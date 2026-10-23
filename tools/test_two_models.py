#!/usr/bin/env python3
# Copyright (C) 2026 reuna-ai
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
test_two_models.py -- two compiled models in one program, and a model fed int8 only.

- Two models built into one program, the second with --prefix (nhwc_compiler/prefix.py), and
  one source file including both headers: it compiles and links, and each model gives exactly
  the bytes it gives built alone. Without the prefix, the same program must fail to link.
- FEMTO_NO_FLOAT_INPUT: the model's float input is never allocated, run_model() is not there,
  and run_model_int8_input() on the caller's own int8 input gives the bytes run_model() gives.

  python tools/test_two_models.py
"""
import os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from femto_host import compile_model, run, HOST
import test_ops

RESULTS = []


def check(ok, label):
    RESULTS.append(bool(ok))
    print(f"{'ok  ' if ok else 'FAIL'} {label}")


def gcc(out, sources, includes, defines=()):
    cmd = ["gcc", "-O2", "-ffp-contract=off", "-w", "-I", HOST] + [f"-I{i}" for i in includes] \
          + [f"-D{d}" for d in defines] + ["-o", out] + sources + ["-lm", "-lpthread"]
    return subprocess.run(cmd, capture_output=True, text=True)


def model_sources(folder, prefix=""):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith(".c") and f.startswith(prefix) and not f.startswith("host_main")
                  and not f.startswith(("esp_nn_", "femto_profile")))


def main():
    work = tempfile.mkdtemp(prefix="femto_two_")
    models = {}

    for name in ("relu", "stem"):
        g, outs = test_ops.CASES[name]()
        path = os.path.join(work, name + ".onnx")
        models[name] = (path, test_ops.quantize(g.model(outs), path))

    x = {name: test_ops.RNG.random(shape, dtype=np.float32) for name, (_, shape) in models.items()}

    # Alone, as femto_host builds them.
    alone = {name: run(compile_model(path, os.path.join(work, f"alone_{name}")), x[name])
             for name, (path, _) in models.items()}

    # Together: relu as it comes, stem prefixed b_.
    a = compile_model(models["relu"][0], os.path.join(work, "a"), gcc=False)
    b = compile_model(models["stem"][0], os.path.join(work, "b"), prefix="b_", gcc=False)
    import json
    io_a = json.load(open(os.path.join(a, "io.json"))) if os.path.exists(os.path.join(a, "io.json")) else None

    def tensors(folder):
        d = json.load(open(os.path.join(folder, "graph_fused.json")))
        san = lambda n: "tensor_" + "".join(c if c.isalnum() else "_" for c in n)
        return [san(n) for n in d["inputs"]], [(san(n), int(np.prod(d["tensors"][n]["shape"]))) for n in d["outputs"]]

    (ia,), oa = tensors(a)
    (ib,), ob = tensors(b)
    main_c = os.path.join(work, "both.c")
    lines = ['#include <stdio.h>', '#include "model.h"', '#include "b_model.h"',
             'int main(int argc,char**argv){', '  if(activations_init()||b_activations_init()) return 1;',
             '  FILE*f=fopen(argv[1],"rb"); FILE*o=fopen(argv[2],"wb"); if(!f||!o) return 2;',
             f'  if(fread((void*){ia},4,{x["relu"].size},f)!={x["relu"].size}) return 3;',
             f'  if(fread((void*)b_{ib},4,{x["stem"].size},f)!={x["stem"].size}) return 3;',
             '  run_model(); b_run_model();']
    lines += [f'  fwrite((void*){n},4,{k},o);' for n, k in oa]
    lines += [f'  fwrite((void*)b_{n},4,{k},o);' for n, k in ob]
    lines += ['  fclose(f); fclose(o); return 0;', '}']
    open(main_c, "w").write("\n".join(lines) + "\n")
    kernels = [os.path.join(a, "esp_nn_conv_s8_simd.c"), os.path.join(a, "esp_nn_depthwise_conv_s8_simd.c")]
    exe = os.path.join(work, "both")
    built = gcc(exe, [main_c] + model_sources(a) + model_sources(b, "b_") + kernels, [a, b])
    check(built.returncode == 0, "two models, one prefixed, one source including both headers: compiles and links"
          + ("" if built.returncode == 0 else f" -- {built.stderr[:300]}"))

    if built.returncode == 0:
        xin, yout = os.path.join(work, "in.bin"), os.path.join(work, "out.bin")
        open(xin, "wb").write(x["relu"].tobytes() + x["stem"].tobytes())
        subprocess.run([exe, xin, yout], check=True)
        got = np.frombuffer(open(yout, "rb").read(), np.float32)
        want = np.concatenate([np.ravel(o) for o in alone["relu"]] + [np.ravel(o) for o in alone["stem"]])
        check(got.tobytes() == want.astype(np.float32).tobytes(), "and each gives exactly the bytes it gives alone")

    # The same two without the prefix: the names clash.
    c = compile_model(models["stem"][0], os.path.join(work, "c"), gcc=False)
    open(os.path.join(work, "clash.c"), "w").write('#include "model.h"\nint main(void){ run_model(); return 0; }\n')
    clash = gcc(os.path.join(work, "clash"), [os.path.join(work, "clash.c")] + model_sources(a) + model_sources(c)
                + kernels, [a])
    check(clash.returncode != 0, "without the prefix, the same two do not link (what the prefix is for)")

    # FEMTO_NO_FLOAT_INPUT: the int8 input written by the caller, no float input taken.
    path, shape = models["relu"]
    d = compile_model(path, os.path.join(work, "d"), gcc=False)
    header = open(os.path.join(d, "model.h")).read()

    if "MODEL_INPUT_INT8" not in header:
        check(False, "the test model has an int8 input entry")
    else:
        n, cc, h, w = shape
        main_c = os.path.join(work, "int8.c")
        open(main_c, "w").write("\n".join([
            '#include <stdio.h>', '#include <stdlib.h>', '#include "model.h"',
            'int main(int argc,char**argv){', '  if(activations_init()) return 1;',
            f'  if({ia}_p) return 5;   /* the float input was not taken */',
            f'  static float x[{cc}][{h}][{w}]; FILE*f=fopen(argv[1],"rb"); FILE*o=fopen(argv[2],"wb");',
            f'  if(fread(x,4,{cc*h*w},f)!={cc*h*w}) return 3;',
            f'  for(int y=0;y<{h};++y) for(int xx=0;xx<{w};++xx) for(int k=0;k<{cc};++k)',
            f'    model_input_int8[(y*{w}+xx)*{cc}+k]=model_quantize_input(x[k][y][xx]);',
            '  run_model_int8_input();'] +
            [f'  fwrite((void*){nm},4,{k},o);' for nm, k in oa] + ['  return 0;', '}']) + "\n")
        exe = os.path.join(work, "int8")
        built = gcc(exe, [main_c] + model_sources(d) + kernels, [d], ["FEMTO_NO_FLOAT_INPUT"])
        check(built.returncode == 0, "built with FEMTO_NO_FLOAT_INPUT" + ("" if built.returncode == 0 else f" -- {built.stderr[:300]}"))

        if built.returncode == 0:
            xin, yout = os.path.join(work, "in8.bin"), os.path.join(work, "out8.bin")
            open(xin, "wb").write(x["relu"].tobytes())
            done = subprocess.run([exe, xin, yout])
            check(done.returncode == 0, f"the float input is never allocated (exit {done.returncode})")
            got = np.frombuffer(open(yout, "rb").read(), np.float32) if done.returncode == 0 else None
            want = np.concatenate([np.ravel(o) for o in alone["relu"]]).astype(np.float32)
            check(got is not None and got.tobytes() == want.tobytes(),
                  "run_model_int8_input() on the caller's int8 input: the bytes run_model() gives")

        uses = '#include "model.h"\nint main(void){ run_model(); return 0; }\n'
        open(os.path.join(work, "float.c"), "w").write(uses)
        refused = gcc(os.path.join(work, "float"), [os.path.join(work, "float.c")] + model_sources(d) + kernels, [d],
                      ["FEMTO_NO_FLOAT_INPUT"])
        check(refused.returncode != 0, "and run_model() is not there to call by mistake")

    print(f"{sum(RESULTS)} of {len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
