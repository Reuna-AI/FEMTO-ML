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
femto_host.py -- compile a model with FEMTO-ML and run the generated C on this PC.

The same C the board runs (esp32s3: the S3 conv kernels fall back to their C
reference, which tools/test_conv_kernels.c proves bit-identical to the vector
paths; ansi_int8: the portable target), built with the PC's gcc. The tests and the
benchmark use it; it is also the quickest way to try a model before flashing.

  from femto_host import compile_model, run
  out = compile_model("model.onnx", "build_dir", target="esp32s3")
  y = run(out, x)        # x: float32 numpy array of the model's input shape
"""
import contextlib, io, json, os, shutil, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILER = os.path.join(ROOT, "nhwc_compiler")
KERNELS = os.path.join(ROOT, "examples", "esp32_yolo26n")
HOST = os.path.join(ROOT, "tools", "host")
SZ = {"float32": 4, "int8": 1, "uint8": 1, "int32": 4, "int64": 8}
NP = {"float32": np.float32, "int8": np.int8, "uint8": np.uint8, "int32": np.int32, "int64": np.int64}


def _san(n):
    return "tensor_" + "".join(c if c.isalnum() else "_" for c in n)


def compile_model(onnx_path, out, target="esp32s3", sram=0, quiet=True, prefix="", gcc=True):
    """The six stages, then gcc. Returns out (holding run, io.json and the sources). With [prefix],
    the model's names and files are prefixed (prefix.py), as for a second model in one firmware;
    without [gcc], only the sources are made."""
    sys.path.insert(0, COMPILER)
    import graph_json, fuse, gen_weights, gen_activations, codegen, prefix as prefixing
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    j = lambda n: os.path.join(out, n)
    saved = gen_activations.SRAM_BUDGET_BYTES
    gen_activations.SRAM_BUDGET_BYTES = sram
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log if quiet else sys.stdout):
            graph_json.build(onnx_path, j("graph.json"))
            graph_json.nhwc_pass(j("graph.json"), j("graph_nhwc.json"))
            fuse.main(j("graph_nhwc.json"), j("graph_fused.json"))
            gen_weights.main(j("graph_fused.json"), out, 5_000_000, target)
            gen_activations.main(j("graph_fused.json"), j("activations.h"), target)
            codegen.main(j("graph_fused.json"), j("model"), target, target == "esp32s3", False)
            if prefix:
                prefixing.apply(out, prefix)
    finally:
        gen_activations.SRAM_BUDGET_BYTES = saved
    with open(j("build.log"), "w") as f:
        f.write(log.getvalue())
    d = json.load(open(j("graph_fused.json")))
    ins = [(n, d["tensors"][n]) for n in d["inputs"]]
    outs = [(n, d["tensors"][n]) for n in d["outputs"]]
    numel = lambda s: int(np.prod(s)) if s else 1
    p = prefix
    c = ['#include <stdio.h>', f'#include "{p}model.h"', 'int main(int argc,char**argv){',
         f'  if({p}activations_init()) return 1;',
         '  FILE*f=fopen(argv[1],"rb"); FILE*o=fopen(argv[2],"wb"); if(!f||!o) return 2;']
    for n, t in ins:
        nb = numel(t["shape"]) * SZ[t["dtype"]]
        c.append(f'  if(fread((void*){p}{_san(n)},1,{nb},f)!={nb}) return 3;')
    c.append(f'  {p}run_model();')
    for n, t in outs:
        c.append(f'  fwrite((void*){p}{_san(n)},1,{numel(t["shape"]) * SZ[t["dtype"]]},o);')
    c += ['  fclose(f); fclose(o); return 0;', '}']
    open(j("host_main.c"), "w").write("\n".join(c) + "\n")
    json.dump({"inputs": [(n, t["shape"], t["dtype"]) for n, t in ins],
               "outputs": [(n, t["shape"], t["dtype"]) for n, t in outs]}, open(j("io.json"), "w"))
    src = [j("host_main.c"), j(p + "model.c"), j(p + "activations.c")]
    src += sorted(os.path.join(out, f) for f in os.listdir(out) if f.startswith(p + "weights_") and f.endswith(".c"))
    if target == "esp32s3":
        for f in os.listdir(KERNELS):
            if f.startswith(("esp_nn_", "femto_profile")) and f.endswith((".c", ".h")):
                shutil.copy2(os.path.join(KERNELS, f), out)
        src += [j("esp_nn_conv_s8_simd.c"), j("esp_nn_depthwise_conv_s8_simd.c")]
    if not gcc:
        return out
    # FEMTO_CFLAGS adds flags after these, e.g. "-mfma -ffp-contract=fast" to build the
    # way the board's compiler does (it fuses a*b+c into one multiply-add by default)
    extra = os.environ.get("FEMTO_CFLAGS", "").split()
    subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-w", "-I", HOST, "-I", out, "-o", j("run")] + src
                   + extra + ["-lm", "-lpthread"], check=True)
    return out


def run(out, *xs):
    """Run the compiled model on input arrays; returns its outputs as numpy arrays."""
    io_ = json.load(open(os.path.join(out, "io.json")))
    xin, yout = os.path.join(out, "in.bin"), os.path.join(out, "out.bin")
    with open(xin, "wb") as f:
        for (n, shape, dt), x in zip(io_["inputs"], xs):
            f.write(np.ascontiguousarray(x, dtype=NP[dt]).reshape(shape).tobytes())
    subprocess.run([os.path.join(out, "run"), xin, yout], check=True)
    raw = open(yout, "rb").read(); res = []; at = 0
    for n, shape, dt in io_["outputs"]:
        k = int(np.prod(shape)) * SZ[dt]
        res.append(np.frombuffer(raw[at:at + k], dtype=NP[dt]).reshape(shape)); at += k
    return res
