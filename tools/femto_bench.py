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
femto_bench.py -- a scorecard for one compiled model, to compare compiler versions.

What it counts, from the generated C (no board needed):
  arena      bytes of working memory the model allocates (PSRAM on the S3)
  traffic    bytes the ops read and write in that memory per run, an estimate:
             each op's stored inputs and outputs once (convs' im2col re-reads and
             the filters are not counted); on the S3 most non-conv time is this
  float      bytes of float tensors stored (4 bytes a value, where int8 is 1)
  ops        ops emitted; copies that became free (views) are not counted
(tools/test_golden.py is what shows a change moved no output bit.)

  python tools/femto_bench.py examples/esp32_yolo26n/yolo26n_q.onnx
  python tools/femto_bench.py model.onnx --json score.json     keep it, to compare later

Measure the board as well: these are proxies, the s3profile build is the truth.
"""
import argparse, json, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from femto_host import compile_model

ESZ = {"float": 4, "int8_t": 1, "uint8_t": 1, "int32_t": 4, "int64_t": 8}


def score(out):
    act = open(os.path.join(out, "activations.h")).read()
    src = open(os.path.join(out, "model.c")).read()
    size, place = {}, {}
    for ct, name, dims in re.findall(r"^\s*(?:extern\s+)?(float|int8_t|int32_t|int64_t|uint8_t)\s+\(?\*?(tensor_\w+?)(?:_p\))?((?:\[\d+\])+)", act, re.M):
        n = 1
        for x in re.findall(r"\d+", dims): n *= int(x)
        size[name] = (n * ESZ[ct], ct)
    for ui, body in re.findall(r"union tensor_union_(\d+) \{(.*?)\};", act, re.S):
        for name in re.findall(r"^\s+\w+ (tensor_\w+)\[", body, re.M):
            place[name] = (ui, 0)
    for name, ui, off in re.findall(r"#define (tensor_\w+) \(\*\(\w+\(\*\)[\[\]\d]+\)\(\(unsigned char\*\)tu(\d+)\+(\d+)\)\)", act):
        place[name] = (ui, int(off))
        if name not in size:
            pass
    for ct, name, dims, off in re.findall(r"//\s*(float|int8_t|int32_t|int64_t|uint8_t) (tensor_\w+)((?:\[\d+\])+) at \+(\d+);", act):
        n = 1
        for x in re.findall(r"\d+", dims): n *= int(x)
        size[name] = (n * ESZ[ct], ct)
    arena = sum(int(b) for b in re.findall(r"union tensor_union_\d+ \{\s*// (\d+) B", act))
    arena += sum(int(b) for b in re.findall(r"// (\d+) B PSRAM", act))
    defs = {fn: (params, body) for fn, params, body in re.findall(r"\nvoid (op_\w+)\(([^)]*)\) \{\n(.*?)\n\}\n", src, re.S)}
    run = src[src.index("static void run_from("):]
    traffic = ops = 0
    for fn, args in re.findall(r"\n\s*(op_\w+)\(([^;]*)\);", run):
        params, body = defs.get(fn, ("", ""))
        names = [m.group(2) for m in (re.search(r"\((\w+)\*\)(tensor_\w+)", a) for a in args.split(",")) if m]
        if "if(y!=x)" in body and len(names) == 2 and place.get(names[0]) == place.get(names[1]) and names[0] in place:
            continue                                     # a view: nothing moves
        ops += 1
        traffic += sum(size.get(t, (0, ""))[0] for t in dict.fromkeys(names))
    floats = sum(b for t, (b, ct) in size.items() if ct == "float")
    return {"arena_bytes": arena, "traffic_bytes": traffic, "float_bytes": floats, "ops": ops}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("onnx"); ap.add_argument("--target", default="esp32s3"); ap.add_argument("--json")
    a = ap.parse_args()
    out = compile_model(a.onnx, tempfile.mkdtemp(prefix="femto_bench_"), target=a.target)
    s = score(out)
    for k, v in s.items():
        print(f"  {k:14s} {v:>12,}")
    if a.json:
        json.dump(s, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
