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
#
# For issues and contributions, visit:
# https://github.com/reuna-ai/synapedge-reuna-ai
# ============================================================================

"""
build_model.py — one-shot orchestrator. Runs the whole compiler on an ONNX model
and writes every intermediate + generated source into an output folder.

  python build_model.py model_quant.onnx  outdir/  [--target ansi_float] [--budget 5000000] [--keep-json]

Pipeline (each step imported and called directly):
  1. graph_json.build   onnx        -> outdir/graph.json           (NCHW IR)
  2. graph_json.nhwc     graph.json -> outdir/graph_nhwc.json       (NHWC layout pass)
  3. fuse.main            "         -> outdir/graph_fused.json       (SiLU fusion)
  4. gen_weights.main    fused      -> outdir/weights.h + weights_N.c (OHWI, <=budget/file)
  5. gen_activations.main  "        -> outdir/activations.h           (union arena)
  6. codegen.main          "        -> outdir/model.c + model.h        (per-op C, --target)

Result: outdir/ is a self-contained C project (model.* + weights.* + activations.h).
Targets: ansi_float (reference) | ansi_int8 | esp32s3 (SIMD).
"""
import sys, os, argparse, importlib.util

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# import the stage modules
import graph_json
import fuse
import gen_weights
import gen_activations
import codegen

def step(msg):
    print(f"\n=== {msg} ===")

def main():
    ap=argparse.ArgumentParser(description="Compile ONNX -> ESP32-S3 NHWC C project")
    ap.add_argument("onnx", help="input quantized ONNX model")
    ap.add_argument("outdir", help="output folder (created if missing)")
    ap.add_argument("--target", choices=["ansi_float","ansi_int8","esp32s3"], default="ansi_float")
    ap.add_argument("--budget", type=int, default=5_000_000, help="max bytes per weights_*.c")
    ap.add_argument("--simd", action="store_true", help="ESP32-S3 vector conv (with --target esp32s3)")
    ap.add_argument("--debug", action="store_true", help="per-node Serial trace (esp32s3)")
    ap.add_argument("--debug", action="store_true", help="per-node trace prints")
    ap.add_argument("--name", default="model", help="basename for generated model.c/.h")
    ap.add_argument("--keep-json", action="store_true", help="keep intermediate .json files")
    a=ap.parse_args()

    if not os.path.isfile(a.onnx):
        print(f"ERROR: onnx not found: {a.onnx}"); sys.exit(1)
    os.makedirs(a.outdir, exist_ok=True)

    j_nchw  = os.path.join(a.outdir,"graph.json")
    j_nhwc  = os.path.join(a.outdir,"graph_nhwc.json")
    j_fused = os.path.join(a.outdir,"graph_fused.json")
    acts_h  = os.path.join(a.outdir,"activations.h")
    model   = os.path.join(a.outdir, a.name)

    step("1/6 ONNX -> JSON (NCHW)")
    graph_json.build(a.onnx, j_nchw)

    step("2/6 NCHW -> NHWC layout pass")
    graph_json.nhwc_pass(j_nchw, j_nhwc)

    step("3/6 SiLU fusion")
    fuse.main(j_nhwc, j_fused)

    step("4/6 weights.h + weights_*.c")
    gen_weights.main(j_fused, a.outdir, a.budget, a.target)

    step("5/6 activations.h (union arena)")
    gen_activations.main(j_fused, acts_h, a.target)

    step(f"6/6 codegen -> {a.name}.c/.h  (target={a.target})")
    codegen.main(j_fused, model, a.target, a.simd, a.debug)

    if not a.keep_json:
        for j in (j_nchw, j_nhwc):        # keep fused (useful for debugging) unless you prefer not
            try: os.remove(j)
            except OSError: pass

    print("\n=== DONE ===")
    print(f"output folder: {a.outdir}")
    print("generated:")
    for f in sorted(os.listdir(a.outdir)):
        p=os.path.join(a.outdir,f); sz=os.path.getsize(p)
        print(f"  {f:32s} {sz:>10,} B")
    print("\nnext: compile the folder, e.g.")
    print(f"  gcc -O2 -I{a.outdir} {model}.c {a.outdir}/weights_*.c -lm -o run   # ansi targets")

if __name__=="__main__":
    main()
