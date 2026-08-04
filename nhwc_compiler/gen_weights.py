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
gen_weights.py — Stage 2 (multi-file). Emit weights split across .c files so no
single file exceeds a byte budget (default 5 MB). NHWC handled: conv weights
flagged "ohwi" in the JSON are transposed OIHW->OHWI to match the ACCX kernel.

Output:
  weights.h        — #include-once header, extern-declares every array
  weights_0.c ...   — definitions, each <= budget; a whole tensor never splits

Each .c #includes weights.h. Your generated model code includes only weights.h.

  python gen_weights.py graph_nhwc.json outdir/ [--budget 5000000]
"""
import sys, os, json, argparse, onnx, numpy as np
from onnx import numpy_helper

NPC={"int8":"int8_t","uint8":"uint8_t","int32":"int32_t","int64":"int64_t",
     "float32":"float","float64":"double"}

def san(n): return "w_"+"".join(c if c.isalnum() else "_" for c in n)

def decl(cname, arr):
    ct=NPC.get(str(arr.dtype),"float")
    dims="".join(f"[{d}]" for d in arr.shape) if arr.ndim else "[1]"
    return ct, dims

def write_def(f, cname, arr):
    ct,dims=decl(cname,arr)
    f.write(f"WEIGHTS_FLASH const {ct} __attribute__((aligned(16))) {cname}{dims} = {{")
    vals=arr.flatten().tolist()
    isf = str(arr.dtype).startswith("float")
    for i,v in enumerate(vals):
        if i%16==0: f.write("\n  ")
        f.write((repr(float(v))+"f" if ct=="float" else repr(float(v))) if isf else str(int(v)))
        f.write(",")
    f.write("\n};\n\n")

def main(json_path, outdir, budget, target="ansi_int8"):
    d=json.load(open(json_path, encoding="utf-8"))
    m=onnx.load(d["onnx_path"]); 
    inits={it.name: numpy_helper.to_array(it) for it in m.graph.initializer}
    T=d["tensors"]
    os.makedirs(outdir, exist_ok=True)

    # collect constants to emit, applying OHWI where flagged
    items=[]  # (cname, arr)
    for name,meta in T.items():
        if not meta.get("const_file"): continue
        if name not in inits: continue
        arr=inits[name]
        if meta.get("depthwise") and arr.ndim==4:
            # depthwise weight in OIHW is [K,1,Kh,Kw]. Repack to [Kh,Kw,K] (HWC) which the
            # esp_nn depthwise SIMD lib expects AND our scalar depthwise kernel reads.
            K,Ci,Kh,Kw = arr.shape          # Ci==1
            arr=np.ascontiguousarray(np.transpose(arr[:, 0, :, :], (1,2,0)))  # [Kh,Kw,K]
        elif meta.get("ohwi") and arr.ndim==4:
            arr=np.ascontiguousarray(np.transpose(arr,(0,2,3,1)))   # OIHW->OHWI
        items.append((san(name), arr))

    # ---- header: extern declarations ----
    with open(os.path.join(outdir,"weights.h"),"w",encoding="utf-8") as h:
        h.write("// Auto-generated weight declarations (int8 NHWC). Do not edit.\n")
        h.write("#pragma once\n#include <stdint.h>\n\n")
        # Force const weights to stay in FLASH, not internal DRAM. On Arduino-ESP32
        # (ESP-IDF underneath), the correct qualifier for user .c files is to place
        # them in the flash rodata via the compiler's default for `const` PLUS ensuring
        # they are not implicitly copied. The reliable, linker-script-free way is to
        # keep them `const` (arduino-esp32 maps file-scope const to flash .rodata).
        h.write("#ifndef WEIGHTS_FLASH\n#define WEIGHTS_FLASH\n#endif\n\n")
        for cname,arr in items:
            ct,dims=decl(cname,arr)
            h.write(f"extern const {ct} __attribute__((aligned(16))) {cname}{dims} WEIGHTS_FLASH;\n")
        h.write(f"\n// {len(items)} constant tensors\n")

    # ---- split .c files by byte budget; a tensor is never split across files ----
    def nbytes(arr): return arr.size*arr.itemsize
    files=[]; cur=[]; cur_bytes=0
    for cname,arr in items:
        b=nbytes(arr)
        if cur and cur_bytes+b>budget:
            files.append(cur); cur=[]; cur_bytes=0
        # a single tensor larger than budget still goes alone in its own file
        cur.append((cname,arr)); cur_bytes+=b
    if cur: files.append(cur)

    for idx,group in enumerate(files):
        path=os.path.join(outdir,f"weights_{idx}.c")
        with open(path,"w",encoding="utf-8") as f:
            f.write(f'// Auto-generated weights part {idx}. Do not edit.\n#include "weights.h"\n\n')
            for cname,arr in group: write_def(f,cname,arr)
        sz=sum(nbytes(a) for _,a in group)
        print(f"  weights_{idx}.c : {len(group)} tensors, {sz/1e6:.2f} MB data")

    print(f"gen_weights -> {outdir}: {len(items)} tensors across {len(files)} file(s), budget {budget/1e6:.1f} MB")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("json"); ap.add_argument("outdir")
    ap.add_argument("--budget", type=int, default=5_000_000)
    ap.add_argument("--target", choices=["esp32s3","ansi_int8","ansi_float"], default="ansi_int8")
    a=ap.parse_args()
    main(a.json, a.outdir, a.budget, a.target)
