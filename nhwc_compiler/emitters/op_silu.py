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
#
# ============================================================================

"""
op_silu.py — SiLU (x*sigmoid(x)) as an emitter-time LUT.
int8 in -> FLOAT out (dequantized silu). The next conv's QuantizeLinear re-quantizes.
Float consumers (Add/Concat) read the float output directly.
LUT[256] indexed by (uint8)(x+128); each entry = silu(in_scale*(q-128 - in_zp)).
"""
import math
from . import register
from .common import cfloat, fsan, numel, quant_c

@register("SiLU")
def emit(ctx, node):
    x=node["inputs"][0]; y=node["outputs"][0]; a=node["attrs"]
    is_ = ctx.val0(a["in_scale"]); iz = int(ctx.val0(a["in_zp"]))
    if is_ is None:
        raise ValueError(f"SiLU {node['name']}: in_scale/zp not inlined; rerun graph_json build.")
    n=numel(ctx.shape(x)); fn=fsan(node["name"])

    # Output quant params: if present, FUSE the quantize into the LUT -> int8 out (one pass,
    # no separate QuantizeLinear). Else fall back to float out (for float consumers).
    # Emit int8 when the graph marks this SiLU's OUTPUT as int8 (i.e. no separate Q follows
    # -- the SiLU output IS the quantized tensor). Requires out params to build the quant LUT.
    # If output is float, emit the float LUT (a separate QuantizeLinear does the quant).
    y_dtype = ctx.T.get(y,{}).get("dtype")
    fused_q = (y_dtype=="int8") and ("out_scale" in a) and ("out_zp" in a)
    os_ = ctx.val0(a["out_scale"]) if fused_q else None
    oz_ = int(ctx.val0(a["out_zp"])) if fused_q else None
    late_q = (y_dtype=="int8") and not fused_q and ctx.T[y].get("q_scale") is not None
    if (y_dtype=="int8") and not fused_q and not late_q:
        raise ValueError(f"SiLU {node['name']}: output marked int8 but out_scale/out_zp missing "
                         f"-- cannot build quant LUT. Check fusion pass.")

    h=ctx.header; b=ctx.body
    if late_q:
        # int8 at the scale of the QuantizeLinear that used to follow the layout ops
        # after it (fuse.py int8_layout_paths): the float table below, quantized on the
        # first run by that QuantizeLinear's own expression -- the same bytes.
        lutf=[]
        for q in range(256):
            xf=is_*((q-128)-iz)
            lutf.append(xf/(1.0+math.exp(-xf)))
        tbl=", ".join(cfloat(v) for v in lutf)
        sig=f"void {fn}(const int8_t* x, int8_t* y)"
        h.append(sig+";")
        os_l=ctx.T[y]["q_scale"]; oz_l=int(ctx.T[y]["q_zp"])
        b.append(f"// SiLU {node['name']}  n={n}  in=({is_:.6g},{iz}) out=({os_l:.6g},{oz_l})  int8->int8 (quantized where made)")
        b.append(sig+" {")
        b.append(f"  static const float lutf[256]={{{tbl}}};")
        b.append(f"  static int8_t lut[256]; static int ready=0;")
        b.append(f"  if(!ready){{ const float inv=1.0f/{cfloat(os_l)}; const int z={oz_l};")
        b.append(f"    for(int q=0;q<256;++q) " + quant_c("lut[q]", "lutf[q]") + " ready=1; }")
        b.append(f"  for(int i=0;i<{n};++i) y[i]=lut[(unsigned char)(x[i]+128)];")
        b.append("}"); b.append("")
        ctx.calls.append((node, fn))
        return
    if fused_q:
        # int8 LUT: quantize( silu(dequant(q)) ) baked in. int8 -> int8, quantize is FREE.
        lut=[]
        for q in range(256):
            xf=is_*((q-128)-iz)
            sv=xf/(1.0+math.exp(-xf))               # silu (float)
            r=int(round(sv/os_))+oz_                 # quantize to int8
            if r<-128: r=-128
            if r>127: r=127
            lut.append(r)
        tbl=", ".join(str(v) for v in lut)
        sig=f"void {fn}(const int8_t* x, int8_t* y)"
        h.append(sig+";")
        b.append(f"// SiLU {node['name']}  n={n}  in=({is_:.6g},{iz}) out=({os_:.6g},{oz_})  int8->int8 (quant fused)")
        b.append(sig+" {")
        b.append(f"  static const int8_t lut[256]={{{tbl}}};")
        b.append(f"  for(int i=0;i<{n};++i) y[i]=lut[(unsigned char)(x[i]+128)];")
        b.append("}"); b.append("")
        if y in ctx.T: ctx.T[y]["dtype"]="int8"
    else:
        # FLOAT LUT (unfused): dequantized silu, for float consumers (Add/Concat).
        lutf=[]
        for q in range(256):
            xf=is_*((q-128)-iz)
            lutf.append(xf/(1.0+math.exp(-xf)))
        tbl=", ".join(cfloat(v) for v in lutf)
        sig=f"void {fn}(const int8_t* x, float* y)"
        h.append(sig+";")
        b.append(f"// SiLU {node['name']}  n={n}  in=({is_:.6g},{iz})  int8->float")
        b.append(sig+" {")
        b.append(f"  static const float lut[256]={{{tbl}}};")
        b.append(f"  for(int i=0;i<{n};++i) y[i]=lut[(unsigned char)(x[i]+128)];")
        b.append("}"); b.append("")
        if y in ctx.T: ctx.T[y]["dtype"]="float32"
    ctx.calls.append((node, fn))
