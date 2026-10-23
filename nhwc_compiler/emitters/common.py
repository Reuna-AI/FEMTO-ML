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

"""Shared helpers for op emitters."""
CDT={"int8":"int8_t","uint8":"uint8_t","int32":"int32_t","float32":"float","int64":"int64_t"}

def san(n):  return "tensor_"+"".join(c if c.isalnum() else "_" for c in n)
def wsan(n): return "w_"+"".join(c if c.isalnum() else "_" for c in n)
def fsan(n): return "op_"+"".join(c if c.isalnum() else "_" for c in n)

def dims_of(shape): return "".join(f"[{x}]" for x in shape) if shape else "[1]"
def numel(shape):
    n=1
    for x in shape: n*=x
    return n if shape else 1

class EmitCtx:
    def __init__(self, tensors, target, simd=False):
        self.T=tensors
        self.target=target          # 'esp32s3' | 'ansi_int8' | 'ansi_float'
        self.simd=simd              # emit ESP32-S3 vector asm for conv?
        self.header=[]
        self.body=[]
        self.calls=[]               # ordered op-call lines for the driver
    def shape(self,name):  return self.T.get(name,{}).get("shape",[])
    def dtype(self,name):  return self.T.get(name,{}).get("dtype","int8")
    def ctype(self,name):  return CDT.get(self.dtype(name),"int8_t")
    def value(self,name):  return self.T.get(name,{}).get("value",None)   # inlined const or None
    def val0(self,name):   # first element of an inlined scalar/1-D const
        v=self.value(name); return (v[0] if isinstance(v,list) and v else v)
    def vals(self,name):   return self.value(name) or []

def cfloat(v):
    """Render a Python number as a valid C float literal (always has a '.'; adds 'f')."""
    import math
    v=float(v)
    if v!=v: return "(0.0f/0.0f)"          # NaN guard
    if math.isinf(v): return "(1e30f)" if v>0 else "(-1e30f)"
    r=repr(v)
    if "e" in r or "E" in r:
        return r+"f"                        # 1.5e-3 -> 1.5e-3f (valid)
    if "." not in r: r+=".0"                # 2 -> 2.0
    return r+"f"

def quant_c(dst, vexpr, inv="inv", z="z"):
    """C for dst = int8 of float vexpr, as QuantizeLinear makes it: one multiply by
    1/scale (a float [inv] = 1.0f/s, declared by the caller), rounded half away from
    zero, plus the zero point [z], saturated. Every op that quantizes uses this, so a
    value quantized where it is made is the same byte as one quantized after."""
    # its own names: vexpr is often a v of the caller's, which "float v=(v)..." would shadow
    return (f"{{ float qv_=({vexpr})*{inv}; qv_ = qv_<-512.0f ? -512.0f : (qv_>512.0f ? 512.0f : qv_);"
            f" int qr_=(int)(qv_<0.0f ? qv_-0.5f : qv_+0.5f)+{z};"
            f" if(qr_<-128)qr_=-128; if(qr_>127)qr_=127; {dst}=(signed char)qr_; }}")

def also_q(node):
    """The QuantizeLinears fuse.py folded into this node: [{src, dst, scale, zp, float_dead}]."""
    return node.get("attrs", {}).get("also_q", [])

def main_outputs(node):
    """The node's own outputs, without the int8 ones folded QuantizeLinears added."""
    dsts = {e["dst"] for e in also_q(node)}
    return [o for o in node["outputs"] if o not in dsts]

def also_q_consts(entries):
    """C declaring each folded QuantizeLinear's 1/scale and zero point: inv0/z0, inv1/z1..."""
    return " ".join(f"const float inv{k}=1.0f/{cfloat(e['scale'])}; const int z{k}={int(e['zp'])};"
                    for k, e in enumerate(entries))

def also_q_stores(entries, vexpr, index):
    """C storing float vexpr, quantized, into each folded QuantizeLinear's output at index."""
    return " ".join(quant_c(f"q{k}[{index}]", vexpr, inv=f"inv{k}", z=f"z{k}")
                    for k, e in enumerate(entries))

def nearest_supported(a,H,W,Ho,Wo):
    """Whether ONNX nearest Resize with these attributes picks source pixel
    floor(o*in/out), the one rule the emitter (and fuse.py's lazy Resize) computes.
    True for whole-number upscales under the modes exporters use; anything else
    (align_corners, fractional scales, ...) is refused rather than computed wrong."""
    ctm=a.get("coordinate_transformation_mode","half_pixel")
    nm=a.get("nearest_mode","round_prefer_floor")
    if Ho%H or Wo%W:
        return False
    if ctm=="asymmetric":
        # source o*in/out exactly: floor, or rounding with ties down, which is floor
        # only while the scale is at most 2 (at 3, o=2 rounds to 1)
        return nm=="floor" or (nm=="round_prefer_floor" and Ho//H<=2 and Wo//W<=2)
    if ctm in ("half_pixel","pytorch_half_pixel"):
        return nm in ("round_prefer_floor","round_prefer_ceil")
    return False
