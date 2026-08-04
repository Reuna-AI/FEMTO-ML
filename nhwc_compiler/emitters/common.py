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
