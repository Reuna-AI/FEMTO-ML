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
lazy.py -- reading a float tensor that fuse.py mark_lazy_floats left unstored.

A lazy tensor is made again where it is read, from the int8 tensor(s) it came from,
by the same C expression its producer would have stored: a SiLU table entry, a
dequantized byte, a channel slice of another tensor, or the float sum of two.

Every tensor here is read as [outer][C] -- C its last dimension, the channels in
NHWC -- so a reader loops over o (pixel) and c (channel) and asks for the C of
value (o, c):

    v = value(ctx, name, "o", "c", params)

params maps each stored tensor the expression reads to the C parameter that holds
it; the reader adds those tensors to its signature with sources(ctx, name).
"""
import math
from .common import cfloat, san, nearest_supported

def is_lazy(ctx, t):
    return bool(ctx.T.get(t, {}).get("lazy"))

def _node(ctx, t):
    return ctx.T[t]["lazy"]["node"]

def _ins(node):
    return node["inputs"][:2] if node["op"] == "Add" else node["inputs"][:1]

def sources(ctx, t):
    """The stored tensors lazy tensor t is made from, in a fixed order."""
    if not is_lazy(ctx, t):
        return [t]
    out = []
    for i in _ins(_node(ctx, t)):
        for s in sources(ctx, i):
            if s not in out:
                out.append(s)
    return out

def _silu_table(ctx, node):
    """The SiLU's float table, once, at file scope: the same entries its emitter makes."""
    name = "lut_" + san(node["outputs"][0])
    done = ctx.__dict__.setdefault("lazy_tables", set())
    if name not in done:
        a = node["attrs"]; is_ = ctx.val0(a["in_scale"]); iz = int(ctx.val0(a["in_zp"]))
        lutf = []
        for q in range(256):
            xf = is_ * ((q - 128) - iz)
            lutf.append(xf / (1.0 + math.exp(-xf)))
        ctx.body.append(f"// SiLU {node['name']}: its table, read where its values are needed")
        ctx.body.append(f"static const float {name}[256]={{{', '.join(cfloat(v) for v in lutf)}}};")
        ctx.body.append("")
        done.add(name)
    return name

def _dq_table(ctx, node):
    """A DequantizeLinear's 256 values, as a table its readers fill on their first run."""
    name = "lutdq_" + san(node["outputs"][0])
    done = ctx.__dict__.setdefault("lazy_tables", set())
    if name not in done:
        s = ctx.val0(node["inputs"][1]); z = int(ctx.val0(node["inputs"][2])) if len(node["inputs"]) > 2 else 0
        ctx.body.append(f"// DequantizeLinear {node['name']}: its values, read where they are needed")
        ctx.body.append(f"static float {name}[256]; static int {name}_ready=0;")
        ctx.body.append(f"static void {name}_fill(void){{ const float s={cfloat(s)}; const int z={z};")
        ctx.body.append(f"  for(int q=-128;q<128;++q) {name}[q+128]=s*(float)(q-z);")
        ctx.body.append(f"  {name}_ready=1; }}")
        ctx.body.append("")
        done.add(name)
    ctx.__dict__.setdefault("lazy_fills", [])
    if name not in ctx.lazy_fills:
        ctx.lazy_fills.append(name)
    return name

def begin(ctx):
    """Call before building a reader's expressions; prelude() then gives its table fills."""
    ctx.lazy_fills = []

def prelude(ctx):
    """C lines a reader runs first: fill the tables its expressions read."""
    return [f"  if(!{n}_ready) {n}_fill();" for n in getattr(ctx, "lazy_fills", [])]

def value(ctx, t, o, c, params):
    """C expression of float t at (pixel o, channel c)."""
    m = ctx.T[t]; C = (m.get("shape") or [1])[-1]
    if not is_lazy(ctx, t):
        p = params[t]
        if m.get("dtype") == "int8" and m.get("q_scale") is not None:
            # as the Add emitter dequantizes an int8 input
            return f"({cfloat(m['q_scale'])}*(float)((int){p}[({o})*{C}+({c})] - ({int(m.get('q_zp', 0))})))"
        return f"{p}[({o})*{C}+({c})]"
    node = _node(ctx, t); op = node["op"]
    if op == "SiLU":
        x = node["inputs"][0]; Cx = ctx.T[x]["shape"][-1]
        return f"{_silu_table(ctx, node)}[(unsigned char)({params[x]}[({o})*{Cx}+({c})]+128)]"
    if op == "DequantizeLinear":
        # a table, filled on the board by the DequantizeLinear's own expression and
        # stored: the same rounded float it used to store, which an inline s*(q-z)
        # would not be once the compiler fuses it into a multiply-add (Xtensa madd.s)
        x = node["inputs"][0]; Cx = ctx.T[x]["shape"][-1]
        return f"{_dq_table(ctx, node)}[(unsigned char)({params[x]}[({o})*{Cx}+({c})]+128)]"
    if op == "Split":
        x = node["inputs"][0]; k = node["outputs"].index(t)
        sizes = split_sizes(ctx, node)
        return value(ctx, x, o, f"{sum(sizes[:k])}+({c})", params)
    if op == "Resize":
        # nearest, as the Resize emitter picks its source pixel
        x = node["inputs"][0]; _, H, W, _ = ctx.T[x]["shape"]; _, Ho, Wo, _ = ctx.T[t]["shape"]
        if not nearest_supported(node["attrs"], H, W, Ho, Wo):
            raise ValueError(f"Resize {node['name']}: nearest {H}x{W}->{Ho}x{Wo} with these modes is not supported")
        sy = f"(int)((float)(({o})/{Wo})*((float){H}/{Ho}))"
        sx = f"(int)((float)(({o})%{Wo})*((float){W}/{Wo}))"
        src = f"((({sy})<{H}?({sy}):{H - 1})*{W}+(({sx})<{W}?({sx}):{W - 1}))"
        return value(ctx, x, src, c, params)
    if op == "Add":
        a, b = node["inputs"][0], node["inputs"][1]
        return f"({value(ctx, a, o, c, params)}+{value(ctx, b, o, c, params)})"
    raise ValueError(f"lazy tensor {t}: no expression for {op}")

def byte_of(ctx, t, o, c, params):
    """When float t at (pixel o, channel c) is made from one int8 byte: (the C that
    reads that byte, a function giving the C of the float from a byte expression B),
    the same float value() gives. None when it is made from two (an Add) or from a
    stored float. A reader that only quantizes t can then quantize all 256 bytes
    once, into a table, and look each value up."""
    m = ctx.T[t]; C = (m.get("shape") or [1])[-1]
    if not is_lazy(ctx, t):
        if m.get("dtype") == "int8" and m.get("q_scale") is not None:
            s, z = cfloat(m["q_scale"]), int(m.get("q_zp", 0))
            return f"{params[t]}[({o})*{C}+({c})]", (lambda B: f"({s}*(float)((int){B} - ({z})))")
        return None
    node = _node(ctx, t); op = node["op"]
    if op in ("SiLU", "DequantizeLinear"):
        x = node["inputs"][0]; Cx = ctx.T[x]["shape"][-1]
        tbl = _silu_table(ctx, node) if op == "SiLU" else _dq_table(ctx, node)
        return f"{params[x]}[({o})*{Cx}+({c})]", (lambda B: f"{tbl}[(unsigned char)({B}+128)]")
    if op == "Split":
        x = node["inputs"][0]; k = node["outputs"].index(t)
        sizes = split_sizes(ctx, node)
        return byte_of(ctx, x, o, f"{sum(sizes[:k])}+({c})", params)
    if op == "Resize":
        x = node["inputs"][0]; _, H, W, _ = ctx.T[x]["shape"]; _, Ho, Wo, _ = ctx.T[t]["shape"]
        if not nearest_supported(node["attrs"], H, W, Ho, Wo):
            raise ValueError(f"Resize {node['name']}: nearest {H}x{W}->{Ho}x{Wo} with these modes is not supported")
        sy = f"(int)((float)(({o})/{Wo})*((float){H}/{Ho}))"
        sx = f"(int)((float)(({o})%{Wo})*((float){W}/{Wo}))"
        src = f"((({sy})<{H}?({sy}):{H - 1})*{W}+(({sx})<{W}?({sx}):{W - 1}))"
        return byte_of(ctx, x, src, c, params)
    return None

def split_sizes(ctx, node):
    """A Split's part sizes, as its emitter works them out."""
    xsh = ctx.shape(node["inputs"][0]); axis = node["attrs"].get("axis", 0)
    if axis < 0: axis += len(xsh)
    outs = [o for o in node["outputs"] if o not in {e["dst"] for e in node["attrs"].get("also_q", [])}]
    sizes = node["attrs"].get("split", None)
    if sizes is None and len(node["inputs"]) > 1:
        sv = ctx.vals(node["inputs"][1])
        if sv: sizes = [int(v) for v in sv]
    if sizes is None:
        each = xsh[axis] // len(outs); sizes = [each] * len(outs)
    return sizes
