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
op_float_ops.py — FLOAT reference emitters for the non-conv ops.
Contract: every activation is float32 EXCEPT the int8 in/out of a QLinearConv.
  DequantizeLinear: int8 -> float32   y = scale*(x - zp)
  QuantizeLinear  : float32 -> int8   y = clamp(round(x/scale)+zp)
Everything else is pure float32. Structural ops read static attrs from JSON.
Signatures pass explicit buffers; shapes/attrs baked as literals.
"""
from . import register, REGISTRY
from .lazy import is_lazy, sources, value, begin as lazy_begin, prelude as lazy_prelude
from .common import nearest_supported, fsan, san, wsan, numel, cfloat, quant_c, also_q, also_q_consts, also_q_stores, main_outputs

def _sig(fn, args): return f"void {fn}({', '.join(args)})"
_CDT={"int8":"int8_t","uint8":"uint8_t","int32":"int32_t","float32":"float","int64":"int64_t"}
def _ct(ctx,name): return _CDT.get(ctx.dtype(name),"float")
def _isq(ctx,name):
    t=ctx.T.get(name,{})
    return t.get("dtype")=="int8" and (t.get("q_scale") is not None)
def _deq(ctx,name):
    t=ctx.T.get(name,{}); return t.get("q_scale",1.0), int(t.get("q_zp",0))


def _assert_not_quantized(ctx, node, *inputs):
    """Safety: ops that don't dequantize must not silently receive a quantized int8 input
    (would read raw ints as data). Raise a clear error instead of emitting wrong code."""
    for nm in inputs:
        t=ctx.T.get(nm,{})
        if t.get("dtype")=="int8" and t.get("q_scale") is not None:
            raise ValueError(
                f"{node['op']} {node['name']}: input '{nm}' is a quantized int8 tensor "
                f"(scale={t.get('q_scale')}, zp={t.get('q_zp')}) but this emitter has no "
                f"dequant handling. Add _isq/_deq logic to the {node['op']} emitter, or "
                f"insert a DequantizeLinear before it.")


def _sub_zp(expr, zp):
    """Render (expr - zp) so negative zp doesn't produce '--' (decrement). e.g. zp=-94 ->
    'expr - (-94)'."""
    zp=int(zp)
    return f"({expr} - ({zp}))"

def _strides(shape):
    """row-major strides for a shape."""
    st=[1]*len(shape)
    for i in range(len(shape)-2,-1,-1): st[i]=st[i+1]*shape[i+1]
    return st

# ---------------- Dequantize / Quantize (the conv boundary) ----------------
@register("DequantizeLinear")
def emit_dq(ctx,node):
    x=node["inputs"][0]; s=node["inputs"][1]; z=node["inputs"][2] if len(node["inputs"])>2 else None
    y=node["outputs"][0]; fn=fsan(node["name"]); n=numel(ctx.shape(x))
    sc=ctx.val0(s); zv=ctx.val0(z) if z is not None else 0
    to_int8 = ctx.T.get(y,{}).get("dtype")=="int8" and ctx.T[y].get("q_scale") is not None
    then_sigmoid = bool(node["attrs"].get("then_sigmoid"))
    sig=_sig(fn,["const int8_t* x", ("int8_t* y" if to_int8 else "float* y")]); ctx.header.append(sig+";")
    b=ctx.body
    b.append(f"// DequantizeLinear {node['name']}"
             + ("  -> int8 at the next QuantizeLinear's scale (fuse.py int8_layout_paths)" if to_int8 else "")
             + ("  -> sigmoid (fuse.py sigmoid_through_layout)" if then_sigmoid else ""))
    b.append(sig+" {")
    if sc is not None:
        b.append(f"  const float s={cfloat(sc)}; const int z={int(zv) if zv is not None else 0};")
    else:  # value not inlined -> read runtime array from weights.h
        zref=(wsan(z)+"[0]") if z is not None else "0"
        b.append(f"  const float s={wsan(s)}[0]; const int z={zref};")
    if to_int8 or then_sigmoid:
        # 256 inputs, so a table -- filled on the first run by the very expressions
        # the separate ops used, hence the same values to the bit.
        ty = "int8_t" if to_int8 else "float"
        b.append(f"  static {ty} lut[256]; static int ready=0;")
        b.append(f"  if(!ready){{")
        if to_int8:
            b.append(f"    const float inv=1.0f/{cfloat(ctx.T[y]['q_scale'])}; const int zo={int(ctx.T[y]['q_zp'])};")
            b.append(f"    for(int q=-128;q<128;++q){{ float d=s*(float)(q-z); " + quant_c("lut[q+128]", "d", z="zo") + " }")
        else:
            b.append(f"    for(int q=-128;q<128;++q){{ float v=s*(float)(q-z); lut[q+128]=1.0f/(1.0f+expf(-v)); }}")
        b.append(f"    ready=1; }}")
        b.append(f"  for(int i=0;i<{n};++i) y[i]=lut[(unsigned char)(x[i]+128)];")
    else:
        b.append(f"  for(int i=0;i<{n};++i) y[i]=s*(float)((int)x[i]-z);")
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,["x","y"]))

@register("QuantizeLinear")
def emit_q(ctx,node):
    x=node["inputs"][0]; s=node["inputs"][1]; z=node["inputs"][2] if len(node["inputs"])>2 else None
    y=node["outputs"][0]; fn=fsan(node["name"]); n=numel(ctx.shape(x))
    sc=ctx.val0(s); zv=ctx.val0(z) if z is not None else 0
    sig=_sig(fn,["const float* x","int8_t* y"]); ctx.header.append(sig+";")
    b=ctx.body
    b.append(f"// QuantizeLinear {node['name']}")
    b.append(sig+" {")
    if sc is not None:
        b.append(f"  const float s={cfloat(sc)}; const int z={int(zv) if zv is not None else 0};")
    else:
        zref=(wsan(z)+"[0]") if z is not None else "0"
        b.append(f"  const float s={wsan(s)}[0]; const int z={zref};")
    # One multiply by 1/s and an inline round half away from zero (lroundf's rule),
    # rather than a float divide and a libm call per element: this is the commonest op
    # in a QDQ graph (every conv input), and on an FPU without a divider each divide is
    # a long instruction sequence. v is clamped first so the int conversion is defined.
    b.append(f"  const float inv=1.0f/s;")
    b.append(f"  for(int i=0;i<{n};++i) " + quant_c("y[i]", "x[i]"))
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,["x","y"]))

# ---------------- elementwise unary ----------------
def _unary(op, cexpr):
    @register(op)
    def _e(ctx,node,cexpr=cexpr,op=op):
        x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"]); n=numel(ctx.shape(x))
        sig=_sig(fn,["const float* x","float* y"]); ctx.header.append(sig+";")
        b=ctx.body; b.append(f"// {op} {node['name']}"); b.append(sig+" {")
        b.append(f"  for(int i=0;i<{n};++i){{ float v=x[i]; y[i]={cexpr}; }}")
        b.append("}"); b.append("")
        ctx.calls.append((node,fn,["x","y"]))
_unary("Sigmoid","1.0f/(1.0f+expf(-v))")
_sigmoid_float = REGISTRY["Sigmoid"]

@register("Sigmoid")
def emit_sigmoid(ctx,node):
    if not node["attrs"].get("identity"):
        return _sigmoid_float(ctx,node)
    x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"]); n=numel(ctx.shape(x))
    sig=_sig(fn,["const float* x","float* y"]); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Sigmoid {node['name']}: done in its DequantizeLinears (fuse.py sigmoid_through_layout)")
    b.append(sig+" {"); b.append(f"  if(y!=x) memcpy(y, x, {n}*sizeof(*x));")
    b.append("}"); b.append(""); ctx.calls.append((node,fn,["x","y"]))

@register("Pow")
def emit_pow(ctx,node):
    x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"]); n=numel(ctx.shape(x))
    e=node["inputs"][1]; ev=ctx.val0(e); ev = ev if ev is not None else 2.0
    sig=_sig(fn,["const float* x","float* y"]); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Pow {node['name']}  exp={ev}"); b.append(sig+" {")
    b.append(f"  const float e={cfloat(ev)};")
    b.append(f"  for(int i=0;i<{n};++i) y[i]=powf(x[i],e);")
    b.append("}"); b.append(""); ctx.calls.append((node,fn,["x","y"]))

# ---------------- elementwise binary (broadcast on trailing dims) ----------------
def _binary(op, cop):
    @register(op)
    def _e(ctx,node,cop=cop,op=op):
        a=node["inputs"][0]; c=node["inputs"][1]; y=node["outputs"][0]; fn=fsan(node["name"])
        na=numel(ctx.shape(a)); nc=numel(ctx.shape(c)); ny=numel(ctx.shape(y))
        sig=_sig(fn,["const float* a","const float* b","float* y"]); ctx.header.append(sig+";")
        b=ctx.body; b.append(f"// {op} {node['name']}  (a={na} b={nc} y={ny})"); b.append(sig+" {")
        # simple broadcast: if b is scalar or same size; else assume b broadcasts by repeat
        b.append(f"  for(int i=0;i<{ny};++i){{ float av=a[i%{na}], bv=b[i%{nc}]; y[i]=av {cop} bv; }}")
        b.append("}"); b.append(""); ctx.calls.append((node,fn,["a","b","y"]))

# ---------------- Concat (static axis) ----------------
@register("Concat")
def emit_concat(ctx,node):
    outs=node["outputs"][0]; axis=node["attrs"].get("axis",0); fn=fsan(node["name"])
    osh=ctx.shape(outs); rank=len(osh)
    if axis<0: axis+=rank
    ins=node["inputs"]
    # GENERAL FIX: the stored axis can be stale relative to the current tensor rank
    # (e.g. a rank-4 axis left on tensors later reduced to rank-3 by a squeeze/reshape).
    # Detect the true concat axis: the dimension along which the input sizes sum to the
    # output size while all other dims match.
    _insh=[ctx.shape(i) for i in ins]
    def _valid_axis(ax):
        if osh and 0<=ax<len(osh) and all(len(a)==len(osh) for a in _insh if a):
            s=0; ok=True
            for a in _insh:
                if not a or len(a)!=len(osh): return False
                s+=a[ax]
                for dd in range(len(osh)):
                    if dd!=ax and a[dd]!=osh[dd]: ok=False
            return ok and s==osh[ax]
        return False
    if not _valid_axis(axis) and osh:
        for cand in range(len(osh)):
            if _valid_axis(cand):
                axis=cand; break
        else:
            # last resort: clamp into range so we at least don't index OOB
            if axis>=len(osh): axis=len(osh)-1
    if ctx.T.get(outs,{}).get("dtype")=="int8":
        return _emit_concat_int8(ctx,node,axis)
    if any(is_lazy(ctx,i) for i in ins):
        return _emit_concat_lazy(ctx,node,axis)
    # output is float; inputs may be float or int8(quantized) -> dequant per input
    extra=also_q(node)
    args=[]
    for i,inp in enumerate(ins):
        ct = "int8_t" if _isq(ctx,inp) else _ct(ctx,inp)
        args.append("const "+ct+"* in"+str(i))
    args.append("float* y")
    args += [f"int8_t* q{k}" for k in range(len(extra))]
    sig=_sig(fn,args); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Concat {node['name']} axis={axis} of {len(ins)} (dequant int8 inputs)")
    b.append(sig+" {")
    # Robust shape handling: derive per-input concat-dim from each INPUT shape, and the
    # total concat dim as their SUM (don't rely on osh[axis] being resolved -- newer
    # models may leave the Concat output shape unresolved).
    in_shapes=[ctx.shape(inp) for inp in ins]
    if axis>=len(osh) or not osh:
        # reconstruct output shape from the first input + summed concat dim
        base = in_shapes[0] if in_shapes and in_shapes[0] else []
        if not base:
            raise ValueError(f"Concat {node['name']}: cannot resolve shapes "
                             f"(inputs={in_shapes}, out={osh}). Run shape inference.")
        cat = sum((ash[axis] if axis<len(ash) else 1) for ash in in_shapes)
        osh_local = list(base); 
        if axis<len(osh_local): osh_local[axis]=cat
        else: osh_local.append(cat)
    else:
        osh_local = list(osh)
    cat_dim = osh_local[axis]
    outer=1
    for dax in osh_local[:axis]: outer*=dax
    inner=1
    for dax in osh_local[axis+1:]: inner*=dax
    b.append(f"  int outer={outer}, inner={inner}, off=0;")
    if extra:
        # and quantized for the QuantizeLinear(s) after it, as each value is placed
        # (fuse.py fold_quantize_into_producer); the float only if anything else reads it
        b.append("  " + also_q_consts(extra))
        keep = not all(e.get("float_dead") for e in extra)
    for i,inp in enumerate(ins):
        ash=in_shapes[i]; ad=ash[axis] if axis<len(ash) else 1
        if extra:
            if _isq(ctx,inp):
                sc,zp=_deq(ctx,inp); val=f"{cfloat(sc)}*(float)((int)in{i}[(o*ad+a)*inner+k]-({zp}))"
            else:
                val=f"in{i}[(o*ad+a)*inner+k]"
            dst=f"(o*{cat_dim}+(off+a))*inner+k"
            b.append(f"  {{ int ad={ad};")
            b.append(f"    for(int o=0;o<outer;++o) for(int a=0;a<ad;++a) for(int k=0;k<inner;++k){{")
            b.append(f"      float v={val}; " + (f"y[{dst}]=v; " if keep else "") + also_q_stores(extra, "v", dst) + " }")
            b.append(f"    off+=ad; }}")
            continue
        if _isq(ctx,inp):
            sc,zp=_deq(ctx,inp)
            b.append(f"  {{ int ad={ad}; const float s={cfloat(sc)}; const int z={zp};")
            b.append(f"    for(int o=0;o<outer;++o) for(int a=0;a<ad;++a) for(int k=0;k<inner;++k)")
            b.append(f"      y[(o*{cat_dim}+(off+a))*inner+k]=s*(float)((int)in{i}[(o*ad+a)*inner+k]-z);")
            b.append(f"    off+=ad; }}")
        else:
            # a run of ad*inner values per outer row: one memcpy, not a loop of three
            b.append(f"  {{ int ad={ad};")
            # (no copy where the input was made in its slab of y: gen_activations)
            b.append(f"    for(int o=0;o<outer;++o) if(y+(o*{cat_dim}+off)*inner!=in{i}+o*ad*inner) memcpy(y+(o*{cat_dim}+off)*inner, in{i}+o*ad*inner, ad*inner*sizeof(*y));")
            b.append(f"    off+=ad; }}")
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,list(ins)+[outs]+[e["dst"] for e in extra]))

def _emit_concat_lazy(ctx,node,axis):
    """A float channel Concat with lazy inputs (fuse.py mark_lazy_floats): each value
    made from its int8 where it is placed, then stored and/or quantized as the plain
    Concat does."""
    outs=node["outputs"][0]; fn=fsan(node["name"]); ins=node["inputs"]
    osh=ctx.shape(outs); C=osh[-1]; outer=numel(osh)//C
    if axis!=len(osh)-1:
        raise ValueError(f"Concat {node['name']}: a lazy input needs the channel axis, not {axis}")
    extra=also_q(node)
    keep = not (extra and all(e.get("float_dead") for e in extra))
    srcs=[]
    for i in ins:
        for s_ in sources(ctx,i):
            if s_ not in srcs: srcs.append(s_)
    params={t:f"s{j}" for j,t in enumerate(srcs)}
    lazy_begin(ctx)
    vals=[value(ctx,i,"o","a",params) for i in ins]
    args=[f"const {'int8_t' if ctx.T[t].get('dtype')=='int8' else 'float'}* s{j}" for j,t in enumerate(srcs)]
    args.append("float* y"); args+=[f"int8_t* q{k}" for k in range(len(extra))]
    sig=_sig(fn,args); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Concat {node['name']} axis={axis} of {len(ins)} (lazy floats made from their int8 here)")
    b.append(sig+" {")
    b.extend(lazy_prelude(ctx))
    if extra: b.append("  " + also_q_consts(extra))
    off=0
    for i,inp in enumerate(ins):
        ad=ctx.shape(inp)[-1]; dst=f"o*{C}+({off}+a)"
        b.append(f"  for(int o=0;o<{outer};++o) for(int a=0;a<{ad};++a){{ float v={vals[i]}; "
                 + (f"y[{dst}]=v; " if keep else "") + (also_q_stores(extra, "v", dst) if extra else "") + " }")
        off+=ad
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,srcs+[outs]+[e["dst"] for e in extra]))

def _emit_concat_int8(ctx,node,axis):
    """Concat whose inputs and output are int8 of one quantization (fuse.py
    int8_layout_paths): a byte copy, a quarter of the float one's traffic."""
    outs=node["outputs"][0]; fn=fsan(node["name"]); ins=node["inputs"]
    osh=ctx.shape(outs); oq=(ctx.T[outs]["q_scale"], ctx.T[outs]["q_zp"])
    for i in ins:
        t=ctx.T.get(i,{})
        if t.get("dtype")!="int8" or (t.get("q_scale"),t.get("q_zp"))!=oq:
            raise ValueError(f"Concat {node['name']}: int8 output but input {i} is not int8 at {oq}")
    args=[f"const int8_t* in{i}" for i in range(len(ins))]+["int8_t* y"]
    sig=_sig(fn,args); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Concat {node['name']} axis={axis} of {len(ins)} (int8, one quantization)")
    b.append(sig+" {")
    outer=1
    for dax in osh[:axis]: outer*=dax
    inner=1
    for dax in osh[axis+1:]: inner*=dax
    cat=osh[axis]
    b.append(f"  int off=0;")
    for i,inp in enumerate(ins):
        ad=ctx.shape(inp)[axis]
        b.append(f"  for(int o=0;o<{outer};++o) if(y+(o*{cat}+off)*{inner}!=in{i}+o*{ad*inner}) memcpy(y+(o*{cat}+off)*{inner}, in{i}+o*{ad*inner}, {ad*inner});")
        b.append(f"  off+={ad};")
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,list(ins)+[outs]))

@register("Split")
def emit_split(ctx,node):
    x=node["inputs"][0]; axis=node["attrs"].get("axis",0); fn=fsan(node["name"])
    xsh=ctx.shape(x); rank=len(xsh)
    if axis<0: axis+=rank
    outs=main_outputs(node)
    extra=also_q(node)
    sizes=node["attrs"].get("split", None)
    if sizes is None and len(node["inputs"])>1:
        # opset-13+ Split: sizes are the 2nd input tensor (now inlined as a value)
        sv=ctx.vals(node["inputs"][1])
        if sv: sizes=[int(v) for v in sv]
    if sizes is None:
        each=xsh[axis]//len(outs); sizes=[each]*len(outs)
    if is_lazy(ctx,x) or any(is_lazy(ctx,o) for o in outs):
        return _emit_split_lazy(ctx,node,x,xsh,axis,outs,extra,sizes)
    ct=_ct(ctx,x)
    args=[f"const {ct}* x"]+[f"{ct}* out{i}" for i in range(len(outs))]+[f"int8_t* q{k}" for k in range(len(extra))]
    sig=_sig(fn,args); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Split {node['name']} axis={axis} sizes={sizes}"); b.append(sig+" {")
    outer=1
    for d in xsh[:axis]: outer*=d
    inner=1
    for d in xsh[axis+1:]: inner*=d
    b.append(f"  int outer={outer}, inner={inner}, off=0;")
    if extra:
        b.append("  " + also_q_consts(extra))
    for i,sz in enumerate(sizes):
        mine=[(k,e) for k,e in enumerate(extra) if e["src"]==outs[i]]
        if mine:
            # this part quantized as it is copied, for the QuantizeLinear(s) after it
            keep = not all(e.get("float_dead") for _,e in mine)
            stores=" ".join(quant_c(f"q{k}[o*sd*inner+j]", "v", inv=f"inv{k}", z=f"z{k}") for k,_ in mine)
            b.append(f"  {{ int sd={sz}; for(int o=0;o<outer;++o) for(int j=0;j<sd*inner;++j){{")
            b.append(f"      float v=x[(o*{xsh[axis]}+off)*inner+j]; " + (f"out{i}[o*sd*inner+j]=v; " if keep else "") + stores + " }")
            b.append(f"    off+=sd; }}")
            continue
        b.append(f"  {{ int sd={sz}; for(int o=0;o<outer;++o)")
        b.append(f"      memcpy(out{i}+o*sd*inner, x+(o*{xsh[axis]}+off)*inner, sd*inner*sizeof(*x));")
        b.append(f"    off+=sd; }}")
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,[x]+list(outs)+[e["dst"] for e in extra]))

def _emit_split_lazy(ctx,node,x,xsh,axis,outs,extra,sizes):
    """A channel Split with a lazy input or output (fuse.py mark_lazy_floats): each
    value made from its int8 source(s) in the loop, the stored parts written as before,
    the lazy ones not at all, and each folded QuantizeLinear's int8 as before."""
    fn=fsan(node["name"]); C=xsh[-1]; outer=numel(xsh)//C
    if axis!=len(xsh)-1:
        raise ValueError(f"Split {node['name']}: a lazy tensor needs the channel axis, not {axis}")
    srcs=sources(ctx,x); params={t:f"s{j}" for j,t in enumerate(srcs)}
    stored=[(i,o) for i,o in enumerate(outs) if not is_lazy(ctx,o)]
    args=[f"const {_ct(ctx,t) if not _isq(ctx,t) else 'int8_t'}* s{j}" for j,t in enumerate(srcs)]
    args+=[f"float* out{i}" for i,_ in stored]+[f"int8_t* q{k}" for k in range(len(extra))]
    sig=_sig(fn,args); ctx.header.append(sig+";")
    b=ctx.body
    lazy_begin(ctx)
    vexpr=value(ctx,x,"o",f"off+j",params)   # tables first, above the function
    b.append(f"// Split {node['name']} axis={axis} sizes={sizes} (lazy floats: made from their int8 here)"); b.append(sig+" {")
    b.extend(lazy_prelude(ctx))
    if extra:
        b.append("  " + also_q_consts(extra))
    off=0
    for i,sz in enumerate(sizes):
        o_name=outs[i]
        mine=[(k,e) for k,e in enumerate(extra) if e["src"]==o_name]
        write=not is_lazy(ctx,o_name) and not (mine and all(e.get("float_dead") for _,e in mine))
        if not write and not mine:
            off+=sz; continue
        stores=" ".join(quant_c(f"q{k}[o*{sz}+j]", "v", inv=f"inv{k}", z=f"z{k}") for k,_ in mine)
        b.append(f"  {{ const int off={off}; for(int o=0;o<{outer};++o) for(int j=0;j<{sz};++j){{")
        b.append(f"      float v={vexpr}; " + (f"out{i}[o*{sz}+j]=v; " if write else "") + stores + " }")
        b.append("    }")
        off+=sz
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,srcs+[o for _,o in stored]+[e["dst"] for e in extra]))

# ---------------- Transpose (general N-D permute, static perm) ----------------
@register("Transpose")
def emit_transpose(ctx,node):
    x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"])
    xsh=ctx.shape(x); perm=node["attrs"].get("perm", list(range(len(xsh)-1,-1,-1)))
    # A Transpose it cannot place is an error, not a copy: a copy is only right by luck.
    # (graph_json now runs shape inference, and the NHWC pass no longer puts 4-D perms
    # on other ranks, so a model this stops really has something unsupported.)
    if not xsh or len(perm)!=len(xsh) or sorted(perm)!=list(range(len(xsh))):
        raise ValueError(f"Transpose {node['name']}: perm {perm} does not fit input shape {xsh}")
    ysh=[xsh[p] for p in perm]; n=numel(xsh)
    xstr=_strides(xsh); ystr=_strides(ysh)
    ct=_ct(ctx,x)
    blocks=_as_2d_transpose(xsh, perm)
    if blocks:
        batch, A, B = blocks
        sig=_sig(fn,[f"const {ct}* x",f"{ct}* y"]); ctx.header.append(sig+";")
        b=ctx.body; b.append(f"// Transpose {node['name']} perm={perm}: [{batch}][{A}][{B}] -> [{batch}][{B}][{A}], in tiles")
        b.append(sig+" {")
        # Element by element, one side of a transpose strides through memory a value at a
        # time -- on an MCU, a PSRAM cache line fetched for each value written. In
        # 16 x 16 tiles both sides stay within a few lines.
        b.append(f"  const int A={A}, B={B}, T=16;")
        b.append(f"  for(int n=0;n<{batch};++n){{ const {ct}* xb=x+n*A*B; {ct}* yb=y+n*A*B;")
        b.append(f"    for(int a0=0;a0<A;a0+=T) for(int b0=0;b0<B;b0+=T){{")
        b.append(f"      const int a1=a0+T<A?a0+T:A, b1=b0+T<B?b0+T:B;")
        b.append(f"      for(int a=a0;a<a1;++a) for(int c=b0;c<b1;++c) yb[c*A+a]=xb[a*B+c]; }} }}")
        b.append("}"); b.append("")
        ctx.calls.append((node,fn,["x","y"])); return
    sig=_sig(fn,[f"const {ct}* x",f"{ct}* y"]); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Transpose {node['name']} perm={perm}"); b.append(sig+" {")
    # emit nested loops over output index, map to input via perm
    idx=[f"i{d}" for d in range(len(ysh))]
    for d,ext in enumerate(ysh):
        b.append("  "+"  "*d+f"for(int {idx[d]}=0;{idx[d]}<{ext};++{idx[d]})")
    # input linear index: sum over output dim d of idx[d]*xstr[perm[d]]
    xin="+".join(f"{idx[d]}*{xstr[perm[d]]}" for d in range(len(ysh)))
    yout="+".join(f"{idx[d]}*{ystr[d]}" for d in range(len(ysh)))
    b.append("  "+"  "*len(ysh)+f"y[{yout}]=x[{xin}];")
    b.append("}"); b.append("")
    ctx.calls.append((node,fn,["x","y"]))

def _as_2d_transpose(shape, perm):
    """(batch, A, B) when [perm] keeps some leading dims and swaps the two groups of dims
    after them -- NHWC <-> NCHW, a matrix's last two axes -- so the transpose is of a
    batch of A x B matrices; else None."""
    k=0
    while k<len(perm) and perm[k]==k: k+=1
    rest=perm[k:]; r=len(perm)
    for m in range(k+1, r):
        if rest==list(range(m, r))+list(range(k, m)):
            batch=numel(shape[:k]) if k else 1
            return batch, numel(shape[k:m]), numel(shape[m:])
    return None

# ---------------- Reshape (pure copy; layout identical) ----------------
@register("Reshape")
def emit_reshape(ctx,node):
    x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"]); n=numel(ctx.shape(x))
    ct=_ct(ctx,x)
    sig=_sig(fn,[f"const {ct}* x",f"{ct}* y"]); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Reshape {node['name']} (memcpy)"); b.append(sig+" {")
    b.append(f"  if(y!=x) memcpy(y, x, {n}*sizeof(*x));")
    b.append("}"); b.append(""); ctx.calls.append((node,fn,["x","y"]))

# ---------------- MaxPool (NHWC, static kernel/stride) ----------------
@register("MaxPool")
def emit_maxpool(ctx,node):
    x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"]); a=node["attrs"]
    _,H,W,C=ctx.shape(x); _,Ho,Wo,_=ctx.shape(y)
    kh,kw=a.get("kernel_shape",[2,2]); sh,sw=a.get("strides",[2,2]); pads=a.get("pads",[0,0,0,0])
    pt,pl=pads[0],pads[1]; ct=_ct(ctx,x)
    sig=_sig(fn,[f"const {ct}* x",f"{ct}* y"]); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// MaxPool {node['name']} {kh}x{kw} s{sh}"); b.append(sig+" {")
    b.append(f"  const int H={H},W={W},C={C},Ho={Ho},Wo={Wo},kh={kh},kw={kw},sh={sh},sw={sw},pt={pt},pl={pl};")
    b.append("""  for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow)for(int c=0;c<C;++c){
    float m=-3.402823e38f;
    for(int i=0;i<kh;++i){int hi=oh*sh-pt+i; if(hi<0||hi>=H)continue;
      for(int j=0;j<kw;++j){int wi=ow*sw-pl+j; if(wi<0||wi>=W)continue;
        float v=x[(hi*W+wi)*C+c]; if(v>m)m=v; }}
    y[(oh*Wo+ow)*C+c]=m;
  }""")
    b.append("}"); b.append(""); ctx.calls.append((node,fn,["x","y"]))

# ---------------- Resize (NHWC, nearest or bilinear; scale from attr/const) ----------------
@register("Resize")
def emit_resize(ctx,node):
    x=node["inputs"][0]; y=node["outputs"][0]; fn=fsan(node["name"]); a=node["attrs"]
    _,H,W,C=ctx.shape(x); _,Ho,Wo,_=ctx.shape(y)
    mode=a.get("mode","nearest"); ct=_ct(ctx,x)
    sig=_sig(fn,[f"const {ct}* x",f"{ct}* y"]); ctx.header.append(sig+";")
    b=ctx.body; b.append(f"// Resize {node['name']} mode={mode} {H}x{W}->{Ho}x{Wo}"); b.append(sig+" {")
    b.append(f"  const int H={H},W={W},C={C},Ho={Ho},Wo={Wo};")
    b.append(f"  const float ry=(float)H/Ho, rx=(float)W/Wo;")
    if mode=="linear" or mode=="bilinear":
        b.append("""  for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow){
    float fy=(oh+0.5f)*ry-0.5f, fx=(ow+0.5f)*rx-0.5f;
    int y0=(int)floorf(fy), x0=(int)floorf(fx); float dy=fy-y0, dx=fx-x0;
    int y1=y0+1,x1=x0+1;
    if(y0<0)y0=0; if(x0<0)x0=0; if(y1>=H)y1=H-1; if(x1>=W)x1=W-1; if(y0>=H)y0=H-1; if(x0>=W)x0=W-1;
    for(int c=0;c<C;++c){
      float a=x[(y0*W+x0)*C+c],b=x[(y0*W+x1)*C+c],cc=x[(y1*W+x0)*C+c],d=x[(y1*W+x1)*C+c];
      float top=a+(b-a)*dx, bot=cc+(d-cc)*dx;
      y[(oh*Wo+ow)*C+c]=top+(bot-top)*dy;
    }
  }""")
    else:
        if not nearest_supported(a,H,W,Ho,Wo):
            raise ValueError(f"Resize {node['name']}: nearest {H}x{W}->{Ho}x{Wo} with "
                             f"coordinate_transformation_mode={a.get('coordinate_transformation_mode')}, "
                             f"nearest_mode={a.get('nearest_mode')} is not what this emitter computes")
        b.append("""  for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow){
    int sy=(int)(oh*ry), sx=(int)(ow*rx); if(sy>=H)sy=H-1; if(sx>=W)sx=W-1;
    for(int c=0;c<C;++c) y[(oh*Wo+ow)*C+c]=x[(sy*W+sx)*C+c];
  }""")
    b.append("}"); b.append(""); ctx.calls.append((node,fn,["x","y"]))


@register("Mul")
def emit_mul(ctx, node):
    a=node["inputs"][0]; b=node["inputs"][1]; y=node["outputs"][0]; fn=fsan(node["name"])
    tb=ctx.T.get(b,{})
    ny=numel(ctx.shape(y)); na=numel(ctx.shape(a)); ct=_ct(ctx,a)
    inlined = tb.get("value") is not None
    fileb   = (tb.get("const_file") is True) or (tb.get("kind")=="init")
    if inlined:
        vals=ctx.vals(b); nb=max(1,len(vals))
        sig=_sig(fn,["const "+ct+"* a", ct+"* y"]); ctx.header.append(sig+";")
        bd=ctx.body; bd.append("// Mul "+node["name"]+" (x * inlined const)"); bd.append(sig+" {")
        if nb==1:
            bd.append("  const float k="+cfloat(vals[0])+";")
            bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i]*k;")
        else:
            tbl=", ".join(cfloat(v) for v in vals)
            bd.append("  static const float k["+str(nb)+"]={"+tbl+"};")
            bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i]*k[i%"+str(nb)+"];")
        bd.append("}"); bd.append(""); ctx.calls.append((node,fn,[a,y]))
    elif fileb:
        nb=max(1,numel(ctx.shape(b))); bct=_ct(ctx,b)
        sig=_sig(fn,["const "+ct+"* a","const "+bct+"* b", ct+"* y"]); ctx.header.append(sig+";")
        bd=ctx.body; bd.append("// Mul "+node["name"]+" (x * file const)"); bd.append(sig+" {")
        bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i]*b[i%"+str(nb)+"];")
        bd.append("}"); bd.append(""); ctx.calls.append((node,fn,[a,b,y]))
    else:
        nb=max(1,numel(ctx.shape(b)))
        sig=_sig(fn,["const "+ct+"* a","const "+ct+"* b", ct+"* y"]); ctx.header.append(sig+";")
        bd=ctx.body; bd.append("// Mul "+node["name"]+" (elementwise)"); bd.append(sig+" {")
        bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i%"+str(na)+"]*b[i%"+str(nb)+"];")
        bd.append("}"); bd.append(""); ctx.calls.append((node,fn,[a,b,y]))


@register("Add")
def emit_add(ctx, node):
    a=node["inputs"][0]; b=node["inputs"][1]; y=node["outputs"][0]; fn=fsan(node["name"])
    tb=ctx.T.get(b,{})
    ny=numel(ctx.shape(y)); na=numel(ctx.shape(a)); ct=_ct(ctx,a)
    inlined = tb.get("value") is not None
    fileb   = (tb.get("const_file") is True) or (tb.get("kind")=="init")
    if inlined:
        vals=ctx.vals(b); nb=max(1,len(vals))
        sig=_sig(fn,["const "+ct+"* a", ct+"* y"]); ctx.header.append(sig+";")
        bd=ctx.body; bd.append("// Add "+node["name"]+" (x + inlined const)"); bd.append(sig+" {")
        if nb==1:
            bd.append("  const float k="+cfloat(vals[0])+";")
            bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i]+k;")
        else:
            tbl=", ".join(cfloat(v) for v in vals)
            bd.append("  static const float k["+str(nb)+"]={"+tbl+"};")
            bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i]+k[i%"+str(nb)+"];")
        bd.append("}"); bd.append(""); ctx.calls.append((node,fn,[a,y]))
    elif fileb:
        nb=max(1,numel(ctx.shape(b))); bct=_ct(ctx,b)
        sig=_sig(fn,["const "+ct+"* a","const "+bct+"* b", ct+"* y"]); ctx.header.append(sig+";")
        bd=ctx.body; bd.append("// Add "+node["name"]+" (x + file const)"); bd.append(sig+" {")
        bd.append("  for(int i=0;i<"+str(ny)+";++i) y[i]=a[i]+b[i%"+str(nb)+"];")
        bd.append("}"); bd.append(""); ctx.calls.append((node,fn,[a,b,y]))
    elif is_lazy(ctx,a) or is_lazy(ctx,b) or is_lazy(ctx,y):
        _emit_add_lazy(ctx,node,a,b,y,fn)
    else:
        nb=max(1,numel(ctx.shape(b)))
        aq=_isq(ctx,a); bq=_isq(ctx,b)
        act = "int8_t" if aq else ct
        bct = "int8_t" if bq else _ct(ctx,b)
        yq = ctx.T.get(y,{}).get("dtype")=="int8" and ctx.T[y].get("q_scale") is not None
        extra=also_q(node)
        sig=_sig(fn,["const "+act+"* a","const "+bct+"* b", ("int8_t* y" if yq else "float* y")]
                 +[f"int8_t* q{k}" for k in range(len(extra))]); ctx.header.append(sig+";")
        bd=ctx.body; bd.append("// Add "+node["name"]+" (elementwise, dequant if q"+(", int8 out" if yq else "")+")"); bd.append(sig+" {")
        ia = "i" if na==ny else f"i%{na}"
        ib = "i" if nb==ny else f"i%{nb}"
        if aq:
            sca,zpa=_deq(ctx,a); av=f"({cfloat(sca)}*(float)((int)a[{ia}] - ({zpa})))"
        else:
            av=f"a[{ia}]"
        if bq:
            scb,zpb=_deq(ctx,b); bv=f"({cfloat(scb)}*(float)((int)b[{ib}] - ({zpb})))"
        else:
            bv=f"b[{ib}]"
        if yq:
            # the float sum, quantized as the QuantizeLinear after it did (fuse.py int8_layout_paths)
            bd.append(f"  const float inv=1.0f/{cfloat(ctx.T[y]['q_scale'])}; const int z={int(ctx.T[y]['q_zp'])};")
            bd.append(f"  for(int i=0;i<{ny};++i){{ float sum={av}+{bv}; " + quant_c("y[i]", "sum") + " }")
        elif extra:
            # and quantized for the QuantizeLinear(s) after it, in the same pass
            keep = not all(e.get("float_dead") for e in extra)
            bd.append("  " + also_q_consts(extra))
            bd.append(f"  for(int i=0;i<{ny};++i){{ float sum={av}+{bv}; " + ("y[i]=sum; " if keep else "")
                      + also_q_stores(extra, "sum", "i") + " }")
        else:
            bd.append(f"  for(int i=0;i<{ny};++i) y[i]={av}+{bv};")
        bd.append("}"); bd.append(""); ctx.calls.append((node,fn,[a,b,y]+[e["dst"] for e in extra]))


def _emit_add_lazy(ctx,node,a,b,y,fn):
    """A same-shape Add with a lazy input or output (fuse.py mark_lazy_floats): each
    input made from its int8 in the loop, the float sum as the plain Add makes it,
    stored only if y is not lazy, and quantized for whatever was folded into it."""
    ysh=ctx.shape(y); C=ysh[-1]; outer=numel(ysh)//C
    if ctx.shape(a)!=ysh or ctx.shape(b)!=ysh:
        raise ValueError(f"Add {node['name']}: a lazy tensor needs equal shapes")
    yq = ctx.T.get(y,{}).get("dtype")=="int8" and ctx.T[y].get("q_scale") is not None
    extra=also_q(node); ylazy=is_lazy(ctx,y)
    srcs=[]
    for t in (a,b):
        for s_ in sources(ctx,t):
            if s_ not in srcs: srcs.append(s_)
    params={t:f"s{j}" for j,t in enumerate(srcs)}
    lazy_begin(ctx)
    av=value(ctx,a,"o","c",params); bv=value(ctx,b,"o","c",params)
    args=[f"const {'int8_t' if ctx.T[t].get('dtype')=='int8' else 'float'}* s{j}" for j,t in enumerate(srcs)]
    if not ylazy: args.append("int8_t* y" if yq else "float* y")
    args+=[f"int8_t* q{k}" for k in range(len(extra))]
    sig=_sig(fn,args); ctx.header.append(sig+";")
    bd=ctx.body; bd.append("// Add "+node["name"]+" (elementwise; lazy floats made from their int8 here)"); bd.append(sig+" {")
    bd.extend(lazy_prelude(ctx))
    if yq:
        bd.append(f"  const float inv=1.0f/{cfloat(ctx.T[y]['q_scale'])}; const int z={int(ctx.T[y]['q_zp'])};")
        bd.append(f"  for(int o=0;o<{outer};++o) for(int c=0;c<{C};++c){{ const int i=o*{C}+c; float sum={av}+{bv}; " + quant_c("y[i]", "sum") + " }")
    else:
        keep = not ylazy and not (extra and all(e.get("float_dead") for e in extra))
        if extra: bd.append("  " + also_q_consts(extra))
        bd.append(f"  for(int o=0;o<{outer};++o) for(int c=0;c<{C};++c){{ const int i=o*{C}+c; float sum={av}+{bv}; "
                  + ("y[i]=sum; " if keep else "") + (also_q_stores(extra, "sum", "i") if extra else "") + " }")
    bd.append("}"); bd.append("")
    ctx.calls.append((node,fn,srcs+([] if ylazy else [y])+[e["dst"] for e in extra]))


# ============================================================================
# Additional ops for attention-based / newer detection models (model-agnostic).
# ============================================================================

@register("Sub")
def emit_sub(ctx, node):
    """Elementwise A - B (with numpy-style broadcasting on leading dims)."""
    a, b = node["inputs"][0], node["inputs"][1]
    y = node["outputs"][0]; fn = fsan(node["name"])
    na = numel(ctx.shape(a)) or 1
    nb = numel(ctx.shape(b)) or 1
    ny = numel(ctx.shape(y)) or max(na, nb)
    cta = "int8_t" if _isq(ctx,a) else "float"
    ctb = "int8_t" if _isq(ctx,b) else "float"
    sig = _sig(fn, [f"const {cta}* a", f"const {ctb}* b", "float* y"])
    ctx.header.append(sig+";")
    bd = ctx.body; bd.append(f"// Sub {node['name']}  n={ny}")
    bd.append(sig+" {")
    # dequant helpers if needed
    aexpr = "(float)a[i%%%d]" % na if not _isq(ctx,a) else None
    if _isq(ctx,a):
        sca,zpa=_deq(ctx,a); aexpr=f"({cfloat(sca)}*(float)((int)a[i%{na}] - ({zpa})))"
    else:
        aexpr=f"a[i%{na}]"
    if _isq(ctx,b):
        scb,zpb=_deq(ctx,b); bexpr=f"({cfloat(scb)}*(float)((int)b[i%{nb}] - ({zpb})))"
    else:
        bexpr=f"b[i%{nb}]"
    bd.append(f"  for(int i=0;i<{ny};++i) y[i]={aexpr}-{bexpr};")
    bd.append("}"); bd.append("")
    ctx.calls.append((node, fn, [a, b, y]))


@register("Slice")
def emit_slice(ctx, node):
    """Slice along one axis. starts/ends/axes/steps come from inputs (opset-10+),
    which are inlined small-int metadata. Handles a single-axis contiguous slice
    with step 1 (the common case); falls back to copy if params are dynamic."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    xsh = ctx.shape(x); ysh = ctx.shape(y)
    ins = node["inputs"]
    def _v(i):
        if i < len(ins):
            vv = ctx.vals(ins[i])
            return list(vv) if vv else None
        return None
    starts = _v(1); ends = _v(2); axes = _v(3); steps = _v(4)
    ct = "int8_t" if _isq(ctx,x) else "float"
    octy = "float"
    sig = _sig(fn, [f"const {ct}* x", f"{octy}* y"])
    ctx.header.append(sig+";")
    bd = ctx.body
    rank = len(xsh)
    # resolve axis (default 0), start, end, step for a SINGLE-axis slice
    ax = (axes[0] if axes else 0)
    if ax < 0: ax += rank
    st = (starts[0] if starts else 0)
    en = (ends[0] if ends else (xsh[ax] if ax < len(xsh) else 0))
    stp = (steps[0] if steps else 1)
    if st < 0: st += xsh[ax]
    if en < 0: en += xsh[ax]
    if en > xsh[ax]: en = xsh[ax]
    # strides
    outer = 1
    for d in xsh[:ax]: outer *= d
    inner = 1
    for d in xsh[ax+1:]: inner *= d
    xdim = xsh[ax]; ydim = max(0, (en - st + (stp-1)) // stp)
    bd.append(f"// Slice {node['name']} axis={ax} [{st}:{en}:{stp}] of dim {xdim}")
    bd.append(sig+" {")
    bd.append(f"  int outer={outer}, inner={inner};")
    if _isq(ctx,x):
        sc,zp=_deq(ctx,x)
        bd.append(f"  const float s={cfloat(sc)}; const int z={zp};")
        bd.append(f"  for(int o=0;o<outer;++o){{ int yi=0;")
        bd.append(f"    for(int a={st};a<{en};a+={stp}){{ for(int k=0;k<inner;++k)")
        bd.append(f"      y[(o*{ydim}+yi)*inner+k]=s*(float)((int)x[(o*{xdim}+a)*inner+k]-z);")
        bd.append(f"    ++yi; }} }}")
    else:
        bd.append(f"  for(int o=0;o<outer;++o){{ int yi=0;")
        bd.append(f"    for(int a={st};a<{en};a+={stp}){{ for(int k=0;k<inner;++k)")
        bd.append(f"      y[(o*{ydim}+yi)*inner+k]=(float)x[(o*{xdim}+a)*inner+k];")
        bd.append(f"    ++yi; }} }}")
    bd.append("}"); bd.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("Softmax")
def emit_softmax(ctx, node):
    """Softmax along `axis` (default -1). Numerically stable (subtract max)."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    xsh = ctx.shape(x); rank = len(xsh); axis = node["attrs"].get("axis", -1)
    if axis < 0: axis += rank
    ct = "int8_t" if _isq(ctx,x) else "float"
    sig = _sig(fn, [f"const {ct}* x", "float* y"])
    ctx.header.append(sig+";")
    outer = 1
    for d in xsh[:axis]: outer *= d
    ad = xsh[axis]
    inner = 1
    for d in xsh[axis+1:]: inner *= d
    bd = ctx.body
    bd.append(f"// Softmax {node['name']} axis={axis} (ad={ad})")
    bd.append(sig+" {")
    bd.append(f"  int outer={outer}, ad={ad}, inner={inner};")
    if _isq(ctx,x):
        sc,zp=_deq(ctx,x); rd=f"({cfloat(sc)}*(float)((int)x[idx] - ({zp})))"
    else:
        rd="x[idx]"
    bd.append(f"  for(int o=0;o<outer;++o) for(int k=0;k<inner;++k){{")
    bd.append(f"    float m=-1e30f;")
    bd.append(f"    for(int a=0;a<ad;++a){{ int idx=(o*ad+a)*inner+k; float v={rd}; if(v>m)m=v; }}")
    bd.append(f"    float s=0.f;")
    bd.append(f"    for(int a=0;a<ad;++a){{ int idx=(o*ad+a)*inner+k; float v={rd}; float e=expf(v-m); y[idx]=e; s+=e; }}")
    bd.append(f"    float inv=1.f/s;")
    bd.append(f"    for(int a=0;a<ad;++a){{ int idx=(o*ad+a)*inner+k; y[idx]*=inv; }}")
    bd.append(f"  }}")
    bd.append("}"); bd.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("MatMul")
def emit_matmul(ctx, node):
    """Batched MatMul: [...,M,K] x [...,K,N] -> [...,M,N].
    Leading (batch) dims must match or broadcast trivially (equal here)."""
    a, b = node["inputs"][0], node["inputs"][1]
    y = node["outputs"][0]; fn = fsan(node["name"])
    ash = ctx.shape(a); bsh = ctx.shape(b); ysh = ctx.shape(y)
    # M,K from a ; K,N from b (last two dims)
    M = ash[-2]; K = ash[-1]; N = bsh[-1]
    batch = 1
    for d in ash[:-2]: batch *= d
    cta = "int8_t" if _isq(ctx,a) else "float"
    ctb = "int8_t" if _isq(ctx,b) else "float"
    sig = _sig(fn, [f"const {cta}* A", f"const {ctb}* B", "float* Y"])
    ctx.header.append(sig+";")
    bd = ctx.body
    bd.append(f"// MatMul {node['name']}  batch={batch} M={M} K={K} N={N}")
    bd.append(sig+" {")
    bd.append(f"  int BATCH={batch}, M={M}, K={K}, N={N};")
    if _isq(ctx,a):
        sca,zpa=_deq(ctx,a); ard=f"({cfloat(sca)}*(float)((int)A[(bt*M+i)*K+p] - ({zpa})))"
    else:
        ard="A[(bt*M+i)*K+p]"
    if _isq(ctx,b):
        scb,zpb=_deq(ctx,b); brd=f"({cfloat(scb)}*(float)((int)B[(bt*K+p)*N+j] - ({zpb})))"
    else:
        brd="B[(bt*K+p)*N+j]"
    bd.append(f"  for(int bt=0;bt<BATCH;++bt)")
    bd.append(f"   for(int i=0;i<M;++i) for(int j=0;j<N;++j){{")
    bd.append(f"     float acc=0.f;")
    bd.append(f"     for(int p=0;p<K;++p) acc+={ard}*{brd};")
    bd.append(f"     Y[(bt*M+i)*N+j]=acc; }}")
    bd.append("}"); bd.append("")
    ctx.calls.append((node, fn, [a, b, y]))


# ============================================================================
# Ops for NMS-free / end2end detection heads (TopK selection). Static shapes.
# Index tensors are emitted as int (int32). float tensors as float.
# ============================================================================

def _is_int_tensor(ctx, name):
    dt = ctx.T.get(name, {}).get("dtype", "")
    return dt in ("int64", "int32", "int16", "int8_idx")

@register("Cast")
def emit_cast(ctx, node):
    _assert_not_quantized(ctx, node, *node["inputs"])
    """Cast: copy with C type conversion. to=1 float, to=7 int64, to=6 int32, etc."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    to = node["attrs"].get("to", 1)
    n = numel(ctx.shape(y)) or numel(ctx.shape(x)) or 1
    # dst type: 1=float32 -> float ; ints -> int
    dst_float = (to == 1 or to == 10 or to == 11)
    src_int = _is_int_tensor(ctx, x)
    ct_src = "int32_t" if src_int else "float"
    ct_dst = "float" if dst_float else "int32_t"
    # mark output dtype
    if y in ctx.T: ctx.T[y]["dtype"] = "float32" if dst_float else "int32"
    sig = _sig(fn, [f"const {ct_src}* x", f"{ct_dst}* y"])
    ctx.header.append(sig+";")
    b = ctx.body; b.append(f"// Cast {node['name']} to={to}  n={n}")
    b.append(sig+" {")
    b.append(f"  for(int i=0;i<{n};++i) y[i]=({ct_dst})x[i];")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("Div")
def emit_div(ctx, node):
    _assert_not_quantized(ctx, node, *node["inputs"])
    """Elementwise A / B. If B is an inlined constant, emit it as a literal so no
    weight array is needed. Integer div when both inputs are int (index math)."""
    a, b = node["inputs"][0], node["inputs"][1]
    y = node["outputs"][0]; fn = fsan(node["name"])
    na = numel(ctx.shape(a)) or 1; nb = numel(ctx.shape(b)) or 1
    ny = numel(ctx.shape(y)) or max(na, nb)
    ai = _is_int_tensor(ctx, a); bi = _is_int_tensor(ctx, b)
    both_int = ai and bi
    cta = "int32_t" if ai else "float"
    octy = "int32_t" if both_int else "float"
    if y in ctx.T: ctx.T[y]["dtype"] = "int32" if both_int else "float32"
    bval = ctx.vals(b)  # inlined constant divisor?
    if bval is not None and len(bval)>=1:
        lit = int(bval[0]) if both_int else float(bval[0])
        litstr = str(lit) if both_int else cfloat(lit)
        sig = _sig(fn, [f"const {cta}* a", f"{octy}* y"])
        ctx.header.append(sig+";")
        bd = ctx.body; bd.append(f"// Div {node['name']} n={ny} by const {lit}")
        bd.append(sig+" {")
        bd.append(f"  for(int i=0;i<{ny};++i) y[i]=a[i%{na}]/({octy}){litstr};")
        bd.append("}"); bd.append("")
        ctx.calls.append((node, fn, [a, y]))
    else:
        ctb = "int32_t" if bi else "float"
        sig = _sig(fn, [f"const {cta}* a", f"const {ctb}* b", f"{octy}* y"])
        ctx.header.append(sig+";")
        bd = ctx.body; bd.append(f"// Div {node['name']} n={ny}")
        bd.append(sig+" {")
        bd.append(f"  for(int i=0;i<{ny};++i) y[i]=a[i%{na}]/b[i%{nb}];")
        bd.append("}"); bd.append("")
        ctx.calls.append((node, fn, [a, b, y]))


@register("Mod")
def emit_mod(ctx, node):
    """Elementwise A % B (integer). Inlined constant modulus -> literal (no array)."""
    a, b = node["inputs"][0], node["inputs"][1]
    y = node["outputs"][0]; fn = fsan(node["name"])
    na = numel(ctx.shape(a)) or 1; nb = numel(ctx.shape(b)) or 1
    ny = numel(ctx.shape(y)) or max(na, nb)
    if y in ctx.T: ctx.T[y]["dtype"] = "int32"
    bval = ctx.vals(b)
    if bval is not None and len(bval)>=1:
        m=int(bval[0])
        sig = _sig(fn, ["const int32_t* a", "int32_t* y"])
        ctx.header.append(sig+";")
        bd = ctx.body; bd.append(f"// Mod {node['name']} n={ny} by const {m}")
        bd.append(sig+" {")
        bd.append(f"  for(int i=0;i<{ny};++i){{ int32_t r=a[i%{na}]%{m}; if(r<0)r+={m}; y[i]=r; }}")
        bd.append("}"); bd.append("")
        ctx.calls.append((node, fn, [a, y]))
    else:
        sig = _sig(fn, ["const int32_t* a", "const int32_t* b", "int32_t* y"])
        ctx.header.append(sig+";")
        bd = ctx.body; bd.append(f"// Mod {node['name']} n={ny}")
        bd.append(sig+" {")
        bd.append(f"  for(int i=0;i<{ny};++i){{ int32_t d=b[i%{nb}]; int32_t r=a[i%{na}]%d; if(r<0)r+=d; y[i]=r; }}")
        bd.append("}"); bd.append("")
        ctx.calls.append((node, fn, [a, b, y]))


@register("Flatten")
def emit_flatten(ctx, node):
    """Flatten: pure reshape (copy). Data layout unchanged in row-major."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    n = numel(ctx.shape(x)) or numel(ctx.shape(y)) or 1
    xi = _is_int_tensor(ctx, x)
    i8 = ctx.T.get(x, {}).get("dtype") in ("int8", "uint8")   # quantized bytes stay bytes
    ct = _ct(ctx, x) if i8 else ("int32_t" if xi else "float")
    if y in ctx.T and not i8: ctx.T[y]["dtype"] = "int32" if xi else "float32"
    sig = _sig(fn, [f"const {ct}* x", f"{ct}* y"])
    ctx.header.append(sig+";")
    b = ctx.body; b.append(f"// Flatten {node['name']} n={n} (copy)")
    b.append(sig+" {"); b.append(f"  if(y!=x) memcpy(y, x, {n}*sizeof(*x));")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("Unsqueeze")
def emit_unsqueeze(ctx, node):
    """Unsqueeze: insert size-1 dims. Data unchanged -> copy."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    n = numel(ctx.shape(x)) or numel(ctx.shape(y)) or 1
    xi = _is_int_tensor(ctx, x)
    i8 = ctx.T.get(x, {}).get("dtype") in ("int8", "uint8")   # quantized bytes stay bytes
    ct = _ct(ctx, x) if i8 else ("int32_t" if xi else "float")
    if y in ctx.T and not i8: ctx.T[y]["dtype"] = "int32" if xi else "float32"
    sig = _sig(fn, [f"const {ct}* x", f"{ct}* y"])
    ctx.header.append(sig+";")
    b = ctx.body; b.append(f"// Unsqueeze {node['name']} n={n} (copy)")
    b.append(sig+" {"); b.append(f"  if(y!=x) memcpy(y, x, {n}*sizeof(*x));")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("Squeeze")
def emit_squeeze(ctx, node):
    """Squeeze: drop size-1 dims. Data unchanged -> copy."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    n = numel(ctx.shape(x)) or numel(ctx.shape(y)) or 1
    xi = _is_int_tensor(ctx, x)
    i8 = ctx.T.get(x, {}).get("dtype") in ("int8", "uint8")   # quantized bytes stay bytes
    ct = _ct(ctx, x) if i8 else ("int32_t" if xi else "float")
    if y in ctx.T and not i8: ctx.T[y]["dtype"] = "int32" if xi else "float32"
    sig = _sig(fn, [f"const {ct}* x", f"{ct}* y"])
    ctx.header.append(sig+";")
    b = ctx.body; b.append(f"// Squeeze {node['name']} n={n} (copy)")
    b.append(sig+" {"); b.append(f"  if(y!=x) memcpy(y, x, {n}*sizeof(*x));")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("Expand")
def emit_expand(ctx, node):
    _assert_not_quantized(ctx, node, *node["inputs"])
    """Expand (broadcast) x to target shape. Target from 2nd input (const shape).
    Implemented for the common case: broadcasting trailing dim (e.g. [1,300,1]->[1,300,80])."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    xsh = ctx.shape(x); ysh = ctx.shape(y)
    nx = numel(xsh) or 1; nyt = numel(ysh) or 1
    xi = _is_int_tensor(ctx, x)
    i8 = ctx.T.get(x, {}).get("dtype") in ("int8", "uint8")   # quantized bytes stay bytes
    ct = _ct(ctx, x) if i8 else ("int32_t" if xi else "float")
    if y in ctx.T and not i8: ctx.T[y]["dtype"] = "int32" if xi else "float32"
    sig = _sig(fn, [f"const {ct}* x", f"{ct}* y"])
    ctx.header.append(sig+";")
    b = ctx.body
    # broadcast factor = ny/nx (each x element repeated rep times along broadcast dim).
    # General row-major broadcast: map each output index back to input by integer-div.
    # For trailing-dim broadcast [.., 1] -> [.., R], rep=R and x index = i / R.
    rep = nyt // nx if nx else 1
    b.append(f"// Expand {node['name']} {xsh}->{ysh} rep={rep}")
    b.append(sig+" {")
    # trailing broadcast: out[i] = x[i / rep_inner] -- but only valid if broadcast is the
    # innermost dim. Detect: last dim of x is 1 and last of y is R.
    if xsh and ysh and len(xsh)==len(ysh) and xsh[-1]==1 and ysh[-1]>1:
        inner = ysh[-1]
        outer = nx  # number of x elements (each becomes `inner` copies)
        b.append(f"  for(int o=0;o<{outer};++o) for(int r=0;r<{inner};++r) y[o*{inner}+r]=x[o];")
    else:
        # generic fallback: tile (repeat whole array) -- correct when x fully broadcasts
        b.append(f"  for(int i=0;i<{nyt};++i) y[i]=x[i%{nx}];")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("ReduceMax")
def emit_reducemax(ctx, node):
    _assert_not_quantized(ctx, node, *node["inputs"])
    """ReduceMax along axis (from 2nd input or attr), keepdims handled by output shape."""
    x = node["inputs"][0]; y = node["outputs"][0]; fn = fsan(node["name"])
    xsh = ctx.shape(x); rank = len(xsh)
    axv = ctx.vals(node["inputs"][1]) if len(node["inputs"])>1 else None
    axis = int(axv[0]) if axv else node["attrs"].get("axes",[rank-1])[0]
    if axis < 0: axis += rank
    outer = 1
    for d in xsh[:axis]: outer *= d
    ad = xsh[axis]
    inner = 1
    for d in xsh[axis+1:]: inner *= d
    xi = _is_int_tensor(ctx, x)
    i8 = ctx.T.get(x, {}).get("dtype") in ("int8", "uint8")   # quantized bytes stay bytes
    ct = _ct(ctx, x) if i8 else ("int32_t" if xi else "float")
    if y in ctx.T and not i8: ctx.T[y]["dtype"] = "int32" if xi else "float32"
    sig = _sig(fn, [f"const {ct}* x", f"{ct}* y"])
    ctx.header.append(sig+";")
    b = ctx.body; b.append(f"// ReduceMax {node['name']} axis={axis} ad={ad}")
    b.append(sig+" {")
    b.append(f"  int outer={outer}, ad={ad}, inner={inner};")
    b.append(f"  for(int o=0;o<outer;++o) for(int k=0;k<inner;++k){{")
    b.append(f"    {ct} m=x[(o*ad+0)*inner+k];")
    b.append(f"    for(int a=1;a<ad;++a){{ {ct} v=x[(o*ad+a)*inner+k]; if(v>m)m=v; }}")
    b.append(f"    y[o*inner+k]=m; }}")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [x, y]))


@register("TopK")
def emit_topk(ctx, node):
    """TopK along axis (last dim here). Two outputs: values (float) + indices (int).
    k from 2nd input (constant). largest+sorted assumed (YOLO usage). Static k."""
    x = node["inputs"][0]; vals_out = node["outputs"][0]; idx_out = node["outputs"][1]
    fn = fsan(node["name"])
    xsh = ctx.shape(x); rank = len(xsh)
    axis = node["attrs"].get("axis", -1)
    if axis < 0: axis += rank
    kv = ctx.vals(node["inputs"][1]) if len(node["inputs"])>1 else None
    K = int(kv[0]) if kv else ctx.shape(vals_out)[axis]
    # collapse to (outer, ad) with axis last (YOLO TopK is on last dim)
    outer = 1
    for d in xsh[:axis]: outer *= d
    ad = xsh[axis]
    if vals_out in ctx.T: ctx.T[vals_out]["dtype"] = "float32"
    if idx_out in ctx.T:  ctx.T[idx_out]["dtype"]  = "int32"
    xi = _is_int_tensor(ctx, x)
    ct = "int32_t" if xi else "float"
    sig = _sig(fn, [f"const {ct}* x", "float* vals", "int32_t* idx"])
    ctx.header.append(sig+";")
    b = ctx.body
    b.append(f"// TopK {node['name']} axis={axis} K={K} ad={ad} outer={outer} (largest,sorted)")
    b.append(sig+" {")
    b.append(f"  const int OUTER={outer}, AD={ad}, K={K};")
    if ad >= K:
        # A K-entry heap, worst at the root: one pass over AD, K*log K to sort,
        # instead of K passes over AD (YOLO26's class TopK: 24000 x 300). Ranked
        # exactly as repeated max ranks: higher value first, a tie to the lower index.
        b.append(f"  static float hv[{K}]; static int32_t hi[{K}];")
        b.append(f"  #define TK_WORSE(i,j) (hv[i]<hv[j] || (hv[i]==hv[j] && hi[i]>hi[j]))")
        b.append(f"  for(int o=0;o<OUTER;++o){{")
        b.append(f"    const {ct}* xo=x+o*AD; int n=0;")
        b.append(f"    for(int a=0;a<AD;++a){{ float v=(float)xo[a];")
        b.append(f"      if(n<K){{ int c=n++; hv[c]=v; hi[c]=a;")
        b.append(f"        while(c>0){{ int p=(c-1)/2; if(!TK_WORSE(c,p)) break;")
        b.append(f"          float tv=hv[c]; hv[c]=hv[p]; hv[p]=tv; int32_t ti=hi[c]; hi[c]=hi[p]; hi[p]=ti; c=p; }}")
        b.append(f"        continue; }}")
        b.append(f"      if(!(v>hv[0])) continue;   /* a tie loses: its index is higher */")
        b.append(f"      hv[0]=v; hi[0]=a; int c=0;")
        b.append(f"      for(;;){{ int l=2*c+1, r=l+1, m=c;")
        b.append(f"        if(l<K && TK_WORSE(l,m)) m=l;")
        b.append(f"        if(r<K && TK_WORSE(r,m)) m=r;")
        b.append(f"        if(m==c) break;")
        b.append(f"        float tv=hv[c]; hv[c]=hv[m]; hv[m]=tv; int32_t ti=hi[c]; hi[c]=hi[m]; hi[m]=ti; c=m; }}")
        b.append(f"    }}")
        b.append(f"    /* pop the worst into the last place, K times: best first */")
        b.append(f"    for(int last=K-1; last>=0; --last){{")
        b.append(f"      vals[o*K+last]=hv[0]; idx[o*K+last]=hi[0];")
        b.append(f"      hv[0]=hv[last]; hi[0]=hi[last]; int c=0;")
        b.append(f"      for(;;){{ int l=2*c+1, r=l+1, m=c;")
        b.append(f"        if(l<last && TK_WORSE(l,m)) m=l;")
        b.append(f"        if(r<last && TK_WORSE(r,m)) m=r;")
        b.append(f"        if(m==c) break;")
        b.append(f"        float tv=hv[c]; hv[c]=hv[m]; hv[m]=tv; int32_t ti=hi[c]; hi[c]=hi[m]; hi[m]=ti; c=m; }}")
        b.append(f"    }}")
        b.append(f"  }}")
        b.append(f"  #undef TK_WORSE")
        b.append("}"); b.append("")
        ctx.calls.append((node, fn, [x, vals_out, idx_out]))
        return
    b.append(f"  for(int o=0;o<OUTER;++o){{")
    b.append(f"    /* simple selection of top-K by repeated max (K<<AD) */")
    b.append(f"    static char used[{ad}];")
    b.append(f"    for(int a=0;a<AD;++a) used[a]=0;")
    b.append(f"    for(int t=0;t<K;++t){{")
    b.append(f"      int best=-1; float bv=-1e30f;")
    b.append(f"      for(int a=0;a<AD;++a){{ if(used[a]) continue; float v=(float)x[o*AD+a]; if(v>bv){{bv=v;best=a;}} }}")
    b.append(f"      used[best]=1; vals[o*K+t]=bv; idx[o*K+t]=best; }}")
    b.append(f"  }}")
    b.append("}"); b.append("")
    # multi-output: driver must pass x, vals, idx
    ctx.calls.append((node, fn, [x, vals_out, idx_out]))


@register("GatherElements")
def emit_gatherelements(ctx, node):
    _assert_not_quantized(ctx, node, *node["inputs"])
    """GatherElements along axis: out[i][j] = data[...idx[i][j]...] per ONNX spec.
    out has SAME shape as indices; each index selects along `axis` of data."""
    data = node["inputs"][0]; idx = node["inputs"][1]
    y = node["outputs"][0]; fn = fsan(node["name"])
    dsh = ctx.shape(data); ish = ctx.shape(idx); rank = len(dsh)
    axis = node["attrs"].get("axis", 0)
    if axis < 0: axis += rank
    di = _is_int_tensor(ctx, data)
    ctd = "int32_t" if di else "float"
    if y in ctx.T: ctx.T[y]["dtype"] = "int32" if di else "float32"
    # strides for data and index (row-major). Index shape drives output.
    def strides(sh):
        st=[1]*len(sh)
        for i in range(len(sh)-2,-1,-1): st[i]=st[i+1]*sh[i+1]
        return st
    dst = strides(dsh); ist = strides(ish)
    sig = _sig(fn, [f"const {ctd}* data", "const int32_t* idx", f"{ctd}* y"])
    ctx.header.append(sig+";")
    b = ctx.body
    b.append(f"// GatherElements {node['name']} axis={axis} data{dsh} idx{ish}")
    b.append(sig+" {")
    nout = numel(ish) or 1
    # iterate every output/index element via multi-index; map to data offset replacing
    # the `axis` coordinate with idx value.
    b.append(f"  const int NOUT={nout};")
    b.append(f"  int ish[{len(ish)}]={{{','.join(str(v) for v in ish)}}};")
    b.append(f"  int dst[{len(dst)}]={{{','.join(str(v) for v in dst)}}};")
    b.append(f"  int ist[{len(ist)}]={{{','.join(str(v) for v in ist)}}};")
    b.append(f"  for(int lin=0;lin<NOUT;++lin){{")
    b.append(f"    int32_t rem=lin, coord[{len(ish)}];")
    b.append(f"    for(int d=0;d<{len(ish)};++d){{ coord[d]=rem/ist[d]; rem%=ist[d]; }}")
    b.append(f"    int gi=idx[lin];")
    b.append(f"    int doff=0;")
    b.append(f"    for(int d=0;d<{len(ish)};++d){{ int c=(d=={axis})?gi:coord[d]; doff+=c*dst[d]; }}")
    b.append(f"    y[lin]=data[doff]; }}")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn, [data, idx, y]))
