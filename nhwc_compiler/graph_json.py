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
graph_json.py  —  Stage 1 of the NHWC int8 compiler (model-agnostic).

  ONNX (quantized, NCHW)  --build-->  graph.json (NCHW)  --nhwc-->  graph_nhwc.json

Layout classification is PURELY BY OP TYPE — no model-specific names anywhere.
Every op is one of:
   Conv/QLinearConv     -> produce NHWC (weights flagged OHWI for the weight gen)
   PASSTHROUGH          -> layout-transparent, keep whatever came in
   SPATIAL_KEEP         -> spatial but transparent to our kernels, stay NHWC
   AXIS_REMAP           -> stay NHWC, remap channel axis 1->3 (only when in NHWC)
   everything else      -> revert to NCHW before it (SAFE default for unknown ops;
                           this is how reshape/transpose/matmul "heads" are handled
                           structurally, without matching any node name)

Unknown op => NCHW revert: always correct, at worst an extra transpose. Watch the
'transposes inserted' count; a spike means some interior op should be added to
PASSTHROUGH/SPATIAL_KEEP.

Usage:
  python graph_json.py build  model_quant.onnx   graph.json
  python graph_json.py nhwc    graph.json         graph_nhwc.json
"""
import sys, json, argparse, onnx
from onnx import numpy_helper

ONNX_DT = {1:"float32",2:"uint8",3:"int8",6:"int32",7:"int64",9:"bool",10:"float16",11:"float64"}

# ---------------- Stage 1a: ONNX -> JSON (verbatim, NCHW) ----------------
def build(onnx_path, out_json):
    m = onnx.load(onnx_path); g = m.graph
    # Every emitter needs its tensors' shapes. A model saved without them (most
    # quantizer outputs) gets ONNX's own shape inference first, rather than the
    # QLinearConv emitter failing on an empty shape.
    known = {vi.name for vi in list(g.input) + list(g.output) + list(g.value_info)}
    known |= {it.name for it in g.initializer}
    if any(o and o not in known for n in g.node if n.op_type != "Constant" for o in n.output):
        m = onnx.shape_inference.infer_shapes(m); g = m.graph
    tensors = {}
    def dt(elem): return ONNX_DT.get(elem, f"elem{elem}")
    NP2={"int8":"int8","uint8":"uint8","int32":"int32","int64":"int64",
         "float32":"float32","float64":"float64"}
    def dt2(npdt): return NP2.get(str(npdt), str(npdt))
    def shp(vi):
        d=[]
        for x in vi.type.tensor_type.shape.dim:
            d.append(x.dim_value if x.HasField("dim_value") else -1)
        return d

    for vi in list(g.input)+list(g.output)+list(g.value_info):
        tensors[vi.name] = {"shape":shp(vi), "dtype":dt(vi.type.tensor_type.elem_type),
                            "kind":"activation", "layout":"nchw", "const_file":False}
    inits={}
    INLINE_MAX=256   # inline scale/zp/small-bias values; big weights stay file-backed

    # ---- collect ALL constant values: initializers + Constant nodes ----
    const_vals={}   # name -> numpy array
    for it in g.initializer:
        const_vals[it.name]=numpy_helper.to_array(it)
    for cn in g.node:
        if cn.op_type=="Constant":
            for a in cn.attribute:
                if a.name=="value":
                    const_vals[cn.output[0]]=numpy_helper.to_array(a.t)

    # Tensors consumed as SHAPE/AXIS/SIZE metadata (emitters read their VALUE).
    # Only these small-int tensors get inlined; everything else (conv bias, Div/Mod/Sub
    # operands, etc.) stays a file-backed array the driver passes as a pointer.
    META_SLOTS = {
        ("Split",1),
        ("Slice",1),("Slice",2),("Slice",3),("Slice",4),
        ("Reshape",1),("Unsqueeze",1),("Squeeze",1),
        ("ReduceMax",1),("ReduceMin",1),("ReduceSum",1),("ReduceMean",1),
        ("TopK",1),("Expand",1),
    }
    _meta_names=set()
    for _n in g.node:
        for _idx,_inp in enumerate(_n.input):
            if (_n.op_type,_idx) in META_SLOTS:
                _meta_names.add(_inp)

    for name,arr in const_vals.items():
        entry = {"shape":list(arr.shape), "dtype":dt2(arr.dtype),
                 "kind":"init", "layout":"none", "const_file":True}
        # inline only SMALL FLOAT/scalar consts (scales, zp). Keep bias/int arrays
        # file-backed: they are per-channel data the kernels index at runtime.
        is_intdata = dt2(arr.dtype) in ("int32","int64") and arr.size>1
        is_weightdata = dt2(arr.dtype) in ("int8","uint8") and arr.size>1   # conv weights: never inline
        # small int64/int32 metadata (split sizes, shapes) up to 8 elems: DO inline as value
        is_small_meta = (dt2(arr.dtype) in ("int32","int64") and arr.size<=8
                         and name in _meta_names)
        if is_small_meta:
            entry["value"]=arr.flatten().tolist(); entry["const_file"]=False
        elif arr.size <= INLINE_MAX and not is_intdata and not is_weightdata:
            entry["value"] = arr.flatten().tolist()
            entry["const_file"] = False
        tensors[name]=entry
        inits[name]=True
    for vi in g.input:
        if vi.name not in inits: tensors[vi.name]["kind"]="input"
    for vi in g.output:
        tensors[vi.name]["kind"]="output"

    nodes=[]
    for n in g.node:
        attrs={}
        for a in n.attribute:
            if a.type==a.INT: attrs[a.name]=a.i
            elif a.type==a.INTS: attrs[a.name]=list(a.ints)
            elif a.type==a.FLOAT: attrs[a.name]=a.f
            elif a.type==a.STRING: attrs[a.name]=a.s.decode()
        nodes.append({"name":n.name, "op":n.op_type,
                      "inputs":list(n.input), "outputs":list(n.output), "attrs":attrs})

    doc={"tensors":tensors, "nodes":nodes,
         "inputs":[v.name for v in g.input], "outputs":[v.name for v in g.output],
         "onnx_path":onnx_path}
    json.dump(doc, open(out_json,"w"), indent=1)
    dyn=sum(1 for t in tensors.values() if any((not isinstance(x,int)) or x<0 for x in t["shape"]))
    print(f"build -> {out_json}: {len(nodes)} nodes, {len(tensors)} tensors, {dyn} with dynamic/unknown dims")

# ---------------- op classification (by TYPE only) ----------------
PASSTHROUGH={"Relu","LeakyRelu","Clip","Sigmoid","Tanh","HardSwish","HardSigmoid","Elu",
             "Mul","Add","Sub","Div","Pow","Sqrt","Abs","Neg","Exp","Log","Min","Max","Mean",
             "QuantizeLinear","DequantizeLinear","QLinearAdd","QLinearMul","QLinearSigmoid",
             "QLinearLeakyRelu","Identity","Cast","Erf","Floor","Round"}
SPATIAL_KEEP={"MaxPool","AveragePool","GlobalAveragePool","Resize","Upsample","Pad","LpPool",
              "QLinearAveragePool","QLinearGlobalAveragePool"}
AXIS_REMAP={"Concat","Split","Softmax","LogSoftmax","QLinearConcat"}
# everything not above -> revert to NCHW (safe default; unknown ops included)

# ---------------- Stage 1b: JSON NCHW -> JSON NHWC ----------------
def _relabel(T,name,layout):
    # auto-create a tensor entry if build() didn't record it (missing value_info)
    if name not in T:
        T[name]={"shape":[], "dtype":"float32", "kind":"activation",
                 "layout":layout, "const_file":False}
        return
    t=T[name]; t["layout"]=layout
    if layout=="nhwc" and len(t.get("shape",[]))==4:
        s=t["shape"]; t["shape"]=[s[0],s[2],s[3],s[1]]     # NCHW dims -> NHWC dims

def nhwc_pass(in_json, out_json):
    d=json.load(open(in_json))
    T=d["tensors"]; state={}; new_nodes=[]; cnt=[0]
    for name,t in T.items():
        state[name] = "const" if t["kind"]=="init" else (
            "nchw" if t["kind"] in ("input","activation","output") else "none")

    def fresh(b): cnt[0]+=1; return f"{b}_T{cnt[0]}"
    def perm_shape(sh,perm):
        return [sh[p] for p in perm] if len(sh)==4 else sh
    def add_transpose(src, perm, want):
        if new_nodes and new_nodes[-1]["op"]=="Transpose" and new_nodes[-1]["outputs"][0]==src:
            pv=new_nodes[-1]; pp=pv["attrs"]["perm"]
            if pp==([0,3,1,2] if perm==[0,2,3,1] else [0,2,3,1]):
                new_nodes.pop(); inp=pv["inputs"][0]; state[inp]=want; return inp
        o=fresh(src); src_sh=T[src]["shape"]
        T[o]={"shape":perm_shape(src_sh,perm),"dtype":T[src]["dtype"],
              "kind":"activation","layout":want,"const_file":False}
        new_nodes.append({"name":o,"op":"Transpose","inputs":[src],"outputs":[o],"attrs":{"perm":perm}})
        state[o]=want; return o
    def ensure(name, want):
        cur=state.get(name,"nchw")
        if cur in ("const","none",want): return name
        # Only a 4-D tensor has an NCHW or NHWC layout. A tensor known to have another
        # rank is the same bytes either way: relabel it instead of inserting a 4-D
        # Transpose its emitter could only turn into a copy. (An unknown shape keeps the
        # Transpose, and the Transpose emitter stops the build.)
        sh=T.get(name,{}).get("shape") or []
        if sh and len(sh)!=4 and all(isinstance(v,int) and v>=0 for v in sh):
            state[name]=want; return name
        if cur=="nchw" and want=="nhwc": return add_transpose(name,[0,2,3,1],"nhwc")
        if cur=="nhwc" and want=="nchw": return add_transpose(name,[0,3,1,2],"nchw")
        return name

    # flag conv weights for OHWI transpose in the weight generator (every conv, no name filter)
    for n in d["nodes"]:
        # flag conv weights for OHWI transpose AND update their shape metadata to OHWI
        # [K,Cp,Kh,Kw] -> [K,Kh,Kw,Cp], so downstream emitters read correct K,Kh,Kw,Cp.
        def _mark_ohwi(wname, group=1):
            t=T.get(wname)
            if not t or t.get("ohwi") or t.get("depthwise"): return
            sh=t.get("shape",[])
            # Depthwise: OIHW with in-ch==1 (per-group) and group==out-ch.
            # Mark distinctly so weight-gen repacks to the layout each conv path needs,
            # instead of the generic OIHW->OHWI (which would transpose the kh/kw kernel).
            is_dw = (len(sh)==4 and group>1 and sh[0]==group and sh[1]==1)
            if is_dw:
                t["depthwise"]=True
                # keep logical OHWI shape [K,Kh,Kw,1] for downstream shape reads
                t["shape"]=[sh[0],sh[2],sh[3],sh[1]]
                return
            t["ohwi"]=True
            if len(sh)==4:                      # OIHW -> OHWI
                t["shape"]=[sh[0],sh[2],sh[3],sh[1]]
        if n["op"]=="QLinearConv": _mark_ohwi(n["inputs"][3], n["attrs"].get("group",1))
        elif n["op"]=="Conv":      _mark_ohwi(n["inputs"][1], n["attrs"].get("group",1))

    def notconst(x): return state.get(x,"const") not in ("const","none")

    for n in d["nodes"]:
        op=n["op"]; nn=dict(n); nn["inputs"]=list(n["inputs"]); nn["attrs"]=dict(n["attrs"])

        if op in ("QLinearConv","Conv"):
            nn["inputs"][0]=ensure(n["inputs"][0],"nhwc")
            for o in n["outputs"]: state[o]="nhwc"; _relabel(T,o,"nhwc")

        elif op in PASSTHROUGH:
            base=state.get(n["inputs"][0],"const"); base="nhwc" if base in ("const","none") else base
            for i,inp in enumerate(n["inputs"]):
                if notconst(inp): nn["inputs"][i]=ensure(inp,base)
            for o in n["outputs"]: state[o]=base; _relabel(T,o,base)

        elif op in SPATIAL_KEEP:
            base=state.get(n["inputs"][0],"nhwc"); base="nhwc" if base in ("const","none") else base
            nn["inputs"][0]=ensure(n["inputs"][0],base)
            for o in n["outputs"]: state[o]=base; _relabel(T,o,base)

        elif op in AXIS_REMAP:
            base=state.get(n["inputs"][0],"const"); base="nhwc" if base in ("const","none") else base
            if base=="nhwc":
                for i,inp in enumerate(n["inputs"]):
                    if notconst(inp): nn["inputs"][i]=ensure(inp,"nhwc")
                # Axis remap NCHW->NHWC applies ONLY to 4-D tensors, where channel axis
                # 1 becomes axis 3. For non-4D tensors (e.g. rank-3 head tensors) the
                # NHWC relabel does not move channels, so the axis must stay unchanged.
                rank=len(T.get(n["inputs"][0],{}).get("shape",[]) or [])
                if "axis" in nn["attrs"] and rank==4:
                    if   nn["attrs"]["axis"]==1:  nn["attrs"]["axis"]=3
                    elif nn["attrs"]["axis"]==-3: nn["attrs"]["axis"]=-1
                for o in n["outputs"]: state[o]="nhwc"; _relabel(T,o,"nhwc")
            else:
                for i,inp in enumerate(n["inputs"]):
                    if notconst(inp): nn["inputs"][i]=ensure(inp,"nchw")
                for o in n["outputs"]: state[o]="nchw"; T[o]["layout"]="nchw"

        else:  # unknown / layout-breaking (Reshape, Transpose, MatMul, Gather, ...) -> NCHW
            for i,inp in enumerate(n["inputs"]):
                if notconst(inp): nn["inputs"][i]=ensure(inp,"nchw")
            for o in n["outputs"]: state[o]="nchw"; T[o]["layout"]="nchw"

        new_nodes.append(nn)

    # graph outputs back to NCHW to preserve external contract
    outs=[]
    for o in d["outputs"]:
        if state.get(o)=="nhwc":
            t=fresh(o)
            sh=T[o]["shape"]
            T[t]={"shape":[sh[p] for p in [0,3,1,2]] if len(sh)==4 else sh,
                  "dtype":T[o]["dtype"],"kind":"output","layout":"nchw","const_file":False}
            new_nodes.append({"name":t,"op":"Transpose","inputs":[o],"outputs":[t],"attrs":{"perm":[0,3,1,2]}})
            T[o]["kind"]="activation"; outs.append(t)
        else: outs.append(o)
    d["outputs"]=outs; d["nodes"]=new_nodes

    n_tr=sum(1 for x in new_nodes if x["op"]=="Transpose")
    json.dump(d, open(out_json,"w"), indent=1)
    print(f"nhwc -> {out_json}: transposes inserted = {n_tr}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd", required=True)
    b=sub.add_parser("build"); b.add_argument("onnx"); b.add_argument("out")
    h=sub.add_parser("nhwc"); h.add_argument("inp"); h.add_argument("out")
    a=ap.parse_args()
    if a.cmd=="build": build(a.onnx, a.out)
    else: nhwc_pass(a.inp, a.out)
