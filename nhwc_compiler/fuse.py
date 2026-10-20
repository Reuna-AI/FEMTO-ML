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
fuse.py — Stage (between nhwc pass and codegen): graph-level fusion.
SEPARATE pass, operates only on the JSON. Emitter stays fusion-unaware except
for the SiLU node (which it maps to the 256-entry LUT you validated).

Fusions implemented (graph rewrite only; no writeback fusion yet):
  1. SiLU:  x*sigmoid(x)  -> single 'SiLU' node.
       float form :  ...->[DequantizeLinear D]->Sigmoid S ; Mul M(inputs: S.out, D.out) ->[QuantizeLinear Q]
       quant form :  QLinearSigmoid S ; QLinearMul M(S.out, x)  (both from same source)
     The SiLU node carries in_scale/in_zp (pre-activation) and out_scale/out_zp
     (post) so the emitter can build the LUT.
  2. Q/DQ cancel: adjacent DequantizeLinear->QuantizeLinear (or Q->DQ) with matching
     scale+zp and nothing else consuming the middle tensor -> drop both.

Model-agnostic: matches by op type + dataflow, never by node name.

  python fuse.py graph_nhwc.json graph_fused.json
"""
import sys, json, argparse

# ---- Fusion directives (edit here; build-time toggles, not CLI flags) ----
# Fold the QuantizeLinear that follows a SiLU into the SiLU's int8 LUT, removing ~50
# redundant full-tensor requant passes. Set False for the original SiLU-float + separate-Q
# behavior (byte-identical to the pre-fusion working build).
FUSE_SILU_QUANTIZE = True
# Quantize where a value is made rather than after the ops that only move it (see
# int8_layout_paths). False for the float layout ops of before October 2026.
INT8_LAYOUT = True
# --------------------------------------------------------------------------

def build_maps(d):
    producer={}   # tensor -> node that writes it
    consumers={}  # tensor -> [nodes that read it]
    for n in d["nodes"]:
        for o in n["outputs"]: producer[o]=n
        for i in n["inputs"]: consumers.setdefault(i,[]).append(n)
    return producer, consumers

def qparams_of_Q(node):
    # QuantizeLinear inputs: [x, y_scale, y_zp]  -> returns (scale_name, zp_name)
    return (node["inputs"][1], node["inputs"][2] if len(node["inputs"])>2 else None)
def qparams_of_DQ(node):
    # DequantizeLinear inputs: [x, x_scale, x_zp]
    return (node["inputs"][1], node["inputs"][2] if len(node["inputs"])>2 else None)

def fuse_silu(d):
    prod,cons=build_maps(d)
    remove=set(); add=[]; rewrite={}   # fused-away tensor -> replacement
    silu_n=0

    for n in d["nodes"]:
        # ---- quantized form: QLinearMul fed by QLinearSigmoid ----
        if n["op"]=="QLinearMul":
            # QLinearMul inputs: A,A_s,A_zp, B,B_s,B_zp, C_s,C_zp
            a=n["inputs"][0]; b=n["inputs"][3]
            pa=prod.get(a); pb=prod.get(b)
            sig = pa if (pa and pa["op"]=="QLinearSigmoid") else (pb if (pb and pb["op"]=="QLinearSigmoid") else None)
            if sig:
                x = sig["inputs"][0]                       # sigmoid's input (pre-activation)
                other = b if sig is pa else a              # the identity branch
                if other == x or prod.get(other) is prod.get(x):
                    in_s, in_zp = sig["inputs"][1], sig["inputs"][2]      # X_scale,X_zp of sigmoid
                    out_s, out_zp = n["inputs"][6], n["inputs"][7]        # C_scale,C_zp of mul
                    o=n["outputs"][0]
                    add.append({"name":o+"_silu","op":"SiLU","inputs":[x],"outputs":[o],
                                "attrs":{"in_scale":in_s,"in_zp":in_zp,"out_scale":out_s,"out_zp":out_zp}})
                    remove.add(id(n)); remove.add(id(sig)); silu_n+=1
                    continue

        # ---- float form: Mul fed by Sigmoid, both from same (dequantized) source ----
        if n["op"]=="Mul":
            a,b=n["inputs"][0],n["inputs"][1]
            pa=prod.get(a); pb=prod.get(b)
            sig = pa if (pa and pa["op"]=="Sigmoid") else (pb if (pb and pb["op"]=="Sigmoid") else None)
            if sig:
                sx=sig["inputs"][0]                        # sigmoid input
                other=b if sig is pa else a
                if other==sx:                              # x * sigmoid(x)
                    # trace quant params: the source is usually a DequantizeLinear D
                    D=prod.get(sx)
                    q_after = None
                    for c in cons.get(n["outputs"][0],[]):
                        if c["op"]=="QuantizeLinear": q_after=c
                    # The DequantizeLinear is dropped below, so nothing but this
                    # Sigmoid and Mul may read it.
                    only_silu = D is not None and all(
                        c is sig or c is n for c in cons.get(sx, []))
                    if D and D["op"]=="DequantizeLinear" and only_silu:
                        in_s,in_zp = qparams_of_DQ(D)
                        x_int = D["inputs"][0]                 # int8 conv output
                        if FUSE_SILU_QUANTIZE and q_after and len(cons.get(n["outputs"][0],[]))==1:
                            out_s,out_zp = qparams_of_Q(q_after)
                            o = q_after["outputs"][0]          # int8 output (feeds next conv)
                            # FUSED: int8 -> SiLU -> int8 (absorb Dequant + Quantize).
                            # Only when the Quantize is the Mul's sole reader: a float
                            # reader too (a residual Add, a Split) needs the float form.
                            rewrite[n["outputs"][0]] = o
                            add.append({"name":o+"_silu","op":"SiLU","inputs":[x_int],"outputs":[o],
                                        "attrs":{"in_scale":in_s,"in_zp":in_zp,
                                                 "out_scale":out_s,"out_zp":out_zp}})
                            remove.add(id(n)); remove.add(id(sig)); remove.add(id(D)); remove.add(id(q_after))
                        else:
                            # UNFUSED: SiLU reads INT8 conv output (LUT is int8-indexed) and
                            # OUTPUTS FLOAT (no out params); a separate Quantize, if any,
                            # requantizes. With none -- the SiLU feeding a Split or a
                            # residual Add in float -- the LUT still replaces the float
                            # Dequantize + Sigmoid (expf) + Mul over the whole tensor.
                            # The SiLU LUT does the dequant internally, so DROP the DequantizeLinear
                            # too, but KEEP the QuantizeLinear (it reads the SiLU float output).
                            silu_out = n["outputs"][0]             # Mul's float output (Q reads it)
                            add.append({"name":silu_out+"_silu","op":"SiLU","inputs":[x_int],
                                        "outputs":[silu_out],
                                        "attrs":{"in_scale":in_s,"in_zp":in_zp}})  # NO out params -> float
                            remove.add(id(n)); remove.add(id(sig)); remove.add(id(D))  # drop D, keep q_after
                            if silu_out in d["tensors"]: d["tensors"][silu_out]["dtype"]="float32"
                        silu_n+=1
                        continue

    new_nodes=[nd for nd in d["nodes"] if id(nd) not in remove]
    new_nodes += add
    for nd in new_nodes:
        nd["inputs"]=[rewrite.get(i,i) for i in nd["inputs"]]
    # graph outputs may also reference a fused-away name
    d["outputs"]=[rewrite.get(o,o) for o in d.get("outputs",[])]
    d["nodes"]=new_nodes

    # register SiLU output dtype: int8 only if FUSED (has out params); else float.
    # CRITICAL: also attach q_scale/q_zp to int8 SiLU outputs so downstream float ops
    # (Add/Concat/Mul residuals) know how to dequantize them. Without this, an int8 SiLU
    # output added to a float tensor is read as raw ints -> garbage.
    for a in add:
        o=a["outputs"][0]
        if o not in d["tensors"]: continue
        if "out_scale" in a["attrs"]:
            d["tensors"][o]["dtype"]="int8"
            # resolve the actual scale/zp values from the referenced tensors
            os_name=a["attrs"]["out_scale"]; oz_name=a["attrs"]["out_zp"]
            def _val(nm):
                t=d["tensors"].get(nm,{}); v=t.get("value")
                return (v[0] if isinstance(v,list) and v else v)
            sc=_val(os_name); zp=_val(oz_name)
            if sc is not None: d["tensors"][o]["q_scale"]=float(sc)
            if zp is not None: d["tensors"][o]["q_zp"]=int(zp)
        else:
            d["tensors"][o]["dtype"]="float32"
    return silu_n


def fuse_silu_quantize(d):
    """Remove QuantizeLinear nodes that directly follow a SiLU when the SiLU already
    carries matching out_scale/out_zp. The SiLU emitter folds the quantize into its
    int8 LUT, so the separate Q is a redundant full-tensor pass. Fusing it away
    eliminates one read+write over the whole activation per SiLU (~50 in YOLO26)."""
    prod,cons=build_maps(d)
    remove=set(); rewrite={}; fused=0
    for n in d["nodes"]:
        if n["op"]!="QuantizeLinear": continue
        src=n["inputs"][0]; p=prod.get(src)
        if not p or p["op"]!="SiLU": continue
        at=p["attrs"]
        if "out_scale" not in at or "out_zp" not in at: continue
        # require SiLU.out_scale/zp == Q.scale/zp (same tensor names => exact match)
        q_s,q_zp = n["inputs"][1], n["inputs"][2]
        if at["out_scale"]!=q_s or at["out_zp"]!=q_zp:
            continue
        # rewire: SiLU should now write DIRECTLY to the Q's output tensor.
        q_out=n["outputs"][0]
        p["outputs"]=[q_out]                    # SiLU emits int8 into the Q's output name
        p["attrs"]["fuse_q"]=1                   # tag: emitter should use int8 fused path
        rewrite[src]=q_out                      # anyone reading SiLU's old output -> Q output
        remove.add(id(n)); fused+=1
    if fused:
        new=[nd for nd in d["nodes"] if id(nd) not in remove]
        for nd in new:
            nd["inputs"]=[rewrite.get(i,i) for i in nd["inputs"]]
        d["outputs"]=[rewrite.get(o,o) for o in d.get("outputs",[])]
        d["nodes"]=new
        # mark fused SiLU outputs int8
        for nd in d["nodes"]:
            if nd["op"]=="SiLU":
                o=nd["outputs"][0]
                if o in d["tensors"]: d["tensors"][o]["dtype"]="int8"
    return fused


def _silu_drop_unfused_outparams(d):
    """For any SiLU still followed by a QuantizeLinear (i.e. NOT fused), remove its
    out_scale/out_zp attrs so the emitter falls back to FLOAT output and the separate
    Q does the requant. Keeps graph + emitter consistent when fusion is off/partial."""
    prod={o:n for n in d["nodes"] for o in n["outputs"]}
    cons={}
    for n in d["nodes"]:
        for i in n["inputs"]: cons.setdefault(i,[]).append(n)
    dropped=0
    for n in d["nodes"]:
        if n["op"]!="SiLU": continue
        o=n["outputs"][0]
        has_q=any(c["op"]=="QuantizeLinear" for c in cons.get(o,[]))
        if has_q and ("out_scale" in n["attrs"] or "out_zp" in n["attrs"]):
            n["attrs"].pop("out_scale",None); n["attrs"].pop("out_zp",None)
            if o in d["tensors"]: d["tensors"][o]["dtype"]="float32"
            dropped+=1
    return dropped

def widen_logit_range(d):
    """Give class-logit convs their positive range back.

    A static-quantized detector's class head is calibrated on images where almost
    every anchor is background, so its logits are almost all negative and the
    calibrated int8 range ends at 0: zero point 127, the int8 maximum. Every
    positive logit -- every confident detection -- is then clamped to 0, and the
    Sigmoid after it can never give more than 0.5: a clear person and a doubtful one
    both score 0.500, and the TopK that ranks anchors by score cannot tell them apart.

    The conv's int32 accumulator holds the true logit; only the requantize clamps it.
    So for a QLinearConv whose int8 output is read only by DequantizeLinear(s) that
    lead, through shape-only ops, to a Sigmoid, and whose zero point is 127, the zero
    point is moved to 0. The scale is kept -- so are the requant multipliers -- and
    the range becomes [-128*s, +127*s] instead of [-255*s, 0]. What is given up is
    logits below -128*s (sigmoid < 1e-5 at the scales seen): nothing a threshold
    could ever pass. Scores below 0.5 are unchanged; above it they are now real.
    """
    prod,cons=build_maps(d)
    T=d["tensors"]
    SHAPE_ONLY={"Reshape","Concat","Transpose","Flatten","Squeeze","Unsqueeze","Identity"}
    widened=0

    def reaches_sigmoid(t, depth=0):
        if depth>6: return False
        for c in cons.get(t,[]):
            if c["op"]=="Sigmoid": return True
            if c["op"] in SHAPE_ONLY and any(reaches_sigmoid(o, depth+1) for o in c["outputs"]):
                return True
        return False

    for n in d["nodes"]:
        if n["op"]!="QLinearConv" or len(n["inputs"])<8: continue
        zp_name=n["inputs"][7]; zt=T.get(zp_name,{})
        v=zt.get("value")
        if not (isinstance(v,list) and len(v)==1 and v[0]==127 and zt.get("dtype")=="int8"): continue
        readers=cons.get(n["outputs"][0],[])
        if not readers or any(r["op"]!="DequantizeLinear" or len(r["inputs"])<3
                              or r["inputs"][2]!=zp_name for r in readers): continue
        # the zero point tensor must belong to this conv and its Dequantizes alone
        if any(u is not n and u not in readers for u in cons.get(zp_name,[])): continue
        if not any(reaches_sigmoid(r["outputs"][0]) for r in readers): continue
        zt["value"]=[0]
        widened+=1
    return widened

# Ops that only move values (or, for MaxPool, pick among them): on int8 they give
# exactly the int8 of what they give on float, quantizing being monotonic.
_LAYOUT = {"Concat", "Split", "Reshape", "Transpose", "MaxPool", "Resize"}

def _data_inputs(n):
    """The inputs of a layout op that carry values (not shapes, scales, sizes)."""
    if n["op"] == "Concat":
        return [i for i in n["inputs"] if i]
    return [n["inputs"][0]] if n["inputs"] else []

def _qparams(d, q):
    """(scale, zero point) of a QuantizeLinear, as numbers, or None if not inlined."""
    T = d["tensors"]
    def v0(name):
        v = T.get(name, {}).get("value")
        return v[0] if isinstance(v, list) and len(v) == 1 else None
    s = v0(q["inputs"][1]) if len(q["inputs"]) > 1 else None
    z = v0(q["inputs"][2]) if len(q["inputs"]) > 2 else 0
    if s is None or z is None or T.get(q["inputs"][2], {}).get("dtype", "int8") != "int8":
        return None
    return (float(s), int(z))

def int8_layout_paths(d):
    """Quantize at the source instead of after the layout ops.

    A QDQ graph is float between convs: a SiLU or residual Add writes float, Split /
    Concat / Reshape / MaxPool / Resize move those floats, and a QuantizeLinear at the
    end makes the next conv's int8. On an MCU those float tensors are 4 bytes a value
    in PSRAM, and moving and requantizing them was a third of YOLO26n's run.

    When every path from a float tensor leads, through layout ops only, to
    QuantizeLinear(s) of the same scale and zero point, the tensor's producer quantizes
    as it writes (SiLU through its LUT, Add after its float add, DequantizeLinear as an
    int8->int8 table), the layout ops move int8, and those QuantizeLinears go. The
    values are bit-identical: the producer computes the same float and quantizes it
    with the QuantizeLinear's own C expression, and moving or taking the max of
    quantized values gives the quantized moved or max value. A tensor that anything
    else reads (a residual Add, a MatMul, the graph's output) stays float.
    """
    prod, cons = build_maps(d)
    T = d["tensors"]
    outputs = set(d.get("outputs", []))
    PRODUCERS = {"SiLU", "Add", "DequantizeLinear"}

    def is_float_act(t):
        m = T.get(t, {})
        return m.get("dtype") == "float32" and m.get("kind") != "init" and t in prod

    def producer_ok(t):
        p = prod[t]
        if p["op"] == "SiLU":
            return "out_scale" not in p["attrs"]          # the float-output form
        if p["op"] == "Add":                              # two activations, float out
            return all(is_float_act(i) or (T.get(i, {}).get("dtype") == "int8" and i in prod)
                       for i in p["inputs"][:2])
        if p["op"] == "DequantizeLinear":
            return _qparams(d, {"inputs": p["inputs"]}) is not None
        if p["op"] == "Resize":
            return p["attrs"].get("mode", "nearest") == "nearest"
        return p["op"] in _LAYOUT

    banned = set()
    while True:
        # what each float tensor would be quantized to, from its readers back
        want = {}
        changed = True
        while changed:
            changed = False
            for t in list(T):
                if t in want or t in banned or t in outputs or not is_float_act(t):
                    continue
                readers = cons.get(t, [])
                if not readers:
                    continue
                params = set()
                for c in readers:
                    if c["op"] == "QuantizeLinear" and c["inputs"][0] == t:
                        qp = _qparams(d, c)
                        params.add(qp if qp else "no")
                    elif c["op"] in _LAYOUT and t in _data_inputs(c) and all(
                            o in want for o in c["outputs"]):
                        params.update(want[o] for o in c["outputs"])
                    else:
                        params.add("no")
                if len(params) == 1 and "no" not in params:
                    want[t] = params.pop()
                    changed = True
        # which of those can be written as int8: their producer can, or is a layout
        # op all of whose value inputs are int8 of the same quantization
        ok = {t for t in want if producer_ok(t)}
        changed = True
        while changed:
            changed = False
            for t in list(ok):
                p = prod[t]
                if p["op"] in _LAYOUT and not all(
                        i in ok and want[i] == want[t] for i in _data_inputs(p)):
                    ok.discard(t); changed = True
                if p["op"] in _LAYOUT and not all(o in ok for o in p["outputs"]):
                    ok.discard(t); changed = True
        new_banned = banned | (set(want) - ok)
        if new_banned == banned:
            break
        banned = new_banned

    # rewrite: int8 tensors, the QuantizeLinears they make redundant gone
    for t in ok:
        s_, z_ = want[t]
        T[t]["dtype"] = "int8"
        T[t]["q_scale"] = s_
        T[t]["q_zp"] = z_
    remove = set(); rewrite = {}
    for n in d["nodes"]:
        if n["op"] == "QuantizeLinear" and n["inputs"][0] in ok:
            rewrite[n["outputs"][0]] = n["inputs"][0]
            remove.add(id(n))
    d["nodes"] = [n for n in d["nodes"] if id(n) not in remove]
    for n in d["nodes"]:
        n["inputs"] = [rewrite.get(i, i) for i in n["inputs"]]
    d["outputs"] = [rewrite.get(o, o) for o in d.get("outputs", [])]
    return len(ok), len(remove)


def sigmoid_through_layout(d):
    """A Sigmoid whose input is DequantizeLinear outputs moved by layout ops only (YOLO
    class scores: three heads, reshaped and concatenated) becomes a table in each
    DequantizeLinear: int8 in, sigmoid(dequant) out, filled once on the device with the
    same C expression -- so the same floats, without an expf per value. The Sigmoid
    node stays, as a copy."""
    prod, cons = build_maps(d)
    done = 0
    for n in d["nodes"]:
        if n["op"] != "Sigmoid":
            continue
        sources, stack, fine = [], [n["inputs"][0]], True
        while stack and fine:
            t = stack.pop()
            p = prod.get(t)
            if p is None:
                fine = False
            elif p["op"] == "DequantizeLinear":
                sources.append(p)
            elif p["op"] in ("Reshape", "Concat", "Transpose", "Flatten") and \
                    all(len(cons.get(o, [])) == 1 for o in p["outputs"]):
                stack.extend(_data_inputs(p))
            else:
                fine = False
            # nothing else may read what is about to become a sigmoid (the graph's
            # caller included)
            if fine and (len(cons.get(t, [])) != 1 or t in d.get("outputs", [])):
                fine = False
        if not fine or not sources:
            continue
        for p in sources:
            p["attrs"]["then_sigmoid"] = 1
        n["attrs"]["identity"] = 1
        done += 1
    return done


FOLD_INTO = ("Concat", "SiLU", "Split", "Add")

def fold_quantize_into_producer(d):
    """A QuantizeLinear that int8_layout_paths could not take away -- its float input is
    also read at another scale, or by a residual Add -- is folded into what makes that
    float: the Concat, SiLU, Split or Add writes the int8 too, in the same pass, so the
    float is not read back from PSRAM to be quantized. Where nothing else reads the
    float (a Concat that only fed this QuantizeLinear), it is not written at all.
    The same float, quantized by the QuantizeLinear's own expression: the same bytes.
    The int8 outputs are listed among the producer's outputs, so the arena keeps them
    alive from there."""
    prod, cons = build_maps(d)
    T = d["tensors"]
    remove = set(); folded = 0
    for q in d["nodes"]:
        if q["op"] != "QuantizeLinear":
            continue
        t = q["inputs"][0]; p = prod.get(t)
        if p is None or T.get(t, {}).get("dtype") != "float32":
            continue
        if p["op"] == "SiLU" and "out_scale" in p["attrs"]:
            continue
        if p["op"] == "Add" and not all(i in prod for i in p["inputs"][:2]):
            continue                                  # a constant operand: another emitter form
        if p["op"] not in FOLD_INTO:
            continue
        qp = _qparams(d, q)
        if qp is None:
            continue
        p["attrs"].setdefault("also_q", []).append(
            {"src": t, "dst": q["outputs"][0], "scale": qp[0], "zp": qp[1]})
        p["outputs"].append(q["outputs"][0])
        remove.add(id(q)); folded += 1
    d["nodes"] = [n for n in d["nodes"] if id(n) not in remove]
    prod, cons = build_maps(d)
    outputs = set(d.get("outputs", []))
    for p in d["nodes"]:
        for e in p["attrs"].get("also_q", []):
            e["float_dead"] = not cons.get(e["src"]) and e["src"] not in outputs
    return folded


def _concat_axis(d, n):
    """A Concat's axis as a non-negative number, or None if its shapes do not resolve."""
    sh = d["tensors"].get(n["outputs"][0], {}).get("shape") or []
    ax = n["attrs"].get("axis", 0)
    if not sh or not -len(sh) <= ax < len(sh):
        return None
    return ax + len(sh) if ax < 0 else ax


def cancel_layout_transposes(d):
    """Two transposes that undo each other, with only data movement between them, both go.

    A detector head (YOLO26n's) takes each scale's NHWC conv output to NCHW
    (Transpose [0,3,1,2] + Reshape to [N,C,H*W]), concatenates the scales along H*W,
    adds the boxes along C, and then transposes the lot back ([0,2,1]) and splits it
    into boxes and scores again. Here every [N,C,L] tensor on that path is followed
    as "the transpose of an [N,L,C] one we already have": an NHWC tensor read as
    [N,H*W,C] is that already, a Concat along L of such tensors is a Concat along L
    of theirs, an identity Sigmoid changes nothing, and the final [0,2,1] Transpose
    is the [N,L,C] tensor itself. A Split that cuts a Concat back into the pieces it
    was made of is those pieces. The values are only moved, never computed: the same
    bytes, without the transposes and the float copies between them.

    Model-agnostic (op types and shapes only); nothing changes in a graph without
    the pattern. Returns how many transposes went."""
    T = d["tensors"]
    prod, cons = build_maps(d)
    outputs = set(d.get("outputs", []))
    shape = lambda t: T.get(t, {}).get("shape") or []
    tposed = {}          # rank-3 [N,C,L] tensor -> a tensor holding it as [N,L,C]
    rename = {}          # tensor -> the tensor its readers take instead
    new = []             # (node to insert after, new node)

    def resolve(t):
        while t in rename:
            t = rename[t]
        return t

    def tensor(name, shp, like):
        base, k = name, 1
        while name in T:                 # one tensor per new node, whatever reads it
            k += 1; name = f"{base}{k}"
        T[name] = {"shape": list(shp), "dtype": T[like]["dtype"], "kind": "activation",
                   "layout": "nchw", "const_file": False}
        for k in ("q_scale", "q_zp"):
            if k in T[like]:
                T[name][k] = T[like][k]
        return name

    def plain(n):
        return not n["attrs"].get("also_q") and len(n["outputs"]) >= 1

    def same_values(a, b):
        """a can stand in for b: same type, and for int8 the same quantization."""
        ta, tb = T.get(a, {}), T.get(b, {})
        return ta.get("dtype") == tb.get("dtype") and \
            (ta.get("q_scale"), ta.get("q_zp")) == (tb.get("q_scale"), tb.get("q_zp"))

    for n in list(d["nodes"]):
        op = n["op"]
        if op == "Reshape" and plain(n):
            t = n["inputs"][0]; p = prod.get(t); y = n["outputs"][0]
            if p is None or p["op"] != "Transpose" or p["attrs"].get("perm") != [0, 3, 1, 2]:
                continue
            x = p["inputs"][0]; xs = shape(x); ys = shape(y)
            if len(xs) != 4 or ys != [xs[0], xs[3], xs[1] * xs[2]]:
                continue
            v = tensor(x + "_as_nlc", [xs[0], xs[1] * xs[2], xs[3]], x)
            new.append((n, {"name": v, "op": "Reshape", "inputs": [x], "outputs": [v], "attrs": {}}))
            tposed[y] = v

        elif op == "Concat" and plain(n):
            y = n["outputs"][0]; ys = shape(y); ax = _concat_axis(d, n)
            ins = [resolve(i) for i in n["inputs"]]
            if len(ys) != 3 or ax is None:
                continue
            if ax == 2 and all(i in tposed for i in ins):
                parts = [tposed[i] for i in ins]
                c = tensor(y + "_as_nlc", [ys[0], ys[2], ys[1]], parts[0])
                new.append((n, {"name": c, "op": "Concat", "inputs": parts, "outputs": [c],
                                "attrs": {"axis": 1}}))
                tposed[y] = c
            elif ax == 1 and any(i in tposed for i in ins) and cons.get(y) and \
                    all(r["op"] == "Transpose" and r["attrs"].get("perm") == [0, 2, 1] for r in cons[y]):
                parts = []
                for i in ins:
                    if i in tposed:
                        parts.append(tposed[i]); continue
                    s = shape(i)
                    u = tensor(i + "_as_nlc", [s[0], s[2], s[1]], i)
                    new.append((n, {"name": u, "op": "Transpose", "inputs": [i], "outputs": [u],
                                    "attrs": {"perm": [0, 2, 1]}}))
                    parts.append(u)
                c = tensor(y + "_as_nlc", [ys[0], ys[2], ys[1]], parts[0])
                new.append((n, {"name": c, "op": "Concat", "inputs": parts, "outputs": [c],
                                "attrs": {"axis": 2}}))
                tposed[y] = c

        elif op == "Sigmoid" and n["attrs"].get("identity"):
            x = resolve(n["inputs"][0])
            if x in tposed:
                tposed[n["outputs"][0]] = tposed[x]

        elif op == "Transpose" and n["attrs"].get("perm") == [0, 2, 1]:
            x = resolve(n["inputs"][0]); y = n["outputs"][0]
            if x in tposed and y not in outputs and same_values(tposed[x], y) \
                    and shape(tposed[x]) == shape(y):
                rename[y] = tposed[x]

        elif op == "Split" and plain(n):
            x = resolve(n["inputs"][0]); xs = shape(x)
            c = prod.get(x) or next((m for _, m in new if m["outputs"][0] == x), None)
            if c is None or c["op"] != "Concat" or c["attrs"].get("also_q") or not xs:
                continue
            ax = n["attrs"].get("axis", 0); ax = ax + len(xs) if ax < 0 else ax
            cx = c["attrs"].get("axis", 0); cx = cx + len(xs) if cx < 0 else cx
            parts = c["inputs"]; outs = n["outputs"]
            if ax != cx or len(parts) != len(outs) or any(o in outputs for o in outs):
                continue
            if all(shape(p) and shape(p) == shape(o) and same_values(p, o) for p, o in zip(parts, outs)):
                for p, o in zip(parts, outs):
                    rename[o] = p

    if not rename:
        return 0
    # the new nodes go in just before the node that made them needed
    order = []
    after = {}
    for anchor, m in new:
        after.setdefault(id(anchor), []).append(m)
    for n in d["nodes"]:
        order.extend(after.get(id(n), []))
        order.append(n)
    for n in order:
        n["inputs"] = [resolve(i) for i in n["inputs"]]
    d["nodes"] = order
    # what nobody reads any more goes, but only nodes that only move values
    gone = 0
    movers = ("Transpose", "Reshape", "Concat", "Split", "Sigmoid", "Flatten", "Unsqueeze")
    while True:
        prod, cons = build_maps(d)
        dead = [n for n in d["nodes"] if n["op"] in movers and plain(n)
                and (n["op"] != "Sigmoid" or n["attrs"].get("identity"))
                and not any(cons.get(o) or o in outputs for o in n["outputs"])]
        if not dead:
            break
        gone += sum(1 for n in dead if n["op"] == "Transpose")
        ids = {id(n) for n in dead}
        d["nodes"] = [n for n in d["nodes"] if id(n) not in ids]
    added = sum(1 for _, m in new if m["op"] == "Transpose" and any(m is n for n in d["nodes"]))
    return gone - added


LAZY_FLOATS = True
LAZY_DEPTH = 4          # how many ops deep a value may be recomputed where it is read

def _last_axis(n, rank):
    ax = n["attrs"].get("axis", 0)
    ax = ax + rank if ax < 0 else ax
    return ax == rank - 1


def mark_lazy_floats(d):
    """Float tensors that are cheaper to recompute where they are read than to store.

    In a C2f block a SiLU table turns an int8 conv output into floats, a Split cuts them
    in two, a residual Add sums one half with another SiLU's floats, and a Concat
    quantizes all three for the next conv. Every one of those floats is a table entry
    (or a dequantized byte, or a sum of two such), so it can be made again from the
    int8 it came from, at the moment it is needed, instead of being written to PSRAM
    at 4 bytes a value and read back. Such a tensor is marked "lazy": it gets no
    memory, the ops that made it only to store it are not emitted, and the ops that
    read it (Split, Add, Concat along the channels) compute it in their own loop, by
    the same C expression its producer used -- so the same float, to the bit.

    A tensor is lazy when its producer is a float SiLU table, a plain
    DequantizeLinear, a channel Split or a same-shape Add, and every reader is a
    channel Split, a same-shape Add or a float channel Concat. The int8 tensors a lazy
    one is made from are listed in its readers' "reads", so the arena keeps them
    alive until then. Returns how many tensors went lazy."""
    if not LAZY_FLOATS:
        return 0
    T = d["tensors"]
    prod, cons = build_maps(d)
    outputs = set(d.get("outputs", []))
    shape = lambda t: T.get(t, {}).get("shape") or []
    def isq(t):
        m = T.get(t, {})
        return m.get("dtype") == "int8" and m.get("q_scale") is not None
    def flt(t):
        return T.get(t, {}).get("dtype") == "float32" and t in prod and T[t].get("kind") != "init"
    def act(t):
        return t in prod and T.get(t, {}).get("kind") != "init" and T[t].get("value") is None

    def add_ok(n):
        a, b = n["inputs"][0], n["inputs"][1]; y = n["outputs"][0]
        if not (act(a) and act(b)) or T[b].get("const_file") or not shape(y):
            return False
        if not (shape(a) == shape(b) == shape(y)):
            return False
        return all(flt(i) or isq(i) for i in (a, b))

    def split_ok(n):
        x = n["inputs"][0]
        return flt(x) and len(shape(x)) >= 2 and _last_axis(n, len(shape(x)))

    def concat_ok(n):
        y = n["outputs"][0]; ys = shape(y)
        if T.get(y, {}).get("dtype") != "float32" or not ys or not _last_axis(n, len(ys)):
            return False
        return all(shape(i)[:-1] == ys[:-1] for i in n["inputs"])

    def resize_ok(n):
        x = n["inputs"][0]; y = n["outputs"][0]
        return (n["attrs"].get("mode", "nearest") == "nearest" and flt(x) and len(shape(x)) == 4
                and len(shape(y)) == 4 and shape(x)[3] == shape(y)[3] and shape(x)[0] == shape(y)[0] == 1)

    def reader_ok(n, t):
        if n["op"] == "Resize": return resize_ok(n) and n["outputs"][0] in lazy
        if n["op"] == "Split": return split_ok(n) and n["inputs"][0] == t
        if n["op"] == "Add": return add_ok(n) and t in n["inputs"][:2]
        if n["op"] == "Concat": return concat_ok(n)
        return False

    def producer_ok(t):
        p = prod.get(t)
        if p is None or not flt(t) or t in outputs:
            return False
        if t in {e["dst"] for e in p["attrs"].get("also_q", [])}:
            return False
        if p["op"] == "SiLU":
            return "out_scale" not in p["attrs"] and T.get(p["inputs"][0], {}).get("dtype") == "int8"
        if p["op"] == "DequantizeLinear":
            return not p["attrs"].get("then_sigmoid") and _qparams(d, {"inputs": p["inputs"]}) is not None
        if p["op"] == "Split":
            return split_ok(p)
        if p["op"] == "Add":
            # float inputs only: an int8 one is dequantized inline (s*q), and a reader
            # computing s*q+b itself could see the compiler fuse that into one
            # multiply-add (Xtensa madd.s), rounding differently from the stored sum
            return add_ok(p) and all(flt(i) for i in p["inputs"][:2])
        if p["op"] == "Resize":
            return resize_ok(p)
        return False

    def depth(t):
        if t not in lazy:
            return 0
        p = prod[t]
        ins = p["inputs"][:2] if p["op"] == "Add" else p["inputs"][:1]
        return 1 + max(depth(i) for i in ins)

    # the largest set in which every lazy tensor's readers can make it themselves,
    # and none is more than LAZY_DEPTH ops of recomputation deep
    lazy = {t for t in list(T) if producer_ok(t) and cons.get(t)}
    while True:
        drop = {t for t in lazy if not all(reader_ok(n, t) for n in cons[t])}
        if not drop:
            deep = [t for t in lazy if depth(t) > LAZY_DEPTH]
            if not deep:
                break
            drop = {max(deep, key=depth)}
        lazy -= drop

    def sources(t):
        if t not in lazy:
            return [t]
        p = prod[t]
        ins = p["inputs"][:2] if p["op"] == "Add" else p["inputs"][:1]
        out = []
        for i in ins:
            for s in sources(i):
                if s not in out:
                    out.append(s)
        return out

    for t in lazy:
        p = prod[t]
        T[t]["lazy"] = {"node": {"name": p["name"], "op": p["op"], "inputs": list(p["inputs"]),
                                 "outputs": list(p["outputs"]), "attrs": dict(p["attrs"])}}
    for n in d["nodes"]:
        reads = []
        for i in n["inputs"]:
            if i in lazy:
                for s in sources(i):
                    if s not in reads:
                        reads.append(s)
        if reads:
            n["reads"] = reads
    # an op whose every output went lazy, with nothing folded into it, is not emitted
    keep = []
    for n in d["nodes"]:
        main = [o for o in n["outputs"] if o not in {e["dst"] for e in n["attrs"].get("also_q", [])}]
        if main and all(o in lazy for o in main) and not n["attrs"].get("also_q"):
            continue
        keep.append(n)
    d["nodes"] = keep
    return len(lazy)


def topo_sort(d):
    prod={o:n for n in d["nodes"] for o in n["outputs"]}
    seen=set(); order=[]
    def visit(n, stack):
        if id(n) in seen: return
        for i in list(n["inputs"])+list(n.get("reads",[])):
            p=prod.get(i)
            if p and id(p) not in seen and id(p) not in stack:
                visit(p, stack|{id(n)})
        seen.add(id(n)); order.append(n)
    for n in d["nodes"]:
        visit(n, set())
    d["nodes"]=order

def main(inp,out):
    d=json.load(open(inp,encoding="utf-8"))
    n_logit=widen_logit_range(d)
    print(f"[fuse] class-logit convs given their positive range back = {n_logit}")
    n_silu=fuse_silu(d)
    # fuse_silu already applies/omits the Q-absorption per FUSE_SILU_QUANTIZE.
    print(f"[fuse] SiLU={n_silu}"
          + ("  (SiLU+Q FUSED: int8->int8)" if FUSE_SILU_QUANTIZE
             else "  (SiLU+Q UNFUSED: float SiLU + separate Q)"))
    if INT8_LAYOUT:
        n_int8, n_q = int8_layout_paths(d)
        print(f"[fuse] tensors quantized at the source = {n_int8}, QuantizeLinears gone = {n_q}")
    n_sig = sigmoid_through_layout(d)
    print(f"[fuse] Sigmoids folded into their DequantizeLinears = {n_sig}")
    if INT8_LAYOUT:
        n_fold = fold_quantize_into_producer(d)
        print(f"[fuse] QuantizeLinears folded into their producers = {n_fold}")
    n_tr = cancel_layout_transposes(d)
    print(f"[fuse] transposes that undid each other, gone = {n_tr}")
    n_lazy = mark_lazy_floats(d)
    print(f"[fuse] float tensors made again where read instead of stored = {n_lazy}")
    topo_sort(d)
    json.dump(d, open(out,"w",encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"fuse -> {out}: SiLU fused = {n_silu}, nodes now {len(d['nodes'])}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    a=ap.parse_args(); main(a.inp,a.out)
