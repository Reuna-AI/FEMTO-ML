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
compare_ort.py — node-by-node validation against onnxruntime.

Runs the ORIGINAL NCHW quantized ONNX in ORT, exposing intermediate conv (and
optional other) outputs, dequantizes them, and prints per-channel values in the
SAME format as the device dump so you can diff directly.

Layout note: ORT is NCHW [1,C,H,W]; the device is NHWC. This script prints
NCHW value at [c][h][w]; the device prints channel c at raster (h,w). Same logical
element -> compare channel c, position k -> (h=k//W, w=k%W).

INPUT: must be identical to the device's. Best path: have the device dump its
input tensor (the float NCHW it feeds the model) to a file, then pass it here with
--input-npy. Otherwise --input-npy a saved preprocessed image, or --random.

Usage:
  # 1) list conv nodes:
  python compare_ort.py model_quant.onnx --list
  # 2) compare, feeding the device's exact input:
  python compare_ort.py model_quant.onnx --input-npy device_input.npy --dump-n 10 --channels 8
  # 3) compare specific nodes only:
  python compare_ort.py model_quant.onnx --input-npy in.npy --nodes /model.0/conv/Conv_quant

Requires: onnx, onnxruntime, numpy
"""
import sys, argparse, numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper

# QLinearConv/Quantize/Dequantize carry scale/zp we can use to dequantize the printout.
# Everything else (Sigmoid, Mul, Add, Pow, Concat, Reshape, Transpose, Split, MaxPool,
# Resize, ...) is exposed as-is; its output dtype (float or int) decides formatting.
def find_outputs(m, kinds):
    """Return [(idx, node_name, op, output_tensor, scale_name_or_None, zp_name_or_None)]
    for nodes whose op_type is in `kinds` (or ALL nodes if kinds is None/empty).
    idx = position of the node in the ONNX graph (topological order)."""
    res=[]
    allk = (not kinds)
    for gi,n in enumerate(m.graph.node):
        if not allk and n.op_type not in kinds: continue
        # every output of every matching node (some ops have >1 output, e.g. Split)
        for oi,y in enumerate(n.output):
            if y=="" : continue
            if n.op_type=="QLinearConv":
                ys=n.input[6] if len(n.input)>6 else None
                yz=n.input[7] if len(n.input)>7 else None
            elif n.op_type in ("QuantizeLinear","DequantizeLinear"):
                ys=n.input[1] if len(n.input)>1 else None
                yz=n.input[2] if len(n.input)>2 else None
            else:
                ys=None; yz=None
            label = n.name if oi==0 else f"{n.name}#out{oi}"
            res.append((gi, label, n.op_type, y, ys, yz))
    return res

def const_val(m, name):
    for it in m.graph.initializer:
        if it.name==name: return numpy_helper.to_array(it)
    return None

def expose_outputs(m, tensor_names):
    """Add given internal tensors as graph outputs so ORT returns them."""
    existing={o.name for o in m.graph.output}
    for t in tensor_names:
        if t in existing: continue
        vi=helper.ValueInfoProto(); vi.name=t
        m.graph.output.append(vi)
    return m


def preprocess_image(path, H=224, W=224, rgb565=True):
    """Replicate the device path: load -> nearest resize to WxH -> (optional RGB565
    5/6/5 quantization) -> normalize per channel to [0,1] -> NCHW float32.
    Device uses r/31, g/63, b/31 after RGB565; without rgb565 uses /255 on 8-bit."""
    from PIL import Image
    img=Image.open(path).convert("RGB").resize((W,H), Image.NEAREST)  # device uses nearest
    arr=np.asarray(img).astype(np.float32)   # HWC, 0..255
    r=arr[:,:,0]; g=arr[:,:,1]; b=arr[:,:,2]
    if rgb565:
        # emulate RGB565: quantize to 5/6/5 bits then normalize by max of each field
        r5=np.floor(r/255.0*31.0); g6=np.floor(g/255.0*63.0); b5=np.floor(b/255.0*31.0)
        rn=r5/31.0; gn=g6/63.0; bn=b5/31.0
    else:
        rn=r/255.0; gn=g/255.0; bn=b/255.0
    chw=np.stack([rn,gn,bn], axis=0)          # CHW
    return chw[None,...].astype(np.float32)    # NCHW [1,3,H,W]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--list", action="store_true", help="list nodes and exit")
    ap.add_argument("--kinds", nargs="*", default=None,
                    help="op types to expose (default: ALL). e.g. --kinds Sigmoid Mul Add Concat Reshape MaxPool Resize")
    ap.add_argument("--all", action="store_true", help="expose every node's output")
    ap.add_argument("--input-npy", help="input tensor .npy (NCHW, matches device)")
    ap.add_argument("--random", action="store_true", help="use fixed random input")
    ap.add_argument("--image", help="image file; preprocessed to match the device (RGB565->resize->[0,1] NCHW)")
    ap.add_argument("--no-rgb565", action="store_true", help="skip RGB565 quantization (use full 8-bit); only if device does too")
    ap.add_argument("--nodes", nargs="*", help="specific conv node names (default: all)")
    ap.add_argument("--dump-n", type=int, default=10)
    ap.add_argument("--channels", type=int, default=8)
    a=ap.parse_args()

    m=onnx.load(a.onnx)
    kinds=None if (a.all or not a.kinds) else set(a.kinds)
    nodes=find_outputs(m, kinds)
    if a.list:
        print(f"{len(nodes)} nodes ({', '.join(a.kinds)}):")
        for gi,nm,op,y,ys,yz in nodes: print(f"  [{gi}] [{op:16s}] {nm}  -> {y}")
        return

    sel=nodes
    if a.nodes:
        want_names=set(a.nodes)
        sel=[c for c in nodes if c[1] in want_names]
        if not sel:
            print("no matching nodes; use --list"); return

    want=[t[3] for t in sel]
    m=expose_outputs(m, want)
    tmp="_ort_exposed.onnx"; onnx.save(m, tmp)

    sess=ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    iname=sess.get_inputs()[0].name
    ishape=[d if isinstance(d,int) else 1 for d in sess.get_inputs()[0].shape]

    if a.image:
        _,C,H,W=ishape if len(ishape)==4 else (1,3,224,224)
        x=preprocess_image(a.image, H=H, W=W, rgb565=(not a.no_rgb565))
        print(f"(preprocessed {a.image}: shape {x.shape}, rgb565={not a.no_rgb565})")
    elif a.input_npy:
        x=np.load(a.input_npy).astype(np.float32)
        if list(x.shape)!=ishape:
            print(f"WARNING: input shape {x.shape} != model {ishape}; reshaping")
            x=x.reshape(ishape)
    else:
        np.random.seed(0); x=np.random.rand(*ishape).astype(np.float32)
        print("(using fixed random input — device must use the SAME array to compare)")

    outs=sess.run(want, {iname:x})
    name2arr=dict(zip(want, outs))

    for gi,nm,op,y,ys,yz in sel:
        arr=name2arr[y]
        is_float = np.issubdtype(arr.dtype, np.floating)
        sc=1.0; zp=0
        if not is_float:
            scv=const_val(m, ys); zpv=const_val(m, yz)
            sc=float(scv.flatten()[0]) if scv is not None else 1.0
            zp=int(zpv.flatten()[0]) if zpv is not None else 0
        # Normalize to (C, spatial...) view. NCHW conv tensors -> [1,C,H,W].
        # For other ranks, treat axis 1 as "channel" and flatten the rest as positions.
        a4=arr
        if a4.ndim>=2 and a4.shape[0]==1:
            a4=a4[0]                      # drop batch -> (C, ...)
        if a4.ndim==1:
            a4=a4.reshape(1,-1)           # treat as 1 channel
        C=a4.shape[0]
        flat=a4.reshape(C,-1)             # (C, positions)
        P=flat.shape[1]
        dt = "float" if is_float else f"int sc={sc:.6g} zp={zp}"
        print(f"== [{gi}] {nm} [{op} ORT shape{tuple(arr.shape)} {dt}]")
        cn=min(C, a.channels)
        for c in range(cn):
            vals=[]
            for k in range(min(a.dump_n, P)):
                if is_float: vals.append(f"{flat[c,k]:.4f}")
                else:
                    v=int(flat[c,k]); vals.append(f"{v}({sc*(v-zp):.4f})")
            print(f"  ch{c}: "+" ".join(vals))

if __name__=="__main__":
    main()
