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
"""
test_ops.py -- small models built here, quantized by onnxruntime the way the YOLO
models were, compiled by FEMTO-ML and checked against onnxruntime.

Each case is a few ops in shapes the YOLO models do not use (odd channel counts,
strides, depthwise, the C2f block, SPPF pooling, upsampling, a detect head that
transposes and splits), so an emitter or a pass that only works for YOLO26n's
shapes fails here. Both targets must give the same bytes, and both must be
within a few quantization steps of onnxruntime (the rounding at exact halves and
the SiLU tables differ in the last bit; nothing more is allowed).

  python tools/test_ops.py            all cases
  python tools/test_ops.py c2f head   some
"""
import os, sys, tempfile
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from femto_host import compile_model, run

RNG = np.random.default_rng(1234)


class G:
    """A float ONNX graph, written NCHW as an exporter would."""
    def __init__(s, shape):
        s.nodes, s.inits, s.k, s.shape = [], [], 0, shape
    def name(s, b):
        s.k += 1; return f"{b}{s.k}"
    def const(s, arr):
        n = s.name("c"); s.inits.append(numpy_helper.from_array(np.asarray(arr), n)); return n
    def node(s, op, ins, n_out=1, **attrs):
        outs = [s.name(op.lower()) for _ in range(n_out)]
        s.nodes.append(helper.make_node(op, ins, outs, name=s.name("n_" + op), **attrs))
        return outs if n_out > 1 else outs[0]
    def conv(s, x, cin, cout, k=3, stride=1, group=1):
        w = (RNG.standard_normal((cout, cin // group, k, k)) * (1.5 / np.sqrt(cin // group * k * k))).astype(np.float32)
        b = (RNG.standard_normal(cout) * 0.1).astype(np.float32)
        return s.node("Conv", [x, s.const(w), s.const(b)], kernel_shape=[k, k], strides=[stride, stride],
                      pads=[k // 2] * 4, group=group)
    def silu(s, x):
        return s.node("Mul", [x, s.node("Sigmoid", [x])])
    def model(s, outs):
        g = helper.make_graph(s.nodes, "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, s.shape)],
                              [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outs], s.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        return onnx.shape_inference.infer_shapes(m)


def case_stem():            # 3 -> 16, stride 2, SiLU
    g = G([1, 3, 32, 32]); return g, [g.silu(g.conv("x", 3, 16, stride=2))]

def case_odd_channels():    # channel counts that are not multiples of 16
    g = G([1, 3, 24, 20]); y = g.silu(g.conv("x", 3, 24)); y = g.silu(g.conv(y, 24, 40, k=1))
    return g, [g.silu(g.conv(y, 40, 20, stride=2))]

def case_depthwise():
    g = G([1, 3, 16, 16]); y = g.silu(g.conv("x", 3, 32))
    return g, [g.silu(g.conv(y, 32, 32, group=32))]

def case_residual():        # two SiLU branches summed
    g = G([1, 3, 16, 16]); a = g.silu(g.conv("x", 3, 16)); b = g.silu(g.conv(a, 16, 16))
    return g, [g.silu(g.conv(g.node("Add", [a, b]), 16, 24, k=1))]

def case_c2f():             # conv, split, bottleneck with residual, concat, conv
    g = G([1, 3, 16, 16]); y = g.silu(g.conv("x", 3, 32, k=1))
    p0, p1 = g.node("Split", [y, g.const(np.array([16, 16], np.int64))], n_out=2, axis=1)
    m = g.silu(g.conv(g.silu(g.conv(p1, 16, 8)), 8, 16))
    c = g.node("Concat", [p0, p1, g.node("Add", [p1, m])], axis=1)
    return g, [g.silu(g.conv(c, 48, 32, k=1))]

def case_sppf():            # max pools 5x5, concatenated
    g = G([1, 3, 16, 16]); y = g.silu(g.conv("x", 3, 16, k=1))
    p1 = g.node("MaxPool", [y], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2])
    p2 = g.node("MaxPool", [p1], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2])
    return g, [g.silu(g.conv(g.node("Concat", [y, p1, p2], axis=1), 48, 16, k=1))]

def case_upsample():        # nearest 2x, concat with a skip, conv
    g = G([1, 3, 16, 16]); a = g.silu(g.conv("x", 3, 16)); b = g.silu(g.conv(a, 16, 24, stride=2))
    u = g.node("Resize", [b, "", g.const(np.array([1, 1, 2, 2], np.float32))], mode="nearest",
               coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    return g, [g.silu(g.conv(g.node("Concat", [u, a], axis=1), 40, 16))]

def case_head():            # two scales to [N,C,HW], concatenated, transposed back, split
    g = G([1, 3, 16, 16]); a = g.silu(g.conv("x", 3, 16)); b = g.silu(g.conv(a, 16, 16, stride=2))
    parts = []
    for t, hw in ((a, 256), (b, 64)):
        y = g.conv(t, 16, 12, k=1)
        parts.append(g.node("Reshape", [y, g.const(np.array([1, 12, hw], np.int64))]))
    c = g.node("Concat", parts, axis=2)
    s = g.node("Sigmoid", [c])
    t = g.node("Transpose", [s], perm=[0, 2, 1])
    box, cls = g.node("Split", [t, g.const(np.array([4, 8], np.int64))], n_out=2, axis=2)
    return g, [box, cls]

def case_residual_linear():  # a conv with no SiLU (a plain DequantizeLinear) in a residual Add
    g = G([1, 3, 16, 16]); a = g.silu(g.conv("x", 3, 16)); b = g.conv(a, 16, 16)
    return g, [g.silu(g.conv(g.node("Add", [a, b]), 16, 8, k=1))]

def case_shared_concat():    # one tensor in two Concats: neither may overwrite the other's parts
    g = G([1, 3, 8, 8]); a = g.silu(g.conv("x", 3, 8))
    b = g.silu(g.conv(a, 8, 8)); c = g.silu(g.conv(a, 8, 8))
    ra, rb, rc = (g.node("Reshape", [t, g.const(np.array([1, 8, 64], np.int64))]) for t in (a, b, c))
    y1 = g.node("Concat", [ra, rb], axis=1); y2 = g.node("Concat", [ra, rc], axis=1)
    k = g.const(np.zeros([1, 16, 64], np.float32))       # read after both are made
    return g, [g.node("Add", [y1, k]), g.node("Add", [y2, k])]

def case_upsample3():       # nearest 3x, asymmetric/floor: supported
    g = G([1, 3, 8, 8]); a = g.silu(g.conv("x", 3, 8))
    u = g.node("Resize", [a, "", g.const(np.array([1, 1, 3, 3], np.float32))], mode="nearest",
               coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    return g, [g.silu(g.conv(u, 8, 8))]

def case_upsample3_round():  # nearest 3x, asymmetric/round_prefer_floor: must be refused
    g = G([1, 3, 8, 8]); a = g.silu(g.conv("x", 3, 8))
    u = g.node("Resize", [a, "", g.const(np.array([1, 1, 3, 3], np.float32))], mode="nearest",
               coordinate_transformation_mode="asymmetric", nearest_mode="round_prefer_floor")
    return g, [g.silu(g.conv(u, 8, 8))]
REFUSED = {"upsample3_round"}      # cases the compiler must stop on, not compile

def case_head_boxes():      # YOLO26n's head: boxes decoded in NCHW, scores sigmoided, joined, back
    g = G([1, 3, 16, 16]); a = g.silu(g.conv("x", 3, 16)); b = g.silu(g.conv(a, 16, 16, stride=2))
    box, cls = [], []
    for t, hw in ((a, 256), (b, 64)):
        box.append(g.node("Reshape", [g.conv(t, 16, 4, k=1), g.const(np.array([1, 4, hw], np.int64))]))
        cls.append(g.node("Reshape", [g.conv(t, 16, 8, k=1), g.const(np.array([1, 8, hw], np.int64))]))
    bx = g.node("Mul", [g.node("Concat", box, axis=2), g.const(np.array(2.0, np.float32))])
    sc = g.node("Sigmoid", [g.node("Concat", cls, axis=2)])
    t = g.node("Transpose", [g.node("Concat", [bx, sc], axis=1)], perm=[0, 2, 1])
    o1, o2 = g.node("Split", [t, g.const(np.array([4, 8], np.int64))], n_out=2, axis=2)
    return g, [o1, o2]

def case_one_channel():     # a conv with one output channel (YuNet's face and object scores):
    g = G([1, 3, 16, 16]); a = g.silu(g.conv("x", 3, 16))   # its bias is one value, and stays an array
    return g, [g.node("Sigmoid", [g.conv(a, 16, 1, k=1)])]

def case_relu():            # ReLU between convs, then a 2x2 max pool (YuNet's backbone)
    g = G([1, 3, 16, 16]); a = g.node("Relu", [g.conv("x", 3, 16)])
    p = g.node("MaxPool", [a], kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    return g, [g.node("Relu", [g.conv(p, 16, 8)])]

CASES = {n[5:]: f for n, f in globals().items() if n.startswith("case_")}


def quantize(m, path):
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    src = path + ".f32.onnx"; onnx.save(m, src)
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    class R(CalibrationDataReader):
        def __init__(s): s.it = iter([{"x": RNG.random(shape, dtype=np.float32)} for _ in range(8)])
        def get_next(s): return next(s.it, None)
    quantize_static(src, path, R(), quant_format=QuantFormat.QOperator, op_types_to_quantize=["Conv"],
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    q = onnx.load(path); onnx.save(onnx.shape_inference.infer_shapes(q), path)
    return shape


def check(name, work):
    import onnxruntime as ort
    g, outs = CASES[name]()
    path = os.path.join(work, name + ".onnx")
    shape = quantize(g.model(outs), path)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    builds = {t: compile_model(path, os.path.join(work, f"{name}_{t}"), target=t) for t in ("esp32s3", "ansi_int8")}
    worst = 0.0
    for trial in range(3):
        x = RNG.random(shape, dtype=np.float32)
        ref = sess.run(None, {"x": x})
        got = {t: run(b, x) for t, b in builds.items()}
        for k, r in enumerate(ref):
            e, a = got["esp32s3"][k], got["ansi_int8"][k]
            if e.tobytes() != a.tobytes():
                return f"targets disagree on output {k}"
            r = np.asarray(r).reshape(e.shape)
            span = float(np.abs(r).max()) or 1.0
            worst = max(worst, float(np.abs(e - r).max()) / span)
    # a few int8 steps of the output's own range
    return None if worst <= 4 / 127 else f"off onnxruntime by {worst:.3f} of the output range"


def main():
    names = sys.argv[1:] or list(CASES)
    work = tempfile.mkdtemp(prefix="femto_ops_")
    failed = 0
    for n in names:
        try:
            why = check(n, work)
            if n in REFUSED:
                why = "compiled, but should have been refused"
        except (Exception, SystemExit) as exc:     # a compiler error is a failure too...
            why = None if n in REFUSED else f"{type(exc).__name__}: {exc}"   # ...unless wanted
        print(f"{'ok  ' if why is None else 'FAIL'} {n}" + ("" if why is None else f"  -- {why}"))
        failed += why is not None
    print(f"{len(names) - failed} of {len(names)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
