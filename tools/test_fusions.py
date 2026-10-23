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
test_fusions.py -- the speed-ups that change what C is made, but must not change a byte.

- fuse.FUSE_CONV_SILU: a SiLU folded into the conv before it, the conv storing each byte
  through the SiLU's table (esp_nn_conv_s8_simd_set_out_lut);
- op_float_ops.CONCAT_BYTE_TABLES: a Concat input made from one int8 byte a value,
  quantized through a 256-entry table made on the first run.

Each of test_ops.py's models that has them is built with both on and with both off, for
both targets, and run on the same inputs: the outputs must be the same bytes, and the
"on" build must actually have folded / tabled something (else the check proves nothing).

  python tools/test_fusions.py
"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from femto_host import compile_model, run, COMPILER
import test_ops

sys.path.insert(0, COMPILER)
import fuse                                   # noqa: E402  (the modules femto_host builds with)
from emitters import op_float_ops             # noqa: E402

CASES = ("stem", "depthwise", "residual", "c2f", "sppf", "upsample", "head", "shared_concat")
RESULTS = []


def check(ok, label):
    RESULTS.append(bool(ok))
    print(f"{'ok  ' if ok else 'FAIL'} {label}")


def build(path, out, target, on):
    fuse.FUSE_CONV_SILU = on
    op_float_ops.CONCAT_BYTE_TABLES = on
    try:
        return compile_model(path, out, target=target)
    finally:
        fuse.FUSE_CONV_SILU = True
        op_float_ops.CONCAT_BYTE_TABLES = True


def main():
    work = tempfile.mkdtemp(prefix="femto_fusions_")
    folded = tabled = 0
    for name in CASES:
        g, outs = test_ops.CASES[name]()
        path = os.path.join(work, name + ".onnx")
        shape = test_ops.quantize(g.model(outs), path)
        xs = [test_ops.RNG.random(shape, dtype=np.float32) for _ in range(3)]
        for target in ("esp32s3", "ansi_int8"):
            on = build(path, os.path.join(work, f"{name}_{target}_on"), target, True)
            off = build(path, os.path.join(work, f"{name}_{target}_off"), target, False)
            src = open(os.path.join(on, "model.c")).read()
            folded += src.count("// its SiLU, folded in")
            tabled += src.count("_state=0;   // byte table")
            same = all(np.concatenate([o.ravel().view(np.uint8) for o in run(on, x)]).tobytes()
                       == np.concatenate([o.ravel().view(np.uint8) for o in run(off, x)]).tobytes()
                       for x in xs)
            check(same, f"{name} ({target}): the same bytes with the fusions on and off")
    check(folded > 0, f"SiLUs folded into convs in these models: {folded}")
    check(tabled > 0, f"Concat inputs quantized through byte tables: {tabled}")
    print(f"{sum(RESULTS)} of {len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
