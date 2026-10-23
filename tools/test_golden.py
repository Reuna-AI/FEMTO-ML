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
test_golden.py -- the shipped YOLO26n, compiled and run on this PC, must give exactly
the bytes it gave when tests/golden_yolo26n.json was written.

Three inputs (the two 256x256 photos in examples/esp32_yolo26n/images.h and seeded noise),
two targets (esp32s3, and the portable ansi_int8, which must agree with it byte for
byte). Any change to the compiler that moves a single output bit fails here. A
change meant to move them (a better rounding, say) is checked against onnxruntime
by other means, then recorded with --update and explained in CHANGELOG.md.

  python tools/test_golden.py            check
  python tools/test_golden.py --update   record the current outputs as golden
"""
import argparse, hashlib, json, os, re, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from femto_host import ROOT, compile_model, run

MODEL = os.path.join(ROOT, "examples", "esp32_yolo26n", "yolo26n_q.onnx")
GOLDEN = os.path.join(ROOT, "tests", "golden_yolo26n.json")


def photo(name):
    """A 256x256 0xRRGGBB picture from images.h (the example sketch's own input)."""
    text = open(os.path.join(ROOT, "examples", "esp32_yolo26n", "images.h")).read()
    body = re.search(r"uint32_t " + name + r"\[\][^{]*\{([^}]*)\}", text).group(1)
    px = np.array([int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)], dtype=np.uint32)
    return np.stack([(px >> 16) & 255, (px >> 8) & 255, px & 255], -1).reshape(256, 256, 3).astype(np.uint8)


def inputs():
    noise = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    for name, img in (("image_1", photo("image_1")), ("image", photo("image")), ("noise", noise)):
        yield name, (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="femto_golden_")
    builds = {t: compile_model(MODEL, os.path.join(work, t), target=t) for t in ("esp32s3", "ansi_int8")}
    got = {}
    for name, x in inputs():
        outs = {t: run(b, x)[0] for t, b in builds.items()}
        if outs["esp32s3"].tobytes() != outs["ansi_int8"].tobytes():
            sys.exit(f"FAIL {name}: the esp32s3 and ansi_int8 builds disagree")
        y = outs["esp32s3"]
        got[name] = hashlib.md5(y.tobytes()).hexdigest()
        top = y.reshape(-1, 6)[:3]
        print(f"{name:8s} {got[name]}  top: " + ", ".join(f"class {int(r[5])} {r[4]:.3f}" for r in top))
    if a.update:
        json.dump(got, open(GOLDEN, "w"), indent=1)
        print(f"recorded {GOLDEN}")
        return
    want = json.load(open(GOLDEN))
    bad = [k for k in want if want[k] != got.get(k)]
    if bad:
        sys.exit(f"FAIL: outputs changed for {', '.join(bad)} (see the docstring before using --update)")
    print("golden: all outputs identical, both targets")


if __name__ == "__main__":
    main()
