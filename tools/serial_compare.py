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
#
# ============================================================================

"""
serial_compare.py — read ESP32 node dumps over serial, compare against onnxruntime.

Eliminates the manual dump-paste-compare loop. The device prints per-node dumps like:
    == /model.10/cv1/conv/... [int8 H8 W8 C256 NHWC] sc=0.0178 zp=-112
      ch0: -111(0.0179) -121(-0.1607) ...
      ch1: ...
This script parses those, runs ORT on the same image, aligns by node name, and reports
the FIRST node whose values grossly diverge (beyond a tolerance), which pinpoints bugs.

USAGE:
  python serial_compare.py --port COM7 --onnx yolo26.onnx --image img.jpg \
      --size 256 [--baud 115200] [--tol 0.15] [--per-channel]

  # or parse a saved serial log instead of live port:
  python serial_compare.py --logfile dump.txt --onnx yolo26.onnx --image img.jpg --size 256

Requires: pyserial, onnxruntime, numpy, pillow
  pip install pyserial onnxruntime numpy pillow
"""
import argparse, re, sys, time
import numpy as np

# ----------------------------- serial / log reading -----------------------------
def read_serial(port, baud, idle_timeout=4.0, hard_timeout=300.0, save=None):
    """Read lines from serial. Stops when: a 'done.' end marker is seen AND stream goes
    idle, OR idle_timeout seconds pass with no new data (after data seen), OR hard_timeout,
    OR Ctrl-C. Saves incrementally to `save` if given so nothing is ever lost."""
    import serial
    ser = serial.Serial(port, baud, timeout=0.5)
    lines, last, start = [], time.time(), time.time()
    got_any = False; saw_done = False
    fh = open(save, "w", encoding="utf-8") if save else None
    print(f"[serial] listening on {port}@{baud} … (reset the board to start dumping)")
    print(f"[serial] will stop on idle >{idle_timeout}s after data, on 'done.', or Ctrl-C")
    try:
        while True:
            raw = ser.readline()
            now = time.time()
            if raw:
                try: s = raw.decode("utf-8", "replace").rstrip("\r\n")
                except: s = str(raw)
                lines.append(s); last = now; got_any = True
                if fh: fh.write(s+"\n"); fh.flush()   # save incrementally
                if s.startswith("==") or s.strip().startswith("detections:"):
                    print("  " + s)
                if s.strip() == "done." or s.strip().startswith("detections:"):
                    saw_done = True
            else:
                if saw_done and got_any and (now - last) > 1.0: break
                if got_any and (now - last) > idle_timeout: break
                if (now - start) > hard_timeout: break
    except KeyboardInterrupt:
        print("\n[serial] Ctrl-C — stopping capture, proceeding to compare")
    finally:
        ser.close()
        if fh: fh.close()
    print(f"[serial] captured {len(lines)} lines" + (f" (saved to {save})" if save else ""))
    return lines

def read_logfile(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\r\n") for ln in f]

# ----------------------------- dump parsing -----------------------------
HDR_4D = re.compile(r"==\s+(\S+)\s+\[int8\s+H(\d+)\s+W(\d+)\s+C(\d+)\s+(NCHW|NHWC)\].*?sc=([-\d.eE]+)\s+zp=(-?\d+)")
HDR_F  = re.compile(r"==\s+(\S+)\s+\[float\s+H(\d+)\s+W(\d+)\s+C(\d+)\s+(NCHW|NHWC)\]")
HDR_FLAT = re.compile(r"==\s+(\S+)\s+\[(?:int8|float)\s+flat\s+C(\d+)\s+per(\d+)\]")
CHLINE = re.compile(r"^\s*ch(\d+):\s+(.*)$")
# each value like "-111(0.0179)" or "0.4531" (float dumps)
VAL = re.compile(r"(-?\d+)\(([-\d.eE]+)\)|(-?[\d.]+(?:[eE][-+]?\d+)?)")

def parse_dumps(lines):
    """Return dict: node_name -> {'layout':.., 'chans': {ch_index: [dequant floats]}}"""
    out = {}
    cur = None
    for ln in lines:
        m4 = HDR_4D.search(ln) or HDR_F.search(ln)
        mflat = HDR_FLAT.search(ln)
        if m4:
            name = m4.group(1)
            layout = m4.group(5) if m4.lastindex>=5 else "NHWC"
            cur = {"layout": layout, "chans": {}}
            out[name] = cur
            continue
        if mflat:
            name = mflat.group(1)
            cur = {"layout":"FLAT", "chans": {}}
            out[name] = cur
            continue
        if cur is not None:
            cm = CHLINE.match(ln)
            if cm:
                ci = int(cm.group(1)); rest = cm.group(2)
                deq = []; raw = []
                for mm in VAL.finditer(rest):
                    if mm.group(2) is not None:      # "int(dequant)" form
                        raw.append(float(mm.group(1)))
                        deq.append(float(mm.group(2)))
                    elif mm.group(3) is not None:    # bare float (float dumps)
                        deq.append(float(mm.group(3)))
                        raw.append(float(mm.group(3)))
                cur["chans"][ci] = deq
                cur.setdefault("chans_raw", {})[ci] = raw
    return out

# ----------------------------- ORT reference -----------------------------
def run_ort(onnx_path, image_path, size, want_nodes):
    import onnxruntime as ort
    import onnx
    from onnx import helper
    from PIL import Image

    model = onnx.load(onnx_path)
    # run shape inference so intermediate tensors get proper type/shape (avoids type 0)
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[ort] shape inference warning: {e}")

    existing = {vi.name for vi in model.graph.output}
    name_to_out = {}
    for n in model.graph.node:
        if n.name:
            name_to_out[n.name] = n.output[0] if n.output else None

    # typed ValueInfo lookup from value_info + inputs + outputs (all carry real types)
    vi_lookup = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        vi_lookup[vi.name] = vi

    add = []
    for wn in want_nodes:
        for nm, outt in name_to_out.items():
            if wn in nm and outt:
                add.append(outt)
    add = [t for t in set(add) if t not in existing]

    exposed = []
    skipped = []
    for t in add:
        if t in vi_lookup:
            model.graph.output.append(vi_lookup[t])   # copy TYPED value info
            exposed.append(t)
        else:
            skipped.append(t)
    if skipped:
        print(f"[ort] {len(skipped)} tensors have no inferred type; exposing untyped via output_names fetch")

    tmp = onnx_path + ".exposed.onnx"
    onnx.save(model, tmp)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(tmp, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0     # HWC, /255
    arr = np.transpose(arr, (2,0,1))[None]               # NCHW
    outs = sess.run(None, {inp.name: arr})
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outs)), name_to_out

# ----------------------------- compare -----------------------------
def ort_channel_rows(arr, n_ch=8, n_val=10):
    """Given an ORT tensor (1,C,H,W) or (1,H,W,C) or (1,C,N), return {ch:[vals]} in
    the same 'first n_ch channels, first n_val spatial' form the device prints."""
    a = np.asarray(arr)
    a = a[0] if a.shape[0]==1 else a
    if a.ndim == 3:
        # assume CHW (ORT is NCHW); flatten spatial
        C = a.shape[0]
        rows = {}
        for c in range(min(n_ch, C)):
            rows[c] = a[c].flatten()[:n_val].tolist()
        return rows
    if a.ndim == 2:
        C = a.shape[0]
        return {c: a[c].flatten()[:n_val].tolist() for c in range(min(n_ch,C))}
    if a.ndim == 1:
        return {0: a[:n_val].tolist()}
    return {}

def compare(dev, ort_map, name_to_out, tol):
    print("\n===== COMPARISON (device vs ORT) =====")
    print(f"{'node':50s} {'status':10s} pos-diff set-diff  note")
    first_bad = None
    for name, d in dev.items():
        # find ORT tensor for this node
        outt = None
        for nm, ot in name_to_out.items():
            if name in nm or nm in name:
                outt = ot; break
        if outt is None or outt not in ort_map:
            print(f"{name[:50]:50s} {'no-ort':10s}   —     (not exposed in ORT)")
            continue
        ort_arr = ort_map[outt]
        is_int = np.issubdtype(np.asarray(ort_arr).dtype, np.integer)
        orows = ort_channel_rows(ort_arr)
        drows = d.get("chans_raw", d["chans"]) if is_int else d["chans"]
        pos_diff = 0.0; set_diff = 0.0; ncmp = 0
        for c in sorted(drows):
            if c not in orows: continue
            dv = np.array(drows[c]); ov = np.array(orows[c])
            k = min(len(dv), len(ov))
            if k==0: continue
            dvi, ovi = dv[:k], ov[:k]
            pos_diff = max(pos_diff, float(np.abs(dvi-ovi).max()))
            # position-independent: compare sorted value sets
            sd = float(np.abs(np.sort(dvi)-np.sort(ovi)).max())
            set_diff = max(set_diff, sd)
            ncmp += k
        # value range for relative comparison
        allv=[]
        for c in sorted(drows):
            if c in orows:
                allv += list(orows[c])
        vr = (max(allv)-min(allv)) if allv else 1.0
        vr = vr if vr>1e-6 else 1.0
        rel_set = set_diff/vr
        rel_pos = pos_diff/vr
        # thresholds: relative >8% of range = real divergence
        REL=0.08
        if not ncmp:
            note="no overlap"; status="no-cmp"
        elif rel_pos <= REL:
            note=""; status="OK"
        elif rel_set <= REL:
            note="SCRAMBLE (same values, wrong pos)"; status="SCRAMBLE"
        else:
            note=f"rel={rel_set*100:.0f}% of range"; status="DIVERGE"
        if status in ("SCRAMBLE","DIVERGE") and first_bad is None:
            first_bad = f"{name}  [{status}: {note}]"
        print(f"{name[:50]:50s} {status:10s} pos={pos_diff:7.3f} set={set_diff:7.3f}  {note}")
    print("======================================")
    if first_bad:
        print(f"\n>>> FIRST DIVERGENCE: {first_bad}")
        print("    (everything above matches within tol; the bug is at/after this node)")
    else:
        print("\n>>> All compared nodes match within tolerance.")
    return first_bad

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--logfile", help="parse a saved serial log instead of live serial")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tol", type=float, default=0.15, help="max |Δ| (dequant units) before flagging divergence")
    ap.add_argument("--save", help="save captured serial lines to this file")
    args = ap.parse_args()

    if args.logfile:
        lines = read_logfile(args.logfile)
    elif args.port:
        lines = read_serial(args.port, args.baud, save=args.save)
    else:
        print("ERROR: provide --port or --logfile"); sys.exit(1)

    dev = parse_dumps(lines)
    if not dev:
        print("No device dumps parsed. Make sure the sketch was built with --dump and the "
              "board printed '== nodename [...]' blocks."); sys.exit(1)
    print(f"[parse] {len(dev)} device node dumps: {', '.join(list(dev)[:6])}"
          + (" …" if len(dev)>6 else ""))

    want = list(dev.keys())
    ort_map, name_to_out = run_ort(args.onnx, args.image, args.size, want)
    compare(dev, ort_map, name_to_out, args.tol)

if __name__ == "__main__":
    main()
