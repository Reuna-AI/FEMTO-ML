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
compare_fuse_modes.py — capture ESP32 node dumps for the FUSED (True) and UNFUSED (False)
builds separately, then compare them node-by-node.

Purpose: prove the SiLU+Quantize fusion is numerically identical (or find the first node
where True and False diverge, which localizes any real fusion bug).

WORKFLOW:
  1. Flash the board with FUSE_SILU_QUANTIZE = True, then:
       python compare_fuse_modes.py capture --mode true  --port COM8 --out dump_true.txt
  2. Flash with FUSE_SILU_QUANTIZE = False, then:
       python compare_fuse_modes.py capture --mode false --port COM8 --out dump_false.txt
  3. Compare the two saved captures:
       python compare_fuse_modes.py compare --true dump_true.txt --false dump_false.txt

  (You can also do capture + compare in one go if you have both files already.)

Dump format expected (what the sketch prints):
  == /node/name [int8 H.. W.. C.. NHWC] sc=.. zp=..
    ch0: v(deq) v(deq) ...
    ch1: ...
  or float:
  == /node/name [float H.. W.. C.. NHWC]
    ch0: v v ...

Requires: pyserial  (only for 'capture'; 'compare' is pure-python)
"""
import argparse, re, sys, time

# ------------------------- serial capture -------------------------
def capture(port, baud, out_path, mode, idle_timeout=5.0, hard_timeout=600.0):
    import serial, os
    if os.path.isdir(out_path):
        # user passed a directory -> pick a sensible filename inside it
        out_path = os.path.join(out_path, f"dump_{mode}.txt")
        print(f"[capture] --out was a directory; writing to {out_path}")
    d=os.path.dirname(out_path)
    if d and not os.path.isdir(d):
        print(f"ERROR: directory does not exist: {d}"); sys.exit(1)
    ser = serial.Serial(port, baud, timeout=0.5)
    fh = open(out_path, "w", encoding="utf-8")
    fh.write(f"# MODE={mode}\n")          # tag the file with which build produced it
    lines=0; last=time.time(); start=time.time(); got=False; done=False
    print(f"[capture:{mode}] listening {port}@{baud} -> {out_path}")
    print(f"[capture:{mode}] reset the board now. Stops on 'detections:'/'done.' + idle, or Ctrl-C.")
    try:
        while True:
            raw=ser.readline(); now=time.time()
            if raw:
                s=raw.decode("utf-8","replace").rstrip("\r\n")
                fh.write(s+"\n"); fh.flush(); lines+=1; last=now; got=True
                if s.startswith("==") or s.startswith("[") or s.strip().startswith("detections:"):
                    print("  "+s)
                if s.strip().startswith("detections:") or s.strip()=="done.":
                    done=True
            else:
                if done and got and (now-last)>1.5: break
                if got and (now-last)>idle_timeout: break
                if (now-start)>hard_timeout: break
    except KeyboardInterrupt:
        print(f"\n[capture:{mode}] Ctrl-C, stopping")
    finally:
        ser.close(); fh.close()
    print(f"[capture:{mode}] saved {lines} lines to {out_path}")

# ------------------------- dump parsing -------------------------
HDR = re.compile(r"==\s+(\S+)\s+\[(int8|float)")
CH  = re.compile(r"^\s*ch(\d+):\s*(.*)$")
VAL = re.compile(r"(-?\d+)\((-?[\d.eE]+)\)|(-?[\d.]+(?:[eE][-+]?\d+)?)")

def parse(path):
    """Return (mode, {node: {ch: [raw_int_or_float,...]}})."""
    mode=None
    nodes={}
    cur=None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("# MODE="):
                mode=ln.strip().split("=",1)[1]; continue
            h=HDR.search(ln)
            if h:
                cur=h.group(1); nodes.setdefault(cur, {}); continue
            cm=CH.match(ln)
            if cm and cur is not None:
                ci=int(cm.group(1)); raw=[]
                for mm in VAL.finditer(cm.group(2)):
                    if mm.group(1) is not None:      # int(deq)
                        raw.append(int(mm.group(1)))
                    elif mm.group(3) is not None:    # bare float
                        raw.append(float(mm.group(3)))
                nodes[cur][ci]=raw
    return mode, nodes

# ------------------------- compare -------------------------
def compare(true_path, false_path, tol=0):
    mt, nt = parse(true_path)
    mf, nf = parse(false_path)
    print(f"\n=== COMPARE  TRUE({mt or '?'}: {true_path})  vs  FALSE({mf or '?'}: {false_path}) ===")
    common=[n for n in nt if n in nf]
    only_t=[n for n in nt if n not in nf]
    only_f=[n for n in nf if n not in nt]
    print(f"nodes: true={len(nt)} false={len(nf)} common={len(common)}")
    if only_t: print(f"  only in TRUE  ({len(only_t)}): {only_t[:4]}{' ...' if len(only_t)>4 else ''}")
    if only_f: print(f"  only in FALSE ({len(only_f)}): {only_f[:4]}{' ...' if len(only_f)>4 else ''}")
    print()
    print(f"{'node':52s} {'status':10s} maxΔ")
    first_diff=None; ndiff=0
    for n in common:
        maxd=0; anymiss=False
        for ci in sorted(nt[n]):
            if ci not in nf[n]: continue
            a=nt[n][ci]; b=nf[n][ci]; k=min(len(a),len(b))
            for i in range(k):
                d=abs(a[i]-b[i])
                if d>maxd: maxd=d
        status = "OK" if maxd<=tol else "DIFFERS"
        if maxd>tol:
            ndiff+=1
            if first_diff is None: first_diff=n
        # only print differing nodes + a few OKs for context
        if maxd>tol or n in common[:3]:
            print(f"{n[:52]:52s} {status:10s} {maxd}")
    print()
    if first_diff:
        print(f">>> FIRST DIVERGENCE between True and False: {first_diff}")
        print(f"    ({ndiff} nodes differ). Everything before this is identical.")
        # show the diverging values
        n=first_diff
        for ci in sorted(nt[n]):
            if ci in nf[n]:
                print(f"    ch{ci} TRUE : {nt[n][ci][:8]}")
                print(f"    ch{ci} FALSE: {nf[n][ci][:8]}")
                break
    else:
        print(">>> TRUE and FALSE are IDENTICAL on all common nodes.")
        print("    The fusion is numerically equivalent -- any detection difference is")
        print("    NOT from the fusion math (check build hygiene / deeper output nodes).")

# ------------------------- main -------------------------
def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest="cmd", required=True)

    c=sub.add_parser("capture", help="read device serial and save to a labeled file")
    c.add_argument("--mode", required=True, choices=["true","false"],
                   help="which build is flashed (FUSE_SILU_QUANTIZE true/false)")
    c.add_argument("--port", required=True)
    c.add_argument("--baud", type=int, default=115200)
    c.add_argument("--out", required=True)

    cmp=sub.add_parser("compare", help="compare two saved captures")
    cmp.add_argument("--true", required=True, dest="true_path")
    cmp.add_argument("--false", required=True, dest="false_path")
    cmp.add_argument("--tol", type=int, default=0, help="max |Δ| allowed before flagging (default 0 = exact)")

    a=ap.parse_args()
    if a.cmd=="capture":
        capture(a.port, a.baud, a.out, a.mode)
    else:
        compare(a.true_path, a.false_path, a.tol)

if __name__=="__main__":
    main()
