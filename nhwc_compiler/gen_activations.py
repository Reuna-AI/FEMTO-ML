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
gen_activations.py — Stage 3. Emit activations.h: intermediate tensors packed into
unions by lifetime, exactly like your target:

    union tensor_union_0 { float tensor_x[..]; ... }; static union tensor_union_0 tu0;

Liveness: for each activation tensor compute [def, last_use] over the topological
node order. Then greedily assign tensors to unions such that NO two tensors sharing
a union have overlapping lifetimes -> they safely alias the same memory. This is a
classic interval-graph coloring; each "color" = one union.

  python gen_activations.py graph_nhwc.json activations.h

Notes:
  * inputs/outputs/consts are NOT unioned (they must persist / live in weights.h);
    they get their own standalone buffers.
  * dtype from the JSON (int8 for activations here). Size = product(shape)*sizeof.
"""
import sys, json

CDT={"int8":"int8_t","uint8":"uint8_t","int32":"int32_t","float32":"float","int64":"int64_t"}
CSZ={"int8":1,"uint8":1,"int32":4,"float32":4,"int64":8}

# ---- Hybrid arena placement (esp32s3 / esp32s2) --------------------------------
# Hot/priority unions go in fast INTERNAL SRAM up to SRAM_BUDGET_BYTES; the rest
# spill to PSRAM. Access is via tuN pointers, so placement is purely an alloc-time
# choice -- kernels and access macros are unchanged.
SRAM_BUDGET_BYTES = 128 * 1024      # hard internal-SRAM budget for the arena
SRAM_PRIORITY     = "hot"           # "hot" = most-accessed unions first
                                    # "large" = largest unions first
# --------------------------------------------------------------------------------

def san(n): return "tensor_"+"".join(c if c.isalnum() else "_" for c in n)

def sh_of(meta): return meta.get('shape',[])

def main(json_path, out_h, target="ansi_int8"):
    d=json.load(open(json_path)); T=d["tensors"]; nodes=d["nodes"]
    order={}                      # node index
    # 1) liveness intervals over node order
    defd={}; lastuse={}
    for idx,n in enumerate(nodes):
        for o in n["outputs"]:
            if o not in defd: defd[o]=idx
        for i in n["inputs"]:
            lastuse[i]=idx
    # tensors that are graph outputs live to the end
    end=len(nodes)
    for o in d["outputs"]: lastuse[o]=end

    # 2) candidates = activation tensors (not const/init, not graph input/output-persistent)
    persistent=set(d["inputs"])|set(d["outputs"])
    cand=[]
    for name,meta in T.items():
        if meta.get("const_file"): continue
        if meta["kind"]=="init": continue
        if name in persistent: continue
        if name not in defd: continue          # never produced (e.g. pure input) -> skip
        sh=meta["shape"]
        if any((not isinstance(x,int)) or x<0 for x in sh):  # dynamic -> give own buffer
            continue
        numel=1
        for x in sh: numel*=x
        cand.append({"name":name,"def":defd[name],"end":lastuse.get(name,defd[name]),
                     "dtype":meta["dtype"],"shape":sh,"bytes":numel*CSZ.get(meta["dtype"],4)})

    # 3) interval-graph greedy coloring: sort by def; assign to first union whose
    #    tensors' intervals don't overlap this one.
    cand.sort(key=lambda t:(t["def"], -t["bytes"]))
    unions=[]                     # each: list of members; and its current 'busy until' per member
    def overlaps(a,b): return not (a["end"]<b["def"] or b["end"]<a["def"])  # inclusive: in/out of same op conflict
    for t in cand:
        placed=False
        for u in unions:
            if all(not overlaps(t,m) for m in u):
                u.append(t); placed=True; break
        if not placed:
            unions.append([t])

    # 3b) Access frequency per tensor = number of nodes that read or write it.
    #     Union access count = sum over its members (members alias but are used at
    #     different times, so total touches approximates time spent in that buffer).
    acc_count={}
    for n in nodes:
        for nm in list(n["inputs"])+list(n["outputs"]):
            acc_count[nm]=acc_count.get(nm,0)+1
    def union_size(u): return max(m["bytes"] for m in u)
    def union_accesses(u): return sum(acc_count.get(m["name"],0) for m in u)

    # 3c) Decide SRAM vs PSRAM per union under the budget (greedy, skip-if-too-big).
    order=list(range(len(unions)))
    if SRAM_PRIORITY=="large":
        order.sort(key=lambda i:-union_size(unions[i]))
    else:  # "hot"
        order.sort(key=lambda i:-union_accesses(unions[i]))
    in_sram=set(); used=0
    for i in order:
        sz=union_size(unions[i])
        if sz<=SRAM_BUDGET_BYTES-used:      # fits in remaining budget
            in_sram.add(i); used+=sz
        # else: skip -> PSRAM, keep trying smaller ones (greedy)
    sram_report=(len(in_sram), used, len(unions)-len(in_sram))

    # 4) emit
    with open(out_h,"w") as f:
        f.write("// Auto-generated activation arena (int8 NHWC). Do not edit.\n")
        f.write("#pragma once\n#include <stdint.h>\n")
        f.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")
        # ---- real union structs + PSRAM-allocated pointers ----
        esp = (target in ("esp32s3","esp32s2"))
        if esp:
            f.write("#include <stdbool.h>\n#include <stddef.h>\n#include <stdlib.h>\n")
            f.write("#include \"esp_heap_caps.h\"\n")
            f.write("// Hybrid arena: hot/priority unions in INTERNAL SRAM (fast), rest in PSRAM.\n")
            f.write(f"// SRAM budget = {SRAM_BUDGET_BYTES} B, priority = {SRAM_PRIORITY}.\n")
            f.write("#define ARENA_ALLOC_SRAM(sz)  heap_caps_malloc((sz), MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT)\n")
            f.write("#define ARENA_ALLOC_PSRAM(sz) heap_caps_malloc((sz), MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT)\n")
            f.write("#define ARENA_FREE(p) heap_caps_free((p))\n\n")
        else:
            f.write("#include <stdlib.h>\n")
            f.write("#define ARENA_ALLOC_SRAM(sz)  malloc((sz))\n")
            f.write("#define ARENA_ALLOC_PSRAM(sz) malloc((sz))\n")
            f.write("#define ARENA_FREE(p) free((p))\n\n")
        for ui,u in enumerate(unions):
            sz=max(m["bytes"] for m in u)
            place="SRAM" if ui in in_sram else "PSRAM"
            f.write(f"union tensor_union_{ui} {{   // {sz} B, {len(u)} tensors, {place}"
                    f" (acc={union_accesses(u)})\n")
            for m in u:
                ct=CDT.get(m["dtype"],"float")
                dims="".join(f"[{x}]" for x in m["shape"]) if m["shape"] else "[1]"
                f.write(f"    {ct} {san(m['name'])}{dims}; // {m['bytes']} B  live[{m['def']},{m['end']}]\n")
            f.write(f"}};\nextern union tensor_union_{ui} *tu{ui};\n\n")

        # Split remaining tensors into: LIVE (produced or graph input/output) vs
        # DEAD (unproduced -> pre-fusion leftovers). Emit them in separate, labeled
        # sections so the header stays auditable.
        done=set(m["name"] for u in unions for m in u)
        produced=set(defd.keys())
        graph_io=set(d["inputs"])|set(d["outputs"])

        # Precise cause-of-death classification. A dead tensor is one that is neither
        # produced by any node nor a graph input/output. WHY it's dead:
        #   'fusion'    : it was consumed only by nodes that fusion removed
        #                 (its consumers no longer exist -> it fed a fused subgraph).
        #   'transpose' : it is the pre-transpose original whose producer got rewired
        #                 to a transposed (_T) output by the NHWC pass.
        #   'unused'    : produced by nobody and consumed by nobody (truly inert).
        # We classify by checking the CURRENT graph, not by name substrings.
        cur_out = produced                         # tensors some surviving node outputs
        cur_in  = set(i for n in d["nodes"] for i in n["inputs"])   # tensors some surviving node reads
        # a transposed replacement exists if some Transpose node takes THIS tensor as input
        transposed_src = set()
        for n in d["nodes"]:
            if n["op"]=="Transpose":
                for i in n["inputs"]: transposed_src.add(i)

        def classify(name):
            if name in transposed_src:      # still fed into a transpose but itself unproduced
                return "transpose"
            if name not in cur_in:          # nobody reads it AND nobody writes it
                return "unused"
            return "fusion"                 # still read by a surviving node but its producer is gone
                                            #  -> producer was a node fusion collapsed

        live=[]; dead={"fusion":[], "transpose":[], "unused":[]}
        for name,meta in T.items():
            if meta.get("const_file") or meta["kind"]=="init": continue
            if name in done: continue
            sh=meta["shape"]
            if any((not isinstance(x,int)) or x<0 for x in sh): continue
            if name in produced or name in graph_io:
                live.append((name,meta))
            else:
                dead[classify(name)].append((name,meta))

        f.write("// persistent / standalone buffers (live: produced or graph I/O)\n")
        # On esp32s3 these can be large (e.g. the [3087][85] output) -> put them in
        # PSRAM via pointers too, else they overflow internal DRAM .bss.
        standalone=[]
        for name,meta in live:
            ctype=CDT.get(meta["dtype"],"float")
            dims="".join(f"[{x}]" for x in meta["shape"]) if meta["shape"] else "[1]"
            numel=1
            for x in meta["shape"]: numel*=x
            nbytes=numel*{"int8":1,"uint8":1,"int32":4,"float":4,"float32":4,"int64":8}.get(meta["dtype"],4)
            if target in ("esp32s3","esp32s2"):
                # pointer + typed macro; alloc in init
                inner="".join(f"[{x}]" for x in meta["shape"][1:]) if len(meta["shape"])>1 else ""
                f.write(f"extern {ctype} (*{san(name)}_p){inner if inner else '[1]'}; // {nbytes} B PSRAM\n")
                standalone.append((name,meta,ctype,inner,nbytes))
            else:
                f.write(f"static {ctype} {san(name)}{dims};\n")
        # access macros for esp32s3 standalone pointers: tensor_x -> (*tensor_x_p)
        for name,meta,ctype,inner,nbytes in standalone:
            if inner:
                f.write(f"#define {san(name)} (*{san(name)}_p)\n")
            else:
                f.write(f"#define {san(name)} ({san(name)}_p)\n")

        LABELS={
          "fusion":    "dead: consumed only by nodes removed during fusion (e.g. Sigmoid/Mul -> SiLU)",
          "transpose": "dead: pre-transpose original, replaced by an NHWC-pass Transpose (_T) output",
          "unused":    "dead: produced and consumed by nothing (inert)",
        }
        for grp in ("fusion","transpose","unused"):
            items=dead[grp]
            if not items: continue
            f.write("\n// -------------------------------------------------------------\n")
            f.write(f"// {LABELS[grp]}\n")
            f.write("// kept commented for auditability; not referenced by run_model. Safe to delete.\n")
            f.write("// -------------------------------------------------------------\n")
            for name,meta in items:
                ctype=CDT.get(meta["dtype"],"float"); dims="".join(f"[{x}]" for x in meta["shape"]) if meta["shape"] else "[1]"
                f.write(f"// static {ctype} {san(name)}{dims};\n")

        # access macros: arrow into the allocated union (members are real arrays)
        f.write("\n// access macros -> tuN->member (real typed arrays)\n")
        for ui,u in enumerate(unions):
            for m in u:
                f.write(f"#define {san(m['name'])} (tu{ui}->{san(m['name'])})\n")
        # init / free
        # DECLARATIONS ONLY in the header (definitions go to activations.c, compiled once)
        f.write("\nint activations_init(void);\nvoid activations_free(void);\n")

        try:
            _standalone=standalone
        except NameError:
            _standalone=[]
        # write the DEFINITIONS to a companion .c next to the header
        import os as _os
        cpath=_os.path.join(_os.path.dirname(out_h) or ".", "activations.c")
        with open(cpath,"w",encoding="utf-8") as g:
            g.write('// Auto-generated arena allocation. Do not edit.\n')
            g.write('#include "activations.h"\n\n')
            # single definitions of the pointer variables
            for ui in range(len(unions)):
                g.write(f"union tensor_union_{ui} *tu{ui} = 0;\n")
            for (name,meta,ctype,inner,nbytes) in _standalone:
                g.write(f"{ctype} (*{san(name)}_p){inner if inner else '[1]'} = 0;\n")
            g.write("\n")
            g.write("int activations_init(void){\n")
            for ui in range(len(unions)):
                alloc = "ARENA_ALLOC_SRAM" if ui in in_sram else "ARENA_ALLOC_PSRAM"
                g.write(f"  tu{ui}=(union tensor_union_{ui}*){alloc}(sizeof(union tensor_union_{ui}));\n")
                # SRAM alloc may fail if internal RAM is fragmented/exhausted -> fall back to PSRAM
                if ui in in_sram:
                    g.write(f"  if(!tu{ui}) tu{ui}=(union tensor_union_{ui}*)ARENA_ALLOC_PSRAM(sizeof(union tensor_union_{ui}));\n")
                g.write(f"  if(!tu{ui}) return {ui+1};\n")
            base=len(unions)
            for j,(name,meta,ctype,inner,nbytes) in enumerate(_standalone):
                cast=f"({ctype}(*){inner if inner else '[1]'})"
                # standalone buffers (often large, e.g. output) -> PSRAM
                g.write(f"  {san(name)}_p={cast}ARENA_ALLOC_PSRAM({nbytes}); if(!{san(name)}_p) return {base+j+1};\n")
            g.write("  return 0;\n}\n")
            g.write("void activations_free(void){\n")
            for ui in range(len(unions)):
                g.write(f"  if(tu{ui}){{ ARENA_FREE(tu{ui}); tu{ui}=0; }}\n")
            for (name,meta,ctype,inner,nbytes) in _standalone:
                g.write(f"  if({san(name)}_p){{ ARENA_FREE({san(name)}_p); {san(name)}_p=0; }}\n")
            g.write("}\n")

        f.write("\n#ifdef __cplusplus\n}\n#endif\n")

    # placement report to stdout
    _ns,_used,_nps=sram_report
    print(f"[arena] unions={len(unions)}  SRAM={_ns} ({_used} B / {SRAM_BUDGET_BYTES} B budget, "
          f"{SRAM_PRIORITY})  PSRAM={_nps}")

    total=sum(max(m['bytes'] for m in u) for u in unions)
    naive=sum(m['bytes'] for u in unions for m in u)
    print(f"gen_activations -> {out_h}: {len(unions)} unions, {len(cand)} tensors")
    print(f"  arena bytes (unioned) = {total}   vs naive = {naive}   saved {100*(1-total/max(naive,1)):.0f}%")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("json"); ap.add_argument("out")
    ap.add_argument("--target", choices=["esp32s3","esp32s2","ansi_int8","ansi_float"], default="ansi_int8")
    a=ap.parse_args(); main(a.json, a.out, a.target)
