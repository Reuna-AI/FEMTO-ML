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
viz.py — visualize the compiler's JSON IR as a dataflow graph (Graphviz DOT).

  python viz.py graph_fused.json graph.dot
  # then:  dot -Tsvg graph.dot -o graph.svg     (needs graphviz installed)
  # or paste graph.dot into https://dreampuf.github.io/GraphvizOnline

Colors: Conv/QLinearConv=blue, SiLU=orange, Transpose=green (spot layout flips),
elementwise/other=gray. Edge labels = tensor name + shape, so you can verify
NHWC shapes and that SiLU fused correctly.
"""
import sys, json, argparse

COLOR={"QLinearConv":"#4477aa","Conv":"#4477aa","SiLU":"#ee8844",
       "Transpose":"#66aa66","QLinearConcat":"#aa66aa","Concat":"#aa66aa",
       "MaxPool":"#888844","Resize":"#888844","Upsample":"#888844"}

def esc(s): return s.replace('"','\\"').replace(":","_")

def main(inp,out):
    d=json.load(open(inp,encoding="utf-8"))
    T=d["tensors"]
    with open(out,"w",encoding="utf-8") as f:
        f.write('digraph G {\n  rankdir=TB;\n  node [shape=box,style=filled,fontname=Helvetica,fontsize=10];\n')
        for i,n in enumerate(d["nodes"]):
            c=COLOR.get(n["op"],"#cccccc")
            label=f'{n["op"]}\\n{esc(n["name"])[:40]}'
            f.write(f'  n{i} [label="{label}",fillcolor="{c}"];\n')
        # edges via producer map
        prod={o:i for i,n in enumerate(d["nodes"]) for o in n["outputs"]}
        for i,n in enumerate(d["nodes"]):
            for inp in n["inputs"]:
                if inp in prod:
                    sh=T.get(inp,{}).get("shape","")
                    lay=T.get(inp,{}).get("layout","")
                    f.write(f'  n{prod[inp]} -> n{i} [label="{esc(inp)[:24]}\\n{sh} {lay}",fontsize=8];\n')
        f.write("}\n")
    print(f"viz -> {out}  ({len(d['nodes'])} nodes). Render: dot -Tsvg {out} -o graph.svg")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
    a=ap.parse_args(); main(a.inp,a.out)
