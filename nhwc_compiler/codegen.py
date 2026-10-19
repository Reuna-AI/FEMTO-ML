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
codegen.py — dispatcher. Walks fused JSON, calls per-op emitters (emitters/op_*.py),
writes model.c + model.h. ONE kernel per op, chosen by --target (no #if in the C).

  python codegen.py graph_fused.json out/model --target esp32s3
  python codegen.py graph_fused.json out/model --target ansi_int8
  python codegen.py graph_fused.json out/model --target ansi_float

Functions take explicit buffer pointers: op_x(x, w, bias, y) / op_silu(x, y).
The driver run_model() calls them in order with the named activation/weight buffers.
"""
import sys, os, json, argparse, importlib, pkgutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emitters
from emitters import REGISTRY
from emitters.common import EmitCtx, san, wsan, cfloat, quant_c

def load_all():
    for _,name,_ in pkgutil.iter_modules(emitters.__path__):
        if name.startswith("op_"): importlib.import_module(f"emitters.{name}")

CDT={"int8":"int8_t","uint8":"uint8_t","int32":"int32_t","float32":"float","int64":"int64_t"}
def buf_ref(ctx, tname):
    """Flat-pointer C expression for a tensor buffer (matches const T* / T* params)."""
    t=ctx.T.get(tname,{})
    ct=CDT.get(t.get("dtype","float32"),"float")
    if t.get("const_file"):                     # file-backed weight array in weights.h
        return f"(const {ct}*){wsan(tname)}"
    if t.get("kind")=="init":                   # (should be inlined; passed only if needed)
        return f"(const {ct}*){wsan(tname)}"
    # activation: arrow-macro yields a real N-D array; cast decays to flat pointer
    return f"({ct}*){san(tname)}"

def main(json_path, out_base, target, simd=False, debug=False, dump=False, dump_n=10, dump_nodes=None, espnn=False):
    load_all()
    d=json.load(open(json_path,encoding="utf-8"))
    ctx=EmitCtx(d["tensors"], target, simd)
    ctx.espnn = espnn
    skipped={}
    for n in d["nodes"]:
        fn=REGISTRY.get(n["op"])
        if fn: fn(ctx,n)
        else: skipped[n["op"]]=skipped.get(n["op"],0)+1

    base=out_base.split("/")[-1].split("\\")[-1]

    # A model that quantizes its float NCHW input and transposes it to NHWC before
    # anything else: the caller can write that int8 NHWC tensor itself (one byte a
    # value, not four, straight from its camera's pixels) and skip both.
    skip_input=None
    if len(ctx.calls)>=2:
        q=ctx.calls[0][0]; t=ctx.calls[1][0]
        if (q["op"]=="QuantizeLinear" and d.get("inputs") and q["inputs"][0]==d["inputs"][0]
                and t["op"]=="Transpose" and t["inputs"][0]==q["outputs"][0]
                and t["attrs"].get("perm")==[0,2,3,1] and len(ctx.shape(t["outputs"][0]))==4
                and ctx.val0(q["inputs"][1]) is not None):
            zq=ctx.val0(q["inputs"][2]) if len(q["inputs"])>2 else 0
            skip_input=(t["outputs"][0], float(ctx.val0(q["inputs"][1])), int(zq or 0))
    # ---- header ----
    with open(out_base+".h","w",encoding="utf-8") as h:
        h.write("// Auto-generated model header. Do not edit.\n#pragma once\n")
        h.write("#include <stdint.h>\n#include <string.h>\n#include <math.h>\n")
        h.write('#include "weights.h"\n#include "activations.h"\n\n')
        h.write(f"// target: {target}\n")
        if target=="esp32s3" and (debug or dump):
            h.write("#include <Arduino.h>\n")
        if target=="esp32s3" and simd:
            h.write('#include "esp_nn_conv_s8_simd.h"\n')
            h.write('#include "esp_nn_depthwise_conv_s8_simd.h"\n')
        h.write(f"#ifndef DEBUG_TRACE\n#define DEBUG_TRACE {1 if (target=='esp32s3' and debug) else 0}\n#endif\n")
        h.write(f"#ifndef DEBUG_DUMP\n#define DEBUG_DUMP {1 if (target=='esp32s3' and dump) else 0}\n#endif\n")
        h.write(f"#ifndef DUMP_N\n#define DUMP_N {dump_n}\n#endif\n")
        h.write("#ifndef MODEL_DEBUG\n#define MODEL_DEBUG %d\n#endif\n" % (1 if debug else 0))
        # FEMTO_PROFILE=1 (a -D flag, nothing regenerated): time every node with
        # femto_profile.c and report by op type and slowest node.
        h.write("#ifndef FEMTO_PROFILE\n#define FEMTO_PROFILE 0\n#endif\n")
        h.write("#if MODEL_DEBUG\n  #include <stdio.h>\n")
        h.write("  #define DBG_NODE(i,op,name) printf(\"[%d] %s %s\\n\",(i),(op),(name))\n")
        h.write("  #define DBG_STAT(name,ptr,n,isf) dbg_stat((name),(ptr),(n),(isf))\n")
        h.write("  #define DBG_END()\n")
        h.write("#elif FEMTO_PROFILE\n  #include \"femto_profile.h\"\n")
        h.write("  #define DBG_NODE(i,op,name) femto_prof_node((i),(op),(name))\n")
        h.write("  #define DBG_STAT(name,ptr,n,isf)\n")
        h.write("  #define DBG_END() femto_prof_end()\n")
        h.write("#else\n  #define DBG_NODE(i,op,name)\n  #define DBG_STAT(name,ptr,n,isf)\n  #define DBG_END()\n#endif\n")
        h.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")
        for line in ctx.header: h.write(line+"\n")
        h.write("\nvoid run_model(void);\n")
        if skip_input:
            name, sc, zp = skip_input
            _,H,W,C = ctx.shape(name)
            h.write("\n/* The input as the first conv reads it: int8, [H][W][C], quantized as below.\n")
            h.write(" * Fill it (model_quantize_input() is the model's own QuantizeLinear) and call\n")
            h.write(" * run_model_int8_input() in place of filling the float input and run_model(). */\n")
            h.write("#define MODEL_INPUT_INT8 1\n")
            h.write(f"#define MODEL_INPUT_H {H}\n#define MODEL_INPUT_W {W}\n#define MODEL_INPUT_C {C}\n")
            h.write(f"#define MODEL_INPUT_SCALE {cfloat(sc)}\n#define MODEL_INPUT_ZP {zp}\n")
            h.write(f"#define model_input_int8 ((int8_t*){san(name)})\n")
            h.write("static inline int8_t model_quantize_input(float x) {\n")
            h.write("  const float inv=1.0f/MODEL_INPUT_SCALE; const int z=MODEL_INPUT_ZP; int8_t y;\n")
            h.write("  " + quant_c("y", "x") + "\n  return y;\n}\n")
            h.write("void run_model_int8_input(void);\n")
        h.write("\n#ifdef __cplusplus\n}\n#endif\n")

    # ---- body + driver ----
    with open(out_base+".c","w",encoding="utf-8") as c:
        c.write(f'// Auto-generated model body (target={target}, simd={simd}). Do not edit.\n')
        c.write(f'#include "{base}.h"\n\n')
        # NOTE: do NOT #include weights_*.c here. Arduino compiles every .c in the
        # sketch folder on its own; including them too causes multiple-definition.
        c.write("#if MODEL_DEBUG\n")
        c.write("static void dbg_stat(const char* nm, const void* p, int n, int isf){\n")
        c.write("  if(!p){ printf(\"   %s: NULL PTR!\\n\", nm); return; }\n")
        c.write("  double mn=1e30,mx=-1e30,sum=0; int k=n<8?n:8;\n")
        c.write("  if(isf){ const float* f=(const float*)p; for(int i=0;i<n;++i){double v=f[i]; if(v<mn)mn=v; if(v>mx)mx=v; sum+=v;}\n")
        c.write("    printf(\"   %s[%d] min=%.4f max=%.4f mean=%.4f first:\", nm,n,mn,mx,sum/n); for(int i=0;i<k;++i) printf(\" %.3f\", f[i]); printf(\"\\n\"); }\n")
        c.write("  else { const signed char* q=(const signed char*)p; for(int i=0;i<n;++i){int v=q[i]; if(v<mn)mn=v; if(v>mx)mx=v; sum+=v;}\n")
        c.write("    printf(\"   %s[%d] min=%.0f max=%.0f mean=%.2f first:\", nm,n,mn,mx,sum/n); for(int i=0;i<k;++i) printf(\" %d\", (int)q[i]); printf(\"\\n\"); } }\n")
        c.write("#endif\n\n")
        for line in ctx.body: c.write(line+"\n")
        # driver: call each op with explicit buffers
        c.write("#if DEBUG_TRACE\n#define TRACE(name) Serial.printf(\"[node] %s\\n\", name); Serial.flush()\n#else\n#define TRACE(name)\n#endif\n\n")
        # activation dumper: prints first 8 channels, DUMP_N vals/channel, raw+dequant.
        # args: name, base pointer (int8 or float), H,W,C, is_int8, scale, zp
        c.write("#if DEBUG_DUMP\n")
        c.write("static void dump_i8(const char* nm, const signed char* p, int H,int W,int C, float sc,int zp,int nchw){\n")
        c.write('  Serial.printf("== %s [int8 H%d W%d C%d %s] sc=%.6g zp=%d\\n", nm,H,W,C,(nchw?"NCHW":"NHWC"),sc,zp);\n')
        c.write("  int cn=C<8?C:8;\n")
        c.write("  for(int c=0;c<cn;++c){ Serial.printf(\"  ch%d:\", c);\n")
        c.write("    for(int k=0;k<DUMP_N;++k){ int h=k/W, w=k%W; if(h>=H)break;\n")
        c.write("      int off = nchw ? (c*H*W + h*W + w) : ((h*W+w)*C+c);\n")
        c.write("      int v=p[off]; Serial.printf(\" %d(%.4f)\", v, sc*(v-zp)); }\n")
        c.write("    Serial.printf(\"\\n\"); } Serial.flush();\n}\n")
        c.write("static void dump_f(const char* nm, const float* p, int H,int W,int C,int nchw){\n")
        c.write('  Serial.printf("== %s [float H%d W%d C%d %s]\\n", nm,H,W,C,(nchw?"NCHW":"NHWC"));\n')
        c.write("  int cn=C<8?C:8;\n")
        c.write("  for(int c=0;c<cn;++c){ Serial.printf(\"  ch%d:\", c);\n")
        c.write("    for(int k=0;k<DUMP_N;++k){ int h=k/W, w=k%W; if(h>=H)break;\n")
        c.write("      int off = nchw ? (c*H*W + h*W + w) : ((h*W+w)*C+c);\n")
        c.write("      Serial.printf(\" %.4f\", p[off]); }\n")
        c.write("    Serial.printf(\"\\n\"); } Serial.flush();\n}\n")
        # generic flat dumper for non-4D tensors: prints C "channels", `per` values each
        c.write("static void dump_flat_i8(const char* nm, const signed char* p, int C,int per, float sc,int zp){\n")
        c.write('  Serial.printf("== %s [int8 flat C%d per%d] sc=%.6g zp=%d\\n", nm,C,per,sc,zp);\n')
        c.write("  int cn=C<8?C:8;\n")
        c.write("  for(int c=0;c<cn;++c){ Serial.printf(\"  ch%d:\", c);\n")
        c.write("    for(int k=0;k<DUMP_N && k<per;++k){ int v=p[c*per+k]; Serial.printf(\" %d(%.4f)\", v, sc*(v-zp)); }\n")
        c.write("    Serial.printf(\"\\n\"); } Serial.flush();\n}\n")
        c.write("static void dump_flat_f(const char* nm, const float* p, int C,int per){\n")
        c.write('  Serial.printf("== %s [float flat C%d per%d]\\n", nm,C,per);\n')
        c.write("  int cn=C<8?C:8;\n")
        c.write("  for(int c=0;c<cn;++c){ Serial.printf(\"  ch%d:\", c);\n")
        c.write("    for(int k=0;k<DUMP_N && k<per;++k){ Serial.printf(\" %.4f\", p[c*per+k]); }\n")
        c.write("    Serial.printf(\"\\n\"); } Serial.flush();\n}\n")
        c.write("#define DUMP_FLAT_I8(nm,p,C,per,sc,zp) dump_flat_i8(nm,(const signed char*)(p),C,per,sc,zp)\n")
        c.write("#define DUMP_FLAT_F(nm,p,C,per) dump_flat_f(nm,(const float*)(p),C,per)\n")
        c.write("#define DUMP_I8(nm,p,H,W,C,sc,zp,nchw) dump_i8(nm,(const signed char*)(p),H,W,C,sc,zp,nchw)\n")
        c.write("#define DUMP_F(nm,p,H,W,C,nchw) dump_f(nm,(const float*)(p),H,W,C,nchw)\n")
        c.write("#else\n#define DUMP_I8(nm,p,H,W,C,sc,zp,nchw)\n#define DUMP_F(nm,p,H,W,C,nchw)\n#define DUMP_FLAT_I8(nm,p,C,per,sc,zp)\n#define DUMP_FLAT_F(nm,p,C,per)\n#endif\n\n")
        c.write("static void run_from(int first){\n  (void)first;\n")
        def dump_call(node):
            outs=node["outputs"]
            if not outs: return ""
            if dump_nodes:
                nm=node["name"]
                if not any(sub in nm for sub in dump_nodes): return ""
            o=outs[0]; t=ctx.T.get(o,{}); sh=t.get("shape",[])
            dt=t.get("dtype","int8"); ref=buf_ref(ctx,o)
            def _numel(dd):
                n=1
                for v in dd:
                    if not isinstance(v,int) or v<0: return 0
                    n*=v
                return n
            if len(sh)==4:
                lay=t.get("layout","nhwc")
                if lay=="nchw":  _,C,H,W=sh; isn=1
                else:            _,H,W,C=sh; isn=0
                if dt in ("int8","uint8"):
                    sc="0.0f"; zp="0"
                    if node["op"]=="QLinearConv" and len(node["inputs"])>=8:
                        sv=ctx.val0(node["inputs"][6]); zv=ctx.val0(node["inputs"][7])
                        if sv is not None: sc=f"{float(sv):.9g}f"
                        if zv is not None: zp=str(int(zv))
                    elif node["op"]=="QuantizeLinear" and len(node["inputs"])>=2:
                        sv=ctx.val0(node["inputs"][1])
                        zv=ctx.val0(node["inputs"][2]) if len(node["inputs"])>2 else 0
                        if sv is not None: sc=f"{float(sv):.9g}f"
                        if zv is not None: zp=str(int(zv))
                    return f'  DUMP_I8("{node["name"]}", {ref}, {H},{W},{C}, {sc},{zp},{isn});\n'
                return f'  DUMP_F("{node["name"]}", {ref}, {H},{W},{C},{isn});\n'
            # ---- non-4D: flat dump. C = axis 1 if rank>=2 (drop batch), else 1 ----
            tot=_numel(sh)
            if tot==0: return ""    # dynamic/unknown shape, skip
            if len(sh)>=2:
                C = sh[1] if sh[0]==1 else sh[0]
            else:
                C = 1
            per = tot // C if C>0 else tot
            isf = 0 if dt in ("int8","uint8") else 1
            if isf:
                return f'  DUMP_FLAT_F("{node["name"]}", {ref}, {C},{per});\n'
            else:
                sc="0.0f"; zp="0"
                if node["op"]=="QuantizeLinear" and len(node["inputs"])>=2:
                    sv=ctx.val0(node["inputs"][1])
                    zv=ctx.val0(node["inputs"][2]) if len(node["inputs"])>2 else 0
                    if sv is not None: sc=f"{float(sv):.9g}f"
                    if zv is not None: zp=str(int(zv))
                return f'  DUMP_FLAT_I8("{node["name"]}", {ref}, {C},{per}, {sc},{zp});\n'
        for _idx,rec in enumerate(ctx.calls):
            node=rec[0]; fn=rec[1]
            if skip_input and _idx in (1, 2):
                c.write("  }\n")
            if skip_input and _idx<2:
                c.write(f"  if(first<={_idx}){{\n")
            c.write(f'  DBG_NODE({_idx}, "{node["op"]}", "{node["name"]}");\n')
            ins=node["inputs"]; outs=node["outputs"]
            if node["op"]=="QLinearConv":
                x=buf_ref(ctx,ins[0]); w=buf_ref(ctx,ins[3])
                bias=(f"(const int32_t*){wsan(ins[8])}") if len(ins)>8 else "0"
                y=buf_ref(ctx,outs[0])
                c.write(f'  TRACE("{node["name"]}");\n  {fn}({x}, {w}, {bias}, {y});\n'); c.write(dump_call(node))
            elif node["op"]=="SiLU" and len(rec)<3:
                c.write(f'  TRACE("{node["name"]}");\n  {fn}({buf_ref(ctx,ins[0])}, {buf_ref(ctx,outs[0])});\n'); c.write(dump_call(node))
            elif len(rec)>=3:
                names=rec[2]
                # If rec[2] holds REAL tensor names (emitter chose exact args), use them.
                if names and all((t in ctx.T) for t in names):
                    exprs=[buf_ref(ctx,t) for t in names]
                else:
                    # positional: pass only real data buffers, then outputs
                    def is_data(t):
                        if t=="": return False
                        m=ctx.T.get(t,{})
                        if m.get("value") is not None: return False
                        if m.get("const_file"):        return False
                        if m.get("kind")=="init":       return False
                        return True
                    data_ins=[t for t in ins if is_data(t)]
                    allt=data_ins+list(outs)
                    exprs=[buf_ref(ctx,t) for t in allt][:len(names)]
                c.write(f'  TRACE("{node["name"]}");\n  {fn}({", ".join(exprs)});\n'); c.write(dump_call(node))
        c.write("  DBG_END();\n}\n\n")
        c.write("void run_model(void){ run_from(0); }\n")
        if skip_input:
            c.write("void run_model_int8_input(void){ run_from(2); }\n")

    n_ok=len(d["nodes"])-sum(skipped.values())
    print(f"codegen[{target}] -> {out_base}.c/.h : {n_ok} ops emitted")
    if skipped:
        print("  UNHANDLED ops (need emitters/op_*.py):")
        for op,cnt in sorted(skipped.items()): print(f"    {op}: {cnt}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("json"); ap.add_argument("out")
    ap.add_argument("--target", choices=["esp32s3","ansi_int8","ansi_float"], default="ansi_int8")
    ap.add_argument("--simd", action="store_true", help="emit ESP32-S3 vector (ACCX) conv; requires --target esp32s3")
    ap.add_argument("--debug", action="store_true", help="emit per-node trace prints")
    ap.add_argument("--dump", action="store_true", help="per-node activation dump (esp32s3)")
    ap.add_argument("--dump-n", type=int, default=10, help="values per channel to dump")
    ap.add_argument("--dump-nodes", nargs="*", default=None,
                    help="only dump these node names (substring match). default: all")
    ap.add_argument("--esp-nn", dest="espnn", action="store_true",
                    help="route eligible convs to esp_nn_conv_s8_simd library (esp32s3)")
    a=ap.parse_args(); main(a.json, a.out, a.target, a.simd, a.debug, a.dump, a.dump_n, a.dump_nodes, a.espnn)
