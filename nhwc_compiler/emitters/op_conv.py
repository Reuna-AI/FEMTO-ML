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
op_conv.py — NHWC int8 QLinearConv, ONE kernel chosen by ctx.target:
  esp32s3    : ACCX SIMD (aligned EE.VLD.128 + EE.VMULAS.S8.ACCX), int8
  ansi_int8  : portable integer reference, int8
  ansi_float : dequantize inputs to float, float MAC, requantize (reference / debug)

Explicit args:  op_x(const int8_t* x, const int8_t* w, const int32_t* bias, int8_t* y)
Per-channel mult[] and adj[] are emitted as COMPILE-TIME CONSTANT arrays using the
inlined scale/zp values from the JSON (no runtime prep). Weights are file-backed
(passed in as `w`); bias passed as `bias`; scales/zp are inlined literals.
"""
from . import register

import math as _math
def _tflm_mult_shift(M):
    """Canonical TFLite QuantizeMultiplier.
       Given real multiplier M in (0,1) (typically), returns (mult, shift) with
       mult in [2^30, 2^31) int32, and integer shift, such that:
           M == mult * 2^(shift-31)
       esp-nn's multiply_by_quantized_mult(acc, mult, shift) is TFLM-compatible:
       it computes round(acc * mult / 2^31) then applies `shift`
       (shift>0 => left shift before/round; TFLite splits into exponent).
       This is the same (mult, shift) TFLite Micro feeds its kernels."""
    if M <= 0.0:
        return (0, 0)
    q, e = _math.frexp(M)             # M = q * 2^e, q in [0.5, 1)
    mult = int(round(q * (1 << 31)))  # q*2^31 in [2^30, 2^31]
    shift = e
    if mult == (1 << 31):             # rounded up to 2^31 -> renormalize
        mult //= 2
        shift += 1
    # clamp shift to int8-ish range TFLite uses
    if shift < -31:
        mult = 0; shift = 0
    return (mult, shift)


def _emit_espnn_call(ctx, node, K,Kh,Kw,Cp,H,W,Ho,Wo,sh,sw,pt,pl,XS,XZ,YS,YZ,WSC,bias_name,x,w,y,fn):
    """Emit a C function that calls the external esp_nn_conv_s8_simd library."""
    # per-channel mult/shift
    mults=[]; shifts=[]
    for c in range(K):
        wsc = WSC[c] if (WSC and c < len(WSC)) else (WSC[0] if WSC else 1.0)
        M = (XS * wsc) / YS
        m,sft = _tflm_mult_shift(M)
        mults.append(m); shifts.append(sft)
    mult_str  = ", ".join(str(v) for v in mults)
    shift_str = ", ".join(str(v) for v in shifts)
    h=ctx.header; b=ctx.body
    sig=f"void {fn}(const int8_t* x, const int8_t* w, const int32_t* bias, int8_t* y)"
    h.append(sig+";")
    b.append(f"// QLinearConv {node['name']} via esp_nn_conv_s8_simd (SIMD lib)")
    b.append(sig+" {")
    b.append(f"  static const int32_t mlt[{K}]={{{mult_str}}};")
    b.append(f"  static const int32_t sft[{K}]={{{shift_str}}};")
    b.append(f"  data_dims_t id={{ .width={W}, .height={H}, .channels={Cp}, .extra=1 }};")
    b.append(f"  data_dims_t fd={{ .width={Kw}, .height={Kh}, .channels={Cp}, .extra={K} }};")
    b.append(f"  data_dims_t od={{ .width={Wo}, .height={Ho}, .channels={K}, .extra=1 }};")
    b.append(f"  conv_params_t cp={{ .in_offset={-XZ}, .out_offset={YZ},")
    b.append(f"      .stride={{{sw},{sh}}}, .padding={{{pl},{pt}}}, .dilation={{1,1}},")
    b.append(f"      .activation={{-128,127}} }};")
    b.append(f"  quant_data_t qd={{ .shift=sft, .mult=mlt }};")
    b.extend(_lut_call(ctx, node, "  esp_nn_conv_s8_simd(&id, x, &fd, w, bias, &od, y, &cp, &qd);"))
    b.append("}"); b.append("")
    ctx.calls.append((node, fn))


def _emit_dw_espnn_call(ctx, node, K,Kh,Kw,H,W,Ho,Wo,sh,sw,pt,pl,XS,XZ,YS,YZ,WSC,bias_name,x,w,y,fn):
    """Emit a call to esp_nn_depthwise_conv_s8_simd (ch_mult=1). Filter weights are
    stored HWC [Kh][Kw][K] (repacked in gen_weights). Per-output-channel mult/shift."""
    mults=[]; shifts=[]
    for c in range(K):
        wsc = WSC[c] if (WSC and c<len(WSC)) else (WSC[0] if WSC else 1.0)
        m,sft=_tflm_mult_shift((XS*wsc)/YS); mults.append(m); shifts.append(sft)
    mult_str=", ".join(str(v) for v in mults); shift_str=", ".join(str(v) for v in shifts)
    h=ctx.header; b=ctx.body
    sig=f"void {fn}(const int8_t* x, const int8_t* w, const int32_t* bias, int8_t* y)"
    h.append(sig+";")
    b.append(f"// Depthwise QLinearConv {node['name']} via esp_nn_depthwise_conv_s8_simd (SIMD lib)")
    b.append(sig+" {")
    b.append(f"  static const int32_t mlt[{K}]={{{mult_str}}};")
    b.append(f"  static const int32_t sft[{K}]={{{shift_str}}};")
    b.append(f"  data_dims_t id={{ .width={W}, .height={H}, .channels={K}, .extra=1 }};")
    b.append(f"  data_dims_t fd={{ .width={Kw}, .height={Kh}, .channels={K}, .extra=1 }};")
    b.append(f"  data_dims_t od={{ .width={Wo}, .height={Ho}, .channels={K}, .extra=1 }};")
    b.append(f"  dw_conv_params_t cp={{ .in_offset={-XZ}, .out_offset={YZ}, .ch_mult=1,")
    b.append(f"      .stride={{{sw},{sh}}}, .padding={{{pl},{pt}}}, .dilation={{1,1}},")
    b.append(f"      .activation={{-128,127}} }};")
    b.append(f"  quant_data_t qd={{ .shift=sft, .mult=mlt }};")
    b.extend(_lut_call(ctx, node, "  esp_nn_depthwise_conv_s8_simd(&id, x, &fd, w, bias, &od, y, &cp, &qd);"))
    b.append("}"); b.append("")
    ctx.calls.append((node, fn))

from .common import fsan, cfloat

def _requant_int(indent):
    return (f"{indent}float sv=(float)acc*mult[k]; int r=(sv>=0)?(int)(sv+0.5f):(int)(sv-0.5f);\n"
            f"{indent}r+=Y_ZP; if(r<-128)r=-128; if(r>127)r=127;\n")


def _emit_depthwise(ctx, node, K,Kh,Kw,H,W,Ho,Wo,sh,sw,pt,pl,XS,XZ,YS,YZ,WSC,bias_name,x,w,y,fn):
    """Depthwise int8 conv: output channel k uses ONLY input channel k.
    Weight layout (OHWI with Cin=1): w[k,kh,kw,0] -> index ((k*Kh+kh)*Kw+kw)*1.
    Requant: per-channel float mult = XS*WSC[k]/YS; adj[k]=bias[k]-x_zp*sum_w[k]."""
    _mi=[]; _si=[]
    for c in range(K):
        wsc = WSC[c] if (WSC and c<len(WSC)) else (WSC[0] if WSC else 1.0)
        m,sft=_tflm_mult_shift(XS*wsc/YS); _mi.append(m); _si.append(sft)
    mi_str=", ".join(str(v) for v in _mi); si_str=", ".join(str(v) for v in _si)
    h=ctx.header; b=ctx.body
    sig=f"void {fn}(const int8_t* x, const int8_t* w, const int32_t* bias, int8_t* y)"
    h.append(sig+";")
    b.append(f"// Depthwise QLinearConv {node['name']}  K={K} {Kh}x{Kw} s{sh} (per-channel, int requant)")
    b.append(sig+" {")
    b.append(f"  const int H={H},W={W},K={K},Kh={Kh},Kw={Kw},Ho={Ho},Wo={Wo};")
    b.append(f"  const int sh={sh},sw={sw},pt={pt},pl={pl}; const int X_ZP={XZ},Y_ZP={YZ};")
    b.append(f"  static const int32_t mlt[{K}]={{{mi_str}}};")
    b.append(f"  static const int32_t sft[{K}]={{{si_str}}};")
    b.extend(_lut_decl(ctx, node, "  "))
    b.append(f"  for(int k=0;k<K;++k){{")
    b.append(f"    long sw_=0; for(int kh=0;kh<Kh;++kh)for(int kw=0;kw<Kw;++kw) sw_+=w[(kh*Kw+kw)*K+k];")
    b.append(f"    int adj=(bias?bias[k]:0) - X_ZP*(int)sw_;")
    b.append(f"    for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow){{")
    b.append(f"      int acc=adj;")
    b.append(f"      for(int kh=0;kh<Kh;++kh){{int hi=oh*sh-pt+kh;")
    b.append(f"        for(int kw=0;kw<Kw;++kw){{int wi=ow*sw-pl+kw;")
    b.append(f"          signed char wv=w[(kh*Kw+kw)*K+k];")
    b.append(f"          if(hi<0||hi>=H||wi<0||wi>=W){{ acc+=X_ZP*(int)wv; continue; }}")
    b.append(f"          acc += (int)x[(hi*W+wi)*K+k]*(int)wv;")
    b.append(f"        }}")
    b.append(f"      }}")
    b.append(f"      int r=_tflm_requant(acc, mlt[k], sft[k]) + Y_ZP;")
    b.append(f"      if(r<-128)r=-128; if(r>127)r=127;")
    b.append(f"      y[(oh*Wo+ow)*K+k]=" + ("silu_lut[r+128];" if _silu_lut(ctx, node) else "(signed char)r;"))
    b.append(f"    }}")
    b.append(f"  }}")
    b.append("}"); b.append("")
    ctx.calls.append((node, fn))

def _silu_lut(ctx, node):
    """The int8 table of a SiLU fuse.py folded into this conv, or None."""
    s = node["attrs"].get("silu")
    if not s:
        return None
    from .op_silu import silu_i8_table
    is_ = ctx.val0(s["in_scale"]); os_ = ctx.val0(s["out_scale"])
    if is_ is None or os_ is None:
        raise ValueError(f"QLinearConv {node['name']}: its folded SiLU's scales are not inlined")
    return silu_i8_table(is_, int(ctx.val0(s["in_zp"])), os_, int(ctx.val0(s["out_zp"])))

def _lut_decl(ctx, node, indent):
    lut = _silu_lut(ctx, node)
    if not lut:
        return []
    return [f"{indent}static const int8_t silu_lut[256]={{{', '.join(str(v) for v in lut)}}};   // its SiLU, folded in"]

def _lut_call(ctx, node, call):
    """The kernel call, with the folded SiLU's table set around it if there is one."""
    if not _silu_lut(ctx, node):
        return [call]
    return _lut_decl(ctx, node, "  ") + ["  esp_nn_conv_s8_simd_set_out_lut(silu_lut);", call,
                                         "  esp_nn_conv_s8_simd_set_out_lut(0);"]

_REQUANT_HELPER = """
// --- TFLite integer requant (matches esp-nn / onnxruntime) ---
// value = SaturatingRoundingDoublingHighMul(acc, mult) then rounding-shift by `shift`.
#ifndef _TFLM_REQUANT_DEFINED
#define _TFLM_REQUANT_DEFINED
static inline int _tflm_srdhm(int a, int b){
    long long ab=(long long)a*(long long)b;
    long long nudge = (ab>=0)? (1<<30) : (1-(1<<30));
    int result=(int)((ab+nudge)/(1LL<<31));
    return result;
}
static inline int _tflm_rounding_shift(int x, int shift){
    // shift is the number of right-shifts (>=0). Rounding to nearest, half away from zero.
    if(shift<=0) return x << (-shift);
    int rem_mask=(1<<shift)-1;
    int rem = x & rem_mask;
    int threshold = (rem_mask>>1) + ((x<0)?1:0);
    int res = (x>>shift) + ((rem>threshold)?1:0);
    return res;
}
static inline int _tflm_requant(int acc, int mult, int shift){
    // Our _tflm_mult_shift returns (mult,shift) with M = mult * 2^(shift-31).
    // Then acc*M = srdhm(acc,mult) * 2^shift.  srdhm already does the /2^31.
    // shift is typically <=0 (M<1), so we RIGHT-shift by (-shift).
    int high = _tflm_srdhm(acc, mult);
    return _tflm_rounding_shift(high, -shift);
}
#endif
"""

@register("QLinearConv")
def emit(ctx, node):
    if not getattr(ctx,"_requant_helper_done",False):
        ctx.header.append(_REQUANT_HELPER)
        ctx._requant_helper_done=True
    ins=node["inputs"]; y=node["outputs"][0]; a=node["attrs"]
    x, x_s, x_zp = ins[0], ins[1], ins[2]
    w, w_s, w_zp = ins[3], ins[4], ins[5]
    y_s, y_zp    = ins[6], ins[7]
    bias         = ins[8] if len(ins)>8 else None

    xs=ctx.shape(x); ws=ctx.shape(w); yshp=ctx.shape(y)
    if len(xs)!=4 or len(ws)!=4 or len(yshp)!=4:
        raise ValueError(f"QLinearConv {node['name']}: needs 4-D NHWC shapes but got "
                         f"x={xs} w={ws} y={yshp}. Run ONNX shape inference so these resolve "
                         f"(e.g. onnx.shape_inference.infer_shapes) before compiling.")
    _,H,W,Cp   = xs
    K,Kh,Kw,wCin = ws
    _,Ho,Wo,_  = yshp
    sh,sw = a.get("strides",[1,1]); pads=a.get("pads",[0,0,0,0]); pt,pl=pads[0],pads[1]
    group = a.get("group",1)
    # Depthwise: group==Cp (each output channel from one input channel). Weight last dim==1.
    is_depthwise = (group>1 and group==Cp and K==Cp and wCin==1)
    if group>1 and not is_depthwise:
        raise ValueError(f"QLinearConv {node['name']}: grouped conv group={group} "
                         f"(Cp={Cp},K={K},wCin={wCin}) not supported (only group=1 or depthwise).")

    # inlined quant constants
    XS=ctx.val0(x_s); XZ=int(ctx.val0(x_zp)); YS=ctx.val0(y_s); YZ=int(ctx.val0(y_zp))
    WSC=ctx.vals(w_s)                          # per-channel weight scales (len K)
    fn=fsan(node["name"])

    # ---- esp-nn SIMD library routing ----
    # When enabled, hand this conv to the validated esp_nn_conv_s8_simd library
    # (NHWC in/out, OHWI filter -- matches our layout). Library folds zero-point
    # internally, so we pass RAW bias and per-channel (mult,shift), NOT adj[]/float mult.
    dil = a.get("dilations",[1,1])
    # Under --simd on esp32s3: hand the conv to the validated esp_nn_conv_s8_simd
    # library instead of emitting inline assembly. Library needs dilation==1 and
    # filter_ch==input_ch (non-grouped); it internally picks 1x1/im2col/c_ref.
    if is_depthwise:
        if ctx.target=="esp32s3" and ctx.simd and dil==[1,1]:
            _emit_dw_espnn_call(ctx, node, K,Kh,Kw,H,W,Ho,Wo,sh,sw,pt,pl,XS,XZ,YS,YZ,WSC,
                                (bias if bias else "0"), x,w,y,fn)
        else:
            _emit_depthwise(ctx, node, K,Kh,Kw,H,W,Ho,Wo,sh,sw,pt,pl,XS,XZ,YS,YZ,WSC,
                            (bias if bias else "0"), x,w,y,fn)
        return
    espnn_ok = (ctx.target=="esp32s3" and ctx.simd and dil==[1,1] and group==1)
    if espnn_ok:
        bias_arg = bias if bias else "0"
        _emit_espnn_call(ctx, node, K,Kh,Kw,Cp,H,W,Ho,Wo,sh,sw,pt,pl,
                         XS,XZ,YS,YZ,WSC,bias_arg,x,w,y,fn)
        return

    # emit compile-time mult[] (needs sum_w for adj -> computed at RUNTIME from w,bias,
    # because weights are file-backed and not inlined; but mult[] IS compile-time).
    mult_vals=", ".join(cfloat(XS*ws/YS) for ws in WSC) if WSC else ""
    _mi=[]; _si=[]
    for c in range(K):
        wsc = WSC[c] if (WSC and c<len(WSC)) else (WSC[0] if WSC else 1.0)
        m,sft=_tflm_mult_shift(XS*wsc/YS); _mi.append(m); _si.append(sft)
    mi_str=", ".join(str(v) for v in _mi); si_str=", ".join(str(v) for v in _si)

    h=ctx.header; b=ctx.body
    sigd=f"void {fn}(const int8_t* x, const int8_t* w, const int32_t* bias, int8_t* y)"
    h.append(sigd+";")
    b.append(f"// QLinearConv {node['name']}  target={ctx.target}  K={K} Cp={Cp} {Kh}x{Kw} s{sh}")
    b.append(sigd+" {")
    b.append(f"  const int H={H},W={W},Cp={Cp},K={K},Kh={Kh},Kw={Kw},Ho={Ho},Wo={Wo};")
    b.append(f"  const int sh={sh},sw={sw},pt={pt},pl={pl},X_ZP={XZ},Y_ZP={YZ};")
    b.append(f"  static const float mult[{K}]={{{mult_vals}}};  // (float, ansi_float only)")
    b.append(f"  static const int32_t mlt[{K}]={{{mi_str}}};")
    b.append(f"  static const int32_t sft[{K}]={{{si_str}}};")
    # adj[k]=bias[k]-x_zp*sum_w[k], computed once from file-backed weights:
    b.append(f"  static int adj[{K}]; static int prep=0;")
    b.append(f"  if(!prep){{ for(int k=0;k<K;++k){{ int s=0;")
    b.append(f"    for(int t=0;t<Kh*Kw*Cp;++t) s+=w[k*Kh*Kw*Cp+t];")
    b.append(f"    adj[k]=(bias?bias[k]:0)-X_ZP*s; }} prep=1; }}")

    if ctx.target=="ansi_float":
        body=_body_ansi_float(XS,XZ,YS,YZ)
    else:
        # scalar int8: correct and alignment-free. Used for esp32s3 when Cp%16!=0,
        # for esp32s3 without --simd, and for ansi_int8.
        body=_body_ansi_int()
    if _silu_lut(ctx, node):
        # the SiLU folded in (fuse.py fold_silu_into_conv): each byte through its table
        b.extend(_lut_decl(ctx, node, "  "))
        store="y[(oh*Wo+ow)*K+k]=(signed char)r;"
        assert body.count(store)==1
        body=body.replace(store, "y[(oh*Wo+ow)*K+k]=silu_lut[r+128];")
    b.append(body)
    b.append("}"); b.append("")

    # driver call
    bn=bias if bias else "0"
    ctx.calls.append((node, fn))

def _body_esp32s3():
    # ACCX SIMD kernel. Safer version: declares q0/q1 and ACCX as clobbered so the
    # compiler doesn't cache live values in them across the asm (the previous version
    # omitted these clobbers, which corrupted results). Each (kh,kw) tap computes an
    # independent 16-lane dot product; taps are summed in `acc`.
    return r"""  for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow)for(int k=0;k<K;++k){
    int acc=adj[k];
    for(int kh=0;kh<Kh;++kh){int hi=oh*sh-pt+kh;
      for(int kw=0;kw<Kw;++kw){int wi=ow*sw-pl+kw;
        const signed char* wp=&w[((k*Kh+kh)*Kw+kw)*Cp];
        int pad=(hi<0||hi>=H||wi<0||wi>=W);
        if(pad){ for(int c=0;c<Cp;++c) acc+=X_ZP*(int)wp[c]; continue; }
        const signed char* xp=&x[(hi*W+wi)*Cp];
        int t16=Cp/16;
        if(t16>0){
          int out=0; const signed char* xx=xp; const signed char* ww=wp; int n=t16;
          __asm__ volatile(
            "ee.zero.accx\n"
            "2:\n"
            "  ee.vld.128.ip q0,%[a],16\n"
            "  ee.vld.128.ip q1,%[b],16\n"
            "  ee.vmulas.s8.accx q0,q1\n"
            "  addi.n %[n],%[n],-1\n"
            "  bnez %[n],2b\n"
            "  ee.srs.accx %[o],%[s],0\n"
            : [o]"=&r"(out), [a]"+&r"(xx), [b]"+&r"(ww), [n]"+&r"(n)
            : [s]"r"(0)
            : "memory");
          acc+=out;
        }
        for(int c=t16*16;c<Cp;++c) acc+=(int)xp[c]*(int)wp[c];
      }}
    int r=_tflm_requant(acc, mlt[k], sft[k]) + Y_ZP;
    if(r<-128)r=-128; if(r>127)r=127;
    y[(oh*Wo+ow)*K+k]=(signed char)r;
  }"""

def _body_ansi_int():
    return r"""  for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow)for(int k=0;k<K;++k){
    int acc=adj[k];
    for(int kh=0;kh<Kh;++kh){int hi=oh*sh-pt+kh;
      for(int kw=0;kw<Kw;++kw){int wi=ow*sw-pl+kw;
        const signed char* wp=&w[((k*Kh+kh)*Kw+kw)*Cp];
        if(hi<0||hi>=H||wi<0||wi>=W){ // padded tap == x_zp
          for(int c=0;c<Cp;++c) acc+=X_ZP*(int)wp[c];
          continue; }
        for(int c=0;c<Cp;++c)
          acc+=(int)x[(hi*W+wi)*Cp+c]*(int)wp[c];
      }}
    int r=_tflm_requant(acc, mlt[k], sft[k]) + Y_ZP;
    if(r<-128)r=-128; if(r>127)r=127;
    y[(oh*Wo+ow)*K+k]=(signed char)r;
  }"""

def _body_ansi_float(XS,XZ,YS,YZ):
    return f"""  const float XS={cfloat(XS)}, YS={cfloat(YS)}; const int XZ={XZ}, YZ={YZ};
  for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow)for(int k=0;k<K;++k){{
    // float reference: dequantize, MAC in float, requantize
    float accf=0.0f;
    for(int kh=0;kh<Kh;++kh){{int hi=oh*sh-pt+kh; if(hi<0||hi>=H)continue;
      for(int kw=0;kw<Kw;++kw){{int wi=ow*sw-pl+kw; if(wi<0||wi>=W)continue;
        for(int c=0;c<Cp;++c){{
          float xf=XS*(float)((int)x[(hi*W+wi)*Cp+c]-XZ);
          float wf=mult[k]*YS/XS*(float)w[((k*Kh+kh)*Kw+kw)*Cp+c]; // w_scale[k]*w
          accf+=xf*wf;
        }}
      }}}}
    if(bias) accf+=(float)bias[k]*mult[k]*YS;  // the int32 bias is at scale XS*w_scale[k]
    int r=(int)lroundf(accf/YS)+YZ; if(r<-128)r=-128; if(r>127)r=127;
    y[(oh*Wo+ow)*K+k]=(signed char)r;
  }}"""
