/*
 * esp_nn_depthwise_conv_s8_simd.c
 *
 * Self-contained s8 NHWC DEPTHWISE convolution for ESP32-S3
 * (with plain-C fallback). See header for documentation.
 *
 * SIMD strategy (ch_mult == 1):
 *   Depthwise conv is elementwise across channels: every channel has its
 *   own small filter and never mixes with other channels. That maps
 *   perfectly onto the S3's 8-lane int16 MAC:
 *
 *   1. One-time per call:
 *      - Convert the whole input to int16 as (value + input_offset), into
 *        a scratch buffer that already includes the spatial padding
 *        (pad cells stored as 0, since pad contributes exactly 0 after
 *        the offset is folded in). Channels padded up to a multiple of 8.
 *      - Convert the filter to int16 with the same channel padding
 *        (pad lanes = 0).
 *   2. Hot loop, per output pixel, per group of 8 channels:
 *        ee.zero.qacc
 *        for each filter tap:  q0 = 8x input16, q1 = 8x filter16
 *                              ee.vmulas.s16.qacc q0, q1   (8 parallel MACs)
 *        extract 8 accumulators from QACC, requantize, store.
 *      No edge checks anywhere in the hot loop.
 *
 *   All int16 buffers are 16-byte aligned by construction (channel count
 *   padded to 8 -> every row/tap offset is a multiple of 16 bytes).
 *
 * SPDX-License-Identifier: Apache-2.0
 * Derived from Espressif esp-nn (Apache-2.0).
 */

#include "esp_nn_depthwise_conv_s8_simd.h"
#include <string.h>
#include <stdlib.h>

#if defined(__has_include)
#  if __has_include("sdkconfig.h")
#    include "sdkconfig.h"
#  endif
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3) && defined(__XTENSA__)
#  define ESP_NN_S3_SIMD 1
#else
#  define ESP_NN_S3_SIMD 0
#endif

/* The rows path's loops, built off the S3 with plain-C stand-ins for the vector
 * instructions (QACC's lanes stored byte for byte as the S3 stores them), so a PC
 * can check them bit-exact against esp_nn_depthwise_conv_s8_ref(). Test builds only
 * (tools/test_dw_kernels.c); never set for a real target. */
#if ESP_NN_S3_SIMD || defined(ESP_NN_S3_EMULATE)
#  define ESP_NN_S3_PATHS 1
#else
#  define ESP_NN_S3_PATHS 0
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

/* ---- TFLM requantization (identical to esp_nn_conv_s8_simd.c) ------ */
static inline int32_t dw_sat_doubling_high_mul(int32_t a, int32_t b)
{
    int64_t result = (int64_t)a * (int64_t)b;
    result += (int64_t)1 << 30;
    result >>= 31;
    if (result > INT32_MAX) result = INT32_MAX;
    if (result < INT32_MIN) result = INT32_MIN;
    return (int32_t)result;
}

static inline int32_t dw_rounding_divide_by_pot(int32_t x, int32_t exponent)
{
    if (exponent <= 0) return x;
    const int32_t mask = ((int32_t)1 << exponent) - 1;
    const int32_t remainder = x & mask;
    int32_t result = x >> exponent;
    int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    if (remainder > threshold) result += 1;
    return result;
}

static inline int32_t dw_requantize(int32_t val, int32_t mult, int32_t shift)
{
    const int32_t left_shift  = shift > 0 ? shift : 0;
    const int32_t right_shift = shift > 0 ? 0 : -shift;
    return dw_rounding_divide_by_pot(
        dw_sat_doubling_high_mul(val * ((int32_t)1 << left_shift), mult),
        right_shift);
}

/* ------------------------------------------------------------------ */
/* Plain-C reference                                                    */
/* ------------------------------------------------------------------ */
void esp_nn_depthwise_conv_s8_ref(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const dw_conv_params_t *conv_params,
                                  const quant_data_t *quant_data)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    const int32_t stride_wd = conv_params->stride.width;
    const int32_t stride_ht = conv_params->stride.height;
    const int32_t filter_wd = filter_dims->width;
    const int32_t filter_ht = filter_dims->height;
    const int32_t out_wd = output_dims->width;
    const int32_t out_ht = output_dims->height;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;
    const int32_t ch_mult = conv_params->ch_mult;

    int32_t out_idx = 0;
    for (int32_t out_y = 0; out_y < out_ht; out_y++) {
        const int32_t base_y = (out_y * stride_ht) - pad_ht;
        for (int32_t out_x = 0; out_x < out_wd; out_x++) {
            const int32_t base_x = (out_x * stride_wd) - pad_wd;
            for (int32_t ch_idx = 0; ch_idx < channels; ch_idx++) {
                for (int32_t m = 0; m < ch_mult; m++) {
                    int32_t result = 0;
                    const int32_t out_ch_idx = m + ch_idx * ch_mult;

                    const int32_t fy_start = max(0, -base_y);
                    const int32_t fx_start = max(0, -base_x);
                    const int32_t fy_end = min(filter_ht, input_ht - base_y);
                    const int32_t fx_end = min(filter_wd, input_wd - base_x);

                    for (int32_t fy = fy_start; fy < fy_end; fy++) {
                        const int32_t iy = base_y + fy;
                        for (int32_t fx = fx_start; fx < fx_end; fx++) {
                            const int32_t ix = base_x + fx;
                            const int32_t in_idx =
                                (iy * input_wd + ix) * channels + ch_idx;
                            const int32_t f_idx =
                                (fy * filter_wd + fx) * (channels * ch_mult)
                                + out_ch_idx;
                            result += (input_data[in_idx] + input_offset)
                                      * filter_data[f_idx];
                        }
                    }
                    if (bias) result += bias[out_ch_idx];
                    result = dw_requantize(result, out_mult[out_ch_idx],
                                           out_shift[out_ch_idx]);
                    result += out_offset;
                    result = max(result, activation_min);
                    result = min(result, activation_max);
                    out_data[out_idx++] = (int8_t)result;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scratch management                                                  */
/* ------------------------------------------------------------------ */
static void *s_user_scratch = NULL;
static int   s_user_scratch_size = 0;
static void *s_auto_scratch = NULL;
static int   s_auto_scratch_size = 0;
static const char *s_last_path = "none";

void esp_nn_depthwise_conv_s8_simd_set_scratch_buf(void *buf, int size_bytes)
{
    s_user_scratch = buf;
    s_user_scratch_size = buf ? size_bytes : 0;
}

const char *esp_nn_depthwise_conv_s8_simd_last_path(void)
{
    return s_last_path;
}

__attribute__((unused))
static void *dw_get_scratch(int need)
{
    if (need <= 0) return (void *)"";
    if (s_user_scratch) {
        return (s_user_scratch_size >= need) ? s_user_scratch : NULL;
    }
    if (s_auto_scratch_size < need) {
        void *p = realloc(s_auto_scratch, need);
        if (!p) return NULL;
        s_auto_scratch = p;
        s_auto_scratch_size = need;
    }
    return s_auto_scratch;
}

int esp_nn_depthwise_conv_s8_simd_get_scratch_size(const data_dims_t *input_dims,
                                                   const data_dims_t *filter_dims,
                                                   const data_dims_t *output_dims,
                                                   const dw_conv_params_t *conv_params)
{
#if !ESP_NN_S3_SIMD
    (void)input_dims; (void)filter_dims; (void)output_dims; (void)conv_params;
    return 0;
#else
    (void)output_dims;
    if (conv_params->ch_mult != 1) return 0; /* falls back to C */

    const int ch_p = (input_dims->channels + 7) & ~7;
    const int padded_wd = input_dims->width + 2 * conv_params->padding.width;
    const int padded_ht = input_dims->height + 2 * conv_params->padding.height;

    const int input16  = padded_wd * padded_ht * ch_p * 2;
    const int filter16 = filter_dims->width * filter_dims->height * ch_p * 2;
    return input16 + filter16 + 64;
#endif
}

/* ================================================================== */
/* ESP32-S3 SIMD path (ch_mult == 1)                                   */
/* ================================================================== */
#if ESP_NN_S3_SIMD

/* MAC one filter tap: 8 int16 inputs x 8 int16 filter lanes -> QACC.
 * Both pointers must be 16-byte aligned. */
static inline void dw_mac8_tap(const int16_t *in16, const int16_t *f16)
{
    __asm__ volatile(
        "ee.vld.128.ip q0, %[d], 0\n"
        "ee.vld.128.ip q1, %[f], 0\n"
        "ee.vmulas.s16.qacc q0, q1\n"
        : : [d] "r" (in16), [f] "r" (f16) : "memory");
}

/* Extract QACC (s16 mode: 8 lanes, 40-bit each, packed 5 bytes/lane).
 * Low 32 bits of each lane are enough for s8 depthwise accumulations. */
static inline void dw_qacc_extract8(int32_t *acc)
{
    int8_t __attribute__((aligned(16))) qraw[24];
    int8_t *qp = qraw;
    __asm__ volatile(
        "ee.st.qacc_l.l.128.ip %[p], 16\n"
        "ee.st.qacc_l.h.32.ip  %[p], -16\n"
        : [p] "+r" (qp) : : "memory");
    acc[0] = *(int32_t *)(qraw + 0);
    acc[1] = *(int32_t *)(qraw + 5);
    acc[2] = *(int32_t *)(qraw + 10);
    acc[3] = *(int32_t *)(qraw + 15);
    qp = qraw;
    __asm__ volatile(
        "ee.st.qacc_h.l.128.ip %[p], 16\n"
        "ee.st.qacc_h.h.32.ip  %[p], -16\n"
        : [p] "+r" (qp) : : "memory");
    acc[4] = *(int32_t *)(qraw + 0);
    acc[5] = *(int32_t *)(qraw + 5);
    acc[6] = *(int32_t *)(qraw + 10);
    acc[7] = *(int32_t *)(qraw + 15);
}

static void dw_conv_s8_mult1_s3(const data_dims_t *input_dims,
                                const int8_t *input_data,
                                const data_dims_t *filter_dims,
                                const int8_t *filter_data,
                                const int32_t *bias,
                                const data_dims_t *output_dims,
                                int8_t *out_data,
                                const dw_conv_params_t *conv_params,
                                const quant_data_t *quant_data,
                                void *scratch)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t channels = input_dims->channels;
    const int32_t filter_wd = filter_dims->width;
    const int32_t filter_ht = filter_dims->height;
    const int32_t out_wd = output_dims->width;
    const int32_t out_ht = output_dims->height;
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    const int32_t stride_wd = conv_params->stride.width;
    const int32_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;
    const int32_t *out_mult = quant_data->mult;
    const int32_t *out_shift = quant_data->shift;

    const int32_t ch_p = (channels + 7) & ~7;        /* channels padded to 8 */
    const int32_t padded_wd = input_wd + 2 * pad_wd;
    const int32_t padded_ht = input_ht + 2 * pad_ht;

    /* Scratch: [input16 (padded) | filter16], both 16-byte aligned. */
    int16_t *input16 = (int16_t *)((uintptr_t)((int8_t *)scratch + 15) & ~15);
    int16_t *filter16 = input16 + padded_wd * padded_ht * ch_p;
    /* ch_p multiple of 8 -> filter16 offset is a multiple of 16 bytes */

    /* --- Phase 1: convert input to int16 with offset + spatial padding.
     * Padding cells and padded channel lanes are 0, which contributes
     * exactly 0 to the MAC (offset already folded into real cells). --- */
    memset(input16, 0, padded_wd * padded_ht * ch_p * sizeof(int16_t));
    for (int32_t y = 0; y < input_ht; y++) {
        const int8_t *src = input_data + y * input_wd * channels;
        int16_t *dst = input16 + ((y + pad_ht) * padded_wd + pad_wd) * ch_p;
        for (int32_t x = 0; x < input_wd; x++) {
            for (int32_t c = 0; c < channels; c++) {
                dst[c] = (int16_t)(src[c] + input_offset);
            }
            src += channels;
            dst += ch_p;
        }
    }

    /* --- Phase 2: convert filter to int16, padded lanes = 0. --- */
    memset(filter16, 0, filter_wd * filter_ht * ch_p * sizeof(int16_t));
    for (int32_t t = 0; t < filter_wd * filter_ht; t++) {
        const int8_t *src = filter_data + t * channels; /* ch_mult == 1 */
        int16_t *dst = filter16 + t * ch_p;
        for (int32_t c = 0; c < channels; c++) {
            dst[c] = (int16_t)src[c];
        }
    }

    /* --- Phase 3: hot loop, no edge checks. --- */
    const int32_t in_row_elems = padded_wd * ch_p;

    for (int32_t out_y = 0; out_y < out_ht; out_y++) {
        const int16_t *row_base = input16 + (out_y * stride_ht) * in_row_elems;
        for (int32_t out_x = 0; out_x < out_wd; out_x++) {
            const int16_t *pix_base = row_base + (out_x * stride_wd) * ch_p;
            int8_t *out_pix = out_data + (out_y * out_wd + out_x) * channels;

            for (int32_t cg = 0; cg < ch_p; cg += 8) {
                __asm__ volatile("ee.zero.qacc");

                const int16_t *f_tap = filter16 + cg;
                for (int32_t fy = 0; fy < filter_ht; fy++) {
                    const int16_t *in_tap = pix_base + fy * in_row_elems + cg;
                    for (int32_t fx = 0; fx < filter_wd; fx++) {
                        dw_mac8_tap(in_tap, f_tap);
                        in_tap += ch_p;
                        f_tap += ch_p;
                    }
                }

                int32_t acc[8];
                dw_qacc_extract8(acc);

                const int32_t lanes = min(8, channels - cg);
                for (int32_t l = 0; l < lanes; l++) {
                    const int32_t oc = cg + l;
                    int32_t r = acc[l];
                    if (bias) r += bias[oc];
                    r = dw_requantize(r, out_mult[oc], out_shift[oc]);
                    r += out_offset;
                    r = max(r, activation_min);
                    r = min(r, activation_max);
                    out_pix[oc] = (int8_t)r;
                }
            }
        }
    }
}

#endif /* ESP_NN_S3_SIMD */

/* ================================================================== */
/* The rows path (ch_mult == 1): a few rows of int16 in the fast buffer  */
/* ================================================================== */
/*
 * Why (October 2026, measured on the board): the path above converts the WHOLE
 * padded input to int16 first -- 674 KB for a 160x128x16 layer -- so either the
 * allocation fails and the plain-C reference runs, or the hot loop streams
 * PSRAM; and it reads QACC's lanes back with unaligned 32-bit loads. Every
 * depthwise layer of YOLO26n and of a YuNet face finder ran at about 11 million
 * MACs a second, slower than the regular convs beside them.
 *
 * Here each output row needs only the k_h input rows under it, in int16 with
 * the padding in place, and those rows sit in the fast buffer the convs already
 * use (internal SRAM; esp_nn_conv_s8_simd_borrow_fast_buf()). A row is converted
 * once and kept while later output rows still read it (a ring of k_h rows). If
 * the rows of every channel do not fit, the channels go in passes of a multiple
 * of 8. QACC is stored and read back a byte at a time, which needs no alignment,
 * and the requant is the regular convs' requant_fast(). With
 * esp_nn_conv_s8_simd_set_cores(2) the other core takes half the output rows,
 * with its own ring.
 *
 * The same integer arithmetic as the reference, each output accumulated in full
 * before it is requantized: bit-exact (tools/test_dw_kernels.c on a PC; and on
 * the board, a self-check against the reference before the first use, which
 * falls back to the old paths if it ever disagrees).
 */
#if ESP_NN_S3_PATHS

#define DW_MAX_K 16                        /* filter rows the ring can hold */
#define DW_PER_CH 4                        /* bias, mult, left shift, right shift */
#define DW_ALIGN16(p) ((void *)(((uintptr_t)(p) + 15) & ~(uintptr_t)15))

#if ESP_NN_S3_SIMD
static inline __attribute__((always_inline)) void rw_zero(void)
{
    __asm__ volatile("ee.zero.qacc" ::: "memory");
}

/* 8 int16 x 8 int16 into QACC's 8 lanes; both 16-byte aligned. */
static inline __attribute__((always_inline)) void rw_mac8(const int16_t *in16, const int16_t *f16)
{
    __asm__ volatile(
        "ee.vld.128.ip q0, %[d], 0\n"
        "ee.vld.128.ip q1, %[f], 0\n"
        "ee.vmulas.s16.qacc q0, q1\n"
        : : [d] "r" (in16), [f] "r" (f16) : "memory");
}

/* QACC to raw[0..19] (lanes 0-3) and raw[32..51] (lanes 4-7), 5 bytes a lane. */
static inline __attribute__((always_inline)) void rw_store(uint8_t *raw)
{
    uint8_t *p = raw;
    __asm__ volatile(
        "ee.st.qacc_l.l.128.ip %[p], 16\n"
        "ee.st.qacc_l.h.32.ip  %[p], -16\n"
        : [p] "+r" (p) : : "memory");
    p = raw + 32;
    __asm__ volatile(
        "ee.st.qacc_h.l.128.ip %[p], 16\n"
        "ee.st.qacc_h.h.32.ip  %[p], -16\n"
        : [p] "+r" (p) : : "memory");
}
#else
/* The PC test: QACC's 8 lanes of 40 bits, and the bytes the S3 stores them as. */
static int64_t s_qacc[8];

static inline void rw_zero(void)
{
    for (int l = 0; l < 8; l++) s_qacc[l] = 0;
}

static inline void rw_mac8(const int16_t *in16, const int16_t *f16)
{
    for (int l = 0; l < 8; l++) s_qacc[l] += (int32_t)in16[l] * f16[l];
}

static inline void rw_store(uint8_t *raw)
{
    for (int l = 0; l < 8; l++) {
        uint8_t *p = raw + (l < 4 ? 5 * l : 32 + 5 * (l - 4));
        const uint64_t v = (uint64_t)s_qacc[l];
        for (int b = 0; b < 5; b++) p[b] = (uint8_t)(v >> (8 * b));
    }
}
#endif

/* A lane's low 32 bits, a byte at a time (lanes 1-3 are not 4-byte aligned). */
static inline __attribute__((always_inline)) int32_t rw_lane(const uint8_t *p)
{
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* dw_requantize(), with the shift split into its left and right parts: the same
 * arithmetic in the same order (as esp_nn_conv_s8_simd.c's requant_fast()). */
static inline __attribute__((always_inline))
int32_t rw_requant(int32_t val, int32_t mult, int32_t left, int32_t right)
{
    const int32_t a = val * ((int32_t)1 << left);
    int32_t x;
    if (__builtin_expect(a == INT32_MIN && mult == INT32_MIN, 0)) {
        x = INT32_MAX;
    } else {
        x = (int32_t)(((int64_t)a * mult + ((int64_t)1 << 30)) >> 31);
    }
    if (right) {
        const int32_t mask = ((int32_t)1 << right) - 1;
        const int32_t threshold = (mask >> 1) + (x < 0);
        x = (x >> right) + ((x & mask) > threshold);
    }
    return x;
}

typedef struct {
    const int8_t *input;
    int8_t *out;
    const int16_t *filter16;      /* [k_h * k_w][ch_p], 16-byte aligned, pad lanes 0 */
    const int32_t *per_ch;        /* ch_p x DW_PER_CH */
    int16_t *ring;                /* this core's k_h rows of row_w x pass int16 */
    int32_t in_w, in_h, ch, ch_p, k_w, k_h, stride_w, stride_h, pad_w, pad_h;
    int32_t out_w, row_w, pass;   /* row_w: columns a row holds; pass: channels at a time */
    int32_t in_offset, out_offset, act_min, act_max;
    int32_t y0, y1;
} dw_rows_job_t;

/* Input row iy, channels [c0, c0 + cn), as int16 (value + in_offset) into d:
 * row_w columns, the padding and the pad lanes 0. */
static void dw_fill_row(const dw_rows_job_t *j, int16_t *d, int32_t iy, int32_t c0, int32_t cn)
{
    if (iy < 0 || iy >= j->in_h) {
        memset(d, 0, (size_t)j->row_w * cn * sizeof(int16_t));
        return;
    }
    const int32_t real = min(cn, j->ch - c0);
    const int32_t xa = min(j->pad_w, j->row_w);
    const int32_t xb = min(j->row_w, j->pad_w + j->in_w);
    const int16_t off = (int16_t)j->in_offset;

    memset(d, 0, (size_t)xa * cn * sizeof(int16_t));
    d += xa * cn;
    const int8_t *src = j->input + ((size_t)iy * j->in_w + (xb > xa ? xa - j->pad_w : 0)) * j->ch + c0;
    for (int32_t x = xa; x < xb; x++) {
        int32_t c = 0;
        for (; c < real; c++) d[c] = (int16_t)(src[c] + off);
        for (; c < cn; c++) d[c] = 0;
        src += j->ch;
        d += cn;
    }
    memset(d, 0, (size_t)(j->row_w - xb) * cn * sizeof(int16_t));
}

static void dw_rows(void *arg)
{
    const dw_rows_job_t *j = (const dw_rows_job_t *)arg;
    const int32_t k_w = j->k_w, k_h = j->k_h, ch = j->ch, ch_p = j->ch_p;
    const int32_t out_offset = j->out_offset, act_min = j->act_min, act_max = j->act_max;
    const int16_t *const filter16 = j->filter16;
    const int32_t *const per_ch = j->per_ch;
    uint8_t raw_buf[64 + 16];
    uint8_t *const raw = (uint8_t *)DW_ALIGN16(raw_buf);

    for (int32_t c0 = 0; c0 < ch_p; c0 += j->pass) {
        const int32_t cn = min(j->pass, ch_p - c0);       /* a multiple of 8 */
        const int32_t real = min(cn, ch - c0);
        const size_t slot = (size_t)j->row_w * cn;
        int32_t held[DW_MAX_K];
        const int16_t *rowp[DW_MAX_K];
        for (int32_t s = 0; s < k_h; s++) held[s] = INT32_MIN;

        for (int32_t out_y = j->y0; out_y < j->y1; out_y++) {
            const int32_t base_y = out_y * j->stride_h - j->pad_h;
            for (int32_t fy = 0; fy < k_h; fy++) {
                const int32_t iy = base_y + fy;
                const int32_t s = ((iy % k_h) + k_h) % k_h;   /* k_h rows in a row: k_h slots */
                int16_t *d = j->ring + s * slot;
                if (held[s] != iy) {
                    dw_fill_row(j, d, iy, c0, cn);
                    held[s] = iy;
                }
                rowp[fy] = d;
            }

            int8_t *orow = j->out + (size_t)out_y * j->out_w * ch + c0;
            for (int32_t out_x = 0; out_x < j->out_w; out_x++) {
                const int32_t col = out_x * j->stride_w * cn;
                int8_t *opix = orow + (size_t)out_x * ch;

                for (int32_t cg = 0; cg < cn && cg < real; cg += 8) {
                    rw_zero();
                    const int16_t *f = filter16 + c0 + cg;
                    for (int32_t fy = 0; fy < k_h; fy++) {
                        const int16_t *in = rowp[fy] + col + cg;
                        for (int32_t fx = 0; fx < k_w; fx++) {
                            rw_mac8(in, f);
                            in += cn;
                            f += ch_p;
                        }
                    }
                    rw_store(raw);

                    const int32_t lanes = min(8, real - cg);
                    const int32_t *q = per_ch + (c0 + cg) * DW_PER_CH;
                    for (int32_t l = 0; l < lanes; l++) {
                        const uint8_t *p = raw + (l < 4 ? 5 * l : 32 + 5 * (l - 4));
                        int32_t r = rw_lane(p) + q[0];
                        r = rw_requant(r, q[1], q[2], q[3]) + out_offset;
                        r = max(r, act_min);
                        r = min(r, act_max);
                        opix[cg + l] = (int8_t)r;
                        q += DW_PER_CH;
                    }
                }
            }
        }
    }
}

/* Lays the call out in the fast buffer and runs it; 0 (nothing done) if even one
 * pass of 8 channels does not fit, or the call is not one this path takes. */
static int dw_rows_run(const data_dims_t *input_dims, const int8_t *input_data,
                       const data_dims_t *filter_dims, const int8_t *filter_data,
                       const int32_t *bias, const data_dims_t *output_dims,
                       int8_t *out_data, const dw_conv_params_t *conv_params,
                       const quant_data_t *quant_data, int *cores_used)
{
    dw_rows_job_t j;
    j.in_w = input_dims->width;
    j.in_h = input_dims->height;
    j.ch = input_dims->channels;
    j.ch_p = (j.ch + 7) & ~7;
    j.k_w = filter_dims->width;
    j.k_h = filter_dims->height;
    j.stride_w = conv_params->stride.width;
    j.stride_h = conv_params->stride.height;
    j.pad_w = conv_params->padding.width;
    j.pad_h = conv_params->padding.height;
    j.out_w = output_dims->width;
    j.in_offset = conv_params->in_offset;
    j.out_offset = conv_params->out_offset;
    j.act_min = conv_params->activation.min;
    j.act_max = conv_params->activation.max;
    const int32_t out_ht = output_dims->height;

    /* int16 holds value + in_offset exactly for an in_offset this size, and the
     * ring has DW_MAX_K slots */
    if (conv_params->ch_mult != 1 || j.ch < 1 || j.k_w < 1 || j.k_h < 1 || j.k_h > DW_MAX_K ||
        j.stride_w < 1 || j.stride_h < 1 || j.pad_w < 0 || j.pad_h < 0 ||
        j.out_w < 1 || out_ht < 1 || j.in_offset < -32639 || j.in_offset > 32639) {
        return 0;
    }
    j.row_w = (j.out_w - 1) * j.stride_w + j.k_w;

    int size = 0;
    int8_t *fast = (int8_t *)esp_nn_conv_s8_simd_borrow_fast_buf(&size);
    if (!fast) return 0;

    const int taps = j.k_w * j.k_h;
    int16_t *filter16 = (int16_t *)DW_ALIGN16(fast);
    int32_t *per_ch = (int32_t *)DW_ALIGN16(filter16 + (size_t)taps * j.ch_p);
    int16_t *ring0 = (int16_t *)DW_ALIGN16(per_ch + (size_t)j.ch_p * DW_PER_CH);
    const long used = (long)((int8_t *)ring0 - fast);

    int cores = (out_ht > 1 && esp_nn_conv_s8_simd_cores() > 1) ? 2 : 1;
    int pass = 0;
    for (;;) {
        /* each core's ring: k_h rows of row_w x pass int16, plus 16 to align the next */
        const long per_core = (size - used - 16L * cores) / cores;
        const long per_pass8 = (long)j.k_h * j.row_w * 8 * (long)sizeof(int16_t);
        pass = per_core > 0 ? (int)(per_core / per_pass8) * 8 : 0;
        if (pass > j.ch_p) pass = j.ch_p;
        if (pass >= 8 || cores == 1) break;
        cores = 1;
    }
    if (pass < 8) return 0;
    j.pass = pass;
    const size_t ring_elems = (size_t)j.k_h * j.row_w * pass;
    int16_t *ring1 = (int16_t *)DW_ALIGN16(ring0 + ring_elems);

    /* the filter as int16, pad lanes 0; per channel its bias, multiplier, shifts */
    memset(filter16, 0, (size_t)taps * j.ch_p * sizeof(int16_t));
    for (int t = 0; t < taps; t++) {
        for (int32_t c = 0; c < j.ch; c++) {
            filter16[(size_t)t * j.ch_p + c] = filter_data[(size_t)t * j.ch + c];
        }
    }
    for (int32_t c = 0; c < j.ch_p; c++) {
        int32_t *q = per_ch + c * DW_PER_CH;
        if (c < j.ch) {
            const int32_t shift = quant_data->shift[c];
            q[0] = bias ? bias[c] : 0;
            q[1] = quant_data->mult[c];
            q[2] = shift > 0 ? shift : 0;
            q[3] = shift > 0 ? 0 : -shift;
        } else {
            q[0] = q[1] = q[2] = q[3] = 0;
        }
    }

    j.input = input_data;
    j.out = out_data;
    j.filter16 = filter16;
    j.per_ch = per_ch;

    if (cores > 1) {
        dw_rows_job_t mine = j, theirs = j;
        const int32_t mid = out_ht / 2;
        mine.ring = ring0;   mine.y0 = 0;     mine.y1 = mid;
        theirs.ring = ring1; theirs.y0 = mid; theirs.y1 = out_ht;
        if (esp_nn_simd_both_cores(dw_rows, &mine, &theirs)) {
            *cores_used = 2;
            return 1;
        }
    }
    j.ring = ring0; j.y0 = 0; j.y1 = out_ht;
    dw_rows(&j);
    *cores_used = 1;
    return 1;
}

/* Before the rows path is first used: the same answers as the reference on a
 * few small layers (odd channel counts, stride 2, a 5x5 filter, both cores if
 * set)? If not, it is never used and the old paths run as before. */
static int s_rows_ok = -1;                 /* -1 not checked yet, 0 failed, 1 passed */

static int dw_rows_check(void)
{
    static const int16_t shapes[][7] = {    /* w, h, ch, k, stride, pad, (spare) */
        {9, 7, 20, 3, 1, 1, 0}, {10, 6, 16, 3, 2, 1, 0}, {7, 6, 4, 5, 1, 2, 0},
        {6, 5, 10, 3, 1, 0, 0}, {5, 4, 33, 1, 1, 0, 0},
    };
    uint32_t seed = 0x5eed1234u;
    int ok = 1;

    for (unsigned i = 0; ok && i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        const int w = shapes[i][0], h = shapes[i][1], ch = shapes[i][2];
        const int k = shapes[i][3], st = shapes[i][4], pad = shapes[i][5];
        const int ow = (w + 2 * pad - k) / st + 1, oh = (h + 2 * pad - k) / st + 1;
        const size_t n_in = (size_t)w * h * ch, n_f = (size_t)k * k * ch, n_out = (size_t)ow * oh * ch;
        int8_t *in = (int8_t *)malloc(n_in), *f = (int8_t *)malloc(n_f);
        int8_t *a = (int8_t *)malloc(n_out), *b = (int8_t *)malloc(n_out);
        int32_t *bias = (int32_t *)malloc(ch * 4), *mult = (int32_t *)malloc(ch * 4);
        int32_t *shift = (int32_t *)malloc(ch * 4);
        if (!in || !f || !a || !b || !bias || !mult || !shift) {
            ok = 0;
        } else {
            for (size_t n = 0; n < n_in; n++) { seed = seed * 1103515245u + 12345u; in[n] = (int8_t)(seed >> 16); }
            for (size_t n = 0; n < n_f; n++)  { seed = seed * 1103515245u + 12345u; f[n] = (int8_t)(seed >> 16); }
            for (int c = 0; c < ch; c++) {
                seed = seed * 1103515245u + 12345u;
                bias[c] = (int32_t)((seed >> 8) & 0xffff) - 32768;
                mult[c] = 1073741824 + (int32_t)((seed >> 4) & 0x3fffffff) / 2;
                shift[c] = (c % 5 == 0) ? 1 : -(int32_t)(4 + (seed >> 28) % 6);
            }
            data_dims_t id = { w, h, ch, 1 }, fd = { k, k, ch, 1 }, od = { ow, oh, ch, 1 };
            dw_conv_params_t cp = { .in_offset = 128 - (int)(i * 37), .out_offset = -5 + (int)i,
                                    .ch_mult = 1, .stride = {st, st}, .padding = {pad, pad},
                                    .dilation = {1, 1}, .activation = {-128, 127} };
            quant_data_t qd = { shift, mult };
            int cores_used = 0;
            esp_nn_depthwise_conv_s8_ref(&id, in, &fd, f, bias, &od, a, &cp, &qd);
            memset(b, 0x55, n_out);
            ok = dw_rows_run(&id, in, &fd, f, bias, &od, b, &cp, &qd, &cores_used) &&
                 memcmp(a, b, n_out) == 0;
        }
        free(in); free(f); free(a); free(b); free(bias); free(mult); free(shift);
    }
    return ok;
}

#endif /* ESP_NN_S3_PATHS */

int esp_nn_depthwise_conv_s8_simd_self_check(void)
{
#if ESP_NN_S3_PATHS
    if (s_rows_ok < 0) s_rows_ok = dw_rows_check();
    return s_rows_ok;
#else
    return -1;
#endif
}

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */
int esp_nn_depthwise_conv_s8_simd(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const dw_conv_params_t *conv_params,
                                  const quant_data_t *quant_data)
{
    if (conv_params->dilation.width > 1 || conv_params->dilation.height > 1) {
        return -1;
    }

#if ESP_NN_S3_PATHS
    if (conv_params->ch_mult == 1 && esp_nn_depthwise_conv_s8_simd_self_check() == 1) {
        int cores_used = 0;
        if (dw_rows_run(input_dims, input_data, filter_dims, filter_data, bias,
                        output_dims, out_data, conv_params, quant_data, &cores_used)) {
            s_last_path = cores_used > 1 ? "dw_rows_simd_2core" : "dw_rows_simd";
            return 0;
        }
    }
#endif

#if ESP_NN_S3_SIMD
    if (conv_params->ch_mult == 1) {
        int need = esp_nn_depthwise_conv_s8_simd_get_scratch_size(
                       input_dims, filter_dims, output_dims, conv_params);
        void *scratch = dw_get_scratch(need);
        if (scratch) {
            s_last_path = "dw_s16_simd";
            dw_conv_s8_mult1_s3(input_dims, input_data, filter_dims,
                                filter_data, bias, output_dims, out_data,
                                conv_params, quant_data, scratch);
            return 0;
        }
        /* scratch too small / alloc failed -> fall through to C */
    }
#endif

    s_last_path = "dw_c_ref";
    esp_nn_depthwise_conv_s8_ref(input_dims, input_data, filter_dims,
                                 filter_data, bias, output_dims, out_data,
                                 conv_params, quant_data);
    return 0;
}
