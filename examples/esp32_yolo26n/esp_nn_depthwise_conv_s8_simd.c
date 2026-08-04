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
