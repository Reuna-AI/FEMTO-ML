/*
 * esp_nn_conv_s8_simd.c
 *
 * Self-contained s8 NHWC convolution for ESP32-S3 (with plain-C fallback).
 * See esp_nn_conv_s8_simd.h for full documentation.
 *
 * SPDX-License-Identifier: Apache-2.0
 * Derived from Espressif esp-nn (Apache-2.0).
 */

#include "esp_nn_conv_s8_simd.h"
#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Target detection                                                    */
/* ------------------------------------------------------------------ */
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

/* The S3 paths' loops, built off the S3 with plain-C stand-ins for the vector
 * instructions, so a PC can check them bit-exact against esp_nn_conv_s8_ref(). Test
 * builds only (tools/test_conv_kernels.c); never set for a real target. */
#if ESP_NN_S3_SIMD || defined(ESP_NN_S3_EMULATE)
#  define ESP_NN_S3_PATHS 1
#else
#  define ESP_NN_S3_PATHS 0
#endif

#if ESP_NN_S3_SIMD
#  include "esp_heap_caps.h"
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

/* ------------------------------------------------------------------ */
/* TFLM-compatible requantization                                      */
/* ------------------------------------------------------------------ */
static inline int32_t nn_saturating_doubling_high_mul(int32_t a, int32_t b)
{
    int64_t result = (int64_t)a * (int64_t)b;
    result += (int64_t)1 << 30;             /* rounding */
    result >>= 31;
    if (result > INT32_MAX) result = INT32_MAX;
    if (result < INT32_MIN) result = INT32_MIN;
    return (int32_t)result;
}

static inline int32_t nn_rounding_divide_by_pot(int32_t x, int32_t exponent)
{
    if (exponent <= 0) return x;
    const int32_t mask = ((int32_t)1 << exponent) - 1;
    const int32_t remainder = x & mask;
    int32_t result = x >> exponent;
    int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    if (remainder > threshold) result += 1;
    return result;
}

static inline int32_t nn_requantize(int32_t val, int32_t mult, int32_t shift)
{
    const int32_t left_shift  = shift > 0 ? shift : 0;
    const int32_t right_shift = shift > 0 ? 0 : -shift;
    return nn_rounding_divide_by_pot(
        nn_saturating_doubling_high_mul(val * ((int32_t)1 << left_shift), mult),
        right_shift);
}

/* ------------------------------------------------------------------ */
/* Plain-C reference (bit-exact vs SIMD paths)                          */
/* ------------------------------------------------------------------ */
void esp_nn_conv_s8_ref(const data_dims_t *input_dims,
                        const int8_t *input_data,
                        const data_dims_t *filter_dims,
                        const int8_t *filter_data,
                        const int32_t *bias,
                        const data_dims_t *output_dims,
                        int8_t *out_data,
                        const conv_params_t *conv_params,
                        const quant_data_t *quant_data)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t in_channels = input_dims->channels;
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
    const int32_t out_channels = output_dims->channels;
    const int32_t *out_shift = quant_data->shift;
    const int32_t *out_mult = quant_data->mult;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t filter_ch = filter_dims->channels ? filter_dims->channels
                                                    : in_channels;
    const int32_t groups = in_channels / filter_ch;
    const int32_t filters_per_group = out_channels / groups;

    for (int32_t out_y = 0; out_y < out_ht; out_y++) {
        for (int32_t out_x = 0; out_x < out_wd; out_x++) {
            for (int32_t oc = 0; oc < out_channels; oc++) {
                int32_t conv_out = 0;
                const int32_t group = oc / filters_per_group;
                const int32_t in_ch_start = group * filter_ch;

                const int32_t base_y = stride_ht * out_y - pad_ht;
                const int32_t base_x = stride_wd * out_x - pad_wd;
                const int32_t fy_start = max(0, -base_y);
                const int32_t fx_start = max(0, -base_x);
                const int32_t fy_end = min(filter_ht, input_ht - base_y);
                const int32_t fx_end = min(filter_wd, input_wd - base_x);

                for (int32_t fy = fy_start; fy < fy_end; fy++) {
                    for (int32_t fx = fx_start; fx < fx_end; fx++) {
                        const int32_t in_row = base_y + fy;
                        const int32_t in_col = base_x + fx;
                        const int32_t in_base = (in_row * input_wd + in_col)
                                                * in_channels + in_ch_start;
                        const int32_t f_base = oc * filter_ch * filter_ht * filter_wd
                                             + (fy * filter_wd + fx) * filter_ch;
                        for (int32_t c = 0; c < filter_ch; c++) {
                            conv_out += (input_data[in_base + c] + input_offset)
                                        * filter_data[f_base + c];
                        }
                    }
                }
                if (bias) conv_out += bias[oc];
                conv_out = nn_requantize(conv_out, out_mult[oc], out_shift[oc]);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_data++ = (int8_t)conv_out;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fast buffer                                                         */
/* ------------------------------------------------------------------ */
/*
 * Why the S3 paths work in blocks of output channels.
 *
 * The S3 reads PSRAM and flash through a 32 KB data cache. The im2col path used to
 * copy a layer's whole filter set to scratch (malloc: PSRAM, being large) and then,
 * for every output pixel, read all of it: out_ch x window bytes, one byte per MAC.
 * Any layer whose filters are bigger than the cache -- most of a detector's compute
 * -- streamed its filters from PSRAM again for each pixel, and the conv ran at the
 * speed of PSRAM, not of the vector unit. The 1x1 path did the same from flash for
 * every eight pixels.
 *
 * Now each conv takes as many output channels' filters as fit in one buffer of
 * internal SRAM (no cache involved, single-cycle loads), runs every pixel against
 * them, then the next block. The input is walked once per block instead of once, and
 * an im2col window or eight-pixel transpose is rebuilt per block, both small beside
 * the MACs they serve. Results are bit-identical: the same arithmetic in another
 * order of output channels, each still accumulated in full before it is requantized.
 */
static int8_t *s_fast = NULL;
static int     s_fast_size = 0;
static int     s_fast_owned = 0;
static const char *s_last_path = "none";

/* Taken by the first conv if the caller gave none: internal SRAM, largest first --
 * 8 KB is enough for the largest im2col window a 3x3 conv over 256 channels needs,
 * plus a block of filters -- and failing that 16 KB of PSRAM, where a block small
 * enough to stay in the data cache still avoids the streaming. A caller that knows
 * what else needs internal SRAM (wifi, TLS) should give its own instead. */
static const int s_fast_tries[] = { 32768, 24576, 16384, 8192 };
#define FAST_PSRAM_SIZE 16384

static void fast_free(void *p)
{
#if ESP_NN_S3_SIMD
    heap_caps_free(p);
#else
    free(p);
#endif
}

void esp_nn_conv_s8_simd_set_fast_buf(void *buf, int size_bytes)
{
    if (s_fast_owned && s_fast) fast_free(s_fast);
    s_fast = (int8_t *)buf;
    s_fast_size = buf ? size_bytes : 0;
    s_fast_owned = 0;
}

int esp_nn_conv_s8_simd_fast_size(void)
{
    return s_fast_size;
}

__attribute__((unused))
static int8_t *fast_buf(int *size)
{
    if (!s_fast) {
        for (unsigned i = 0; i < sizeof(s_fast_tries) / sizeof(s_fast_tries[0]); i++) {
#if ESP_NN_S3_SIMD
            void *p = heap_caps_malloc(s_fast_tries[i], MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
            void *p = malloc(s_fast_tries[i]);
#endif
            if (p) { s_fast = (int8_t *)p; s_fast_size = s_fast_tries[i]; s_fast_owned = 1; break; }
        }
#if ESP_NN_S3_SIMD
        if (!s_fast) {
            void *p = heap_caps_malloc(FAST_PSRAM_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (p) { s_fast = (int8_t *)p; s_fast_size = FAST_PSRAM_SIZE; s_fast_owned = 1; }
        }
#endif
    }
    *size = s_fast_size;
    return s_fast;
}

/* Kept for callers of the old API: the S3 paths no longer need scratch beyond the
 * fast buffer, so a scratch buffer given here is used as the fast buffer. */
void esp_nn_conv_s8_simd_set_scratch_buf(void *buf, int size_bytes)
{
    esp_nn_conv_s8_simd_set_fast_buf(buf, size_bytes);
}

const char *esp_nn_conv_s8_simd_last_path(void)
{
    return s_last_path;
}

#define ALIGN16(p) ((int8_t *)(((uintptr_t)(p) + 15) & ~(uintptr_t)15))

/* Path predicates ---------------------------------------------------- */
__attribute__((unused))
static int can_use_1x1(const data_dims_t *in, const data_dims_t *filt,
                       const conv_params_t *p)
{
    return filt->width == 1 && filt->height == 1 &&
           p->stride.width == 1 && p->stride.height == 1 &&
           p->padding.width == 0 && p->padding.height == 0 &&
           (filt->channels == 0 || filt->channels == in->channels);
}

__attribute__((unused))
static int is_plain_conv(const data_dims_t *in, const data_dims_t *filt)
{
    return (filt->channels == 0 || filt->channels == in->channels);
}

/* Output channels per block for each path, given [size] bytes of fast buffer;
 * 0 if even one does not fit. */
__attribute__((unused))
static int block_1x1(int size, int in_ch, int out_ch)
{
    const int tbuf = (in_ch / 8) * 128 + 16;
    int block = (size - tbuf - 32) / in_ch;
    if (block > out_ch) block = out_ch;
    return block < 1 ? 0 : block;
}

__attribute__((unused))
static int block_im2col(int size, int window_len, int out_ch)
{
    const int wla = (window_len + 15) & ~15;
    int block = (size - wla - 64) / (wla + (int)sizeof(int32_t));
    if (block > out_ch) block = out_ch;
    return block < 1 ? 0 : block;
}

int esp_nn_conv_s8_simd_get_scratch_size(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    (void)input_dims; (void)filter_dims; (void)output_dims; (void)conv_params;
    return 0;   /* the S3 paths use the fast buffer only */
}

/* ================================================================== */
/* ESP32-S3 vector primitives                                          */
/* ================================================================== */
#if ESP_NN_S3_SIMD

/* ---- aligned int8 dot product via ACCX ---------------------------- *
 * a, b must be 16-byte aligned; len must be a multiple of 16.          */
static inline int32_t s3_dot_s8_aligned(const int8_t *a, const int8_t *b,
                                        int len)
{
    int32_t acc;
    int32_t tmp;
    __asm__ volatile("ee.zero.accx");
    while (len >= 32) {
        __asm__ volatile(
            "ee.vld.128.ip q0, %[a], 16\n"
            "ee.vld.128.ip q1, %[b], 16\n"
            "ee.vld.128.ip q2, %[a], 16\n"
            "ee.vld.128.ip q3, %[b], 16\n"
            "ee.vmulas.s8.accx q0, q1\n"
            "ee.vmulas.s8.accx q2, q3\n"
            : [a] "+r" (a), [b] "+r" (b) : : "memory");
        len -= 32;
    }
    if (len >= 16) {
        __asm__ volatile(
            "ee.vld.128.ip q0, %[a], 16\n"
            "ee.vld.128.ip q1, %[b], 16\n"
            "ee.vmulas.s8.accx q0, q1\n"
            : [a] "+r" (a), [b] "+r" (b) : : "memory");
    }
    __asm__ volatile(
        "movi.n %[t], 0\n"
        "ee.srs.accx %[acc], %[t], 0\n"
        : [acc] "=r" (acc), [t] "=r" (tmp) : : "memory");
    return acc;
}

/* ---- 1x1 path: transpose 8 pixels + QACC parallel MAC -------------- */

static inline void transpose_8x8_s16_simd(const int8_t *input, int stride,
                                          int16_t offset16, int16_t *out_buf)
{
    const int8_t *p = input;
    int16_t *out = out_buf;
    int16_t *off_ptr = &offset16;

    __asm__ volatile(
        "ee.vldbc.16 q5, %[off]\n"
        "ee.zero.q q7\n"
        "ee.vld.l.64.xp q0, %[p], %[s]\n"
        "ee.vld.l.64.xp q1, %[p], %[s]\n"
        "ee.vld.h.64.xp q0, %[p], %[s]\n"
        "ee.vld.h.64.xp q1, %[p], %[s]\n"
        "ee.vld.l.64.xp q2, %[p], %[s]\n"
        "ee.vzip.8 q0, q1\n"
        "ee.vld.l.64.xp q3, %[p], %[s]\n"
        "ee.vld.h.64.xp q2, %[p], %[s]\n"
        "ee.vld.h.64.ip q3, %[p], 0\n"
        "ee.vzip.16 q0, q1\n"
        "ee.vzip.8 q2, q3\n"
        "ee.vzip.16 q2, q3\n"
        "ee.vzip.32 q0, q2\n"
        "ee.vcmp.lt.s8 q4, q2, q7\n"
        "ee.vzip.8 q2, q4\n"
        "ee.vcmp.lt.s8 q6, q0, q7\n"
        "ee.vzip.8 q0, q6\n"
        "ee.vadds.s16 q0, q0, q5\n"
        "ee.vst.128.ip q0, %[out], 16\n"
        "ee.vadds.s16 q6, q6, q5\n"
        "ee.vst.128.ip q6, %[out], 16\n"
        "ee.vadds.s16 q2, q2, q5\n"
        "ee.vst.128.ip q2, %[out], 16\n"
        "ee.vadds.s16 q4, q4, q5\n"
        "ee.vst.128.ip q4, %[out], 16\n"
        "ee.vzip.32 q1, q3\n"
        "ee.vcmp.lt.s8 q4, q3, q7\n"
        "ee.vzip.8 q3, q4\n"
        "ee.vcmp.lt.s8 q6, q1, q7\n"
        "ee.vzip.8 q1, q6\n"
        "ee.vadds.s16 q1, q1, q5\n"
        "ee.vst.128.ip q1, %[out], 16\n"
        "ee.vadds.s16 q6, q6, q5\n"
        "ee.vst.128.ip q6, %[out], 16\n"
        "ee.vadds.s16 q3, q3, q5\n"
        "ee.vst.128.ip q3, %[out], 16\n"
        "ee.vadds.s16 q4, q4, q5\n"
        "ee.vst.128.ip q4, %[out], 16\n"
        : [p] "+r" (p), [out] "+r" (out), [off] "+r" (off_ptr)
        : [s] "r" (stride)
        : "memory");
}

static inline void mac_8pos_8ch_simd(const int16_t *data_buf,
                                     const int8_t *filter)
{
    int8_t __attribute__((aligned(16))) f_aligned[16];
    memcpy(f_aligned, filter, 8);

    const int16_t *dp = data_buf;
    const int8_t *fp = f_aligned;
    __asm__ volatile(
        "ee.zero.q q5\n"
        "ee.vld.l.64.ip q7, %[f], 0\n"
        "ee.vld.128.ip q0, %[d], 16\n"
        "ee.vld.128.ip q1, %[d], 16\n"
        "ee.vcmp.lt.s8 q6, q7, q5\n"
        "ee.vzip.8 q7, q6\n"
        "ee.vsmulas.s16.qacc.ld.incp q2, %[d], q0, q7, 0\n"
        "ee.vsmulas.s16.qacc.ld.incp q3, %[d], q1, q7, 1\n"
        "ee.vsmulas.s16.qacc.ld.incp q0, %[d], q2, q7, 2\n"
        "ee.vsmulas.s16.qacc.ld.incp q1, %[d], q3, q7, 3\n"
        "ee.vsmulas.s16.qacc.ld.incp q2, %[d], q0, q7, 4\n"
        "ee.vsmulas.s16.qacc.ld.incp q3, %[d], q1, q7, 5\n"
        "ee.vsmulas.s16.qacc q2, q7, 6\n"
        "ee.vsmulas.s16.qacc q3, q7, 7\n"
        : [d] "+r" (dp), [f] "+r" (fp) : : "memory");
}

static inline void qacc_zero(void)
{
    __asm__ volatile("ee.zero.qacc");
}

/* The eight 40-bit QACC lanes, low 32 bits each. */
static inline void qacc_read8(int32_t *qacc)
{
    int8_t __attribute__((aligned(16))) qraw[24];
    int8_t *qp = qraw;
    __asm__ volatile(
        "ee.st.qacc_l.l.128.ip %[p], 16\n"
        "ee.st.qacc_l.h.32.ip  %[p], -16\n"
        : [p] "+r" (qp) : : "memory");
    memcpy(&qacc[0], qraw + 0, 4);
    memcpy(&qacc[1], qraw + 5, 4);
    memcpy(&qacc[2], qraw + 10, 4);
    memcpy(&qacc[3], qraw + 15, 4);
    qp = qraw;
    __asm__ volatile(
        "ee.st.qacc_h.l.128.ip %[p], 16\n"
        "ee.st.qacc_h.h.32.ip  %[p], -16\n"
        : [p] "+r" (qp) : : "memory");
    memcpy(&qacc[4], qraw + 0, 4);
    memcpy(&qacc[5], qraw + 5, 4);
    memcpy(&qacc[6], qraw + 10, 4);
    memcpy(&qacc[7], qraw + 15, 4);
}

#elif defined(ESP_NN_S3_EMULATE)

/* Plain-C stand-ins with the vector instructions' results, for the PC test. */
static inline int32_t s3_dot_s8_aligned(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

static inline void transpose_8x8_s16_simd(const int8_t *input, int stride,
                                          int16_t offset16, int16_t *out_buf)
{
    for (int ch = 0; ch < 8; ch++)
        for (int pos = 0; pos < 8; pos++)
            out_buf[ch * 8 + pos] = (int16_t)(input[pos * stride + ch] + offset16);
}

static int64_t s_emul_qacc[8];

static inline void qacc_zero(void)
{
    memset(s_emul_qacc, 0, sizeof(s_emul_qacc));
}

static inline void mac_8pos_8ch_simd(const int16_t *data_buf, const int8_t *filter)
{
    for (int ch = 0; ch < 8; ch++)
        for (int pos = 0; pos < 8; pos++)
            s_emul_qacc[pos] += (int32_t)data_buf[ch * 8 + pos] * filter[ch];
}

static inline void qacc_read8(int32_t *qacc)
{
    for (int i = 0; i < 8; i++) qacc[i] = (int32_t)s_emul_qacc[i];
}

#endif

#if ESP_NN_S3_PATHS

static inline void transpose_8x8_s16_c(const int8_t *input, int stride,
                                       int32_t input_offset, int16_t *out_buf)
{
    for (int ch = 0; ch < 8; ch++) {
        for (int pos = 0; pos < 8; pos++) {
            out_buf[ch * 8 + pos] =
                (int16_t)(input[pos * stride + ch] + input_offset);
        }
    }
}

static inline int8_t requant_out(int32_t acc, int32_t mult, int32_t shift,
                                 int32_t out_offset, int32_t act_min, int32_t act_max)
{
    acc = nn_requantize(acc, mult, shift);
    acc += out_offset;
    acc = max(acc, act_min);
    acc = min(acc, act_max);
    return (int8_t)acc;
}

/* ---- 1x1, stride 1, no pad: blocks of output channels -------------- */
static void conv_s8_1x1_s3(const int8_t *input,
                           int input_wd, int input_ht, int in_channels,
                           int32_t input_offset,
                           const int8_t *filter_data, const int32_t *bias,
                           int8_t *out_data, int out_channels,
                           int32_t out_offset,
                           const int32_t *out_shift, const int32_t *out_mult,
                           int32_t activation_min, int32_t activation_max,
                           int8_t *fast, int fast_size, int block)
{
    const int size = input_wd * input_ht;
    const int ch8 = in_channels / 8;
    const int use_simd_transpose = (in_channels % 8 == 0) &&
                                   (((uintptr_t)input & 7) == 0);
    const int16_t offset16 = (int16_t)input_offset;

    /* fast buffer: [transposed 8 pixels, int16 | this block's filter rows] */
    int16_t *tbuf = (int16_t *)ALIGN16(fast);
    int8_t *filt_block = ALIGN16((int8_t *)tbuf + ch8 * 128);
    (void)fast_size;

    for (int oc0 = 0; oc0 < out_channels; oc0 += block) {
        const int nb = min(block, out_channels - oc0);
        memcpy(filt_block, filter_data + (size_t)oc0 * in_channels,
               (size_t)nb * in_channels);

        int pos = 0;
        for (; pos + 7 < size; pos += 8) {
            const int8_t *in_base = input + (size_t)pos * in_channels;

            for (int cg = 0; cg < ch8; cg++) {
                int16_t *cg_buf = tbuf + cg * 64;
                if (use_simd_transpose) {
                    transpose_8x8_s16_simd(in_base + cg * 8, in_channels,
                                           offset16, cg_buf);
                } else {
                    transpose_8x8_s16_c(in_base + cg * 8, in_channels,
                                        input_offset, cg_buf);
                }
            }
            __asm__ volatile("" ::: "memory");

            int8_t *out_pos = out_data + (size_t)pos * out_channels + oc0;
            for (int ob = 0; ob < nb; ob++) {
                const int oc = oc0 + ob;
                const int8_t *filt = filt_block + (size_t)ob * in_channels;

                qacc_zero();
                for (int cg = 0; cg < ch8; cg++) {
                    mac_8pos_8ch_simd(tbuf + cg * 64, filt + cg * 8);
                }

                int32_t qacc[8];
                qacc_read8(qacc);

                for (int c = ch8 * 8; c < in_channels; c++) {
                    int16_t f = (int16_t)filt[c];
                    for (int p = 0; p < 8; p++) {
                        qacc[p] += ((int32_t)in_base[p * in_channels + c]
                                    + input_offset) * f;
                    }
                }

                const int32_t b = bias ? bias[oc] : 0;
                for (int p = 0; p < 8; p++) {
                    out_pos[p * out_channels + ob] =
                        requant_out(qacc[p] + b, out_mult[oc], out_shift[oc],
                                    out_offset, activation_min, activation_max);
                }
            }
        }

        /* leftover pixels */
        for (; pos < size; pos++) {
            const int8_t *in_ptr = input + (size_t)pos * in_channels;
            for (int ob = 0; ob < nb; ob++) {
                const int oc = oc0 + ob;
                const int8_t *filt = filt_block + (size_t)ob * in_channels;
                int32_t acc = 0;
                for (int c = 0; c < in_channels; c++) {
                    acc += ((int32_t)in_ptr[c] + input_offset) * (int32_t)filt[c];
                }
                if (bias) acc += bias[oc];
                out_data[(size_t)pos * out_channels + oc] =
                    requant_out(acc, out_mult[oc], out_shift[oc],
                                out_offset, activation_min, activation_max);
            }
        }
    }
}

/* ---- general path: im2col + ACCX dot product, blocks of output channels --- *
 * Handles padding, any stride, any filter size, any in_ch.                     */
static void conv_s8_im2col_s3(const data_dims_t *input_dims,
                              const int8_t *input_data,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const conv_params_t *conv_params,
                              const quant_data_t *quant_data,
                              int8_t *fast, int fast_size, int block)
{
    const int32_t input_wd = input_dims->width;
    const int32_t input_ht = input_dims->height;
    const int32_t in_ch = input_dims->channels;
    const int32_t filter_wd = filter_dims->width;
    const int32_t filter_ht = filter_dims->height;
    const int32_t out_wd = output_dims->width;
    const int32_t out_ht = output_dims->height;
    const int32_t out_ch = output_dims->channels;
    const int32_t pad_wd = conv_params->padding.width;
    const int32_t pad_ht = conv_params->padding.height;
    const int32_t stride_wd = conv_params->stride.width;
    const int32_t stride_ht = conv_params->stride.height;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t window_len = filter_wd * filter_ht * in_ch;
    const int32_t window_len_aligned = (window_len + 15) & ~15;
    const int8_t pad_val = (int8_t)(-input_offset);
    (void)fast_size;

    /* fast buffer: [im2col window | corrections | this block's filters] */
    int8_t *im2col_buf = ALIGN16(fast);
    int32_t *corrections = (int32_t *)ALIGN16(im2col_buf + window_len_aligned);
    int8_t *aligned_filter = ALIGN16((int8_t *)(corrections + block));

    memset(im2col_buf + window_len, 0, window_len_aligned - window_len);

    /* NOTE on the correction trick: it assumes every window element is
     * (value + input_offset) * filter. Padded elements are written as
     * -input_offset so they contribute exactly 0 after correction. */

    const int32_t row_bytes = filter_wd * in_ch;

    /* interior region where the whole window is inside the input */
    int32_t safe_y_start = (pad_ht + stride_ht - 1) / stride_ht;
    int32_t safe_y_end = (input_ht - filter_ht + pad_ht) / stride_ht + 1;
    int32_t safe_x_start = (pad_wd + stride_wd - 1) / stride_wd;
    int32_t safe_x_end = (input_wd - filter_wd + pad_wd) / stride_wd + 1;
    safe_y_start = min(safe_y_start, out_ht);
    safe_y_end = min(max(safe_y_end, safe_y_start), out_ht);
    safe_x_start = min(safe_x_start, out_wd);
    safe_x_end = min(max(safe_x_end, safe_x_start), out_wd);

    for (int32_t oc0 = 0; oc0 < out_ch; oc0 += block) {
        const int32_t nb = min(block, out_ch - oc0);

        /* This block's filters, zero-padded to the aligned length so the dot
         * product over it is exact, and filter_sum * input_offset (+ bias). */
        const int8_t *fptr = filter_data + (size_t)oc0 * window_len;
        int8_t *af = aligned_filter;
        for (int32_t ob = 0; ob < nb; ob++) {
            int32_t sum = 0;
            for (int32_t i = 0; i < window_len; i++) sum += fptr[i];
            corrections[ob] = sum * input_offset + (bias ? bias[oc0 + ob] : 0);
            memcpy(af, fptr, window_len);
            memset(af + window_len, 0, window_len_aligned - window_len);
            fptr += window_len;
            af += window_len_aligned;
        }

        int8_t *out_ptr = out_data + oc0;
        for (int32_t out_y = 0; out_y < out_ht; out_y++) {
            const int32_t base_y = out_y * stride_ht - pad_ht;
            const int is_safe_y = (out_y >= safe_y_start && out_y < safe_y_end);

            for (int32_t out_x = 0; out_x < out_wd; out_x++) {
                const int32_t base_x = out_x * stride_wd - pad_wd;
                int8_t *buf = im2col_buf;

                if (is_safe_y && out_x >= safe_x_start && out_x < safe_x_end) {
                    for (int32_t fy = 0; fy < filter_ht; fy++) {
                        const int8_t *src = input_data +
                            ((base_y + fy) * input_wd + base_x) * in_ch;
                        memcpy(buf, src, row_bytes);
                        buf += row_bytes;
                    }
                } else {
                    for (int32_t fy = 0; fy < filter_ht; fy++) {
                        const int32_t in_y = base_y + fy;
                        if (in_y >= 0 && in_y < input_ht) {
                            for (int32_t fx = 0; fx < filter_wd; fx++) {
                                const int32_t in_x = base_x + fx;
                                if (in_x >= 0 && in_x < input_wd) {
                                    memcpy(buf,
                                           input_data + (in_y * input_wd + in_x) * in_ch,
                                           in_ch);
                                } else {
                                    memset(buf, pad_val, in_ch);
                                }
                                buf += in_ch;
                            }
                        } else {
                            memset(buf, pad_val, row_bytes);
                            buf += row_bytes;
                        }
                    }
                }

                const int8_t *filter_ptr = aligned_filter;
                for (int32_t ob = 0; ob < nb; ob++) {
                    const int32_t oc = oc0 + ob;
                    int32_t conv_out = s3_dot_s8_aligned(im2col_buf, filter_ptr,
                                                         window_len_aligned);
                    out_ptr[ob] = requant_out(conv_out + corrections[ob],
                                              quant_data->mult[oc], quant_data->shift[oc],
                                              out_offset, activation_min, activation_max);
                    filter_ptr += window_len_aligned;
                }
                out_ptr += out_ch;
            }
        }
    }
}

#endif /* ESP_NN_S3_PATHS */

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */
int esp_nn_conv_s8_simd(const data_dims_t *input_dims,
                        const int8_t *input_data,
                        const data_dims_t *filter_dims,
                        const int8_t *filter_data,
                        const int32_t *bias,
                        const data_dims_t *output_dims,
                        int8_t *out_data,
                        const conv_params_t *conv_params,
                        const quant_data_t *quant_data)
{
    if (conv_params->dilation.width > 1 || conv_params->dilation.height > 1) {
        return -1; /* dilation not supported */
    }

#if ESP_NN_S3_PATHS
    if (is_plain_conv(input_dims, filter_dims)) {
        int size = 0;
        int8_t *fast = fast_buf(&size);
        const int out_ch = output_dims->channels;

        if (fast && can_use_1x1(input_dims, filter_dims, conv_params)) {
            int block = block_1x1(size, input_dims->channels, out_ch);
            if (block) {
                s_last_path = "1x1_simd";
                conv_s8_1x1_s3(input_data,
                               input_dims->width, input_dims->height,
                               input_dims->channels, conv_params->in_offset,
                               filter_data, bias, out_data,
                               out_ch, conv_params->out_offset,
                               quant_data->shift, quant_data->mult,
                               conv_params->activation.min,
                               conv_params->activation.max,
                               fast, size, block);
                return 0;
            }
        } else if (fast) {
            int window = filter_dims->width * filter_dims->height * input_dims->channels;
            int block = block_im2col(size, window, out_ch);
            if (block) {
                s_last_path = "im2col_simd";
                conv_s8_im2col_s3(input_dims, input_data, filter_dims, filter_data,
                                  bias, output_dims, out_data, conv_params,
                                  quant_data, fast, size, block);
                return 0;
            }
        }
    }
#endif

    /* grouped conv, non-S3 target, or no fast buffer to be had */
    s_last_path = "c_ref";
    esp_nn_conv_s8_ref(input_dims, input_data, filter_dims, filter_data,
                       bias, output_dims, out_data, conv_params, quant_data);
    return 0;
}
