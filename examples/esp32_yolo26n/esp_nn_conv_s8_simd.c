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
#  include "freertos/FreeRTOS.h"
#  include "freertos/task.h"
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
 * How the S3 path is laid out, and why (October 2026, measured on the board and
 * counted instruction by instruction in Espressif's QEMU):
 *
 * 1. Blocks of output channels. The S3 reads PSRAM and flash through a 32 KB data
 *    cache. The old im2col path copied a layer's whole filter set to scratch (PSRAM)
 *    and read all of it for every output pixel, one byte per MAC: any layer with more
 *    filters than the cache ran at PSRAM speed. Now each conv takes as many output
 *    channels' filters as fit in one buffer of internal SRAM, runs every pixel
 *    against them, then the next block. YOLO26n at 256x256: 11 s -> 5.1 s a run.
 *
 * 2. One path for every plain conv, 1x1 included. The old 1x1 path transposed
 *    eight pixels to int16 and accumulated them in QACC; reading QACC's 40-bit lanes
 *    back cost eight calls to memcpy for every eight outputs, and its MAC loop
 *    another memcpy per eight channels. A 1x1 conv is an im2col conv whose window is
 *    one pixel: through the ACCX dot product it takes 1.6x (48 channels in) to 3x
 *    (384 in) fewer instructions.
 *
 * 3. Per output, only what must be per output. Each block's requant multipliers,
 *    shifts and filter-sum corrections sit beside its filters, the requant is inlined
 *    and branch-free, and the dot product loads its next operands in the same
 *    instruction that multiplies the current ones.
 *
 * 4. Both cores. With esp_nn_conv_s8_simd_set_cores(2) a worker on the other core
 *    takes the lower half of each block's output rows (its own im2col window, the
 *    block's filters shared, read-only). Off by default: the caller knows what else
 *    the other core does.
 *
 * Results are bit-identical to esp_nn_conv_s8_ref() throughout: the same integer
 * arithmetic, in another order of output channels and pixels, each output
 * accumulated in full before it is requantized (tools/test_conv_kernels.c, on a PC
 * and on the S3 vector unit in QEMU).
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

/* Kept for callers of the old API: the S3 path needs no scratch beyond the fast
 * buffer, so a scratch buffer given here is used as the fast buffer. */
void esp_nn_conv_s8_simd_set_scratch_buf(void *buf, int size_bytes)
{
    esp_nn_conv_s8_simd_set_fast_buf(buf, size_bytes);
}

const char *esp_nn_conv_s8_simd_last_path(void)
{
    return s_last_path;
}

int esp_nn_conv_s8_simd_get_scratch_size(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params)
{
    (void)input_dims; (void)filter_dims; (void)output_dims; (void)conv_params;
    return 0;   /* the S3 path uses the fast buffer only */
}

#define ALIGN16(p) ((int8_t *)(((uintptr_t)(p) + 15) & ~(uintptr_t)15))

__attribute__((unused))
static int is_plain_conv(const data_dims_t *in, const data_dims_t *filt)
{
    return (filt->channels == 0 || filt->channels == in->channels);
}

/* Per output channel in a block: its aligned filter, and four int32 -- the
 * filter-sum correction (+ bias), the requant multiplier, and the shift split into
 * its left and right parts. */
#define PER_OC_WORDS 4

/* Output channels per block for [size] bytes of fast buffer, with an im2col window
 * for each of [windows] cores; 0 if even one channel does not fit. The 16 bytes per
 * window and the 96 at the end are alignment, and the 16 bytes the dot product's
 * last load may read past a window or a filter. */
__attribute__((unused))
static int block_size(int size, int window_len, int out_ch, int windows)
{
    const int wla = (window_len + 15) & ~15;
    int block = (size - windows * (wla + 16) - 96) / (wla + PER_OC_WORDS * (int)sizeof(int32_t));
    if (block > out_ch) block = out_ch;
    return block < 1 ? 0 : block;
}

/* ================================================================== */
/* ESP32-S3 vector primitives                                          */
/* ================================================================== */
#if ESP_NN_S3_SIMD

/* ---- aligned int8 dot product via ACCX ---------------------------- *
 * a, b 16-byte aligned; len a multiple of 16, at least 16. Each multiply also
 * loads the next 16 bytes of a, so a pair of a and b is three instructions,
 * not four; the last of those loads reads 16 bytes past each, unused.          */
static inline __attribute__((always_inline))
int32_t s3_dot_s8_aligned(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc;
    int32_t tmp;
    int n = len >> 4;                      /* 16-byte vectors */

    __asm__ volatile(
        "ee.zero.accx\n"
        "ee.vld.128.ip q0, %[a], 16\n"
        "ee.vld.128.ip q1, %[b], 16\n"
        : [a] "+r" (a), [b] "+r" (b) : : "memory");

    while (n >= 2) {
        __asm__ volatile(
            "ee.vld.128.ip q3, %[b], 16\n"
            "ee.vmulas.s8.accx.ld.ip q2, %[a], 16, q0, q1\n"
            "ee.vld.128.ip q1, %[b], 16\n"
            "ee.vmulas.s8.accx.ld.ip q0, %[a], 16, q2, q3\n"
            : [a] "+r" (a), [b] "+r" (b) : : "memory");
        n -= 2;
    }
    if (n) {
        __asm__ volatile("ee.vmulas.s8.accx q0, q1\n" ::: "memory");
    }
    __asm__ volatile(
        "movi.n %[t], 0\n"
        "ee.srs.accx %[acc], %[t], 0\n"
        : [acc] "=r" (acc), [t] "=r" (tmp) : : "memory");
    return acc;
}

#elif defined(ESP_NN_S3_EMULATE)

/* Plain C with the vector instructions' result, for the PC test. */
static inline int32_t s3_dot_s8_aligned(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

#endif

#if ESP_NN_S3_PATHS

/* nn_requantize(), inlined, with the shift already split into its left and right
 * parts: the same arithmetic in the same order. The doubling high multiply can only
 * overflow for INT32_MIN * INT32_MIN, which saturates as before. */
static inline __attribute__((always_inline))
int32_t requant_fast(int32_t val, int32_t mult, int32_t left, int32_t right)
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

/* One block of output channels over rows [y0, y1): everything a core needs. */
typedef struct {
    const int8_t *input;
    int8_t *out;                 /* output, already offset to the block's first channel */
    const int8_t *filters;       /* nb aligned filters, wla bytes each */
    const int32_t *per_oc;       /* nb x {correction, mult, shift} */
    int8_t *window;              /* this core's im2col window */
    int32_t in_w, in_h, in_ch, k_w, k_h, stride_w, stride_h, pad_w, pad_h;
    int32_t out_w, out_ch, nb, window_len, wla;
    int32_t out_offset, act_min, act_max, pad_val;
    int32_t safe_x0, safe_x1, safe_y0, safe_y1;
    int32_t y0, y1;
} rows_job_t;

static void conv_rows(const rows_job_t *j)
{
    const int32_t row_bytes = j->k_w * j->in_ch;
    int8_t *out_ptr = j->out + (size_t)j->y0 * j->out_w * j->out_ch;
    /* In locals: the dot product's asm clobbers memory, which would otherwise have
     * every field reloaded for every output. */
    const int8_t *const window = j->window;
    const int8_t *const filters = j->filters;
    const int32_t *const per_oc = j->per_oc;
    const int32_t nb = j->nb, wla = j->wla, out_ch = j->out_ch;
    const int32_t out_offset = j->out_offset, act_min = j->act_min, act_max = j->act_max;

    for (int32_t out_y = j->y0; out_y < j->y1; out_y++) {
        const int32_t base_y = out_y * j->stride_h - j->pad_h;
        const int is_safe_y = (out_y >= j->safe_y0 && out_y < j->safe_y1);

        for (int32_t out_x = 0; out_x < j->out_w; out_x++) {
            const int32_t base_x = out_x * j->stride_w - j->pad_w;
            int8_t *buf = j->window;

            if (is_safe_y && out_x >= j->safe_x0 && out_x < j->safe_x1) {
                const int8_t *src = j->input + ((size_t)base_y * j->in_w + base_x) * j->in_ch;
                for (int32_t fy = 0; fy < j->k_h; fy++) {
                    memcpy(buf, src, row_bytes);
                    buf += row_bytes;
                    src += (size_t)j->in_w * j->in_ch;
                }
            } else {
                for (int32_t fy = 0; fy < j->k_h; fy++) {
                    const int32_t in_y = base_y + fy;
                    if (in_y >= 0 && in_y < j->in_h) {
                        for (int32_t fx = 0; fx < j->k_w; fx++) {
                            const int32_t in_x = base_x + fx;
                            if (in_x >= 0 && in_x < j->in_w) {
                                memcpy(buf, j->input + ((size_t)in_y * j->in_w + in_x) * j->in_ch,
                                       j->in_ch);
                            } else {
                                memset(buf, j->pad_val, j->in_ch);
                            }
                            buf += j->in_ch;
                        }
                    } else {
                        memset(buf, j->pad_val, row_bytes);
                        buf += row_bytes;
                    }
                }
            }

            const int8_t *filter_ptr = filters;
            const int32_t *q = per_oc;
            for (int32_t ob = 0; ob < nb; ob++) {
                int32_t acc = s3_dot_s8_aligned(window, filter_ptr, wla) + q[0];
                acc = requant_fast(acc, q[1], q[2], q[3]) + out_offset;
                acc = max(acc, act_min);
                acc = min(acc, act_max);
                out_ptr[ob] = (int8_t)acc;
                filter_ptr += wla;
                q += PER_OC_WORDS;
            }
            out_ptr += out_ch;
        }
    }
}

/* ---- the other core ------------------------------------------------- */
static int s_cores = 1;

#if ESP_NN_S3_SIMD
static TaskHandle_t s_worker = NULL;
static TaskHandle_t s_waiter = NULL;
static rows_job_t s_job;

static void worker_main(void *arg)
{
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        conv_rows(&s_job);
        xTaskNotifyGive(s_waiter);
    }
}

int esp_nn_conv_s8_simd_set_cores(int cores)
{
    if (cores >= 2 && !s_worker) {
        const BaseType_t other = xPortGetCoreID() ? 0 : 1;
        if (xTaskCreatePinnedToCore(worker_main, "femto_nn", 4096, NULL,
                                    uxTaskPriorityGet(NULL), &s_worker, other) != pdPASS) {
            s_worker = NULL;
        }
    }
    s_cores = (cores >= 2 && s_worker) ? 2 : 1;
    return s_cores;
}

/* Rows [mid, y1) on the worker, [y0, mid) here; back when both are done. */
static void run_rows(rows_job_t *mine, const rows_job_t *theirs)
{
    s_job = *theirs;
    s_waiter = xTaskGetCurrentTaskHandle();
    xTaskNotifyGive(s_worker);
    conv_rows(mine);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}
#else
int esp_nn_conv_s8_simd_set_cores(int cores)
{
    /* the PC test: the two halves one after the other, each with its own window */
    s_cores = cores >= 2 ? 2 : 1;
    return s_cores;
}

static void run_rows(rows_job_t *mine, const rows_job_t *theirs)
{
    conv_rows(mine);
    conv_rows(theirs);
}
#endif

/* ---- every plain conv: im2col + ACCX dot product, blocks of output channels -- *
 * Handles 1x1, padding, any stride, any filter size, any in_ch.                   */
static void conv_s8_im2col_s3(const data_dims_t *input_dims,
                              const int8_t *input_data,
                              const data_dims_t *filter_dims,
                              const int8_t *filter_data,
                              const int32_t *bias,
                              const data_dims_t *output_dims,
                              int8_t *out_data,
                              const conv_params_t *conv_params,
                              const quant_data_t *quant_data,
                              int8_t *fast, int block, int cores)
{
    rows_job_t j;
    j.input = input_data;
    j.in_w = input_dims->width;
    j.in_h = input_dims->height;
    j.in_ch = input_dims->channels;
    j.k_w = filter_dims->width;
    j.k_h = filter_dims->height;
    j.stride_w = conv_params->stride.width;
    j.stride_h = conv_params->stride.height;
    j.pad_w = conv_params->padding.width;
    j.pad_h = conv_params->padding.height;
    j.out_w = output_dims->width;
    j.out_ch = output_dims->channels;
    j.out_offset = conv_params->out_offset;
    j.act_min = conv_params->activation.min;
    j.act_max = conv_params->activation.max;
    j.window_len = j.k_w * j.k_h * j.in_ch;
    j.wla = (j.window_len + 15) & ~15;
    /* Padding is written as -input_offset so that, after the correction below
     * (filter sum x input_offset), it contributes exactly 0. */
    j.pad_val = (int8_t)(-conv_params->in_offset);
    const int32_t out_ht = output_dims->height;
    const int32_t input_offset = conv_params->in_offset;

    /* interior region where the whole window is inside the input */
    int32_t sy0 = (j.pad_h + j.stride_h - 1) / j.stride_h;
    int32_t sy1 = (j.in_h - j.k_h + j.pad_h) / j.stride_h + 1;
    int32_t sx0 = (j.pad_w + j.stride_w - 1) / j.stride_w;
    int32_t sx1 = (j.in_w - j.k_w + j.pad_w) / j.stride_w + 1;
    j.safe_y0 = min(sy0, out_ht);
    j.safe_y1 = min(max(sy1, j.safe_y0), out_ht);
    j.safe_x0 = min(sx0, j.out_w);
    j.safe_x1 = min(max(sx1, j.safe_x0), j.out_w);

    /* fast buffer: [window per core | per-oc words | this block's filters] */
    int8_t *window0 = ALIGN16(fast);
    int8_t *window1 = ALIGN16(window0 + j.wla + 16);
    int32_t *per_oc = (int32_t *)ALIGN16(window1 + (cores > 1 ? j.wla + 16 : 0));
    int8_t *filters = ALIGN16((int8_t *)(per_oc + PER_OC_WORDS * block));

    /* zero the windows' tails once: the dot product runs over the aligned length */
    memset(window0 + j.window_len, 0, j.wla - j.window_len);
    if (cores > 1) memset(window1 + j.window_len, 0, j.wla - j.window_len);

    j.filters = filters;
    j.per_oc = per_oc;

    for (int32_t oc0 = 0; oc0 < j.out_ch; oc0 += block) {
        j.nb = min(block, j.out_ch - oc0);

        const int8_t *fptr = filter_data + (size_t)oc0 * j.window_len;
        int8_t *af = filters;
        int32_t *q = per_oc;
        for (int32_t ob = 0; ob < j.nb; ob++) {
            int32_t sum = 0;
            for (int32_t i = 0; i < j.window_len; i++) sum += fptr[i];
            q[0] = sum * input_offset + (bias ? bias[oc0 + ob] : 0);
            q[1] = quant_data->mult[oc0 + ob];
            q[2] = quant_data->shift[oc0 + ob] > 0 ? quant_data->shift[oc0 + ob] : 0;
            q[3] = quant_data->shift[oc0 + ob] > 0 ? 0 : -quant_data->shift[oc0 + ob];
            memcpy(af, fptr, j.window_len);
            memset(af + j.window_len, 0, j.wla - j.window_len);
            fptr += j.window_len;
            af += j.wla;
            q += PER_OC_WORDS;
        }

        j.out = out_data + oc0;
        if (cores > 1 && out_ht > 1) {
            rows_job_t mine = j, theirs = j;
            const int32_t mid = out_ht / 2;
            mine.window = window0;   mine.y0 = 0;     mine.y1 = mid;
            theirs.window = window1; theirs.y0 = mid; theirs.y1 = out_ht;
            run_rows(&mine, &theirs);
        } else {
            j.window = window0; j.y0 = 0; j.y1 = out_ht;
            conv_rows(&j);
        }
    }
}

#else

int esp_nn_conv_s8_simd_set_cores(int cores)
{
    (void)cores;
    return 1;
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
    /* The S3 path pads with -in_offset as an int8, so in_offset must be -127..128 (an
     * int8 input's zero point negated); outside that, the reference does it right. */
    if (is_plain_conv(input_dims, filter_dims) &&
        conv_params->in_offset >= -127 && conv_params->in_offset <= 128) {
        int size = 0;
        int8_t *fast = fast_buf(&size);
        const int window = filter_dims->width * filter_dims->height * input_dims->channels;
        int cores = s_cores;
        int block = fast ? block_size(size, window, output_dims->channels, cores) : 0;
        if (!block && cores > 1) {
            cores = 1;
            block = fast ? block_size(size, window, output_dims->channels, 1) : 0;
        }
        if (block) {
            s_last_path = cores > 1 ? "im2col_simd_2core" : "im2col_simd";
            conv_s8_im2col_s3(input_dims, input_data, filter_dims, filter_data,
                              bias, output_dims, out_data, conv_params,
                              quant_data, fast, block, cores);
            return 0;
        }
    }
#endif

    /* grouped conv, non-S3 target, or no fast buffer to be had */
    s_last_path = "c_ref";
    esp_nn_conv_s8_ref(input_dims, input_data, filter_dims, filter_data,
                       bias, output_dims, out_data, conv_params, quant_data);
    return 0;
}
