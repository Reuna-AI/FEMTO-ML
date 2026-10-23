/*
 * esp_nn_conv_s8_simd.h
 *
 * Single-file, self-contained int8 (s8) 2D convolution for ESP32-S3,
 * NHWC layout, TFLite-Micro compatible quantization.
 *
 * Paths selected automatically at runtime:
 *   1. Every plain conv on the S3 -> im2col + ACCX SIMD dot product, in blocks of
 *                                    output channels whose filters fit in the fast
 *                                    buffer (1x1, padding, any stride, any filter
 *                                    size), on one core or two
 *   2. Grouped convs, non-S3      -> plain C fallback (bit-exact same results)
 *
 * ---------------------------------------------------------------------------
 * USAGE
 * ---------------------------------------------------------------------------
 *   #include "esp_nn_conv_s8_simd.h"
 *
 *   data_dims_t in_dims  = { .width = 26, .height = 26, .channels = 128, .extra = 1 };
 *   data_dims_t filt_dims= { .width = 3,  .height = 3,  .channels = 128, .extra = 128 };
 *   data_dims_t out_dims = { .width = 12, .height = 12, .channels = 128, .extra = 1 };
 *
 *   conv_params_t params = {
 *       .in_offset  = 128,          // -input_zero_point... (TFLM: -zp_in)
 *       .out_offset = -128,         // output zero point
 *       .stride     = {2, 2},
 *       .padding    = {0, 0},
 *       .dilation   = {1, 1},       // must be 1,1
 *       .activation = {-128, 127},
 *   };
 *   quant_data_t quant = { .shift = per_ch_shift, .mult = per_ch_mult };
 *
 *   // 1) (optional) give the library its fast buffer: internal SRAM, 8-48 KB.
 *   //    Each conv runs in blocks of output channels whose filters fit in it
 *   //    (see "Fast buffer" in the .c). If you skip this, the first conv takes
 *   //    the largest of 32/24/16/8 KB of internal SRAM it can get, or 16 KB of
 *   //    PSRAM -- without regard to what wifi or TLS will need later.
 *   esp_nn_conv_s8_simd_set_fast_buf(my_sram_buffer, my_sram_buffer_size);
 *
 *   // 2) run the convolution
 *   esp_nn_conv_s8_simd(&in_dims, input, &filt_dims, filter, bias,
 *                       &out_dims, output, &params, &quant);
 *
 * Data layout (all int8 NHWC):
 *   input : [in_ht][in_wd][in_ch]
 *   filter: [out_ch][filt_ht][filt_wd][in_ch]     (TFLM OHWI)
 *   output: [out_ht][out_wd][out_ch]
 *   bias  : int32[out_ch] or NULL
 *   mult/shift: int32[out_ch] per-channel requantization (TFLM convention)
 *
 * Constraints:
 *   - dilation must be 1
 *   - grouped/depthwise conv (filter channels != input channels) falls back
 *     to the C reference path
 *
 * A bit-exact C reference `esp_nn_conv_s8_ref()` with the same signature is
 * exported for validation/benchmarking.
 * ---------------------------------------------------------------------------
 */

#ifndef ESP_NN_CONV_S8_SIMD_H
#define ESP_NN_CONV_S8_SIMD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* If you already use esp-nn, define ESP_NN_SIMD_USE_ESP_NN_DEFS before
 * including this header to reuse its types instead of the local copies. */
#ifdef ESP_NN_SIMD_USE_ESP_NN_DEFS
#include <esp_nn_defs.h>
#else

typedef struct {
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t extra;      /* batch (input/output) or out_channels (filter) */
} data_dims_t;

typedef struct {
    int32_t width;
    int32_t height;
} dims2_t;

typedef struct {
    int32_t min;
    int32_t max;
} activation_t;

typedef struct {
    int32_t in_offset;
    int32_t out_offset;
    dims2_t stride;
    dims2_t padding;
    dims2_t dilation;
    activation_t activation;
} conv_params_t;

typedef struct {
    const int32_t *shift;   /* per-output-channel shift */
    const int32_t *mult;    /* per-output-channel multiplier */
} quant_data_t;

#endif /* ESP_NN_SIMD_USE_ESP_NN_DEFS */

/**
 * Give the S3 paths their fast buffer (internal SRAM; any alignment). Pass NULL,0
 * to let the library take its own on the next conv.
 */
void esp_nn_conv_s8_simd_set_fast_buf(void *buf, int size_bytes);

/** Bytes of fast buffer in use; 0 before the first conv if none was given. */
int esp_nn_conv_s8_simd_fast_size(void);

/**
 * 2: run each conv on both cores, a worker on the other core taking half of the
 * output rows (created on the first call, pinned to the core this is not called
 * from, at the caller's priority). 1: this core only (the default). Call it from
 * the task that runs the model. Returns the number of cores now in use.
 */
int esp_nn_conv_s8_simd_set_cores(int cores);

/** The number of cores the convs run on now: 1, or 2 after set_cores(2). */
int esp_nn_conv_s8_simd_cores(void);

/**
 * The fast buffer itself, taken now if no conv has taken it yet (NULL if none can
 * be had), and its size in *size. For another kernel run from the same task
 * between convs -- the depthwise conv borrows it for its rows -- never during one.
 */
void *esp_nn_conv_s8_simd_borrow_fast_buf(int *size);

/**
 * With esp_nn_conv_s8_simd_set_cores(2): fn(theirs) on the worker on the other
 * core while fn(mine) runs here; returns 1 once both are done. Returns 0, having
 * run nothing, on one core. For kernels that split their rows like the convs.
 */
int esp_nn_simd_both_cores(void (*fn)(void *), void *mine, void *theirs);

/**
 * Kept for the old API: always 0 now, as the S3 paths need only the fast buffer.
 */
int esp_nn_conv_s8_simd_get_scratch_size(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params);

/** Kept for the old API: the same as esp_nn_conv_s8_simd_set_fast_buf(). */
void esp_nn_conv_s8_simd_set_scratch_buf(void *buf, int size_bytes);

/**
 * Run the s8 NHWC convolution. Picks the fastest available path.
 * Returns 0 on success, -1 on scratch allocation failure or bad params.
 */
int esp_nn_conv_s8_simd(const data_dims_t *input_dims,
                        const int8_t *input_data,
                        const data_dims_t *filter_dims,
                        const int8_t *filter_data,
                        const int32_t *bias,
                        const data_dims_t *output_dims,
                        int8_t *out_data,
                        const conv_params_t *conv_params,
                        const quant_data_t *quant_data);

/**
 * Plain-C bit-exact reference implementation (for testing / non-S3 targets).
 */
void esp_nn_conv_s8_ref(const data_dims_t *input_dims,
                        const int8_t *input_data,
                        const data_dims_t *filter_dims,
                        const int8_t *filter_data,
                        const int32_t *bias,
                        const data_dims_t *output_dims,
                        int8_t *out_data,
                        const conv_params_t *conv_params,
                        const quant_data_t *quant_data);

/** Name of the path chosen by the last esp_nn_conv_s8_simd() call
 *  ("im2col_simd", "im2col_simd_2core", "c_ref"). Handy for benchmarks. */
const char *esp_nn_conv_s8_simd_last_path(void);

#ifdef __cplusplus
}
#endif

#endif /* ESP_NN_CONV_S8_SIMD_H */
