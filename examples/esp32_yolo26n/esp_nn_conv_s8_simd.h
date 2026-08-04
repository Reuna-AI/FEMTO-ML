/*
 * esp_nn_conv_s8_simd.h
 *
 * Single-file, self-contained int8 (s8) 2D convolution for ESP32-S3,
 * NHWC layout, TFLite-Micro compatible quantization.
 *
 * Paths selected automatically at runtime:
 *   1. 1x1 / stride 1 / no pad  -> transpose + QACC parallel MAC (8 pixels at once)
 *   2. Everything else           -> im2col + ACCX SIMD dot product
 *                                   (handles padding, any stride, any filter size)
 *   3. Non-S3 targets            -> plain C fallback (bit-exact same results)
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
 *   // 1) (optional but recommended) give the library a scratch buffer.
 *   //    If you skip this, it mallocs/caches one internally.
 *   int need = esp_nn_conv_s8_simd_get_scratch_size(&in_dims, &filt_dims,
 *                                                   &out_dims, &params);
 *   esp_nn_conv_s8_simd_set_scratch_buf(my_buffer, my_buffer_size);
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
 * Scratch bytes required for the given conv shape (worst case for the path
 * that will actually be chosen). Returns 0 if no scratch is needed.
 */
int esp_nn_conv_s8_simd_get_scratch_size(const data_dims_t *input_dims,
                                         const data_dims_t *filter_dims,
                                         const data_dims_t *output_dims,
                                         const conv_params_t *conv_params);

/**
 * Provide a user-owned scratch buffer (any alignment; aligned internally).
 * Pass NULL,0 to revert to internal malloc'd scratch.
 */
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
 *  ("1x1_simd", "im2col_simd", "c_ref"). Handy for benchmarks. */
const char *esp_nn_conv_s8_simd_last_path(void);

#ifdef __cplusplus
}
#endif

#endif /* ESP_NN_CONV_S8_SIMD_H */
