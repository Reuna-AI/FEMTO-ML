/*
 * esp_nn_depthwise_conv_s8_simd.h
 *
 * Single-file, self-contained int8 (s8) DEPTHWISE 2D convolution for
 * ESP32-S3, NHWC layout, TFLite-Micro compatible quantization.
 *
 * Companion to esp_nn_conv_s8_simd.h (regular convolution). This header
 * includes it to reuse the common types (data_dims_t, quant_data_t, ...),
 * so keep both .h/.c pairs together in your project.
 *
 * Paths selected automatically at runtime:
 *   1. ch_mult == 1 on ESP32-S3 -> the rows path: the few input rows each
 *      output row needs, as int16 with the padding in place, in the regular
 *      convs' fast buffer (internal SRAM), QACC 8-lane elementwise MAC, on
 *      both cores after esp_nn_conv_s8_simd_set_cores(2). Checked against the
 *      reference before its first use; never used if that check fails.
 *   2. ch_mult == 1 on ESP32-S3, the rows path unavailable -> the whole input
 *      converted to int16 once, in a scratch buffer (the old path)
 *   3. ch_mult > 1, or non-S3   -> plain C fallback (bit-exact)
 *
 * ---------------------------------------------------------------------------
 * USAGE
 * ---------------------------------------------------------------------------
 *   #include "esp_nn_depthwise_conv_s8_simd.h"
 *
 *   data_dims_t in_dims  = { 24, 24, 32, 1 };   // w, h, ch, batch
 *   data_dims_t filt_dims= { 3, 3, 32, 1 };     // w, h, ch, ch_mult
 *   data_dims_t out_dims = { 24, 24, 32, 1 };   // out_ch = ch * ch_mult
 *
 *   dw_conv_params_t params = {
 *       .in_offset  = 128,
 *       .out_offset = -128,
 *       .stride     = {1, 1},
 *       .padding    = {1, 1},
 *       .dilation   = {1, 1},          // must be 1,1
 *       .activation = {-128, 127},
 *       .ch_mult    = 1,
 *   };
 *   quant_data_t quant = { per_out_ch_shift, per_out_ch_mult };
 *
 *   int need = esp_nn_depthwise_conv_s8_simd_get_scratch_size(
 *                  &in_dims, &filt_dims, &out_dims, &params);
 *   esp_nn_depthwise_conv_s8_simd_set_scratch_buf(buf, buf_size); // optional
 *
 *   esp_nn_depthwise_conv_s8_simd(&in_dims, input, &filt_dims, filter,
 *                                 bias, &out_dims, output, &params, &quant);
 *
 * Data layout (int8 NHWC):
 *   input : [in_ht][in_wd][channels]
 *   filter: [filt_ht][filt_wd][channels * ch_mult]   <-- NOTE: different
 *           from regular conv! Depthwise filters are HWC(-mult), not OHWI.
 *   output: [out_ht][out_wd][channels * ch_mult]
 *   bias / mult / shift: int32[channels * ch_mult]
 *
 * Scratch note: the SIMD path converts the (spatially padded) input to
 * int16, so scratch is roughly 2 x padded input size + 2 x filter size.
 * For a 96x96x24 input with pad 1 that's ~460 KB -> use PSRAM scratch for
 * big layers, or let the internal allocator grab it.
 * ---------------------------------------------------------------------------
 */

#ifndef ESP_NN_DEPTHWISE_CONV_S8_SIMD_H
#define ESP_NN_DEPTHWISE_CONV_S8_SIMD_H

#include "esp_nn_conv_s8_simd.h"   /* shared types */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ESP_NN_SIMD_USE_ESP_NN_DEFS
typedef struct {
    int32_t in_offset;
    int32_t out_offset;
    int32_t ch_mult;            /* out_ch = in_ch * ch_mult */
    dims2_t stride;
    dims2_t padding;
    dims2_t dilation;
    activation_t activation;
} dw_conv_params_t;
#endif

/** Scratch bytes needed (0 if the C fallback will be used). */
int esp_nn_depthwise_conv_s8_simd_get_scratch_size(const data_dims_t *input_dims,
                                                   const data_dims_t *filter_dims,
                                                   const data_dims_t *output_dims,
                                                   const dw_conv_params_t *conv_params);

/** User scratch buffer (any alignment). NULL,0 reverts to internal malloc. */
void esp_nn_depthwise_conv_s8_simd_set_scratch_buf(void *buf, int size_bytes);

/** Run depthwise conv. Returns 0 on success, -1 on bad params / alloc fail. */
int esp_nn_depthwise_conv_s8_simd(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const dw_conv_params_t *conv_params,
                                  const quant_data_t *quant_data);

/** Bit-exact plain-C reference (validation / non-S3). */
void esp_nn_depthwise_conv_s8_ref(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const dw_conv_params_t *conv_params,
                                  const quant_data_t *quant_data);

/** Path used by last call: "dw_rows_simd_2core", "dw_rows_simd", "dw_s16_simd"
 *  or "dw_c_ref". */
const char *esp_nn_depthwise_conv_s8_simd_last_path(void);

/** The rows path's check against the reference, run now if not run yet:
 *  1 passed (the rows path is used), 0 failed (never used), -1 no rows path
 *  on this target. The first depthwise conv runs it if nothing has. */
int esp_nn_depthwise_conv_s8_simd_self_check(void);

#ifdef __cplusplus
}
#endif

#endif /* ESP_NN_DEPTHWISE_CONV_S8_SIMD_H */
