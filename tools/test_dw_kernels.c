/*
 * test_dw_kernels.c -- the depthwise conv paths against the plain-C reference, bit for bit.
 *
 * On a PC (the rows path's loops, with plain-C stand-ins for the vector instructions
 * that store QACC's lanes byte for byte as the S3 does):
 *   gcc -O2 -DESP_NN_S3_EMULATE -I examples/esp32_yolo26n tools/test_dw_kernels.c \
 *       examples/esp32_yolo26n/esp_nn_depthwise_conv_s8_simd.c \
 *       examples/esp32_yolo26n/esp_nn_conv_s8_simd.c -lm -o test_dw && ./test_dw
 *
 * On an ESP32-S3 the same file built into a sketch calls test_dw_kernels_main() and
 * prints the result: there it checks the real vector code.
 *
 * Shapes: every depthwise layer YOLO26n and the YuNet face finder use, plus odd ones
 * (channel counts not a multiple of 8, one channel, stride 2, no padding, 5x5, 1x3,
 * 3x1 and 1x1 filters, an output narrower than the padding), each with a range of
 * fast-buffer sizes -- all channels at once, passes of 8, and too small for even 8
 * (the reference then runs) -- on one core and on two, aligned and not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_nn_depthwise_conv_s8_simd.h"

typedef struct { int in_w, in_h, ch, k_w, k_h, stride, pad; } shape_t;

static const shape_t SHAPES[] = {
    /* YOLO26n at 256 x 256 */
    {8, 8, 128, 3, 3, 1, 1}, {32, 32, 64, 3, 3, 1, 1}, {32, 32, 80, 3, 3, 1, 1},
    {16, 16, 128, 3, 3, 1, 1}, {16, 16, 80, 3, 3, 1, 1}, {8, 8, 256, 3, 3, 1, 1},
    {8, 8, 80, 3, 3, 1, 1},
    /* the YuNet face finder at 320 x 256 */
    {160, 128, 16, 3, 3, 1, 1}, {80, 64, 16, 3, 3, 1, 1}, {80, 64, 32, 3, 3, 1, 1},
    {80, 64, 64, 3, 3, 1, 1}, {40, 32, 64, 3, 3, 1, 1}, {20, 16, 64, 3, 3, 1, 1},
    {10, 8, 64, 3, 3, 1, 1}, {10, 8, 4, 3, 3, 1, 1}, {10, 8, 10, 3, 3, 1, 1},
    {20, 16, 4, 3, 3, 1, 1}, {20, 16, 10, 3, 3, 1, 1}, {40, 32, 4, 3, 3, 1, 1},
    {40, 32, 10, 3, 3, 1, 1},
    /* odd ones */
    {9, 7, 20, 3, 3, 1, 1}, {10, 10, 24, 3, 3, 2, 1}, {11, 9, 13, 3, 3, 2, 1},
    {7, 6, 3, 5, 5, 1, 2}, {6, 5, 40, 3, 3, 1, 0}, {5, 4, 33, 1, 1, 1, 0},
    {12, 3, 8, 3, 1, 1, 1}, {3, 12, 9, 1, 3, 1, 1}, {1, 1, 17, 3, 3, 1, 1},
    {2, 2, 1, 3, 3, 1, 1}, {13, 5, 7, 5, 5, 2, 2}, {4, 4, 16, 3, 3, 3, 1},
};

static const int FAST_SIZES[] = { 49152, 16384, 4096, 900 };
static const int CORES[] = { 1, 2 };

static unsigned s_seed = 54321;
static int rnd(void) { s_seed = s_seed * 1103515245u + 12345u; return (int)((s_seed >> 16) & 0x7fff); }

static int s_paths_rows = 0, s_paths_other = 0;

static int run_one(const shape_t *s, int fast_size, int misalign, int cores)
{
    const int out_w = (s->in_w + 2 * s->pad - s->k_w) / s->stride + 1;
    const int out_h = (s->in_h + 2 * s->pad - s->k_h) / s->stride + 1;
    const size_t n_in = (size_t)s->in_w * s->in_h * s->ch;
    const size_t n_f = (size_t)s->k_w * s->k_h * s->ch;
    const size_t n_out = (size_t)out_w * out_h * s->ch;

    int8_t *in_raw = malloc(n_in + 16), *in = in_raw + misalign;
    int8_t *filt = malloc(n_f), *a = malloc(n_out), *b_raw = malloc(n_out + 16), *b = b_raw + misalign;
    int32_t *bias = malloc(s->ch * 4), *mult = malloc(s->ch * 4), *shift = malloc(s->ch * 4);
    void *fast_raw = malloc(fast_size + 16);

    if (!in_raw || !filt || !a || !b_raw || !bias || !mult || !shift || !fast_raw) {
        printf("skip %3dx%-3d c%-3d k%dx%d  (no memory for it here)\n", s->in_w, s->in_h, s->ch, s->k_w, s->k_h);
        free(in_raw); free(filt); free(a); free(b_raw); free(bias); free(mult); free(shift); free(fast_raw);
        return -1;
    }

    for (size_t i = 0; i < n_in; i++) in[i] = (int8_t)(rnd() & 0xff);
    for (size_t i = 0; i < n_f; i++) filt[i] = (int8_t)(rnd() & 0xff);
    for (int i = 0; i < s->ch; i++) {
        bias[i] = (rnd() - 16384) * 4;
        mult[i] = 1073741824 + rnd() * 30000;
        shift[i] = (i % 7 == 3) ? 1 : -(4 + rnd() % 8);
    }

    data_dims_t id = { s->in_w, s->in_h, s->ch, 1 };
    data_dims_t fd = { s->k_w, s->k_h, s->ch, 1 };
    data_dims_t od = { out_w, out_h, s->ch, 1 };
    /* drawn one at a time: the order an initializer's expressions run in is the compiler's */
    const int32_t in_offset = rnd() % 256 - 127;
    const int32_t out_offset = rnd() % 256 - 128;
    const int32_t act_min = (rnd() % 4 == 0) ? -100 : -128;
    const int32_t act_max = (rnd() % 4 == 0) ? 90 : 127;
    dw_conv_params_t cp = { .in_offset = in_offset, .out_offset = out_offset, .ch_mult = 1,
                            .stride = {s->stride, s->stride}, .padding = {s->pad, s->pad},
                            .dilation = {1, 1}, .activation = {act_min, act_max} };
    quant_data_t qd = { shift, mult };

    esp_nn_depthwise_conv_s8_ref(&id, in, &fd, filt, bias, &od, a, &cp, &qd);
    /* every other run with an output table (an activation folded in): the reference's
     * bytes through it are what the kernel must store */
    static int8_t lut[256];
    const int with_lut = (rnd() & 1);
    if (with_lut) {
        for (int q = 0; q < 256; q++) lut[q] = (int8_t)(rnd() & 0xff);
        for (size_t q = 0; q < n_out; q++) a[q] = lut[(uint8_t)(a[q] + 128)];
    }
    /* the fast buffer at an odd address too: the path aligns what it lays out in it */
    esp_nn_conv_s8_simd_set_fast_buf((int8_t *)fast_raw + (misalign ? 5 : 0), fast_size);
    esp_nn_conv_s8_simd_set_cores(cores);
    memset(b, 0x55, n_out);
    esp_nn_conv_s8_simd_set_out_lut(with_lut ? lut : NULL);
    esp_nn_depthwise_conv_s8_simd(&id, in, &fd, filt, bias, &od, b, &cp, &qd);
    esp_nn_conv_s8_simd_set_out_lut(NULL);

    const char *path = esp_nn_depthwise_conv_s8_simd_last_path();
    if (strncmp(path, "dw_rows", 7) == 0) s_paths_rows++; else s_paths_other++;

    size_t bad = 0;
    for (size_t i = 0; i < n_out; i++) bad += a[i] != b[i];
    printf("%s %3dx%-3d c%-3d k%dx%d s%d p%d  fast %5d  %s %s  %s\n",
           bad ? "FAIL" : "ok  ", s->in_w, s->in_h, s->ch, s->k_w, s->k_h, s->stride, s->pad,
           fast_size, misalign ? "misaligned" : "aligned   ", with_lut ? "lut" : "   ", path);
    if (bad) {
        printf("     %zu of %zu outputs differ; the first:", bad, n_out);
        int shown = 0;
        for (size_t i = 0; i < n_out && shown < 6; i++)
            if (a[i] != b[i]) {
                printf(" [pixel %zu ch %zu: %d not %d]", i / s->ch, i % s->ch, b[i], a[i]);
                shown++;
            }
        printf("\n");
    }

    esp_nn_conv_s8_simd_set_fast_buf(NULL, 0);
    free(in_raw); free(filt); free(a); free(b_raw); free(bias); free(mult); free(shift); free(fast_raw);
    return bad ? 1 : 0;
}

int test_dw_kernels_main(void)
{
    int failed = 0, runs = 0;
    const int check = esp_nn_depthwise_conv_s8_simd_self_check();
    printf("self-check: %d\n", check);

    for (unsigned i = 0; i < sizeof(SHAPES) / sizeof(SHAPES[0]); i++) {
        for (unsigned j = 0; j < sizeof(FAST_SIZES) / sizeof(FAST_SIZES[0]); j++) {
            for (unsigned c = 0; c < sizeof(CORES) / sizeof(CORES[0]); c++) {
                int r = run_one(&SHAPES[i], FAST_SIZES[j], (i + j + c) % 3 == 2 ? 3 : 0, CORES[c]);
                if (r < 0) continue;
                failed += r;
                runs++;
            }
        }
    }
    esp_nn_conv_s8_simd_set_cores(1);
    printf("\n%d of %d passed (%d by the rows path, %d by another)\n",
           runs - failed, runs, s_paths_rows, s_paths_other);
#if defined(ESP_NN_S3_EMULATE)
    /* the emulated build must have checked the rows path, not fallen back everywhere */
    if (check != 1 || s_paths_rows < runs / 2) {
        printf("FAIL the rows path was not used (self-check %d, %d runs by it)\n", check, s_paths_rows);
        failed++;
    }
#endif
    return failed;
}

#ifndef TEST_DW_NO_MAIN
int main(void) { return test_dw_kernels_main() ? 1 : 0; }
#endif
