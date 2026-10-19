/*
 * test_conv_kernels.c -- the conv paths against the plain-C reference, bit for bit.
 *
 * On a PC (the S3 paths' loops, with plain-C stand-ins for the vector instructions):
 *   gcc -O2 -DESP_NN_S3_EMULATE -I examples/esp32_yolo26n tools/test_conv_kernels.c \
 *       examples/esp32_yolo26n/esp_nn_conv_s8_simd.c -o test_conv && ./test_conv
 *
 * On an ESP32-S3 (or Espressif's QEMU, which runs the S3 vector instructions), the
 * same file built into a sketch calls test_conv_kernels_main() and prints the result:
 * there it checks the real vector code.
 *
 * Shapes: every conv shape YOLO26n uses, plus odd ones (channels not a multiple of 8,
 * stride 2, padding, a misaligned input), each with a range of fast-buffer sizes so
 * that a layer runs in one block of output channels, in several, and in one channel
 * at a time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_nn_conv_s8_simd.h"

typedef struct { int in_w, in_h, in_c, k, stride, pad, out_c; } shape_t;

static const shape_t SHAPES[] = {
    /* YOLO26n at 256 x 256 */
    {256, 256, 3, 3, 2, 1, 16}, {128, 128, 16, 3, 2, 1, 32}, {64, 64, 16, 3, 1, 1, 8},
    {64, 64, 8, 3, 1, 1, 16}, {64, 64, 64, 3, 2, 1, 64}, {32, 32, 128, 3, 2, 1, 128},
    {16, 16, 128, 3, 2, 1, 256}, {32, 32, 256, 1, 1, 0, 64}, {64, 64, 48, 1, 1, 0, 64},
    {16, 16, 384, 1, 1, 0, 128}, {8, 8, 256, 1, 1, 0, 256}, {32, 32, 64, 3, 1, 1, 64},
    /* the same channel counts on small pictures, for a board without PSRAM */
    {8, 8, 64, 3, 2, 1, 64}, {8, 8, 128, 3, 2, 1, 128}, {6, 6, 128, 3, 2, 1, 256},
    {8, 8, 256, 1, 1, 0, 64}, {8, 8, 48, 1, 1, 0, 64}, {9, 9, 3, 3, 2, 1, 16},
    /* odd ones */
    {13, 11, 12, 1, 1, 0, 7}, {9, 7, 5, 3, 1, 1, 3}, {10, 10, 20, 3, 2, 1, 33},
    {7, 5, 24, 5, 1, 2, 9}, {6, 6, 8, 1, 1, 0, 1}, {5, 9, 40, 3, 2, 0, 17},
};

static const int FAST_SIZES[] = { 49152, 8192, 3000, 1400 };

static unsigned s_seed = 12345;
static int rnd(void) { s_seed = s_seed * 1103515245u + 12345u; return (int)((s_seed >> 16) & 0x7fff); }

static int run_one(const shape_t *s, int fast_size, int misalign)
{
    const int out_w = (s->in_w + 2 * s->pad - s->k) / s->stride + 1;
    const int out_h = (s->in_h + 2 * s->pad - s->k) / s->stride + 1;
    const size_t n_in = (size_t)s->in_w * s->in_h * s->in_c;
    const size_t n_f = (size_t)s->out_c * s->k * s->k * s->in_c;
    const size_t n_out = (size_t)out_w * out_h * s->out_c;

    int8_t *in_raw = malloc(n_in + 16), *in = in_raw + misalign;
    int8_t *filt = malloc(n_f), *a = malloc(n_out), *b = malloc(n_out);
    int32_t *bias = malloc(s->out_c * 4), *mult = malloc(s->out_c * 4), *shift = malloc(s->out_c * 4);
    void *fast = malloc(fast_size);

    if (!in_raw || !filt || !a || !b || !bias || !mult || !shift || !fast) {
        /* a board without PSRAM (QEMU) cannot hold the largest layers */
        printf("skip %3dx%-3d c%-3d k%d -> c%-3d  (no memory for it here)\n",
               s->in_w, s->in_h, s->in_c, s->k, s->out_c);
        free(in_raw); free(filt); free(a); free(b); free(bias); free(mult); free(shift); free(fast);
        return -1;
    }

    for (size_t i = 0; i < n_in; i++) in[i] = (int8_t)(rnd() & 0xff);
    for (size_t i = 0; i < n_f; i++) filt[i] = (int8_t)(rnd() & 0xff);
    for (int i = 0; i < s->out_c; i++) {
        bias[i] = (rnd() - 16384) * 4;
        mult[i] = 1073741824 + rnd() * 30000;
        shift[i] = -(6 + rnd() % 6);
    }

    data_dims_t id = { s->in_w, s->in_h, s->in_c, 1 };
    data_dims_t fd = { s->k, s->k, s->in_c, s->out_c };
    data_dims_t od = { out_w, out_h, s->out_c, 1 };
    /* in_offset is minus the input's zero point, so -127..128: the im2col path pads
     * with -in_offset, and -(-128) is not an int8. Drawn in order, one at a time (the
     * order an initializer's expressions are evaluated in is the compiler's choice). */
    const int32_t in_offset = rnd() % 256 - 127;
    const int32_t out_offset = rnd() % 256 - 128;
    conv_params_t cp = { .in_offset = in_offset, .out_offset = out_offset,
                         .stride = {s->stride, s->stride}, .padding = {s->pad, s->pad},
                         .dilation = {1, 1}, .activation = {-128, 127} };
    quant_data_t qd = { shift, mult };

    esp_nn_conv_s8_ref(&id, in, &fd, filt, bias, &od, a, &cp, &qd);
    esp_nn_conv_s8_simd_set_fast_buf(fast, fast_size);
    memset(b, 0x55, n_out);
    esp_nn_conv_s8_simd(&id, in, &fd, filt, bias, &od, b, &cp, &qd);

    size_t bad = 0;
    for (size_t i = 0; i < n_out; i++) bad += a[i] != b[i];
    printf("%s %3dx%-3d c%-3d k%d s%d p%d -> c%-3d  fast %5d  %s  %s\n",
           bad ? "FAIL" : "ok  ", s->in_w, s->in_h, s->in_c, s->k, s->stride, s->pad, s->out_c,
           fast_size, misalign ? "misaligned" : "aligned   ", esp_nn_conv_s8_simd_last_path());
    if (bad) {
        printf("     %zu of %zu outputs differ; the first:", bad, n_out);
        int shown = 0;
        for (size_t i = 0; i < n_out && shown < 6; i++)
            if (a[i] != b[i]) {
                printf(" [pixel %zu ch %zu: %d not %d]", i / s->out_c, i % s->out_c, b[i], a[i]);
                shown++;
            }
        printf("\n");
    }

    esp_nn_conv_s8_simd_set_fast_buf(NULL, 0);
    free(in_raw); free(filt); free(a); free(b); free(bias); free(mult); free(shift); free(fast);
    return bad ? 1 : 0;
}

int test_conv_kernels_main(void)
{
    int failed = 0, runs = 0;
    for (unsigned i = 0; i < sizeof(SHAPES) / sizeof(SHAPES[0]); i++) {
        for (unsigned j = 0; j < sizeof(FAST_SIZES) / sizeof(FAST_SIZES[0]); j++) {
            int r = run_one(&SHAPES[i], FAST_SIZES[j], (i + j) % 3 == 2 ? 3 : 0);
            if (r < 0) continue;
            failed += r;
            runs++;
        }
    }
    printf("\n%d of %d passed\n", runs - failed, runs);
    return failed;
}

#ifndef TEST_CONV_NO_MAIN
int main(void) { return test_conv_kernels_main() ? 1 : 0; }
#endif
