# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- ESP32-S3 convolutions run in blocks of output channels whose filters fit in a "fast
  buffer" of internal SRAM (`esp_nn_conv_s8_simd_set_fast_buf()`, else taken on the first
  conv). Before, the im2col path re-read a layer's whole filter set from PSRAM for every
  output pixel (and the 1x1 path from flash for every eight): any layer whose filters
  exceed the S3's 32 KB data cache -- most of YOLO26n's MACs -- ran at PSRAM speed.
  Bit-identical results (`tools/test_conv_kernels.c`: PC, and the S3 vector unit in QEMU).
  Measured on an ESP32-S3 N16R8 beside a camera's live view: 11 s -> 5.1 s a run.
- The S3's 1x1 QACC path is gone: every plain conv takes the im2col + ACCX path, a 1x1
  conv being an im2col conv one pixel wide. Reading QACC's 40-bit lanes back cost eight
  `memcpy` calls per eight outputs, and its MAC loop another per eight channels; the
  ACCX path takes 1.6x (48 channels in) to 3x (384 in) fewer instructions (counted in
  QEMU with `-icount`).
- The ACCX dot product loads its next operands in the instruction that multiplies the
  current ones; each block keeps its requant multipliers and split shifts beside its
  filters; the requant is inlined; the row loop keeps its job in registers.
- `esp_nn_conv_s8_simd_set_cores(2)`: a worker on the other core takes half of each
  block's output rows (its own im2col window, the block's filters shared).
- `fuse.py` `int8_layout_paths`: where every path from a float tensor leads, through
  Split / Concat / Reshape / Transpose / MaxPool / nearest Resize only, to
  QuantizeLinears of one scale, the producer (SiLU, Add, DequantizeLinear) quantizes as
  it writes, the layout ops move int8, and those QuantizeLinears go. Bit-identical: the
  producer quantizes with the QuantizeLinear's own C expression (`quant_c`), and its
  tables are filled on the device by that expression.
- `fuse.py` `sigmoid_through_layout`: YOLO26n's final class Sigmoid (107,520 `expf`s,
  0.15 s on the S3) is a 256-entry table in each of its three DequantizeLinears.
- `FUSE_SILU_QUANTIZE` on by default, where the Quantize is the SiLU's only reader.
- SiLU becomes a table lookup wherever the graph has Dequantize -> Sigmoid -> Mul, not
  only where a QuantizeLinear follows: removes YOLO26n's 38 float `expf` Sigmoids.
- QuantizeLinear multiplies by 1/scale and rounds inline instead of a float divide and
  `lroundf` per element.
- TopK is one pass with a K-entry heap instead of K passes (YOLO26n: 300 x 24000);
  same order, ties to the lower index.

### Added
- `FEMTO_PROFILE=1`: the generated `run_model()` times every node (`femto_profile.c`) and
  `femto_prof_report()` gives the time per op type and the slowest nodes.
- `tools/test_conv_kernels.c`: the conv paths against the C reference, bit for bit, over
  every YOLO26n conv shape and odd ones, with fast buffers from one block to many.
- Depthwise convolution support (scalar kernel + esp-nn depthwise SIMD library integration).
- TFLite-style integer requantization across all convolution paths.
- Hybrid fast-RAM / large-RAM (e.g. SRAM/PSRAM) arena placement, configurable via
  `SRAM_BUDGET_BYTES` and `SRAM_PRIORITY`. An `esp32s2` arena target is accepted but only
  partially wired (arena placement done; not yet a full codegen target).
- SiLU + Quantize fusion as a toggleable build directive (`FUSE_SILU_QUANTIZE`).
- Compile-time safety guard that rejects quantized int8 inputs to ops without dequant handling.
- Validation tools: `serial_compare.py` (device vs onnxruntime) and
  `compare_fuse_modes.py` (fused vs unfused device runs).

### Fixed
- The S3 im2col path pads with `-in_offset` as an int8, wrong for `in_offset == -128`
  (an input zero point of 128, which no int8 tensor has, but a uint8 one could); such a
  conv now takes the C reference.
- Class scores could never exceed 0.5 (the open question in the docs: "~0.5 confidence").
  The class-logit convs were calibrated on mostly-background anchors, so their int8 range
  ended at 0 (zero point 127) and every positive logit was clamped to 0: sigmoid(0) = 0.5
  for every confident detection, and TopK could not rank them. `fuse.py` now moves such a
  conv's zero point to 0 when its output reaches a Sigmoid (`widen_logit_range`): the
  scale and multipliers are unchanged, scores below 0.5 are unchanged, and above it they
  are real again (a clear person ~0.88 rather than 0.500).
- The float SiLU fusion dropped its DequantizeLinear without checking that nothing else
  read it.
- Depthwise weights were run through the generic OHWI transpose, transposing the kernel
  spatially (kh <-> kw); now repacked explicitly to [Kh,Kw,K].
- Quantized residual `Add`/`Concat` inputs were not being dequantized (raw int8 added to
  float), corrupting residual blocks throughout the backbone; int8 activation outputs now
  carry `q_scale`/`q_zp` so dequant-aware ops handle them correctly.
- Negative zero-points rendered as `x--N` (a decrement) in dequant expressions; now emitted
  as `x - (-N)`.
- Integer requant right-shift used `31 - shift`, zeroing outputs; corrected to `-shift`.

## [0.1.0] - 2026
- Initial validated pipeline on YOLOv5n (CNN) and YOLO26n (attention) for ESP32-S3.
