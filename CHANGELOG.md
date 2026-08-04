# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
