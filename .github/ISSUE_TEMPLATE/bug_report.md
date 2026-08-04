---
name: Bug report
about: Report incorrect generated code, a crash, or a device-vs-ORT mismatch
title: "[BUG] "
labels: bug
---

**Describe the bug**
A clear description of what went wrong.

**Model / target**
- Model (and input size):
- Target: `esp32s3` / `esp32s2` / `ansi_int8` / `ansi_float`
- `--simd`: yes / no
- `FUSE_SILU_QUANTIZE`: true / false

**To reproduce**
Command(s) used to compile, and how you ran it on-device.

**Expected vs actual**
What you expected (e.g. onnxruntime output) vs what the device produced.

**Validation output**
If possible, paste the first divergence from `serial_compare.py` (device vs ORT).

**Environment**
- OS:
- Python version:
- onnx / onnxruntime versions:
