## Summary
What does this PR change and why?

## Validation
- [ ] Numerics validated against onnxruntime (`serial_compare.py`)
- Model / image used:
- Where device and ORT agreed (node range):

## Checklist
- [ ] New quantized-input ops dequantize or trip the `_assert_not_quantized` guard
- [ ] Regenerated from ONNX (not codegen-only) when testing fusion/layout changes
- [ ] Docs / CHANGELOG updated if behavior changed
- [ ] No generated artifacts or `__pycache__` committed
