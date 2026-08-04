# Contributing to Synapedge

Thanks for your interest in improving Synapedge! This project is a model-agnostic
ONNX → C compiler for quantized int8 networks on the ESP32-S3. Contributions of new
operators, targets, bug fixes, and documentation are all welcome.

## Ground rules

- By submitting a contribution, you agree it is licensed under the **Apache License 2.0**.
- Be respectful and constructive — see the [Code of Conduct](CODE_OF_CONDUCT.md).
- Keep changes focused; one logical change per pull request.

## Development setup

```bash
git clone https://github.com/reuna-ai/synapedge.git
cd synapedge
pip install -r requirements.txt
```

## The golden rule: validate against onnxruntime

Synapedge's correctness contract is **node-by-node agreement with onnxruntime**. Any change
that affects generated numerics must be validated:

```bash
python tools/serial_compare.py --port <PORT> \
    --onnx path/to/model_q.onnx --image path/to/image.jpg --size 256
```

A PR that changes emitters, requantization, layout, or fusion should include a note on how
you validated it (which model, which image, and where device vs ORT agreed).

## Adding a new operator

1. Add an emitter in `nhwc_compiler/emitters/` (follow `op_float_ops.py` for elementwise ops
   or `op_conv.py` for compute-heavy ops).
2. If the op can receive a **quantized int8** input, it MUST either dequantize it
   (`_isq`/`_deq`) or the build will (intentionally) fail via the `_assert_not_quantized`
   guard. Silently reading int8 as float is the single most common correctness bug in this
   codebase — the guard exists to catch it.
3. Register the op with `@register("OpName")`.
4. Validate against onnxruntime and add a short note to the PR.

## Adding a new target

- Wire the target into `codegen.py`'s `--target` choices and route convs to the correct
  kernel (SIMD vs scalar).
- Reuse the hybrid arena in `gen_activations.py` where memory is constrained.

## Common pitfalls (learned the hard way)

- **Stale bytecode.** Clear `__pycache__` before regenerating, or set
  `PYTHONDONTWRITEBYTECODE=1`. Stale `.pyc` files cause header/body desync and
  "my change didn't apply" confusion.
- **Regenerate from ONNX.** Fusion and layout happen in early stages. Re-running only
  `codegen` on an already-transformed `graph_fused.json` will not pick up fusion/build-stage
  changes. Delete `graph_fused.json` and `model.*` first.
- **Negative zero-points.** When emitting dequant, render `x - zp` so a negative zp becomes
  `x - (-N)`, not `x--N` (which is a decrement and won't compile).

## Code style

- Python: keep emitters small and readable; prefer explicit over clever.
- Generated C: must compile clean under the Arduino/ESP-IDF toolchain with no warnings where
  practical.

## Pull request checklist

- [ ] Change is focused and described clearly.
- [ ] Numerics validated against onnxruntime (state the model/image).
- [ ] New quantized-input ops dequantize or trip the safety guard.
- [ ] Docs / CHANGELOG updated if behavior changed.
- [ ] `__pycache__` and generated artifacts are not committed (see `.gitignore`).
