
<p align="center">
  <img width="518" height="207" alt="FEMTO_01" src="https://github.com/user-attachments/assets/0b7713e3-c92e-4a48-bbed-b71f5f35d27d" />
</p>
A model-agnostic ONNX → C compiler for quantized int8 neural networks on microcontrollers.

FEMTO-ML turns a quantized int8 ONNX model into portable, standalone C that runs on
resource-constrained microcontrollers. It is model-agnostic — operators are handled as
first-class emitters rather than hard-coded model graphs — and target-agnostic: the same
model compiles to portable scalar C for any MCU, or to a hardware-accelerated backend where
one exists.

It has been validated on both a pure CNN (**YOLOv5n**) and a transformer-attention detector
(**YOLO26n**), producing correct on-device detections verified node-by-node against
onnxruntime.

<!-- Replace with your own badge URLs / CI once set up -->
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Features

- **Model-agnostic** — per-op C emitters (conv, depthwise conv, SiLU, Add, Mul, Concat,
  Split, Slice, Softmax, MatMul, and the attention / NMS tail), not a fixed model template.
- **Portable by default** — emits plain C that compiles for any microcontroller with a C
  toolchain; hardware acceleration is an opt-in backend, not a requirement.
- **Quantized int8** — TFLite-style integer requantization across all conv paths.
- **NHWC layout** — activations are laid out for efficient per-op access.
- **Transformer support** — handles attention blocks (MatMul + Softmax), depthwise
  positional-encoding convs, and embedded end-to-end NMS.
- **Union arena** — activation memory is packed via interval-graph coloring, with an optional
  **hybrid fast-RAM / large-RAM** placement for parts that expose two memory tiers
  (e.g. internal SRAM + PSRAM).
- **Optional fusions** — SiLU + Quantize fusion (toggleable build directive).
- **Validation tooling** — scripts to capture device dumps and diff them against onnxruntime.

---

## Targets and portability

FEMTO-ML is designed to run models on **any microcontroller**. The compiler always produces
portable C; where a target exposes SIMD or a vendor kernel library, FEMTO-ML can route the
heavy operators (convolutions) to it, falling back to scalar C everywhere else.

`--target` selects the code path for the compute-heavy operators:

| Target        | Convolutions           | Status                                                |
|---------------|------------------------|-------------------------------------------------------|
| `ansi_float`  | float (portable)       | Reference / debugging (default)                       |
| `ansi_int8`   | scalar int8 (portable) | **Portable baseline** — any MCU with a C toolchain    |
| `esp32s3`     | esp-nn SIMD + scalar   | Hardware-accelerated backend (validated)              |

> The portable `ansi_int8` target is the path for bringing up a new microcontroller: it needs
> only a standard C toolchain. Adding a new accelerated backend means teaching `codegen.py`
> and the conv emitter to call that platform's kernels — see
> [CONTRIBUTING.md](CONTRIBUTING.md).

The ESP32-S3 is currently the reference hardware-accelerated backend (esp-nn SIMD, PSRAM).
It is one example of what the architecture supports, not the limit of it — the portable
scalar path is the primary route to any other microcontroller.

**Experimental:** the hybrid fast-RAM / large-RAM arena in `gen_activations.py` also accepts
an `esp32s2` target for RAM-constrained parts that expose two memory tiers (internal SRAM +
PSRAM). This is **partially wired** — the arena placement is implemented, but `esp32s2` is
not yet a full `codegen` target (convolutions would use the scalar path). Treat it as a
work-in-progress, not a finished backend.

---

## Repository layout

```
.
├── nhwc_compiler/            # the compiler
│   ├── build_model.py        # orchestrator — runs the full pipeline (entry point)
│   ├── graph_json.py         # stage 1-2: ONNX -> JSON IR, NCHW -> NHWC
│   ├── fuse.py               # stage 3: graph-level fusion (SiLU, Q/DQ)
│   ├── gen_weights.py        # stage 4: weight emission (OHWI / depthwise repack)
│   ├── gen_activations.py    # stage 5: arena allocation (interval coloring, hybrid RAM)
│   ├── codegen.py            # stage 6: dispatcher / C emitter (per-target)
│   ├── emitters/             # per-op C generators
│   │   ├── op_conv.py        #   conv, depthwise, integer requant
│   │   ├── op_silu.py        #   SiLU LUT (int8 or float)
│   │   ├── op_float_ops.py   #   Add/Mul/Concat/Softmax/MatMul/... (dequant-aware)
│   │   └── common.py         #   shared emitter helpers
│   ├── compare_ort.py        # validation: run ONNX in onnxruntime, dump per-node values
│   └── viz.py                # visualize the JSON IR as a Graphviz dataflow graph
├── tools/                    # device-side validation utilities
│   ├── serial_compare.py     # device dump vs onnxruntime, node-by-node
│   └── compare_fuse_modes.py # fused vs unfused device runs
├── docs/                     # architecture & bug write-ups (with diagrams)
├── examples/                 # sample sketches / usage
├── LICENSE                   # Apache-2.0
├── NOTICE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
└── README.md
```

> The orchestrator entry point is `nhwc_compiler/build_model.py`. Adapt paths to your layout.

---

## Quick start

### Prerequisites
- Python 3.10+
- `onnx`, `onnxruntime`, `numpy` (for the compiler and validation tools)
- A C toolchain for your target microcontroller
- For an accelerated backend (e.g. ESP32-S3): the vendor kernel sources for that platform —
  see `examples/`.

### Install
```bash
git clone https://github.com/reuna-ai/FEMTO-ML.git
cd FEMTO-ML
pip install -r requirements.txt
```

### Compile a model
```bash
# Portable baseline (any MCU): scalar int8 C
python nhwc_compiler/build_model.py path/to/model_q.onnx path/to/project/ \
    --target ansi_int8

# Accelerated backend example (ESP32-S3 with esp-nn SIMD)
python nhwc_compiler/build_model.py path/to/model_q.onnx path/to/sketch/ \
    --target esp32s3 --simd
```

This writes `model.c`, `model.h`, weight files, and `activations.*` into your project folder.
Build and flash with your platform's toolchain. For an accelerated backend, add that
platform's kernel sources to the build (see `examples/`).

> **Important:** fusion and layout decisions happen in early stages. If you change a build
> directive (e.g. `FUSE_SILU_QUANTIZE`), regenerate **from ONNX** — do not re-run codegen on
> an already-transformed IR. Delete `graph_fused.json` and `model.*`, and clear any
> `__pycache__` folders first.

### Command-line flags (`build_model.py`)

```
python nhwc_compiler/build_model.py <model_q.onnx> <outdir/> [flags]
```

| Flag             | Values / type                               | Default       | Meaning                                                          |
|------------------|---------------------------------------------|---------------|------------------------------------------------------------------|
| `onnx`           | path (positional)                           | —             | Input quantized ONNX model                                       |
| `outdir`         | path (positional)                           | —             | Output folder for the generated C project (created if missing)   |
| `--target`       | `ansi_float` \| `ansi_int8` \| `esp32s3`    | `ansi_float`  | Code path for the compute-heavy ops (see Targets table)          |
| `--simd`         | flag                                        | off           | Emit ESP32-S3 vector (ACCX) convolutions; use with `--target esp32s3` |
| `--budget`       | int (bytes)                                 | 5,000,000     | Max bytes per generated `weights_*.c` file (splits large weights)|
| `--debug`        | flag                                        | off           | Emit per-node trace prints on device                             |
| `--name`         | string                                      | `model`       | Basename for the generated `model.c` / `model.h`                 |
| `--keep-json`    | flag                                        | off           | Keep intermediate `.json` IR files instead of cleaning them up   |
| `--prefix`       | C identifier, e.g. `face_`                  | none          | Prefix the files and every name the model makes visible, so two models can be built into one firmware |

The `codegen.py` stage (invoked by the orchestrator, or runnable directly) additionally
supports device‑validation flags: `--dump` (per-node activation dump over serial),
`--dump-n N` (values per channel to dump), `--dump-nodes NAME...` (limit the dump to matching
node names), and `--esp-nn` (route eligible convs to the `esp_nn_conv_s8_simd` library on
ESP32-S3). Use `--dump` only for validation runs — the serial prints dominate runtime.

### Build-time directives (source constants)

These are edited at the top of the relevant source file, not passed on the command line —
they affect graph transformation and arena layout, so they take effect only on a full
regeneration from ONNX:

| Directive             | File                 | Default | Meaning                                             |
|-----------------------|----------------------|---------|-----------------------------------------------------|
| `FUSE_SILU_QUANTIZE`  | `fuse.py`            | `True`  | Fold post-SiLU Quantize into an int8 SiLU LUT       |
| `INT8_LAYOUT`         | `fuse.py`            | `True`  | Quantize where a value is made, move int8 after     |
| `LAZY_FLOATS`         | `fuse.py`            | `True`  | Make table-derived floats again where read, unstored|
| `SRAM_BUDGET_BYTES`   | `gen_activations.py` | 128 KiB | Fast-RAM budget for the hybrid arena (experimental) |
| `SRAM_PRIORITY`       | `gen_activations.py` | `"hot"` | `"hot"` (most-accessed) or `"large"` (biggest-first)|

One switch is set when the generated C is compiled, not when it is generated:
`-DFEMTO_NO_FLOAT_INPUT` leaves the model's float input unallocated, for a caller that writes
the int8 input itself (`model_input_int8`, then `run_model_int8_input()`); `run_model()` is then
not defined, and a model without an int8 input entry refuses to build with it.

---

## Validation

FEMTO-ML's correctness contract is **node-by-node agreement with onnxruntime**. Three tools
support this:

**Device vs onnxruntime** — capture the device's per-node dump over serial and diff it against
ORT on the same image, auto-aligned by node name (reports the first divergence):
```bash
python tools/serial_compare.py --port <PORT> \
    --onnx path/to/model_q.onnx --image path/to/image.jpg --size 256
```

**onnxruntime reference dump** — run the original NCHW quantized ONNX in onnxruntime, exposing
intermediate outputs and printing per-channel values in the *same format* as the device dump,
so you can diff by eye or feed a golden reference:
```bash
python nhwc_compiler/compare_ort.py path/to/model_q.onnx --image path/to/image.jpg
```

**Fused vs unfused builds** — prove the SiLU+Quantize fusion is numerically equivalent (flash
each build, capture, then diff):
```bash
python tools/compare_fuse_modes.py capture --mode false --port <PORT> --out dump_false.txt
python tools/compare_fuse_modes.py capture --mode true  --port <PORT> --out dump_true.txt
python tools/compare_fuse_modes.py compare --true dump_true.txt --false dump_false.txt
```

## Testing on a PC

No board needed: the generated C is built with gcc and run (CI runs all of these):
```bash
python tools/test_golden.py      # YOLO26n gives exactly its recorded outputs (esp32s3 and ansi_int8)
FEMTO_CFLAGS=-DESP_NN_S3_EMULATE python tools/test_golden.py   # the same through the S3 paths' loops
python tools/test_ops.py         # small models of other shapes against onnxruntime
python tools/test_two_models.py  # two models in one program (--prefix); a model fed int8 only
python tools/femto_bench.py examples/esp32_yolo26n/yolo26n_q.onnx   # memory and traffic scorecard
```
The S3 kernels' own loops, bit for bit against their plain-C references, with plain-C
stand-ins for the vector instructions (`ESP_NN_S3_EMULATE`):
```bash
gcc -O2 -DESP_NN_S3_EMULATE -I examples/esp32_yolo26n tools/test_conv_kernels.c examples/esp32_yolo26n/esp_nn_conv_s8_simd.c -lm -o test_conv && ./test_conv
gcc -O2 -DESP_NN_S3_EMULATE -I examples/esp32_yolo26n tools/test_dw_kernels.c examples/esp32_yolo26n/esp_nn_depthwise_conv_s8_simd.c examples/esp32_yolo26n/esp_nn_conv_s8_simd.c -lm -o test_dw && ./test_dw
```
A compiler change that should not change results must leave `test_golden.py` passing. One
that should (a better rounding, say) is checked against onnxruntime first, then recorded
with `python tools/test_golden.py --update` and explained in CHANGELOG.md.

## Inspecting the graph

`viz.py` renders the compiler's JSON IR as a Graphviz dataflow graph — useful for
understanding how a model was transformed (fused SiLUs, quantize boundaries, layout):
```bash
python nhwc_compiler/viz.py graph_fused.json graph.dot
dot -Tsvg graph.dot -o graph.svg          # requires Graphviz installed
# or paste graph.dot into an online Graphviz viewer
```

---

## Performance notes

Inference on a detection model is dominated by convolution compute (~98% of MACs), so the
biggest win comes from routing those convolutions to a hardware SIMD backend.

Measured on **ESP32-S3** (validated targets):

| Model    | Scalar (`ansi_int8` path) | SIMD (`esp32s3 --simd`) | Speed-up |
|----------|---------------------------|-------------------------|----------|
| YOLOv5n  | ~24 s                     | ~5 s                    | ~5×      |

The SIMD path is only as fast as its operands arrive: on the S3, PSRAM and flash are read
through a 32 KB data cache, so the convs run in blocks of output channels whose filters fit in
a buffer of internal SRAM (`esp_nn_conv_s8_simd_set_fast_buf()`), instead of streaming a
layer's filters from PSRAM for every output pixel. Depthwise convs use the same buffer between
convs for the few int16 input rows each output row needs, rather than an int16 copy of their
whole input. Build with `-D FEMTO_PROFILE=1` and call
`femto_prof_report()` to see where each run's time goes, node by node. Input resolution
(e.g. 192×192 or 160×160) cuts MACs quadratically on top, as a model re-export rather than a
compiler change. See [`docs/`](docs/) for the full breakdown.

---

## Documentation

- [`docs/SYNAPEDGE_DOCUMENTATION.md`](docs/FEMTO-ML_DOCUMENTATION.md) — architecture,
  file-by-file reference, and detailed write-ups of the correctness bugs found and fixed
  (with diagrams).

## Contributing

Contributions are welcome — new operators, new microcontroller backends, and fixes alike.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
By contributing you agree that your contributions are licensed under the Apache License 2.0.

A good first step when adding a new operator or target: run `serial_compare.py` against
onnxruntime and confirm node-by-node agreement before opening a PR.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgements

- The ONNX and onnxruntime projects, used as the reference for validation.
- Espressif's **esp-nn** SIMD kernels, used by the ESP32-S3 accelerated backend.
- YOLOv5 and YOLO26 model architectures used as validation targets (subject to their own
  respective licenses — see their upstream projects).
