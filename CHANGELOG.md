# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (fourth round, October 2026)
- Depthwise convs on the ESP32-S3 (`esp_nn_depthwise_conv_s8_simd.c`). Measured on the board,
  every depthwise layer of YOLO26n and of a YuNet face finder ran at about 11 million MACs a
  second -- slower than the regular convs beside them, and in YuNet, which is mostly depthwise
  and pointwise, they were over half its time. The old S3 path converted the whole padded
  input to int16 first (674 KB for a 160x128x16 layer: in PSRAM if it could be had, and the
  plain-C reference if not) and read QACC's lanes back with unaligned 32-bit loads. The new
  rows path keeps only the k_h input rows each output row needs, as int16 with the padding in
  place, in the regular convs' fast buffer (internal SRAM, borrowed between convs), each row
  converted once while later rows still read it; channels go in passes of a multiple of 8 when
  all of them do not fit; QACC is read back a byte at a time; the requant is the convs'
  `requant_fast()`; and with `esp_nn_conv_s8_simd_set_cores(2)` the other core takes half the
  output rows. Before its first use it checks itself against the reference on five small
  layers and, if it ever disagrees, is never used (the old paths run as before);
  `esp_nn_depthwise_conv_s8_simd_self_check()` says which.
- Measured on the board (ESP32-S3, 23 October 2026; the check passed): the YuNet face finder,
  as the JARVIS house cam times it in normal use, 1838 -> 1319 ms (1012 ms with nothing else
  talking to the board, as the other core is then free); YOLO26n about 2.2 -> 2.0 s, its
  depthwise layers (68 and 48 ms the slowest before) now under 28 ms each.
- The convs' second-core worker runs any function handed to it (`esp_nn_simd_both_cores()`),
  so the depthwise conv shares it; `esp_nn_conv_s8_simd_cores()` and
  `esp_nn_conv_s8_simd_borrow_fast_buf()` are new.
- Checked: `tools/test_dw_kernels.c` (new) runs every depthwise shape of both models and odd
  ones (channel counts not a multiple of 8, stride 2 and 3, 5x5, 1x3, 3x1 and 1x1 filters, no
  padding, an output narrower than the padding) with four fast-buffer sizes, one core and two,
  aligned and not: bit-exact against the reference, with QACC's lanes stored byte for byte as
  the S3 stores them. YOLO26n's golden outputs are unchanged on both targets, also built with
  `ESP_NN_S3_EMULATE` (CI runs that too), and a YuNet face finder gives the same bytes either way.

### Added (third round, October 2026)
- `--prefix` (`nhwc_compiler/prefix.py`): a model's files and every name it makes visible
  (functions, weights, arena unions and pointers, tensor macros, `MODEL_INPUT_*`) get a prefix,
  so two models build into one firmware -- a person detector and a face finder, say. The names
  are read from the generated headers, not from a list. `tools/test_two_models.py` builds two
  models into one program, one source including both headers: each gives exactly the bytes it
  gives alone; without the prefix the same program does not link.
- `FEMTO_NO_FLOAT_INPUT`, defined when the C is compiled: the float input buffer is not
  allocated (786 KB for YOLO26n at 256x256) for a caller that writes the int8 input itself,
  and `run_model()` is left out so it cannot be called by mistake.
- A `Relu` emitter: a model quantized with only its convs quantized keeps float ReLUs between
  them (YuNet's backbone, quantized that way, stopped at "no emitter for Relu").

### Fixed (third round)
- A conv with one output channel did not build: its bias, a single value, was inlined as a
  constant while the generated C still named a bias array that was never written (YuNet's
  face and object scores: "undeclared w_..._bias_quantized"). A conv's bias is now always a
  weight array.
- `test_ops.py` has the two cases that found these: `one_channel` and `relu`.

Checked with YuNet (OpenCV's face detector, MIT), quantized to int8 at 320x256 and compiled for
esp32s3: on 186 house-camera pictures it was not tuned on, the faces found match onnxruntime's
int8 model in 185 and the float model in 184, scores within 0.03; arena 0.9 MB. YOLO26n's
golden outputs are unchanged on both targets.

### Changed (second round, October 2026)
Measured with `tools/femto_bench.py` on YOLO26n (esp32s3 target), the first round's build -> this one:
working memory 4.94 -> 3.94 MB, memory traffic per run (estimate) 46.2 -> 24.3 MB,
float tensors stored 17.9 -> 7.7 MB, ops 334 -> 268. The model's output is the same to
the bit as the first round's, built and run on a PC (`tools/test_golden.py`: three
inputs, both targets), with the compiler's multiply-add fusion off and on
(`FEMTO_CFLAGS="-mfma -ffp-contract=fast"`, which is how the board's compiler builds);
the same for YOLOv5n quantized from Ultralytics' v7.0 release. On the board itself, and
its run time, still to be measured with the s3profile build.
- Views cost nothing: a Reshape, Flatten, Unsqueeze, Squeeze or an identity Sigmoid
  gets its output at its input's address (`gen_activations.py`), and its copy is skipped
  (`if(y!=x)`). A Concat that only copies, along an axis with nothing but 1s before
  it, has each input written straight into its slab of the output -- unless one of its
  inputs (or a view of it) is in another such Concat, where the two Concats' other
  parts would land in the same place. 17 views and 2 Concats in YOLO26n.
- `fuse.py` `cancel_layout_transposes`: YOLO26n's head takes each scale's NHWC class
  scores to NCHW, concatenates them, and transposes the lot back before splitting it.
  The two transposes undo each other, so both go, with the Concat and Split between
  them: the class scores land where TopK reads them with no copy at all. Before, the
  [1024][80] transpose alone took about 82 ms, its 4 KB stride likely fighting the S3's
  data cache. It also takes a Split that cuts a Concat back into the very pieces it was
  made of to be those pieces. Model-agnostic: it follows op types and shapes only.
- `fuse.py` `mark_lazy_floats`: a float tensor that is only a SiLU table entry, a
  dequantized byte, a channel slice, a nearest upsample or the sum of two of those is
  no longer stored. The ops that read it (channel Split, same-shape Add, channel
  Concat) make it again from the int8 it came from, with its producer's own C
  expression, so the same float. A dequantized byte comes from a 256-entry table the
  board fills with the DequantizeLinear's own expression, not an inline s*(q-z) that
  the compiler could fuse with an add into one differently rounded multiply-add; for
  the same reason an Add with an int8 input still stores its sum. YOLO26n's C2f blocks
  wrote and re-read about 10 MB of such floats a run; now they read the int8 (a quarter
  of the bytes). `LAZY_FLOATS = False` turns it off.
- The NHWC pass relabels a tensor known to have a rank other than 4 instead of
  inserting a 4-D Transpose its emitter could only turn into a copy (four such copies of
  up to 450 KB in YOLO26n's head).

### Fixed (second round)
- The portable targets returned nothing: for `ansi_int8` / `ansi_float`,
  `activations.h` declared the graph's input and output `static`, so every file that
  included it had its own copies, and the caller filled an input and read an output
  the model never used. They are `extern` now, defined once in `activations.c`;
  `ansi_int8` gives the same bytes as the esp32s3 build.
- `ansi_float` convolutions left out their bias.
- `build_model.py` stopped at once (it declared `--debug` twice).
- `compare_ort.py`: onnxruntime 1.29's graph optimizer rejects the shipped model
  (its QDQ propagation makes a QuantizeLinear with mismatched types); it now runs the
  model as written, as `serial_compare.py` already did.
- An op with no emitter was skipped with a printed note, leaving C that compiled and
  computed the wrong thing; codegen now stops. A Transpose whose perm does not fit its
  input's shape was emitted as a copy ("verify!"); it is an error now (and a tensor whose
  shape cannot be resolved keeps its layout Transpose, so it reaches that error). A
  nearest Resize the emitter does not compute exactly (align_corners, fractional
  scales, rounding modes that differ from floor at its scale) is refused instead of
  computed wrongly.
- `sigmoid_through_layout` could fold the Sigmoid into a tensor that was also a graph
  output, handing the caller sigmoided values; it now leaves such a tensor alone.
- There was no Squeeze emitter (Squeeze was skipped, see above). Flatten, Unsqueeze,
  Squeeze, Expand and ReduceMax keep an int8 tensor's bytes as bytes instead of
  treating it as float.
- `graph_json.py` runs ONNX shape inference when the model does not carry its shapes
  (onnxruntime's quantizer does not save them), instead of the conv emitter failing.

### Added (second round)
- `tools/femto_host.py`: compile a model and run the generated C on a PC.
- `tools/test_golden.py` and `tests/golden_yolo26n.json`: the shipped YOLO26n must give
  the recorded bytes, on both targets.
- `tools/test_ops.py`: small models of other shapes (odd channel counts, strides,
  depthwise, C2f, SPPF, 2x and 3x upsampling, a transposing head, one tensor in two
  Concats, and a Resize that must be refused), quantized by onnxruntime,
  compiled, and checked against onnxruntime and between targets.
- `tools/femto_bench.py`: the scorecard above.
- CI runs all of these, and the conv kernel test, on every push and pull request; the
  golden check twice, the second time with multiply-add fusion on as on the board.

### Changed (first round)
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
  Measured: 5.1 s -> 3.3 s a run.
- `fuse.py` `fold_quantize_into_producer`: a QuantizeLinear whose float input is also
  read elsewhere (at another scale, or by a residual Add) is done by the Concat, SiLU,
  Split or Add that makes the float, in the same pass, so the float is never read back
  to be quantized; where nothing else reads it, the float is not written at all. 36 of
  YOLO26n's 39 remaining QuantizeLinears. Bit-identical, on four test images.
- A model whose first ops quantize its float NCHW input and transpose it to NHWC gets
  `run_model_int8_input()`, `model_input_int8` and `model_quantize_input()` (its own
  QuantizeLinear) in `model.h`: the caller writes the int8 NHWC input directly, a byte a
  value instead of four, and both ops are skipped. The same bytes as `run_model()`.
- Transpose is tiled 16 x 16 wherever it is a batched 2-D transpose (NHWC <-> NCHW, a
  matrix's last two axes): element by element, one side wrote a PSRAM cache line per
  value. Concat, Split and Reshape copy rows with `memcpy`; int8 SiLU tables run a word
  of four values at a time; Add drops its modulo where both sides are the whole tensor.
- `FUSE_SILU_QUANTIZE` on by default, where the Quantize is the SiLU's only reader.
- SiLU becomes a table lookup wherever the graph has Dequantize -> Sigmoid -> Mul, not
  only where a QuantizeLinear follows: removes YOLO26n's 38 float `expf` Sigmoids.
  Not bit-identical to what it replaced: each table entry is SiLU worked out in double
  precision and rounded once, where `expf` on the board (and on a PC) rounds its own
  way, so an entry can differ in the last bit. On YOLO26n's test photos every output
  is the same; on random-noise inputs an int8 value occasionally rounds the other
  way from layer 19 on, moving a low-scoring box by about a pixel.
- QuantizeLinear multiplies by 1/scale and rounds inline instead of a float divide and
  `lroundf` per element. `x*(1/s)` can differ from `x/s` in the last bit, so a value
  that falls exactly between two int8 steps may round to the other one.
- TopK is one pass with a K-entry heap instead of K passes (YOLO26n: 300 x 24000):
  the same order, except among equal scores, which now come lowest index first as
  ONNX's TopK specifies (the K-pass version ordered ties differently).

### Added (first round)
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

### Fixed (first round)
- `quant_c`'s temporaries could shadow the caller's `v` (`float v=(v)*inv`, reading an
  uninitialized value); caught by the bit-exact test before release, its names are now
  its own.
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
