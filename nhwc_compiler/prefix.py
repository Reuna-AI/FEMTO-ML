# Copyright (C) 2026 reuna-ai
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
prefix.py -- give one compiled model its own names, so two models can live in one firmware.

Every model FEMTO-ML compiles defines the same names: run_model(), activations_init(), tu0,
op_Conv_0_quant(), MODEL_INPUT_W, files model.h and activations.h ... Two of them in one sketch
(a detector and a face finder, say) clash at the compiler and the linker. apply(outdir, "face_")
renames, in a folder build_model.py has written:

- the files: model.c/.h, activations.c/.h, weights.h, weights_*.c -> face_model.c ..., and the
  #includes between them;
- every name they make visible to another file: the functions and buffers the headers declare,
  the arena's unions, the tensors' access macros and the MODEL_INPUT_* / model_* helpers.

The esp-nn kernels and femto_profile are shared by both models and left alone; so are names a
model only uses inside its own files (static tables and helpers). What is renamed is found in the
generated headers themselves, not from a list of patterns, so a new kind of name a later stage
adds is renamed too as long as a header declares it.
"""
import os, re

GENERATED = ("model", "activations", "weights")   # basenames whose files are renamed
IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def _files(outdir, name="model"):
    out = []
    for f in sorted(os.listdir(outdir)):
        base, ext = os.path.splitext(f)
        if ext in (".c", ".h", ".cpp") and (base in (name, "activations", "weights") or re.fullmatch(r"weights_\d+", base)):
            out.append(f)
    return out


def _declared(text):
    """The names a generated header makes visible to other files."""
    names = set()
    names |= set(re.findall(rf"^\s*#define\s+({IDENT})", text, re.M))
    names |= set(re.findall(rf"\b(?:union|struct)\s+({IDENT})\s*\{{", text))
    # Declarations: "type name(args);", "extern type name[...]...;", "extern type (*name)[...]"
    for line in text.splitlines():
        line = re.sub(r"//.*$", "", line).strip()      # "extern float (*t_p)[3]; // 12 KB PSRAM"
        if line.startswith(("#", "//", "/*", "*")) or not line.endswith(";"):
            continue
        m = re.match(rf"^(?:extern\s+)?(?:const\s+)?(?:unsigned\s+|signed\s+)?{IDENT}[\s\*]+(?:__attribute__\(\(.*?\)\)\s*)?"
                     rf"\(?\s*\*?\s*({IDENT})\s*[\(\[\)]", line)
        if m:
            names.add(m.group(1))
        m = re.match(rf"^extern\s+union\s+{IDENT}\s*\*\s*({IDENT})\s*;", line)
        if m:
            names.add(m.group(1))
    # inline helpers the header defines in full: "static inline type name(...) {"
    names |= set(re.findall(rf"static\s+inline\s+{IDENT}[\s\*]+({IDENT})\s*\(", text))
    return names


# Names every model shares and must keep: the toolchain's and the shared kernels' own, and the
# switches a build sets for all of its models at once.
SHARED = {"WEIGHTS_FLASH", "ARENA_ALLOC_SRAM", "ARENA_ALLOC_PSRAM", "ARENA_FREE", "DEBUG_TRACE", "DEBUG_DUMP",
          "DUMP_N", "MODEL_DEBUG", "FEMTO_PROFILE", "DBG_NODE", "DBG_STAT", "DBG_END", "TRACE",
          "_TFLM_REQUANT_DEFINED", "FEMTO_NO_FLOAT_INPUT"}


def apply(outdir, prefix, name="model"):
    """Rename the model compiled into [outdir] with [prefix] (e.g. "face_"). Returns {old: new} of the names."""
    if not re.fullmatch(IDENT, prefix):
        raise ValueError(f"prefix must be a C identifier: {prefix!r}")

    files = _files(outdir, name)
    texts = {f: open(os.path.join(outdir, f), encoding="utf-8").read() for f in files}
    names = set()

    for f, text in texts.items():
        if f.endswith(".h"):
            names |= _declared(text)

    names -= SHARED
    names = {n for n in names if not n.startswith("__")}
    renamed = {n: prefix + n for n in names}
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b") if names else None

    for f, text in texts.items():
        if pattern:
            text = pattern.sub(lambda m: renamed[m.group(1)], text)

        # #include "model.h" -> #include "face_model.h" (only the generated files' own)
        for g in files:
            text = text.replace(f'#include "{g}"', f'#include "{prefix}{g}"')

        os.remove(os.path.join(outdir, f))
        with open(os.path.join(outdir, prefix + f), "w", encoding="utf-8") as out:
            out.write(text)

    return renamed
