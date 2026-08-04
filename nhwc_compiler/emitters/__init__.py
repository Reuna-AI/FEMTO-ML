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
#
# For issues and contributions, visit:
#
# ============================================================================

"""
Per-op C emitters. Each module registers a generator for one op type via @register.
The dispatcher (codegen.py) walks the fused JSON and calls the matching emitter.

Emitter contract:
    def emit(ctx, node) -> None
        ctx : EmitCtx  (buffers, tensor table, helpers, name sanitizer)
        node: the JSON node dict {name, op, inputs, outputs, attrs}
    The emitter writes:
      - a C function into ctx.body (or a call, if kernel is in a shared runtime)
      - any #include / decl into ctx.header
    Each op wraps its optimized/scalar variants in  #if USE_SIMD_<OP> ... #else ... #endif
"""
REGISTRY = {}
def register(op):
    def deco(fn): REGISTRY[op]=fn; return fn
    return deco
