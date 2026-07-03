# AMX v1 Intrinsics Wrapper

Stable, header-only C API over Apple's undocumented AMX instructions. Targets **Apple Silicon AArch64** only (`amx_backend_aarch64.h`).

Low-level macros remain in `src/headers/aarch64.h`. Emulation reference: `src/emulate/`.

## Quick start

```c
#include "amx_intrinsics.h"

amx_f32x16_t x, y, z;
// ... fill x.lane[], y.lane[], z.lane[] (64-byte aligned types) ...

amx_scope_t scope = amx_scope_begin();
amx_ldx(amx_ldx_op_single(x.lane, 0));
amx_ldy(amx_ldy_op_single(y.lane, 0));
amx_ldz(amx_ldz_op_single(z.lane, 0));
amx_fma32(amx_fma32_op_vec_acc(0, 0, 0));  /* vector: z[i] += x[i]*y[i] */
amx_stz(amx_stz_op_single(z.lane, 0));
amx_scope_end(&scope);
```

**Rules:**

1. Pair `amx_scope_begin()` / `amx_scope_end()` (maps to `AMX_SET` / `AMX_CLR`).
2. Pointers in operands must fit in **56 bits** (`amx_intr_ptr56()`).
3. All `amx_*x*` vector types are **`_Alignas(64)`** — required for load/store.
4. Prefer typed operand builders over raw `.bits`.

## Vector vs matrix mode

| Mode | Builder example | Effect |
|------|-----------------|--------|
| Vector | `amx_fma32_op_vec_acc(z_row, x_off, y_off)` | One Z row: elementwise MAC |
| Matrix | `amx_fma32_op_mat_acc(z_row, x_off, y_off)` | Outer product into Z subgrid |

Widths:

| Op family | X/Y lanes per 64B strip | Z tile rows | `z_row` mask (mat) |
|-----------|-------------------------:|------------:|-------------------:|
| FMA16 | 32 | 32 | `& 1` |
| FMA32 | 16 | 16 | `& 3` |
| FMA64 | 8 | 8 | `& 7` |

Matrix tests use logical tile rows `j` mapped to AMX Z rows as `j * stride + (z_base & mask)` (see `src/test/amx_mat_correctness.c`).

FMS16/32/64 mirror FMA with the same vec/mat builders.

## Load / store

`amx_ldx` / `amx_ldy` / `amx_ldz` and `amx_stx` / `amx_sty` / `amx_stz` take typed `*_op_single(ptr, row)` operands. Multi-row panel helpers are **not** in v1 yet (see `docs/Intrinsics-Roadmap.md`).

## Build and test

```bash
make test_all          # emulation + intrinsics v1 + mat gold (recommended gate)
make test              # full instruction emulation sweep
make test_intrinsics_v1
make test_mat          # fma16/f32/f64 matrix gold vs emulate
```

Matrix gold tests: smoke, `z_base`, accumulation, builder mask, 100-seed HW vs `emulate_AMX_FMA*` / `emulate_AMX_FMS*` for f16/f32/f64.

## What v1 does not cover

- `extr*`, `vecint`, `matint`, `mac16`, `genlut`, `matfp` families
- Multi-row / streaming ld/st builders
- Inference GEMM / K-loop kernels (autoresearch closed negative vs Accelerate)

See `docs/Intrinsics-Roadmap.md` for expansion order.

## Advanced patterns

- **Z-group rotation** (throughput micro-optimization): `docs/Advanced-Z-Rotation.md` — not default API; poor fit for latency and real K-loop traffic.

## Further reading

- `docs/RegisterFile.md` — X/Y/Z layout
- `docs/instructions/` — per-instruction reference
