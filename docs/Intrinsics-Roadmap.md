# Intrinsics v1 Roadmap (scoping)

**Branch:** `feat/intrinsics-wrapper`  
**Gate:** `make test_all`

## Done (v1 baseline)

- Scope guards, typed FMA/FMS f16/f32/f64 vec+mat builders
- ld/st single-row operands
- Vector correctness (`amx_intrinsics_v1_test.c`)
- Matrix gold f16/f32/f64 (`amx_mat_correctness.c`)

## Next (when needed by consumers)

| Priority | Item | Notes |
|----------|------|-------|
| P1 | `amx_ldz` / `amx_stz` multi-row panel helpers | Only with correctness tests per width |
| P2 | FMS matrix gold tests | Mirror FMA mat coverage |
| P3 | `vecfp` / `matfp` wrappers | After FMA family stable |

## Deferred (no v1 work without explicit ask)

| Item | Reason |
|------|--------|
| `extr_*` | Operand complexity; rare in matmul paths |
| `vecint` / `matint` | Separate ALU semantics; needs int gold suite |
| `mac16` / `genlut` | Niche crypto / LUT paths |
| Multi-core / pthread overlap | Autoresearch discard on AMX |
| Inference kernels (K-loop, block GEMM) | Accelerate wins throughput and energy |

## API principles

1. Header-only `static inline`; no linker dependency.
2. Typed `.bits` operands at API boundary, not raw `uint64_t` at call sites.
3. New family = emulate cross-check before perf work.
4. No default API for autoresearch-only patterns (see `docs/Advanced-Z-Rotation.md`).
