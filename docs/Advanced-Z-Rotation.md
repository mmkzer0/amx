# Advanced: Four-Way Z-Group Rotation

**Status:** Documented pattern only — **not** part of the default v1 intrinsics API.

## What it is

FMA32 matrix mode has ~4-issue latency on the same Z bank. Interleaving four `z_base` groups (`0..3`) in the mat_acc operand breaks RAW hazards and raised **compute-only** microkernel throughput to ~2 TFLOP/s in autoresearch (`a3e6ee4` on branch `autoresearch/fma32-mat-kernel`).

## When it helps

- Tight inner loop issuing many `amx_fma32(amx_fma32_op_mat_acc(zb+k, ...))` on **loaded** X/Y with Z already in AMX
- High iteration count, compute-only timing (no per-iter full tile load/store)

## When it does not help

- Real K-loop panels: Z load/store per strip dominates (~8 GFLOPS vs ~2000 GFLOPS compute-only)
- Latency-sensitive single-shot ops (`iters==1` light path is the opposite tradeoff)
- Inference-shaped traffic vs Accelerate (see autoresearch closeout on `autoresearch/fma32-mat-kernel`)

## Sketch (not exported API)

```c
/* Pseudocode: rotate z_base 0,1,2,3 across fused mat_acc chain */
for (int k = 0; k < iters; ++k) {
    amx_fma32(amx_fma32_op_mat_acc((uint8_t)(k & 3u), 0, 0));
}
/* Requires session setup: zero/load all four Z groups before compute,
   merge groups back to memory after — see autoresearch P6 docs. */
```

For production matmul on Apple Silicon, use **Accelerate** (`cblas_*`) unless you have a non-BLAS shape requirement.

## Reference

- `docs/P6-FMA32-Z-Rotation.md` (on autoresearch branch)
- `docs/Autoresearch-Plugin-Handoff.md` closeout (experiment branch)
