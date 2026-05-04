#include "amx_intrinsics.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(_Alignof(amx_u8x64_t) >= 64, "amx_u8x64_t must be 64-byte aligned");
_Static_assert(_Alignof(amx_f16x32_t) >= 64, "amx_f16x32_t must be 64-byte aligned");
_Static_assert(_Alignof(amx_f32x16_t) >= 64, "amx_f32x16_t must be 64-byte aligned");
_Static_assert(_Alignof(amx_f64x8_t) >= 64, "amx_f64x8_t must be 64-byte aligned");

/* ─── timing ─────────────────────────────────────────────────────────────── */

static uint64_t now_ns(void) {
#if defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
#endif
}

/* ─── deterministic RNG ───────────────────────────────────────────────────── */

static uint32_t rng_state = 0x12345678u;

static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static float rng_f32(void) {
    return (float)((int32_t)(rng_next() >> 8)) * (1.0f / 16777216.0f);
}

static double rng_f64(void) {
    uint64_t u = ((uint64_t)rng_next() << 32) | rng_next();
    return (double)(int64_t)(u >> 8) * (1.0 / (double)(1ull << 56));
}

/* ─── buffer allocation helper ───────────────────────────────────────────── */

static int alloc_aligned(void** ptr, size_t size) {
    void* p = 0;
    int err = posix_memalign(&p, 64, size);
    if (err == 0) *ptr = p;
    return err;
}

/* ─── operand builder tests ───────────────────────────────────────────────── */

static void test_operand_builders(void) {
    const uint64_t p = 0xFEDCBA9876543210ull;
    const uint64_t ptr56 = p & ((1ull << 56) - 1);

    amx_ldx_op_t ldx = amx_ldx_op_single((const void*)(uintptr_t)p, 11);
    assert(ldx.bits == (ptr56 | (3ull << 56)));

    amx_ldy_op_t ldy = amx_ldy_op_single((const void*)(uintptr_t)p, 8);
    assert(ldy.bits == (ptr56 | (0ull << 56)));

    amx_stx_op_t stx = amx_stx_op_single((void*)(uintptr_t)p, 15);
    assert(stx.bits == (ptr56 | (7ull << 56)));

    amx_sty_op_t sty = amx_sty_op_single((void*)(uintptr_t)p, 2);
    assert(sty.bits == (ptr56 | (2ull << 56)));

    amx_ldz_op_t ldz = amx_ldz_op_single((const void*)(uintptr_t)p, 129);
    assert(ldz.bits == (ptr56 | (1ull << 56)));

    amx_stz_op_t stz = amx_stz_op_single((void*)(uintptr_t)p, 63);
    assert(stz.bits == (ptr56 | (63ull << 56)));

    amx_fma32_op_t fma32v = amx_fma32_op_vec_acc(127, 0x3FF, 0x2FF);
    assert(fma32v.bits == ((1ull << 63) | (63ull << 20) | (0x1FFull << 10) | 0x0FFull));

    amx_fma32_op_t fma32m = amx_fma32_op_mat_acc(65, 123, 456);
    assert(fma32m.bits == ((1ull << 20) | (123ull << 10) | 456ull));

    amx_fms64_op_t fms64v = amx_fms64_op_vec_acc(9, 88, 77);
    assert(fms64v.bits == ((1ull << 63) | (9ull << 20) | (88ull << 10) | 77ull));

    amx_fms16_op_t fms16m = amx_fms16_op_mat_acc(7, 1, 2);
    assert(fms16m.bits == ((7ull << 20) | (1ull << 10) | 2ull));
}

/* ─── correctness: FMA32 vector mode ──────────────────────────────────────── */

static float test_fma32_vec_max_err = 0.0f;

static void test_runtime_fma32_vec_correctness(void) {
    amx_f32x16_t x, y, out;
    for (int i = 0; i < 16; ++i) {
        x.lane[i] = (float)(i + 1);
        y.lane[i] = 2.0f;
        out.lane[i] = -1.0f;
    }

    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x.lane, 0));
    amx_ldy(amx_ldy_op_single(y.lane, 0));
    amx_fma32(amx_fma32_op_vec_acc(0, 0, 0));
    amx_stz(amx_stz_op_single(out.lane, 0));
    amx_scope_end(&scope);

    float max_err = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float expected = x.lane[i] * y.lane[i];
        float err = fabsf(out.lane[i] - expected);
        if (err > max_err) max_err = err;
    }
    test_fma32_vec_max_err = max_err;
    assert(max_err < 1e-5f);
}

/* ─── correctness: FMA64 vector mode ──────────────────────────────────────── */

static double test_fma64_vec_max_err = 0.0;

static void test_runtime_fma64_vec_correctness(void) {
    amx_f64x8_t x, y, out;
    for (int i = 0; i < 8; ++i) {
        x.lane[i] = (double)(i + 1);
        y.lane[i] = 3.0;
        out.lane[i] = -1.0;
    }

    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x.lane, 0));
    amx_ldy(amx_ldy_op_single(y.lane, 0));
    amx_fma64(amx_fma64_op_vec_acc(0, 0, 0));
    amx_stz(amx_stz_op_single(out.lane, 0));
    amx_scope_end(&scope);

    double max_err = 0.0;
    for (int i = 0; i < 8; ++i) {
        double expected = x.lane[i] * y.lane[i];
        double err = fabs(out.lane[i] - expected);
        if (err > max_err) max_err = err;
    }
    test_fma64_vec_max_err = max_err;
    assert(max_err < 1e-12);
}

/* ─── correctness: FMA16 vector mode ──────────────────────────────────────── */

static float test_fma16_vec_max_err = 0.0f;

static void test_runtime_fma16_vec_correctness(void) {
    amx_f16x32_t x, y, out;
    for (int i = 0; i < 32; ++i) {
        float v = (float)(i + 1);
        x.lane[i] = v;
        y.lane[i] = v * 2.0f;
        out.lane[i] = -1.0f;
    }

    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x.lane, 0));
    amx_ldy(amx_ldy_op_single(y.lane, 0));
    amx_fma16(amx_fma16_op_vec_acc(0, 0, 0));
    amx_stz(amx_stz_op_single(out.lane, 0));
    amx_scope_end(&scope);

    float max_err = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float expected = (float)x.lane[i] * (float)y.lane[i];
        float got = (float)out.lane[i];
        float err = fabsf(got - expected);
        if (err > max_err) max_err = err;
    }
    test_fma16_vec_max_err = max_err;
    assert(max_err < 0.001f);
}

/* ─── correctness: FMS32 vector mode ─────────────────────────────────────── */

static float test_fms32_vec_max_err = 0.0f;

static void test_runtime_fms32_vec_correctness(void) {
    amx_f32x16_t x, y, out;
    for (int i = 0; i < 16; ++i) {
        x.lane[i] = (float)(i + 1);
        y.lane[i] = 3.0f;
        out.lane[i] = 123.0f;
    }

    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x.lane, 0));
    amx_ldy(amx_ldy_op_single(y.lane, 0));
    amx_fms32(amx_fms32_op_vec_acc(0, 0, 0));
    amx_stz(amx_stz_op_single(out.lane, 0));
    amx_scope_end(&scope);

    float max_err = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float expected = -(x.lane[i] * y.lane[i]);
        float err = fabsf(out.lane[i] - expected);
        if (err > max_err) max_err = err;
    }
    test_fms32_vec_max_err = max_err;
    assert(max_err < 1e-5f);
}

/* ─── correctness: FMS64 vector mode ─────────────────────────────────────── */

static double test_fms64_vec_max_err = 0.0;

static void test_runtime_fms64_vec_correctness(void) {
    amx_f64x8_t x, y, out;
    for (int i = 0; i < 8; ++i) {
        x.lane[i] = (double)(i + 1);
        y.lane[i] = 5.0;
        out.lane[i] = 777.0;
    }

    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x.lane, 0));
    amx_ldy(amx_ldy_op_single(y.lane, 0));
    amx_fms64(amx_fms64_op_vec_acc(0, 0, 0));
    amx_stz(amx_stz_op_single(out.lane, 0));
    amx_scope_end(&scope);

    double max_err = 0.0;
    for (int i = 0; i < 8; ++i) {
        double expected = -(x.lane[i] * y.lane[i]);
        double err = fabs(out.lane[i] - expected);
        if (err > max_err) max_err = err;
    }
    test_fms64_vec_max_err = max_err;
    assert(max_err < 1e-12);
}

/* ─── correctness: FMS16 vector mode ─────────────────────────────────────── */

static float test_fms16_vec_max_err = 0.0f;

static void test_runtime_fms16_vec_correctness(void) {
    amx_f16x32_t x, y, out;
    for (int i = 0; i < 32; ++i) {
        float v = (float)(i + 1);
        x.lane[i] = v;
        y.lane[i] = v * 4.0f;
        out.lane[i] = 888.0f;
    }

    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x.lane, 0));
    amx_ldy(amx_ldy_op_single(y.lane, 0));
    amx_fms16(amx_fms16_op_vec_acc(0, 0, 0));
    amx_stz(amx_stz_op_single(out.lane, 0));
    amx_scope_end(&scope);

    float max_err = 0.0f;
    for (int i = 0; i < 32; ++i) {
        float expected = -((float)x.lane[i] * (float)y.lane[i]);
        float got = (float)out.lane[i];
        float err = fabsf(got - expected);
        if (err > max_err) max_err = err;
    }
    test_fms16_vec_max_err = max_err;
    assert(max_err < 0.001f);
}

/* ─── scope guard ─────────────────────────────────────────────────────────── */

static void test_scope_end_idempotent(void) {
    amx_scope_t scope = amx_scope_begin();
    amx_scope_end(&scope);
    amx_scope_end(&scope);
    assert(scope.active == 0u);
}

/* ─── multi-run statistics helpers ─────────────────────────────────────────── */

#define WARMUP_ITERS  16
#define NUM_RUNS       5

typedef struct {
    double min;
    double median;
    double max;
    double stddev;
} run_stats_t;

static run_stats_t compute_stats(double values[NUM_RUNS]) {
    for (int i = 0; i < NUM_RUNS - 1; ++i) {
        for (int j = i + 1; j < NUM_RUNS; ++j) {
            if (values[j] < values[i]) {
                double t = values[i]; values[i] = values[j]; values[j] = t;
            }
        }
    }
    double min_v    = values[0];
    double max_v    = values[NUM_RUNS - 1];
    double median_v = (NUM_RUNS % 2 == 0)
        ? (values[NUM_RUNS / 2 - 1] + values[NUM_RUNS / 2]) * 0.5
        : values[NUM_RUNS / 2];

    double mean = 0.0;
    for (int i = 0; i < NUM_RUNS; ++i) mean += values[i];
    mean /= NUM_RUNS;
    double var = 0.0;
    for (int i = 0; i < NUM_RUNS; ++i) {
        double d = values[i] - mean;
        var += d * d;
    }
    var /= NUM_RUNS;

    return (run_stats_t){ min_v, median_v, max_v, sqrt(var) };
}

/* ─── FMA32 vector throughput (warmup + multi-run, cache sensitivity) ─────── */

static double run_fma32_vec_throughput_at_size(uint32_t chunks, double* out_gflops) {
    amx_f32x16_t* x = 0;
    amx_f32x16_t* y = 0;
    amx_f32x16_t* out = 0;

    /* try allocation at requested size, fall back to smaller if needed */
    while (chunks >= 16) {
        if (alloc_aligned((void**)&x,   (size_t)chunks * sizeof(*x))   == 0 &&
            alloc_aligned((void**)&y,   (size_t)chunks * sizeof(*y))   == 0 &&
            alloc_aligned((void**)&out, (size_t)chunks * sizeof(*out)) == 0) {
            break;
        }
        free(x); free(y); free(out); x = y = out = 0;
        chunks >>= 1;
    }
    assert(x && y && out);

    /* fill buffers before timing */
    for (uint32_t c = 0; c < chunks; ++c) {
        for (int i = 0; i < 16; ++i) {
            x[c].lane[i] = rng_f32();
            y[c].lane[i] = rng_f32();
            out[c].lane[i] = 0.0f;
        }
    }

    const uint64_t op_bits = amx_fma32_op_vec_acc(0, 0, 0).bits | (1ull << 27);
    const amx_fma32_op_t op_mul = amx_fma32_op_raw(op_bits);

    volatile uintptr_t sink = 0;

    /* warmup */
    for (int w = 0; w < WARMUP_ITERS; ++w) {
        amx_scope_t scope = amx_scope_begin();
        for (uint32_t c = 0; c < chunks; ++c) {
            amx_ldx(amx_ldx_op_single(x[c].lane, 0));
            amx_ldy(amx_ldy_op_single(y[c].lane, 0));
            amx_fma32(op_mul);
            amx_stz(amx_stz_op_single(out[c].lane, 0));
        }
        amx_scope_end(&scope);
        sink += (uintptr_t)out;
    }
    (void)sink;

    /* reset output before measurement runs */
    for (uint32_t c = 0; c < chunks; ++c)
        for (int i = 0; i < 16; ++i)
            out[c].lane[i] = 0.0f;

    double gibps_arr[NUM_RUNS];
    double gflops_arr[NUM_RUNS];

    for (int r = 0; r < NUM_RUNS; ++r) {
        uint64_t t0 = now_ns();
        amx_scope_t scope = amx_scope_begin();
        for (uint32_t c = 0; c < chunks; ++c) {
            amx_ldx(amx_ldx_op_single(x[c].lane, 0));
            amx_ldy(amx_ldy_op_single(y[c].lane, 0));
            amx_fma32(op_mul);
            amx_stz(amx_stz_op_single(out[c].lane, 0));
        }
        amx_scope_end(&scope);
        uint64_t t1 = now_ns();

        double sec   = (double)(t1 - t0) * 1e-9;
        double elems = (double)chunks * 16.0;
        double gib   = elems * (double)sizeof(float) * 3.0 / (1024.0 * 1024.0 * 1024.0);
        gibps_arr[r]  = gib / sec;
        gflops_arr[r] = 2.0 * elems / sec / 1e9;
    }

    (void)compute_stats(gibps_arr);
    *out_gflops = gflops_arr[NUM_RUNS / 2];

    /* correctness spot-check on last run output */
    double max_err = 0.0;
    for (uint32_t s = 0; s < 256; ++s) {
        uint32_t c = (s * 4099u) % chunks;
        uint32_t i = (s * 13u) & 15u;
        double ref = (double)x[c].lane[i] * (double)y[c].lane[i];
        double err = fabs((double)out[c].lane[i] - ref);
        if (err > max_err) max_err = err;
    }
    assert(max_err < 1e-4);

    free(x); free(y); free(out);
    return gibps_arr[NUM_RUNS / 2];
}

/* ─── latency measurement ─────────────────────────────────────────────────── */

static double measure_fma32_vec_latency_ns(void) {
    amx_f32x16_t x, y, out;
    for (int i = 0; i < 16; ++i) {
        x.lane[i] = rng_f32();
        y.lane[i] = rng_f32();
        out.lane[i] = 0.0f;
    }

    const uint64_t op_bits = amx_fma32_op_vec_acc(0, 0, 0).bits | (1ull << 27);
    const amx_fma32_op_t op_mul = amx_fma32_op_raw(op_bits);
    const int ITERS = 1 << 20;
    volatile uintptr_t sink = 0;

    /* warmup */
    for (int w = 0; w < WARMUP_ITERS; ++w) {
        amx_scope_t scope = amx_scope_begin();
        for (int i = 0; i < 1024; ++i) {
            amx_ldx(amx_ldx_op_single(x.lane, 0));
            amx_ldy(amx_ldy_op_single(y.lane, 0));
            amx_fma32(op_mul);
            amx_stz(amx_stz_op_single(out.lane, 0));
        }
        amx_scope_end(&scope);
        sink += (uintptr_t)&out;
    }
    (void)sink;

    double times[NUM_RUNS];
    for (int r = 0; r < NUM_RUNS; ++r) {
        uint64_t t0 = now_ns();
        amx_scope_t scope = amx_scope_begin();
        for (int i = 0; i < ITERS; ++i) {
            amx_ldx(amx_ldx_op_single(x.lane, 0));
            amx_ldy(amx_ldy_op_single(y.lane, 0));
            amx_fma32(op_mul);
            amx_stz(amx_stz_op_single(out.lane, 0));
        }
        amx_scope_end(&scope);
        uint64_t t1 = now_ns();
        times[r] = (double)(t1 - t0) / ITERS;
    }
    (void)out;
    return times[NUM_RUNS / 2];
}

static double measure_fma64_vec_latency_ns(void) {
    amx_f64x8_t x, y, out;
    for (int i = 0; i < 8; ++i) {
        x.lane[i] = rng_f64();
        y.lane[i] = rng_f64();
        out.lane[i] = 0.0;
    }

    const uint64_t op_bits = amx_fma64_op_vec_acc(0, 0, 0).bits | (1ull << 27);
    const amx_fma64_op_t op_mul = amx_fma64_op_raw(op_bits);
    const int ITERS = 1 << 20;
    volatile uintptr_t sink = 0;

    for (int w = 0; w < WARMUP_ITERS; ++w) {
        amx_scope_t scope = amx_scope_begin();
        for (int i = 0; i < 1024; ++i) {
            amx_ldx(amx_ldx_op_single(x.lane, 0));
            amx_ldy(amx_ldy_op_single(y.lane, 0));
            amx_fma64(op_mul);
            amx_stz(amx_stz_op_single(out.lane, 0));
        }
        amx_scope_end(&scope);
        sink += (uintptr_t)&out;
    }
    (void)sink;

    double times[NUM_RUNS];
    for (int r = 0; r < NUM_RUNS; ++r) {
        uint64_t t0 = now_ns();
        amx_scope_t scope = amx_scope_begin();
        for (int i = 0; i < ITERS; ++i) {
            amx_ldx(amx_ldx_op_single(x.lane, 0));
            amx_ldy(amx_ldy_op_single(y.lane, 0));
            amx_fma64(op_mul);
            amx_stz(amx_stz_op_single(out.lane, 0));
        }
        amx_scope_end(&scope);
        uint64_t t1 = now_ns();
        times[r] = (double)(t1 - t0) / ITERS;
    }
    (void)out;
    return times[NUM_RUNS / 2];
}

/* ─── FMA32 matrix throughput (outer product one 16x16 tile per call) ─────── */

static double run_fma32_mat_throughput_gibps(uint32_t z_rows) {
    amx_f32x16_t* x;
    amx_f32x16_t* y;
    amx_f32x16_t* z;

    assert(posix_memalign((void**)&x, 64, 16 * sizeof(amx_f32x16_t)) == 0);
    assert(posix_memalign((void**)&y, 64, 16 * sizeof(amx_f32x16_t)) == 0);
    assert(posix_memalign((void**)&z, 64, (size_t)z_rows * sizeof(amx_f32x16_t)) == 0);

    for (int i = 0; i < 16; ++i) {
        x->lane[i] = rng_f32();
        y->lane[i] = rng_f32();
    }
    for (uint32_t r = 0; r < z_rows; ++r)
        for (int i = 0; i < 16; ++i)
            z[r].lane[i] = 0.0f;

    volatile uintptr_t sink = 0;

    /* warmup */
    for (int w = 0; w < WARMUP_ITERS; ++w) {
        amx_scope_t scope = amx_scope_begin();
        for (uint32_t row = 0; row < z_rows; ++row)
            amx_ldz(amx_ldz_op_single(&z[row].lane[0], row & 15));
        amx_ldx(amx_ldx_op_single(x->lane, 0));
        amx_ldy(amx_ldy_op_single(y->lane, 0));
        amx_fma32(amx_fma32_op_mat_acc(0, 0, 0));
        amx_scope_end(&scope);
        sink += (uintptr_t)z;
    }
    (void)sink;

    uint64_t t0 = now_ns();
    amx_scope_t scope = amx_scope_begin();
    for (uint32_t row = 0; row < z_rows; ++row)
        amx_ldz(amx_ldz_op_single(&z[row].lane[0], row & 15));
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    amx_fma32(amx_fma32_op_mat_acc(0, 0, 0));
    amx_scope_end(&scope);
    uint64_t t1 = now_ns();

    double sec = (double)(t1 - t0) * 1e-9;
    /* bytes: z_rows * 16 * 4 bytes * (z_read + z_write + x_read + y_read) */
    double gib = (double)z_rows * 16.0 * 4.0 * 4.0 / (1024.0 * 1024.0 * 1024.0);
    double gibps = gib / sec;

    free(x); free(y); free(z);
    return gibps;
}

/* ─── main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== amx_intrinsics_v1_test ===\n");
    printf("Correctness:\n");

    test_operand_builders();
    printf("  operand builders ... OK\n");

    rng_state = 0x12345678u;

    test_runtime_fma32_vec_correctness();
    printf("  FMA32 vec ... OK (max_err=%.3e)\n", (double)test_fma32_vec_max_err);

    test_runtime_fma64_vec_correctness();
    printf("  FMA64 vec ... OK (max_err=%.3e)\n", test_fma64_vec_max_err);

    test_runtime_fma16_vec_correctness();
    printf("  FMA16 vec ... OK (max_err=%.3e)\n", (double)test_fma16_vec_max_err);

    test_runtime_fms32_vec_correctness();
    printf("  FMS32 vec ... OK (max_err=%.3e)\n", (double)test_fms32_vec_max_err);

    test_runtime_fms64_vec_correctness();
    printf("  FMS64 vec ... OK (max_err=%.3e)\n", test_fms64_vec_max_err);

    test_runtime_fms16_vec_correctness();
    printf("  FMS16 vec ... OK (max_err=%.3e)\n", (double)test_fms16_vec_max_err);

    test_scope_end_idempotent();
    printf("  scope guard ... OK\n");

    /* throughput: cache sensitivity, FMA32 vec */
    printf("Throughput (median of %d runs, %d warmup iterations):\n", NUM_RUNS, WARMUP_ITERS);
    printf("  FMA32 vec   4KB ... ");
    { double gf = 0; double gbps = run_fma32_vec_throughput_at_size(64, &gf); printf("%6.1f GiB/s  %6.1f GFLOP/s\n", gbps, gf); }
    printf("  FMA32 vec  64KB ... ");
    { double gf = 0; double gbps = run_fma32_vec_throughput_at_size(1024, &gf); printf("%6.1f GiB/s  %6.1f GFLOP/s\n", gbps, gf); }
    printf("  FMA32 vec   1MB ... ");
    { double gf = 0; double gbps = run_fma32_vec_throughput_at_size(16384, &gf); printf("%6.1f GiB/s  %6.1f GFLOP/s\n", gbps, gf); }
    printf("  FMA32 vec  16MB ... ");
    { double gf = 0; double gbps = run_fma32_vec_throughput_at_size(262144, &gf); printf("%6.1f GiB/s  %6.1f GFLOP/s\n", gbps, gf); }
    printf("  FMA32 vec 256MB ... ");
    { double gf = 0; double gbps = run_fma32_vec_throughput_at_size(4194304, &gf); printf("%6.1f GiB/s  %6.1f GFLOP/s\n", gbps, gf); }

    printf("  FMA32 mat   4KB ... %.1f GiB/s\n", run_fma32_mat_throughput_gibps(256));
    printf("  FMA32 mat  64KB ... %.1f GiB/s\n", run_fma32_mat_throughput_gibps(4096));

    /* latency */
    printf("Latency (median of %d runs, %d warmup iterations):\n", NUM_RUNS, WARMUP_ITERS);
    printf("  FMA32 vec ... %.1f ns/op\n", measure_fma32_vec_latency_ns());
    printf("  FMA64 vec ... %.1f ns/op\n", measure_fma64_vec_latency_ns());

    printf("PASS\n");
    return 0;
}
