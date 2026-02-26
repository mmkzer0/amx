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

static uint64_t now_ns(void) {
#if defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
#endif
}

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

static void test_runtime_fma32_vec(void) {
    amx_f32x16_t x;
    amx_f32x16_t y;
    amx_f32x16_t out;
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

    for (int i = 0; i < 16; ++i) {
        const float expected = x.lane[i] * y.lane[i];
        assert(fabsf(out.lane[i] - expected) < 0.00001f);
    }
}

static void test_runtime_fms32_vec(void) {
    amx_f32x16_t x;
    amx_f32x16_t y;
    amx_f32x16_t out;
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

    for (int i = 0; i < 16; ++i) {
        const float expected = -(x.lane[i] * y.lane[i]);
        assert(fabsf(out.lane[i] - expected) < 0.00001f);
    }
}

static void test_scope_end_idempotent(void) {
    amx_scope_t scope = amx_scope_begin();
    amx_scope_end(&scope);
    amx_scope_end(&scope);
    assert(scope.active == 0u);
}

static void test_runtime_fma32_dram_throughput(void) {
    uint32_t chunks = 1024u * 1024u; // 64MB per operand buffer
    amx_f32x16_t* x = 0;
    amx_f32x16_t* y = 0;
    amx_f32x16_t* out = 0;
    while (chunks >= (256u * 1024u)) {
        if (posix_memalign((void**)&x, 64, (size_t)chunks * sizeof(*x)) != 0) {
            x = 0;
        }
        if (posix_memalign((void**)&y, 64, (size_t)chunks * sizeof(*y)) != 0) {
            y = 0;
        }
        if (posix_memalign((void**)&out, 64, (size_t)chunks * sizeof(*out)) != 0) {
            out = 0;
        }
        if (x && y && out) {
            break;
        }
        free(x); x = 0;
        free(y); y = 0;
        free(out); out = 0;
        chunks >>= 1;
    }
    assert(x && y && out);

    uint32_t rng = 0x12345678u;
    for (uint32_t c = 0; c < chunks; ++c) {
        for (int i = 0; i < 16; ++i) {
            rng = rng * 1664525u + 1013904223u;
            x[c].lane[i] = (float)((int32_t)(rng >> 8)) * (1.0f / 16777216.0f);
            rng = rng * 1664525u + 1013904223u;
            y[c].lane[i] = (float)((int32_t)(rng >> 8)) * (1.0f / 16777216.0f);
            out[c].lane[i] = 0.0f;
        }
    }

    const uint64_t op_bits = amx_fma32_op_vec_acc(0, 0, 0).bits | (1ull << 27); // skip Z input: out = x*y
    const amx_fma32_op_t op_mul = amx_fma32_op_raw(op_bits);

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

    const uint64_t elapsed_ns = (t1 > t0) ? (t1 - t0) : 1;
    const double elems = (double)chunks * 16.0;
    const double sec = (double)elapsed_ns * 1e-9;
    const double gib = (elems * (double)(sizeof(float) * 3)) / (1024.0 * 1024.0 * 1024.0);
    const double gibps = gib / sec;
    const double gflops = (2.0 * elems) / sec / 1e9;

    double max_abs_err = 0.0;
    for (uint32_t s = 0; s < 256; ++s) {
        uint32_t c = (s * 4099u) % chunks;
        uint32_t i = (s * 13u) & 15u;
        double ref = (double)x[c].lane[i] * (double)y[c].lane[i];
        double err = fabs((double)out[c].lane[i] - ref);
        if (err > max_abs_err) max_abs_err = err;
    }
    assert(isfinite(gibps) && gibps > 0.0);
    assert(isfinite(gflops) && gflops > 0.0);
    assert(max_abs_err < 1e-4);

    printf("amx_intrinsics_v1_test: throughput chunks=%u (%.1f MiB/buffer), %.2f GiB/s, %.2f GFLOP/s, max_abs_err=%.3g\n",
           chunks,
           ((double)chunks * (double)sizeof(*x)) / (1024.0 * 1024.0),
           gibps,
           gflops,
           max_abs_err);

    free(x);
    free(y);
    free(out);
}

int main(void) {
    test_operand_builders();
    test_runtime_fma32_vec();
    test_runtime_fms32_vec();
    test_scope_end_idempotent();
    test_runtime_fma32_dram_throughput();
    puts("amx_intrinsics_v1_test: OK");
    return 0;
}
