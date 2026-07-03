#include "amx_intrinsics.h"
#include "emulate.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(_Alignof(amx_f16x32_t) >= 64, "amx_f16x32_t must be 64-byte aligned");
_Static_assert(_Alignof(amx_f32x16_t) >= 64, "amx_f32x16_t must be 64-byte aligned");
_Static_assert(_Alignof(amx_f64x8_t) >= 64, "amx_f64x8_t must be 64-byte aligned");

extern void emulate_AMX_FMA16(amx_state* state, uint64_t operand);
extern void emulate_AMX_FMA32(amx_state* state, uint64_t operand);
extern void emulate_AMX_FMA64(amx_state* state, uint64_t operand);
extern void emulate_AMX_FMS16(amx_state* state, uint64_t operand);
extern void emulate_AMX_FMS32(amx_state* state, uint64_t operand);
extern void emulate_AMX_FMS64(amx_state* state, uint64_t operand);

static uint32_t rng_state = 0x12345678u;

static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static float rng_f32(void) {
    return (float)((int32_t)(rng_next() >> 8)) * (1.0f / 16777216.0f);
}

static _Float16 rng_f16(void) {
    return (_Float16)rng_f32();
}

static double rng_f64(void) {
    return (double)rng_f32();
}

/* ─── FMA16 matrix (32×32) ───────────────────────────────────────────────── */

static const float F16_EPS = 1e-2f;

static void f16_fill_z(amx_f16x32_t z_tile[32], _Float16 value) {
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            z_tile[j].lane[i] = value;
        }
    }
}

static void f16_assert_outer(const amx_f16x32_t z_tile[32],
                             const amx_f16x32_t* x,
                             const amx_f16x32_t* y,
                             _Float16 z_addend) {
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            const float expected = (float)z_addend + (float)x->lane[i] * (float)y->lane[j];
            const float err = fabsf((float)z_tile[j].lane[i] - expected);
            assert(err < F16_EPS);
        }
    }
}

static void f16_assert_equal(const amx_f16x32_t* a, const amx_f16x32_t* b) {
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            const float err = fabsf((float)a[j].lane[i] - (float)b[j].lane[i]);
            assert(err < F16_EPS);
        }
    }
}

static void f16_run_hw(const amx_f16x32_t* x,
                       const amx_f16x32_t* y,
                       amx_f16x32_t z_tile[32],
                       uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 32; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 2) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fma16(amx_fma16_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 32; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 2) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void f16_run_emu(const amx_f16x32_t* x,
                        const amx_f16x32_t* y,
                        amx_f16x32_t z_tile[32],
                        uint8_t z_base) {
    amx_state state;
    memset(&state, 0, sizeof(state));
    memcpy(state.x[0].f16, x->lane, sizeof(x->lane));
    memcpy(state.y[0].f16, y->lane, sizeof(y->lane));
    for (int j = 0; j < 32; ++j) {
        memcpy(state.z[(j * 2) + z_base].f16, z_tile[j].lane, sizeof(z_tile[j].lane));
    }
    emulate_AMX_FMA16(&state, amx_fma16_op_mat_acc(z_base, 0, 0).bits);
    for (int j = 0; j < 32; ++j) {
        memcpy(z_tile[j].lane, state.z[(j * 2) + z_base].f16, sizeof(z_tile[j].lane));
    }
}

static void f16_setup_smoke(amx_f16x32_t* x, amx_f16x32_t* y) {
    for (int i = 0; i < 32; ++i) {
        x->lane[i] = (_Float16)(float)(i + 1);
        y->lane[i] = (_Float16)2.0f;
    }
}

static void test_fma16_mat_smoke(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_tile[32];
    f16_setup_smoke(&x, &y);
    f16_fill_z(z_tile, (_Float16)0.0f);
    f16_run_hw(&x, &y, z_tile, 0);
    f16_assert_outer(z_tile, &x, &y, (_Float16)0.0f);
}

static void test_fma16_mat_z_base(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_tile[32];
    f16_setup_smoke(&x, &y);
    for (uint8_t z_base = 1; z_base <= 1; ++z_base) {
        f16_fill_z(z_tile, (_Float16)0.0f);
        f16_run_hw(&x, &y, z_tile, z_base);
        f16_assert_outer(z_tile, &x, &y, (_Float16)0.0f);
    }
}

static void test_fma16_mat_accumulation(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_tile[32];
    f16_setup_smoke(&x, &y);
    f16_fill_z(z_tile, (_Float16)1.0f);
    f16_run_hw(&x, &y, z_tile, 0);
    f16_assert_outer(z_tile, &x, &y, (_Float16)1.0f);
}

static void test_fma16_mat_z_row_mask(void) {
    assert(amx_fma16_op_mat_acc(2, 0, 0).bits == amx_fma16_op_mat_acc(0, 0, 0).bits);
    assert(amx_fma16_op_mat_acc(3, 0, 0).bits == amx_fma16_op_mat_acc(1, 0, 0).bits);
}

static void test_fma16_mat_vs_emulate_random(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_hw[32];
    amx_f16x32_t z_emu[32];

    for (int seed_idx = 0; seed_idx < 100; ++seed_idx) {
        rng_state = 0x12345678u + (uint32_t)seed_idx * 0x9e3779b9u;
        for (int i = 0; i < 32; ++i) {
            x.lane[i] = rng_f16();
            y.lane[i] = rng_f16();
        }
        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 32; ++i) {
                const _Float16 v = rng_f16();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }
        const uint8_t z_base = (uint8_t)(rng_next() & 1u);
        f16_run_hw(&x, &y, z_hw, z_base);
        f16_run_emu(&x, &y, z_emu, z_base);
        f16_assert_equal(z_hw, z_emu);
    }
}

/* ─── FMA32 matrix (16×16) ───────────────────────────────────────────────── */

static const float F32_EPS = 1e-5f;

static void f32_fill_z(amx_f32x16_t z_tile[16], float value) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            z_tile[j].lane[i] = value;
        }
    }
}

static void f32_assert_outer(const amx_f32x16_t z_tile[16],
                             const amx_f32x16_t* x,
                             const amx_f32x16_t* y,
                             float z_addend) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            const float expected = z_addend + x->lane[i] * y->lane[j];
            assert(fabsf(z_tile[j].lane[i] - expected) < F32_EPS);
        }
    }
}

static void f32_assert_equal(const amx_f32x16_t* a, const amx_f32x16_t* b) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            assert(fabsf(a[j].lane[i] - b[j].lane[i]) < F32_EPS);
        }
    }
}

static void f32_run_hw(const amx_f32x16_t* x,
                       const amx_f32x16_t* y,
                       amx_f32x16_t z_tile[16],
                       uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 16; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 4) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fma32(amx_fma32_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 16; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 4) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void f32_run_emu(const amx_f32x16_t* x,
                        const amx_f32x16_t* y,
                        amx_f32x16_t z_tile[16],
                        uint8_t z_base) {
    amx_state state;
    memset(&state, 0, sizeof(state));
    memcpy(state.x[0].f32, x->lane, sizeof(x->lane));
    memcpy(state.y[0].f32, y->lane, sizeof(y->lane));
    for (int j = 0; j < 16; ++j) {
        memcpy(state.z[(j * 4) + z_base].f32, z_tile[j].lane, sizeof(z_tile[j].lane));
    }
    emulate_AMX_FMA32(&state, amx_fma32_op_mat_acc(z_base, 0, 0).bits);
    for (int j = 0; j < 16; ++j) {
        memcpy(z_tile[j].lane, state.z[(j * 4) + z_base].f32, sizeof(z_tile[j].lane));
    }
}

static void f32_setup_smoke(amx_f32x16_t* x, amx_f32x16_t* y) {
    for (int i = 0; i < 16; ++i) {
        x->lane[i] = (float)(i + 1);
        y->lane[i] = 2.0f;
    }
}

static void test_fma32_mat_smoke(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];
    f32_setup_smoke(&x, &y);
    f32_fill_z(z_tile, 0.0f);
    f32_run_hw(&x, &y, z_tile, 0);
    f32_assert_outer(z_tile, &x, &y, 0.0f);
}

static void test_fma32_mat_z_base(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];
    f32_setup_smoke(&x, &y);
    for (uint8_t z_base = 1; z_base <= 3; ++z_base) {
        f32_fill_z(z_tile, 0.0f);
        f32_run_hw(&x, &y, z_tile, z_base);
        f32_assert_outer(z_tile, &x, &y, 0.0f);
    }
}

static void test_fma32_mat_accumulation(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];
    f32_setup_smoke(&x, &y);
    f32_fill_z(z_tile, 1.0f);
    f32_run_hw(&x, &y, z_tile, 0);
    f32_assert_outer(z_tile, &x, &y, 1.0f);
}

static void test_fma32_mat_z_row_mask(void) {
    assert(amx_fma32_op_mat_acc(4, 0, 0).bits == amx_fma32_op_mat_acc(0, 0, 0).bits);
    assert(amx_fma32_op_mat_acc(5, 0, 0).bits == amx_fma32_op_mat_acc(1, 0, 0).bits);
}

static void test_fma32_mat_vs_emulate_random(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_hw[16];
    amx_f32x16_t z_emu[16];

    for (int seed_idx = 0; seed_idx < 100; ++seed_idx) {
        rng_state = 0x12345678u + (uint32_t)seed_idx * 0x9e3779b9u;
        for (int i = 0; i < 16; ++i) {
            x.lane[i] = rng_f32();
            y.lane[i] = rng_f32();
        }
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 16; ++i) {
                const float v = rng_f32();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }
        const uint8_t z_base = (uint8_t)(rng_next() & 3u);
        f32_run_hw(&x, &y, z_hw, z_base);
        f32_run_emu(&x, &y, z_emu, z_base);
        f32_assert_equal(z_hw, z_emu);
    }
}

/* ─── FMA64 matrix (8×8) ─────────────────────────────────────────────────── */

static const double F64_EPS = 1e-12;

static void f64_fill_z(amx_f64x8_t z_tile[8], double value) {
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            z_tile[j].lane[i] = value;
        }
    }
}

static void f64_assert_outer(const amx_f64x8_t z_tile[8],
                             const amx_f64x8_t* x,
                             const amx_f64x8_t* y,
                             double z_addend) {
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            const double expected = z_addend + x->lane[i] * y->lane[j];
            assert(fabs(z_tile[j].lane[i] - expected) < F64_EPS);
        }
    }
}

static void f64_assert_equal(const amx_f64x8_t* a, const amx_f64x8_t* b) {
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            assert(fabs(a[j].lane[i] - b[j].lane[i]) < F64_EPS);
        }
    }
}

static void f64_run_hw(const amx_f64x8_t* x,
                       const amx_f64x8_t* y,
                       amx_f64x8_t z_tile[8],
                       uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 8; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 8) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fma64(amx_fma64_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 8; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 8) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void f64_run_emu(const amx_f64x8_t* x,
                        const amx_f64x8_t* y,
                        amx_f64x8_t z_tile[8],
                        uint8_t z_base) {
    amx_state state;
    memset(&state, 0, sizeof(state));
    memcpy(state.x[0].f64, x->lane, sizeof(x->lane));
    memcpy(state.y[0].f64, y->lane, sizeof(y->lane));
    for (int j = 0; j < 8; ++j) {
        memcpy(state.z[(j * 8) + z_base].f64, z_tile[j].lane, sizeof(z_tile[j].lane));
    }
    emulate_AMX_FMA64(&state, amx_fma64_op_mat_acc(z_base, 0, 0).bits);
    for (int j = 0; j < 8; ++j) {
        memcpy(z_tile[j].lane, state.z[(j * 8) + z_base].f64, sizeof(z_tile[j].lane));
    }
}

static void f64_setup_smoke(amx_f64x8_t* x, amx_f64x8_t* y) {
    for (int i = 0; i < 8; ++i) {
        x->lane[i] = (double)(i + 1);
        y->lane[i] = 2.0;
    }
}

static void test_fma64_mat_smoke(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_tile[8];
    f64_setup_smoke(&x, &y);
    f64_fill_z(z_tile, 0.0);
    f64_run_hw(&x, &y, z_tile, 0);
    f64_assert_outer(z_tile, &x, &y, 0.0);
}

static void test_fma64_mat_z_base(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_tile[8];
    f64_setup_smoke(&x, &y);
    for (uint8_t z_base = 1; z_base <= 7; ++z_base) {
        f64_fill_z(z_tile, 0.0);
        f64_run_hw(&x, &y, z_tile, z_base);
        f64_assert_outer(z_tile, &x, &y, 0.0);
    }
}

static void test_fma64_mat_accumulation(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_tile[8];
    f64_setup_smoke(&x, &y);
    f64_fill_z(z_tile, 1.0);
    f64_run_hw(&x, &y, z_tile, 0);
    f64_assert_outer(z_tile, &x, &y, 1.0);
}

static void test_fma64_mat_z_row_mask(void) {
    assert(amx_fma64_op_mat_acc(8, 0, 0).bits == amx_fma64_op_mat_acc(0, 0, 0).bits);
    assert(amx_fma64_op_mat_acc(9, 0, 0).bits == amx_fma64_op_mat_acc(1, 0, 0).bits);
}

static void test_fma64_mat_vs_emulate_random(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_hw[8];
    amx_f64x8_t z_emu[8];

    for (int seed_idx = 0; seed_idx < 100; ++seed_idx) {
        rng_state = 0x12345678u + (uint32_t)seed_idx * 0x9e3779b9u;
        for (int i = 0; i < 8; ++i) {
            x.lane[i] = rng_f64();
            y.lane[i] = rng_f64();
        }
        for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < 8; ++i) {
                const double v = rng_f64();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }
        const uint8_t z_base = (uint8_t)(rng_next() & 7u);
        f64_run_hw(&x, &y, z_hw, z_base);
        f64_run_emu(&x, &y, z_emu, z_base);
        f64_assert_equal(z_hw, z_emu);
    }
}

/* ─── FMS16 matrix (32×32) ───────────────────────────────────────────────── */

static void f16_fms_assert_outer(const amx_f16x32_t z_tile[32],
                                 const amx_f16x32_t* x,
                                 const amx_f16x32_t* y,
                                 _Float16 z_addend) {
    for (int j = 0; j < 32; ++j) {
        for (int i = 0; i < 32; ++i) {
            const float expected = (float)z_addend - (float)x->lane[i] * (float)y->lane[j];
            assert(fabsf((float)z_tile[j].lane[i] - expected) < F16_EPS);
        }
    }
}

static void f16_fms_run_hw(const amx_f16x32_t* x,
                           const amx_f16x32_t* y,
                           amx_f16x32_t z_tile[32],
                           uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 32; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 2) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fms16(amx_fms16_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 32; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 2) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void f16_fms_run_emu(const amx_f16x32_t* x,
                            const amx_f16x32_t* y,
                            amx_f16x32_t z_tile[32],
                            uint8_t z_base) {
    amx_state state;
    memset(&state, 0, sizeof(state));
    memcpy(state.x[0].f16, x->lane, sizeof(x->lane));
    memcpy(state.y[0].f16, y->lane, sizeof(y->lane));
    for (int j = 0; j < 32; ++j) {
        memcpy(state.z[(j * 2) + z_base].f16, z_tile[j].lane, sizeof(z_tile[j].lane));
    }
    emulate_AMX_FMS16(&state, amx_fms16_op_mat_acc(z_base, 0, 0).bits);
    for (int j = 0; j < 32; ++j) {
        memcpy(z_tile[j].lane, state.z[(j * 2) + z_base].f16, sizeof(z_tile[j].lane));
    }
}

static void test_fms16_mat_smoke(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_tile[32];
    f16_setup_smoke(&x, &y);
    f16_fill_z(z_tile, (_Float16)0.0f);
    f16_fms_run_hw(&x, &y, z_tile, 0);
    f16_fms_assert_outer(z_tile, &x, &y, (_Float16)0.0f);
}

static void test_fms16_mat_z_base(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_tile[32];
    f16_setup_smoke(&x, &y);
    f16_fill_z(z_tile, (_Float16)0.0f);
    f16_fms_run_hw(&x, &y, z_tile, 1);
    f16_fms_assert_outer(z_tile, &x, &y, (_Float16)0.0f);
}

static void test_fms16_mat_accumulation(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_tile[32];
    f16_setup_smoke(&x, &y);
    f16_fill_z(z_tile, (_Float16)1.0f);
    f16_fms_run_hw(&x, &y, z_tile, 0);
    f16_fms_assert_outer(z_tile, &x, &y, (_Float16)1.0f);
}

static void test_fms16_mat_z_row_mask(void) {
    assert(amx_fms16_op_mat_acc(2, 0, 0).bits == amx_fms16_op_mat_acc(0, 0, 0).bits);
    assert(amx_fms16_op_mat_acc(3, 0, 0).bits == amx_fms16_op_mat_acc(1, 0, 0).bits);
}

static void test_fms16_mat_vs_emulate_random(void) {
    amx_f16x32_t x, y;
    amx_f16x32_t z_hw[32];
    amx_f16x32_t z_emu[32];

    for (int seed_idx = 0; seed_idx < 100; ++seed_idx) {
        rng_state = 0x12345678u + (uint32_t)seed_idx * 0x9e3779b9u;
        for (int i = 0; i < 32; ++i) {
            x.lane[i] = rng_f16();
            y.lane[i] = rng_f16();
        }
        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < 32; ++i) {
                const _Float16 v = rng_f16();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }
        const uint8_t z_base = (uint8_t)(rng_next() & 1u);
        f16_fms_run_hw(&x, &y, z_hw, z_base);
        f16_fms_run_emu(&x, &y, z_emu, z_base);
        f16_assert_equal(z_hw, z_emu);
    }
}

/* ─── FMS32 matrix (16×16) ───────────────────────────────────────────────── */

static void f32_fms_assert_outer(const amx_f32x16_t z_tile[16],
                                 const amx_f32x16_t* x,
                                 const amx_f32x16_t* y,
                                 float z_addend) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            const float expected = z_addend - x->lane[i] * y->lane[j];
            assert(fabsf(z_tile[j].lane[i] - expected) < F32_EPS);
        }
    }
}

static void f32_fms_run_hw(const amx_f32x16_t* x,
                           const amx_f32x16_t* y,
                           amx_f32x16_t z_tile[16],
                           uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 16; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 4) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fms32(amx_fms32_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 16; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 4) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void f32_fms_run_emu(const amx_f32x16_t* x,
                            const amx_f32x16_t* y,
                            amx_f32x16_t z_tile[16],
                            uint8_t z_base) {
    amx_state state;
    memset(&state, 0, sizeof(state));
    memcpy(state.x[0].f32, x->lane, sizeof(x->lane));
    memcpy(state.y[0].f32, y->lane, sizeof(y->lane));
    for (int j = 0; j < 16; ++j) {
        memcpy(state.z[(j * 4) + z_base].f32, z_tile[j].lane, sizeof(z_tile[j].lane));
    }
    emulate_AMX_FMS32(&state, amx_fms32_op_mat_acc(z_base, 0, 0).bits);
    for (int j = 0; j < 16; ++j) {
        memcpy(z_tile[j].lane, state.z[(j * 4) + z_base].f32, sizeof(z_tile[j].lane));
    }
}

static void test_fms32_mat_smoke(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];
    f32_setup_smoke(&x, &y);
    f32_fill_z(z_tile, 0.0f);
    f32_fms_run_hw(&x, &y, z_tile, 0);
    f32_fms_assert_outer(z_tile, &x, &y, 0.0f);
}

static void test_fms32_mat_z_base(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];
    f32_setup_smoke(&x, &y);
    for (uint8_t z_base = 1; z_base <= 3; ++z_base) {
        f32_fill_z(z_tile, 0.0f);
        f32_fms_run_hw(&x, &y, z_tile, z_base);
        f32_fms_assert_outer(z_tile, &x, &y, 0.0f);
    }
}

static void test_fms32_mat_accumulation(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];
    f32_setup_smoke(&x, &y);
    f32_fill_z(z_tile, 1.0f);
    f32_fms_run_hw(&x, &y, z_tile, 0);
    f32_fms_assert_outer(z_tile, &x, &y, 1.0f);
}

static void test_fms32_mat_z_row_mask(void) {
    assert(amx_fms32_op_mat_acc(4, 0, 0).bits == amx_fms32_op_mat_acc(0, 0, 0).bits);
    assert(amx_fms32_op_mat_acc(5, 0, 0).bits == amx_fms32_op_mat_acc(1, 0, 0).bits);
}

static void test_fms32_mat_vs_emulate_random(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_hw[16];
    amx_f32x16_t z_emu[16];

    for (int seed_idx = 0; seed_idx < 100; ++seed_idx) {
        rng_state = 0x12345678u + (uint32_t)seed_idx * 0x9e3779b9u;
        for (int i = 0; i < 16; ++i) {
            x.lane[i] = rng_f32();
            y.lane[i] = rng_f32();
        }
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 16; ++i) {
                const float v = rng_f32();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }
        const uint8_t z_base = (uint8_t)(rng_next() & 3u);
        f32_fms_run_hw(&x, &y, z_hw, z_base);
        f32_fms_run_emu(&x, &y, z_emu, z_base);
        f32_assert_equal(z_hw, z_emu);
    }
}

/* ─── FMS64 matrix (8×8) ─────────────────────────────────────────────────── */

static void f64_fms_assert_outer(const amx_f64x8_t z_tile[8],
                                 const amx_f64x8_t* x,
                                 const amx_f64x8_t* y,
                                 double z_addend) {
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            const double expected = z_addend - x->lane[i] * y->lane[j];
            assert(fabs(z_tile[j].lane[i] - expected) < F64_EPS);
        }
    }
}

static void f64_fms_run_hw(const amx_f64x8_t* x,
                           const amx_f64x8_t* y,
                           amx_f64x8_t z_tile[8],
                           uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 8; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 8) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fms64(amx_fms64_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 8; ++j) {
        const uint8_t amx_row = (uint8_t)((j * 8) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void f64_fms_run_emu(const amx_f64x8_t* x,
                            const amx_f64x8_t* y,
                            amx_f64x8_t z_tile[8],
                            uint8_t z_base) {
    amx_state state;
    memset(&state, 0, sizeof(state));
    memcpy(state.x[0].f64, x->lane, sizeof(x->lane));
    memcpy(state.y[0].f64, y->lane, sizeof(y->lane));
    for (int j = 0; j < 8; ++j) {
        memcpy(state.z[(j * 8) + z_base].f64, z_tile[j].lane, sizeof(z_tile[j].lane));
    }
    emulate_AMX_FMS64(&state, amx_fms64_op_mat_acc(z_base, 0, 0).bits);
    for (int j = 0; j < 8; ++j) {
        memcpy(z_tile[j].lane, state.z[(j * 8) + z_base].f64, sizeof(z_tile[j].lane));
    }
}

static void test_fms64_mat_smoke(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_tile[8];
    f64_setup_smoke(&x, &y);
    f64_fill_z(z_tile, 0.0);
    f64_fms_run_hw(&x, &y, z_tile, 0);
    f64_fms_assert_outer(z_tile, &x, &y, 0.0);
}

static void test_fms64_mat_z_base(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_tile[8];
    f64_setup_smoke(&x, &y);
    for (uint8_t z_base = 1; z_base <= 7; ++z_base) {
        f64_fill_z(z_tile, 0.0);
        f64_fms_run_hw(&x, &y, z_tile, z_base);
        f64_fms_assert_outer(z_tile, &x, &y, 0.0);
    }
}

static void test_fms64_mat_accumulation(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_tile[8];
    f64_setup_smoke(&x, &y);
    f64_fill_z(z_tile, 1.0);
    f64_fms_run_hw(&x, &y, z_tile, 0);
    f64_fms_assert_outer(z_tile, &x, &y, 1.0);
}

static void test_fms64_mat_z_row_mask(void) {
    assert(amx_fms64_op_mat_acc(8, 0, 0).bits == amx_fms64_op_mat_acc(0, 0, 0).bits);
    assert(amx_fms64_op_mat_acc(9, 0, 0).bits == amx_fms64_op_mat_acc(1, 0, 0).bits);
}

static void test_fms64_mat_vs_emulate_random(void) {
    amx_f64x8_t x, y;
    amx_f64x8_t z_hw[8];
    amx_f64x8_t z_emu[8];

    for (int seed_idx = 0; seed_idx < 100; ++seed_idx) {
        rng_state = 0x12345678u + (uint32_t)seed_idx * 0x9e3779b9u;
        for (int i = 0; i < 8; ++i) {
            x.lane[i] = rng_f64();
            y.lane[i] = rng_f64();
        }
        for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < 8; ++i) {
                const double v = rng_f64();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }
        const uint8_t z_base = (uint8_t)(rng_next() & 7u);
        f64_fms_run_hw(&x, &y, z_hw, z_base);
        f64_fms_run_emu(&x, &y, z_emu, z_base);
        f64_assert_equal(z_hw, z_emu);
    }
}

/* ─── main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== amx_mat_correctness ===\n");

    test_fma16_mat_smoke();
    printf("  fma16_mat_smoke ... OK\n");
    test_fma16_mat_z_base();
    printf("  fma16_mat_z_base ... OK\n");
    test_fma16_mat_accumulation();
    printf("  fma16_mat_accumulation ... OK\n");
    test_fma16_mat_z_row_mask();
    printf("  fma16_mat_z_row_mask ... OK\n");
    test_fma16_mat_vs_emulate_random();
    printf("  fma16_mat_vs_emulate_random ... OK\n");

    test_fma32_mat_smoke();
    printf("  fma32_mat_smoke ... OK\n");
    test_fma32_mat_z_base();
    printf("  fma32_mat_z_base ... OK\n");
    test_fma32_mat_accumulation();
    printf("  fma32_mat_accumulation ... OK\n");
    test_fma32_mat_z_row_mask();
    printf("  fma32_mat_z_row_mask ... OK\n");
    test_fma32_mat_vs_emulate_random();
    printf("  fma32_mat_vs_emulate_random ... OK\n");

    test_fma64_mat_smoke();
    printf("  fma64_mat_smoke ... OK\n");
    test_fma64_mat_z_base();
    printf("  fma64_mat_z_base ... OK\n");
    test_fma64_mat_accumulation();
    printf("  fma64_mat_accumulation ... OK\n");
    test_fma64_mat_z_row_mask();
    printf("  fma64_mat_z_row_mask ... OK\n");
    test_fma64_mat_vs_emulate_random();
    printf("  fma64_mat_vs_emulate_random ... OK\n");

    test_fms16_mat_smoke();
    printf("  fms16_mat_smoke ... OK\n");
    test_fms16_mat_z_base();
    printf("  fms16_mat_z_base ... OK\n");
    test_fms16_mat_accumulation();
    printf("  fms16_mat_accumulation ... OK\n");
    test_fms16_mat_z_row_mask();
    printf("  fms16_mat_z_row_mask ... OK\n");
    test_fms16_mat_vs_emulate_random();
    printf("  fms16_mat_vs_emulate_random ... OK\n");

    test_fms32_mat_smoke();
    printf("  fms32_mat_smoke ... OK\n");
    test_fms32_mat_z_base();
    printf("  fms32_mat_z_base ... OK\n");
    test_fms32_mat_accumulation();
    printf("  fms32_mat_accumulation ... OK\n");
    test_fms32_mat_z_row_mask();
    printf("  fms32_mat_z_row_mask ... OK\n");
    test_fms32_mat_vs_emulate_random();
    printf("  fms32_mat_vs_emulate_random ... OK\n");

    test_fms64_mat_smoke();
    printf("  fms64_mat_smoke ... OK\n");
    test_fms64_mat_z_base();
    printf("  fms64_mat_z_base ... OK\n");
    test_fms64_mat_accumulation();
    printf("  fms64_mat_accumulation ... OK\n");
    test_fms64_mat_z_row_mask();
    printf("  fms64_mat_z_row_mask ... OK\n");
    test_fms64_mat_vs_emulate_random();
    printf("  fms64_mat_vs_emulate_random ... OK\n");

    printf("PASS\n");
    return 0;
}
