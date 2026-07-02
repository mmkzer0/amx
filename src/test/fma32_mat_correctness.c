#include "amx_intrinsics.h"
#include "emulate.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(_Alignof(amx_f32x16_t) >= 64, "amx_f32x16_t must be 64-byte aligned");

extern void emulate_AMX_FMA32(amx_state* state, uint64_t operand);

static const float EPSILON = 1e-5f;

/* ─── deterministic RNG (same LCG as amx_intrinsics_v1_test) ─────────────── */

static uint32_t rng_state = 0x12345678u;

static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static float rng_f32(void) {
    return (float)((int32_t)(rng_next() >> 8)) * (1.0f / 16777216.0f);
}

/* ─── helpers ────────────────────────────────────────────────────────────── */

static void fill_z_tile(amx_f32x16_t z_tile[16], float value) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            z_tile[j].lane[i] = value;
        }
    }
}

static void assert_tile_outer_product(const amx_f32x16_t z_tile[16],
                                      const amx_f32x16_t* x,
                                      const amx_f32x16_t* y,
                                      float z_addend) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            float expected = z_addend + x->lane[i] * y->lane[j];
            float err = fabsf(z_tile[j].lane[i] - expected);
            assert(err < EPSILON);
        }
    }
}

static void assert_tiles_equal(const amx_f32x16_t* a, const amx_f32x16_t* b) {
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            float err = fabsf(a[j].lane[i] - b[j].lane[i]);
            assert(err < EPSILON);
        }
    }
}

static void run_hw_fma32_mat(const amx_f32x16_t* x,
                             const amx_f32x16_t* y,
                             amx_f32x16_t z_tile[16],
                             uint8_t z_base) {
    amx_scope_t scope = amx_scope_begin();
    amx_ldx(amx_ldx_op_single(x->lane, 0));
    amx_ldy(amx_ldy_op_single(y->lane, 0));
    for (int j = 0; j < 16; ++j) {
        uint8_t amx_row = (uint8_t)((j * 4) + z_base);
        amx_ldz(amx_ldz_op_single(z_tile[j].lane, amx_row));
    }
    amx_fma32(amx_fma32_op_mat_acc(z_base, 0, 0));
    for (int j = 0; j < 16; ++j) {
        uint8_t amx_row = (uint8_t)((j * 4) + z_base);
        amx_stz(amx_stz_op_single(z_tile[j].lane, amx_row));
    }
    amx_scope_end(&scope);
}

static void run_emulate_fma32_mat(const amx_f32x16_t* x,
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

static void setup_smoke_xy(amx_f32x16_t* x, amx_f32x16_t* y) {
    for (int i = 0; i < 16; ++i) {
        x->lane[i] = (float)(i + 1);
        y->lane[i] = 2.0f;
    }
}

/* ─── tests ──────────────────────────────────────────────────────────────── */

static void test_smoke_tile(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];

    setup_smoke_xy(&x, &y);
    fill_z_tile(z_tile, 0.0f);
    run_hw_fma32_mat(&x, &y, z_tile, 0);
    assert_tile_outer_product(z_tile, &x, &y, 0.0f);
}

static void test_z_base(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];

    setup_smoke_xy(&x, &y);
    for (uint8_t z_base = 1; z_base <= 3; ++z_base) {
        fill_z_tile(z_tile, 0.0f);
        run_hw_fma32_mat(&x, &y, z_tile, z_base);
        assert_tile_outer_product(z_tile, &x, &y, 0.0f);
    }
}

static void test_accumulation(void) {
    amx_f32x16_t x, y;
    amx_f32x16_t z_tile[16];

    setup_smoke_xy(&x, &y);
    fill_z_tile(z_tile, 1.0f);
    run_hw_fma32_mat(&x, &y, z_tile, 0);
    assert_tile_outer_product(z_tile, &x, &y, 1.0f);
}

static void test_mat_z_row_mask(void) {
    assert(amx_fma32_op_mat_acc(4, 0, 0).bits == amx_fma32_op_mat_acc(0, 0, 0).bits);
    assert(amx_fma32_op_mat_acc(5, 0, 0).bits == amx_fma32_op_mat_acc(1, 0, 0).bits);
}

static void test_vs_emulate_random(void) {
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
                float v = rng_f32();
                z_hw[j].lane[i] = v;
                z_emu[j].lane[i] = v;
            }
        }

        uint8_t z_base = (uint8_t)(rng_next() & 3u);

        run_hw_fma32_mat(&x, &y, z_hw, z_base);
        run_emulate_fma32_mat(&x, &y, z_emu, z_base);
        assert_tiles_equal(z_hw, z_emu);
    }
}

/* ─── main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== fma32_mat_correctness ===\n");

    test_smoke_tile();
    printf("  test_smoke_tile ... OK\n");

    test_z_base();
    printf("  test_z_base ... OK\n");

    test_accumulation();
    printf("  test_accumulation ... OK\n");

    test_mat_z_row_mask();
    printf("  test_mat_z_row_mask ... OK\n");

    test_vs_emulate_random();
    printf("  test_vs_emulate_random ... OK\n");

    printf("PASS\n");
    return 0;
}
