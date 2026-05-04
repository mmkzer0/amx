#pragma once

#include <stdint.h>

#include "amx_backend_aarch64.h"

#define AMX_INTR_OP_TYPE(name) \
    typedef struct name { \
        uint64_t bits; \
    } name

AMX_INTR_OP_TYPE(amx_ldx_op_t);
AMX_INTR_OP_TYPE(amx_ldy_op_t);
AMX_INTR_OP_TYPE(amx_stx_op_t);
AMX_INTR_OP_TYPE(amx_sty_op_t);
AMX_INTR_OP_TYPE(amx_ldz_op_t);
AMX_INTR_OP_TYPE(amx_stz_op_t);

AMX_INTR_OP_TYPE(amx_fma16_op_t);
AMX_INTR_OP_TYPE(amx_fma32_op_t);
AMX_INTR_OP_TYPE(amx_fma64_op_t);
AMX_INTR_OP_TYPE(amx_fms16_op_t);
AMX_INTR_OP_TYPE(amx_fms32_op_t);
AMX_INTR_OP_TYPE(amx_fms64_op_t);

typedef struct __attribute__((aligned(64))) {
    uint8_t lane[64];
} amx_u8x64_t;

typedef struct __attribute__((aligned(64))) {
    _Float16 lane[32];
} amx_f16x32_t;

typedef struct __attribute__((aligned(64))) {
    float lane[16];
} amx_f32x16_t;

typedef struct __attribute__((aligned(64))) {
    double lane[8];
} amx_f64x8_t;

typedef struct amx_scope_t {
    unsigned active;
} amx_scope_t;

static inline uint64_t amx_intr_ptr56(const void* ptr) {
    return ((uint64_t)(uintptr_t)ptr) & ((1ull << 56) - 1);
}

static inline amx_scope_t amx_scope_begin(void) {
    amx_backend_set();
    return (amx_scope_t){1u};
}

static inline void amx_scope_end(amx_scope_t* scope) {
    if (scope != 0 && scope->active) {
        amx_backend_clr();
        scope->active = 0u;
    }
}

static inline amx_ldx_op_t amx_ldx_op_raw(uint64_t bits) { return (amx_ldx_op_t){bits}; }
static inline amx_ldy_op_t amx_ldy_op_raw(uint64_t bits) { return (amx_ldy_op_t){bits}; }
static inline amx_stx_op_t amx_stx_op_raw(uint64_t bits) { return (amx_stx_op_t){bits}; }
static inline amx_sty_op_t amx_sty_op_raw(uint64_t bits) { return (amx_sty_op_t){bits}; }
static inline amx_ldz_op_t amx_ldz_op_raw(uint64_t bits) { return (amx_ldz_op_t){bits}; }
static inline amx_stz_op_t amx_stz_op_raw(uint64_t bits) { return (amx_stz_op_t){bits}; }
static inline amx_fma16_op_t amx_fma16_op_raw(uint64_t bits) { return (amx_fma16_op_t){bits}; }
static inline amx_fma32_op_t amx_fma32_op_raw(uint64_t bits) { return (amx_fma32_op_t){bits}; }
static inline amx_fma64_op_t amx_fma64_op_raw(uint64_t bits) { return (amx_fma64_op_t){bits}; }
static inline amx_fms16_op_t amx_fms16_op_raw(uint64_t bits) { return (amx_fms16_op_t){bits}; }
static inline amx_fms32_op_t amx_fms32_op_raw(uint64_t bits) { return (amx_fms32_op_t){bits}; }
static inline amx_fms64_op_t amx_fms64_op_raw(uint64_t bits) { return (amx_fms64_op_t){bits}; }

static inline amx_ldx_op_t amx_ldx_op_single(const void* ptr, uint8_t x_row) {
    return (amx_ldx_op_t){amx_intr_ptr56(ptr) | (((uint64_t)x_row & 0x7ull) << 56)};
}

static inline amx_ldy_op_t amx_ldy_op_single(const void* ptr, uint8_t y_row) {
    return (amx_ldy_op_t){amx_intr_ptr56(ptr) | (((uint64_t)y_row & 0x7ull) << 56)};
}

static inline amx_stx_op_t amx_stx_op_single(void* ptr, uint8_t x_row) {
    return (amx_stx_op_t){amx_intr_ptr56(ptr) | (((uint64_t)x_row & 0x7ull) << 56)};
}

static inline amx_sty_op_t amx_sty_op_single(void* ptr, uint8_t y_row) {
    return (amx_sty_op_t){amx_intr_ptr56(ptr) | (((uint64_t)y_row & 0x7ull) << 56)};
}

static inline amx_ldz_op_t amx_ldz_op_single(const void* ptr, uint8_t z_row) {
    return (amx_ldz_op_t){amx_intr_ptr56(ptr) | (((uint64_t)z_row & 0x3Full) << 56)};
}

static inline amx_stz_op_t amx_stz_op_single(void* ptr, uint8_t z_row) {
    return (amx_stz_op_t){amx_intr_ptr56(ptr) | (((uint64_t)z_row & 0x3Full) << 56)};
}

static inline uint64_t amx_intr_fma_like_bits(uint8_t vector_mode, uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return ((uint64_t)(vector_mode & 1u) << 63) |
           (((uint64_t)z_row & 0x3Full) << 20) |
           (((uint64_t)x_off_b & 0x1FFull) << 10) |
           ((uint64_t)y_off_b & 0x1FFull);
}

static inline amx_fma16_op_t amx_fma16_op_vec_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fma16_op_t){amx_intr_fma_like_bits(1u, z_row, x_off_b, y_off_b)};
}

static inline amx_fma16_op_t amx_fma16_op_mat_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fma16_op_t){amx_intr_fma_like_bits(0u, z_row, x_off_b, y_off_b)};
}

static inline amx_fma32_op_t amx_fma32_op_vec_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fma32_op_t){amx_intr_fma_like_bits(1u, z_row, x_off_b, y_off_b)};
}

static inline amx_fma32_op_t amx_fma32_op_mat_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fma32_op_t){amx_intr_fma_like_bits(0u, z_row, x_off_b, y_off_b)};
}

static inline amx_fma64_op_t amx_fma64_op_vec_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fma64_op_t){amx_intr_fma_like_bits(1u, z_row, x_off_b, y_off_b)};
}

static inline amx_fma64_op_t amx_fma64_op_mat_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fma64_op_t){amx_intr_fma_like_bits(0u, z_row, x_off_b, y_off_b)};
}

static inline amx_fms16_op_t amx_fms16_op_vec_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fms16_op_t){amx_intr_fma_like_bits(1u, z_row, x_off_b, y_off_b)};
}

static inline amx_fms16_op_t amx_fms16_op_mat_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fms16_op_t){amx_intr_fma_like_bits(0u, z_row, x_off_b, y_off_b)};
}

static inline amx_fms32_op_t amx_fms32_op_vec_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fms32_op_t){amx_intr_fma_like_bits(1u, z_row, x_off_b, y_off_b)};
}

static inline amx_fms32_op_t amx_fms32_op_mat_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fms32_op_t){amx_intr_fma_like_bits(0u, z_row, x_off_b, y_off_b)};
}

static inline amx_fms64_op_t amx_fms64_op_vec_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fms64_op_t){amx_intr_fma_like_bits(1u, z_row, x_off_b, y_off_b)};
}

static inline amx_fms64_op_t amx_fms64_op_mat_acc(uint8_t z_row, uint16_t x_off_b, uint16_t y_off_b) {
    return (amx_fms64_op_t){amx_intr_fma_like_bits(0u, z_row, x_off_b, y_off_b)};
}

static inline void amx_ldx(amx_ldx_op_t op) { amx_backend_ldx(op.bits); }
static inline void amx_ldy(amx_ldy_op_t op) { amx_backend_ldy(op.bits); }
static inline void amx_stx(amx_stx_op_t op) { amx_backend_stx(op.bits); }
static inline void amx_sty(amx_sty_op_t op) { amx_backend_sty(op.bits); }
static inline void amx_ldz(amx_ldz_op_t op) { amx_backend_ldz(op.bits); }
static inline void amx_stz(amx_stz_op_t op) { amx_backend_stz(op.bits); }

static inline void amx_fma16(amx_fma16_op_t op) { amx_backend_fma16(op.bits); }
static inline void amx_fma32(amx_fma32_op_t op) { amx_backend_fma32(op.bits); }
static inline void amx_fma64(amx_fma64_op_t op) { amx_backend_fma64(op.bits); }
static inline void amx_fms16(amx_fms16_op_t op) { amx_backend_fms16(op.bits); }
static inline void amx_fms32(amx_fms32_op_t op) { amx_backend_fms32(op.bits); }
static inline void amx_fms64(amx_fms64_op_t op) { amx_backend_fms64(op.bits); }

#undef AMX_INTR_OP_TYPE
