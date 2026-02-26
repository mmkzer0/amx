#pragma once

#include <stdint.h>

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "AMX intrinsics require Apple Silicon AArch64 (__APPLE__ && __aarch64__)."
#endif

#if defined(__has_include)
#if __has_include("amx_aarch64.h")
#include "amx_aarch64.h"
#elif __has_include("aarch64.h")
#include "aarch64.h"
#else
#error "Missing AMX backend header: expected amx_aarch64.h or aarch64.h."
#endif
#else
#include "aarch64.h"
#endif

static inline void amx_backend_set(void) { AMX_SET(); }
static inline void amx_backend_clr(void) { AMX_CLR(); }

static inline void amx_backend_ldx(uint64_t op) { AMX_LDX(op); }
static inline void amx_backend_ldy(uint64_t op) { AMX_LDY(op); }
static inline void amx_backend_stx(uint64_t op) { AMX_STX(op); }
static inline void amx_backend_sty(uint64_t op) { AMX_STY(op); }
static inline void amx_backend_ldz(uint64_t op) { AMX_LDZ(op); }
static inline void amx_backend_stz(uint64_t op) { AMX_STZ(op); }

static inline void amx_backend_fma16(uint64_t op) { AMX_FMA16(op); }
static inline void amx_backend_fma32(uint64_t op) { AMX_FMA32(op); }
static inline void amx_backend_fma64(uint64_t op) { AMX_FMA64(op); }
static inline void amx_backend_fms16(uint64_t op) { AMX_FMS16(op); }
static inline void amx_backend_fms32(uint64_t op) { AMX_FMS32(op); }
static inline void amx_backend_fms64(uint64_t op) { AMX_FMS64(op); }
