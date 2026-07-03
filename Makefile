BUILD_DIR := build

CFLAGS := -O2 -g -Isrc/headers

# --- test ---
test: $(BUILD_DIR)/a.out
	$(BUILD_DIR)/a.out

$(BUILD_DIR)/a.out: src/test/test.c \
		src/emulate/ldst.c src/emulate/extr.c \
		src/emulate/fma.c src/emulate/fms.c \
		src/emulate/genlut.c src/emulate/mac16.c \
		src/emulate/matfp.c src/emulate/matint.c \
		src/emulate/vecfp.c src/emulate/vecint.c \
		src/headers/emulate.h src/headers/aarch64.h \
		| $(BUILD_DIR)
	gcc $(CFLAGS) -o $@ \
		src/test/test.c \
		src/emulate/ldst.c src/emulate/extr.c \
		src/emulate/fma.c src/emulate/fms.c \
		src/emulate/genlut.c src/emulate/mac16.c \
		src/emulate/matfp.c src/emulate/matint.c \
		src/emulate/vecfp.c src/emulate/vecint.c

# --- perf ---
perf: $(BUILD_DIR)/perf

$(BUILD_DIR)/perf: src/perf/perf.c src/headers/aarch64.h | $(BUILD_DIR)
	python3 src/perf/perf_kernels.py $(BUILD_DIR)/perf_kernels.c
	gcc $(CFLAGS) -o $@ src/perf/perf.c $(BUILD_DIR)/perf_kernels.c

# --- amx_intrinsics_v1_test ---
$(BUILD_DIR)/amx_intrinsics_v1_test: \
		src/test/amx_intrinsics_v1_test.c \
		src/headers/amx_intrinsics.h \
		src/headers/amx_backend_aarch64.h \
		src/headers/aarch64.h \
		| $(BUILD_DIR)
	gcc $(CFLAGS) -o $@ src/test/amx_intrinsics_v1_test.c -lm

test_intrinsics_v1: $(BUILD_DIR)/amx_intrinsics_v1_test
	$<

# --- amx_mat_correctness ---
$(BUILD_DIR)/amx_mat_correctness: \
		src/test/amx_mat_correctness.c \
		src/emulate/fma.c \
		src/emulate/fms.c \
		src/headers/amx_intrinsics.h \
		src/headers/amx_backend_aarch64.h \
		src/headers/aarch64.h \
		src/headers/emulate.h \
		| $(BUILD_DIR)
	gcc $(CFLAGS) -o $@ src/test/amx_mat_correctness.c src/emulate/fma.c src/emulate/fms.c -lm

test_mat: $(BUILD_DIR)/amx_mat_correctness
	$<

# Alias for backward compatibility
test_fma32_mat: test_mat

# --- unified correctness gate ---
test_all: test test_intrinsics_v1 test_mat
	@echo "test_all: PASS"

.PHONY: test test_intrinsics_v1 test_mat test_fma32_mat test_all perf clean

# --- clean ---
clean:
	rm -rf $(BUILD_DIR)

# --- helpers ---
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
