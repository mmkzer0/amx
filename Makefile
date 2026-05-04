BUILD_DIR := build

# --- test ---
test: $(BUILD_DIR)/a.out
	$(BUILD_DIR)/a.out

$(BUILD_DIR)/a.out: test.c emulate.h aarch64.h ldst.c extr.c fma.c fms.c genlut.c mac16.c matfp.c matint.c vecfp.c vecint.c | $(BUILD_DIR)
	gcc -O2 -g -o $@ test.c ldst.c extr.c fma.c fms.c genlut.c mac16.c matfp.c matint.c vecfp.c vecint.c

# --- perf ---
perf: $(BUILD_DIR)/perf
	python3 perf_kernels.py
	gcc -O2 -g -o $(BUILD_DIR)/perf perf.c perf_kernels.c

$(BUILD_DIR)/perf: perf.c perf_kernels.c aarch64.h | $(BUILD_DIR)
	gcc -O2 -g -o $@ perf.c perf_kernels.c

# --- amx_intrinsics_v1_test ---
$(BUILD_DIR)/amx_intrinsics_v1_test: amx_intrinsics_v1_test.c amx_intrinsics.h amx_backend_aarch64.h aarch64.h | $(BUILD_DIR)
	gcc -O2 -g -o $@ amx_intrinsics_v1_test.c -lm

test_intrinsics_v1: $(BUILD_DIR)/amx_intrinsics_v1_test
	$<

# --- clean ---
clean:
	rm -rf $(BUILD_DIR)

# --- helpers ---
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: test test_intrinsics_v1 perf clean
