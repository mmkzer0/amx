test: a.out
	./a.out

a.out: test.c emulate.h aarch64.h ldst.c extr.c fma.c fms.c genlut.c mac16.c matfp.c matint.c vecfp.c vecint.c
	gcc -O2 -g test.c ldst.c extr.c fma.c fms.c genlut.c mac16.c matfp.c matint.c vecfp.c vecint.c

perf: perf_kernels.py perf.c aarch64.h
	python3 perf_kernels.py
	gcc -O2 -g -o perf perf.c perf_kernels.c

amx_intrinsics_v1_test: amx_intrinsics_v1_test.c amx_intrinsics.h amx_backend_aarch64.h aarch64.h
	gcc -O2 -g -o amx_intrinsics_v1_test amx_intrinsics_v1_test.c -lm

test_intrinsics_v1: amx_intrinsics_v1_test
	./amx_intrinsics_v1_test

.PHONY: test test_intrinsics_v1
