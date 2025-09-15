#!/bin/bash

# Simple RISC-V omatcopy performance comparison script
# Direct comparison of omatcopy_ct.c vs omatcopy_ct_rvv.c

set -e

# Configuration
RISCV_TOOLCHAIN="/qemu/riscv"
CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
CFLAGS="-O3 -march=rv64gcv -static"
TEST_SIZES=(64 128 256 512 1024)
TEST_ITERATIONS=1000

echo "=== Simple RISC-V omatcopy Performance Comparison ==="
echo "Toolchain: ${RISCV_TOOLCHAIN}"
echo "Compiler: ${CC}"
echo "Flags: ${CFLAGS}"
echo

# Check if RISC-V toolchain exists
if [ ! -f "${CC}" ]; then
    echo "Error: RISC-V compiler not found at ${CC}"
    echo "Please ensure RISC-V toolchain is installed at ${RISCV_TOOLCHAIN}"
    exit 1
fi

# Create standalone test for scalar version
cat > test_scalar.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

typedef long BLASLONG;
typedef double FLOAT;

// Scalar implementation (simplified from omatcopy_ct.c)
int omatcopy_ct_scalar(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i, j;
    FLOAT *aptr, *bptr;
    
    if (rows <= 0) return(0);
    if (cols <= 0) return(0);
    
    aptr = a;
    
    if (alpha == 0.0) {
        for (i = 0; i < cols; i++) {
            bptr = &b[i];
            for (j = 0; j < rows; j++) {
                bptr[j * ldb] = 0.0;
            }
        }
        return(0);
    }
    
    if (alpha == 1.0) {
        for (i = 0; i < cols; i++) {
            bptr = &b[i];
            for (j = 0; j < rows; j++) {
                bptr[j * ldb] = aptr[j];
            }
            aptr += lda;
        }
        return(0);
    }
    
    for (i = 0; i < cols; i++) {
        bptr = &b[i];
        for (j = 0; j < rows; j++) {
            bptr[j * ldb] = alpha * aptr[j];
        }
        aptr += lda;
    }
    
    return(0);
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <rows> <cols> <iterations>\n", argv[0]);
        return 1;
    }
    
    BLASLONG rows = atol(argv[1]);
    BLASLONG cols = atol(argv[2]);
    int iterations = atoi(argv[3]);
    
    // Allocate matrices
    FLOAT *a = (FLOAT*)malloc(rows * cols * sizeof(FLOAT));
    FLOAT *b = (FLOAT*)malloc(rows * cols * sizeof(FLOAT));
    
    if (!a || !b) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrix A
    for (BLASLONG i = 0; i < rows * cols; i++) {
        a[i] = (FLOAT)(i % 100) / 10.0;
    }
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        omatcopy_ct_scalar(rows, cols, 1.0, a, rows, b, cols);
    }
    
    // Benchmark
    double start_time = get_time();
    for (int i = 0; i < iterations; i++) {
        omatcopy_ct_scalar(rows, cols, 1.0, a, rows, b, cols);
    }
    double end_time = get_time();
    
    double total_time = end_time - start_time;
    double avg_time = total_time / iterations;
    double gflops = (2.0 * rows * cols * iterations) / (total_time * 1e9);
    
    printf("SCALAR,%ld,%ld,%d,%.6f,%.6f,%.3f\n", 
           rows, cols, iterations, total_time, avg_time, gflops);
    
    free(a);
    free(b);
    return 0;
}
EOF

# Create standalone test for RVV version
cat > test_rvv.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <riscv_vector.h>

typedef long BLASLONG;
typedef double FLOAT;

// RVV macros for double precision
#define VSETVL_MAX              __riscv_vsetvlmax_e64m8()
#define VSETVL(n)               __riscv_vsetvl_e64m8(n)
#define FLOAT_V_T               vfloat64m8_t
#define VLEV_FLOAT              __riscv_vle64_v_f64m8
#define VSEV_FLOAT              __riscv_vse64_v_f64m8
#define VSSEV_FLOAT             __riscv_vsse64_v_f64m8
#define VFMULVF_FLOAT           __riscv_vfmul_vf_f64m8
#define VFMVVF_FLOAT            __riscv_vfmv_v_f_f64m8

// RVV implementation (from omatcopy_ct_rvv.c)
int omatcopy_ct_rvv(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i, j;
    FLOAT *aptr, *bptr;
    size_t vl;
    
    FLOAT_V_T va;
    if (rows <= 0) return(0);
    if (cols <= 0) return(0);
    
    aptr = a;
    
    if (alpha == 0.0) {
        vl = VSETVL_MAX;
        va = VFMVVF_FLOAT(0, vl);
        for (i = 0; i < cols; i++) {
            bptr = &b[i];
            for (j = 0; j < rows; j += vl) {
                vl = VSETVL(rows - j);
                VSSEV_FLOAT(bptr + j * ldb, sizeof(FLOAT) * ldb, va, vl);
            }
        }
        return(0);
    }
    
    if (alpha == 1.0) {
        for (i = 0; i < cols; i++) {
            bptr = &b[i];
            for (j = 0; j < rows; j += vl) {
                vl = VSETVL(rows - j);
                va = VLEV_FLOAT(aptr + j, vl);
                VSSEV_FLOAT(bptr + j * ldb, sizeof(FLOAT) * ldb, va, vl);
            }
            aptr += lda;
        }
        return(0);
    }
    
    for (i = 0; i < cols; i++) {
        bptr = &b[i];
        for (j = 0; j < rows; j += vl) {
            vl = VSETVL(rows - j);
            va = VLEV_FLOAT(aptr + j, vl);
            va = VFMULVF_FLOAT(va, alpha, vl);
            VSSEV_FLOAT(bptr + j * ldb, sizeof(FLOAT) * ldb, va, vl);
        }
        aptr += lda;
    }
    
    return(0);
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <rows> <cols> <iterations>\n", argv[0]);
        return 1;
    }
    
    BLASLONG rows = atol(argv[1]);
    BLASLONG cols = atol(argv[2]);
    int iterations = atoi(argv[3]);
    
    // Allocate matrices
    FLOAT *a = (FLOAT*)malloc(rows * cols * sizeof(FLOAT));
    FLOAT *b = (FLOAT*)malloc(rows * cols * sizeof(FLOAT));
    
    if (!a || !b) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrix A
    for (BLASLONG i = 0; i < rows * cols; i++) {
        a[i] = (FLOAT)(i % 100) / 10.0;
    }
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        omatcopy_ct_rvv(rows, cols, 1.0, a, rows, b, cols);
    }
    
    // Benchmark
    double start_time = get_time();
    for (int i = 0; i < iterations; i++) {
        omatcopy_ct_rvv(rows, cols, 1.0, a, rows, b, cols);
    }
    double end_time = get_time();
    
    double total_time = end_time - start_time;
    double avg_time = total_time / iterations;
    double gflops = (2.0 * rows * cols * iterations) / (total_time * 1e9);
    
    printf("RVV,%ld,%ld,%d,%.6f,%.6f,%.3f\n", 
           rows, cols, iterations, total_time, avg_time, gflops);
    
    free(a);
    free(b);
    return 0;
}
EOF

echo "Building standalone test programs..."

# Compile scalar version
echo "Compiling scalar version..."
${CC} ${CFLAGS} -o test_scalar test_scalar.c
if [ $? -ne 0 ]; then
    echo "Failed to compile scalar version"
    exit 1
fi

# Compile RVV version
echo "Compiling RVV version..."
${CC} ${CFLAGS} -o test_rvv test_rvv.c
if [ $? -ne 0 ]; then
    echo "Failed to compile RVV version"
    exit 1
fi

echo "Compilation successful!"
echo

# Create results file
RESULTS_FILE="omatcopy_benchmark_results.csv"
echo "Version,Rows,Cols,Iterations,TotalTime(s),AvgTime(s),GFLOPS" > ${RESULTS_FILE}

echo "Running benchmarks..."
echo "Results will be saved to: ${RESULTS_FILE}"
echo
echo "Format: Version,Rows,Cols,Iterations,TotalTime(s),AvgTime(s),GFLOPS"
echo "----------------------------------------"

# Run benchmarks for different matrix sizes
for size in "${TEST_SIZES[@]}"; do
    echo "Testing ${size}x${size} matrices..."
    
    # Test scalar version
    ./test_scalar ${size} ${size} ${TEST_ITERATIONS} | tee -a ${RESULTS_FILE}
    
    # Test RVV version
    ./test_rvv ${size} ${size} ${TEST_ITERATIONS} | tee -a ${RESULTS_FILE}
    
    echo
done

echo "Benchmark completed!"
echo "Results saved to: ${RESULTS_FILE}"
echo
echo "To transfer files to sg2044:"
echo "1. Copy the compiled binaries and results:"
echo "   scp test_scalar test_rvv ${RESULTS_FILE} user@sg2044:/path/to/test/"
echo "2. Run on sg2044:"
echo "   ./test_scalar 1024 1024 1000"
echo "   ./test_rvv 1024 1024 1000"
echo
echo "Binary information:"
file test_scalar test_rvv 2>/dev/null || echo "file command not available"
ls -lh test_scalar test_rvv

# Clean up source files
rm -f test_scalar.c test_rvv.c

echo
echo "Script completed successfully!"