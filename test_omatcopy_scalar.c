#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define BLASLONG long
#define FLOAT double

// Include scalar implementation
#include "kernel/riscv64/omatcopy_ct.c"

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
        CNAME(rows, cols, 1.0, a, rows, b, cols);
    }
    
    // Benchmark
    double start_time = get_time();
    for (int i = 0; i < iterations; i++) {
        CNAME(rows, cols, 1.0, a, rows, b, cols);
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
