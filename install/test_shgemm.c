#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <riscv_vector.h>

void print_matrix(float *C, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    const int M = 2, N = 2, K = 2;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A[M x K], row-major
    hfloat16 A[4] = {1.0, 2.0,
                     3.0, 4.0};

    // B[K x N], row-major
    hfloat16 B[4] = {5.0, 6.0,
                     7.0, 8.0};

    // C[M x N], row-major
    float C[4] = {0};

    // Call OpenBLAS float16 GEMM
    cblas_shgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K,
                 alpha,
                 A, K,  // lda = K
                 B, N,  // ldb = N
                 beta,
                 C, N); // ldc = N

    printf("Result C = A*B:\n");
    print_matrix(C, M, N);
    return 0;
}

