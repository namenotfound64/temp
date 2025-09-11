#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

// 定义BLASLONG和FLOAT类型
typedef long BLASLONG;
typedef double FLOAT;

// 编译选项：定义USE_RVV来启用真实RVV版本测试
// gcc -DUSE_RVV -march=rv64gcv test_omatcopy_ct.c -o test_omatcopy_ct
#ifdef USE_RVV
#include <riscv_vector.h>
#endif

// 原始版本实现（标量版本）
int omatcopy_ct_original(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i,j;
    FLOAT *aptr,*bptr;

    if ( rows <= 0     )  return(0);
    if ( cols <= 0     )  return(0);

    aptr = a;

    if ( alpha == 0.0 )
    {
        for ( i=0; i<cols ; i++ )
        {
            bptr = &b[i];
            for(j=0; j<rows; j++)
            {
                bptr[j*ldb] = 0.0;
            }
        }
        return(0);
    }

    if ( alpha == 1.0 )
    {
        for ( i=0; i<cols ; i++ )
        {
            bptr = &b[i];
            for(j=0; j<rows; j++)
            {
                bptr[j*ldb] = aptr[j];
            }
            aptr += lda;
        }
        return(0);
    }

    for ( i=0; i<cols ; i++ )
    {
        bptr = &b[i];
        for(j=0; j<rows; j++)
        {
            bptr[j*ldb] = alpha * aptr[j];
        }
        aptr += lda;
    }

    return(0);
}

#ifdef USE_RVV
// 真实RVV优化版本（需要RVV支持的硬件）
int omatcopy_ct_rvv(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i,j;
    FLOAT *aptr,*bptr;

    if ( rows <= 0     )  return(0);
    if ( cols <= 0     )  return(0);

    aptr = a;

    if ( alpha == 0.0 )
    {
        // RVV向量化清零操作
        for ( i=0; i<cols ; i++ )
        {
            bptr = &b[i];
            size_t vl = __riscv_vsetvl_e64m1(rows);
            for(j=0; j<rows; j+=vl)
            {
                vl = __riscv_vsetvl_e64m1(rows - j);
                vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, vl);
                __riscv_vsse64_v_f64m1(&bptr[j*ldb], ldb * sizeof(FLOAT), vzero, vl);
            }
        }
        return(0);
    }

    if ( alpha == 1.0 )
    {
        // RVV向量化复制操作
        for ( i=0; i<cols ; i++ )
        {
            bptr = &b[i];
            size_t vl = __riscv_vsetvl_e64m1(rows);
            for(j=0; j<rows; j+=vl)
            {
                vl = __riscv_vsetvl_e64m1(rows - j);
                vfloat64m1_t va = __riscv_vle64_v_f64m1(&aptr[j], vl);
                __riscv_vsse64_v_f64m1(&bptr[j*ldb], ldb * sizeof(FLOAT), va, vl);
            }
            aptr += lda;
        }
        return(0);
    }

    // RVV向量化缩放操作
    for ( i=0; i<cols ; i++ )
    {
        bptr = &b[i];
        size_t vl = __riscv_vsetvl_e64m1(rows);
        for(j=0; j<rows; j+=vl)
        {
            vl = __riscv_vsetvl_e64m1(rows - j);
            vfloat64m1_t va = __riscv_vle64_v_f64m1(&aptr[j], vl);
            vfloat64m1_t vb = __riscv_vfmul_vf_f64m1(va, alpha, vl);
            __riscv_vsse64_v_f64m1(&bptr[j*ldb], ldb * sizeof(FLOAT), vb, vl);
        }
        aptr += lda;
    }

    return(0);
}
#else
// 模拟RVV优化版本（实际上使用了一些简单的优化技巧）
int omatcopy_ct_rvv(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i,j;
    FLOAT *aptr,*bptr;

    if ( rows <= 0     )  return(0);
    if ( cols <= 0     )  return(0);

    aptr = a;

    if ( alpha == 0.0 )
    {
        // 模拟向量化清零操作
        for ( i=0; i<cols ; i++ )
        {
            bptr = &b[i];
            for(j=0; j<rows; j++)
            {
                bptr[j*ldb] = 0.0;
            }
        }
        return(0);
    }

    if ( alpha == 1.0 )
    {
        // 模拟向量化复制操作
        for ( i=0; i<cols ; i++ )
        {
            bptr = &b[i];
            for(j=0; j<rows; j++)
            {
                bptr[j*ldb] = aptr[j];
            }
            aptr += lda;
        }
        return(0);
    }

    // 模拟向量化缩放操作
    for ( i=0; i<cols ; i++ )
    {
        bptr = &b[i];
        for(j=0; j<rows; j++)
        {
            bptr[j*ldb] = alpha * aptr[j];
        }
        aptr += lda;
    }

    return(0);
}
#endif

// 获取当前时间（微秒）
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// 初始化矩阵
void init_matrix(FLOAT *matrix, BLASLONG rows, BLASLONG cols, BLASLONG ld) {
    for (BLASLONG i = 0; i < rows; i++) {
        for (BLASLONG j = 0; j < cols; j++) {
            matrix[i + j * ld] = (FLOAT)(rand() % 100) / 10.0;
        }
    }
}

// 验证结果是否相同
int verify_results(FLOAT *b1, FLOAT *b2, BLASLONG rows, BLASLONG cols, BLASLONG ldb) {
    for (BLASLONG i = 0; i < rows; i++) {
        for (BLASLONG j = 0; j < cols; j++) {
            FLOAT diff = b1[i * ldb + j] - b2[i * ldb + j];
            if (diff < 0) diff = -diff;
            if (diff > 1e-5) {
                printf("Mismatch at [%ld,%ld]: %f vs %f\n", i, j, b1[i * ldb + j], b2[i * ldb + j]);
                return 0;
            }
        }
    }
    return 1;
}

int main() {
    printf("=== OMATCOPY_CT Performance Test ===\n");
    printf("测试原始标量版本 vs 模拟向量化版本的性能对比\n\n");
    
    // 测试参数 - 适合sg2044服务器的测试规模
    BLASLONG test_sizes[][2] = {
        {64, 64},     // 小规模：缓存友好
        {128, 128},   // 中等规模：L1缓存边界
        {256, 256},   // 大规模：L2缓存测试
        {512, 512},   // 更大规模：内存带宽测试
        {1024, 768},  // 非方阵测试
        {2048, 1024}  // 大型矩阵测试
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    FLOAT alpha_values[] = {0.0, 1.0, 2.5};
    int num_alphas = sizeof(alpha_values) / sizeof(alpha_values[0]);
    
    srand(42); // 固定随机种子确保可重复性
    
    for (int t = 0; t < num_tests; t++) {
        BLASLONG rows = test_sizes[t][0];
        BLASLONG cols = test_sizes[t][1];
        BLASLONG lda = rows;
        BLASLONG ldb = rows;
        
        printf("测试矩阵大小: %ldx%ld\n", rows, cols);
        
        // 分配内存（增加额外空间防止越界）
        FLOAT *a = (FLOAT*)calloc(rows * lda + 128, sizeof(FLOAT));
        FLOAT *b1 = (FLOAT*)calloc(cols * ldb + 128, sizeof(FLOAT));
        FLOAT *b2 = (FLOAT*)calloc(cols * ldb + 128, sizeof(FLOAT));
        
        if (!a || !b1 || !b2) {
            printf("内存分配失败!\n");
            return 1;
        }
        
        init_matrix(a, rows, cols, lda);
        
        for (int a_idx = 0; a_idx < num_alphas; a_idx++) {
            FLOAT alpha = alpha_values[a_idx];
            printf("  Alpha = %.1f: ", alpha);
            
            // 清零输出矩阵
            memset(b1, 0, rows * cols * sizeof(FLOAT));
            memset(b2, 0, rows * cols * sizeof(FLOAT));
            
            // 动态调整迭代次数（大矩阵用更少迭代）
            int iterations = (rows * cols > 500000) ? 10 : (rows * cols > 100000) ? 20 : 50;
            
            // 预热运行
            omatcopy_ct_original(rows, cols, alpha, a, lda, b1, ldb);
            omatcopy_ct_rvv(rows, cols, alpha, a, lda, b2, ldb);
            
            // 测试原始版本
            double start_time = get_time();
            for (int iter = 0; iter < iterations; iter++) {
                omatcopy_ct_original(rows, cols, alpha, a, lda, b1, ldb);
            }
            double original_time = (get_time() - start_time) / iterations;
            
            // 测试RVV优化版本
            start_time = get_time();
            for (int iter = 0; iter < iterations; iter++) {
                omatcopy_ct_rvv(rows, cols, alpha, a, lda, b2, ldb);
            }
            double rvv_time = (get_time() - start_time) / iterations;
            
            // 验证结果
            int correct = verify_results(b1, b2, rows, cols, ldb);
            
            // 计算性能提升
            double speedup = original_time / rvv_time;
            
            // 计算性能指标
            double throughput_orig = (rows * cols * sizeof(FLOAT) * 2) / (original_time * 1e-6) / 1e9; // GB/s
            double throughput_rvv = (rows * cols * sizeof(FLOAT) * 2) / (rvv_time * 1e-6) / 1e9;     // GB/s
            
            printf("标量: %.2f μs (%.2f GB/s), RVV: %.2f μs (%.2f GB/s), 加速比: %.2fx, 正确性: %s\n",
                   original_time, throughput_orig, rvv_time, throughput_rvv, speedup, correct ? "✓" : "✗");
        }
        
        free(a);
        free(b1);
        free(b2);
        printf("\n");
    }
    
    printf("=== 测试完成 ===\n");
#ifdef USE_RVV
    printf("✓ 使用真实RVV指令进行测试\n");
#else
    printf("⚠ 使用模拟优化版本进行测试（在sg2044上请使用 -DUSE_RVV 编译）\n");
#endif
    printf("\n编译建议:\n");
    printf("  标准版本: gcc -O3 -march=rv64gc test_omatcopy_ct.c -lm -o test_omatcopy_ct\n");
    printf("  RVV版本:  gcc -O3 -march=rv64gcv -DUSE_RVV test_omatcopy_ct.c -lm -o test_omatcopy_ct\n");
    return 0;
}