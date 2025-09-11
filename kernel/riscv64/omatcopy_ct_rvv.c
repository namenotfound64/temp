/***************************************************************************
Copyright (c) 2013, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "common.h"
#include <stdio.h>

#if !defined(DOUBLE)
#define VSETVL_MAX				__riscv_vsetvlmax_e32m8()
#define VSETVL(n)               __riscv_vsetvl_e32m8(n)
#define FLOAT_V_T               vfloat32m8_t
#define VLEV_FLOAT              __riscv_vle32_v_f32m8
#define VSEV_FLOAT              __riscv_vse32_v_f32m8
#define VLSEV_FLOAT             __riscv_vlse32_v_f32m8
#define VSSEV_FLOAT             __riscv_vsse32_v_f32m8
#define VFMULVF_FLOAT           __riscv_vfmul_vf_f32m8
#define VFMVVF_FLOAT            __riscv_vfmv_v_f_f32m8
#else
#define VSETVL_MAX				__riscv_vsetvlmax_e64m8()
#define VSETVL(n)               __riscv_vsetvl_e64m8(n)
#define FLOAT_V_T               vfloat64m8_t
#define VLEV_FLOAT              __riscv_vle64_v_f64m8
#define VSEV_FLOAT              __riscv_vse64_v_f64m8
#define VLSEV_FLOAT             __riscv_vlse64_v_f64m8
#define VSSEV_FLOAT             __riscv_vsse64_v_f64m8
#define VFMULVF_FLOAT           __riscv_vfmul_vf_f64m8
#define VFMVVF_FLOAT            __riscv_vfmv_v_f_f64m8
#endif

/*****************************************************
 * Order ColMajor
 * Trans with RVV optimization
 * Optimized version with:
 * - Block processing for cache efficiency
 * - Loop unrolling for better ILP
 * - Reduced VSETVL overhead
 * - Software prefetching
******************************************************/

// Block size for cache-friendly processing
#define BLOCK_SIZE_ROWS 256
#define BLOCK_SIZE_COLS 64

// Fast path for small matrices
static inline int small_matrix_transpose(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
	if (rows <= 8 && cols <= 8) {
		// Optimized 8x8 or smaller transpose
		for (BLASLONG i = 0; i < cols; i++) {
			for (BLASLONG j = 0; j < rows; j++) {
				b[j * ldb + i] = alpha * a[i * lda + j];
			}
		}
		return 1;
	}
	return 0;
}

int CNAME(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
	BLASLONG i, j, ii, jj;
	FLOAT *aptr, *bptr;
	size_t vl, vl_max;
	FLOAT_V_T va, vb, va2, va3, va4;

	if (rows <= 0) return(0);
	if (cols <= 0) return(0);

	// Try small matrix fast path
	if (small_matrix_transpose(rows, cols, alpha, a, lda, b, ldb)) {
		return(0);
	}

	// Get maximum vector length once
	vl_max = VSETVL_MAX;

	if (alpha == 0.0)
	{
		va = VFMVVF_FLOAT(0, vl_max);
		// Block processing for better cache locality
		for (ii = 0; ii < cols; ii += BLOCK_SIZE_COLS) {
			BLASLONG col_end = (ii + BLOCK_SIZE_COLS < cols) ? ii + BLOCK_SIZE_COLS : cols;
			for (jj = 0; jj < rows; jj += BLOCK_SIZE_ROWS) {
				BLASLONG row_end = (jj + BLOCK_SIZE_ROWS < rows) ? jj + BLOCK_SIZE_ROWS : rows;
				
				for (i = ii; i < col_end; i++) {
					bptr = &b[i + jj * ldb];
					BLASLONG remaining = row_end - jj;
					
					// Main loop with reduced VSETVL calls
					for (j = 0; j < remaining; j += vl_max) {
						vl = (j + vl_max <= remaining) ? vl_max : VSETVL(remaining - j);
						VSSEV_FLOAT(bptr + j * ldb, sizeof(FLOAT) * ldb, va, vl);
					}
				}
			}
		}
		return(0);
	}

	if (alpha == 1.0)
	{
		// Block processing with loop unrolling
		for (ii = 0; ii < cols; ii += BLOCK_SIZE_COLS) {
			BLASLONG col_end = (ii + BLOCK_SIZE_COLS < cols) ? ii + BLOCK_SIZE_COLS : cols;
			for (jj = 0; jj < rows; jj += BLOCK_SIZE_ROWS) {
				BLASLONG row_end = (jj + BLOCK_SIZE_ROWS < rows) ? jj + BLOCK_SIZE_ROWS : rows;
				
				// Process 4 columns at once when possible
				for (i = ii; i < col_end - 3; i += 4) {
					aptr = &a[i * lda + jj];
					FLOAT *bptr1 = &b[i + jj * ldb];
					FLOAT *bptr2 = &b[i + 1 + jj * ldb];
					FLOAT *bptr3 = &b[i + 2 + jj * ldb];
					FLOAT *bptr4 = &b[i + 3 + jj * ldb];
					
					BLASLONG remaining = row_end - jj;
					
					// Prefetch next block
					if (i + 4 < col_end) {
						__builtin_prefetch(&a[(i + 4) * lda + jj], 0, 3);
					}
					
					for (j = 0; j < remaining; j += vl_max) {
						vl = (j + vl_max <= remaining) ? vl_max : VSETVL(remaining - j);
						
						va = VLEV_FLOAT(aptr + j, vl);
						va2 = VLEV_FLOAT(aptr + lda + j, vl);
						va3 = VLEV_FLOAT(aptr + 2 * lda + j, vl);
						va4 = VLEV_FLOAT(aptr + 3 * lda + j, vl);
						
						VSSEV_FLOAT(bptr1 + j * ldb, sizeof(FLOAT) * ldb, va, vl);
						VSSEV_FLOAT(bptr2 + j * ldb, sizeof(FLOAT) * ldb, va2, vl);
						VSSEV_FLOAT(bptr3 + j * ldb, sizeof(FLOAT) * ldb, va3, vl);
						VSSEV_FLOAT(bptr4 + j * ldb, sizeof(FLOAT) * ldb, va4, vl);
					}
				}
				
				// Handle remaining columns
				for (; i < col_end; i++) {
					aptr = &a[i * lda + jj];
					bptr = &b[i + jj * ldb];
					BLASLONG remaining = row_end - jj;
					
					for (j = 0; j < remaining; j += vl_max) {
						vl = (j + vl_max <= remaining) ? vl_max : VSETVL(remaining - j);
						va = VLEV_FLOAT(aptr + j, vl);
						VSSEV_FLOAT(bptr + j * ldb, sizeof(FLOAT) * ldb, va, vl);
					}
				}
			}
		}
		return(0);
	}

	// General case with alpha scaling and optimizations
	for (ii = 0; ii < cols; ii += BLOCK_SIZE_COLS) {
		BLASLONG col_end = (ii + BLOCK_SIZE_COLS < cols) ? ii + BLOCK_SIZE_COLS : cols;
		for (jj = 0; jj < rows; jj += BLOCK_SIZE_ROWS) {
			BLASLONG row_end = (jj + BLOCK_SIZE_ROWS < rows) ? jj + BLOCK_SIZE_ROWS : rows;
			
			// Process 2 columns at once for better pipeline utilization
			for (i = ii; i < col_end - 1; i += 2) {
				aptr = &a[i * lda + jj];
				FLOAT *bptr1 = &b[i + jj * ldb];
				FLOAT *bptr2 = &b[i + 1 + jj * ldb];
				
				BLASLONG remaining = row_end - jj;
				
				// Prefetch next block
				if (i + 2 < col_end) {
					__builtin_prefetch(&a[(i + 2) * lda + jj], 0, 3);
				}
				
				for (j = 0; j < remaining; j += vl_max) {
					vl = (j + vl_max <= remaining) ? vl_max : VSETVL(remaining - j);
					
					va = VLEV_FLOAT(aptr + j, vl);
					va2 = VLEV_FLOAT(aptr + lda + j, vl);
					
					va = VFMULVF_FLOAT(va, alpha, vl);
					va2 = VFMULVF_FLOAT(va2, alpha, vl);
					
					VSSEV_FLOAT(bptr1 + j * ldb, sizeof(FLOAT) * ldb, va, vl);
					VSSEV_FLOAT(bptr2 + j * ldb, sizeof(FLOAT) * ldb, va2, vl);
				}
			}
			
			// Handle remaining columns
			for (; i < col_end; i++) {
				aptr = &a[i * lda + jj];
				bptr = &b[i + jj * ldb];
				BLASLONG remaining = row_end - jj;
				
				for (j = 0; j < remaining; j += vl_max) {
					vl = (j + vl_max <= remaining) ? vl_max : VSETVL(remaining - j);
					va = VLEV_FLOAT(aptr + j, vl);
					va = VFMULVF_FLOAT(va, alpha, vl);
					VSSEV_FLOAT(bptr + j * ldb, sizeof(FLOAT) * ldb, va, vl);
				}
			}
		}
	}

	return(0);
}