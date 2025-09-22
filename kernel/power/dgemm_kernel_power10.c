/*********************************************************************************
Copyright (c) 2020, The OpenBLAS Project
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
**********************************************************************************/
#include "common.h"
#include <altivec.h>

typedef __vector unsigned char  vec_t;
typedef FLOAT v4sf_t __attribute__ ((vector_size (16)));
#if !__has_builtin(__builtin_vsx_assemble_pair)
#define __builtin_vsx_assemble_pair __builtin_mma_assemble_pair
#endif

#if !__has_builtin(__builtin_vsx_disassemble_pair)
#define __builtin_vsx_disassemble_pair __builtin_mma_disassemble_pair
#endif

#ifdef TRMMKERNEL
#define SAVE_ACC(ACC, J)  \
          __builtin_mma_disassemble_acc ((void *)result, ACC); \
          rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] = result[0] * alpha; \
          rowC = (v4sf_t *) &CO[1*ldc+J]; \
          rowC[0] = result[1] * alpha; \
          rowC = (v4sf_t *) &CO[2*ldc+J]; \
          rowC[0] = result[2] * alpha; \
          rowC = (v4sf_t *) &CO[3*ldc+J]; \
          rowC[0] = result[3] * alpha;
#define SAVE_ACC1(ACC, J)  \
          __builtin_mma_disassemble_acc ((void *)result, ACC); \
          rowC = (v4sf_t *) &CO[4* ldc+J]; \
          rowC[0] = result[0] * alpha; \
          rowC = (v4sf_t *) &CO[5*ldc+J]; \
          rowC[0] = result[1] * alpha; \
          rowC = (v4sf_t *) &CO[6*ldc+J]; \
          rowC[0] = result[2] * alpha; \
          rowC = (v4sf_t *) &CO[7*ldc+J]; \
          rowC[0] = result[3] * alpha;
#define  SAVE2x4_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] = result[0] * alpha; \
	  rowC = (v4sf_t *) &CO[1* ldc+J]; \
          rowC[0] = result[1] * alpha;
#else
#define SAVE_ACC(ACC, J)  \
          __builtin_mma_disassemble_acc ((void *)result, ACC); \
          rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] += result[0] * alpha; \
          rowC = (v4sf_t *) &CO[1*ldc+J]; \
          rowC[0] += result[1] * alpha; \
          rowC = (v4sf_t *) &CO[2*ldc+J]; \
          rowC[0] += result[2] * alpha; \
          rowC = (v4sf_t *) &CO[3*ldc+J]; \
          rowC[0] += result[3] * alpha;
#define SAVE_ACC1(ACC, J)  \
          __builtin_mma_disassemble_acc ((void *)result, ACC); \
          rowC = (v4sf_t *) &CO[4* ldc+J]; \
          rowC[0] += result[0] * alpha; \
          rowC = (v4sf_t *) &CO[5*ldc+J]; \
          rowC[0] += result[1] * alpha; \
          rowC = (v4sf_t *) &CO[6*ldc+J]; \
          rowC[0] += result[2] * alpha; \
          rowC = (v4sf_t *) &CO[7*ldc+J]; \
          rowC[0] += result[3] * alpha;
#define  SAVE2x4_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] += result[0] * alpha; \
	  rowC = (v4sf_t *) &CO[1* ldc+J]; \
          rowC[0] += result[1] * alpha;
#endif

#define PREFETCH1(x, y) asm volatile ("dcbt %0, %1" : : "r" (x), "b" (y) : "memory");

#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
#define REFRESH_TEMP_BK(x, y) \
            temp = k - off;
#elif defined(LEFT)
#define REFRESH_TEMP_BK(x, y) \
            temp = off + x;
#else
#define REFRESH_TEMP_BK(x, y) \
            temp = off + y;
#endif
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
#define REFRESH_POINTERS(x, y) \
          BO = B; \
          REFRESH_TEMP_BK(x, y)
#else
#define REFRESH_POINTERS(x, y) \
          AO += off * x; \
          BO = B + off * y; \
          REFRESH_TEMP_BK(x, y)
#endif

#ifdef LEFT
#define REFRESH_OFF(x) \
            off += x;
#else
#define REFRESH_OFF(x)
#endif

#ifdef LEFT
#define UPDATE_TEMP(x, y) \
            temp -= x;
#else
#define UPDATE_TEMP(x, y) \
            temp -= y;
#endif

#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
#define REFRESH_TMP_AFTER_SAVE(x, y) \
            temp = k - off; \
            UPDATE_TEMP(x, y) \
            AO += temp * x; \
            BO += temp * y;
#else
#define REFRESH_TMP_AFTER_SAVE(x, y)
#endif

#define REFRESH_AFTER_SAVE(x,y) \
        REFRESH_TMP_AFTER_SAVE(x, y) \
        REFRESH_OFF(x)
/*************************************************************************************
* GEMM Kernel
*************************************************************************************/
int
CNAME (BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha, FLOAT * A, FLOAT * B,
       FLOAT * C, BLASLONG ldc
#ifdef TRMMKERNEL
       , BLASLONG offset
#endif
  )
{
  BLASLONG i1;
#if defined(TRMMKERNEL)
  BLASLONG off;
#endif
#if defined(TRMMKERNEL) && !defined(LEFT)
  off = -offset;
#endif
  v4sf_t valpha = { alpha, alpha };
  for (i1 = 0; i1 < (n >> 3); i1++)
    {
      BLASLONG j, temp;
      FLOAT *CO;
      FLOAT *AO;
#if defined(TRMMKERNEL) && defined(LEFT)
      off = offset;
#endif
      CO = C;
      C += ldc << 3;
      AO = A;
      PREFETCH1 (A, 128);
      PREFETCH1 (A, 256);
      for (j = 0; j < (m >> 3); j++)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (8, 8);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3, acc4,acc5,acc6,acc7;
	  BLASLONG l = 1;
	  vec_t *rowA = (vec_t *) & AO[0];
	  __vector_pair rowB, rowB1;
	  rowB = *((__vector_pair *)((void *)&BO[0]));
	  rowB1 = *((__vector_pair *)((void *)&BO[4]));
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB1, rowA[0]);
	  __builtin_mma_xvf64ger (&acc2, rowB, rowA[1]);
	  __builtin_mma_xvf64ger (&acc3, rowB1, rowA[1]);
	  __builtin_mma_xvf64ger (&acc4, rowB, rowA[2]);
	  __builtin_mma_xvf64ger (&acc5, rowB1, rowA[2]);
	  __builtin_mma_xvf64ger (&acc6, rowB, rowA[3]);
	  __builtin_mma_xvf64ger (&acc7, rowB1, rowA[3]);
	  for (l = 1; l + 15 < temp; l += 16)
	{

		vec_t *rowA0 = (vec_t *)&AO[(l + 0) << 3];
		__vector_pair rowB0 = *((__vector_pair *)((void *)&BO[(l + 0) << 3]));
		__vector_pair rowB0_1 = *((__vector_pair *)((void *)&BO[((l + 0) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB0, rowA0[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB0_1, rowA0[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB0, rowA0[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB0_1, rowA0[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB0, rowA0[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB0_1, rowA0[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB0, rowA0[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB0_1, rowA0[3]);

		vec_t *rowA1 = (vec_t *)&AO[(l + 1) << 3];
		__vector_pair rowB1 = *((__vector_pair *)((void *)&BO[(l + 1) << 3]));
		__vector_pair rowB1_1 = *((__vector_pair *)((void *)&BO[((l + 1) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB1, rowA1[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB1_1, rowA1[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB1, rowA1[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB1_1, rowA1[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB1, rowA1[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB1_1, rowA1[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB1, rowA1[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB1_1, rowA1[3]);

		vec_t *rowA2 = (vec_t *)&AO[(l + 2) << 3];
		__vector_pair rowB2 = *((__vector_pair *)((void *)&BO[(l + 2) << 3]));
		__vector_pair rowB2_1 = *((__vector_pair *)((void *)&BO[((l + 2) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB2, rowA2[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB2_1, rowA2[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB2, rowA2[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB2_1, rowA2[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB2, rowA2[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB2_1, rowA2[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB2, rowA2[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB2_1, rowA2[3]);

		vec_t *rowA3 = (vec_t *)&AO[(l + 3) << 3];
		__vector_pair rowB3 = *((__vector_pair *)((void *)&BO[(l + 3) << 3]));
		__vector_pair rowB3_1 = *((__vector_pair *)((void *)&BO[((l + 3) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB3, rowA3[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB3_1, rowA3[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB3, rowA3[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB3_1, rowA3[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB3, rowA3[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB3_1, rowA3[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB3, rowA3[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB3_1, rowA3[3]);

		vec_t *rowA4 = (vec_t *)&AO[(l + 4) << 3];
		__vector_pair rowB4 = *((__vector_pair *)((void *)&BO[(l + 4) << 3]));
		__vector_pair rowB4_1 = *((__vector_pair *)((void *)&BO[((l + 4) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB4, rowA4[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB4_1, rowA4[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB4, rowA4[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB4_1, rowA4[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB4, rowA4[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB4_1, rowA4[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB4, rowA4[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB4_1, rowA4[3]);
		
		vec_t *rowA5 = (vec_t *)&AO[(l + 5) << 3];
		__vector_pair rowB5 = *((__vector_pair *)((void *)&BO[(l + 5) << 3]));
		__vector_pair rowB5_1 = *((__vector_pair *)((void *)&BO[((l + 5) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB5, rowA5[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB5_1, rowA5[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB5, rowA5[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB5_1, rowA5[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB5, rowA5[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB5_1, rowA5[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB5, rowA5[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB5_1, rowA5[3]);

		vec_t *rowA6 = (vec_t *)&AO[(l + 6) << 3];
		__vector_pair rowB6 = *((__vector_pair *)((void *)&BO[(l + 6) << 3]));
		__vector_pair rowB6_1 = *((__vector_pair *)((void *)&BO[((l + 6) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB6, rowA6[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB6_1, rowA6[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB6, rowA6[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB6_1, rowA6[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB6, rowA6[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB6_1, rowA6[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB6, rowA6[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB6_1, rowA6[3]);

		vec_t *rowA7 = (vec_t *)&AO[(l + 7) << 3];
		__vector_pair rowB7 = *((__vector_pair *)((void *)&BO[(l + 7) << 3]));
		__vector_pair rowB7_1 = *((__vector_pair *)((void *)&BO[((l + 7) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB7, rowA7[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB7_1, rowA7[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB7, rowA7[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB7_1, rowA7[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB7, rowA7[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB7_1, rowA7[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB7, rowA7[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB7_1, rowA7[3]);

		vec_t *rowA8 = (vec_t *)&AO[(l + 8) << 3];
		__vector_pair rowB8 = *((__vector_pair *)((void *)&BO[(l + 8) << 3]));
		__vector_pair rowB8_1 = *((__vector_pair *)((void *)&BO[((l + 8) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB8, rowA8[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB8_1, rowA8[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB8, rowA8[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB8_1, rowA8[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB8, rowA8[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB8_1, rowA8[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB8, rowA8[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB8_1, rowA8[3]);

		vec_t *rowA9 = (vec_t *)&AO[(l + 9) << 3];
		__vector_pair rowB9 = *((__vector_pair *)((void *)&BO[(l + 9) << 3]));
		__vector_pair rowB9_1 = *((__vector_pair *)((void *)&BO[((l + 9) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB9, rowA9[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB9_1, rowA9[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB9, rowA9[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB9_1, rowA9[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB9, rowA9[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB9_1, rowA9[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB9, rowA9[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB9_1, rowA9[3]);

		vec_t *rowA10 = (vec_t *)&AO[(l + 10) << 3];
		__vector_pair rowB10 = *((__vector_pair *)((void *)&BO[(l + 10) << 3]));
		__vector_pair rowB10_1 = *((__vector_pair *)((void *)&BO[((l + 10) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB10, rowA10[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB10_1, rowA10[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB10, rowA10[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB10_1, rowA10[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB10, rowA10[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB10_1, rowA10[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB10, rowA10[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB10_1, rowA10[3]);

		vec_t *rowA11 = (vec_t *)&AO[(l + 11) << 3];
		__vector_pair rowB11 = *((__vector_pair *)((void *)&BO[(l + 11) << 3]));
		__vector_pair rowB11_1 = *((__vector_pair *)((void *)&BO[((l + 11) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB11, rowA11[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB11_1, rowA11[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB11, rowA11[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB11_1, rowA11[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB11, rowA11[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB11_1, rowA11[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB11, rowA11[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB11_1, rowA11[3]);

		vec_t *rowA12 = (vec_t *)&AO[(l + 12) << 3];
		__vector_pair rowB12 = *((__vector_pair *)((void *)&BO[(l + 12) << 3]));
		__vector_pair rowB12_1 = *((__vector_pair *)((void *)&BO[((l + 12) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB12, rowA12[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB12_1, rowA12[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB12, rowA12[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB12_1, rowA12[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB12, rowA12[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB12_1, rowA12[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB12, rowA12[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB12_1, rowA12[3]);

		vec_t *rowA13 = (vec_t *)&AO[(l + 13) << 3];
		__vector_pair rowB13 = *((__vector_pair *)((void *)&BO[(l + 13) << 3]));
		__vector_pair rowB13_1 = *((__vector_pair *)((void *)&BO[((l + 13) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB13, rowA13[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB13_1, rowA13[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB13, rowA13[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB13_1, rowA13[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB13, rowA13[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB13_1, rowA13[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB13, rowA13[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB13_1, rowA13[3]);

		vec_t *rowA14 = (vec_t *)&AO[(l + 14) << 3];
		__vector_pair rowB14 = *((__vector_pair *)((void *)&BO[(l + 14) << 3]));
		__vector_pair rowB14_1 = *((__vector_pair *)((void *)&BO[((l + 14) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB14, rowA14[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB14_1, rowA14[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB14, rowA14[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB14_1, rowA14[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB14, rowA14[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB14_1, rowA14[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB14, rowA14[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB14_1, rowA14[3]);

		vec_t *rowA15 = (vec_t *)&AO[(l + 15) << 3];
		__vector_pair rowB15 = *((__vector_pair *)((void *)&BO[(l + 15) << 3]));
		__vector_pair rowB15_1 = *((__vector_pair *)((void *)&BO[((l + 15) << 3) + 4]));
		__builtin_mma_xvf64gerpp(&acc0, rowB15, rowA15[0]);
		__builtin_mma_xvf64gerpp(&acc1, rowB15_1, rowA15[0]);
		__builtin_mma_xvf64gerpp(&acc2, rowB15, rowA15[1]);
		__builtin_mma_xvf64gerpp(&acc3, rowB15_1, rowA15[1]);
		__builtin_mma_xvf64gerpp(&acc4, rowB15, rowA15[2]);
		__builtin_mma_xvf64gerpp(&acc5, rowB15_1, rowA15[2]);
		__builtin_mma_xvf64gerpp(&acc6, rowB15, rowA15[3]);
		__builtin_mma_xvf64gerpp(&acc7, rowB15_1, rowA15[3]);

	}
	  for (; l < temp; l++)
	    {
	      rowA = (vec_t *) & AO[l << 3];
	      rowB = *((__vector_pair *)((void *)&BO[l << 3]));
	      rowB1 = *((__vector_pair *)((void *)&BO[(l << 3) + 4]));
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB1, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc2, rowB, rowA[1]);
	      __builtin_mma_xvf64gerpp (&acc3, rowB1, rowA[1]);
	      __builtin_mma_xvf64gerpp (&acc4, rowB, rowA[2]);
	      __builtin_mma_xvf64gerpp (&acc5, rowB1, rowA[2]);
	      __builtin_mma_xvf64gerpp (&acc6, rowB, rowA[3]);
	      __builtin_mma_xvf64gerpp (&acc7, rowB1, rowA[3]);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC1 (&acc1, 0);
	  SAVE_ACC (&acc2, 2);
	  SAVE_ACC1 (&acc3, 2);
	  SAVE_ACC (&acc4, 4);
	  SAVE_ACC1 (&acc5, 4);
	  SAVE_ACC (&acc6, 6);
	  SAVE_ACC1 (&acc7, 6);
	  CO += 8;
	  AO += temp << 3;
	  BO += temp << 3;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (8, 8)
#endif
	}
      if (m & 4)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (4, 8);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  BLASLONG l = 0;
	  vec_t *rowA = (vec_t *) & AO[0];
	  __vector_pair rowB, rowB1;
	  rowB = *((__vector_pair *)((void *)&BO[0]));
	  rowB1 = *((__vector_pair *)((void *)&BO[4]));
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB1, rowA[0]);
	  __builtin_mma_xvf64ger (&acc2, rowB, rowA[1]);
	  __builtin_mma_xvf64ger (&acc3, rowB1, rowA[1]);
	  for (l = 1; l < temp; l++)
	    {
	      rowA = (vec_t *) & AO[l << 2];
	      rowB = *((__vector_pair *)((void *)&BO[l << 3]));
	      rowB1 = *((__vector_pair *)((void *)&BO[(l << 3) + 4]));
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB1, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc2, rowB, rowA[1]);
	      __builtin_mma_xvf64gerpp (&acc3, rowB1, rowA[1]);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC1 (&acc1, 0);
	  SAVE_ACC (&acc2, 2);
	  SAVE_ACC1 (&acc3, 2);
	  CO += 4;
	  AO += temp << 2;
	  BO += temp << 3;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (4, 8)
#endif
	}
      if (m & 2)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (2, 8);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  BLASLONG l = 0;
	  vec_t *rowA = (vec_t *) & AO[0];
	  __vector_pair rowB, rowB1;
	  rowB = *((__vector_pair *)((void *)&BO[0]));
	  rowB1 = *((__vector_pair *)((void *)&BO[4]));
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB1, rowA[0]);
	  for (l = 1; l < temp; l++)
	    {
	      rowA = (vec_t *) & AO[l << 1];
	      rowB = *((__vector_pair *)((void *)&BO[l << 3]));
	      rowB1 = *((__vector_pair *)((void *)&BO[(l << 3) + 4]));
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB1, rowA[0]);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC1 (&acc1, 0);
	  CO += 2;
	  AO += temp << 1;
	  BO += temp << 3;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (2, 8)
#endif
	}
      if (m & 1)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (1, 8);
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0 };
	  v4sf_t t1 = { 0, 0 };
	  v4sf_t t2 = { 0, 0 };
	  v4sf_t t3 = { 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowA = { AO[l], AO[l] };
	      v4sf_t rowB = { BO[l << 3], BO[(l << 3) + 1] };
	      v4sf_t rowB1 = { BO[(l << 3) + 2], BO[(l << 3) + 3] };
	      v4sf_t rowB2 = { BO[(l << 3) + 4], BO[(l << 3) + 5] };
	      v4sf_t rowB3 = { BO[(l << 3) + 6], BO[(l << 3) + 7] };
	      t += rowA * rowB;
	      t1 += rowA * rowB1;
	      t2 += rowA * rowB2;
	      t3 += rowA * rowB3;
	    }
	  t = t * valpha;
	  t1 = t1 * valpha;
	  t2 = t2 * valpha;
	  t3 = t3 * valpha;
#if defined(TRMMKERNEL)
	  CO[0 * ldc] = t[0];
	  CO[1 * ldc] = t[1];
	  CO[2 * ldc] = t1[0];
	  CO[3 * ldc] = t1[1];
	  CO[4 * ldc] = t2[0];
	  CO[5 * ldc] = t2[1];
	  CO[6 * ldc] = t3[0];
	  CO[7 * ldc] = t3[1];
#else
	  CO[0 * ldc] += t[0];
	  CO[1 * ldc] += t[1];
	  CO[2 * ldc] += t1[0];
	  CO[3 * ldc] += t1[1];
	  CO[4 * ldc] += t2[0];
	  CO[5 * ldc] += t2[1];
	  CO[6 * ldc] += t3[0];
	  CO[7 * ldc] += t3[1];
#endif
	  CO += 1;
	  AO += temp;
	  BO += temp << 3;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (1, 8)
#endif
	}
#if defined(TRMMKERNEL) && !defined(LEFT)
      off += 8;                 // number of values in A
#endif
      B += k << 3;
    }
  if (n & 4)
    {
      BLASLONG j, temp;
      FLOAT *CO;
      FLOAT *AO;
#if defined(TRMMKERNEL) && defined(LEFT)
      off = offset;
#endif
      CO = C;
      C += ldc << 2;
      AO = A;
      PREFETCH1 (A, 128);
      PREFETCH1 (A, 256);
      for (j = 0; j < (m >> 3); j++)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (8, 4);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  BLASLONG l = 0;
	  vec_t *rowA = (vec_t *) & AO[0];
	  __vector_pair rowB;
	  rowB = *((__vector_pair *)((void *)&BO[0]));
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB, rowA[1]);
	  __builtin_mma_xvf64ger (&acc2, rowB, rowA[2]);
	  __builtin_mma_xvf64ger (&acc3, rowB, rowA[3]);
	  for (l = 1; l < temp; l++)
	    {
	      rowA = (vec_t *) & AO[l << 3];
	      rowB = *((__vector_pair *)((void *)&BO[l << 2]));
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB, rowA[1]);
	      __builtin_mma_xvf64gerpp (&acc2, rowB, rowA[2]);
	      __builtin_mma_xvf64gerpp (&acc3, rowB, rowA[3]);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc2, 4);
	  SAVE_ACC (&acc1, 2);
	  SAVE_ACC (&acc3, 6);
	  CO += 8;
	  AO += temp << 3;
	  BO += temp << 2;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (8, 4)
#endif
	}
      if (m & 4)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (4, 4);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  BLASLONG l = 0;
	  vec_t *rowA = (vec_t *) & AO[0];
	  __vector_pair rowB;
	  rowB = *((__vector_pair *)((void *)&BO[0]));
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB, rowA[1]);
	  for (l = 1; l < temp; l++)
	    {
	      rowA = (vec_t *) & AO[l << 2];
	      rowB = *((__vector_pair *)((void *)&BO[l << 2]));
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB, rowA[1]);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc1, 2);
	  CO += 4;
	  AO += temp << 2;
	  BO += temp << 2;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (4, 4)
#endif
	}
      if (m & 2)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (2, 4);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0;
	  BLASLONG l = 0;
	  vec_t *rowA = (vec_t *) & AO[0];
	  __vector_pair rowB;
	  rowB = *((__vector_pair *)((void *)&BO[0]));
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  for (l = 1; l < temp; l++)
	    {
	      rowA = (vec_t *) & AO[l << 1];
	      rowB = *((__vector_pair *)((void *)&BO[l << 2]));
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	    }
	  SAVE_ACC (&acc0, 0);
	  CO += 2;
	  AO += temp << 1;
	  BO += temp << 2;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (2, 4)
#endif
	}
      if (m & 1)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (1, 4);
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0 };
	  v4sf_t t1 = { 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowA = { AO[l], AO[l] };
	      v4sf_t rowB = { BO[l << 2], BO[(l << 2) + 1] };
	      v4sf_t rowB1 = { BO[(l << 2) + 2], BO[(l << 2) + 3] };
	      t += rowA * rowB;
	      t1 += rowA * rowB1;
	    }
	  t = t * valpha;
	  t1 = t1 * valpha;
#if defined(TRMMKERNEL)
	  CO[0 * ldc] = t[0];
	  CO[1 * ldc] = t[1];
	  CO[2 * ldc] = t1[0];
	  CO[3 * ldc] = t1[1];
#else
	  CO[0 * ldc] += t[0];
	  CO[1 * ldc] += t[1];
	  CO[2 * ldc] += t1[0];
	  CO[3 * ldc] += t1[1];
#endif
	  CO += 1;
	  AO += temp;
	  BO += temp << 2;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (1, 4)
#endif
	}
#if defined(TRMMKERNEL) && !defined(LEFT)
      off += 4;                 // number of values in A
#endif
      B += k << 2;
    }
  if (n & 2)
    {
      BLASLONG j, temp;
#if defined(TRMMKERNEL) && defined(LEFT)
      off = offset;
#endif
      FLOAT *CO;
      FLOAT *AO;
      CO = C;
      C += ldc << 1;
      AO = A;
      for (j = 0; j < (m >> 3); j++)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (8, 2);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  BLASLONG l = 0;
	  __vector_pair rowB;
	  vec_t *rb = (vec_t *) & BO[0];
	  __builtin_vsx_assemble_pair (&rowB, rb[0], rb[0]);
	  vec_t *rowA = (vec_t *) & AO[0];
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB, rowA[1]);
	  __builtin_mma_xvf64ger (&acc2, rowB, rowA[2]);
	  __builtin_mma_xvf64ger (&acc3, rowB, rowA[3]);
	  for (l = 1; l < temp; l++)
	    {
	      rb = (vec_t *) & BO[l << 1];
	      __builtin_vsx_assemble_pair (&rowB, rb[0], rb[0]);
	      rowA = (vec_t *) & AO[l << 3];
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB, rowA[1]);
	      __builtin_mma_xvf64gerpp (&acc2, rowB, rowA[2]);
	      __builtin_mma_xvf64gerpp (&acc3, rowB, rowA[3]);
	    }
	  SAVE2x4_ACC (&acc0, 0);
	  SAVE2x4_ACC (&acc1, 2);
	  SAVE2x4_ACC (&acc2, 4);
	  SAVE2x4_ACC (&acc3, 6);
	  CO += 8;
	  AO += temp << 3;
	  BO += temp << 1;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (8, 2)
#endif
	}
      if (m & 4)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (4, 2);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  BLASLONG l = 0;
	  __vector_pair rowB;
	  vec_t *rb = (vec_t *) & BO[0];
	  __builtin_vsx_assemble_pair (&rowB, rb[0], rb[0]);
	  vec_t *rowA = (vec_t *) & AO[0];
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  __builtin_mma_xvf64ger (&acc1, rowB, rowA[1]);
	  for (l = 1; l < temp; l++)
	    {
	      rb = (vec_t *) & BO[l << 1];
	      __builtin_vsx_assemble_pair (&rowB, rb[0], rb[0]);
	      rowA = (vec_t *) & AO[l << 2];
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	      __builtin_mma_xvf64gerpp (&acc1, rowB, rowA[1]);
	    }
	  SAVE2x4_ACC (&acc0, 0);
	  SAVE2x4_ACC (&acc1, 2);
	  CO += 4;
	  AO += temp << 2;
	  BO += temp << 1;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (4, 2)
#endif
	}
      if (m & 2)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (2, 2);
#else
          BO = B;
          temp = k;
#endif
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0;
	  BLASLONG l = 0;
	  __vector_pair rowB;
	  vec_t *rb = (vec_t *) & BO[0];
	  __builtin_vsx_assemble_pair (&rowB, rb[0], rb[0]);
	  vec_t *rowA = (vec_t *) & AO[0];
	  __builtin_mma_xvf64ger (&acc0, rowB, rowA[0]);
	  for (l = 1; l < temp; l++)
	    {
	      rb = (vec_t *) & BO[l << 1];
	      __builtin_vsx_assemble_pair (&rowB, rb[0], rb[0]);
	      rowA = (vec_t *) & AO[l << 1];
	      __builtin_mma_xvf64gerpp (&acc0, rowB, rowA[0]);
	    }
	  SAVE2x4_ACC (&acc0, 0);
	  CO += 2;
	  AO += temp << 1;
	  BO += temp << 1;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (2, 2)
#endif
	}
      if (m & 1)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (1, 2);
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowA = { AO[l], AO[l] };
	      v4sf_t rowB = { BO[l << 1], BO[(l << 1) + 1] };
	      t += rowA * rowB;
	    }
	  t = t * valpha;
#if defined(TRMMKERNEL)
	  CO[0 * ldc] = t[0];
	  CO[1 * ldc] = t[1];
#else
	  CO[0 * ldc] += t[0];
	  CO[1 * ldc] += t[1];
#endif
	  CO += 1;
	  AO += temp;
	  BO += temp << 1;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (1, 2)
#endif
	}
#if defined(TRMMKERNEL) && !defined(LEFT)
      off += 2;                 // number of values in A
#endif
      B += k << 1;
    }
  if (n & 1)
    {
      BLASLONG i, temp;
#if defined(TRMMKERNEL) && defined(LEFT)
      off = offset;
#endif
      FLOAT *CO;
      FLOAT *AO;
      CO = C;
      C += ldc;
      AO = A;
      for (i = 0; i < (m >> 3); i++)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (8, 1)
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0 };
	  v4sf_t t1 = { 0, 0 };
	  v4sf_t t2 = { 0, 0 };
	  v4sf_t t3 = { 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowB = { BO[l], BO[l] };
	      v4sf_t rowA = { AO[l << 3], AO[(l << 3) + 1] };
	      v4sf_t rowA1 = { AO[(l << 3) + 2], AO[(l << 3) + 3] };
	      v4sf_t rowA2 = { AO[(l << 3) + 4], AO[(l << 3) + 5] };
	      v4sf_t rowA3 = { AO[(l << 3) + 6], AO[(l << 3) + 7] };
	      t += rowA * rowB;
	      t1 += rowA1 * rowB;
	      t2 += rowA2 * rowB;
	      t3 += rowA3 * rowB;
	    }
	  t = t * valpha;
	  t1 = t1 * valpha;
	  t2 = t2 * valpha;
	  t3 = t3 * valpha;
#if defined(TRMMKERNEL)
	  CO[0] = t[0];
	  CO[1] = t[1];
	  CO[2] = t1[0];
	  CO[3] = t1[1];
	  CO[4] = t2[0];
	  CO[5] = t2[1];
	  CO[6] = t3[0];
	  CO[7] = t3[1];
#else
	  CO[0] += t[0];
	  CO[1] += t[1];
	  CO[2] += t1[0];
	  CO[3] += t1[1];
	  CO[4] += t2[0];
	  CO[5] += t2[1];
	  CO[6] += t3[0];
	  CO[7] += t3[1];
#endif
	  AO += temp << 3;
	  BO += temp;
	  CO += 8;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (8, 1)
#endif
	}
      if (m & 4)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (4, 1)
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0 };
	  v4sf_t t1 = { 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowB = { BO[l], BO[l] };
	      v4sf_t rowA = { AO[l << 2], AO[(l << 2) + 1] };
	      v4sf_t rowA1 = { AO[(l << 2) + 2], AO[(l << 2) + 3] };
	      t += rowA * rowB;
	      t1 += rowA1 * rowB;
	    }
	  t = t * valpha;
	  t1 = t1 * valpha;
#if defined(TRMMKERNEL)
	  CO[0] = t[0];
	  CO[1] = t[1];
	  CO[2] = t1[0];
	  CO[3] = t1[1];
#else
	  CO[0] += t[0];
	  CO[1] += t[1];
	  CO[2] += t1[0];
	  CO[3] += t1[1];
#endif
	  AO += temp << 2;
	  BO += temp;
	  CO += 4;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (4, 1)
#endif
	}
      if (m & 2)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (2, 1)
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowB = { BO[l], BO[l] };
	      v4sf_t rowA = { AO[l << 1], AO[(l << 1) + 1] };
	      t += rowA * rowB;
	    }
	  t = t * valpha;
#if defined(TRMMKERNEL)
	  CO[0] = t[0];
	  CO[1] = t[1];
#else
	  CO[0] += t[0];
	  CO[1] += t[1];
#endif
	  AO += temp << 1;
	  BO += temp;
	  CO += 2;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (2, 1)
#endif
	}
      if (m & 1)
	{
	  FLOAT *BO;
#if defined(TRMMKERNEL)
          REFRESH_POINTERS (1, 1)
#else
          BO = B;
          temp = k;
#endif
	  BLASLONG l = 0;
	  FLOAT t = 0;
	  for (l = 0; l < temp; l++)
	    {
	      t += AO[l] * BO[l];
	    }
	  AO += temp;
	  BO += temp;
#if defined(TRMMKERNEL)
	  CO[0] = t * alpha;
#else
	  CO[0] += t * alpha;
#endif
	  CO += 1;
#if defined(TRMMKERNEL)
          REFRESH_AFTER_SAVE (1, 1)
#endif
	}
#if defined(TRMMKERNEL) && !defined(LEFT)
      off += 1;                 // number of values in A
#endif
      B += k;
    }
  return 0;
}
