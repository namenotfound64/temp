/***************************************************************************
Copyright (c) 2025 The OpenBLAS Project
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
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/
#include "../common.h"
#include <stdint.h>
#include <stdio.h>

#include <arm_neon.h>

#define SGEMM BLASFUNC(sgemm)
#define BGEMM BLASFUNC(bgemm)
#define BGEMM_LARGEST 256

typedef union
{
  unsigned short v;
#if defined(_AIX)
  struct __attribute__((packed))
#else
  struct
#endif
  {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    unsigned short s:1;
    unsigned short e:8;
    unsigned short m:7;
#else
    unsigned short m:7;
    unsigned short e:8;
    unsigned short s:1;
#endif
  } bits;
} bfloat16_bits;

typedef union
{
  float v;
#if defined(_AIX)
  struct __attribute__((packed))
#else
  struct
#endif
  {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint32_t s:1;
    uint32_t e:8;
    uint32_t m:23;
#else
    uint32_t m:23;
    uint32_t e:8;
    uint32_t s:1;
#endif
  } bits;
} float32_bits;

float
float16to32 (bfloat16_bits f16)
{
  float32_bits f32;
  f32.bits.s = f16.bits.s;
  f32.bits.e = f16.bits.e;
  f32.bits.m = (uint32_t) f16.bits.m << 16;
  return f32.v;
}

bfloat16
float32to16 (float32_bits f32)
{
  bfloat16_bits f16;
  f16.bits.s = f32.bits.s;
  f16.bits.e = f32.bits.e;
  f16.bits.m = (f32.bits.m >> 16) & 0x7f;
  return f16.v;
}

static float truncate_float(float value) {
  bfloat16_bits f16 = (bfloat16_bits)float32to16((float32_bits)value);
  return float16to32(f16);
}

void *malloc_safe(size_t size) {
  if (size == 0)
    return malloc(1);
  else
    return malloc(size);
}

int
main (int argc, char *argv[])
{
  blasint m, n, k;
  int i, j, l;
  blasint x, y;
  int ret = 0;
  int loop = BGEMM_LARGEST;
  char transA = 'N', transB = 'N';
  float alpha = 1.0, beta = 0.0;
  bfloat16 alpha_bf16 = float32to16((float32_bits)alpha);
  bfloat16 beta_bf16 = float32to16((float32_bits)beta);

  for (x = 0; x <= loop; x++)
  {
    if ((x > 100) && (x != BGEMM_LARGEST)) continue;
    m = k = n = x;
    float *A = (float *)malloc_safe(m * k * sizeof(FLOAT));
    float *B = (float *)malloc_safe(k * n * sizeof(FLOAT));
    float *C = (float *)malloc_safe(m * n * sizeof(FLOAT));
    bfloat16_bits *AA = (bfloat16_bits *)malloc_safe(m * k * sizeof(bfloat16_bits));
    bfloat16_bits *BB = (bfloat16_bits *)malloc_safe(k * n * sizeof(bfloat16_bits));
    bfloat16_bits *CC = (bfloat16_bits *)malloc_safe(k * n * sizeof(bfloat16_bits));
    FLOAT *DD = (FLOAT *)malloc_safe(m * n * sizeof(FLOAT));
    if ((A == NULL) || (B == NULL) || (C == NULL) || (AA == NULL) || (BB == NULL) ||
        (DD == NULL) || (CC == NULL))
      return 1;
    bfloat16 atmp,btmp;
    blasint one=1;

    for (j = 0; j < m; j++)
    {
      for (i = 0; i < k; i++)
      {
        A[j * k + i] = ((FLOAT) rand () / (FLOAT) RAND_MAX) + 0.5;
        sbstobf16_(&one, &A[j*k+i], &one, &atmp, &one);
        AA[j * k + i].v = atmp;
      }
    }
    for (j = 0; j < n; j++)
    {
      for (i = 0; i < k; i++)
      {
        B[j * k + i] = ((FLOAT) rand () / (FLOAT) RAND_MAX) + 0.5;
        sbstobf16_(&one, &B[j*k+i], &one, &btmp, &one);
        BB[j * k + i].v = btmp;
      }
    }
    for (y = 0; y < 4; y++)
    {
      if ((y == 0) || (y == 2)) {
        transA = 'N';
      } else {
        transA = 'T';
      }
      if ((y == 0) || (y == 1)) {
        transB = 'N';
      } else {
        transB = 'T';
      }

      memset(CC, 0, m * n * sizeof(bfloat16_bits));
      memset(DD, 0, m * n * sizeof(FLOAT));
      memset(C, 0, m * n * sizeof(FLOAT));

      SGEMM (&transA, &transB, &m, &n, &k, &alpha, A,
        &m, B, &k, &beta, C, &m);
      BGEMM (&transA, &transB, &m, &n, &k, &alpha_bf16, (bfloat16*) AA,
        &m, (bfloat16*)BB, &k, &beta_bf16, (bfloat16*)CC, &m);

      for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
        {
          for (l = 0; l < k; l++)
            if (transA == 'N' && transB == 'N')
            {
              DD[i * m + j] +=
                float16to32 (AA[l * m + j]) * float16to32 (BB[l + k * i]);
            } else if (transA == 'T' && transB == 'N')
            {
              DD[i * m + j] +=
                float16to32 (AA[k * j + l]) * float16to32 (BB[l + k * i]);
            } else if (transA == 'N' && transB == 'T')
            {
              DD[i * m + j] +=
                float16to32 (AA[l * m + j]) * float16to32 (BB[i + l * n]);
            } else if (transA == 'T' && transB == 'T')
            {
              DD[i * m + j] +=
                float16to32 (AA[k * j + l]) * float16to32 (BB[i + l * n]);
            }
          if (fabs(float16to32(CC[i * m + j]) - truncate_float(C[i * m + j])) > 2.0) {
            ret++;
          }
          if (fabs(float16to32(CC[i * m + j]) - truncate_float(DD[i * m + j])) > 1.0) {
            ret++;
          }
            
        }
    }
    free(A);
    free(B);
    free(C);
    free(AA);
    free(BB);
    free(CC);
    free(DD);
  }

  if (ret != 0) {
    fprintf (stderr, "FATAL ERROR BGEMM - Return code: %d\n", ret);
    return ret;
  }
}
