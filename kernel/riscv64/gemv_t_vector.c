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
#if !defined(DOUBLE)
#define VSETVL(n) RISCV_RVV(vsetvl_e32m8)(n)
#define VSETVL_MAX_M1 RISCV_RVV(vsetvlmax_e32m1)
#define FLOAT_V_T vfloat32m8_t
#define FLOAT_V_T_M1 vfloat32m1_t
#define VLEV_FLOAT RISCV_RVV(vle32_v_f32m8)
#define VLSEV_FLOAT RISCV_RVV(vlse32_v_f32m8)
#ifdef RISCV_0p10_INTRINSICS
#define VFREDSUM_FLOAT(va, vb, gvl) vfredusum_vs_f32m8_f32m1(v_res, va, vb, gvl)
#else
#define VFREDSUM_FLOAT RISCV_RVV(vfredusum_vs_f32m8_f32m1)
#endif
#define VFMULVV_FLOAT RISCV_RVV(vfmul_vv_f32m8)
#define VFMVVF_FLOAT RISCV_RVV(vfmv_v_f_f32m8)
#define VFMVVF_FLOAT_M1 RISCV_RVV(vfmv_v_f_f32m1)
#define xint_t int
#else
#define VSETVL(n) RISCV_RVV(vsetvl_e64m8)(n)
#define VSETVL_MAX_M1 RISCV_RVV(vsetvlmax_e64m1)
#define FLOAT_V_T vfloat64m8_t
#define FLOAT_V_T_M1 vfloat64m1_t
#define VLEV_FLOAT RISCV_RVV(vle64_v_f64m8)
#define VLSEV_FLOAT RISCV_RVV(vlse64_v_f64m8)
#ifdef RISCV_0p10_INTRINSICS
#define VFREDSUM_FLOAT(va, vb, gvl) vfredusum_vs_f64m8_f64m1(v_res, va, vb, gvl)
#else
#define VFREDSUM_FLOAT RISCV_RVV(vfredusum_vs_f64m8_f64m1)
#endif
#define VFMULVV_FLOAT RISCV_RVV(vfmul_vv_f64m8)
#define VFMVVF_FLOAT RISCV_RVV(vfmv_v_f_f64m8)
#define VFMVVF_FLOAT_M1 RISCV_RVV(vfmv_v_f_f64m1)
#define xint_t long long
#endif

int CNAME(BLASLONG m, BLASLONG n, BLASLONG dummy1, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y, FLOAT *buffer)
{
    BLASLONG i = 0, j = 0, k = 0;
    BLASLONG ix = 0, iy = 0;
    FLOAT *a_ptr = a;
    FLOAT temp;

    FLOAT_V_T va, vr, vx;
    unsigned int gvl = 0;
    FLOAT_V_T_M1 v_res;
    size_t vlmax = VSETVL_MAX_M1();

#ifndef RISCV_0p10_INTRINSICS
    FLOAT_V_T va0, va1, va2, va3, vr0, vr1, vr2, vr3;
    FLOAT_V_T_M1 vec0, vec1, vec2, vec3;
    FLOAT *a_ptrs[4], *y_ptrs[4];
#endif

    if(inc_x == 1){
#ifndef RISCV_0p10_INTRINSICS
        BLASLONG anr = n - n % 4;
        for (; i < anr; i += 4) {
            gvl = VSETVL(m);
            j = 0;
            for (int l = 0; l < 4; l++) {
                a_ptrs[l] = a + (i + l) * lda;
                y_ptrs[l] = y + (i + l) * inc_y;
            }
            vec0 = VFMVVF_FLOAT_M1(0.0, vlmax);
            vec1 = VFMVVF_FLOAT_M1(0.0, vlmax);
            vec2 = VFMVVF_FLOAT_M1(0.0, vlmax);
            vec3 = VFMVVF_FLOAT_M1(0.0, vlmax);
            vr0 = VFMVVF_FLOAT(0.0, gvl);
            vr1 = VFMVVF_FLOAT(0.0, gvl);
            vr2 = VFMVVF_FLOAT(0.0, gvl);
            vr3 = VFMVVF_FLOAT(0.0, gvl);
            for (k = 0; k < m / gvl; k++) {
                va0 = VLEV_FLOAT(a_ptrs[0] + j, gvl);
                va1 = VLEV_FLOAT(a_ptrs[1] + j, gvl);
                va2 = VLEV_FLOAT(a_ptrs[2] + j, gvl);
                va3 = VLEV_FLOAT(a_ptrs[3] + j, gvl);

                vx = VLEV_FLOAT(x + j, gvl);
                vr0 = VFMULVV_FLOAT(va0, vx, gvl);
                vr1 = VFMULVV_FLOAT(va1, vx, gvl);
                vr2 = VFMULVV_FLOAT(va2, vx, gvl);
                vr3 = VFMULVV_FLOAT(va3, vx, gvl);
                // Floating-point addition does not satisfy the associative law, that is, (a + b) + c ≠ a + (b + c),
                // so piecewise multiplication and reduction must be performed inside the loop body.
                vec0 = VFREDSUM_FLOAT(vr0, vec0, gvl);
                vec1 = VFREDSUM_FLOAT(vr1, vec1, gvl);
                vec2 = VFREDSUM_FLOAT(vr2, vec2, gvl);
                vec3 = VFREDSUM_FLOAT(vr3, vec3, gvl);
                j += gvl;
            }
            if (j < m) {
                gvl = VSETVL(m - j);
                va0 = VLEV_FLOAT(a_ptrs[0] + j, gvl);
                va1 = VLEV_FLOAT(a_ptrs[1] + j, gvl);
                va2 = VLEV_FLOAT(a_ptrs[2] + j, gvl);
                va3 = VLEV_FLOAT(a_ptrs[3] + j, gvl);

                vx = VLEV_FLOAT(x + j, gvl);
                vr0 = VFMULVV_FLOAT(va0, vx, gvl);
                vr1 = VFMULVV_FLOAT(va1, vx, gvl);
                vr2 = VFMULVV_FLOAT(va2, vx, gvl);
                vr3 = VFMULVV_FLOAT(va3, vx, gvl);
                vec0 = VFREDSUM_FLOAT(vr0, vec0, gvl);
                vec1 = VFREDSUM_FLOAT(vr1, vec1, gvl);
                vec2 = VFREDSUM_FLOAT(vr2, vec2, gvl);
                vec3 = VFREDSUM_FLOAT(vr3, vec3, gvl);
            }
            *y_ptrs[0] += alpha * (FLOAT)(EXTRACT_FLOAT(vec0));
            *y_ptrs[1] += alpha * (FLOAT)(EXTRACT_FLOAT(vec1));
            *y_ptrs[2] += alpha * (FLOAT)(EXTRACT_FLOAT(vec2));
            *y_ptrs[3] += alpha * (FLOAT)(EXTRACT_FLOAT(vec3));
        }
        // deal with the tail
        for (; i < n; i++) {
            v_res = VFMVVF_FLOAT_M1(0, vlmax);
            gvl = VSETVL(m);
            j = 0;
            a_ptrs[0] = a + i * lda;
            y_ptrs[0] = y + i * inc_y;
            vr0 = VFMVVF_FLOAT(0, gvl);
            for (k = 0; k < m / gvl; k++) {
                va0 = VLEV_FLOAT(a_ptrs[0] + j, gvl);
                vx = VLEV_FLOAT(x + j, gvl);
                vr0 = VFMULVV_FLOAT(va0, vx, gvl);
                v_res = VFREDSUM_FLOAT(vr0, v_res, gvl);
                j += gvl;
            }
            if (j < m) {
                gvl = VSETVL(m - j);
                va0 = VLEV_FLOAT(a_ptrs[0] + j, gvl);
                vx = VLEV_FLOAT(x + j, gvl);
                vr0 = VFMULVV_FLOAT(va0, vx, gvl);
                v_res = VFREDSUM_FLOAT(vr0, v_res, gvl);
            }
            *y_ptrs[0] += alpha * (FLOAT)(EXTRACT_FLOAT(v_res));
        }
#else
    for(i = 0; i < n; i++){
        v_res = VFMVVF_FLOAT_M1(0, 1);
        gvl = VSETVL(m);
        j = 0;
        vr = VFMVVF_FLOAT(0, gvl);
        for(k = 0; k < m/gvl; k++){
            va = VLEV_FLOAT(&a_ptr[j], gvl);
            vx = VLEV_FLOAT(&x[j], gvl);
            vr = VFMULVV_FLOAT(va, vx, gvl);                // could vfmacc here and reduce outside loop
            v_res = VFREDSUM_FLOAT(vr, v_res, gvl);         // but that reordering diverges far enough from scalar path to make tests fail
            j += gvl;
        }
        if(j < m){
            gvl = VSETVL(m-j);
            va = VLEV_FLOAT(&a_ptr[j], gvl);
            vx = VLEV_FLOAT(&x[j], gvl);
            vr = VFMULVV_FLOAT(va, vx, gvl);
            v_res = VFREDSUM_FLOAT(vr, v_res, gvl);
        }
        temp = (FLOAT)EXTRACT_FLOAT(v_res);
        y[iy] += alpha * temp;


        iy += inc_y;
        a_ptr += lda;
    }
#endif
    } else {
        BLASLONG stride_x = inc_x * sizeof(FLOAT);
        for(i = 0; i < n; i++){
            v_res = VFMVVF_FLOAT_M1(0, 1);
            gvl = VSETVL(m);
            j = 0;
            ix = 0;
            vr = VFMVVF_FLOAT(0, gvl);
            for(k = 0; k < m/gvl; k++){
                va = VLEV_FLOAT(&a_ptr[j], gvl);
                vx = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                vr = VFMULVV_FLOAT(va, vx, gvl);
                v_res = VFREDSUM_FLOAT(vr, v_res, gvl);
                j += gvl;
                ix += inc_x * gvl;
            }
            if(j < m){
                gvl = VSETVL(m-j);
                va = VLEV_FLOAT(&a_ptr[j], gvl);
                vx = VLSEV_FLOAT(&x[ix], stride_x, gvl);
                vr = VFMULVV_FLOAT(va, vx, gvl);
                v_res = VFREDSUM_FLOAT(vr, v_res, gvl);
                }
                temp = (FLOAT)EXTRACT_FLOAT(v_res);
                y[iy] += alpha * temp;


                iy += inc_y;
                a_ptr += lda;
            }
        }

    return (0);
}
