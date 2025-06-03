#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
	unsigned int gvl = __riscv_vsetvl_e32m2(8);
	float *A = (float *)malloc(4 * 4 * sizeof(float));
	for (int i =0;i<4*4;i++){
		A[i]=i%10;
	}
	vfloat32m2_t A0 = __riscv_vle32_v_f32m2(&A[0], gvl);
	float tmp[8];
    	__riscv_vse32_v_f32m2(tmp, A0, gvl);

	printf("A0 vector contents:\n");
	    for (int i = 0; i < gvl; i++) {
		printf("tmp[%d] = %.2f\n", i, tmp[i]);
	    }

	    free(A);
	return 0;
}
