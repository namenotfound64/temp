import numpy as np
import torch
# 设置矩阵尺寸
M, K, N = 31, 31, 31  # 可修改为更大规模

# 生成随机输入矩阵，类型为float16
A = np.random.randint(0, 11, size=(M, K)).astype(np.float16)
B = np.random.randint(0, 11, size=(K, N)).astype(np.float16)
A_torch = torch.tensor(A, dtype=torch.float16, device='cuda')
B_torch = torch.tensor(B, dtype=torch.float16, device='cuda')
C_torch = torch.matmul(A_torch, B_torch)
C_ref = C_torch.cpu().numpy().astype(np.float32)

def format_array_c(name, array, c_type="hfloat16"):
    flat = array.flatten()
    elements = ", ".join(f"{x:.5f}" for x in flat)
    return f"{c_type} {name}[{len(flat)}] = {{ {elements} }};\n"

def format_array_c_float(name, array):
    flat = array.flatten()
    elements = ", ".join(f"{x:.5f}" for x in flat)
    return f"float {name}[{len(flat)}] = {{ {elements} }};\n"

# 写入C文件
with open("generated_test.c", "w") as f:
    f.write('#include <stdio.h>\n')
    f.write('#include <stdlib.h>\n')
    f.write('#include <string.h>\n')
    f.write('#include <cblas.h>\n\n')

    f.write(f"const int M = {M}, K = {K}, N = {N};\n")
    f.write("const float alpha = 1.0f, beta = 0.0f;\n\n")

    f.write(format_array_c("A", A))
    f.write(format_array_c("B", B))
    f.write(f"float C[{M*N}] = {{ 0 }};\n\n")

    f.write("int main() {\n")
    f.write("    cblas_shgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n")
    f.write("                 M, N, K,\n")
    f.write("                 alpha,\n")
    f.write("                 A, K,\n")
    f.write("                 B, N,\n")
    f.write("                 beta,\n")
    f.write("                 C, N);\n\n")

    f.write('    printf("Result C = A * B:\\n");\n')
    f.write("    for (int i = 0; i < M * N; i++) {\n")
    f.write("        printf(\"%.5f \", C[i]);\n")
    f.write("        if ((i + 1) % N == 0) printf(\"\\n\");\n")
    f.write("    }\n")
    f.write("    return 0;\n")
    f.write("}\n\n")

    f.write("// Reference result computed in Python:\n")
    c_ref_flat = ", ".join(f"{x:.5f}" for x in C_ref.flatten())
    f.write(f"// C_ref = {{ {c_ref_flat} }}\n")

