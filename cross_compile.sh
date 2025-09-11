#!/bin/bash

# RISC-V交叉编译脚本
# 用于在x86_64主机上编译RISC-V二进制文件，然后传输到真实RISC-V服务器测试

set -e

echo "=== RISC-V交叉编译脚本 ==="
echo "编译器: riscv64-unknown-linux-gnu-gcc"
echo "目标架构: RISC-V 64位"
echo ""

# 检查交叉编译器是否存在
if ! command -v riscv64-unknown-linux-gnu-gcc &> /dev/null; then
    echo "错误: 未找到 riscv64-unknown-linux-gnu-gcc 交叉编译器"
    echo "请确保已安装RISC-V工具链"
    exit 1
fi

echo "编译器版本:"
riscv64-unknown-linux-gnu-gcc --version | head -1
echo ""

# 编译标量版本
echo "编译标量版本..."
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gc -static \
    -o test_omatcopy_ct_scalar test_omatcopy_ct.c -lm
echo "✓ 标量版本编译完成: test_omatcopy_ct_scalar"

# 编译RVV版本
echo "编译RVV版本..."
riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -DUSE_RVV -static \
    -o test_omatcopy_ct_rvv test_omatcopy_ct.c -lm
echo "✓ RVV版本编译完成: test_omatcopy_ct_rvv"

# 显示文件信息
echo ""
echo "=== 编译结果 ==="
ls -lh test_omatcopy_ct_*
echo ""
echo "文件架构信息:"
file test_omatcopy_ct_scalar test_omatcopy_ct_rvv

echo ""
echo "=== 使用说明 ==="
echo "1. 将以下文件传输到RISC-V服务器:"
echo "   - test_omatcopy_ct_scalar (标量版本)"
echo "   - test_omatcopy_ct_rvv (RVV版本)"
echo ""
echo "2. 在RISC-V服务器上运行测试:"
echo "   ./test_omatcopy_ct_scalar  # 测试标量版本"
echo "   ./test_omatcopy_ct_rvv     # 测试RVV版本"
echo ""
echo "3. 传输命令示例:"
echo "   scp test_omatcopy_ct_* user@riscv-server:/path/to/test/"
echo ""
echo "编译完成！"