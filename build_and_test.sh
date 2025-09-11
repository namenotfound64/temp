#!/bin/bash

# OMATCOPY_CT 性能测试编译脚本
# 适用于 SG2044 服务器

echo "=== OMATCOPY_CT 性能测试编译脚本 ==="
echo "适用于 SG2044 RISC-V 服务器"
echo

# 检查编译器
if command -v riscv64-unknown-linux-gnu-gcc &> /dev/null; then
    CC="riscv64-unknown-linux-gnu-gcc"
    echo "使用 RISC-V 交叉编译器"
elif command -v gcc &> /dev/null; then
    CC="gcc"
    echo "使用系统 GCC 编译器"
else
    echo "错误: 未找到合适的编译器"
    exit 1
fi

# 显示编译器版本
echo "编译器版本:"
$CC --version | head -1
echo

# 编译标准版本（无RVV）
echo "[1/3] 编译标准版本（标量优化）..."
if [[ "$CC" == *"riscv64"* ]]; then
    $CC -O3 -march=rv64gc test_omatcopy_ct.c -lm -o test_omatcopy_ct_scalar -static
else
    $CC -O3 test_omatcopy_ct.c -lm -o test_omatcopy_ct_scalar
fi
if [ $? -eq 0 ]; then
    echo "✓ 标准版本编译成功: test_omatcopy_ct_scalar"
else
    echo "✗ 标准版本编译失败"
    exit 1
fi

# 编译RVV版本
echo "[2/3] 编译RVV优化版本..."
if [[ "$CC" == *"riscv64"* ]]; then
    $CC -O3 -march=rv64gcv -DUSE_RVV test_omatcopy_ct.c -lm -o test_omatcopy_ct_rvv -static
else
    $CC -O3 -DUSE_RVV test_omatcopy_ct.c -lm -o test_omatcopy_ct_rvv
fi
if [ $? -eq 0 ]; then
    echo "✓ RVV版本编译成功: test_omatcopy_ct_rvv"
else
    echo "⚠ RVV版本编译失败（可能不支持RVV扩展）"
    echo "  将只运行标量版本测试"
fi

echo
echo "[3/3] 编译完成！"
echo

# 检查CPU信息
echo "=== CPU 信息 ==="
if [ -f /proc/cpuinfo ]; then
    echo "CPU型号:"
    grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs
    echo "CPU核心数: $(nproc)"
    
    # 检查RVV支持
    if grep -q "v" /proc/cpuinfo; then
        echo "✓ 检测到向量扩展支持"
    else
        echo "⚠ 未检测到向量扩展支持"
    fi
fi
echo

# 运行测试
echo "=== 开始性能测试 ==="
echo

# 如果是交叉编译，提示用户需要在目标平台运行
if [[ "$CC" == *"riscv64"* ]]; then
    echo "⚠ 检测到交叉编译环境，生成的可执行文件需要在 RISC-V 平台上运行"
    echo "请将以下文件传输到目标 RISC-V 系统:"
    echo "  - test_omatcopy_ct_scalar (标量版本)"
    if [ -f "test_omatcopy_ct_rvv" ]; then
        echo "  - test_omatcopy_ct_rvv (RVV优化版本)"
    fi
    echo
    echo "在目标系统上运行:"
    echo "  ./test_omatcopy_ct_scalar  # 运行标量版本"
    if [ -f "test_omatcopy_ct_rvv" ]; then
        echo "  ./test_omatcopy_ct_rvv     # 运行RVV版本"
    fi
else
    if [ -f "test_omatcopy_ct_rvv" ]; then
        echo "运行 RVV 优化版本测试:"
        echo "----------------------------------------"
        ./test_omatcopy_ct_rvv
        echo
    fi
    
    echo "运行标量版本测试:"
    echo "----------------------------------------"
    ./test_omatcopy_ct_scalar
fi

echo
echo "=== 测试完成 ==="
echo "文件说明:"
echo "  test_omatcopy_ct_scalar - 标量优化版本"
if [ -f "test_omatcopy_ct_rvv" ]; then
    echo "  test_omatcopy_ct_rvv    - RVV向量化版本"
fi
echo "  test_omatcopy_ct.c      - 源代码文件"
echo "  build_and_test.sh       - 本编译脚本"
echo
echo "编译器信息:"
echo "  使用编译器: $CC"
if [[ "$CC" == *"riscv64"* ]]; then
    echo "  目标架构: RISC-V 64位"
    echo "  编译模式: 交叉编译 (静态链接)"
else
    echo "  目标架构: 本机架构"
    echo "  编译模式: 本地编译"
fi