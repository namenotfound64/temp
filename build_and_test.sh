#!/bin/bash

# OMATCOPY_CT 性能测试编译脚本
# 适用于 SG2044 服务器

echo "=== OMATCOPY_CT 性能测试编译脚本 ==="
echo "适用于 SG2044 RISC-V 服务器"
echo

# 检查编译器
if ! command -v gcc &> /dev/null; then
    echo "错误: 未找到 GCC 编译器"
    exit 1
fi

# 显示 GCC 版本
echo "GCC 版本:"
gcc --version | head -1
echo

# 编译标准版本（无RVV）
echo "[1/3] 编译标准版本（标量优化）..."
gcc -O3 -march=rv64gc test_omatcopy_ct.c -lm -o test_omatcopy_ct_scalar
if [ $? -eq 0 ]; then
    echo "✓ 标准版本编译成功: test_omatcopy_ct_scalar"
else
    echo "✗ 标准版本编译失败"
    exit 1
fi

# 编译RVV版本
echo "[2/3] 编译RVV优化版本..."
gcc -O3 -march=rv64gcv -DUSE_RVV test_omatcopy_ct.c -lm -o test_omatcopy_ct_rvv
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

if [ -f "test_omatcopy_ct_rvv" ]; then
    echo "运行 RVV 优化版本测试:"
    echo "----------------------------------------"
    ./test_omatcopy_ct_rvv
    echo
fi

echo "运行标量版本测试:"
echo "----------------------------------------"
./test_omatcopy_ct_scalar

echo
echo "=== 测试完成 ==="
echo "文件说明:"
echo "  test_omatcopy_ct_scalar - 标量优化版本"
if [ -f "test_omatcopy_ct_rvv" ]; then
    echo "  test_omatcopy_ct_rvv    - RVV向量化版本"
fi
echo "  test_omatcopy_ct.c      - 源代码文件"
echo "  build_and_test.sh       - 本编译脚本"