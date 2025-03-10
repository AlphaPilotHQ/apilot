#!/usr/bin/env bash

# 定义Python解释器，如果没有指定则使用python3
python=$1
shift 1
[[ -z $python ]] && python=python3

echo "===== 开始安装APilot量化交易框架 ====="

# 升级pip和wheel
echo "升级pip和wheel..."
$python -m pip install --upgrade pip wheel

# 安装TA-Lib（技术分析库）
echo "检查并安装TA-Lib..."
function install-ta-lib()
{
    echo "安装TA-Lib C库..."
    export HOMEBREW_NO_AUTO_UPDATE=true
    brew install ta-lib
}

function ta-lib-exists()
{
    ta-lib-config --libs > /dev/null
}

ta-lib-exists || install-ta-lib

# 按正确顺序安装依赖
echo "安装核心数值计算依赖..."
$python -m pip install numpy==1.26.4

echo "安装TA-Lib Python包..."
$python -m pip install ta-lib==0.4.24

# 安装其他依赖
echo "安装其他Python依赖..."
$python -m pip install -r requirements.txt

# 安装APilot框架
echo "安装APilot框架..."
$python -m pip install .

echo "===== APilot量化交易框架安装完成 ====="
echo "使用官方PyPI源安装，无UI组件版本"
