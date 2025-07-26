#!/usr/bin/env bash
set -e

# 金融研报自动生成系统启动脚本
# 支持公司报告、行业报告、宏观报告的生成

mode="$1"

echo "🚀 启动金融研报自动生成系统..."
echo "📋 运行模式: $mode"

# 设置UTF-8编码环境变量
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# 确保输出目录存在
mkdir -p /app/outputs

if [ "$mode" = "company" ]; then
    if [ $# -ne 3 ]; then
        echo "❌ 参数错误！公司报告需要3个参数"
        echo "用法: bash run.sh company <company_name> <company_code>"
        echo "示例: bash run.sh company \"商汤科技\" \"00020.HK\""
        exit 1
    fi
    
    company_name="$2"
    company_code="$3"
    echo "🏢 生成公司报告: $company_name ($company_code)"
    # 直接调用公司报告脚本（按比赛要求）
    python run_company_research_report.py --company_name "$company_name" --company_code "$company_code"
    
elif [ "$mode" = "industry" ]; then
    if [ $# -ne 2 ]; then
        echo "❌ 参数错误！行业报告需要2个参数"
        echo "用法: bash run.sh industry <industry_name>"
        echo "示例: bash run.sh industry \"智能风控&大数据征信服务\""
        exit 1
    fi
    
    industry_name="$2"
    echo "🏭 生成行业报告: $industry_name"
    # 直接调用行业报告脚本（按比赛要求）
    python run_industry_research_report.py --industry_name "$industry_name"
    
elif [ "$mode" = "macro" ]; then
    if [ $# -ne 3 ]; then
        echo "❌ 参数错误！宏观报告需要3个参数"
        echo "用法: bash run.sh macro <marco_name> <time>"
        echo "示例: bash run.sh macro \"生成式AI基建与算力投资趋势\" \"2023-2026\""
        exit 1
    fi
    
    marco_name="$2"
    time_range="$3"
    echo "🌍 生成宏观报告: $marco_name ($time_range)"
    # 直接调用宏观报告脚本（按比赛要求）
    python run_marco_research_report.py --marco_name "$marco_name" --time "$time_range"
    
else
    echo "❌ 无效的运行模式: $mode"
    echo ""
    echo "📖 使用说明:"
    echo "  bash run.sh company  <company_name>  <company_code>"
    echo "  bash run.sh industry <industry_name>"
    echo "  bash run.sh macro    <marco_name>    <time>"
    echo ""
    echo "📝 示例:"
    echo "  bash run.sh company  \"商汤科技\" \"00020.HK\""
    echo "  bash run.sh industry \"智能风控&大数据征信服务\""
    echo "  bash run.sh macro    \"生成式AI基建与算力投资趋势\" \"2023-2026\""
    exit 1
fi

echo "✅ 报告生成完成!"