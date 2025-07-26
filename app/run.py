#!/usr/bin/env python3
"""
金融研报自动生成系统 - 主程序
支持公司报告、行业报告、宏观报告的生成
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='金融研报自动生成系统')
    parser.add_argument("mode", choices=["company", "industry", "macro"], 
                       help="报告类型: company(公司), industry(行业), macro(宏观)")
    parser.add_argument("--company_name", help="公司名称，例如：商汤科技")
    parser.add_argument("--company_code", help="公司股票代码，例如：00020.HK")
    parser.add_argument("--industry_name", help="行业名称，例如：智能风控&大数据征信服务")
    parser.add_argument("--marco_name", help="宏观主题，例如：生成式AI基建与算力投资趋势")
    parser.add_argument("--time", help="时间范围，例如：2023-2026")
    
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)

    # 设置环境变量以支持UTF-8编码
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    if args.mode == "company":
        if not args.company_name or not args.company_code:
            print("❌ 公司报告需要提供 --company_name 和 --company_code 参数")
            sys.exit(1)
        
        print(f"🏢 生成公司报告: {args.company_name} ({args.company_code})")
        cmd = [
            sys.executable, "run_company_research_report.py",
            "--company_name", args.company_name,
            "--company_code", args.company_code
        ]
        
    elif args.mode == "industry":
        if not args.industry_name:
            print("❌ 行业报告需要提供 --industry_name 参数")
            sys.exit(1)
            
        print(f"🏭 生成行业报告: {args.industry_name}")
        cmd = [
            sys.executable, "run_industry_research_report.py",
            "--industry_name", args.industry_name
        ]
        
    elif args.mode == "macro":
        if not args.marco_name or not args.time:
            print("❌ 宏观报告需要提供 --marco_name 和 --time 参数")
            sys.exit(1)
            
        print(f"🌍 生成宏观报告: {args.marco_name} ({args.time})")
        cmd = [
            sys.executable, "run_marco_research_report.py",
            "--marco_name", args.marco_name,
            "--time", args.time
        ]

    try:
        # 使用UTF-8环境变量运行子进程
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, encoding='utf-8')
        print("✅ 报告生成成功!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ 报告生成失败: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
