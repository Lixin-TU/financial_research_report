"""
整合的金融研报生成器
包含数据采集、分析和深度研报生成的完整流程
- 第一阶段：数据采集与基础分析
- 第二阶段：深度研报生成与格式化输出
"""

import os
import glob
import time
import json
import yaml
import re
import shutil
import requests
from datetime import datetime
from dotenv import load_dotenv
import importlib
from urllib.parse import urlparse

from data_analysis_agent import quick_analysis
from data_analysis_agent.config.llm_config import LLMConfig
from data_analysis_agent.utils.llm_helper import LLMHelper
from utils.get_shareholder_info import get_shareholder_info, get_table_content
from utils.get_financial_statements import get_all_financial_statements, save_financial_statements_to_csv
from utils.identify_competitors import identify_competitors_with_ai
from utils.get_stock_intro import get_stock_intro, save_stock_intro_to_txt
from duckduckgo_search import DDGS
from utils.search_engine import SearchEngine

class IntegratedResearchReportGenerator:
    """整合的研报生成器类"""
    
    def __init__(self, target_company="商汤科技", target_company_code="00020", target_company_market="HK", search_engine="ddg"):
        # 环境变量与全局配置
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        # 打印模型
        print(f"🔧 使用的模型: {self.model}")
        self.target_company = target_company
        self.target_company_code = target_company_code
        self.target_company_market = target_company_market
        
        # 搜索引擎配置
        self.search_engine = SearchEngine(search_engine)
        print(f"🔍 搜索引擎已配置为: {search_engine.upper()}")
        
        # 目录配置
        self.data_dir = "./download_financial_statement_files"
        self.company_info_dir = "./company_info"
        self.industry_info_dir = "./industry_info"
        
        # 创建必要的目录
        for dir_path in [self.data_dir, self.company_info_dir, self.industry_info_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # LLM配置
        self.llm_config = LLMConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.7,
            max_tokens=16384,
        )
        self.llm = LLMHelper(self.llm_config)
        
        # 存储分析结果
        self.analysis_results = {}
    
    def stage1_data_collection(self):
        """第一阶段：数据采集与基础分析"""
        print("\n" + "="*80)
        print("🚀 开始第一阶段：数据采集与基础分析")
        print("="*80)
        
        # 1. 获取竞争对手列表
        print("🔍 识别竞争对手...")
        other_companies = identify_competitors_with_ai(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model,
            company_name=self.target_company
        )
        listed_companies = [company for company in other_companies if company.get('market') != "未上市"]
        
        # 2. 获取目标公司财务数据
        print(f"\n📊 获取目标公司 {self.target_company} 的财务数据...")
        target_financials = get_all_financial_statements(
            stock_code=self.target_company_code,
            market=self.target_company_market,
            period="年度",
            verbose=False
        )
        save_financial_statements_to_csv(
            financial_statements=target_financials,
            stock_code=self.target_company_code,
            market=self.target_company_market,
            company_name=self.target_company,
            period="年度",
            save_dir=self.data_dir
        )
        
        # 3. 获取竞争对手的财务数据
        print("\n📊 获取竞争对手的财务数据...")
        competitors_financials = {}
        for company in listed_companies:
            company_name = company.get('name')
            company_code = company.get('code')
            market_str = company.get('market', '')
            
            if "A" in market_str:
                market = "A"
                if not (company_code.startswith('SH') or company_code.startswith('SZ')):
                    if company_code.startswith('6'):
                        company_code = f"SH{company_code}"
                    else:
                        company_code = f"SZ{company_code}"
            elif "港" in market_str:
                market = "HK"
            
            print(f"  获取 {company_name}({market}:{company_code}) 的财务数据")
            try:
                company_financials = get_all_financial_statements(
                    stock_code=company_code,
                    market=market,
                    period="年度",
                    verbose=False
                )
                save_financial_statements_to_csv(
                    financial_statements=company_financials,
                    stock_code=company_code,
                    market=market,
                    company_name=company_name,
                    period="年度",
                    save_dir=self.data_dir
                )
                competitors_financials[company_name] = company_financials
                time.sleep(2)
            except Exception as e:
                print(f"  获取 {company_name} 财务数据失败: {e}")
        
        # 4. 获取公司基础信息
        print("\n🏢 获取公司基础信息...")
        all_base_info_targets = [(self.target_company, self.target_company_code, self.target_company_market)]
        
        for company in listed_companies:
            company_name = company.get('name')
            company_code = company.get('code')
            market_str = company.get('market', '')
            if "A" in market_str:
                market = "A"
                if not (company_code.startswith('SH') or company_code.startswith('SZ')):
                    if company_code.startswith('6'):
                        company_code = f"SH{company_code}"
                    else:
                        company_code = f"SZ{company_code}"
            elif "港" in market_str:
                market = "HK"
            all_base_info_targets.append((company_name, company_code, market))
        
        # 添加特定公司如百度
        all_base_info_targets.append(("百度", "09888", "HK"))
        
        for company_name, company_code, market in all_base_info_targets:
            print(f"  获取 {company_name}({market}:{company_code}) 的基础信息")
            company_info = get_stock_intro(company_code, market=market)
            if company_info:
                save_path = os.path.join(self.company_info_dir, f"{company_name}_{market}_{company_code}_info.txt")
                save_stock_intro_to_txt(company_code, market, save_path)
                print(f"    信息已保存到: {save_path}")
            else:
                print(f"    未能获取到 {company_name} 的基础信息")
            time.sleep(1)
        
        # 5. 搜索行业信息
        print("\n🔍 搜索行业信息...")
        all_search_results = {}
          # 搜索目标公司行业信息
        target_search_keywords = f"{self.target_company} 行业地位 市场份额 竞争分析 业务模式"
        print(f"  正在搜索: {target_search_keywords}")
        # 进行目标公司搜索
        target_results = self.search_engine.search(target_search_keywords, 10)
        all_search_results[self.target_company] = target_results

        # 搜索竞争对手行业信息
        for company in listed_companies:
            company_name = company.get('name')
            search_keywords = f"{company_name} 行业地位 市场份额 业务模式 发展战略"
            print(f"  正在搜索: {search_keywords}")
            competitor_results = self.search_engine.search(search_keywords, 10)
            all_search_results[company_name] = competitor_results
            # 增加延迟避免请求过于频繁
            time.sleep(self.search_engine.delay * 2)
        
        # 保存搜索结果
        search_results_file = os.path.join(self.industry_info_dir, "all_search_results.json")
        with open(search_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_search_results, f, ensure_ascii=False, indent=2)
        
        # 6. 运行财务分析
        print("\n📈 运行财务分析...")
        
        # 单公司分析
        results = self.analyze_companies_in_directory(self.data_dir, self.llm_config)
        
        # 两两对比分析
        comparison_results = self.run_comparison_analysis(
            self.data_dir, self.target_company, self.llm_config
        )
        
        # 合并所有报告
        merged_results = self.merge_reports(results, comparison_results)
        
        # 商汤科技估值与预测分析
        sensetime_files = self.get_sensetime_files(self.data_dir)
        sensetime_valuation_report = None
        if sensetime_files:
            sensetime_valuation_report = self.analyze_sensetime_valuation(sensetime_files, self.llm_config)
        
        # 7. 整理所有分析结果
        print("\n📋 整理分析结果...")
        
        # 整理公司信息
        company_infos = self.get_company_infos(self.company_info_dir)
        company_infos = self.llm.call(
            f"请整理以下公司信息内容，确保格式清晰易读，并保留关键信息：\n{company_infos}",
            system_prompt="你是一个专业的公司信息整理师。",
            max_tokens=16384,
            temperature=0.5
        )
        
        # 整理股权信息
        info = get_shareholder_info()
        shangtang_shareholder_info = info.get("tables")
        table_content = get_table_content(shangtang_shareholder_info)
        shareholder_analysis = self.llm.call(
            "请分析以下股东信息表格内容：\n" + table_content,
            system_prompt="你是一个专业的股东信息分析师。",
            max_tokens=16384,
            temperature=0.5
        )
        
        # 整理行业信息搜索结果
        with open(search_results_file, 'r', encoding='utf-8') as f:
            all_search_results = json.load(f)
        search_res = ""
        for company, results in all_search_results.items():
            search_res += f"【{company}搜索信息开始】\n"
            for result in results:
                search_res += f"标题: {result.get('title', '无标题')}\n"
                search_res += f"链接: {result.get('href', '无链接')}\n"
                search_res += f"摘要: {result.get('body', '无摘要')}\n"
                search_res += "----\n"
            search_res += f"【{company}搜索信息结束】\n\n"
        
        # 保存阶段一结果
        formatted_report = self.format_final_reports(merged_results)
        
        # 统一保存为markdown
        md_output_file = f"财务研报汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 公司基础信息\n\n## 整理后公司信息\n\n{company_infos}\n\n")
            f.write(f"# 股权信息分析\n\n{shareholder_analysis}\n\n")
            f.write(f"# 行业信息搜索结果\n\n{search_res}\n\n")
            f.write(f"# 财务数据分析与两两对比\n\n{formatted_report}\n\n")
            if sensetime_valuation_report and isinstance(sensetime_valuation_report, dict):
                f.write(f"# 商汤科技估值与预测分析\n\n{sensetime_valuation_report.get('final_report', '未生成报告')}\n\n")
        
        print(f"\n✅ 第一阶段完成！基础分析报告已保存到: {md_output_file}")
        
        # 存储结果供第二阶段使用
        self.analysis_results = {
            'md_file': md_output_file,
            'company_infos': company_infos,
            'shareholder_analysis': shareholder_analysis,
            'search_res': search_res,
            'formatted_report': formatted_report,
            'sensetime_valuation_report': sensetime_valuation_report
        }
        
        return md_output_file
    
    def stage2_deep_report_generation(self, md_file_path):
        """第二阶段：深度研报生成"""
        print("\n" + "="*80)
        print("🚀 开始第二阶段：深度研报生成")
        print("="*80)
        
        # 处理图片路径
        print("🖼️ 处理图片路径...")
        new_md_path = md_file_path.replace('.md', '_images.md')
        images_dir = os.path.join(os.path.dirname(md_file_path), 'images')
        self.extract_images_from_markdown(md_file_path, images_dir, new_md_path)
        
        # 加载报告内容
        report_content = self.load_report_content(new_md_path)
        background = self.get_background()
        
        # 生成大纲
        print("\n📋 生成报告大纲...")
        parts = self.generate_outline(self.llm, background, report_content)
        
        # 分段生成深度研报
        print("\n✍️ 开始分段生成深度研报...")
        full_report = ['# 商汤科技公司研报\n']
        prev_content = ''
        
        for idx, part in enumerate(parts):
            part_title = part.get('part_title', f'部分{idx+1}')
            print(f"\n  正在生成：{part_title}")
            is_last = (idx == len(parts) - 1)
            section_text = self.generate_section(
                self.llm, part_title, prev_content, background, report_content, is_last
            )
            full_report.append(section_text)
            print(f"  ✅ 已完成：{part_title}")
            prev_content = '\n'.join(full_report)
        
        # 保存最终报告
        final_report = '\n\n'.join(full_report)
        output_file = f"深度财务研报分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self.save_markdown(final_report, output_file)
        
        # 格式化和转换
        print("\n🎨 格式化报告...")
        self.format_markdown(output_file)
        
        print("\n📄 转换为Word文档...")
        self.convert_to_docx(output_file)
        
        print(f"\n✅ 第二阶段完成！深度研报已保存到: {output_file}")
        return output_file
    
    def run_full_pipeline(self):
        """运行完整流程"""
        print("\n" + "="*100)
        print("🎯 启动整合的金融研报生成流程")
        print("="*100)
        
        # 第一阶段：数据采集与基础分析
        md_file = self.stage1_data_collection()
        
        # 第二阶段：深度研报生成
        final_report = self.stage2_deep_report_generation(md_file)
        
        print("\n" + "="*100)
        print("🎉 完整流程执行完毕！")
        print(f"📊 基础分析报告: {md_file}")
        print(f"📋 深度研报: {final_report}")
        print("="*100)
        
        return md_file, final_report

    # ========== 辅助方法（从原始脚本移植） ==========
    
    def get_company_infos(self, data_dir="./company_info"):
        """获取公司信息"""
        all_files = os.listdir(data_dir)
        company_infos = ""
        for file in all_files:
            if file.endswith(".txt"):
                company_name = file.split(".")[0]
                with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                company_infos += f"【公司信息开始】\n公司名称: {company_name}\n{content}\n【公司信息结束】\n\n"
        return company_infos
    
    def get_company_files(self, data_dir):
        """获取公司文件"""
        all_files = glob.glob(f"{data_dir}/*.csv")
        companies = {}
        for file in all_files:
            filename = os.path.basename(file)
            company_name = filename.split("_")[0]
            companies.setdefault(company_name, []).append(file)
        return companies
    
    def analyze_individual_company(self, company_name, files, llm_config, query=None, verbose=True):
        """分析单个公司"""
        if query is None:
            query = "基于表格的数据，分析有价值的内容，并绘制相关图表。最后生成汇报给我。"
        report = quick_analysis(
            query=query, files=files, llm_config=llm_config, 
            absolute_path=True, max_rounds=20
        )
        return report
    
    def format_final_reports(self, all_reports):
        """格式化最终报告"""
        formatted_output = []
        for company_name, report in all_reports.items():
            formatted_output.append(f"【{company_name}财务数据分析结果开始】")
            final_report = report.get("final_report", "未生成报告")
            formatted_output.append(final_report)
            formatted_output.append(f"【{company_name}财务数据分析结果结束】")
            formatted_output.append("")
        return "\n".join(formatted_output)
    
    def analyze_companies_in_directory(self, data_directory, llm_config, query="基于表格的数据，分析有价值的内容，并绘制相关图表。最后生成汇报给我。"):
        """分析目录中的所有公司"""
        company_files = self.get_company_files(data_directory)
        all_reports = {}
        for company_name, files in company_files.items():
            report = self.analyze_individual_company(company_name, files, llm_config, query, verbose=False)
            if report:
                all_reports[company_name] = report
        return all_reports
    
    def compare_two_companies(self, company1_name, company1_files, company2_name, company2_files, llm_config):
        """比较两个公司"""
        query = "基于两个公司的表格的数据，分析有共同点的部分，绘制对比分析的表格，并绘制相关图表。最后生成汇报给我。"
        all_files = company1_files + company2_files
        report = quick_analysis(
            query=query,
            files=all_files,
            llm_config=llm_config,
            absolute_path=True,
            max_rounds=20
        )
        return report
    
    def run_comparison_analysis(self, data_directory, target_company_name, llm_config):
        """运行对比分析"""
        company_files = self.get_company_files(data_directory)
        if not company_files or target_company_name not in company_files:
            return {}
        competitors = [company for company in company_files.keys() if company != target_company_name]
        comparison_reports = {}
        for competitor in competitors:
            comparison_key = f"{target_company_name}_vs_{competitor}"
            report = self.compare_two_companies(
                target_company_name, company_files[target_company_name],
                competitor, company_files[competitor],
                llm_config
            )
            if report:
                comparison_reports[comparison_key] = {
                    'company1': target_company_name,
                    'company2': competitor,
                    'report': report
                }
        return comparison_reports
    
    def merge_reports(self, individual_reports, comparison_reports):
        """合并报告"""
        merged = {}
        for company, report in individual_reports.items():
            merged[company] = report
        for comp_key, comp_data in comparison_reports.items():
            merged[comp_key] = comp_data['report']
        return merged
    
    def get_sensetime_files(self, data_dir):
        """获取商汤科技的财务数据文件"""
        all_files = glob.glob(f"{data_dir}/*.csv")
        sensetime_files = []
        for file in all_files:
            filename = os.path.basename(file)
            company_name = filename.split("_")[0]
            if "商汤" in company_name or "SenseTime" in company_name:
                sensetime_files.append(file)
        return sensetime_files
    
    def analyze_sensetime_valuation(self, files, llm_config):
        """分析商汤科技的估值与预测"""
        query = "基于三大表的数据，构建估值与预测模型，模拟关键变量变化对财务结果的影响,并绘制相关图表。最后生成汇报给我。"
        report = quick_analysis(
            query=query, files=files, llm_config=llm_config, absolute_path=True, max_rounds=20
        )
        return report
    
    # ========== 深度研报生成相关方法 ==========
    
    def load_report_content(self, md_path):
        """加载报告内容"""
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def get_background(self):
        """获取背景信息"""
        return '''
本报告基于自动化采集与分析流程，涵盖如下环节：
- 公司基础信息等数据均通过akshare、公开年报、主流财经数据源自动采集。
- 财务三大报表数据来源：东方财富-港股-财务报表-三大报表 (https://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/index)
- 主营业务信息来源：同花顺-主营介绍 (https://basic.10jqka.com.cn/new/000066/operate.html)
- 股东结构信息来源：同花顺-股东信息 (https://basic.10jqka.com.cn/HK0020/holder.html) 通过网页爬虫技术自动采集
- 行业信息通过DuckDuckGo等公开搜索引擎自动抓取，引用了权威新闻、研报、公司公告等。
- 财务分析、对比分析、估值与预测均由大模型（如GPT-4）自动生成，结合了行业对标、财务比率、治理结构等多维度内容。
- 相关数据与分析均在脚本自动化流程下完成，确保数据来源可追溯、分析逻辑透明。
- 详细引用与外部链接已在正文中标注。
- 数据接口说明与免责声明见文末。
'''
    
    def generate_outline(self, llm, background, report_content):
        """生成大纲"""
        outline_prompt = f"""
你是一位顶级金融分析师和研报撰写专家。请基于以下背景和财务研报汇总内容，生成一份详尽的《商汤科技公司研报》分段大纲，要求：
- 以yaml格式输出，务必用```yaml和```包裹整个yaml内容，便于后续自动分割。
- 每一项为一个主要部分，每部分需包含：
  - part_title: 章节标题
  - part_desc: 本部分内容简介
- 章节需覆盖公司基本面、财务分析、行业对比、估值与预测、治理结构、投资建议、风险提示、数据来源等。
- 只输出yaml格式的分段大纲，不要输出正文内容。

【背景说明开始】
{background}
【背景说明结束】

【财务研报汇总内容开始】
{report_content}
【财务研报汇总内容结束】
"""
        outline_list = llm.call(
            outline_prompt,
            system_prompt="你是一位顶级金融分析师和研报撰写专家，善于结构化、分段规划输出，分段大纲必须用```yaml包裹，便于后续自动分割。",
            max_tokens=4096,
            temperature=0.3
        )
        print("\n===== 生成的分段大纲如下 =====\n")
        print(outline_list)
        try:
            if '```yaml' in outline_list:
                yaml_block = outline_list.split('```yaml')[1].split('```')[0]
            else:
                yaml_block = outline_list
            parts = yaml.safe_load(yaml_block)
            if isinstance(parts, dict):
                parts = list(parts.values())
        except Exception as e:
            print(f"[大纲yaml解析失败] {e}")
            parts = []
        return parts
    
    def generate_section(self, llm, part_title, prev_content, background, report_content, is_last):
        """生成章节"""
        section_prompt = f"""
你是一位顶级金融分析师和研报撰写专家。请基于以下内容，直接输出\"{part_title}\"这一部分的完整研报内容。

**重要要求：**
1. 直接输出完整可用的研报内容，以\"## {part_title}\"开头
2. 在正文中引用数据、事实、图片等信息时，适当位置插入参考资料符号（如[1][2][3]），符号需与文末引用文献编号一致
3. **图片引用要求（务必严格遵守）：**
   - 只允许引用【财务研报汇总内容】中真实存在的图片地址（格式如：./images/图片名字.png），必须与原文完全一致。
   - 禁止虚构、杜撰、改编、猜测图片地址，未在【财务研报汇总内容】中出现的图片一律不得引用。
   - 如需插入图片，必须先在【财务研报汇总内容】中查找，未找到则不插入图片，绝不编造图片。
   - 如引用了不存在的图片，将被判为错误输出。
4. 不要输出任何【xxx开始】【xxx结束】等分隔符
5. 不要输出\"建议补充\"、\"需要添加\"等提示性语言
6. 不要编造图片地址或数据
7. 内容要详实、专业，可直接使用

**数据来源标注：**
- 财务数据标注：（数据来源：东方财富-港股-财务报表[1]）
- 主营业务信息标注：（数据来源：同花顺-主营介绍[2]）
- 股东结构信息标注：（数据来源：同花顺-股东信息网页爬虫[3]）

【本次任务】
{part_title}

【已生成前文】
{prev_content}

【背景说明开始】
{background}
【背景说明结束】

【财务研报汇总内容开始】
{report_content}
【财务研报汇总内容结束】
"""
        if is_last:
            section_prompt += """
请在本节最后以"引用文献"格式，列出所有正文中用到的参考资料，格式如下：
[1] 东方财富-港股-财务报表: https://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/index
[2] 同花顺-主营介绍: https://basic.10jqka.com.cn/new/000066/operate.html
[3] 同花顺-股东信息: https://basic.10jqka.com.cn/HK0020/holder.html
"""
        section_text = llm.call(
            section_prompt,
            system_prompt="你是顶级金融分析师，专门生成完整可用的研报内容。输出必须是完整的研报正文，无需用户修改。严格禁止输出分隔符、建议性语言或虚构内容。只允许引用真实存在于【财务研报汇总内容】中的图片地址，严禁虚构、猜测、改编图片路径。如引用了不存在的图片，将被判为错误输出。",
            max_tokens=16384,
            temperature=0.5
        )
        return section_text
    
    def save_markdown(self, content, output_file):
        """保存markdown文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n📁 深度财务研报分析已保存到: {output_file}")
    
    def format_markdown(self, output_file):
        """格式化markdown文件"""
        try:
            import subprocess
            format_cmd = ["mdformat", output_file]
            subprocess.run(format_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"✅ 已用 mdformat 格式化 Markdown 文件: {output_file}")
        except Exception as e:
            print(f"[提示] mdformat 格式化失败: {e}\n请确保已安装 mdformat (pip install mdformat)")
    
    def convert_to_docx(self, output_file, docx_output=None):
        """转换为Word文档"""
        if docx_output is None:
            docx_output = output_file.replace('.md', '.docx')
        try:
            import subprocess
            import os
            pandoc_cmd = [
                "pandoc",
                output_file,
                "-o",
                docx_output,
                "--standalone",
                "--resource-path=.",
                "--extract-media=."
            ]
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            subprocess.run(pandoc_cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
            print(f"\n📄 Word版报告已生成: {docx_output}")
        except subprocess.CalledProcessError as e:
            print(f"[提示] pandoc转换失败。错误信息: {e.stderr}")
            print("[建议] 检查图片路径是否正确，或使用 --extract-media 选项")
        except Exception as e:
            print(f"[提示] 若需生成Word文档，请确保已安装pandoc。当前转换失败: {e}")
    
    # ========== 图片处理相关方法 ==========
    
    def ensure_dir(self, path):
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def is_url(self, path):
        """判断是否为URL"""
        return path.startswith('http://') or path.startswith('https://')
    
    def download_image(self, url, save_path):
        """下载图片"""
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"[下载失败] {url}: {e}")
            return False
    
    def copy_image(self, src, dst):
        """复制图片"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"[复制失败] {src}: {e}")
            return False
    
    def extract_images_from_markdown(self, md_path, images_dir, new_md_path):
        """从markdown中提取图片"""
        self.ensure_dir(images_dir)
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 匹配 ![alt](path) 形式的图片
        pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        matches = pattern.findall(content)
        used_names = set()
        replace_map = {}
        not_exist_set = set()

        for img_path in matches:
            img_path = img_path.strip()
            # 取文件名
            if self.is_url(img_path):
                filename = os.path.basename(urlparse(img_path).path)
            else:
                filename = os.path.basename(img_path)
            # 防止重名
            base, ext = os.path.splitext(filename)
            i = 1
            new_filename = filename
            while new_filename in used_names:
                new_filename = f"{base}_{i}{ext}"
                i += 1
            used_names.add(new_filename)
            new_img_path = os.path.join(images_dir, new_filename)
            # 下载或复制
            img_exists = True
            if self.is_url(img_path):
                success = self.download_image(img_path, new_img_path)
                if not success:
                    img_exists = False
            else:
                # 支持绝对和相对路径
                abs_img_path = img_path
                if not os.path.isabs(img_path):
                    abs_img_path = os.path.join(os.path.dirname(md_path), img_path)
                if not os.path.exists(abs_img_path):
                    print(f"[警告] 本地图片不存在: {abs_img_path}")
                    img_exists = False
                else:
                    self.copy_image(abs_img_path, new_img_path)
            # 记录替换
            if img_exists:
                replace_map[img_path] = f'./images/{new_filename}'
            else:
                not_exist_set.add(img_path)

        # 替换 markdown 内容，不存在的图片直接删除整个图片语法
        def replace_func(match):
            orig = match.group(1).strip()
            if orig in not_exist_set:
                return ''  # 删除不存在的图片语法
            return match.group(0).replace(orig, replace_map.get(orig, orig))

        new_content = pattern.sub(replace_func, content)
        with open(new_md_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"图片处理完成！新文件: {new_md_path}")


def main():
    """主函数"""
    import argparse
    
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='整合的金融研报生成器')
    parser.add_argument('--search-engine', choices=['ddg', 'sogou'], default='sogou',
                       help='搜索引擎选择: ddg (DuckDuckGo) 或 sogou (搜狗), 默认: ddg')
    parser.add_argument('--company', default='商汤科技', help='目标公司名称')
    parser.add_argument('--code', default='00020', help='股票代码')
    parser.add_argument('--market', default='HK', help='市场代码')
    
    args = parser.parse_args()
    
    # 创建生成器实例
    generator = IntegratedResearchReportGenerator(
        target_company=args.company,
        target_company_code=args.code, 
        target_company_market=args.market,
        search_engine=args.search_engine
    )
    
    # 运行完整流程
    basic_report, deep_report = generator.run_full_pipeline()
    
    print("\n" + "="*100)
    print("🎯 程序执行完毕！生成的文件：")
    print(f"📊 基础分析报告: {basic_report}")
    print(f"📋 深度研报: {deep_report}")
    print("="*100)


if __name__ == "__main__":
    main()
