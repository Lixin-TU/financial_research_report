import os
import time
import yaml
import openai
from duckduckgo_search import DDGS
from pocketflow import Node, Flow
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 从环境变量中初始化 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

class IndustryResearchFlow(Node):  # 研报生成的决策节点
    def prep(self, shared):
        context = shared.get("context", [])
        context_str = yaml.dump(context, allow_unicode=True)
        industry = shared["industry"]  # 行业名称
        focus_areas = shared.get("focus_areas", [])
        return industry, context_str, focus_areas

    def exec(self, inputs):
        industry, context, focus_areas = inputs
        print(f"\n正在分析{industry}研究进度...")
        prompt = f"""
针对宏观经济研究，基于已有信息：{context}
关注指标：{', '.join(focus_areas)}

请分析并判断下一步行动：
1) 搜索更多信息
2) 开始生成某个章节内容
3) 完成研报生成

要特别关注：
- 核心经济指标数据（GDP、CPI、利率、汇率、制造业PMI、房地产开发投资完成情况、工业企业利润、工业用电量）的最新数据和趋势
- 重要政策文件与解读（如货币政策、财政政策等）
- 区域经济对比数据与分析
- 全球经济形势与联动影响
- 潜在风险因素与预警信号

请以 YAML 格式输出：
```yaml
action: search/generate/complete  # search表示继续搜索，generate表示生成章节，complete表示完成
reason: 做出此判断的原因
search_terms: # 如果是search，列出要搜索的关键词列表
  - 关键词1 
  - 关键词2
section: # 如果是generate，指定要生成的章节名称
  name: 章节名称 # 如：宏观经济概况/政策解读/风险分析等
  focus: 重点关注内容 # 具体要分析的要点
```"""
        resp = call_llm(prompt)
        yaml_str = resp.split("```yaml")[1].split("```", 1)[0].strip()
        result = yaml.safe_load(yaml_str)
        
        # 打印决策结果
        print(f"决策结果: {result['action']}")
        print(f"决策原因: {result['reason']}")
        if result['action'] == 'search':
            print("需要搜索的关键词:", result['search_terms'])
        elif result['action'] == 'generate':
            print(f"即将生成章节: {result['section']['name']}")
        
        return result

    def post(self, shared, prep_res, exec_res):
        action = exec_res.get("action")
        if action == "search":
            shared["search_terms"] = exec_res.get("search_terms", [])
            print("\n=== 开始信息搜索阶段 ===")
        elif action == "generate":
            shared["current_section"] = exec_res.get("section", {})
            print("\n=== 开始章节生成阶段 ===")
        elif action == "complete":
            print("\n=== 开始完成研报阶段 ===")
        return action

class SearchInfo(Node):  # 信息搜索节点
    def prep(self, shared):
        return shared.get("search_terms", [])

    def exec(self, search_terms):
        all_results = []
        total = len(search_terms)
        for i, term in enumerate(search_terms, 1):
            print(f"\n搜索关键词 ({i}/{total}): {term}")
            results = search_web(term)
            print(f"找到 {len(list(results))} 条相关信息")
            all_results.append({"term": term, "results": results})
            time.sleep(15)  # 避免请求过快
        return all_results

    def post(self, shared, prep_res, exec_res):
        context_list = shared.get("context", [])
        context_list.extend(exec_res)
        shared["context"] = context_list
        print("\n信息搜索完成，返回决策节点...")
        return "search_done"

class GenerateSection(Node):  # 章节生成节点
    def prep(self, shared):
        return (
            shared.get("industry"),
            shared.get("current_section", {}),
            shared.get("context", []),
            shared.get("focus_areas", [])
        )

    def exec(self, inputs):
        industry, section, context, focus_areas = inputs
        print(f"\n开始生成 {section['name']} 章节...")
        context_str = yaml.dump(context, allow_unicode=True)
        prompt = f"""
研究主题：{industry}
章节名称：{section['name']}
重点关注：{section['focus']}
核心指标：{', '.join(focus_areas)}
参考资料：{context_str}

请生成一个专业、详实的宏观经济研报章节。要求：
1. 数据支撑充分
   - 提供最新的经济指标数据
   - 使用权威来源的统计数据
   - 展示关键数据的历史趋势

2. 逻辑严谨
   - 建立清晰的因果关系
   - 解释政策传导机制
   - 分析各指标间的相互影响

3. 分析深入
   - 挖掘数据背后的原因
   - 评估政策效果
   - 预测未来趋势

4. 结构清晰
   - 层次分明
   - 重点突出
   - 观点明确

5. 专业视角
   - 使用专业术语
   - 多维度分析
   - 提供具体建议

如果涉及风险分析，请特别关注：
- 潜在的系统性风险
- 外部冲击因素
- 政策调整风险
- 区域性风险
- 具体防范建议
"""
        content = call_llm(prompt)
        print(f"章节 {section['name']} 生成完成!")
        print(f"内容长度: {len(content)} 字符")
        # 打印前100个字符预览
        print(f"内容预览: {content[:100]}...")
        return {
            "name": section["name"],
            "content": content
        }
        

    def post(self, shared, prep_res, exec_res):
        sections = shared.get("generated_sections", [])
        sections.append(exec_res)
        shared["generated_sections"] = sections
        print("\n返回决策节点，继续分析下一步...")
        return "continue"

class CompleteReport(Node):  # 研报完成节点
    def prep(self, shared):
        return (
            shared.get("industry"),
            shared.get("generated_sections", [])
        )

    def exec(self, inputs):
        industry, sections = inputs
        print(f"\n=== 开始整合最终研报 ===")
        # 整合所有章节内容
        content = f"# {industry}行业研究报告\n\n"
        for section in sections:
            print(f"添加章节: {section['name']}")
            content += f"\n## {section['name']}\n\n{section['content']}\n"
        return content

    def post(self, shared, prep_res, exec_res):
        print(f"\n=== 研报生成完成！===")
        print(f"研报已保存到 '研报2.md' 文件中")
        shared["report"] = exec_res
        return None

def call_llm(prompt: str) -> str:
    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def search_web(term: str):
    with DDGS() as ddgs:
        results = ddgs.text(keywords=term, region="cn-zh", max_results=20)
    return results

"""
示例用法
"""
if __name__ == "__main__":
    # 构建工作流
    research = IndustryResearchFlow()
    search = SearchInfo()
    generate = GenerateSection()
    complete = CompleteReport()
    
    # 设置转换关系
    research - "search" >> search
    research - "generate" >> generate
    research - "complete" >> complete
    search - "search_done" >> research
    generate - "continue" >> research


     # 运行工作流
    flow = Flow(start=research)
    shared_state = {
        "industry": "生成式AI基建与算力投资趋势（2023-2026）",
        "focus_areas": ["GDP", "CPI", "利率", "汇率", "制造业PMI", "房地产开发投资完成情况", "工业企业利润", "工业用电量"],
        "analysis_type": "macro_economic"
    }
    result = flow.run(shared_state)
    
    # 保存结果
    if result:
        with open("宏观研报.md", "w", encoding="utf-8") as f:
            f.write(result)
