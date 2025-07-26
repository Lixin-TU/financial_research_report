# AFAC2025挑战组-赛题四：智能体赋能的金融多模态报告自动化生成

![AFAC Logo](./app/afac_LOGO.png)

A榜：rank 4  (2025-07-14)\
B榜：rank 17 (2025-07-24)

## 🏗️ 系统架构设计

### 核心架构
```
智能研报生成系统
├── 入口层: run.py, run.sh (命令行接口)
├── 业务层: 公司/行业/宏观报告生成模块
├── 工作流层: 搜索→生成→评估→改进循环控制
├── 服务层: LLM调用、信息搜索、质量评估
└── 工具层: 图表生成、文档处理、数据分析
```

### 核心功能模块

**📊 报告生成引擎**
- 公司报告: 财务分析 + 业务评估 + 估值模型 + 投资建议
- 行业报告: 产业链分析 + 竞争格局 + 趋势预测
- 宏观报告: 经济指标 + 政策分析 + 风险评估

**🔍 智能工作流**
- 多轮搜索策略 (最多6轮)
- CSA合规验证 (8维度评分)
- 自动改进循环 (最多8次迭代)

**📈 多模态输出**
- 专业Word文档 + Markdown格式
- 4类金融分析图表自动生成
- YAML格式评估报告

## 🚀 本地部署指南

### 环境要求
- **Python**: 3.9-3.11
- **内存**: 8GB+ RAM  
- **系统**: Windows/macOS/Linux

### 快速部署
```bash
# 1. 克隆项目
git clone <项目地址>
cd financial_research_report/docker_image

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥 (创建.env文件)
cd app
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3" >> .env
echo "OPENAI_MODEL=deepseek-r1-250528" >> .env

# 4. 测试运行
python run_company_research_report.py --company_name "商汤科技" --company_code "00020.HK"
```

### 依赖包说明
```bash
# 核心包
pip install matplotlib seaborn pandas numpy python-docx PyYAML
pip install python-dotenv openai requests beautifulsoup4 nest-asyncio

# 金融数据
pip install akshare efinance duckdb

# 搜索功能  
pip install duckduckgo-search
```

### 中文字体配置
```bash
# Windows: 系统自带，无需配置
# macOS: brew install font-source-han-sans
# Ubuntu: sudo apt install fonts-wqy-zenhei fonts-wqy-microhei
```

## 🎯 使用接口

### 命令行接口
```bash
# 公司报告
python run_company_research_report.py --company_name "公司名" --company_code "股票代码"

# 行业报告  
python run_industry_research_report.py --industry_name "行业名称"

# 宏观报告
python run_marco_research_report.py --marco_name "主题名称" --time "时间范围"
```

### Docker接口
```bash
# 构建镜像
docker build --no-cache -t research-agent:latest -f docker_image\Dockerfile docker_image

# 运行报告 (示例)
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh company "商汤科技" "00020.HK"
```

## 📊 测试用例

### B榜测试
```bash
# 公司报告 - 4Paradigm
python run_company_research_report.py --company_name "4Paradigm" --company_code "06682.HK"

# 行业报告 - 中国智能服务机器人产业
python run_industry_research_report.py --industry_name "中国智能服务机器人产业"

# 宏观报告 - 国家级人工智能+政策效果评估 (2023-2025)
python run_marco_research_report.py --marco_name "国家级人工智能+政策效果评估" --time "2023-2025"
```

### A榜测试
```bash
# 公司报告 - 商汤科技
python run_company_research_report.py --company_name "商汤科技" --company_code "00020.HK"

# 行业报告 - 智能风控&大数据征信服务
python run_industry_research_report.py --industry_name "智能风控&大数据征信服务"

# 宏观报告 - 生成式AI基建与算力投资趋势 (2023-2026)
python run_marco_research_report.py --marco_name "生成式AI基建与算力投资趋势" --time "2023-2026"
```

### Docker测试
```bash
# B榜Docker测试
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh company "4Paradigm" "06682.HK"
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh industry "中国智能服务机器人产业"
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh macro "国家级人工智能+政策效果评估" "2023-2025"

# A榜Docker测试
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh company "商汤科技" "00020.HK"
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh industry "智能风控&大数据征信服务"
docker run --rm -v ${PWD}\docker_outputs:/app/outputs research-agent:latest bash /app/run.sh macro "生成式AI基建与算力投资趋势" "2023-2026"
```

## 🔧 技术特性

**CSA合规保障**
- 严格遵循《发布证券研究报告暂行规定》
- 论点-论据链完整性验证
- 必要披露信息自动生成

**质量控制体系**  
- 8维度评估: 完整性、逻辑性、专业性、数据性、创新性、实用性、合规性、可读性
- 智能改进循环: 自动识别问题并优化
- 严格阈值: 总分≥8.5且CSA完全合规

**多模态支持**
- 中文字体自适应配置
- 专业金融图表生成
- Word文档格式化输出

## 📁 输出文件

每次运行生成：
- `{主题}_严格CSA合规研报_{时间戳}.docx` - Word格式报告
- `{主题}_严格CSA合规研报_{时间戳}.md` - Markdown格式  
- `{主题}_严格CSA合规评估_{时间戳}.yaml` - 质量评估报告
- `{主题}_图表_*.png` - 专业分析图表

## ⚠️ 注意事项

**API配置**
- 支持OpenAI、火山引擎等多种API
- 请妥善保管API密钥，不要提交到版本控制

**依赖模型说明**
- 本系统不包含大模型参数文件
- 请确保API服务可用且有足够配额
- 建议使用DeepSeek-V3或DeepSeek-R1等高性能模型

**合规声明**
- 生成报告仅供研究参考，不构成投资建议
- 严格遵循证券业协会相关规定
- 请根据实际需要进行人工审核

---