# -*- coding: utf-8 -*-
"""
Data Analysis Agent Package

一个基于LLM的智能数据分析代理，专门为Jupyter Notebook环境设计。
"""

from .config.llm_config import LLMConfig
from .utils.code_executor import CodeExecutor
from .data_analysis_agent import DataAnalysisAgent

__version__ = "1.0.0"
__author__ = "Data Analysis Agent Team"

# 主要导出类
__all__ = [
    "DataAnalysisAgent",
    "LLMConfig", 
    "CodeExecutor",
]

# 便捷函数
def create_agent(llm_config=None, output_dir="outputs", max_rounds=30,absolute_path=False):
    """
    创建一个数据分析智能体实例
    
    Args:
        config: LLM配置，如果为None则使用默认配置
        output_dir: 输出目录
        max_rounds: 最大分析轮数
        session_dir: 指定会话目录（可选，此参数暂不支持）
        
    Returns:
        DataAnalysisAgent: 智能体实例
    """
    if llm_config is None:
        llm_config = LLMConfig()
    return DataAnalysisAgent(llm_config=llm_config, output_dir=output_dir, max_rounds=max_rounds,absolute_path=absolute_path)

def quick_analysis(query,files=None, llm_config=None, output_dir="outputs", max_rounds=10,absolute_path=False):
    """
    快速数据分析函数
    
    Args:
        query: 分析需求（自然语言）
        files: 数据文件路径列表
        output_dir: 输出目录
        max_rounds: 最大分析轮数
        
    Returns:
        dict: 分析结果
    """
    agent = create_agent(llm_config=llm_config, output_dir=output_dir, max_rounds=max_rounds, absolute_path=absolute_path)
    return agent.analyze(query, files)
