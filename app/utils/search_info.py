
"""
搜索信息模块
支持 DuckDuckGo 和搜狗搜索
"""

from search_engine import create_search_engine

def search_company_industry_info(company_name: str, engine: str = "sogou", max_results: int = 10):
    """
    搜索公司行业信息
    
    Args:
        company_name: 公司名称
        engine: 搜索引擎 ('ddg' 或 'sogou')
        max_results: 最大结果数
        
    Returns:
        搜索结果列表
    """
    search_engine = create_search_engine(engine)
    keywords = f"{company_name}的行业地位"
    return search_engine.search(keywords, max_results)

# 示例用法
if __name__ == "__main__":
    # 使用 DuckDuckGo 搜索
    print("=== DuckDuckGo 搜索结果 ===")
    ddg_results = search_company_industry_info("百度", engine="ddg", max_results=5)
    for i, result in enumerate(ddg_results, 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['description'][:100]}...")
    
    # 尝试使用搜狗搜索
    print("\n=== 搜狗搜索结果 ===")
    try:
        sogou_results = search_company_industry_info("百度", engine="sogou", max_results=5)
        for i, result in enumerate(sogou_results, 1):
            print(f"{i}. {result['title']}")
            print(f"   {result['description'][:100]}...")
    except Exception as e:
        print(f"搜狗搜索不可用: {e}")