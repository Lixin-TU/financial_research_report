import requests
from bs4 import BeautifulSoup


def get_sensetime_company_info():
    """获取商汤科技公司介绍信息"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get("https://www.sensetime.com/cn/about-index#1", headers=headers)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(strip=True)
    except:
        return ""


if __name__ == "__main__":
    text = get_sensetime_company_info()
    print(text)
