# data_downloader.py - 自动下载并融合PubMed数据
import requests
import json
from config import PUBMED_DOWNLOAD_URL

def download_pubmed_data(max_articles=300):
    """从Hugging Face下载PubMed数据"""
    try:
        response = requests.get(PUBMED_DOWNLOAD_URL, stream=True)
        articles = []
        for i, line in enumerate(response.iter_lines()):
            if i >= max_articles:
                break
            if line:
                article = json.loads(line)
                articles.append({
                    "title": article.get("title", ""),
                    "abstract": article.get("abstract", ""),
                    "source": "PubMed",
                    "publish_time": article.get("pubmed_id", "")[:4]  # 提取年份
                })
        return articles
    except Exception as e:
        print(f"PubMed数据下载失败: {e}")
        return []
