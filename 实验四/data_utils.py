# data_utils.py
import json
import os

# 你原有的函数保持不变
def load_local_pubmed_data(filepath="./data/Open-Patients.jsonl", max_articles=300):
    """仅加载本地PubMed数据"""
    try:
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            articles = []
            for i, line in enumerate(f):
                if i >= max_articles:
                    break
                if line.strip():
                    article = json.loads(line)
                    articles.append({
                        "title": article.get("title", ""),
                        "abstract": article.get("abstract", ""),
                        "source": "PubMed",
                        "publish_time": article.get("pubmed_id", "")[:4]
                    })
            return articles
    except Exception as e:
        print(f"数据加载失败: {e}")
        return []

# 新增这个函数（app.py需要）
def load_data(filepath):
    """从JSON文件加载数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载 {len(data)} 条数据从 {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ 数据文件不存在: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"❌ JSON解析错误: {filepath}")
        return []
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return []
