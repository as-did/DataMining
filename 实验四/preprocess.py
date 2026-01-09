import os
import json
import re


def split_text(text, chunk_size=500, chunk_overlap=50):
    """将文本分割成指定大小的块，并带有重叠。"""
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text):
            break
        if start < chunk_size and len(chunks) > 1 and chunks[-1] == chunks[-2][chunk_size - chunk_overlap:]:
            chunks.pop()
            start = len(text)

    if start < len(text) and start > 0:
        last_chunk = text[start - chunk_size + chunk_overlap:]
        if chunks and last_chunk != chunks[-1]:
            if not chunks[-1].endswith(last_chunk):
                chunks.append(last_chunk)
        elif not chunks:
            chunks.append(last_chunk)

    return [c.strip() for c in chunks if c.strip()]


def load_local_jsonl_data(filepath, max_articles=300):
    """直接加载本地JSONL文件"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return []

    if os.path.getsize(filepath) == 0:
        print(f"⚠️ 文件为空: {filepath}")
        return []

    print(f"📄 正在加载: {filepath}")
    articles = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_articles:
                break
            line = line.strip()
            if not line:
                continue

            try:
                article = json.loads(line)
                articles.append({
                    "title": article.get("title", ""),
                    "abstract": article.get("abstract", ""),
                    "source": "PubMed",
                    "publish_time": article.get("pubmed_id", "")[:4] if article.get("pubmed_id") else ""
                })
            except json.JSONDecodeError:
                print(f"⚠️ 第 {i + 1} 行格式错误，跳过: {line[:50]}...")
                continue

    print(f"✅ 成功加载 {len(articles)} 篇文章")
    return articles


def main():
    # --- 配置 ---
    txt_directory = './data/'
    jsonl_filepath = './data/Open-Patients.jsonl'  # 你的文件名
    output_json_path = './data/processed_data.json'
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    print(f"开始处理目录 '{txt_directory}' 中的文件...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 处理本地TXT文件
    all_data = []
    file_count = 0
    chunk_count = 0

    txt_files = [f for f in os.listdir(txt_directory) if f.endswith('.txt')]
    print(f"找到 {len(txt_files)} 个 TXT 文件。")

    for filename in txt_files:
        filepath = os.path.join(txt_directory, filename)
        print(f"  处理文件: {filename} ...")
        file_count += 1

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                main_text = f.read().strip()

            title = os.path.splitext(filename)[0]

            if main_text:
                chunks = split_text(main_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                print(f"    分割成 {len(chunks)} 个块。")

                for i, chunk in enumerate(chunks):
                    chunk_count += 1
                    entry = {
                        "id": f"{filename}_{i}",
                        "title": title,
                        "abstract": chunk,
                        "source_file": filename,
                        "chunk_index": i
                    }
                    all_data.append(entry)
            else:
                print(f"    警告：文件 {filename} 内容为空。")

        except Exception as e:
            print(f"    处理文件 {filename} 时出错: {e}")

    # --- 加载JSONL数据 ---
    print("\n加载JSONL数据...")
    pubmed_articles = load_local_jsonl_data(jsonl_filepath)

    if pubmed_articles:
        all_data.extend(pubmed_articles)

    # --- 保存为 JSON ---
    total_count = len(all_data)
    print(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个文本块，"
          f"加载 {len(pubmed_articles)} 篇PubMed文章，总计 {total_count} 条数据。")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 结果已保存到: {output_json_path}")
    except Exception as e:
        print(f"❌ 错误：无法写入 JSON 文件 {output_json_path}: {e}")


if __name__ == "__main__":
    main()
