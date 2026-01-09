# check_config.py
print("=" * 50)
print("验证 config.py 内容")
print("=" * 50)

try:
    import config

    # 检查所有必要变量
    required_vars = [
        "CHROMA_DATA_PATH",
        "COLLECTION_NAME",
        "EMBEDDING_MODEL_NAME",
        "GENERATION_MODEL_NAME",
        "TOP_K",
        "MAX_ARTICLES_TO_INDEX",
        "id_to_doc_map"
    ]

    for var in required_vars:
        if hasattr(config, var):
            value = getattr(config, var)
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未定义")

except Exception as e:
    print(f"❌ 导入config.py失败: {e}")

print("=" * 50)
