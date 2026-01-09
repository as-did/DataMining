# verify_import_chain.py
try:
    from config import CHROMA_DATA_PATH

    print(f"✅ 步骤1: config.py导入成功 -> {CHROMA_DATA_PATH}")

    from chromadb_utils import get_chroma_client

    print("✅ 步骤2: chromadb_utils导入成功")

    # 测试运行时访问
    client = get_chroma_client()
    print("✅ 步骤3: 函数调用成功")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()
