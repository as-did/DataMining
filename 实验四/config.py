# config.py - 删除查询优化配置
# =====================================
# ========== 查询预处理配置 ==========
QUERY_PREPROCESSING_ENABLED = True  # 是否启用查询预处理
QUERY_PREPROCESSING_TEMPERATURE = 0.1  # 预处理温度（越低越稳定）
QUERY_PREPROCESSING_MAX_TOKENS = 128  # 预处理后最大长度

# ========== ChromaDB配置 ==========
CHROMA_DATA_PATH = "./chroma_data"
COLLECTION_NAME = "medical_rag_chroma"
EMBEDDING_DIM = 384

# ========== 数据配置 ==========
DATA_FILE = "./data/processed_data.json"
PUBMED_RAW_FILE = "./data/Open-Patients.jsonl"
PUBMED_DOWNLOAD_URL = "https://huggingface.co/datasets/ncbi/pubmed/resolve/main/pubmed_test.jsonl"

# ========== 模型配置 ==========
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# ========== 索引和搜索参数 ==========
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 3

# ========== 生成参数 ==========
MAX_NEW_TOKENS_GEN = 150
TEMPERATURE = 0.3
TOP_P = 0.8
REPETITION_PENALTY = 1.1

# ========== 全局文档映射 ==========
id_to_doc_map = {}

# 删除查询优化相关配置
