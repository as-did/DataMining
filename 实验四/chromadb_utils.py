# chromadb_utils.py - 完整ChromaDB实现
# ======================================
import streamlit as st
import chromadb
from chromadb.config import Settings
import time
import os

# 导入配置
from config import (
    CHROMA_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map
)


@st.cache_resource
def get_chroma_client():
    """
    初始化ChromaDB客户端（持久化模式）
    - 首次调用：创建数据库目录
    - 后续调用：复用现有连接
    """
    try:
        st.write(f"Initializing ChromaDB with data path: {CHROMA_DATA_PATH}")

        # 确保数据目录存在
        os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

        # 初始化持久化客户端
        client = chromadb.PersistentClient(
            path=CHROMA_DATA_PATH,
            settings=Settings(
                anonymized_telemetry=False,  # 关闭匿名统计
                allow_reset=True  # 允许重置collection
            )
        )

        st.success("ChromaDB client initialized!")
        return client

    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None


@st.cache_resource
def setup_chroma_collection(_client):
    """
    创建或获取Collection
    - 自动检测collection是否存在
    - 不存在时创建并配置HNSW索引
    """
    if not _client:
        st.error("Chroma client not available.")
        return False

    try:
        collection_name = COLLECTION_NAME

        # 尝试获取现有collection
        try:
            collection = _client.get_collection(name=collection_name)
            st.write(f"Found existing collection: '{collection_name}'.")
        except Exception:
            # 创建新collection（自动使用HNSW索引）
            st.write(f"Collection '{collection_name}' not found. Creating...")
            collection = _client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",  # 使用余弦相似度
                    "hnsw:construction_ef": 100,  # 索引构建参数
                    "hnsw:M": 16
                },
                get_or_create=True
            )
            st.success(f"Collection '{collection_name}' created with HNSW index.")

        # 获取文档数量
        count = collection.count()
        st.write(f"Collection '{collection_name}' ready. Current entity count: {count}")
        return True

    except Exception as e:
        st.error(f"Error setting up Chroma collection: {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """
    检查并索引数据到ChromaDB
    - 自动跳过已索引的文档（基于数量对比）
    - 批量生成嵌入向量
    - 更新全局id_to_doc_map
    """
    global id_to_doc_map  # 修改全局映射

    if not client:
        st.error("Chroma client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    collection = client.get_collection(name=collection_name)

    # 获取当前文档数
    current_count = collection.count()
    st.write(f"Entities currently in Chroma collection: {current_count}")

    # 限制数据量
    data_to_index = data[:MAX_ARTICLES_TO_INDEX]

    # 准备数据
    texts_to_encode = []
    metadatas = []
    ids = []
    temp_id_map = {}

    with st.spinner("Preparing data for indexing..."):
        for i, doc in enumerate(data_to_index):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            content = f"Title: {title}\nAbstract: {abstract}".strip()
            if not content:
                continue

            doc_id = str(i)  # ChromaDB要求字符串ID
            texts_to_encode.append(content)
            metadatas.append({
                "title": title,
                "source": doc.get('source', ''),
                "publish_time": doc.get('publish_time', '')
            })
            ids.append(doc_id)

            # 更新临时映射
            temp_id_map[int(doc_id)] = {
                'title': title,
                'abstract': abstract,
                'content': content
            }

    if not texts_to_encode:
        st.error("No valid text content found in the data to index.")
        return False

    # 判断是否需要重新索引
    if current_count < len(texts_to_encode):
        st.warning(f"Indexing required ({current_count}/{len(texts_to_encode)} documents).")

        # 生成嵌入向量
        st.write(f"Embedding {len(texts_to_encode)} documents...")
        start_embed = time.time()
        embeddings = embedding_model.encode(
            texts_to_encode,
            show_progress_bar=True,
            normalize_embeddings=True  # 重要：归一化用于余弦相似度
        )
        end_embed = time.time()
        st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

        # 批量插入到ChromaDB
        st.write("Inserting data into ChromaDB...")
        start_insert = time.time()
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts_to_encode,
            metadatas=metadatas,
            ids=ids
        )
        end_insert = time.time()
        st.success(
            f"Successfully indexed {len(texts_to_encode)} documents. Insert took {end_insert - start_insert:.2f} seconds.")

        # 更新全局映射（仅在成功后）
        id_to_doc_map.update(temp_id_map)
        return True
    else:
        st.write("Data indexing is complete.")
        # 如果全局映射为空，填充它
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True


def search_similar_documents(client, query, embedding_model):
    """
    在ChromaDB中进行向量搜索
    - 返回ID列表和距离列表（与Milvus接口兼容）
    - 距离已转换为余弦相似度分数
    """
    if not client or not embedding_model:
        st.error("Chroma client or embedding model not available for search.")
        return [], []

    collection_name = COLLECTION_NAME
    collection = client.get_collection(name=collection_name)

    # 生成查询向量
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    # 执行搜索
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"]
        )
    except Exception as e:
        st.error(f"Error during ChromaDB search: {e}")
        return [], []

    # 处理结果（转换为Milvus兼容格式）
    if not results or not results['ids'][0]:
        return [], []

    # ChromaDB返回的IDs是字符串，需转回int
    retrieved_ids = [int(doc_id) for doc_id in results['ids'][0]]

    # 距离值已经是余弦相似度（1-相似度），需要转换
    # Milvus期望的是相似度分数（越高越好）
    if 'distances' in results:
        distances = [1.0 - d for d in results['distances'][0]]
    else:
        distances = []

    return retrieved_ids, distances
