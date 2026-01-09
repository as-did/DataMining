# app.py - 修复版：确认按钮可触发检索
# ======================================
import streamlit as st
import time
import os
import shutil
import re
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ 未找到HF_TOKEN！请检查.env文件")
    st.stop()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_TOKEN'] = HF_TOKEN

from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, COLLECTION_NAME, CHROMA_DATA_PATH, id_to_doc_map,
    QUERY_PREPROCESSING_ENABLED, QUERY_PREPROCESSING_MAX_TOKENS, QUERY_PREPROCESSING_TEMPERATURE
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from chromadb_utils import get_chroma_client, setup_chroma_collection, index_data_if_needed, search_similar_documents
from rag_core import generate_answer_stream, preprocess_query, extract_medical_keywords

# ========== CSS样式 ==========
st.markdown("""
<style>
:root { --primary-color: #2563eb; --secondary-color: #10b981; --accent-color: #f59e0b; }
.medical-title { background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 2.5rem; }
.doc-card { background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-left: 4px solid var(--primary-color); padding: 1rem; margin-bottom: 0.75rem; border-radius: 8px; }
.streaming-cursor { color: var(--primary-color); animation: blink 1s infinite; }
@keyframes blink { 0%,50% { opacity: 1; } 51%,100% { opacity: 0; } }
.keyword-tag { background: #e6f7ff; color: #1890ff; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.9rem; margin: 0.25rem; display: inline-block; }
.original-query { background: #fffbe6; border-left: 4px solid #faad14; padding: 1rem; margin: 1rem 0; border-radius: 6px; }
.processed-query { background: #f6ffed; border-left: 4px solid #52c41a; padding: 1rem; margin: 1rem 0; border-radius: 6px; }
.confirmation-box { border: 2px dashed #d9d9d9; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ========== 系统初始化 ==========
st.markdown('<h1 class="medical-title">📄 医疗RAG智能助手</h1>', unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """初始化所有核心组件"""
    client = get_chroma_client()
    if not client or not setup_chroma_collection(client):
        return None, None, None, None
    embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    gen_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME, hf_token=HF_TOKEN)
    return client, embed_model, gen_model, tokenizer


chroma_client, embedding_model, generation_model, tokenizer = initialize_system()

if not chroma_client or not embedding_model or not generation_model or not tokenizer:
    st.error("❌ 系统初始化失败")
    st.stop()


# ========== 数据加载与索引 ==========
@st.cache_data(show_spinner=False)
def load_and_index_data():
    """加载数据"""
    pubmed_data = load_data(DATA_FILE)
    if not pubmed_data:
        return [{"title": "示例", "abstract": "示例内容", "content": "示例"}]
    return pubmed_data


pubmed_data = load_and_index_data()

# 检查是否需要重建索引
reindex_marker = os.path.join(os.path.dirname(CHROMA_DATA_PATH), "NEED_REINDEX")
if os.path.exists(reindex_marker):
    st.warning("🔔 检测到重建索引标记，正在重建...")
    if os.path.exists(CHROMA_DATA_PATH):
        shutil.rmtree(CHROMA_DATA_PATH, ignore_errors=True)
    os.remove(reindex_marker)
    st.rerun()

# 执行索引
with st.status("📚 正在加载知识库...", expanded=False):
    indexing_successful = index_data_if_needed(chroma_client, pubmed_data, embedding_model)
    if indexing_successful:
        st.success(f"✅ 知识库加载完成！共索引 {len(pubmed_data)} 篇文档")

# ========== 主交互界面 ==========
st.markdown("---")

# 初始化session_state
if 'query_state' not in st.session_state:
    st.session_state.query_state = {
        'original': "",
        'processed': "",
        'keywords': [],
        'is_processed': False,
        'confirmed_query': "",
        'is_confirmed': False
    }

if indexing_successful:
    # 第一步：用户输入
    st.markdown("### 📝 第一步：输入医学问题")
    query = st.text_area(
        "请详细描述您的医学问题（支持口语化表达）：",
        value=st.session_state.query_state['original'],
        placeholder="例如：我感冒了，流鼻涕，鼻子不通气，应该吃什么药？",
        height=100
    )
    st.session_state.query_state['original'] = query

    col1, col2 = st.columns([1, 3])
    with col1:
        preprocess_disabled = query.strip() == "" or st.session_state.query_state['is_processed']
        if st.button("🤖 分析并优化问题", disabled=preprocess_disabled, use_container_width=True):
            st.session_state.query_state['is_processed'] = True
            with st.status("🔍 正在分析问题...", expanded=True):
                processed = preprocess_query(query, generation_model, tokenizer)
                keywords = extract_medical_keywords(processed)
                st.session_state.query_state.update({
                    'processed': processed,
                    'keywords': keywords,
                    'confirmed_query': processed,
                    'is_confirmed': False
                })
            st.rerun()

    # 第二步：展示预处理结果
    if st.session_state.query_state['is_processed']:
        st.markdown("### 🔬 第二步：问题分析与优化结果")
        with st.expander("📊 点击查看详细分析", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始问题：**")
                st.markdown(f'<div class="original-query">{st.session_state.query_state["original"]}</div>',
                            unsafe_allow_html=True)
                st.markdown("**识别到的医学关键词：**")
                if st.session_state.query_state['keywords']:
                    for kw in st.session_state.query_state['keywords']:
                        st.markdown(f'<span class="keyword-tag">{kw}</span>', unsafe_allow_html=True)
                else:
                    st.caption("未检测到明显的医学关键词差异")
            with col2:
                st.markdown("**优化后问题：**")
                st.markdown(f'<div class="processed-query">{st.session_state.query_state["processed"]}</div>',
                            unsafe_allow_html=True)

        # 第三步：用户确认和编辑
        st.markdown("### ✏️ 第三步：确认或修改优化结果")
        final_query = st.text_area(
            "您可以在此修改优化后的查询（或直接确认使用）：",
            value=st.session_state.query_state['confirmed_query'],
            height=80
        )
        st.session_state.query_state['confirmed_query'] = final_query

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ 确认并检索", type="primary", disabled=not st.session_state.query_state['is_processed'],
                         use_container_width=True):
                st.session_state.query_state['is_confirmed'] = True
                st.rerun()

    # 第四步：检索和生成（当确认后）
    if st.session_state.query_state['is_confirmed']:
        final_query = st.session_state.query_state['confirmed_query']
        start_time = time.time()

        with st.status("🔍 正在检索相关文献...", expanded=True):
            retrieved_ids, distances = search_similar_documents(chroma_client, final_query, embedding_model)
            if retrieved_ids:
                st.write(f"✅ 找到 {len(retrieved_ids)} 篇相关文档")
            else:
                st.warning("⚠️ 未找到相关文献")

        if retrieved_ids:
            retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]
            if retrieved_docs:
                st.markdown("### 📚 参考医学证据")
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(
                        f'<div class="doc-card"><strong>📄 文档 {i + 1}:</strong> {doc["title"]}<br><small>相关性: {1 - distances[i]:.2%}</small></div>',
                        unsafe_allow_html=True
                    )
                st.markdown("---")

                st.markdown("### 💡 智能答案")
                answer_container = st.empty()
                try:
                    full_answer = ""
                    for token in generate_answer_stream(final_query, retrieved_docs, generation_model, tokenizer):
                        if token:
                            full_answer += token
                            answer_container.markdown(full_answer + '<span class="streaming-cursor">▌</span>',
                                                      unsafe_allow_html=True)
                    answer_container.markdown(full_answer)
                except Exception as e:
                    st.error(f"❌ 生成错误: {e}")

        end_time = time.time()
        st.success(f"✅ 回答生成完成！总耗时: {end_time - start_time:.2f} 秒")

        # 第五步：重新开始
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 新问题", use_container_width=True):
                st.session_state.query_state = {
                    'original': "", 'processed': "", 'keywords': [],
                    'is_processed': False, 'confirmed_query': "", 'is_confirmed': False
                }
                st.rerun()

# ========== 侧边栏配置 ==========
st.sidebar.header("⚙️ 系统配置")
st.sidebar.markdown(f"**向量存储:** ChromaDB")
st.sidebar.markdown(f"**数据路径:** `{os.path.abspath(CHROMA_DATA_PATH)}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.success("✅ Token已配置")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")

preprocess_enabled = st.sidebar.toggle("启用查询预处理", value=True)
if preprocess_enabled:
    st.sidebar.info("已启用：问题分析 → 关键词识别 → 专业改写")

if st.sidebar.button("清空历史"):
    st.session_state.clear()
    st.rerun()
