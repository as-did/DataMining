# models.py - 支持HF Token的模型加载
# ======================================
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@st.cache_resource
def load_embedding_model(model_name):
    """加载嵌入模型"""
    st.write(f"正在加载嵌入模型: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("✅ 嵌入模型加载成功")
        return model
    except Exception as e:
        st.error(f"❌ 嵌入模型加载失败: {e}")
        return None


@st.cache_resource
def load_generation_model(model_name, hf_token=None):
    """加载生成模型，支持HF Token避免限流"""
    st.write(f"正在加载生成模型: {model_name}...")

    # 调试信息
    st.write(f"Token接收状态: {hf_token is not None}")
    if hf_token:
        st.write(f"Token前10位: {hf_token[:10]}...")

    try:
        # 使用token参数进行身份验证
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=hf_token
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        st.success("✅ 生成模型和分词器加载成功")
        return model, tokenizer

    except Exception as e:
        st.error(f"❌ 生成模型加载失败: {e}")

        # 智能错误诊断
        if "429" in str(e):
            st.error("🚨 429限流错误！Token未生效或无效")
            st.info(f"请检查Token: {hf_token[:10] if hf_token else 'None'}")
        elif "401" in str(e):
            st.error("🚨 401未授权！Token无效或权限不足")
        elif "quota" in str(e).lower():
            st.warning("⚠️ 可能是HuggingFace下载配额不足")

        return None, None
