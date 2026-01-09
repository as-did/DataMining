import streamlit as st
import torch
import time
import re
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY, QUERY_PREPROCESSING_MAX_TOKENS, \
    QUERY_PREPROCESSING_TEMPERATURE


def extract_medical_keywords(processed_query):
    """
    基于医学词典的语义关键词提取
    识别真正的医学术语，而非简单的字符串差集
    """
    if not processed_query:
        return []

    # 常见医学术语后缀模式
    medical_suffixes = r'(炎|症|病|瘤|癌|症|征|畸形|损伤|感染|障碍|功能不全|衰竭|出血|梗死|栓塞|痛|痒|肿|胀|晕|麻)'

    # 常见前缀模式
    medical_prefixes = r'(超|亚|急|慢|特|原|继|反|再)'

    words = processed_query.split()
    keywords = []

    for word in words:
        # 规则1：长度大于3且包含医学后缀
        if len(word) > 3 and re.search(medical_suffixes, word):
            keywords.append(word)
        # 规则2：包含常见前缀+医学后缀（如"超敏反应"）
        elif re.match(medical_prefixes + r'.*' + medical_suffixes, word):
            keywords.append(word)
        # 规则3：纯英文医学术语（如"DNA"、"RNA"）
        elif re.match(r'^[A-Z]{2,}$', word):
            keywords.append(word)
        # 规则4：常见医学操作词
        elif word in ['诊断', '治疗', '预防', '检查', '手术', '药物']:
            keywords.append(word)

    return keywords[:6]  # 最多返回6个关键词


def preprocess_query(user_input, gen_model, tokenizer):
    """
    增强版查询预处理：带强制信息保留和多层验证

    核心改进：
    1. Prompt明确禁止生成通用建议
    2. 强制保留原始关键信息
    3. 4层输出验证（关键词保留、长度、语义、黑名单）
    4. 模型失败时立即回退到可靠的规则处理
    """
    if not gen_model or not tokenizer:
        return rule_based_preprocess(user_input)  # 直接回退

    # ========== 关键改进1：强制保留原始信息的Prompt ==========
    prompt = f"""作为医学AI检索助手，请优化以下查询以提高检索准确性。

**核心要求（必须遵守）：**
1.  **必须保留**  所有原始关键信息（疾病、症状、药物、治疗方式等）
2. **必须转换**口语化为专业医学术语（如"鼻子堵"→"鼻塞"）
3. **可以补充**相关医学维度（诊断、病因、预防等）
4.  **禁止删除**  任何原始信息或生成通用建议
5. **必须输出**专业医学查询，不能是通用回答

**合格示例：**
原始："鼻子堵了，该吃什么药？"
优化："鼻塞 药物治疗" ✓（保留了鼻塞和用药）

**失败示例：**
原始："鼻子堵了，该吃什么药？"
优化："医生建议吃点什么" ❌（丢失了所有关键信息）

**失败示例：**
原始："鼻子堵了，该吃什么药？"
优化："鼻塞" ❌（丢失了"药物治疗"信息）

**转换规则：**
- 鼻子堵/鼻塞 → 鼻塞
- 吃什么药/用药 → 药物治疗
- 鼻炎/鼻窦炎 → 鼻炎

请优化以下查询（只输出优化结果，不解释）：
原始："user_input"
优化："鼻塞 药物治疗 鼻炎"  # 示例格式
---
原始："{user_input}"
优化：
""".replace('user_input', user_input)  # 确保变量正确插入

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        input_length = inputs['input_ids'].shape[1]

        # 生成参数：调整为更保守的输出
        generation_kwargs = {
            "max_new_tokens": QUERY_PREPROCESSING_MAX_TOKENS,
            "temperature": QUERY_PREPROCESSING_TEMPERATURE,  # 使用更低温度
            "top_p": 0.95,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = gen_model.generate(**inputs, **generation_kwargs)

        processed_query = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


        # 清理：只保留第一行，并移除可能的标签
        if '\n' in processed_query:
            processed_query = processed_query.split('\n')[0].strip()

        # 移除可能生成的标签
        processed_query = processed_query.replace('优化：', '').replace('结果：', '').strip()

        # ========== 关键改进3：4层输出质量验证 ==========

        # 验证1：必须包含原始语义关键词（使用模糊匹配）
        original_concepts = extract_concepts(user_input)
        processed_concepts = extract_concepts(processed_query)

        # 检查是否丢失了核心概念（如"鼻塞"对应"鼻子堵"）
        concept_loss = False
        for orig_concept in original_concepts:
            if not any(semantic_match(orig_concept, proc_concept) for proc_concept in processed_concepts):
                concept_loss = True
                break

        if concept_loss:
            print(f"⚠️ 预处理失败：丢失了原始概念。原始：{original_concepts}，处理后：{processed_concepts}")
            return rule_based_preprocess(user_input)

        # 验证2：长度不能太短（至少保留原查询的一半长度）
        if len(processed_query) < len(user_input) * 0.5:
            print(f"⚠️ 预处理失败：输出太短。原始：{len(user_input)}字符，处理后：{len(processed_query)}字符")
            return rule_based_preprocess(user_input)

        # 验证3：不能是通用短语（黑名单检查）
        generic_phrases = ['医生建议', '吃点什么', '怎么治疗', '怎么办', '看医生', '去医院', '治疗建议', '咨询医生']
        if any(phrase in processed_query for phrase in generic_phrases) and len(processed_query) < 20:
            print(f"⚠️ 预处理失败：生成了通用短语。输出：{processed_query}")
            return rule_based_preprocess(user_input)

        # 验证4：必须有医学术语
        if not has_medical_terms(processed_query):
            print(f"⚠️ 预处理失败：未识别到医学术语。输出：{processed_query}")
            return rule_based_preprocess(user_input)

        return processed_query

    except Exception as e:
        print(f"⚠️ 预处理异常：{e}，回退到规则处理")
        return rule_based_preprocess(user_input)


def rule_based_preprocess(user_input):
    # 医学术语映射表（覆盖常见症状和查询）
    term_mapping = {
        # 症状
        '鼻子堵': '鼻塞',
        '鼻塞': '鼻塞',
        '流鼻涕': '鼻溢',
        '流鼻涕': '鼻溢',
        '发烧': '发热',
        '发热': '发热',
        '拉肚子': '腹泻',
        '腹泻': '腹泻',
        '头疼': '头痛',
        '头痛': '头痛',
        '头晕': '眩晕',
        '眩晕': '眩晕',
        '咳嗽': '咳嗽',
        '出血': '出血',
        '出血了': '出血',
        '痒': '瘙痒',
        '瘙痒': '瘙痒',
        '肿': '肿胀',
        '肿胀': '肿胀',
        '痛': '疼痛',
        '疼痛': '疼痛',

        # 治疗查询
        '吃药': '药物治疗',
        '用药': '药物治疗',
        '吃什么药': '药物治疗',
        '该用什么': '治疗',
        '怎么治疗': '治疗',
        '怎么办': '治疗',
        '咋治': '治疗',
        '咋整': '治疗',
        '咋弄': '治疗',
        '如何治': '治疗',

        # 疾病
        '感冒': '上呼吸道感染',
        '鼻炎': '鼻炎',
        '鼻窦炎': '鼻窦炎',
        '过敏': '过敏反应',
        '肺炎': '肺炎',
        '胃炎': '胃炎',
        '肠炎': '肠炎',
    }

    # 提取原始关键词
    keywords = []
    # 先匹配最长词组
    for colloquial in sorted(term_mapping.keys(), key=len, reverse=True):
        if colloquial in user_input:
            keywords.append(term_mapping[colloquial])
            user_input = user_input.replace(colloquial, '')  # 避免重复匹配

    # 去重
    keywords = list(dict.fromkeys(keywords))

    # 如果有关键词，添加通用医学维度
    if keywords:
        if any(k in user_input for k in ['药', '治疗', '怎么办', '咋治']):
            keywords.extend(['诊断', '病因', '预防'])
        # 限制数量
        keywords = keywords[:5]

    result = " ".join(keywords)

    # 确保不为空
    if not result:
        # 最坏情况：返回原始输入+通用词
        result = user_input + " 治疗 诊断"

    print(f"✅ 规则预处理成功：{user_input} → {result}")
    return result


def extract_concepts(text):
    """提取文本中的核心概念（用于语义匹配）"""
    # 移除标点
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = text.split()
    # 保留名词性词汇（简化）
    return [w for w in words if len(w) > 1]


def semantic_match(concept1, concept2):
    """检查两个概念是否语义匹配（简化版）"""
    # 直接相等或包含关系
    if concept1 in concept2 or concept2 in concept1:
        return True

    # 同义词映射
    synonyms = {
        '鼻子堵': ['鼻塞', '堵', '堵了'],
        '鼻塞': ['鼻子堵', '堵'],
        '吃药': ['药物', '用药', '治疗', '吃什么药', '该用什么药'],
        '治疗': ['用药', '吃药', '治疗', '咋治', '怎么办'],
    }

    for key, values in synonyms.items():
        if concept1 in values and concept2 in [key] + values:
            return True
        if concept2 in values and concept1 in [key] + values:
            return True

    return False


def has_medical_terms(text):
    """检查文本是否包含医学术语"""
    medical_patterns = [
        r'\b\w*(?:炎|症|病|瘤|癌|征|畸形|损伤|感染|障碍|功能不全|衰竭|出血|梗死|栓塞|痛|痒|肿|胀|晕|麻)\b',
        r'\b(?:药物|治疗|诊断|病因|预防|并发症|手术|护理|康复|检查|疗法|方案)\w*\b',
    ]

    for pattern in medical_patterns:
        if re.search(pattern, text):
            return True

    return False


def generate_answer_stream(query, context_docs, gen_model, tokenizer):

    if not context_docs:
        yield "⚠️ 未找到相关文献来回答您的问题。"
        return

    if not gen_model or not tokenizer:
        yield "❌ 生成组件未加载。"
        return

    try:
        # 改进上下文构建
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):
            title = doc.get('title', '未知标题')
            content = doc.get('content', doc.get('abstract', ''))
            if content and len(content.strip()) > 50:
                content_preview = content[:1000] if len(content) > 1000 else content
                context_parts.append(f"文档{i + 1}《{title}》：\n{content_preview}")

        context = "\n\n---\n\n".join(context_parts)

        if not context or len(context.strip()) < 100:
            yield "⚠️ 检索到的文档内容过短，无法生成有效答案。请尝试更具体的问题。"
            return

        # 增强提示词：明确要求详细回答
        prompt = f"""基于以下医学文献，请详细回答用户问题。请提供完整、准确且易于理解的答案。

参考文献：
{context}

用户问题：{query}

请提供简洁的医学解答：
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        input_length = inputs['input_ids'].shape[1]

        gen_model.eval()
        if hasattr(gen_model, 'generation_config'):
            gen_model.generation_config.output_scores = False

        past_key_values = None
        current_tokens = inputs['input_ids']

        # 实时解码，避免累积导致的提前终止
        min_length = 50  # 最少生成50个token

        for step in range(MAX_NEW_TOKENS_GEN):
            with torch.no_grad():
                if past_key_values is None:
                    outputs = gen_model(current_tokens, use_cache=True)
                else:
                    outputs = gen_model(current_tokens[:, -1:], past_key_values=past_key_values, use_cache=True)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # 调整采样参数
                next_token_logits = logits / (TEMPERATURE * 0.6)
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

                # 只有在生成足够内容后才允许EOS
                if next_token.item() == tokenizer.eos_token_id and step > min_length:
                    break

                current_tokens = torch.cat([current_tokens, next_token], dim=-1)

                # 实时解码并输出
                new_text = tokenizer.decode(next_token[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # 转换为 UTF-8 编码以确保流输出中文时没有乱码
                if new_text and new_text.isprintable() and not new_text.isspace():
                    yield new_text

    except Exception as e:
        yield f"生成错误: {e}"
        yield "\n💡 建议：请检查模型状态或重新启动应用。"

