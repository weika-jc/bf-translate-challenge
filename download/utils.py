from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch


embed_model = None
client = OpenAI(base_url="http://10.0.0.2:1234/v1", api_key='hellofromtheotherside~')

# 约定不支持的数据返回该错误信息, 采样时忽略
InvalidDatasetError = 'InvalidDatasetError'


def load_embed_model(use_mps=True):
    global embed_model
    device = 'cpu'
    if use_mps and torch.backends.mps.is_available():
        device = 'mps'
        print(f'using mps for acceleration')
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    return embed_model


def check_semantic_similarity(text1: str, text2: str) -> float:
    # 编码为向量
    emb1 = embed_model.encode(text1, convert_to_tensor=True)
    emb2 = embed_model.encode(text2, convert_to_tensor=True)
    # 计算余弦相似度
    similarity = util.cos_sim(emb1, emb2).item()
    return similarity


def check_content_similarity(text1: str, text2: str) -> float:
    completion = client.chat.completions.create(
        model="google/gemma-4-e2b",
        messages=[{ "role": "user", "content": f"判断下这两个句子在内容上的相似度，并给出相似度评分, 仅输出一个0~100的整数。\n句子1：{text1}\n句子2：{text2}" }],
        temperature=0.01,
        reasoning_effort="low",
    )
    try:
        return int(completion.choices[0].message.content) / 100
    except Exception:
        return 0
