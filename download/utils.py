from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch


embed_model = None
client = OpenAI(base_url="http://10.0.0.2:1234/v1", api_key='hellofromtheotherside~')


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


def main():
    src1 = [
        'カリガリ博士の小屋',
        '僕の婚約者だ',
        '故郷ハレシュテンバルでの出来事だ',
        'お祭りがひらかれた',
        'ある香具師がやってきた',
        '親友のアランだ',
        'ハレシュテンバル祭へ　皆様　ぜひお越しを！',
        'フランシス　お祭りへいこうぜ',
        'きょうは市の係官は機嫌が悪い',
        'カリガリ博士',
    ]
    src2 = [
        'THE CABINET OF DR. CALIGARI',
        'She is my bride...',
        'In the small town, where I was born...',
        '...a traveling fair had arrived.',
        'Him...',
        'My friend, Alan...',
        'Come to the HOLSTENWALL FAIR! WONDERS! MARVELS!',
        'Come, Francis-- let\'s go to the fair.',
        'The Town-Clerk is in a bad mood today.',
        'Dr. Caligari',
    ]

    for s1, s2 in zip(src1, src2):
        similarity = check_semantic_similarity(s1, s2)
        # 抽取的样本数量不多, 速度也比较快, 这一步可以门槛高一点, 但是担心太高会限制语义
        print(f'debug: embed相似度: {similarity}')
        if similarity < 0.7:
            continue

        if similarity < 0.8:
            # 模型二次检查, 不行就丢弃
            similarity = check_content_similarity(s1, s2)
            print(f'debug: model相似度: {similarity}')
            if similarity < 0.8:
                continue

        print(f'这一组可以: {s1} -> {s2}')

if __name__ == '__main__':
    main()
