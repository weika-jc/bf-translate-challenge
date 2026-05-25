from datasets import DatasetDict, load_dataset


DATASET_NAME = 'Helsinki-NLP/tatoeba_mt'

# ISO 639-1 -> ISO 639-3, Tatoeba 配置名使用 3 字母代码
LANG_CODE_MAP = {
    'en': 'eng',
    'es': 'spa',
    'ja': 'jpn',
    'de': 'deu',
    'fr': 'fra',
    'nl': 'nld',
    'da': 'dan',
    'no': 'nor',
    'it': 'ita',
}

def extract_pair(example: dict) -> tuple[str, str]:
    src = example.get('sourceString') or ''
    tgt = example.get('targetString') or ''
    if not src and not tgt and isinstance(example.get('translation'), dict):
        values = list(example['translation'].values())
        if len(values) == 2:
            src, tgt = values[0], values[1]
    return src or '', tgt or ''


def read_all_pairs(dataset_dict: DatasetDict, swap: bool) -> list[tuple[str, str]]:
    """遍历所有 split 流式过滤并写出, 避免把整份语料读进内存."""
    pairs = []
    for split_name in dataset_dict:
        for example in dataset_dict[split_name]:
            first, second = extract_pair(example)
            src_clean = first.strip()
            tgt_clean = second.strip()
            if swap:
                src_clean, tgt_clean = tgt_clean, src_clean

            if not src_clean or not tgt_clean:
                continue

            pairs.append((src_clean, tgt_clean))
    return pairs


def load_tatoeba_dataset(src: str, tgt: str) -> tuple[list[tuple[str, str]] | None, str]:
    '''根据参数语言顺序返回 (src_data, tgt_data) 的数组以及错误原因'''
    src3 = LANG_CODE_MAP.get(src)
    tgt3 = LANG_CODE_MAP.get(tgt)
    if not src3 or not tgt3:
        return None, ''

    dataset_dict: DatasetDict | None = None
    dataset_swap = False
    last_error = None
    for first, second, swap in ((src3, tgt3, False), (tgt3, src3, True)):
        config = f'{first}-{second}'
        try:
            dataset_dict = load_dataset(DATASET_NAME, config, verification_mode='no_checks')
            dataset_swap = swap
            break
        except Exception as e:
            last_error = e
            continue

    if dataset_dict is None:
        return None, str(last_error) or 'dataset not found'

    return read_all_pairs(dataset_dict, dataset_swap), None
