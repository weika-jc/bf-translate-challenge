from datasets import DatasetDict, load_dataset
from ..utils import InvalidDatasetError


_mtnt_dataset: DatasetDict | None = None


def load_full_dataset() -> DatasetDict:
    global _mtnt_dataset
    if _mtnt_dataset is None:
        _mtnt_dataset = load_dataset('sjelassi/mtnt', 'default', split='test', verification_mode='no_checks')
    return _mtnt_dataset


def read_all_pairs(dataset_dict: DatasetDict, src: str, tgt: str) -> list[tuple[str, str]]:
    """从dataset_dict中找出给定语言对的样本. mtnt双向都有数据"""
    pairs = []
    for line in dataset_dict:
        swap = False
        if line['language_pair'] == f'{src}-{tgt}':
            swap = False
        elif line['language_pair'] == f'{tgt}-{src}':
            swap = True
        else:
            continue

        src_line = line.get('source', '')
        src_line = src_line.strip()
        tgt_line = line.get('target', '')
        tgt_line = tgt_line.strip()
        if not src_line or not tgt_line:
            continue

        if swap:
            src_line, tgt_line = tgt_line, src_line
        pairs.append((src_line, tgt_line))
    return pairs


def load_mtnt_dataset(src: str, tgt: str) -> tuple[list[tuple[str, str]] | None, str]:
    '''根据参数语言顺序返回 (src_data, tgt_data) 的数组以及错误原因'''
    supported = ['en', 'ja', 'fr']
    if src not in supported or tgt not in supported or 'en' not in (src, tgt):
        return None, InvalidDatasetError

    try:
        dataset = load_full_dataset()
    except Exception as e:
        return None, str(e)

    return read_all_pairs(dataset, src, tgt), None
