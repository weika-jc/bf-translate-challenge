import itertools
import os
import random
import time

from .dataset import fetch_handlers
from .utils import InvalidDatasetError, check_semantic_similarity, load_embed_model


def check_similarity(src: str, tgt: str) -> bool:
    '''判断两个句子内容上是否一致'''
    return check_semantic_similarity(src, tgt) >= 0.8


def pick_pairs(pairs: list[tuple[str, str]], count: int, need_filter: bool) -> list[tuple[str, str]]:
    indexes = set(range(len(pairs)))
    result = []
    while len(result) < count and len(indexes) > 0:
        index = random.choice(tuple(indexes))
        indexes.remove(index)
        src, tgt = pairs[index]
        if need_filter and check_similarity(src, tgt) < 0.8:
            continue
        result.append((src, tgt))
    return result


def fetch_all(dst: str, count = 1000):
    '''加载所有源数据, 抽取满足条件的样本'''
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)

    embed_model = load_embed_model(use_mps=False)
    if embed_model is None:
        print('failed to load embed model')
        return

    for name, handler in fetch_handlers.items():
        lang_pairs = list(itertools.combinations(handler['languages'], 2))
        for src, tgt in lang_pairs:
            print(f'fetching {name}@{src}-{tgt}... ', end='')
            dst_path = os.path.join(dst, name)
            src_path = os.path.join(dst_path, f'{src}.{tgt}.txt')
            tgt_path = os.path.join(dst_path, f'{tgt}.{src}.txt')
            if os.path.exists(src_path) and os.path.exists(tgt_path):
                print('already exists, skipped')
                continue

            start_time = time.time()
            pairs, reason = handler['handler'](src, tgt)
            if pairs is None:
                if reason == InvalidDatasetError:
                    print(f'{name} not supported, skipped')
                else:
                    print(f'{name} failed: {reason}')
                continue
            if len(pairs) == 0:
                print(f'{name} failed: no valid samples')
                continue

            print(f'{len(pairs)} pairs fetched, filtering... ', end='')
            pairs = pick_pairs(pairs, count, handler.get('need_filter', True))
            if len(pairs) == 0:
                print(f'{name} failed: no valid samples')
                continue

            print(f'{len(pairs)} pairs filtered, writing to files... ', end='')
            os.makedirs(dst_path, exist_ok=True)
            with (
                open(src_path, 'w', encoding='utf-8') as fsrc,
                open(tgt_path, 'w', encoding='utf-8') as ftgt,
            ):
                for src_line, tgt_line in pairs:
                    fsrc.write(src_line + '\n')
                    ftgt.write(tgt_line + '\n')

            print(f'done in {time.time() - start_time:.2f}s')
