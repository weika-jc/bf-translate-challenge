import tempfile
import urllib.error
import urllib.request
import os
import random
import shutil
from datasets import DownloadManager

DATASET_NAME = 'open_subtitles'
OPUS_MOSES_URL = 'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/{src}-{tgt}.txt.zip'
OPUS_FILE_TEMPLATE = 'OpenSubtitles.{pair_folder}.{lang}'


def opus_url_exists(url: str) -> bool:
    request = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False


def download_files(url: str, src: str, tgt: str) -> tuple[str, str, str] | None:
    dm = DownloadManager(dataset_name=DATASET_NAME, data_dir=tempfile.gettempdir())
    try:
        extracted_dir = dm.download_and_extract(url)
    except Exception as e:
        print(f'下载失败: {e}')
        return None

    src_path = os.path.join(extracted_dir, OPUS_FILE_TEMPLATE.format(pair_folder=f'{src}-{tgt}', lang=src))
    tgt_path = os.path.join(extracted_dir, OPUS_FILE_TEMPLATE.format(pair_folder=f'{src}-{tgt}', lang=tgt))
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        print('下载失败, 语料文件不存在')
        shutil.rmtree(extracted_dir, ignore_errors=True)
        return None

    return src_path, tgt_path, extracted_dir


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for count, _ in enumerate(f):
            pass
    return count + 1


def read_all_pairs(src_path: str, tgt_path: str) -> list[tuple[str, str]]:
    pairs = []
    # 数据过大, 随机丢弃一部分
    discard_score = None
    line_count = count_lines(src_path)
    if line_count > 3_000_000:
        discard_score = 1_000_000 / line_count

    with open(src_path, 'r', encoding='utf-8') as fsrc, open(tgt_path, 'r', encoding='utf-8') as ftgt:
        for src_line, tgt_line in zip(fsrc, ftgt):
            if discard_score is not None and random.random() > discard_score:
                continue
            src_clean = src_line.strip()
            tgt_clean = tgt_line.strip()
            if not src_clean or not tgt_clean:
                continue
            pairs.append((src_clean, tgt_clean))
    return pairs


def load_subtitle_dataset(src: str, tgt: str) -> tuple[list[tuple[str, str]] | None, str]:
    '''根据参数语言顺序返回 (src_data, tgt_data) 的数组以及错误原因'''
    last_error = None
    for first, second, swap in ((src, tgt, False), (tgt, src, True)):
        url = OPUS_MOSES_URL.format(src=first, tgt=second)
        if not opus_url_exists(url):
            continue

        extracted_dir = None
        try:
            result = download_files(url, first, second)
            if result is None:
                continue
            src_path, tgt_path, extracted_dir = result
            if src_path is None or tgt_path is None:
                continue
            if swap:
                src_path, tgt_path = tgt_path, src_path
            return read_all_pairs(src_path, tgt_path), None
        except Exception as e:
            last_error = e
            continue
        finally:
            if extracted_dir is not None:
                shutil.rmtree(extracted_dir, ignore_errors=True)
    if last_error is None:
        last_error = 'dataset not found'
    return None, str(last_error)
