import asyncio
import csv
import os
import sys
from .utils import rate, translate

input_dir = 'output'
output_dir = 'result'
limit = 10
model_name = 'claude-haiku-4-6'
model_id = 'arn:aws:bedrock:us-west-2:686465264859:prompt/7ZJL56AKIU'  # test model
# model_name = 'gpt-oss-120b'
# model_id = 'arn:aws:bedrock:us-west-2:686465264859:prompt/1FWKA83D84'  # alter model

records = []
record_lock = asyncio.Lock()
CSV_FIELDS = ['dataset', 'src', 'tgt', 'raw', 'ref', 'trans', 'score', 'input_tokens', 'output_tokens', 'total_tokens', 'latency_ms']

async def evaluate_file(dataset: str, data_path: str, ref_path: str, src: str, tgt: str):
    src_lines = open(data_path, 'r', encoding='utf-8').readlines()
    tgt_lines = open(ref_path, 'r', encoding='utf-8').readlines()
    processed = 0
    for raw, ref in zip(src_lines, tgt_lines):
        raw = raw.strip()
        ref = ref.strip()
        if not raw or not ref:
            continue
        result = await translate(model_id, raw, tgt)
        trans = result['text'] if result else None
        score = await rate(raw, trans, tgt, ref)
        record = {
            'dataset': dataset,
            'src': src,
            'tgt': tgt,
            'raw': raw,
            'ref': ref,
            'trans': trans,
            'score': score,
            'input_tokens': result.get('input_tokens') if result else None,
            'output_tokens': result.get('output_tokens') if result else None,
            'total_tokens': result.get('total_tokens') if result else None,
            'latency_ms': result.get('latency_ms') if result else None,
        }
        async with record_lock:
            records.append(record)
        processed += 1
        if processed >= limit:
            break


async def evaluate_source(dataset: str, source_dir: str):
    files = os.listdir(source_dir)
    for file in files:
        if not file.endswith('.txt'):
            continue
        src, tgt = file.split('.')[:2]
        src_path = os.path.join(source_dir, file)
        tgt_path = os.path.join(source_dir, f'{tgt}.{src}.txt')
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            continue
        await evaluate_file(dataset, src_path, tgt_path, src, tgt)


async def main():
    if not os.path.exists(input_dir):
        print(f'input directory {input_dir} does not exist')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    datasets = os.listdir(input_dir)
    tasks = []
    for dataset in datasets:
        source_dir = os.path.join(input_dir, dataset)
        if not os.path.isdir(source_dir):
            continue
        tasks.append(asyncio.create_task(evaluate_source(dataset, source_dir)))

    await asyncio.gather(*tasks)

    csv_path = os.path.join(output_dir, f'{model_name}.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    asyncio.run(main())
