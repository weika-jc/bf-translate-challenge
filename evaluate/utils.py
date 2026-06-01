import asyncio
import time
import boto3
from botocore.config import Config
from .prompt import PROMPT_REF, PROMPT_NOREF

session = boto3.Session(profile_name='aigc')
bedrock_runtime = session.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    config=Config(
        retries={'max_attempts': 3, 'mode': 'standard'},
        connect_timeout=10,
        read_timeout=30,
    ),
)
sonnet_model_id = 'arn:aws:bedrock:us-west-2:686465264859:inference-profile/us.anthropic.claude-sonnet-4-6'


async def _converse(**kwargs):
    return await asyncio.to_thread(bedrock_runtime.converse, **kwargs)


async def translate(model_id: str, txt: str, tgt: str) -> dict | None:
    print(f'[debug] translate {txt} -> {tgt}')
    message = f'"{txt}" -> {tgt}'
    start = time.perf_counter()
    response = await _converse(modelId=model_id, promptVariables={ 'input': { 'text': message } })
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    text = None
    for content in response['output']['message']['content']:
        if 'text' in content:
            text = content['text']
            break
    if text is None:
        return None

    usage = response.get('usage', {})
    return {
        'text': text,
        'input_tokens': usage.get('inputTokens'),
        'output_tokens': usage.get('outputTokens'),
        'total_tokens': usage.get('totalTokens'),
        'latency_ms': latency_ms,
    }


async def invoke_generic_model(model_id: str, txt: str) -> str:
    messages = [{ 'role': 'user', 'content': [{ 'text': txt }] }]
    response = await _converse(modelId=model_id, messages=messages)
    return response['output']['message']['content'][0]['text']


async def rate(raw: str, trans: str, tgt: str, ref: str | None = None) -> int | None:
    prompt = PROMPT_REF if ref is not None else PROMPT_NOREF
    message = prompt.format(raw=raw, tgt=tgt, trans=trans, ref=ref)
    for _ in range(3):
        try:
            score = await invoke_generic_model(sonnet_model_id, message)
            return int(score)
        except Exception as e:
            print(f'[debug] rate failed: {e}')
            continue
