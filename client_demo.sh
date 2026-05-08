#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

MAX_TOKENS=${1:-128}
PORT=${PORT:-8000}
SERVER=${SERVER:-http://localhost:$PORT}
API_KEY=${API_KEY:-your-secret-key}
# MODE: auto (probe), chat (force /v1/chat/completions), completions (force /v1/completions)
MODE=${MODE:-auto}

python3 -u -c "
import requests, time, json, sys

server = '$SERVER'
api_key = '$API_KEY'
max_tokens = $MAX_TOKENS
mode = '$MODE'
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

# Optional sampling overrides from env vars
sampling_overrides = {}
if '$TEMPERATURE' != '':
    sampling_overrides['temperature'] = float('$TEMPERATURE')
if '$REPETITION_PENALTY' != '':
    sampling_overrides['repetition_penalty'] = float('$REPETITION_PENALTY')

# Wait for server to be ready
print('Waiting for server...', end='', flush=True)
while True:
    try:
        resp = requests.get(f'{server}/tt-liveness', headers=headers, timeout=2)
        if resp.status_code == 200 and resp.json().get('model_ready'):
            break
    except (requests.ConnectionError, requests.Timeout, Exception):
        pass
    print('.', end='', flush=True)
    time.sleep(5)
print(f' ready!')

models_resp = requests.get(f'{server}/v1/models', headers=headers).json()
model = models_resp['data'][0]['id'] if models_resp.get('data') else 'unknown'

# Detect chat vs completions support. Base models lack a chat_template and 500
# on /v1/chat/completions; instruct/chat models work fine. Probe with a 1-token
# request — fails fast at template-load before real inference for base models.
if mode == 'auto':
    probe = requests.post(
        f'{server}/v1/chat/completions',
        headers=headers,
        json={'model': model, 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1, 'stream': False},
        timeout=60,
    )
    mode = 'chat' if probe.status_code == 200 else 'completions'

print(f'Model: {model} | Max tokens: {max_tokens} | Mode: {mode}')
if mode == 'completions':
    print('(base model — using /v1/completions; no chat roles)')
print()

conversation = []  # list[dict] for chat mode; running text for completions mode
completion_text = ''
print('(conversation history is maintained across turns, type \"new\" to reset)')
print()

while True:
    try:
        prompt = input('Prompt (q to quit): ')
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if prompt.strip().lower() == 'q':
        break
    if prompt.strip().lower() == 'new':
        conversation.clear()
        completion_text = ''
        print('-- conversation reset --')
        print()
        continue
    if not prompt.strip():
        continue

    if mode == 'chat':
        conversation.append({'role': 'user', 'content': prompt})
        url = f'{server}/v1/chat/completions'
        payload = {'model': model, 'messages': conversation, 'max_tokens': max_tokens, 'stream': True}
    else:
        completion_text += prompt
        url = f'{server}/v1/completions'
        payload = {'model': model, 'prompt': completion_text, 'max_tokens': max_tokens, 'stream': True}

    print('Response: ', end='', flush=True)

    start = time.perf_counter()
    ttft = None
    token_count = 0
    assistant_text = ''
    usage = {}

    resp = requests.post(url, headers=headers, json={**payload, **sampling_overrides}, stream=True)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):
            continue
        data = line[len('data: '):]   # str.removeprefix is 3.9+; prefix already verified above
        if data.strip() == '[DONE]':
            break
        chunk = json.loads(data)
        choice = chunk['choices'][0]
        # /v1/chat/completions streams 'delta.content'; /v1/completions streams 'text'
        text = choice.get('delta', {}).get('content', '') if mode == 'chat' else choice.get('text', '')
        finish = choice.get('finish_reason')
        if finish and finish == 'error':
            err_msg = text.strip() if text else 'Unknown error'
            print(f'\n\033[91m[SERVER ERROR: {err_msg}]\033[0m', flush=True)
            break
        if text and ttft is None:
            ttft = time.perf_counter() - start
        if text:
            print(text, end='', flush=True)
            assistant_text += text
            token_count += 1
        if finish:
            usage = chunk.get('usage', {})
            break

    if mode == 'chat':
        conversation.append({'role': 'assistant', 'content': assistant_text})
    else:
        completion_text += assistant_text

    elapsed = time.perf_counter() - start
    ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    print()
    print(f'[{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {token_count/elapsed:.1f} tok/s | in:{prompt_tokens} out:{completion_tokens}]')
    print()
"
