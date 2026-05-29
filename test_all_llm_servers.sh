#!/bin/bash
# Find all running tt-inference-server-* containers, extract each one's
# published host port, and run test_llm_server.sh against it.
#
# Usage:
#   ./test_all_llm_servers.sh                                       # default mode
#   ./test_all_llm_servers.sh --health                              # readiness only
#   ./test_all_llm_servers.sh --concurrent 4                        # 4 streams/server
#                                                                   # in parallel across
#                                                                   # all servers. Default
#                                                                   # prompt: "Tell me a quick story".
#   ./test_all_llm_servers.sh --concurrent 4 "Write a haiku"        # custom prompt
#   ./test_all_llm_servers.sh --concurrent 4 --max-tokens 64        # longer outputs
#   ./test_all_llm_servers.sh --concurrent 4 --max-tokens 64 "..."  # both
#   HOST=foo ./test_all_llm_servers.sh
#   API_KEY=xxx ./test_all_llm_servers.sh
#
# Notes:
# - Only containers named with the prefix "tt-inference-server-" are considered
#   (matches what run.py auto-generates: tt-inference-server-<short-uuid>).
# - Skips containers that don't respond healthy on /health (still warming up
#   or already crashed).
# - Skips containers whose /v1/models returns no entries (e.g. CNN servers).
# - --health mode reports per-port readiness without sending any tokens or
#   invoking test_llm_server.sh. /health returns 200 once the model is loaded
#   and serving; 405 (or other non-200) while still warming up.
# - --concurrent mode is fast (no model warmup wait, fails fast on non-ready
#   servers): used to watch multiple servers stream side-by-side. Output style
#   mirrors client_demo_concurrent.sh — one streaming line per (server,stream)
#   redrawn in-place, followed by TTFT / tok-per-sec per stream and total wall
#   time. Prompt defaults to "Tell me a quick story".

set -u

HEALTH_ONLY=0
CONCURRENT_N=0
PROMPT_TEXT=""
MAX_TOKENS=32

# Parse args. Recognized flags: --health | --concurrent N | --max-tokens N.
# A bare positional (no leading --) is taken as the prompt text and only makes
# sense in --concurrent mode.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --health)
      HEALTH_ONLY=1
      shift ;;
    --concurrent)
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --concurrent requires a positive integer (e.g. --concurrent 4)" >&2
        exit 2
      fi
      CONCURRENT_N="$1"
      shift ;;
    --max-tokens|-n)
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --max-tokens requires a positive integer (e.g. --max-tokens 64)" >&2
        exit 2
      fi
      MAX_TOKENS="$1"
      shift ;;
    --*)
      echo "ERROR: unknown flag: $1" >&2
      echo "Usage: $0 [--health | --concurrent N [--max-tokens N] [PROMPT]]" >&2
      exit 2 ;;
    *)
      # Bare arg: treat as prompt text (concurrent mode only).
      PROMPT_TEXT="$1"
      shift ;;
  esac
done

HOST="${HOST:-localhost}"
TEST_SCRIPT="$(dirname "$0")/test_llm_server.sh"

if [[ "$HEALTH_ONLY" -eq 0 && "$CONCURRENT_N" -eq 0 && ! -x "$TEST_SCRIPT" ]]; then
  echo "ERROR: $TEST_SCRIPT not found or not executable" >&2
  exit 1
fi

# List running tt-inference-server-* containers + their host ports.
# `docker ps --format` gives us "ID|NAME|PORTS" lines; we parse PORTS like:
#   "0.0.0.0:8010->8000/tcp"  → host port 8010
# Uses portable grep -oE (works on mawk/gawk/busybox alike).
entries=()
while IFS='|' read -r cid name ports; do
  [[ -z "$cid" ]] && continue
  # Extract first :NNNN-> match and strip the punctuation around it
  # tr -d ':>-' treats '-' as literal because it's the trailing char (avoids
  # tr interpreting ':->' as a character range).
  port=$(echo "$ports" | grep -oE ':[0-9]+->' | head -1 | tr -d ':>-')
  [[ -n "$port" ]] && entries+=("${cid}|${name}|${port}")
done < <(
  docker ps --filter "name=^tt-inference-server-" \
    --format '{{.ID}}|{{.Names}}|{{.Ports}}'
)

if [[ ${#entries[@]} -eq 0 ]]; then
  echo "No running tt-inference-server-* containers found."
  exit 0
fi

echo "Found ${#entries[@]} tt-inference-server container(s):"
printf '%s\n' "${entries[@]}" | awk -F'|' '{printf "  - %s (port %s) [%s]\n", $2, $3, $1}'
echo

if [[ "$CONCURRENT_N" -gt 0 ]]; then
  # Build "port|model_short_name" list for ready LLM servers only.
  servers=()
  for entry in "${entries[@]}"; do
    IFS='|' read -r cid name port <<< "$entry"
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 \
      -H "Authorization: Bearer ${API_KEY:-your-secret-key}" \
      "http://${HOST}:${port}/health" 2>/dev/null)
    [[ "$code" != "200" ]] && continue
    full=$(curl -s --max-time 3 \
      -H "Authorization: Bearer ${API_KEY:-your-secret-key}" \
      "http://${HOST}:${port}/v1/models" 2>/dev/null \
      | jq -r '.data[0].id // empty' 2>/dev/null)
    [[ -z "$full" ]] && continue
    # Short tag: drop the "org/" prefix if present (Qwen/Qwen3-4B -> Qwen3-4B).
    short="${full##*/}"
    servers+=("${port}|${full}|${short}")
  done

  if [[ ${#servers[@]} -eq 0 ]]; then
    echo "No ready LLM servers (none returned /health=200 with non-empty /v1/models)."
    exit 0
  fi

  : "${PROMPT_TEXT:=Tell me a quick story}"

  echo "Concurrent mode: $CONCURRENT_N streams per server × ${#servers[@]} server(s)"
  echo "Prompt: \"$PROMPT_TEXT\"  |  Max tokens: $MAX_TOKENS"
  for s in "${servers[@]}"; do
    IFS='|' read -r port full short <<< "$s"
    echo "  - $short (port $port)"
  done
  echo

  # Inline Python: dispatch (servers × N) parallel streaming threads, redraw
  # in-place, then print per-stream stats and total wall time. Mirrors the
  # tagging/formatting of ~/scripts/client_demo_concurrent.sh.
  export TT_SERVERS_JSON="$(
    python3 -c "
import json, os
servers = []
for s in '''${servers[*]}'''.split():
    port, full, short = s.split('|', 2)
    servers.append({'port': int(port), 'full': full, 'short': short})
print(json.dumps(servers))"
  )"

  HOST="$HOST" API_KEY="${API_KEY:-your-secret-key}" \
  CONCURRENT_N="$CONCURRENT_N" MAX_TOKENS="$MAX_TOKENS" PROMPT_TEXT="$PROMPT_TEXT" \
  python3 -u <<'PYEOF'
import json, os, sys, time, threading, shutil
import requests

host = os.environ['HOST']
api_key = os.environ['API_KEY']
n = int(os.environ['CONCURRENT_N'])
max_tokens = int(os.environ['MAX_TOKENS'])
prompt = os.environ['PROMPT_TEXT']
servers = json.loads(os.environ['TT_SERVERS_JSON'])

headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

# Build per-stream descriptors: (server_short, server_full, port, idx_in_server)
streams = []
for srv in servers:
    for i in range(n):
        streams.append({
            'short': srv['short'], 'full': srv['full'],
            'port': srv['port'], 'idx': i + 1, 'of': n,
        })
total = len(streams)

# tag width for alignment, e.g. "[Falcon3-7B-Instruct/3]" -> 22 chars
tag_width = max(len(f"[{s['short']}/{s['idx']}]") for s in streams)

lock = threading.Lock()
buffers = [''] * total
metrics = [None] * total

def cols():
    return shutil.get_terminal_size((120, 24)).columns

def redraw():
    """Move cursor up `total` lines and redraw each stream's current line."""
    sys.stdout.write(f'\033[{total}A')
    width = cols()
    for i, s in enumerate(streams):
        tag = f"[{s['short']}/{s['idx']}]".ljust(tag_width)
        max_text = width - tag_width - 1
        text = buffers[i].replace('\n', ' ').replace('\r', ' ')
        if max_text > 3 and len(text) > max_text:
            text = text[:max_text - 3] + '...'
        sys.stdout.write(f'\r\033[K{tag} {text}\n')
    sys.stdout.flush()

def run_stream(i, s):
    start = time.perf_counter()
    ttft = None
    token_count = 0
    try:
        resp = requests.post(
            f"http://{host}:{s['port']}/v1/chat/completions",
            headers=headers,
            json={
                'model': s['full'],
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'stream': True,
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
            data = line[len('data: '):]
            if data.strip() == '[DONE]':
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = chunk['choices'][0]
            text = choice.get('delta', {}).get('content', '')
            finish = choice.get('finish_reason')
            if text and ttft is None:
                ttft = time.perf_counter() - start
            if text:
                token_count += 1
                with lock:
                    buffers[i] += text
                    redraw()
            if finish:
                break
    except Exception as e:
        with lock:
            buffers[i] += f' [ERROR: {type(e).__name__}: {e}]'
            redraw()
        return
    elapsed = time.perf_counter() - start
    if ttft and token_count > 1:
        tps = (token_count - 1) / (elapsed - ttft)
        tps_str = f'{tps:.1f} tok/s'
    else:
        tps_str = 'n/a tok/s'
    ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
    metrics[i] = f'{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {tps_str}'

# Reserve `total` blank lines for the redraw region.
for s in streams:
    tag = f"[{s['short']}/{s['idx']}]".ljust(tag_width)
    print(f"{tag} ")
sys.stdout.flush()

threads = [threading.Thread(target=run_stream, args=(i, s)) for i, s in enumerate(streams)]
wall_start = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall_elapsed = time.perf_counter() - wall_start

print()
for i, s in enumerate(streams):
    tag = f"[{s['short']}/{s['idx']}]".ljust(tag_width)
    print(f'{tag} {metrics[i] if metrics[i] else "no metrics (error?)"}')
print(f'--- Wall time: {wall_elapsed:.2f}s ---')
PYEOF
  exit $?
fi

if [[ "$HEALTH_ONLY" -eq 1 ]]; then
  ready=0
  warming=0
  for entry in "${entries[@]}"; do
    IFS='|' read -r cid name port <<< "$entry"
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 \
      "http://${HOST}:${port}/health" 2>/dev/null)
    # /v1/models returns 200 even during warmup; use it to get the model id.
    model=$(curl -s --max-time 3 "http://${HOST}:${port}/v1/models" 2>/dev/null \
      | jq -r '.data[0].id // empty' 2>/dev/null)
    [[ -z "$model" ]] && model="?"
    if [[ "$code" == "200" ]]; then
      echo "[$name] port $port: READY ($model)"
      ready=$((ready+1))
    else
      echo "[$name] port $port: warming ($code) ($model)"
      warming=$((warming+1))
    fi
  done
  echo
  echo "Health summary: $ready ready, $warming warming (out of ${#entries[@]} container(s))"
  [[ $warming -eq 0 ]]
  exit
fi

pass=0
fail=0
skip=0

for entry in "${entries[@]}"; do
  IFS='|' read -r cid name port <<< "$entry"
  hr="============================================================"
  echo
  echo "$hr"
  echo "[$name] port=$port"
  echo "$hr"

  # Quick health probe before running the full test script.
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 \
    -H "Authorization: Bearer ${API_KEY:-your-secret-key}" \
    "http://${HOST}:${port}/health" 2>/dev/null)
  if [[ "$code" != "200" ]]; then
    echo "SKIP: /health returned HTTP $code (still warming up, crashed, or non-LLM)"
    skip=$((skip+1))
    continue
  fi

  # Skip CNN servers — they expose /v1/models but return data=[]
  models=$(curl -s --max-time 3 \
    -H "Authorization: Bearer ${API_KEY:-your-secret-key}" \
    "http://${HOST}:${port}/v1/models" 2>/dev/null)
  if [[ -z "$models" ]] || ! echo "$models" | jq -e '.data | length > 0' > /dev/null 2>&1; then
    echo "SKIP: /v1/models has no entries (likely a CNN server, not an LLM)"
    skip=$((skip+1))
    continue
  fi

  if HOST="$HOST" PORT="$port" "$TEST_SCRIPT"; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
done

echo
echo "$hr"
echo "Summary:  $pass passed, $fail failed, $skip skipped (out of ${#entries[@]} container(s))"
echo "$hr"

# Non-zero exit if anything failed; useful for CI-style invocations.
[[ $fail -eq 0 ]]
