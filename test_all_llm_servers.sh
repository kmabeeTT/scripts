#!/bin/bash
# Discover running LLM servers and run test_llm_server.sh against each. Two
# discovery sources are combined (deduped by host port):
#   1. Docker: running tt-inference-server-* containers + published host ports
#      (parsed from `docker ps`). Skipped silently if docker isn't installed.
#   2. Local processes: manually spun-up uvicorn/vllm servers (e.g. a wheel
#      server launched by launch_*.sh) — the bound port is read straight from
#      the process command line, so containerless servers are found too.
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
#   ./test_all_llm_servers.sh --concurrent 1 --seq                  # one server at a
#                                                                   # time (no cross-server
#                                                                   # host-CPU contention)
#   ./test_all_llm_servers.sh --concurrent 1 --ports 8101,8102      # use explicit ports
#                                                                   # instead of docker
#                                                                   # discovery (models found
#                                                                   # via /v1/models)
#   ./test_all_llm_servers.sh --concurrent 4 --max-tokens 64 "..."  # both
#   HOST=foo ./test_all_llm_servers.sh
#   API_KEY=xxx ./test_all_llm_servers.sh
#
# Notes:
# - Docker discovery only considers containers named with the prefix
#   "tt-inference-server-" (matches what run.py auto-generates:
#   tt-inference-server-<short-uuid>). If docker isn't on PATH (e.g. running
#   inside a container), this source is skipped silently.
# - Local-process discovery scans for uvicorn/vllm processes and reads their
#   --port from the command line. Non-LLM ports are harmless: the /health +
#   /v1/models gate skips anything that isn't an OpenAI-compatible LLM.
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
SEQ=0
PORTS_CSV=""

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
    --seq)
      # Concurrent mode only: run servers ONE AT A TIME (each server's N
      # streams still run together), so cross-server host-CPU contention
      # (cpu_sampling runs on the host) doesn't depress each server's tok/s.
      SEQ=1
      shift ;;
    --ports)
      # Use explicit, comma-separated host ports instead of discovering
      # docker containers (e.g. --ports 8101,8102). Works for non-container
      # servers too; the model on each port is discovered via /v1/models.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "ERROR: --ports requires comma-separated port numbers (e.g. --ports 8101,8102)" >&2
        exit 2
      fi
      PORTS_CSV="$1"
      shift ;;
    --*)
      echo "ERROR: unknown flag: $1" >&2
      echo "Usage: $0 [--health | --concurrent N [--seq] [--max-tokens N] [PROMPT]] [--ports P1,P2,...]" >&2
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

# Build the list of (cid|name|port) entries to test from the two discovery
# sources below. `--ports` overrides both and uses the given ports verbatim.
entries=()
if [[ -n "$PORTS_CSV" ]]; then
  # Explicit ports: skip discovery entirely (also works for non-container
  # servers, e.g. a uvicorn wheel server). cid/name are synthesized; the model
  # name on each port is discovered from /v1/models downstream.
  IFS=',' read -ra _ports <<< "$PORTS_CSV"
  for p in "${_ports[@]}"; do
    entries+=("-|port-${p}|${p}")
  done
else
  # 1. Docker containers (only if docker is installed). `docker ps --format`
  #    gives "ID|NAME|PORTS" lines; we parse PORTS like:
  #      "0.0.0.0:8010->8000/tcp"  → host port 8010
  #    Uses portable grep -oE (works on mawk/gawk/busybox alike).
  if command -v docker >/dev/null 2>&1; then
    while IFS='|' read -r cid name ports; do
      [[ -z "$cid" ]] && continue
      # Extract first :NNNN-> match and strip the punctuation around it
      # tr -d ':>-' treats '-' as literal because it's the trailing char (avoids
      # tr interpreting ':->' as a character range).
      port=$(echo "$ports" | grep -oE ':[0-9]+->' | head -1 | tr -d ':>-')
      [[ -n "$port" ]] && entries+=("${cid}|${name}|${port}")
    done < <(
      docker ps --filter "name=^tt-inference-server-" \
        --format '{{.ID}}|{{.Names}}|{{.Ports}}' 2>/dev/null
    )
  fi

  # 2. Local uvicorn/vllm processes (manually spun-up, non-container servers).
  #    Read the bound port straight from each matching process command line
  #    (handles both "--port 8004" and "--port=8004"). cid is "-"; the label
  #    is just the server kind + port. Anything that isn't actually an LLM is
  #    filtered out later by the /health + /v1/models probe.
  while IFS= read -r cmd; do
    case "$cmd" in
      *uvicorn*|*vllm*|*main:app*|*openai.api_server*) ;;
      *) continue ;;
    esac
    lport=$(printf '%s\n' "$cmd" | grep -oE -- '--port[= ]+[0-9]+' | head -1 | grep -oE '[0-9]+$')
    [[ -z "$lport" ]] && continue
    if   [[ "$cmd" == *uvicorn* ]]; then lbl=uvicorn
    elif [[ "$cmd" == *vllm*    ]]; then lbl=vllm
    else                                 lbl=server; fi
    entries+=("-|${lbl}-${lport}|${lport}")
  done < <(ps -eo args= 2>/dev/null)
fi

# Sort entries by host port (numeric, field 3) so every section below — the
# "Found" list, the per-server list, and the concurrent live/summary lines —
# is ordered by port, and dedupe by port (a server seen by both docker and the
# process scan should appear once; first wins, so the docker row is kept).
if [[ ${#entries[@]} -gt 0 ]]; then
  mapfile -t entries < <(printf '%s\n' "${entries[@]}" | sort -t'|' -k3,3n | awk -F'|' '!seen[$3]++')
fi

if [[ ${#entries[@]} -eq 0 ]]; then
  echo "No servers found (no tt-inference-server-* containers, no local uvicorn/vllm LLM processes)."
  exit 0
fi

if [[ -n "$PORTS_CSV" ]]; then
  echo "Using ${#entries[@]} specified port(s):"
else
  echo "Found ${#entries[@]} server(s) (docker containers + local uvicorn/vllm):"
fi
# Container rows show name+port+id; local/specified rows ("-|...") show name+port.
printf '%s\n' "${entries[@]}" | awk -F'|' '{ if ($1=="-") printf "  - %s (port %s)\n", $2, $3; else printf "  - %s (port %s) [%s]\n", $2, $3, $1 }'
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
  CONCURRENT_N="$CONCURRENT_N" MAX_TOKENS="$MAX_TOKENS" PROMPT_TEXT="$PROMPT_TEXT" SEQ="$SEQ" \
  python3 -u <<'PYEOF'
import json, os, sys, time, threading, shutil
import requests

host = os.environ['HOST']
api_key = os.environ['API_KEY']
n = int(os.environ['CONCURRENT_N'])
max_tokens = int(os.environ['MAX_TOKENS'])
prompt = os.environ['PROMPT_TEXT']
seq = os.environ.get('SEQ', '0') == '1'
servers = json.loads(os.environ['TT_SERVERS_JSON'])

headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

# Port-prefixed tag so identical model names on different ports are
# distinguishable, e.g. "[8101] [Llama-3.2-3B-Instruct/1]".
def tag_of(s):
    return f"[{s['port']}] [{s['short']}/{s['idx']}]"

def make_streams(server_list):
    out = []
    for srv in server_list:
        for i in range(n):
            out.append({'short': srv['short'], 'full': srv['full'],
                        'port': srv['port'], 'idx': i + 1, 'of': n})
    return out

# Compute tag width across ALL streams so alignment is stable across waves.
tag_width = max(len(tag_of(s)) for s in make_streams(servers))

def cols():
    return shutil.get_terminal_size((120, 24)).columns

def run_wave(streams):
    """Run one wave of streams concurrently, redraw in place, print per-stream
    summary. Returns the wave's wall time. Buffers/metrics are wave-local so
    waves can run one after another (--seq)."""
    total = len(streams)
    buffers = [''] * total
    metrics = [None] * total
    lock = threading.Lock()

    def redraw():
        sys.stdout.write(f'\033[{total}A')
        width = cols()
        for i, s in enumerate(streams):
            tag = tag_of(s).ljust(tag_width)
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
                json={'model': s['full'],
                      'messages': [{'role': 'user', 'content': prompt}],
                      'max_tokens': max_tokens, 'stream': True},
                stream=True, timeout=300)
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
            tps_str = f'{(token_count - 1) / (elapsed - ttft):.1f} tok/s'
        else:
            tps_str = 'n/a tok/s'
        ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
        metrics[i] = f'{token_count} tokens | {ttft_str} | {elapsed:.2f}s | {tps_str}'

    # Reserve `total` blank lines for the redraw region.
    for s in streams:
        print(f"{tag_of(s).ljust(tag_width)} ")
    sys.stdout.flush()

    threads = [threading.Thread(target=run_stream, args=(i, s)) for i, s in enumerate(streams)]
    ws = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    we = time.perf_counter() - ws

    print()
    for i, s in enumerate(streams):
        print(f'{tag_of(s).ljust(tag_width)} {metrics[i] if metrics[i] else "no metrics (error?)"}')
    return we

if seq:
    # One server at a time (its N streams still run together) so cross-server
    # host-CPU contention (cpu_sampling) doesn't depress each server's tok/s.
    grand = 0.0
    for srv in servers:
        we = run_wave(make_streams([srv]))
        grand += we
        print(f'--- {srv["short"]} (port {srv["port"]}) wall: {we:.2f}s ---')
        print()
    print(f'=== sequential-across-servers total wall: {grand:.2f}s ===')
else:
    we = run_wave(make_streams(servers))
    print(f'--- Wall time: {we:.2f}s ---')
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
