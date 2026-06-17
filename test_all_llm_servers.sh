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
#   ./test_all_llm_servers.sh --concurrent 32 --isl 1024 --osl 128  # fixed input/output
#                                                                   # lengths: synthetic
#                                                                   # ~1024-token prompt
#                                                                   # (calibrated via
#                                                                   # /tokenize), exactly
#                                                                   # 128 output tokens
#                                                                   # (ignore_eos)
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
# - --isl N / --osl N (concurrent mode) set the input / output sequence lengths.
#   --isl builds a synthetic but readable prompt calibrated to ~N input tokens
#   via each server's /tokenize endpoint (per-model, since tokenizers differ).
#   --osl sets max_tokens to N (an upper bound) and requests ignore_eos: on
#   servers that honor it (e.g. stock vLLM) output is exactly N tokens; servers
#   that ignore it (some TT vLLM builds) may stop earlier at the natural EOS.
#   Either way the per-stream summary reports the ACTUAL ISL/OSL from the
#   response's usage field (ground truth, incl. chat-template tokens).

set -u

HEALTH_ONLY=0
CONCURRENT_N=0
PROMPT_TEXT=""
MAX_TOKENS=32
SEQ=0
PORTS_CSV=""
ISL=0          # target input  seq length (tokens); 0 = use PROMPT_TEXT as-is
OSL=0          # target output seq length (tokens); 0 = use MAX_TOKENS, EOS allowed

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
    --isl)
      # Concurrent mode only: build a synthetic prompt calibrated (via the
      # server's /tokenize endpoint) to ~N input tokens, so you can sweep
      # input lengths (1024, 2048, ...). Overrides any positional PROMPT.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --isl requires a positive integer (e.g. --isl 1024)" >&2
        exit 2
      fi
      ISL="$1"
      shift ;;
    --osl)
      # Concurrent mode only: output sequence length. Sets max_tokens to N AND
      # forces ignore_eos so the model generates EXACTLY N tokens (clean tok/s
      # at a fixed length). Takes precedence over --max-tokens.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --osl requires a positive integer (e.g. --osl 128)" >&2
        exit 2
      fi
      OSL="$1"
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
      echo "Usage: $0 [--health | --concurrent N [--seq] [--max-tokens N] [--isl N] [--osl N] [PROMPT]] [--ports P1,P2,...]" >&2
      exit 2 ;;
    *)
      # Bare arg: treat as prompt text (concurrent mode only).
      PROMPT_TEXT="$1"
      shift ;;
  esac
done

# --osl is the output sequence length: it sets max_tokens and (in the Python
# block) forces ignore_eos so generation runs to exactly OSL tokens. Wins over
# --max-tokens.
[[ "$OSL" -gt 0 ]] && MAX_TOKENS="$OSL"

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
  if [[ "$ISL" -gt 0 ]]; then
    echo "ISL: exactly $ISL input tokens (raw token IDs via /v1/completions; non-English, no calibration)  |  OSL: $MAX_TOKENS output-token cap (ignore_eos requested)"
  else
    echo "Prompt: \"$PROMPT_TEXT\"  |  Max tokens: $MAX_TOKENS$([[ "$OSL" -gt 0 ]] && echo ' (ignore_eos requested)')"
  fi
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
  ISL="$ISL" OSL="$OSL" \
  python3 -u <<'PYEOF'
import json, os, sys, time, threading, shutil
import requests

host = os.environ['HOST']
api_key = os.environ['API_KEY']
n = int(os.environ['CONCURRENT_N'])
max_tokens = int(os.environ['MAX_TOKENS'])
prompt = os.environ['PROMPT_TEXT']
seq = os.environ.get('SEQ', '0') == '1'
isl = int(os.environ.get('ISL', '0'))
osl = int(os.environ.get('OSL', '0'))
servers = json.loads(os.environ['TT_SERVERS_JSON'])

headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

# When --osl is given we force the model to emit exactly OSL tokens (ignore the
# EOS token), so tok/s is measured over a fixed output length.
force_len = osl > 0

# Token mode (--isl): instead of generating English text and round-tripping the
# slow /tokenize endpoint to calibrate its length, we send a raw list of token
# IDs straight to /v1/completions. The input length is then EXACTLY `isl`
# tokens with zero server calls and no chat-template overhead. Output is
# non-English gibberish (we don't care — this path is for throughput numbers).
token_mode = isl > 0
# An ordinary mid-vocab content token id, valid in every real model's vocab
# (vocabs are 32k-150k+); used only as raw /v1/completions *input*, so it never
# affects generation (output stops on max_tokens / EOS, not on input content).
SAFE_TOKEN_ID = 1000
token_prompt = [SAFE_TOKEN_ID] * isl if token_mode else None

if token_mode:
    print(f"Token mode: sending exactly {isl} input token(s) per stream as raw token IDs "
          f"via /v1/completions (no /tokenize calibration; output is non-English).")
    print()

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
    # Detect the REAL terminal/pane width via the tty ioctl. Deliberately does
    # NOT use shutil.get_terminal_size, which consults $COLUMNS first — that var
    # is frequently stale in tmux (it keeps the pre-split width), which makes us
    # draw wider than the pane and flood it with soft-wraps. Try each std fd and
    # then /dev/tty; fall back to a SAFE narrow 80 (under-fill, never overflow).
    for fd in (1, 2, 0):
        try:
            return os.get_terminal_size(fd).columns
        except OSError:
            pass
    try:
        with open('/dev/tty') as _tty:
            return os.get_terminal_size(_tty.fileno()).columns
    except OSError:
        return 80

def _char_cells(ch):
    """Display width of a single char: 2 for CJK/fullwidth/emoji (which render
    double-wide but count as one Python char), else 1."""
    o = ord(ch)
    if (0x1100 <= o <= 0x115F or 0x2329 <= o <= 0x232A or
            0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3 or
            0xF900 <= o <= 0xFAFF or 0xFE10 <= o <= 0xFE19 or
            0xFE30 <= o <= 0xFE6F or 0xFF00 <= o <= 0xFF60 or
            0xFFE0 <= o <= 0xFFE6 or 0x1F300 <= o <= 0x1FAFF or
            0x20000 <= o <= 0x3FFFD):
        return 2
    return 1

def fit_cells_tail(s, max_cells):
    """Keep the TAIL of `s` whose on-screen width is <= max_cells (counting
    wide glyphs as 2), so the newest tokens stay visible and older text scrolls
    off the left. Returns (text, used_cells, truncated)."""
    if max_cells <= 0:
        return '', 0, len(s) > 0
    out, used = [], 0
    # Walk from the end, prepending until we'd exceed the budget.
    for ch in reversed(s):
        cw = _char_cells(ch)
        if used + cw > max_cells:
            return ''.join(reversed(out)), used, True
        out.append(ch)
        used += cw
    return ''.join(reversed(out)), used, False

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
            # Budget cells for the text: pane width minus the tag, the space
            # after it, and a 1-cell right margin so we never touch the edge.
            # Show the TAIL (newest tokens) so the line scrolls left and stays
            # visibly alive; older text slides off behind a leading ellipsis.
            # Sized by DISPLAY width (CJK glyphs are 2 cells) so the line can
            # never wrap — wrapping would break the \033[A cursor-up redraw.
            max_cells = width - tag_width - 2
            text = buffers[i].replace('\n', ' ').replace('\r', ' ')
            if max_cells > 1:
                # Reserve 1 cell for the leading '…' when we've truncated.
                fitted, _, trunc = fit_cells_tail(text, max_cells - 1)
                text = ('…' + fitted) if trunc else fitted
            else:
                text = ''
            sys.stdout.write(f'\r\033[K{tag} {text}\n')
        sys.stdout.flush()

    def run_stream(i, s):
        start = time.perf_counter()
        ttft = None
        token_count = 0
        # Token mode sends exactly `isl` token IDs, so ISL is known a priori
        # (this build's /v1/completions stream omits the usage chunk). Chat mode
        # fills both in from the final usage chunk.
        prompt_toks = isl if token_mode else None   # actual ISL
        completion_toks = None   # actual OSL, from response usage
        # Token mode (--isl): raw token-ID prompt to /v1/completions (exact ISL,
        # no chat template). Otherwise: English text to /v1/chat/completions.
        if token_mode:
            endpoint = 'completions'
            body = {'model': s['full'], 'prompt': token_prompt,
                    'max_tokens': max_tokens, 'stream': True,
                    'stream_options': {'include_usage': True}}
        else:
            endpoint = 'chat/completions'
            body = {'model': s['full'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': max_tokens, 'stream': True,
                    'stream_options': {'include_usage': True}}
        if force_len:
            body['ignore_eos'] = True
        try:
            resp = requests.post(
                f"http://{host}:{s['port']}/v1/{endpoint}",
                headers=headers, json=body, stream=True, timeout=300)
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
                # Final usage chunk (include_usage) carries empty choices.
                usage = chunk.get('usage')
                if usage:
                    prompt_toks = usage.get('prompt_tokens', prompt_toks)
                    completion_toks = usage.get('completion_tokens', completion_toks)
                choices = chunk.get('choices') or []
                if not choices:
                    continue
                choice = choices[0]
                # /v1/completions streams 'text'; chat streams 'delta.content'.
                text = (choice.get('text', '') if token_mode
                        else choice.get('delta', {}).get('content', ''))
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
        out_toks = completion_toks if completion_toks is not None else token_count
        if ttft and out_toks > 1:
            tps_str = f'{(out_toks - 1) / (elapsed - ttft):.1f} tok/s'
        else:
            tps_str = 'n/a tok/s'
        ttft_str = f'TTFT: {ttft*1000:.0f}ms' if ttft else 'TTFT: n/a'
        # Show actual ISL/OSL from usage when available.
        io_str = ''
        if prompt_toks is not None:
            io_str = f'ISL {prompt_toks} | OSL {out_toks} | '
        metrics[i] = f'{io_str}{ttft_str} | {elapsed:.2f}s | {tps_str}'

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
