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
#                                                                   # lengths: exact
#                                                                   # 1024-token raw
#                                                                   # token-ID prompt,
#                                                                   # exactly 128 output
#                                                                   # tokens (ignore_eos)
#   ./test_all_llm_servers.sh --concurrent 1,4 --isl 128 \
#       --osl 128,256,512,1024                                      # SWEEP: cross product
#                                                                   # of concurrency × ISL
#                                                                   # × OSL (here 8 points);
#                                                                   # prints a comparison
#                                                                   # table at the end
#   ./test_all_llm_servers.sh --concurrent 1 --osl 1024 \
#       --rep-penalty 1.0                                           # send repetition_penalty
#                                                                   # explicitly (1.0 disables
#                                                                   # it; unset => server
#                                                                   # default, e.g. 1.1)
#   ./test_all_llm_servers.sh --concurrent 1 --osl 1024 --temperature 0  # send temperature
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
CONCURRENT_CSV=""   # comma list of concurrency levels, e.g. "1,4"; empty = not concurrent mode
PROMPT_TEXT=""
MAX_TOKENS=32
SEQ=0
PORTS_CSV=""
ISL_CSV="0"        # comma list of input  seq lengths (tokens); 0 = use PROMPT_TEXT as-is
OSL_CSV="0"        # comma list of output seq lengths (tokens); 0 = use MAX_TOKENS, EOS allowed
REP_PENALTY_FLAG="${REP_PENALTY:-}"   # repetition_penalty to send; empty = let server default apply
TEMPERATURE_FLAG="${TEMPERATURE:-}"   # temperature to send; empty = let server default apply
SEED_FLAG="${SEED:-}"                 # seed to send; empty = no seed (unseeded fast path)
REQ_TIMEOUT_FLAG="${REQ_TIMEOUT:-1800}"  # per-request read timeout (s); large-ISL cold compiles exceed the old 300s
RAMP_FLAG="${RAMP:-0}"                    # ms to stagger concurrent stream launches (0 = all-at-once burst)

# Parse args. Recognized flags: --health | --concurrent N | --max-tokens N.
# A bare positional (no leading --) is taken as the prompt text and only makes
# sense in --concurrent mode.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --health)
      HEALTH_ONLY=1
      shift ;;
    --concurrent)
      # One or more concurrency levels, comma-separated (e.g. 4 or 1,4,8). With
      # multiple values (and/or multiple --isl/--osl) the script sweeps the
      # cross product and prints a comparison table at the end.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "ERROR: --concurrent requires positive integer(s), comma-separated (e.g. --concurrent 4 or 1,4,8)" >&2
        exit 2
      fi
      CONCURRENT_CSV="$1"
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
      # Concurrent mode only: input sequence length(s), comma-separated to sweep
      # (e.g. 128 or 128,512,1024). Each value sends a raw token-ID prompt of
      # EXACTLY that many tokens via /v1/completions (no calibration). Overrides
      # any positional PROMPT.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "ERROR: --isl requires positive integer(s), comma-separated (e.g. --isl 1024 or 128,512,1024)" >&2
        exit 2
      fi
      ISL_CSV="$1"
      shift ;;
    --osl)
      # Concurrent mode only: output sequence length(s), comma-separated to
      # sweep (e.g. 128 or 128,256,512). Each value sets max_tokens AND forces
      # ignore_eos so the model generates EXACTLY that many tokens (clean tok/s
      # at a fixed length). Takes precedence over --max-tokens.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "ERROR: --osl requires positive integer(s), comma-separated (e.g. --osl 128 or 128,256,512)" >&2
        exit 2
      fi
      OSL_CSV="$1"
      shift ;;
    --seq)
      # Concurrent mode only: run servers ONE AT A TIME (each server's N
      # streams still run together), so cross-server host-CPU contention
      # (cpu_sampling runs on the host) doesn't depress each server's tok/s.
      SEQ=1
      shift ;;
    --rep-penalty)
      # repetition_penalty to send on every request (float, e.g. 1.0 or 1.1).
      # Unset => server default applies (tt-media-server defaults 1.1, which
      # triggers O(N^2) decode; see issue #4278). Use 1.0 to disable it.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "ERROR: --rep-penalty requires a non-negative number (e.g. --rep-penalty 1.0)" >&2
        exit 2
      fi
      REP_PENALTY_FLAG="$1"
      shift ;;
    --temperature|--temp)
      # temperature to send on every request (float, e.g. 0 or 0.6). Unset =>
      # server default applies.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "ERROR: --temperature requires a non-negative number (e.g. --temperature 0)" >&2
        exit 2
      fi
      TEMPERATURE_FLAG="$1"
      shift ;;
    --seed)
      # seed to send on every request (non-negative integer). Unset => no seed
      # is sent (the fast unseeded path). NOTE: on the TT device sampler a seed
      # does NOT make output deterministic (tt-xla#4539) yet still forces the
      # slow seeded sampling path (~5x decode); use this only to reproduce/A-B
      # that cost. See SCOPE_seeded_qsamples_perstep_cost.md.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --seed requires a non-negative integer (e.g. --seed 42)" >&2
        exit 2
      fi
      SEED_FLAG="$1"
      shift ;;
    --timeout)
      # per-request read timeout in seconds (default 1800). Raise for large-ISL
      # runs whose first request triggers a cold prefill compile (the old fixed
      # 300s killed conc-1 ISL>=16k before TTFT). Lower it to fail fast.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --timeout requires a positive integer (seconds, e.g. --timeout 1800)" >&2
        exit 2
      fi
      REQ_TIMEOUT_FLAG="$1"
      shift ;;
    --ramp)
      # Stagger concurrent stream launches by N ms instead of firing all at t=0.
      # Avoids the synchronized prefill stampede (chunked-prefill backlog) so the
      # decode/TTFT numbers reflect steady-state serving rather than a thundering
      # herd. 0 (default) = burst. Has no effect at --concurrent 1.
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --ramp requires a non-negative integer (ms, e.g. --ramp 250)" >&2
        exit 2
      fi
      RAMP_FLAG="$1"
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
      echo "Usage: $0 [--health | --concurrent N[,N...] [--seq] [--max-tokens N] [--isl N[,N...]] [--osl N[,N...]] [--rep-penalty F] [--temperature F] [--seed N] [PROMPT]] [--ports P1,P2,...]" >&2
      exit 2 ;;
    *)
      # Bare arg: treat as prompt text (concurrent mode only).
      PROMPT_TEXT="$1"
      shift ;;
  esac
done

# --osl (output sequence length) sets max_tokens per value and forces ignore_eos
# in the Python block; the per-OSL value wins over --max-tokens. The mapping is
# done per combo inside Python now that OSL can be a list.

HOST="${HOST:-localhost}"
TEST_SCRIPT="$(dirname "$0")/test_llm_server.sh"

if [[ "$HEALTH_ONLY" -eq 0 && -z "$CONCURRENT_CSV" && ! -x "$TEST_SCRIPT" ]]; then
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

if [[ -n "$CONCURRENT_CSV" ]]; then
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

  # Count datapoints in the cross product for the banner (concurrency × ISL × OSL).
  _n_conc=$(awk -F, '{print NF}' <<< "$CONCURRENT_CSV")
  _n_isl=$(awk -F, '{print NF}' <<< "$ISL_CSV")
  _n_osl=$(awk -F, '{print NF}' <<< "$OSL_CSV")
  _n_pts=$(( _n_conc * _n_isl * _n_osl ))
  if [[ "$_n_pts" -gt 1 ]]; then
    echo "Concurrent sweep: concurrency=[$CONCURRENT_CSV] × ISL=[$ISL_CSV] × OSL=[$OSL_CSV] = ${_n_pts} datapoints/server"
  else
    _isl_desc=$([[ "$ISL_CSV" == "0" ]] && echo "prompt=\"$PROMPT_TEXT\"" || echo "ISL $ISL_CSV (token IDs)")
    _osl_desc=$([[ "$OSL_CSV" == "0" ]] && echo "max_tokens $MAX_TOKENS" || echo "OSL $OSL_CSV (ignore_eos)")
    echo "Concurrent mode: $CONCURRENT_CSV stream(s)/server  |  $_isl_desc  |  $_osl_desc"
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
  CONCURRENT_CSV="$CONCURRENT_CSV" MAX_TOKENS="$MAX_TOKENS" PROMPT_TEXT="$PROMPT_TEXT" SEQ="$SEQ" \
  ISL_CSV="$ISL_CSV" OSL_CSV="$OSL_CSV" \
  REP_PENALTY="$REP_PENALTY_FLAG" TEMPERATURE="$TEMPERATURE_FLAG" SEED="$SEED_FLAG" \
  REQ_TIMEOUT="$REQ_TIMEOUT_FLAG" RAMP="$RAMP_FLAG" \
  python3 -u <<'PYEOF'
import json, os, re, sys, time, threading, shutil, random, statistics
import requests

host = os.environ['HOST']
api_key = os.environ['API_KEY']
prompt = os.environ['PROMPT_TEXT']
seq = os.environ.get('SEQ', '0') == '1'
default_max_tokens = int(os.environ['MAX_TOKENS'])
servers = json.loads(os.environ['TT_SERVERS_JSON'])
# --rep-penalty / --temperature (or REP_PENALTY / TEMPERATURE env): when set, send
# the value explicitly on every request. Empty -> server default applies
# (tt-media-server defaults repetition_penalty=1.1, which triggers O(N^2) decode;
# see issue #4278). Lets you A/B e.g. 1.1 vs 1.0.
rep_penalty = os.environ.get('REP_PENALTY') or None
temperature = os.environ.get('TEMPERATURE') or None
# --seed (or SEED env): when set, send seed on every request. Empty -> no seed.
# WARNING: on the TT device sampler a seed is NOT honored (tt-xla#4539) but still
# triggers the slow seeded sampling path (~5x decode); for A/B only.
seed = os.environ.get('SEED') or None
# Per-request read timeout (s). The old fixed 300s killed conc-1 large-ISL runs
# whose first request pays a cold prefill compile (TTFT > 300s) before any token.
req_timeout = float(os.environ.get('REQ_TIMEOUT') or 1800)
ramp_s = float(os.environ.get('RAMP') or 0) / 1000.0  # stagger between stream launches
# Strip ALL control/escape chars from streamed text before the in-place redraw:
# generated garbage (esp. at large random-token ISL) can contain ESC / C0 / C1
# bytes that corrupt the cursor-up math and flood the screen.
_CTRL = re.compile(r'[\x00-\x1f\x7f-\x9f]')

headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

def _ints(csv, default):
    csv = (csv or '').strip()
    if not csv:
        return [default]
    return [int(x) for x in csv.split(',') if x.strip() != '']

# Each flag is a comma list; we sweep the cross product (one "datapoint" each).
concurrency_list = _ints(os.environ.get('CONCURRENT_CSV', ''), 1)
isl_list = _ints(os.environ.get('ISL_CSV', '0'), 0)   # 0 = English chat prompt
osl_list = _ints(os.environ.get('OSL_CSV', '0'), 0)   # 0 = default max_tokens, EOS allowed
combos = [(c, i, o) for c in concurrency_list for i in isl_list for o in osl_list]

# Raw input token IDs are drawn from a conservative band that is valid (and
# non-special) in every real model's vocab — the smallest LLM vocabs are 32k,
# so ids in [1000, 21000) are always in range. Used only as /v1/completions
# *input*, so the exact ids never affect generation (output stops on
# max_tokens / EOS, not on input content).
TOK_LO, TOK_BAND = 1000, 20000
# A per-invocation salt so repeat runs cold-prefill instead of being served from
# the server's PREFIX CACHE (identical prompts cache; an all-constant or
# sequential prompt also makes a shorter ISL a prefix of a longer one). Seeding
# per (run, isl) makes every ISL in a sweep a distinct sequence AND makes each
# invocation different — so TTFT/prefill is measured cold, not from cache.
RUN_SALT = random.Random().randrange(1 << 30)

def _token_prompt(isl, uniq=""):
    # Seed per (run, isl, uniq) so EVERY request gets a distinct sequence — across
    # ISLs, across concurrency levels, AND across streams within a wave. Without
    # the uniq key, conc 1/16/32 at the same ISL shared one prompt, so the conc=1
    # datapoint (first under --seq) warmed the PREFIX CACHE and the conc=16/32
    # datapoints read warm KV -> bogus (too-low) TTFT.
    rng = random.Random(f"{RUN_SALT}:{isl}:{uniq}")
    return [TOK_LO + rng.randrange(TOK_BAND) for _ in range(isl)]

def make_cfg(n, isl, osl):
    """Resolve one datapoint into a runnable config. Token mode (--isl) sends a
    raw token-ID prompt of exactly `isl` tokens via /v1/completions (no slow
    /tokenize calibration; output is non-English). --osl forces ignore_eos for a
    fixed output length and wins over --max-tokens."""
    token_mode = isl > 0
    return {
        'n': n, 'isl': isl, 'osl': osl,
        'max_tokens': osl if osl > 0 else default_max_tokens,
        'token_mode': token_mode,
        'force_len': osl > 0,
        # Prompts are built per-stream in run_wave (unique per concurrency+idx),
        # not shared here — otherwise prefix caching fakes TTFT across datapoints.
    }

# Port-prefixed tag so identical model names on different ports are
# distinguishable, e.g. "[8101] [Llama-3.2-3B-Instruct/1]".
def tag_of(s):
    return f"[{s['port']}] [{s['short']}/{s['idx']}]"

def make_streams(server_list, n):
    out = []
    for srv in server_list:
        for i in range(n):
            out.append({'short': srv['short'], 'full': srv['full'],
                        'port': srv['port'], 'idx': i + 1, 'of': n})
    return out

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

def run_wave(streams, cfg):
    """Run one wave of streams concurrently, redraw in place, print per-stream
    summary. Returns (wall_time, [per-stream sample dicts that succeeded]).
    Buffers/samples are wave-local so waves can run one after another."""
    total = len(streams)
    buffers = [''] * total
    samples = [None] * total
    # One UNIQUE prompt per stream (keyed by concurrency + stream idx) so prefix
    # caching can't serve warmed KV across streams or earlier datapoints (which
    # faked TTFT). Pre-built before launch so dispatch isn't staggered by gen.
    prompts = ([_token_prompt(cfg['isl'], (cfg['n'], s['idx'])) for s in streams]
               if cfg['token_mode'] else [None] * total)
    # Tag width is wave-local: concurrency (and thus the /idx suffix width) can
    # differ between datapoints, so alignment is computed per wave.
    tag_width = max((len(tag_of(s)) for s in streams), default=1)
    lock = threading.Lock()

    _last_draw = [0.0]

    def redraw(force=False):
        # Coalesce repaints: at high concurrency every token from every stream
        # would otherwise trigger a full N-line repaint (thousands/sec) -> flood
        # and flicker. Cap to ~20 Hz; callers force a final repaint after join.
        now = time.perf_counter()
        if not force and now - _last_draw[0] < 0.05:
            return
        _last_draw[0] = now
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
            text = _CTRL.sub(' ', buffers[i])
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
        itls = []          # inter-token gaps (s) -> TPOT / steady-state decode rate
        prev_tok = None
        base = {'port': s['port'], 'short': s['short'], 'idx': s['idx'], 'ok': False}
        # Token mode sends exactly `isl` token IDs, so ISL is known a priori
        # (this build's /v1/completions stream omits the usage chunk). Chat mode
        # fills both in from the final usage chunk.
        prompt_toks = cfg['isl'] if cfg['token_mode'] else None   # actual ISL
        completion_toks = None   # actual OSL, from response usage
        # Token mode (--isl): raw token-ID prompt to /v1/completions (exact ISL,
        # no chat template). Otherwise: English text to /v1/chat/completions.
        if cfg['token_mode']:
            endpoint = 'completions'
            body = {'model': s['full'], 'prompt': prompts[i],
                    'max_tokens': cfg['max_tokens'], 'stream': True,
                    'stream_options': {'include_usage': True}}
        else:
            endpoint = 'chat/completions'
            body = {'model': s['full'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': cfg['max_tokens'], 'stream': True,
                    'stream_options': {'include_usage': True}}
        if cfg['force_len']:
            body['ignore_eos'] = True
        if rep_penalty is not None:
            body['repetition_penalty'] = float(rep_penalty)
        if temperature is not None:
            body['temperature'] = float(temperature)
        if seed is not None:
            body['seed'] = int(seed)
        try:
            resp = requests.post(
                f"http://{host}:{s['port']}/v1/{endpoint}",
                headers=headers, json=body, stream=True, timeout=req_timeout)
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
                text = (choice.get('text', '') if cfg['token_mode']
                        else choice.get('delta', {}).get('content', ''))
                finish = choice.get('finish_reason')
                if text:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start          # first token = prefill latency
                    elif prev_tok is not None:
                        itls.append(now - prev_tok)  # decode inter-token gap
                    prev_tok = now
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
            samples[i] = base
            return
        elapsed = time.perf_counter() - start
        out_toks = completion_toks if completion_toks is not None else token_count
        # Steady-state decode: TPOT = median inter-token gap; decode tok/s = 1/TPOT.
        # Robust to prefill-backlog stalls that confound (out-1)/(elapsed-ttft).
        tpot = statistics.median(itls) if itls else None
        decode_tps = (1.0 / tpot) if tpot else None
        # Prefill throughput: prompt tokens prefilled per second of TTFT.
        prefill_tps = (prompt_toks / ttft) if (ttft and prompt_toks) else None
        samples[i] = {**base, 'ok': True, 'ttft_ms': ttft * 1000 if ttft else None,
                      'elapsed': elapsed, 'isl': prompt_toks, 'out_toks': out_toks,
                      'tpot_ms': tpot * 1000 if tpot else None,
                      'decode_tps': decode_tps, 'prefill_tps': prefill_tps}

    # Reserve `total` blank lines for the redraw region.
    for s in streams:
        print(f"{tag_of(s).ljust(tag_width)} ")
    sys.stdout.flush()

    threads = [threading.Thread(target=run_stream, args=(i, s)) for i, s in enumerate(streams)]
    ws = time.perf_counter()
    for j, t in enumerate(threads):
        t.start()
        if ramp_s and j < len(threads) - 1:
            time.sleep(ramp_s)   # stagger launches to avoid a synchronized prefill stampede
    for t in threads:
        t.join()
    we = time.perf_counter() - ws
    redraw(force=True)  # ensure the final token state is painted (throttle may have skipped it)

    print()
    for i, s in enumerate(streams):
        smp = samples[i]
        if smp and smp.get('ok'):
            io = f"ISL {smp['isl']} | OSL {smp['out_toks']} | " if smp['isl'] is not None else ''
            ttft_str = f"TTFT: {smp['ttft_ms']:.0f}ms" if smp['ttft_ms'] is not None else 'TTFT: n/a'
            pf_str = f"prefill {smp['prefill_tps']:.0f} tok/s" if smp['prefill_tps'] is not None else 'prefill n/a'
            dec_str = (f"decode {smp['decode_tps']:.1f} tok/s ({smp['tpot_ms']:.0f}ms TPOT)"
                       if smp['decode_tps'] is not None else 'decode n/a')
            line = f"{io}{ttft_str} | {pf_str} | {dec_str} | {smp['elapsed']:.2f}s"
        else:
            line = 'no metrics (error?)'
        print(f'{tag_of(s).ljust(tag_width)} {line}')
    return we, [s for s in samples if s and s.get('ok')]

def fmt_cfg(cfg):
    ins = (f"ISL {cfg['isl']} (token IDs)" if cfg['token_mode']
           else f'prompt="{prompt}"')
    outs = (f"OSL {cfg['osl']} (ignore_eos)" if cfg['force_len']
            else f"max_tokens {cfg['max_tokens']}")
    return f"concurrency {cfg['n']} | {ins} | {outs}"

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

rows = []   # one aggregated row per (datapoint, server)

def record(cfg, server_list, samples, wall):
    """Aggregate a wave's per-stream samples into one row per server. agg tok/s
    is total output tokens / wall (whole-server throughput at this concurrency);
    tok/s/req is the mean per-stream decode rate."""
    for srv in server_list:
        ss = [s for s in samples if s['port'] == srv['port']]
        if not ss:
            # No successful streams (all errored/timed out). Record a visible
            # FAILED row instead of silently dropping the datapoint.
            rows.append({
                'short': srv['short'], 'conc': cfg['n'], 'nstreams': 0,
                'isl_t': cfg['isl'], 'osl_t': cfg['osl'],
                'isl': cfg['isl'], 'osl': cfg['osl'],
                'ttft_p50': None, 'ttft_p90': None, 'ttft_max': None,
                'prefill_tps': None, 'decode_tps': None, 'tps_agg': None,
                'wall': wall, 'failed': True,
            })
            continue
        out_sum = sum(s['out_toks'] for s in ss)
        ttfts = [s['ttft_ms'] for s in ss if s['ttft_ms'] is not None]
        rows.append({
            'short': srv['short'], 'conc': cfg['n'], 'nstreams': len(ss),
            'isl_t': cfg['isl'], 'osl_t': cfg['osl'],   # requested targets (sort keys)
            'isl': mean([s['isl'] for s in ss]),        # measured (displayed)
            'osl': out_sum / len(ss),
            # Prefill: TTFT spread (p50/p90/max exposes the chunked-prefill backlog).
            'ttft_p50': _pct(ttfts, 50), 'ttft_p90': _pct(ttfts, 90),
            'ttft_max': max(ttfts) if ttfts else None,
            'prefill_tps': mean([s['prefill_tps'] for s in ss]),
            # Decode: steady-state per-user rate (1/median-ITL), mean across streams.
            'decode_tps': mean([s['decode_tps'] for s in ss]),
            'tps_agg': out_sum / wall if wall > 0 else None,   # whole-system throughput
            'wall': wall,
        })

multi = len(combos) > 1
for ci, cfg in enumerate(make_cfg(*c) for c in combos):
    if multi:
        print(f"########## datapoint {ci + 1}/{len(combos)}: {fmt_cfg(cfg)} ##########")
    if seq:
        # One server at a time (its N streams still run together) so cross-server
        # host-CPU contention (cpu_sampling) doesn't depress each server's tok/s.
        grand = 0.0
        for srv in servers:
            we, samples = run_wave(make_streams([srv], cfg['n']), cfg)
            grand += we
            # With a single stream the per-stream line already prints the wall;
            # only show the per-server wall line when N>1 streams are aggregated.
            if cfg['n'] > 1:
                print(f'--- {srv["short"]} (port {srv["port"]}) wall: {we:.2f}s ---')
            print()
            record(cfg, [srv], samples, we)
        if len(servers) > 1:
            print(f'=== datapoint sequential total wall: {grand:.2f}s ===')
            print()
    else:
        we, samples = run_wave(make_streams(servers, cfg['n']), cfg)
        # Redundant when there's only one stream total (1 server, N=1): the
        # per-stream line already shows the wall.
        if len(servers) * cfg['n'] > 1:
            print(f'--- wall: {we:.2f}s ---')
        print()
        record(cfg, servers, samples, we)

# Comparison table — only worth printing when there is more than one row to
# compare (multiple datapoints and/or multiple servers).
if len(rows) > 1:
    def _f(v, nd):
        return f'{v:.{nd}f}' if v is not None else '-'
    hdr = ['model', 'conc', 'ISL', 'OSL',
           'TTFT p50', 'TTFT p90', 'TTFT max', 'prefill t/s', 'decode t/s',
           'agg t/s', 'wall s']
    table = [hdr]
    # Sort by requested targets so each (ISL,OSL) block lists concurrencies
    # adjacently — the agg-tok/s scaling reads straight down.
    for r in sorted(rows, key=lambda r: (r['short'], r['isl_t'], r['osl_t'], r['conc'])):
        if r.get('failed'):
            table.append([r['short'], str(r['conc']), _f(r['isl'], 0), _f(r['osl'], 0),
                          'FAIL', 'FAIL', 'FAIL', 'FAIL', 'FAIL', 'FAIL', _f(r['wall'], 2)])
        else:
            table.append([r['short'], str(r['conc']), _f(r['isl'], 0), _f(r['osl'], 0),
                          _f(r['ttft_p50'], 0), _f(r['ttft_p90'], 0), _f(r['ttft_max'], 0),
                          _f(r['prefill_tps'], 0), _f(r['decode_tps'], 1),
                          _f(r['tps_agg'], 1), _f(r['wall'], 2)])
    widths = [max(len(row[c]) for row in table) for c in range(len(hdr))]
    span = sum(widths) + 3 * (len(hdr) - 1)
    print()
    print('=' * span)
    print('SWEEP SUMMARY  (TTFT p50/p90/max = prefill latency spread; prefill t/s = ISL/TTFT;')
    print('               decode t/s = steady-state 1/median-inter-token; agg t/s = total out toks / wall)')
    print('=' * span)
    for ri, row in enumerate(table):
        cells = [(row[c].ljust(widths[c]) if c == 0 else row[c].rjust(widths[c]))
                 for c in range(len(hdr))]
        print('   '.join(cells))
        if ri == 0:
            print('   '.join('-' * widths[c] for c in range(len(hdr))))
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
