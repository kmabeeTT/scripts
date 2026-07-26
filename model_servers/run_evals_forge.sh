#!/bin/bash
# Run a forge LLM's evals against an ALREADY-RUNNING server, via run.py.
# Generic over model + port (defaults reproduce the Llama-3.1-8B-Instruct case).
#
# Copied verbatim (except the `cd` line below, adapted so this can live
# outside a tt-inference-server checkout) from:
#   https://github.com/tenstorrent/tt-inference-server/commit/517210492
#
# Start a server first, from the tt-xla venv, e.g.:
#   cd <tt-xla checkout> && source venv/activate
#   cd <tt-inference-server checkout>/tt-media-server
#   DEVICE_IDS=0 PORT=8012 ./launch_llama_8b.sh
#
# Drives `run.py --workflow evals`, which reads the model's EvalConfig in
# evals/eval_config.py and handles BOTH eval code paths:
#   - meta_* (e.g. meta_ifeval)  -> WorkflowVenvType.EVALS_META (builds work_dir)
#   - longbench_* / mmlu_pro / gpqa -> WorkflowVenvType.EVALS_COMMON (lm-eval-harness)
# A plain `lm_eval` one-liner cannot reproduce the meta_* tasks, which is why
# this wraps run.py rather than calling lm_eval directly.
#
# Usage:
#   ./run_evals_forge.sh                                   # default model, ALL tasks, ci-nightly
#   ./run_evals_forge.sh --model Qwen3-8B --port 8019      # different model + port
#   ./run_evals_forge.sh --mode smoke-test                 # ALL tasks, ~1% (fast smoke)
#   ./run_evals_forge.sh --task longbench_code_e           # one task, first 20 docs
#   ./run_evals_forge.sh --model Qwen3-4B --port 8010 --task mmlu_pro --samples 50
#   ./run_evals_forge.sh --task meta_gpqa --samples all    # one task, full doc set
#
# Hang hunting (repeat the same eval run until something wedges):
#   ./run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 \
#     --loops 20 --dir falcon3_7b_instruct_evals_0.25_loop |& tee master.log
# Each iteration's full output goes to <dir>/iter_NN.log; the master log (what
# you tee) carries only one START and one result line per iteration.
#
# Options (--flag value or --flag=value):
#   --model NAME   model name as in evals/eval_config.py / model spec
#                  (default Llama-3.1-8B-Instruct).
#   --port P       server port (default 8012).
#   --task NAME    run a single eval task (uses --eval-samples doc-id filter).
#                  Default: unset -> run ALL tasks for the model.
#   --samples N    single-task mode only: run docs 0..N-1 (default 20).
#                  ("all" runs the full task set for that task.)
#   --mode MODE    all-tasks mode only: ci-nightly | smoke-test | ci-long | ci-commit
#                  (default ci-nightly, matching CI). Ignored when --task is set
#                  (--eval-samples and --limit-samples-mode are mutually exclusive).
#   --device DEV   tt device (default p150).
#   --server-url U target a non-localhost server, e.g. http://10.0.0.5 (default 127.0.0.1)
#   --loops N      run the whole eval invocation N times back-to-back (default 1).
#                  A hang shows up as an iteration that never prints a result
#                  line -- that is the one to attach gdb/py-spy to.
#   --dir DIR      mkdir -p DIR and send each iteration's output to
#                  DIR/iter_NN.log, keeping the master log to progress lines
#                  only. Relative paths resolve against your invoking cwd, not
#                  the tt-inference-server checkout. Without --dir, iteration
#                  output is interleaved into the master log as usual.
#   --timeout SECS abort an iteration that exceeds SECS and stop the loop
#                  (no default -- unset means wait forever).
#                  WARNING for #4521-class hangs: the timeout SIGTERMs run.py,
#                  which disconnects the eval client. A client disconnect
#                  aborts the in-flight requests and can force-clear the
#                  server's wedged state -- i.e. it can destroy the evidence
#                  you are trying to catch. To catch a hang in progress, run
#                  WITHOUT --timeout (or set it generously) and let the loop
#                  block on the wedged iteration.
#   -h, --help     show this help
#
# meta_*/gpqa need HF access (HF_TOKEN); longbench_*/mmlu_pro are open. Default
# API key is "your-secret-key".
set -e

# A tt-xla venv's PYTHONPATH makes tt-xla's own tests/utils.py shadow
# tt-inference-server-2's utils/url_helpers.py ("ModuleNotFoundError: No
# module named 'jax.lax'" from tt-xla's tests/utils.py, not the real error).
# Equivalent to always invoking this script as `env -u PYTHONPATH ...`.
unset PYTHONPATH

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit "${1:-0}"; }

# Captured before the cd below so a relative --dir resolves where the user ran
# the script, matching where their `tee` master log lands.
INVOKE_PWD="$PWD"

MODEL="Llama-3.1-8B-Instruct"; PORT="8012"; DEVICE="p150"
TASK=""; SAMPLES="20"; MODE="ci-nightly"; SERVER_URL=""
LOOPS="1"; LOOP_DIR=""; ITER_TIMEOUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --model|--port|--device|--task|--samples|--mode|--server-url|--loops|--dir|--timeout)
      key="$1"; val="${2:-}"; shift 2 || { echo "ERROR: $key needs a value"; exit 1; } ;;
    --model=*|--port=*|--device=*|--task=*|--samples=*|--mode=*|--server-url=*|--loops=*|--dir=*|--timeout=*)
      key="${1%%=*}"; val="${1#*=}"; shift ;;
    *) echo "ERROR: unknown arg '$1'"; usage 1 ;;
  esac
  case "$key" in
    --model) MODEL="$val" ;;
    --port) PORT="$val" ;;
    --device) DEVICE="$val" ;;
    --task) TASK="$val" ;;
    --samples) SAMPLES="$val" ;;
    --mode) MODE="$val" ;;
    --server-url) SERVER_URL="$val" ;;
    --loops) LOOPS="$val" ;;
    --dir) LOOP_DIR="$val" ;;
    --timeout) ITER_TIMEOUT="$val" ;;
  esac
done

case "$LOOPS" in ''|*[!0-9]*) echo "ERROR: --loops must be a positive integer, got '$LOOPS'"; exit 1 ;; esac
[ "$LOOPS" -ge 1 ] || { echo "ERROR: --loops must be >= 1"; exit 1; }
if [ -n "$ITER_TIMEOUT" ]; then
  case "$ITER_TIMEOUT" in ''|*[!0-9]*) echo "ERROR: --timeout must be a positive integer (seconds), got '$ITER_TIMEOUT'"; exit 1 ;; esac
  [ "$ITER_TIMEOUT" -ge 1 ] || { echo "ERROR: --timeout must be >= 1"; exit 1; }
fi
if [ -n "$LOOP_DIR" ]; then
  case "$LOOP_DIR" in /*) ;; *) LOOP_DIR="$INVOKE_PWD/$LOOP_DIR" ;; esac
  mkdir -p "$LOOP_DIR"
fi

# ADAPTED from the original (which cd'd to its own dirname, since it lived at
# the tt-inference-server repo root): this copy lives in ~/scripts, so cd to
# a parameterized tt-inference-server checkout instead.
cd "${TT_INFERENCE_SERVER_ROOT:-$HOME/tt-inference-server}"

# Preflight: confirm the server is up.
HOST_FOR_CHECK="${SERVER_URL:-http://127.0.0.1}"
if ! curl -sf "${HOST_FOR_CHECK}:${PORT}/health" >/dev/null 2>&1 \
   && ! curl -sf "${HOST_FOR_CHECK}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: no healthy server at ${HOST_FOR_CHECK}:${PORT} — start one first (a launch_*.sh on this port, from the tt-xla venv)."; exit 1
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-your-secret-key}}"
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN not set — meta_*/gpqa need HF access (cached datasets may suffice)."

# --dev-mode: forge LLM specs live in the dev catalog (CI runs run.py --dev-mode);
# without it run.py looks in 'prod' and errors "does not support ... forge-vllm-plugin".
ARGS=(--model "$MODEL" --tt-device "$DEVICE" --engine forge
      --impl forge-vllm-plugin --workflow evals --service-port "$PORT"
      --dev-mode --skip-system-sw-validation)
[ -n "$SERVER_URL" ] && ARGS+=(--server-url "$SERVER_URL")

if [ -n "$TASK" ]; then
  if [ "$SAMPLES" = "all" ]; then
    # "all" = run the full, unrestricted doc set. A null value for the task's
    # --eval-samples entry still selects the task (task_configs.py's task
    # selection only checks dict keys) but applies no --samples limit at all,
    # so lm-eval runs its natural full set for every (sub)task regardless of
    # size. A flat index-range cap (e.g. range(4000)) is the wrong way to do
    # this: lm-eval hard-rejects it for any (sub)task with fewer than that many
    # examples ("Elements of --samples should be in the interval [0,k-1]...")
    # and silently under-covers anything bigger (e.g. full mmlu_pro ~12k).
    echo "[evals] --samples all -> no --samples limit (full natural task size)"
    EVAL_SAMPLES=$(python3 -c "import json,sys; print(json.dumps({sys.argv[1]: None}))" "$TASK")
  else
    EVAL_SAMPLES=$(python3 -c "import json,sys; print(json.dumps({'$TASK': list(range(int(sys.argv[1])))}))" "$SAMPLES")
  fi
  echo "[evals] model=$MODEL single task=$TASK samples=$SAMPLES port=$PORT"
  ARGS+=(--eval-samples "$EVAL_SAMPLES")
else
  echo "[evals] model=$MODEL ALL tasks  mode=$MODE port=$PORT"
  ARGS+=(--limit-samples-mode "$MODE")
fi

# Note: not echoing ${ARGS[*]} — in single-task mode it contains the (large)
# --eval-samples index list, which spams the log.

# Single run, no per-iteration dir: behave exactly as before (exec, straight to
# stdout, caller's tee sees everything).
if [ "$LOOPS" = "1" ] && [ -z "$LOOP_DIR" ] && [ -z "$ITER_TIMEOUT" ]; then
  exec python3 run.py "${ARGS[@]}"
fi

# --- loop mode ---------------------------------------------------------------
# Progress lines go to stderr so they stay visible/interleaved correctly in the
# master log even when an iteration's own output is redirected to a file.
ts() { date '+%Y-%m-%d %H:%M:%S'; }
hms() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }
say() { echo "$*" >&2; }

say "[$(ts)] loop start: loops=$LOOPS model=$MODEL port=$PORT${LOOP_DIR:+ dir=$LOOP_DIR}${ITER_TIMEOUT:+ timeout=${ITER_TIMEOUT}s}"
[ -z "$ITER_TIMEOUT" ] && say "[$(ts)] no --timeout: a hung iteration will block here indefinitely (by design — attach gdb/py-spy to the EngineCore then)."

loop_start=$(date +%s)
passed=0; failed=0; timed_out=0; last_iter=0
width=${#LOOPS}

for i in $(seq 1 "$LOOPS"); do
  last_iter="$i"
  iter_tag=$(printf "%0${width}d" "$i")
  if [ -n "$LOOP_DIR" ]; then
    iter_log="$LOOP_DIR/iter_${iter_tag}.log"
  else
    iter_log=""
  fi

  # Cheap liveness probe. A #4521-style wedge still answers /v1/models (uvicorn
  # is alive, only generation is stuck), so this does not false-positive on a
  # hang — it only catches a server that actually died or was killed.
  if ! curl -sf "${HOST_FOR_CHECK}:${PORT}/health" >/dev/null 2>&1 \
     && ! curl -sf "${HOST_FOR_CHECK}:${PORT}/v1/models" >/dev/null 2>&1; then
    say "[$(ts)] iter ${i}/${LOOPS} ABORT   server at ${HOST_FOR_CHECK}:${PORT} is not answering — stopping loop"
    break
  fi

  say "[$(ts)] iter ${i}/${LOOPS} START${iter_log:+   log=$iter_log}"
  iter_start=$(date +%s)

  rc=0
  if [ -n "$ITER_TIMEOUT" ]; then
    # --foreground so the timeout applies to this non-interactive child; -k
    # follows with SIGKILL 30s later if run.py ignores the TERM.
    if [ -n "$iter_log" ]; then
      timeout --foreground -k 30 "$ITER_TIMEOUT" python3 run.py "${ARGS[@]}" >"$iter_log" 2>&1 || rc=$?
    else
      timeout --foreground -k 30 "$ITER_TIMEOUT" python3 run.py "${ARGS[@]}" || rc=$?
    fi
  else
    if [ -n "$iter_log" ]; then
      python3 run.py "${ARGS[@]}" >"$iter_log" 2>&1 || rc=$?
    else
      python3 run.py "${ARGS[@]}" || rc=$?
    fi
  fi

  elapsed=$(( $(date +%s) - iter_start ))
  if [ -n "$ITER_TIMEOUT" ] && { [ "$rc" = "124" ] || [ "$rc" = "137" ]; }; then
    timed_out=$(( timed_out + 1 ))
    say "[$(ts)] iter ${i}/${LOOPS} TIMEOUT after ${elapsed}s ($(hms $elapsed)) — exceeded --timeout ${ITER_TIMEOUT}s, stopping loop"
    say "[$(ts)] NOTE: run.py was killed, which disconnects the eval client; that disconnect may have already cleared a wedged server."
    break
  elif [ "$rc" = "0" ]; then
    passed=$(( passed + 1 ))
    say "[$(ts)] iter ${i}/${LOOPS} DONE    rc=0 elapsed=${elapsed}s ($(hms $elapsed))"
  else
    failed=$(( failed + 1 ))
    say "[$(ts)] iter ${i}/${LOOPS} FAIL    rc=${rc} elapsed=${elapsed}s ($(hms $elapsed))${iter_log:+ — see $iter_log}"
  fi
done

total=$(( $(date +%s) - loop_start ))
say "[$(ts)] loop end: ${passed} passed, ${failed} failed, ${timed_out} timed out, of ${last_iter}/${LOOPS} started — total $(hms $total)"
[ -n "$LOOP_DIR" ] && say "[$(ts)] per-iteration logs: $LOOP_DIR"

# Non-zero if anything went wrong, so a wrapping script can notice.
[ "$failed" = "0" ] && [ "$timed_out" = "0" ]
