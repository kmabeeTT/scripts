#!/bin/bash
# Quick smoke test for tt-inference-server LLM endpoints (vLLM- or media-forge-style).
# Usage:
#   ./test_llm_server.sh                                        # localhost:8000
#   HOST=qb2-120-p01t06 PORT=8000 ./test_llm_server.sh
#   API_KEY=xxx ./test_llm_server.sh                            # if started without --no-auth
#   MODEL=meta-llama/Llama-3.2-1B-Instruct ./test_llm_server.sh # override auto-detected model

set -u

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
BASE="http://${HOST}:${PORT}"
# For forge/media LLM servers, auth is always enforced (--no-auth is vLLM-only).
# For vLLM with --no-auth, the header is harmless. Override with API_KEY=... if needed.
API_KEY="${API_KEY:-your-secret-key}"
AUTH=(-H "Authorization: Bearer ${API_KEY}")

hr() { printf '\n--- %s ---\n' "$*"; }

hr "Server: ${BASE}"

hr "GET /health"
code=$(curl -s -o /tmp/.tt_health -w "%{http_code}" "${AUTH[@]}" "${BASE}/health")
echo "HTTP ${code}  body: $(cat /tmp/.tt_health)"

hr "GET /v1/models"
models_json=$(curl -s "${AUTH[@]}" "${BASE}/v1/models")
echo "${models_json}" | jq .

# Pick a model: explicit MODEL env var wins, else first id from /v1/models.
if [[ -z "${MODEL:-}" ]]; then
  MODEL=$(echo "${models_json}" | jq -r '.data[0].id // empty')
fi
if [[ -z "${MODEL}" ]]; then
  echo "Could not determine a model; set MODEL=<name> and retry." >&2
  exit 1
fi
echo "Using MODEL=${MODEL}"

CHAT_PROMPT="Say hello in five words."
COMPLETION_PROMPT="The capital of France is"

hr "POST /v1/chat/completions"
echo "Prompt: ${CHAT_PROMPT}"
curl -s -X POST "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "${MODEL}" --arg p "${CHAT_PROMPT}" '{
        model: $m,
        messages: [{role:"user", content:$p}],
        max_tokens: 32,
        temperature: 0.0
      }')" \
  "${BASE}/v1/chat/completions" | jq .

hr "POST /v1/completions  (legacy text)"
echo "Prompt: ${COMPLETION_PROMPT}"
curl -s -X POST "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "${MODEL}" --arg p "${COMPLETION_PROMPT}" '{
        model: $m,
        prompt: $p,
        max_tokens: 16,
        temperature: 0.0
      }')" \
  "${BASE}/v1/completions" | jq .
