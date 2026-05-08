#!/bin/bash
# Probe a tt-inference-server LLM endpoint, detect chat vs completions support,
# and print a copy-pasteable curl command appropriate for the served model.
# Usage:
#   ./print_curl_example.sh
#   HOST=qb2-120-p01t06 PORT=8010 ./print_curl_example.sh
#   API_KEY=xxx PROMPT="Write a haiku" MAX_TOKENS=128 ./print_curl_example.sh

set -u

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
BASE="http://${HOST}:${PORT}"
API_KEY="${API_KEY:-your-secret-key}"
PROMPT="${PROMPT:-Tell me a quick joke.}"
MAX_TOKENS="${MAX_TOKENS:-64}"

models_json=$(curl -fsS -H "Authorization: Bearer ${API_KEY}" "${BASE}/v1/models") || {
  echo "Failed to GET ${BASE}/v1/models — is the server up?" >&2
  exit 1
}
MODEL=$(echo "${models_json}" | jq -r '.data[0].id // empty')
if [[ -z "${MODEL}" ]]; then
  echo "No model returned from ${BASE}/v1/models" >&2
  exit 1
fi

# Probe /v1/chat/completions with a 1-token request. Base models lack a
# chat_template and 500 at template-load before any real inference runs.
probe_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Authorization: Bearer ${API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
  "${BASE}/v1/chat/completions")

if [[ "${probe_status}" == "200" ]]; then
  MODE=chat
else
  MODE=completions
fi

echo "Server: ${BASE}"
echo "Model:  ${MODEL}"
echo "Mode:   ${MODE} (chat probe → HTTP ${probe_status})"
echo
echo "# Copy/paste:"
echo

if [[ "${MODE}" == "chat" ]]; then
  cat <<EOF
curl -sS -X POST "${BASE}/v1/chat/completions" \\
  -H "Authorization: Bearer ${API_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${MODEL}",
    "messages": [{"role": "user", "content": "${PROMPT}"}],
    "max_tokens": ${MAX_TOKENS}
  }' | jq .
EOF
else
  cat <<EOF
curl -sS -X POST "${BASE}/v1/completions" \\
  -H "Authorization: Bearer ${API_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${MODEL}",
    "prompt": "${PROMPT}",
    "max_tokens": ${MAX_TOKENS}
  }' | jq .
EOF
fi

echo
echo "# (add  \"stream\": true  to the JSON body for SSE streaming)"
