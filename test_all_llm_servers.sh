#!/bin/bash
# Find all running tt-inference-server-* containers, extract each one's
# published host port, and run test_llm_server.sh against it.
#
# Usage:
#   ./test_all_llm_servers.sh              # localhost, default API key
#   HOST=foo ./test_all_llm_servers.sh     # remote host
#   API_KEY=xxx ./test_all_llm_servers.sh
#
# Notes:
# - Only containers named with the prefix "tt-inference-server-" are considered
#   (matches what run.py auto-generates: tt-inference-server-<short-uuid>).
# - Skips containers that don't respond healthy on /health (still warming up
#   or already crashed).
# - Skips containers whose /v1/models returns no entries (e.g. CNN servers).

set -u

HOST="${HOST:-localhost}"
TEST_SCRIPT="$(dirname "$0")/test_llm_server.sh"

if [[ ! -x "$TEST_SCRIPT" ]]; then
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
