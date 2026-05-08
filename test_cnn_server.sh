#!/bin/bash
# Quick smoke test for tt-media-server CNN endpoints (e.g. resnet-50).
# Usage:
#   ./test_cnn_server.sh                          # localhost:8010
#   HOST=qb2-120-p01t06 PORT=8010 ./test_cnn_server.sh
#   API_KEY=xxx ./test_cnn_server.sh              # if server started without --no-auth
#   IMAGE=/path/to/pic.jpg ./test_cnn_server.sh   # use your own image

set -u

HOST="${HOST:-localhost}"
PORT="${PORT:-8010}"
BASE="http://${HOST}:${PORT}"
IMAGE="${IMAGE:-/tmp/tt_test_dog.jpg}"
# tt-media-server always enforces auth (--no-auth on run.py is vLLM-only).
# Default key is the container fallback in api_key_checker.py.
API_KEY="${API_KEY:-your-secret-key}"
AUTH=(-H "Authorization: Bearer ${API_KEY}")

hr() { printf '\n--- %s ---\n' "$*"; }

hr "Server: ${BASE}"

hr "GET /health"
code=$(curl -s -o /tmp/.tt_health -w "%{http_code}" "${AUTH[@]}" "${BASE}/health")
echo "HTTP ${code}  body: $(cat /tmp/.tt_health)"

hr "GET /v1/models"
curl -s "${AUTH[@]}" "${BASE}/v1/models" | jq . || cat /tmp/.tt_models 2>/dev/null

if [[ ! -f "${IMAGE}" ]]; then
  hr "Fetching sample image -> ${IMAGE}"
  # PyTorch's canonical golden retriever image used in resnet tutorials.
  # Should classify confidently as "golden retriever" (ImageNet class 207).
  curl -fsSL -A "Mozilla/5.0" -o "${IMAGE}" \
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg" \
    || { echo "Failed to download sample image; set IMAGE=/path/to/file.jpg" >&2; exit 1; }
fi

hr "POST /v1/cnn/search-image  (file=${IMAGE})"
curl -s -X POST "${AUTH[@]}" \
  -F "file=@${IMAGE}" \
  -F "top_k=5" \
  -F "min_confidence=10.0" \
  "${BASE}/v1/cnn/search-image" | jq .
