#!/usr/bin/env bash
# Resolve the component commits (tt-xla, tt-mlir, tt-metal) baked into a
# tt-forge wheel, then print each as a one-line git-log-style entry by querying
# the component's GitHub repo — same format as `git log --date=short
# --format="%cd (%h) by %cn (Author %ae) : %s"`.
#
# The commit manifest isn't on the tt-forge release page; it lives in the
# pjrt-plugin-tt wheel's METADATA "Summary" field. By DEFAULT this reads that
# metadata WITHOUT downloading the ~130 MB wheel: it tries the PEP 658
# `<wheel>.metadata` sidecar, then falls back to HTTP range requests over the
# wheel zip (fetches ~1% of the bytes). Use --download to fetch the full wheel.
#
# Usage:
#   tt_forge_commits.sh 1.2.0
#   tt_forge_commits.sh https://github.com/tenstorrent/tt-forge/releases/tag/1.2.0
#   tt_forge_commits.sh 1.2.0.dev20260526002843
#   tt_forge_commits.sh --download 1.2.0          # full pip download instead
#   tt_forge_commits.sh /path/to/pjrt_plugin_tt-*.whl   # read a local wheel
#
# Env overrides: PIP (default pip3, only for --download), TTPYPI_INDEX.
# Requires: gh (authenticated), python3; --download also needs pip + unzip.
set -euo pipefail

DOWNLOAD=0; ARG=""
for a in "$@"; do
  case "$a" in
    --download) DOWNLOAD=1 ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *)          ARG="$a" ;;
  esac
done
[ -n "$ARG" ] || { echo "usage: $(basename "$0") [--download] <tt-forge version | release URL | wheel path>" >&2; exit 2; }

PIP="${PIP:-pip3}"
INDEX="${TTPYPI_INDEX:-https://pypi.eng.aws.tenstorrent.com}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# --- Get the METADATA "Summary:" line (the commit manifest) ----------------
SUMMARY=""; BUILT=""
if [[ "$ARG" == *.whl ]]; then
  [ -f "$ARG" ] || { echo "ERROR: wheel not found: $ARG" >&2; exit 1; }
  echo "source: local wheel $(basename "$ARG")" >&2
  SUMMARY="$(unzip -p "$ARG" '*.dist-info/METADATA' 2>/dev/null | sed -n 's/^Summary: //p' | head -1)"
else
  case "$ARG" in
    http*://*) VER="${ARG%/}"; VER="${VER##*/}" ;;   # last path segment of URL
    *)         VER="$ARG" ;;
  esac
  VER="${VER#v}"
  if [ "$DOWNLOAD" -eq 1 ]; then
    echo "source: pip download pjrt-plugin-tt==$VER (full wheel)" >&2
    "$PIP" download "pjrt-plugin-tt==$VER" --no-deps -d "$TMP" --index-url "$INDEX" \
      >/dev/null 2>"$TMP/pip.err" || { echo "ERROR: download failed:" >&2; tail -4 "$TMP/pip.err" >&2; exit 1; }
    WHL="$(ls "$TMP"/pjrt_plugin_tt-*.whl 2>/dev/null | head -1)"
    SUMMARY="$(unzip -p "$WHL" '*.dist-info/METADATA' 2>/dev/null | sed -n 's/^Summary: //p' | head -1)"
  else
    echo "source: remote metadata for pjrt-plugin-tt==$VER (no full download)" >&2
    SUMMARY="$(python3 - "$INDEX" "$VER" <<'PY'
import sys, io, re, zipfile, urllib.request, urllib.parse
index, ver = sys.argv[1].rstrip('/'), sys.argv[2]
proj = f"{index}/pjrt-plugin-tt/"
try:
    page = urllib.request.urlopen(proj, timeout=30).read().decode("utf-8", "replace")
except Exception as e:
    sys.stderr.write(f"ERROR: cannot read simple index {proj}: {e}\n"); sys.exit(3)
cands = []
for h in re.findall(r'href=["\']([^"\']+)["\']', page):
    u = urllib.parse.urljoin(proj, h); fn = u.split('/')[-1].split('#')[0]
    if fn.startswith(f"pjrt_plugin_tt-{ver}-") and fn.endswith(".whl"):
        cands.append((( 'cp312' in fn) + ('manylinux' in fn), u, fn))
if not cands:
    sys.stderr.write(f"ERROR: no pjrt_plugin_tt-{ver}-*.whl at {proj}\n"); sys.exit(3)
cands.sort(reverse=True)
wurl = cands[0][1].split('#')[0]   # drop PEP503 #sha256=... fragment

def emit_summary(text):
    for l in text.splitlines():
        if l.startswith("Summary:"):
            print(l[len("Summary: "):]); return True
    return False

# 1) PEP 658 sidecar (one tiny GET) ----------------------------------------
try:
    md = urllib.request.urlopen(wurl + ".metadata", timeout=30).read().decode("utf-8", "replace")
    if emit_summary(md):
        sys.stderr.write(f"# via PEP 658 sidecar: {wurl.split('/')[-1]}.metadata\n"); sys.exit(0)
except Exception:
    pass

# 2) HTTP range requests over the wheel zip --------------------------------
class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0; self.fetched = 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else self.pos+off if whence==1 else self.size+off; return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0: return b""
        end = min(self.pos + n, self.size) - 1
        data = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})).read()
        self.fetched += len(data); self.pos += len(data); return data
hf = HTTPRangeFile(wurl); z = zipfile.ZipFile(hf)
name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
ok = emit_summary(z.read(name).decode("utf-8", "replace"))
sys.stderr.write(f"# via HTTP range: fetched {hf.fetched/1024:.0f} KiB of {hf.size/1048576:.0f} MiB ({hf.fetched*100.0/hf.size:.2f}%) from {wurl.split('/')[-1]}\n")
sys.exit(0 if ok else 4)
PY
)" || { echo "ERROR: remote metadata read failed (try --download)" >&2; exit 1; }
  fi
fi
[ -n "$SUMMARY" ] || { echo "ERROR: no 'Summary:' commit manifest found" >&2; exit 1; }

# --- Parse the commit manifest --------------------------------------------
# 'commit=' (tt-xla) must not be preceded by a word char or '-' (those are the
# tt-mlir-commit= / tt-metal-commit= keys).
XLA="$(grep -oP '(?<![\w-])commit=\K[0-9a-f]+'  <<<"$SUMMARY" | head -1)"
MLIR="$(grep -oP 'tt-mlir-commit=\K[0-9a-f]+'   <<<"$SUMMARY" | head -1)"
METAL="$(grep -oP 'tt-metal-commit=\K[0-9a-f]+' <<<"$SUMMARY" | head -1)"
BUILT="$(grep -oP 'built-date=\K[0-9-]+'        <<<"$SUMMARY" | head -1)"

echo "built-date: ${BUILT:-?}"
echo

# --- Look up each commit in its repo, print git-log-style ------------------
show() {  # label  repo  sha
  local label="$1" repo="$2" sha="$3" line
  if [ -z "$sha" ]; then printf "%-9s %s\n" "$label" "(not in wheel manifest)"; return; fi
  if ! line="$(gh api "repos/tenstorrent/$repo/commits/$sha" \
        --jq '"\(.commit.committer.date[0:10]) (\(.sha[0:9])) by \(.commit.committer.name) (Author \(.commit.author.email)) : \(.commit.message | split("\n")[0])"' 2>"$TMP/gh.err")"; then
    line="LOOKUP FAILED ($repo@${sha:0:9}) — $(tail -1 "$TMP/gh.err" 2>/dev/null)"
  fi
  printf "%-9s %s\n" "$label" "$line"
}

show "tt-xla"   tt-xla   "$XLA"
show "tt-mlir"  tt-mlir  "$MLIR"
show "tt-metal" tt-metal "$METAL"
