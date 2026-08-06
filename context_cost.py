#!/usr/bin/env python3
"""context_cost.py -- estimate how much Claude context a file or folder would cost.

Answers "if I pull this whole directory into a session, what does that cost me?"
before you actually do it and blow the window.

Counting method, in order of preference:
  1. --exact       : Anthropic's count_tokens API. Exact for Claude, needs
                     ANTHROPIC_API_KEY, costs a round trip per batch.
  2. tiktoken      : cl100k_base (OpenAI's tokenizer). Default when installed.
                     Claude's tokenizer differs, so treat as +/-15%.
  3. char heuristic: bytes / chars-per-token, per file type. Rough fallback.

Binary files, .git, venvs, caches and the like are skipped (see SKIP_DIRS /
BINARY_EXTS). Symlinks are not followed.

Usage:
  context_cost.py PATH [PATH ...] [options]

  context_cost.py ~/debug-docs/paged_fill_cache_128k_hang-5897
  context_cost.py src/ tests/ --ext .py,.md --top 30
  context_cost.py bigfile.log --window 1000000
  context_cost.py . --exact              # needs ANTHROPIC_API_KEY

Options:
  --top N          show only the N largest files (default 25; 0 = all)
  --ext LIST       comma-separated extensions to include, e.g. .py,.md
  --window N       context window to compare against (default 200000)
  --min-tokens N   hide files under N tokens from the table (default 0)
  --exact          use the Anthropic count_tokens API instead of estimating
  --model NAME     model for --exact (default claude-sonnet-4-5)
  --json           machine-readable output
  --no-recurse     do not descend into subdirectories
  -h, --help       this help
"""
import argparse
import json
import os
import sys

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", "venv", ".venv", "env", ".tox", "build",
    "dist", ".eggs", "site-packages", ".idea", ".vscode", "third_party",
}
BINARY_EXTS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz",
    ".tgz", ".bz2", ".xz", ".7z", ".tar", ".whl", ".pt", ".pth", ".safetensors",
    ".onnx", ".npy", ".npz", ".parquet", ".db", ".sqlite", ".woff", ".woff2",
    ".ttf", ".mp4", ".mov", ".wav", ".jpeg",
}
# chars-per-token for the crude fallback; code tokenizes denser than prose
CPT_DEFAULT = 3.8
CPT_BY_EXT = {
    ".py": 3.2, ".cpp": 3.0, ".cc": 3.0, ".c": 3.0, ".h": 3.0, ".hpp": 3.0,
    ".mlir": 2.6, ".ll": 2.6, ".json": 2.8, ".yaml": 3.2, ".yml": 3.2,
    ".sh": 3.2, ".log": 3.0, ".csv": 2.8, ".md": 3.9, ".txt": 3.9,
}
# tiktoken (cl100k) tends to run slightly under Claude for code-heavy text.
CLAUDE_FUDGE = 1.10


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def looks_binary(path, sniff=2048):
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff)
    except OSError:
        return True
    if b"\0" in chunk:
        return True
    if not chunk:
        return False
    # high ratio of non-text bytes
    text = bytes(range(32, 127)) + b"\n\r\t\f\b"
    nontext = sum(b not in text for b in chunk)
    return nontext / len(chunk) > 0.30


def collect(paths, exts, recurse):
    files = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            files.append(p)
            continue
        if not os.path.isdir(p):
            print(f"warning: {p} not found, skipping", file=sys.stderr)
            continue
        for root, dirs, names in os.walk(p, followlinks=False):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            if not recurse:
                dirs[:] = []
            for name in names:
                fp = os.path.join(root, name)
                if os.path.islink(fp):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in BINARY_EXTS:
                    continue
                if exts and ext not in exts:
                    continue
                files.append(fp)
    return sorted(set(files))


def count_tokens(files, mode, model):
    """Return (rows, skipped). rows = [(path, bytes, tokens, method)]."""
    rows, skipped = [], []
    enc = None
    if mode == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    client = None
    if mode == "exact":
        import anthropic
        client = anthropic.Anthropic()

    for fp in files:
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        if size == 0:
            continue
        if looks_binary(fp):
            skipped.append((fp, size, "binary"))
            continue
        try:
            text = open(fp, "r", encoding="utf-8", errors="replace").read()
        except OSError as e:
            skipped.append((fp, size, f"unreadable: {e}"))
            continue

        if mode == "exact":
            try:
                r = client.messages.count_tokens(
                    model=model, messages=[{"role": "user", "content": text}])
                rows.append((fp, size, r.input_tokens, "exact"))
                continue
            except Exception as e:  # noqa: BLE001
                print(f"warning: exact count failed for {fp} ({e}); estimating",
                      file=sys.stderr)
        if enc is not None:
            n = int(len(enc.encode(text, disallowed_special=())) * CLAUDE_FUDGE)
            rows.append((fp, size, n, "tiktoken"))
        else:
            cpt = CPT_BY_EXT.get(os.path.splitext(fp)[1].lower(), CPT_DEFAULT)
            rows.append((fp, size, int(len(text) / cpt), "chars"))
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--ext", default="")
    ap.add_argument("--window", type=int, default=200_000)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-recurse", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    if args.help or not args.paths:
        print(__doc__)
        return 0

    exts = {e if e.startswith(".") else "." + e
            for e in (x.strip() for x in args.ext.split(",")) if e} if args.ext else None

    mode = "chars"
    if args.exact:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: --exact needs ANTHROPIC_API_KEY", file=sys.stderr)
            return 2
        mode = "exact"
    else:
        try:
            import tiktoken  # noqa: F401
            mode = "tiktoken"
        except ImportError:
            pass

    files = collect(args.paths, exts, not args.no_recurse)
    if not files:
        print("no matching files")
        return 1

    rows, skipped = count_tokens(files, mode, args.model)
    rows.sort(key=lambda r: -r[2])
    total_tok = sum(r[2] for r in rows)
    total_bytes = sum(r[1] for r in rows)

    if args.json:
        print(json.dumps({
            "mode": mode, "total_tokens": total_tok, "total_bytes": total_bytes,
            "window": args.window, "pct_of_window": round(100 * total_tok / args.window, 2),
            "files": [{"path": p, "bytes": b, "tokens": t} for p, b, t, _ in rows],
            "skipped": [{"path": p, "bytes": b, "reason": r} for p, b, r in skipped],
        }, indent=2))
        return 0

    shown = [r for r in rows if r[2] >= args.min_tokens]
    if args.top > 0:
        shown = shown[:args.top]

    width = max((len(os.path.relpath(r[0])) for r in shown), default=20)
    width = min(max(width, 20), 80)
    print(f"{'file':<{width}}  {'size':>8}  {'tokens':>9}  {'%':>6}")
    print("-" * (width + 30))
    for p, b, t, _ in shown:
        rel = os.path.relpath(p)
        if len(rel) > width:
            rel = "..." + rel[-(width - 3):]
        pct = 100 * t / total_tok if total_tok else 0
        print(f"{rel:<{width}}  {human(b):>8}  {t:>9,}  {pct:>5.1f}%")

    hidden = len(rows) - len(shown)
    if hidden > 0:
        print(f"{'... and ' + str(hidden) + ' more files':<{width}}  "
              f"{'':>8}  {sum(r[2] for r in rows[len(shown):]):>9,}")

    print("-" * (width + 30))
    print(f"{'TOTAL (' + str(len(rows)) + ' files)':<{width}}  "
          f"{human(total_bytes):>8}  {total_tok:>9,}")
    print()
    pct = 100 * total_tok / args.window
    bar = "#" * min(40, int(pct / 2.5))
    print(f"context window {args.window:,}: {pct:.1f}% used  [{bar:<40}]")
    if pct > 100:
        print(f"  -> does NOT fit; over by {total_tok - args.window:,} tokens")
    elif pct > 50:
        print(f"  -> fits, but leaves only {args.window - total_tok:,} tokens to work in")

    by_ext = {}
    for p, _, t, _ in rows:
        by_ext.setdefault(os.path.splitext(p)[1].lower() or "(none)", [0, 0])
        by_ext[os.path.splitext(p)[1].lower() or "(none)"][0] += t
        by_ext[os.path.splitext(p)[1].lower() or "(none)"][1] += 1
    if len(by_ext) > 1:
        print("\nby extension:")
        for ext, (t, c) in sorted(by_ext.items(), key=lambda kv: -kv[1][0])[:10]:
            print(f"  {ext:<12} {t:>9,} tokens  ({c} files)")

    if skipped:
        print(f"\nskipped {len(skipped)} file(s): "
              + ", ".join(os.path.basename(p) for p, _, _ in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))

    note = {"exact": "exact (Anthropic count_tokens API)",
            "tiktoken": f"estimated via tiktoken cl100k x{CLAUDE_FUDGE} — expect +/-15% vs Claude",
            "chars": "crude chars/token heuristic — install tiktoken for better"}[mode]
    print(f"\ncounting method: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
