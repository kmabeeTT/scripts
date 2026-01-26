#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to compare two MLIR files for functional equivalence.

This script normalizes MLIR IR files by:
1. Removing location attributes (#loc definitions and loc("...") annotations)
2. Normalizing whitespace
3. Comparing the resulting structures

Usage:
    python compare_mlir.py <file1.mlir> <file2.mlir>
"""

import argparse
import re
import sys
from difflib import unified_diff
from pathlib import Path


def normalize_mlir(content: str, strip_locations: bool = True) -> str:
    """
    Normalize MLIR content by removing location information and normalizing whitespace.

    Args:
        content: Raw MLIR file content
        strip_locations: Whether to remove location attributes

    Returns:
        Normalized MLIR content
    """
    lines = content.split("\n")
    normalized_lines = []

    for line in lines:
        # Skip location attribute definitions at the top of the file or trailing definitions
        if strip_locations and re.match(r"^\s*#loc\d*\s*=\s*loc\(", line):
            continue

        # Remove inline location annotations like: loc("p0.1") or loc(#loc123)
        if strip_locations:
            line = re.sub(r'\s+loc\("[^"]+"\)', "", line)
            line = re.sub(r"\s+loc\(#loc\d*\)", "", line)

        # Normalize multiple spaces to single space
        line = re.sub(r"\s+", " ", line).strip()

        # Skip empty lines
        if not line:
            continue

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def extract_ops(content: str) -> dict:
    """
    Extract operation types and counts from MLIR content.

    Returns:
        Dictionary mapping operation names to their counts
    """
    ops = {}

    # Match operations like: %0 = "ttir.something"(...)
    # or: ttir.something(...)
    op_pattern = r'(?:^|[^"a-zA-Z_])([a-z_]+\.[a-z_]+)(?:\(|<|{)'

    for match in re.finditer(op_pattern, content):
        op_name = match.group(1)
        ops[op_name] = ops.get(op_name, 0) + 1

    return ops


def extract_function_signatures(content: str) -> list:
    """
    Extract function signatures from MLIR content.

    Returns:
        List of function signature information (name, arg count, return count)
    """
    signatures = []

    # Match func.func declarations
    func_pattern = r"func\.func\s+@(\w+)\s*\("

    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)

        # Count arguments (rough approximation)
        # Look for the signature after the function name
        start_pos = match.end()
        # Find the matching closing parenthesis
        paren_count = 1
        pos = start_pos
        while pos < len(content) and paren_count > 0:
            if content[pos] == "(":
                paren_count += 1
            elif content[pos] == ")":
                paren_count -= 1
            pos += 1

        signature_text = content[start_pos:pos]
        # Count %arg occurrences
        arg_count = len(re.findall(r"%arg\d+", signature_text))

        signatures.append(
            {
                "name": func_name,
                "arg_count": arg_count,
            }
        )

    return signatures


def compare_mlir_files(
    file1: Path, file2: Path, show_diff: bool = False, output_dir: Path = None
) -> bool:
    """
    Compare two MLIR files for functional equivalence.

    Args:
        file1: Path to first MLIR file
        file2: Path to second MLIR file
        show_diff: Whether to show detailed diff
        output_dir: Directory to save normalized files (default: /tmp)

    Returns:
        True if files are functionally equivalent, False otherwise
    """
    print(f"Comparing:\n  {file1}\n  {file2}\n")

    # Read both files
    content1 = file1.read_text()
    content2 = file2.read_text()

    print(f"File sizes:")
    print(
        f"  {file1.name}: {len(content1):,} bytes, {len(content1.splitlines())} lines"
    )
    print(
        f"  {file2.name}: {len(content2):,} bytes, {len(content2.splitlines())} lines"
    )
    print()

    # Normalize both files
    print("Normalizing files (stripping location attributes)...")
    norm1 = normalize_mlir(content1, strip_locations=True)
    norm2 = normalize_mlir(content2, strip_locations=True)

    # Save normalized files automatically
    if output_dir is None:
        output_dir = Path("/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_file1 = output_dir / f"{file1.stem}_normalized.mlir"
    norm_file2 = output_dir / f"{file2.stem}_normalized.mlir"

    norm_file1.write_text(norm1)
    norm_file2.write_text(norm2)

    print(f"Normalized files saved to:")
    print(f"  {norm_file1}")
    print(f"  {norm_file2}")
    print()

    print(f"Normalized sizes:")
    print(f"  {file1.name}: {len(norm1):,} bytes, {len(norm1.splitlines())} lines")
    print(f"  {file2.name}: {len(norm2):,} bytes, {len(norm2.splitlines())} lines")
    print()

    # Check if normalized content is identical
    are_identical = norm1 == norm2

    if are_identical:
        print("✅ Files are FUNCTIONALLY EQUIVALENT (identical after normalization)")
    else:
        print("⚠️  Files differ after normalization")

    # Provide manual inspection commands
    print()
    print("=" * 80)
    print("Manual inspection commands:")
    print("=" * 80)
    print(f"# View normalized file 1:")
    print(f"less {norm_file1}")
    print()
    print(f"# View normalized file 2:")
    print(f"less {norm_file2}")
    print()
    print(f"# Side-by-side diff:")
    print(f"diff -y {norm_file1} {norm_file2} | less")
    print()
    print(f"# Unified diff:")
    print(f"diff -u {norm_file1} {norm_file2} | less")
    print()
    print(f"# Context diff:")
    print(f"diff -c {norm_file1} {norm_file2} | less")
    print()
    print(f"# Using vimdiff (if available):")
    print(f"vimdiff {norm_file1} {norm_file2}")
    print()
    print(f"# Using colordiff (if available):")
    print(f"colordiff -u {norm_file1} {norm_file2} | less -R")
    print("=" * 80)
    print()

    if not are_identical:
        return False

    # Extract and compare operation types
    print("Comparing operation types and counts...")
    ops1 = extract_ops(content1)
    ops2 = extract_ops(content2)

    all_ops = sorted(set(ops1.keys()) | set(ops2.keys()))

    ops_match = True
    for op in all_ops:
        count1 = ops1.get(op, 0)
        count2 = ops2.get(op, 0)
        if count1 != count2:
            print(f"  ❌ {op}: {count1} vs {count2}")
            ops_match = False
        else:
            print(f"  ✅ {op}: {count1}")

    print()

    # Extract and compare function signatures
    print("Comparing function signatures...")
    sigs1 = extract_function_signatures(content1)
    sigs2 = extract_function_signatures(content2)

    print(f"  File 1 functions: {len(sigs1)}")
    print(f"  File 2 functions: {len(sigs2)}")

    if len(sigs1) != len(sigs2):
        print(f"  ❌ Different number of functions")
    else:
        for sig1, sig2 in zip(sigs1, sigs2):
            if sig1["name"] == sig2["name"] and sig1["arg_count"] == sig2["arg_count"]:
                print(f"  ✅ {sig1['name']}: {sig1['arg_count']} args")
            else:
                print(
                    f"  ❌ Mismatch: {sig1['name']}({sig1['arg_count']} args) vs "
                    f"{sig2['name']}({sig2['arg_count']} args)"
                )

    print()

    # Show diff if requested
    if show_diff and not are_identical:
        print("=" * 80)
        print("DIFF PREVIEW (first 100 lines):")
        print("=" * 80)

        lines1 = norm1.splitlines(keepends=True)
        lines2 = norm2.splitlines(keepends=True)

        diff = unified_diff(
            lines1,
            lines2,
            fromfile=str(norm_file1),
            tofile=str(norm_file2),
            lineterm="",
        )

        # Show first 100 lines of diff
        diff_lines = list(diff)
        if diff_lines:
            for i, line in enumerate(diff_lines[:100]):
                print(line.rstrip())

            if len(diff_lines) > 100:
                print(f"\n... and {len(diff_lines) - 100} more lines of diff ...")
                print(f"Run the diff commands above to see full output")
        else:
            print("(no differences)")

        print("=" * 80)
        print()

    return are_identical and ops_match and len(sigs1) == len(sigs2)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two MLIR files for functional equivalence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file1", type=Path, help="First MLIR file")
    parser.add_argument("file2", type=Path, help="Second MLIR file")
    parser.add_argument(
        "--diff",
        "-d",
        action="store_true",
        help="Show detailed diff preview after normalization",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        metavar="DIR",
        default="/tmp",
        type=Path,
        help="Directory to save normalized files (default: /tmp)",
    )

    args = parser.parse_args()

    # Check files exist
    if not args.file1.exists():
        print(f"Error: {args.file1} does not exist", file=sys.stderr)
        sys.exit(1)
    if not args.file2.exists():
        print(f"Error: {args.file2} does not exist", file=sys.stderr)
        sys.exit(1)

    # Compare files (normalized files are automatically saved)
    equivalent = compare_mlir_files(
        args.file1, args.file2, show_diff=args.diff, output_dir=args.output_dir
    )

    # Exit with appropriate code
    sys.exit(0 if equivalent else 1)


if __name__ == "__main__":
    main()
