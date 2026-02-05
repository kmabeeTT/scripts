#!/usr/bin/env python3
"""
Display arch_overrides information from test config YAML files.

Usage:
    show_arch_overrides.py <file_or_directory>

Examples:
    # Single file
    show_arch_overrides.py tests/runner/test_config/torch/test_config_inference_single_device.yaml

    # All YAML files in directory (recursive)
    show_arch_overrides.py tests/runner/test_config/

    # Grep for specific arch
    show_arch_overrides.py tests/runner/test_config/ | grep qb2-blackhole
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Set


def extract_arch_overrides_from_file(file_path: Path) -> Dict[str, Set[str]]:
    """
    Extract test entries with their arch_overrides from a YAML file.

    Returns:
        Dict mapping test_name -> set of arch names that have overrides
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    entries = {}
    current_test = None
    in_arch_overrides = False
    current_archs = set()

    for line in lines:
        # Match test entry name (starts with 2 spaces, ends with colon)
        if re.match(r'^  [a-zA-Z0-9_/\-]+:', line):
            # Save previous entry if we were tracking one
            if current_test and current_archs:
                entries[current_test] = current_archs

            # Start new entry
            current_test = line.strip().rstrip(':')
            in_arch_overrides = False
            current_archs = set()

        # Check if we're entering arch_overrides section
        elif line.strip() == 'arch_overrides:' and current_test:
            in_arch_overrides = True

        # Check for architecture entries (6 spaces indent)
        elif in_arch_overrides and re.match(r'^      ([a-zA-Z0-9\-]+):', line):
            arch = line.strip().rstrip(':')
            current_archs.add(arch)

        # Exit arch_overrides if we hit a different section
        elif in_arch_overrides and line.strip() and not line.startswith('      ') and not line.startswith('    arch_overrides'):
            if current_archs:
                entries[current_test] = current_archs
            in_arch_overrides = False

    # Save last entry if needed
    if current_test and current_archs:
        entries[current_test] = current_archs

    return entries


def find_yaml_files(path: Path) -> list[Path]:
    """Find all YAML files recursively if path is directory, or return single file."""
    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(path.rglob("*.yaml")) + sorted(path.rglob("*.yml"))
    else:
        raise ValueError(f"Path not found: {path}")


def print_table(results: list[tuple[str, str, str]]):
    """Print results as a table with columns: file, test_name, archs."""
    if not results:
        print("No arch_overrides found.")
        return

    # Calculate column widths
    max_file_len = max(len(r[0]) for r in results)
    max_test_len = max(len(r[1]) for r in results)
    max_archs_len = max(len(r[2]) for r in results)

    # Header
    header_file = "File"
    header_test = "Test Name"
    header_archs = "Arch Overrides"

    col1_width = max(max_file_len, len(header_file))
    col2_width = max(max_test_len, len(header_test))
    col3_width = max(max_archs_len, len(header_archs))

    # Print header
    print(f"{header_file:<{col1_width}}  {header_test:<{col2_width}}  {header_archs}")
    print("-" * (col1_width + col2_width + col3_width + 4))

    # Print rows
    for file_name, test_name, archs in results:
        print(f"{file_name:<{col1_width}}  {test_name:<{col2_width}}  {archs}")


def main():
    parser = argparse.ArgumentParser(
        description="Display arch_overrides information from test config YAML files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test_config_inference_single_device.yaml
  %(prog)s tests/runner/test_config/
  %(prog)s tests/runner/test_config/ | grep qb2-blackhole
        """
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to YAML file or directory"
    )
    parser.add_argument(
        "-s", "--sort-by-arch",
        action="store_true",
        help="Sort output by architecture list instead of file name"
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Find all YAML files
    yaml_files = find_yaml_files(args.path)

    if not yaml_files:
        print(f"No YAML files found in: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Collect results
    results = []
    for yaml_file in yaml_files:
        try:
            entries = extract_arch_overrides_from_file(yaml_file)

            # Get relative path for cleaner output
            try:
                file_name = str(yaml_file.relative_to(Path.cwd()))
            except ValueError:
                file_name = str(yaml_file)

            for test_name, archs in entries.items():
                archs_str = ", ".join(sorted(archs))
                results.append((file_name, test_name, archs_str))

        except Exception as e:
            print(f"Warning: Error processing {yaml_file}: {e}", file=sys.stderr)

    # Sort results
    if args.sort_by_arch:
        results.sort(key=lambda x: (x[2], x[0], x[1]))
    else:
        results.sort(key=lambda x: (x[0], x[1]))

    # Print table
    print_table(results)

    # Print summary
    print()
    print(f"Total entries with arch_overrides: {len(results)}")

    # Count by architecture
    arch_counts = {}
    for _, _, archs_str in results:
        for arch in archs_str.split(", "):
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

    if arch_counts:
        print("\nArch override counts:")
        for arch in sorted(arch_counts.keys()):
            print(f"  {arch}: {arch_counts[arch]}")


if __name__ == "__main__":
    main()
