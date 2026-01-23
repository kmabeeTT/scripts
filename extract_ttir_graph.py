#!/usr/bin/env python3
"""
Extract TTIR graph from TT-XLA debug logs.

Usage:
    python extract_ttir_graph.py <log_file> [--output <output_file>]

Example:
    python extract_ttir_graph.py test_debug.log --output ttir_graph.mlir
"""

import re
import sys
import argparse
from pathlib import Path


def extract_ttir_module(content):
    """Extract TTIR module from logs using 'MLIR Module' markers."""

    lines = content.split('\n')

    # Find MLIR Module markers
    mlir_modules = []
    for i, line in enumerate(lines):
        if 'MLIR Module' in line:
            # Extract module type (vhlo, shlo, ttir, ttnn, etc.)
            match = re.search(r'MLIR Module (\w+):', line)
            if match:
                module_type = match.group(1)
                mlir_modules.append((i, module_type))

    if not mlir_modules:
        print("  ⚠️  No 'MLIR Module' markers found, falling back to module search...", file=sys.stderr)
        return extract_ttir_module_fallback(lines)

    # Find TTIR module
    ttir_start = None
    ttir_end = None
    for idx, (line_num, module_type) in enumerate(mlir_modules):
        if module_type == 'ttir':
            ttir_start = line_num + 1  # Skip the marker line itself
            # End is the start of next module or end of file
            if idx + 1 < len(mlir_modules):
                ttir_end = mlir_modules[idx + 1][0]
            else:
                ttir_end = len(lines)
            break

    if ttir_start is None:
        print("  ❌ No 'MLIR Module ttir:' marker found", file=sys.stderr)
        return None

    # Extract TTIR section
    ttir_section = '\n'.join(lines[ttir_start:ttir_end])
    ttir_op_count = len(re.findall(r'= "ttir\.[a-z_]+"', ttir_section))

    print(f"  ✓ TTIR module at line {ttir_start} ({ttir_op_count} operations, {ttir_end - ttir_start} lines)")

    return ttir_section


def extract_ttir_module_fallback(lines):
    """Fallback: Extract TTIR module by searching for module declarations."""

    # Find all module declarations
    module_starts = []
    for i, line in enumerate(lines):
        if line.startswith('module @SyncTensorsGraph'):
            module_starts.append(i)

    if not module_starts:
        return None

    # Check each module for actual TTIR operations
    ttir_modules = []
    for idx, start_line in enumerate(module_starts):
        # Determine end of this module (start of next module or end of file)
        if idx + 1 < len(module_starts):
            end_line = module_starts[idx + 1]
        else:
            # Search for next major section marker
            end_line = len(lines)
            for i in range(start_line + 1, len(lines)):
                if lines[i].startswith('Executing operation:') or lines[i].startswith('module @'):
                    end_line = i
                    break

        # Check if this module contains actual TTIR operations
        module_section = '\n'.join(lines[start_line:end_line])
        ttir_op_count = len(re.findall(r'= "ttir\.[a-z_]+"', module_section))

        if ttir_op_count > 5:  # More than 5 actual TTIR operations
            ttir_modules.append((start_line + 1, ttir_op_count, module_section))
            print(f"  ✓ TTIR graph at line {start_line + 1} ({ttir_op_count} operations)")

    if not ttir_modules:
        return None

    # Return the first TTIR module (usually the failing one)
    return ttir_modules[0][2]


def extract_ttir_graph(log_file, output_file=None):
    """Extract the TTIR graph (After ConvertStableHLOToTTIR) from log."""

    with open(log_file, 'r') as f:
        content = f.read()

    # Method 1: Try to find explicit IR dump markers (verbose log format)
    pattern = r'// -----// IR Dump After ConvertStableHLOToTTIR.*?\n(.*?)(?=\n\n// -----// IR Dump|\nExecuting operation|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        print(f"✅ Found {len(matches)} TTIR graph(s) (verbose log format)")
        ttir_graph = matches[0]
    else:
        # Method 2: Look for TTIR modules directly (simplified log format)
        print("ℹ️  No explicit IR dump markers found, searching for TTIR modules...")
        ttir_graph = extract_ttir_module(content)

        if not ttir_graph:
            print("❌ No TTIR graph found in log file.", file=sys.stderr)
            print("\nTo enable IR dumps, run your test with:", file=sys.stderr)
            print("  export MLIR_ENABLE_DUMP=1", file=sys.stderr)
            print("  pytest <your_test> 2>&1 | tee debug.log", file=sys.stderr)
            return None

        print(f"✅ Found TTIR graph (simplified log format)")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(ttir_graph)
        print(f"✅ TTIR graph written to: {output_file}")
    else:
        print("\n" + "="*80)
        print("TTIR GRAPH:")
        print("="*80)
        print(ttir_graph[:2000])  # Show first 2000 chars
        if len(ttir_graph) > 2000:
            print(f"\n... ({len(ttir_graph) - 2000} more characters)")

    return ttir_graph


def main():
    parser = argparse.ArgumentParser(
        description='Extract TTIR graph from TT-XLA debug logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to stdout (truncated)
  python extract_ttir_graph.py test_debug.log

  # Extract to file
  python extract_ttir_graph.py test_debug.log --output ttir.mlir

  # Count TTIR operations
  python extract_ttir_graph.py test_debug.log --output ttir.mlir
  grep -E '"ttir\.[a-z_]+' ttir.mlir | wc -l
        """
    )
    parser.add_argument('log_file', help='Path to debug log file')
    parser.add_argument('-o', '--output', help='Output file for TTIR graph')

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    result = extract_ttir_graph(args.log_file, args.output)
    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
