#!/usr/bin/env python3
"""
Extract MLIR graphs from TT-XLA debug logs.

Usage:
    python extract_ttir_graph.py <log_file> [options]

Examples:
    # Extract TTIR graphs (default)
    python extract_ttir_graph.py test_debug.log

    # Extract all graph types
    python extract_ttir_graph.py test_debug.log --type all

    # Extract specific type
    python extract_ttir_graph.py test_debug.log --type ttnn

    # Filter by TTNN operation
    python extract_ttir_graph.py test_debug.log --filter '%294 = "ttnn.rms_norm"'
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


class MLIRGraph:
    """Represents an MLIR graph at a specific IR level."""
    def __init__(self, graph_num, ir_type, start_line, end_line, content):
        self.graph_num = graph_num
        self.ir_type = ir_type
        self.start_line = start_line
        self.end_line = end_line
        self.content = content
        self.op_count = self._count_ops()
        self.line_count = end_line - start_line

    def _count_ops(self):
        """Count operations in this graph."""
        if self.ir_type == 'vhlo':
            pattern = r'vhlo\.'
        elif self.ir_type in ['shlo', 'shlo_frontend', 'shlo_compiler']:
            pattern = r'stablehlo\.'
        elif self.ir_type == 'ttir':
            pattern = r'= "ttir\.[a-z_]+"'
        elif self.ir_type == 'ttnn':
            pattern = r'= "ttnn\.[a-z_]+"'
        else:
            return 0
        return len(re.findall(pattern, self.content))

    def write_to_file(self, output_dir):
        """Write graph to file."""
        # Normalize type name for filename
        type_name = self.ir_type.replace('_', '')
        filename = f"graph_{self.graph_num}_{type_name}.mlir"
        filepath = Path(output_dir) / filename

        with open(filepath, 'w') as f:
            f.write(self.content)

        return filepath


def parse_mlir_modules(log_file):
    """Parse all MLIR modules from log and group into graphs."""

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find all MLIR Module markers
    mlir_markers = []
    for i, line in enumerate(lines):
        if 'MLIR Module' in line:
            match = re.search(r'MLIR Module (\w+):', line)
            if match:
                module_type = match.group(1)
                mlir_markers.append((i, module_type))

    if not mlir_markers:
        print("❌ No 'MLIR Module' markers found in log file.", file=sys.stderr)
        return []

    # Group modules into graphs
    # Each compilation produces a sequence of modules
    # We detect a new graph when we see 'vhlo' or when the sequence repeats
    graphs = []
    graph_num = 1

    for idx, (line_num, module_type) in enumerate(mlir_markers):
        # Check if this is the start of a new graph
        if module_type == 'vhlo' and idx > 0:
            graph_num += 1
        elif idx > 0:
            # Check if we're seeing a module type we've already seen in this graph
            # This indicates a new graph starting without vhlo marker
            prev_types = [mlir_markers[i][1] for i in range(max(0, idx - 6), idx)]
            if module_type in prev_types:
                graph_num += 1

        # Determine content boundaries
        start_line = line_num + 1  # Skip marker line
        if idx + 1 < len(mlir_markers):
            end_line = mlir_markers[idx + 1][0]
        else:
            end_line = len(lines)

        # Extract content
        content = ''.join(lines[start_line:end_line])

        # Create graph object
        graph = MLIRGraph(graph_num, module_type, start_line + 1, end_line + 1, content)
        graphs.append(graph)

    return graphs


def filter_graphs_by_operation(graphs, filter_pattern):
    """Filter graphs by searching for a pattern in TTNN graphs."""

    # Find TTNN graphs that match the filter
    matching_graph_nums = set()

    for graph in graphs:
        if graph.ir_type == 'ttnn':
            if filter_pattern in graph.content or re.search(re.escape(filter_pattern), graph.content):
                matching_graph_nums.add(graph.graph_num)
                print(f"  ✓ Found pattern in graph {graph.graph_num} (ttnn)")

    if not matching_graph_nums:
        print(f"  ❌ Pattern not found in any TTNN graphs", file=sys.stderr)
        return []

    # Return all graphs (any IR type) that match the graph numbers
    filtered = [g for g in graphs if g.graph_num in matching_graph_nums]
    return filtered


def print_summary_table(graphs, output_dir):
    """Print summary table of extracted graphs."""

    if not graphs:
        print("\n❌ No graphs extracted")
        return

    # Group by graph number
    by_graph_num = defaultdict(list)
    for graph in graphs:
        by_graph_num[graph.graph_num].append(graph)

    print("\n" + "=" * 100)
    print("EXTRACTED GRAPHS SUMMARY")
    print("=" * 100)
    print(f"{'Graph':<8} {'IR Type':<18} {'Lines':>8} {'Ops':>8}   {'Output File':<60}")
    print("-" * 100)

    for graph_num in sorted(by_graph_num.keys()):
        for graph in sorted(by_graph_num[graph_num], key=lambda g: ['vhlo', 'shlo', 'shlo_frontend', 'shlo_compiler', 'ttir', 'ttnn'].index(g.ir_type) if g.ir_type in ['vhlo', 'shlo', 'shlo_frontend', 'shlo_compiler', 'ttir', 'ttnn'] else 99):
            type_name = graph.ir_type.replace('_', '')
            filename = f"graph_{graph.graph_num}_{type_name}.mlir"
            filepath = Path(output_dir) / filename
            print(f"{graph.graph_num:<8} {graph.ir_type:<18} {graph.line_count:>8} {graph.op_count:>8}   {filepath}")

    print("-" * 100)

    # Statistics
    total_graphs = len(by_graph_num)
    total_ir_types = len(graphs)
    total_ops = sum(g.op_count for g in graphs)
    total_lines = sum(g.line_count for g in graphs)

    print(f"Total: {total_graphs} graph(s), {total_ir_types} IR representation(s), {total_ops:,} operations, {total_lines:,} lines")
    print("=" * 100)


def extract_graphs(log_file, graph_types, output_dir, filter_pattern=None):
    """Extract graphs of specified types from log file."""

    # Parse all modules
    print(f"📖 Parsing MLIR modules from: {log_file}")
    all_graphs = parse_mlir_modules(log_file)

    if not all_graphs:
        return []

    print(f"✅ Found {len(all_graphs)} MLIR module(s)")

    # Filter by pattern if specified
    if filter_pattern:
        print(f"\n🔍 Filtering by pattern: {filter_pattern}")
        all_graphs = filter_graphs_by_operation(all_graphs, filter_pattern)

        if not all_graphs:
            return []

    # Filter by requested types
    if 'all' not in graph_types:
        type_set = set()
        for t in graph_types:
            if t == 'shlo':
                # Include all shlo variants
                type_set.update(['shlo', 'shlo_frontend', 'shlo_compiler'])
            else:
                type_set.add(t)

        all_graphs = [g for g in all_graphs if g.ir_type in type_set]

    if not all_graphs:
        print("❌ No graphs match the specified type(s)", file=sys.stderr)
        return []

    # Write graphs to files
    print(f"\n💾 Writing graphs to: {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for graph in all_graphs:
        filepath = graph.write_to_file(output_dir)
        written_files.append(filepath)

    print(f"✅ Wrote {len(written_files)} file(s)")

    # Print summary
    print_summary_table(all_graphs, output_dir)

    return all_graphs


def main():
    parser = argparse.ArgumentParser(
        description='Extract MLIR graphs from TT-XLA debug logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract TTIR graphs (default)
  python extract_ttir_graph.py test_debug.log

  # Extract all graph types
  python extract_ttir_graph.py test_debug.log --type all

  # Extract TTNN graphs only
  python extract_ttir_graph.py test_debug.log --type ttnn

  # Extract multiple types
  python extract_ttir_graph.py test_debug.log --type ttir --type ttnn

  # Filter by TTNN operation (extracts all IR types for matching graph)
  python extract_ttir_graph.py test_debug.log --filter '%294 = "ttnn.rms_norm"'

  # Filter and extract specific type
  python extract_ttir_graph.py test_debug.log --type ttir --filter 'ttnn.rms_norm'

  # Custom output directory
  python extract_ttir_graph.py test_debug.log --output-dir /my/path
        """
    )

    parser.add_argument('log_file', help='Path to debug log file')

    parser.add_argument(
        '--type', '-t',
        action='append',
        dest='types',
        choices=['all', 'vhlo', 'shlo', 'ttir', 'ttnn'],
        help='IR type to extract (can specify multiple times, default: ttir)'
    )

    parser.add_argument(
        '--filter', '-f',
        help='Filter graphs by TTNN operation pattern (searches TTNN graphs, extracts all IR types for matches)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='/tmp',
        help='Output directory for extracted graphs (default: /tmp)'
    )

    args = parser.parse_args()

    # Validate log file
    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    # Default to ttir if no type specified
    graph_types = args.types if args.types else ['ttir']

    # Extract graphs
    result = extract_graphs(
        args.log_file,
        graph_types,
        args.output_dir,
        args.filter
    )

    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
