# Development Guidelines for Claude

## Commit Message Style

When committing changes, always start the commit subject line with the name of the primary file being modified, followed by a dash and brief description.

### Format
```
filename.py - Brief description of changes

Optional longer description with more details.
Can span multiple lines if needed.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Examples

**Good:**
- `extract_mlir_graphs.py - Fix extraction bug and add subdirectory option`
- `compare_graphs.py - Add support for comparing multiple graph types`
- `analyze_ttir.py - Improve operation counting logic`

**Bad:**
- `Fix extraction bug and add subdirectory option` (missing filename)
- `Fixed bug` (too vague)
- `Updates` (too vague)

### Rationale

- Makes it easy to scan git history and see which files were changed
- Helps when filtering commits by file or component
- Provides better context when reviewing git log output
- Follows common convention used in many projects

## Multi-file Commits

If multiple unrelated scripts are changed, list them:
```
extract_mlir_graphs.py, compare_graphs.py - Add common utilities
```

If changes span many files or are repo-wide, use a component prefix:
```
scripts: Refactor common IR parsing logic
```
