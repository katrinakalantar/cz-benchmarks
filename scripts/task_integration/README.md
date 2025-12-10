# Task Integration Automation Tool

This directory contains tools to automate the integration of research-level task implementations into the cz-benchmarks framework using Claude Code.

## Overview

Converting a research implementation into a framework-integrated task typically requires:
- Creating 3+ new files (~600-1000 lines of code)
- Modifying 4-5 existing files
- Following specific patterns and conventions
- Writing comprehensive tests
- Updating documentation

The automation tool uses Claude Code's intelligent code generation to handle this process, reducing manual effort from 2-4 hours to 10-15 minutes.

## Quick Start

### Prerequisites

1. **Claude Code CLI**: Install from [https://docs.claude.com/en/docs/claude-code](https://docs.claude.com/en/docs/claude-code)
2. **Python 3.8+**: For the analysis script
3. **Git**: For branch management

### Basic Usage

```bash
# Navigate to the repo root
cd /path/to/cz-benchmarks

# Run the integration tool
./scripts/task_integration/integrate_task.sh \
    --task-name "MyTask" \
    --implementation path/to/implementation.py
```

This will:
1. Analyze your implementation file
2. Create a new feature branch
3. Generate a comprehensive prompt for Claude Code
4. Launch Claude Code to perform the integration
5. Guide you through validation

## Files in This Directory

| File | Purpose |
|------|---------|
| `integrate_task.sh` | Main script that orchestrates the integration process |
| `analyze_implementation.py` | Python script that extracts components from research code using AST parsing |
| `prompt_template.txt` | Template for the Claude Code prompt with detailed instructions |
| `validate_integration.sh` | Script to verify the integration is complete and correct |
| `README.md` | This file - documentation and usage guide |

## Detailed Usage

### Command-Line Options

```bash
./scripts/task_integration/integrate_task.sh [OPTIONS]

Required:
  --task-name NAME           Task name in PascalCase (e.g., "RareCellDetection")
  --implementation PATH      Path to your implementation.py file

Optional:
  --display-name NAME        Human-readable name (default: derived from task-name)
  --category CATEGORY        Task category (default: single_cell)
                             Options: single_cell, spatial, multimodal, root
  --baseline MODEL           Baseline model (default: PCA)
                             Options: PCA, LabelPrediction, None, Custom
  --config FILE              YAML config with additional parameters
  --branch-name NAME         Custom branch name (default: feat/add-{task-name})
  --skip-branch-creation     Use current branch instead of creating new one
  --dry-run                  Generate prompt without launching Claude Code
  --help                     Show help message
```

### Examples

#### Example 1: Basic Integration

```bash
# Simple single-cell task with PCA baseline
./scripts/task_integration/integrate_task.sh \
    --task-name "RareCellDetection" \
    --implementation ./research/rare_cell_detection.py
```

#### Example 2: Spatial Task

```bash
# Spatial task with custom configuration
./scripts/task_integration/integrate_task.sh \
    --task-name "SpatialClustering" \
    --implementation ./spatial_task.py \
    --category spatial \
    --baseline None
```

#### Example 3: With Configuration File

Create a config file `task_config.yaml`:
```yaml
description: "Evaluates embeddings on rare cell type identification"
requires_multiple_datasets: false
additional_notes: |
  This task uses stratified k-fold cross-validation with
  multiple classifiers to ensure robust evaluation.
```

Then run:
```bash
./scripts/task_integration/integrate_task.sh \
    --task-name "MyTask" \
    --implementation ./my_task.py \
    --config ./task_config.yaml
```

#### Example 4: Dry Run (Preview Prompt)

```bash
# See the generated prompt without executing
./scripts/task_integration/integrate_task.sh \
    --task-name "TestTask" \
    --implementation ./test.py \
    --dry-run
```

## What the Tool Does

### Step 1: Implementation Analysis

The tool analyzes your `implementation.py` file using Python's AST parser to extract:
- Function and class definitions
- Import statements
- Type hints and docstrings
- Metrics used
- Constants and variables

### Step 2: Prompt Generation

Creates a detailed prompt for Claude Code that includes:
- Your task name and configuration
- Extracted components from your implementation
- Full source code
- Step-by-step integration instructions
- Framework patterns to follow
- Validation checklist

### Step 3: Claude Code Integration

Launches Claude Code with the generated prompt. Claude Code will:
1. Create the task implementation file with TaskInput, TaskOutput, and Task classes
2. Add new metrics if needed (types.py, implementations.py)
3. Update module exports (__init__.py)
4. Generate comprehensive test file (10+ test functions)
5. Update integration tests and documentation
6. Create an example file (optional)
7. Run validation (tests, linting)

### Step 4: Validation

After Claude Code completes, you can validate the integration:

```bash
./scripts/task_integration/validate_integration.sh \
    --task-name "MyTask" \
    --category single_cell
```

## Preparing Your Implementation File

For best results, your `implementation.py` should contain:

### 1. Clear Function/Class Structure

```python
def identify_rare_cells(adata, rarity_threshold=0.05, obs="cell_type"):
    """
    Identify rare cell types in the dataset.

    Args:
        adata: AnnData object with cell annotations
        rarity_threshold: Proportion threshold for rarity (0-1)
        obs: Column name in adata.obs containing cell type labels

    Returns:
        Dictionary with rare cell types and their counts
    """
    # Your implementation here
    pass
```

### 2. Type Hints and Docstrings

```python
from typing import List, Dict, Tuple
import anndata as ad

def compute_metrics(
    predictions: List[int],
    labels: List[int]
) -> Dict[str, float]:
    """Compute classification metrics."""
    # Implementation
    pass
```

### 3. Metric Calculations

```python
from sklearn.metrics import balanced_accuracy_score, f1_score

# The tool will detect these metrics
accuracy = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
```

### 4. Parameters and Defaults

```python
# Clear parameters with sensible defaults
RARITY_THRESHOLD = 0.05
MIN_CELLS = 10
N_SPLITS = 5
```

## Integration Checklist

After running the tool, verify:

- [ ] Task implementation file created (`src/czbenchmarks/tasks/{category}/{task_name}.py`)
- [ ] Three Pydantic models defined: TaskInput, TaskOutput, Task
- [ ] Methods implemented: `_run_task`, `_compute_metrics`
- [ ] Metrics updated if needed (`metrics/types.py`, `metrics/implementations.py`)
- [ ] Module exports updated (`tasks/{category}/__init__.py`)
- [ ] Test file created with 10+ tests (`tests/tasks/test_{task_name}.py`)
- [ ] Integration tests updated (`tests/tasks/test_tasks.py`)
- [ ] Documentation updated (`docs/source/developer_guides/tasks.md`)
- [ ] Example file created (optional, `examples/{task_name}_example.py`)
- [ ] All tests pass
- [ ] No linting errors
- [ ] Task can be imported and instantiated

## Troubleshooting

### Issue: "Claude Code CLI not found"

**Solution**: Install Claude Code:
```bash
# Follow installation instructions at:
# https://docs.claude.com/en/docs/claude-code
```

### Issue: "Analysis found errors in implementation file"

**Solution**: Check your implementation.py for Python syntax errors:
```bash
python3 -m py_compile path/to/implementation.py
```

### Issue: "Branch already exists"

**Solution**: Either:
1. Use a different branch name: `--branch-name feat/my-custom-branch`
2. Delete the existing branch: `git branch -D feat/add-mytask-task`
3. Use current branch: `--skip-branch-creation`

### Issue: Tests fail after integration

**Solution**:
1. Review the test failures carefully
2. Check if test data fixtures match your task requirements
3. Update the task implementation or test file as needed
4. Re-run validation: `./scripts/task_integration/validate_integration.sh --task-name "MyTask"`

### Issue: Import errors

**Solution**:
1. Verify module exports in `__init__.py`
2. Check class names match exactly (PascalCase)
3. Try importing manually:
   ```python
   from czbenchmarks.tasks.single_cell.my_task import MyTask
   ```

## Advanced Usage

### Custom Prompt Modifications

If you need to customize the prompt:

1. Run with `--dry-run` to generate the prompt:
   ```bash
   ./scripts/task_integration/integrate_task.sh \
       --task-name "MyTask" \
       --implementation ./task.py \
       --dry-run
   ```

2. Edit the generated prompt file (location shown in output)

3. Run Claude Code manually:
   ```bash
   claude --prompt "$(cat /path/to/prompt.txt)"
   ```

### Using Existing Branch

If you want to add to an existing branch:

```bash
# Check out your branch first
git checkout feat/my-existing-branch

# Run without creating a new branch
./scripts/task_integration/integrate_task.sh \
    --task-name "MyTask" \
    --implementation ./task.py \
    --skip-branch-creation
```

### Batch Processing

For multiple tasks:

```bash
#!/bin/bash
for task_file in tasks/*.py; do
    task_name=$(basename "$task_file" .py | sed 's/_//g' | sed 's/.*/\u&/')
    ./scripts/task_integration/integrate_task.sh \
        --task-name "$task_name" \
        --implementation "$task_file"
done
```

## Framework Patterns Reference

### Task Structure

Every task needs these components:

1. **TaskInput** (Pydantic model)
   - Inherits from `TaskInput`
   - Defines all parameters
   - Has field validators

2. **TaskOutput** (Pydantic model)
   - Inherits from `TaskOutput`
   - Stores results and intermediate data

3. **Task** (main class)
   - Inherits from `Task`
   - Has `display_name`, `input_model`, `baseline_model`
   - Implements `_run_task(cell_representation, task_input)`
   - Implements `_compute_metrics(task_input, task_output)`

### File Organization

```
cz-benchmarks/
├── src/czbenchmarks/
│   ├── tasks/
│   │   ├── single_cell/
│   │   │   ├── __init__.py          # Exports
│   │   │   └── my_task.py           # Implementation
│   │   └── task.py                  # Base classes
│   └── metrics/
│       ├── types.py                 # MetricType enum
│       └── implementations.py       # Metric functions
├── tests/
│   └── tasks/
│       ├── test_my_task.py          # Task-specific tests
│       └── test_tasks.py            # Integration tests
├── examples/
│   └── my_task_example.py           # Usage example
└── docs/
    └── source/developer_guides/
        └── tasks.md                 # Documentation
```

## Contributing

If you find issues or have suggestions for improving the automation tool:

1. Check existing issues in the repo
2. Create a new issue with:
   - Description of the problem
   - Example implementation.py that caused issues
   - Expected vs actual behavior
3. Submit a PR if you have a fix

## FAQ

**Q: Can I use this with non-Python implementation files?**
A: No, the tool currently only supports Python files. Convert your implementation to Python first.

**Q: What if my task doesn't fit the standard patterns?**
A: The tool handles most common cases. For highly specialized tasks, you may need to manually adjust the generated code after Claude Code finishes.

**Q: How much does it cost to run?**
A: Each integration uses Claude Code's API, costing approximately $0.10-0.50 in API tokens depending on implementation complexity.

**Q: Can I run this offline?**
A: No, Claude Code requires an internet connection to access the Claude API.

**Q: What if I need to integrate multiple related tasks?**
A: Run the tool multiple times, once for each task. Use descriptive branch names or create sub-branches.

**Q: How do I know if my implementation is complex enough to need Claude Code?**
A: If your implementation has:
- Multiple metrics (3+)
- Complex validation logic
- Multiple classifiers or methods
- Non-standard patterns
Then Claude Code's intelligence is valuable. For simple tasks, template-based generation might suffice (future feature).

## Next Steps

After successful integration:

1. **Review the code**: Claude Code generates high-quality code, but always review
2. **Run full test suite**: `pytest tests/ -v`
3. **Test with real data**: Use the generated example file as a starting point
4. **Update documentation**: Add any task-specific notes
5. **Create pull request**: Follow the repo's PR guidelines
6. **Request review**: Get feedback from maintainers

## Support

- Documentation: [cz-benchmarks docs](https://cz-benchmarks.readthedocs.io)
- Issues: GitHub Issues in the cz-benchmarks repo
- Claude Code: [https://docs.claude.com/en/docs/claude-code](https://docs.claude.com/en/docs/claude-code)

---

**Version**: 1.0.0
**Last Updated**: 2025-12-10
**Maintained By**: CZ Biohub
