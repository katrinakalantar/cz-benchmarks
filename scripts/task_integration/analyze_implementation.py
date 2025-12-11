#!/usr/bin/env python3
"""
Analyze a research implementation.py file to extract key components for task integration.

This script uses AST parsing to extract:
- Function and class definitions
- Import statements
- Docstrings and comments
- Variable assignments
- Metrics used
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


class ImplementationAnalyzer(ast.NodeVisitor):
    """AST visitor to extract key components from implementation code."""

    def __init__(self):
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.imports: List[str] = []
        self.metrics: Set[str] = set()
        self.variables: List[Dict[str, Any]] = []
        self.constants: Dict[str, Any] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions."""
        func_info = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "docstring": ast.get_docstring(node),
            "lineno": node.lineno,
            "is_async": False,
        }

        # Extract return type hint if present
        if node.returns:
            func_info["return_type"] = ast.unparse(node.returns)

        # Extract argument type hints
        arg_types = {}
        for arg in node.args.args:
            if arg.annotation:
                arg_types[arg.arg] = ast.unparse(arg.annotation)
        if arg_types:
            func_info["arg_types"] = arg_types

        # Extract default values
        defaults = {}
        if node.args.defaults:
            # Match defaults to args (defaults are right-aligned)
            num_defaults = len(node.args.defaults)
            num_args = len(node.args.args)
            start_idx = num_args - num_defaults
            for i, default in enumerate(node.args.defaults):
                arg_name = node.args.args[start_idx + i].arg
                try:
                    defaults[arg_name] = ast.unparse(default)
                except:
                    defaults[arg_name] = "<complex_default>"
        if defaults:
            func_info["defaults"] = defaults

        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract async function definitions."""
        func_info = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "docstring": ast.get_docstring(node),
            "lineno": node.lineno,
            "is_async": True,
        }
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions."""
        class_info = {
            "name": node.name,
            "bases": [ast.unparse(base) for base in node.bases],
            "docstring": ast.get_docstring(node),
            "lineno": node.lineno,
            "methods": [],
        }

        # Extract methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                class_info["methods"].append({
                    "name": item.name,
                    "args": [arg.arg for arg in item.args.args],
                    "is_async": isinstance(item, ast.AsyncFunctionDef),
                })

        self.classes.append(class_info)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statements."""
        for alias in node.names:
            if alias.asname:
                self.imports.append(f"import {alias.name} as {alias.asname}")
            else:
                self.imports.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from...import statements."""
        module = node.module or ""
        for alias in node.names:
            if alias.asname:
                self.imports.append(f"from {module} import {alias.name} as {alias.asname}")
            else:
                self.imports.append(f"from {module} import {alias.name}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_info = {
                    "name": target.id,
                    "lineno": node.lineno,
                }
                try:
                    var_info["value"] = ast.unparse(node.value)
                except:
                    var_info["value"] = "<complex_value>"

                self.variables.append(var_info)

                # Check if it's a constant (uppercase)
                if target.id.isupper():
                    try:
                        self.constants[target.id] = ast.literal_eval(node.value)
                    except:
                        self.constants[target.id] = "<non_literal>"

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function calls to detect metric usage."""
        try:
            func_name = ast.unparse(node.func)

            # Common metric patterns
            metric_keywords = [
                "accuracy", "precision", "recall", "f1", "auc", "roc",
                "mse", "mae", "rmse", "r2", "score", "metric",
                "balanced_accuracy", "mcc", "specificity", "silhouette",
                "davies_bouldin", "calinski_harabasz", "adjusted_rand",
            ]

            if any(keyword in func_name.lower() for keyword in metric_keywords):
                self.metrics.add(func_name)
        except:
            pass

        self.generic_visit(node)


def analyze_implementation(file_path: Path) -> Dict[str, Any]:
    """
    Analyze an implementation file and extract key components.

    Args:
        file_path: Path to the implementation.py file

    Returns:
        Dictionary containing extracted components
    """
    # Read the file
    with open(file_path, "r") as f:
        source_code = f.read()

    # Parse the AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return {
            "error": f"Syntax error in file: {e}",
            "source_code": source_code,
        }

    # Analyze the AST
    analyzer = ImplementationAnalyzer()
    analyzer.visit(tree)

    # Compile results
    results = {
        "file_path": str(file_path),
        "source_code": source_code,
        "functions": analyzer.functions,
        "classes": analyzer.classes,
        "imports": analyzer.imports,
        "metrics": sorted(list(analyzer.metrics)),
        "variables": analyzer.variables,
        "constants": analyzer.constants,
    }

    # Add summary statistics
    results["summary"] = {
        "num_functions": len(analyzer.functions),
        "num_classes": len(analyzer.classes),
        "num_imports": len(analyzer.imports),
        "num_metrics": len(analyzer.metrics),
        "has_main_function": any(f["name"] == "main" for f in analyzer.functions),
        "has_run_function": any(
            f["name"] in ["run", "run_task", "execute", "compute"]
            for f in analyzer.functions
        ),
    }

    return results


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: analyze_implementation.py <implementation.py>", file=sys.stderr)
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Analyze the implementation
    results = analyze_implementation(file_path)

    # Output as JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
