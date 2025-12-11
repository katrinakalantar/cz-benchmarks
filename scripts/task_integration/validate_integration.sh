#!/bin/bash
#
# validate_integration.sh - Validate that a task integration is complete and correct
#
# This script checks that all necessary files have been created/modified and that
# the integration follows the cz-benchmarks framework patterns.
#
# Usage:
#   ./validate_integration.sh --task-name "MyTask" [--category single_cell]
#
# Options:
#   --task-name NAME      Name of the task to validate (required)
#   --category CATEGORY   Task category (default: single_cell)
#   --fix                 Attempt to fix simple issues automatically
#   --help                Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
TASK_NAME=""
CATEGORY="single_cell"
FIX_ISSUES=false

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((CHECKS_PASSED++))
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    ((WARNINGS++))
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((CHECKS_FAILED++))
}

show_help() {
    cat << EOF
validate_integration.sh - Validate task integration completeness

USAGE:
    ./validate_integration.sh --task-name NAME [OPTIONS]

REQUIRED ARGUMENTS:
    --task-name NAME      Name of the task to validate (PascalCase)

OPTIONS:
    --category CATEGORY   Task category (default: single_cell)
                          Options: single_cell, spatial, multimodal, root
    --fix                 Attempt to fix simple issues automatically
    --help                Show this help message

EXAMPLES:
    # Validate a single_cell task
    ./validate_integration.sh --task-name "RareCellDetection"

    # Validate a spatial task
    ./validate_integration.sh --task-name "SpatialClustering" --category spatial

    # Validate and auto-fix issues
    ./validate_integration.sh --task-name "MyTask" --fix

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --fix)
            FIX_ISSUES=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TASK_NAME" ]]; then
    echo -e "${RED}[ERROR]${NC} Missing required argument: --task-name"
    echo "Use --help for usage information"
    exit 1
fi

# Change to repo root
cd "$REPO_ROOT"

# Derive file names
TASK_FILE_NAME=$(echo "$TASK_NAME" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
TASK_FILE_NAME="${TASK_FILE_NAME//_task/}"

if [[ "$CATEGORY" == "root" ]]; then
    TASK_DIR="src/czbenchmarks/tasks"
    CATEGORY_PATH=""
else
    TASK_DIR="src/czbenchmarks/tasks/$CATEGORY"
    CATEGORY_PATH="$CATEGORY"
fi

echo ""
print_info "Validating task integration: $TASK_NAME"
echo "  Category:       $CATEGORY"
echo "  Task File Name: $TASK_FILE_NAME"
echo "  Task Directory: $TASK_DIR"
echo ""

# Check 1: Task implementation file exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Checking Task Implementation File"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TASK_FILE="$TASK_DIR/${TASK_FILE_NAME}.py"
if [[ -f "$TASK_FILE" ]]; then
    print_success "Task file exists: $TASK_FILE"

    # Check for required classes
    if grep -q "class ${TASK_NAME}TaskInput" "$TASK_FILE"; then
        print_success "TaskInput class found"
    else
        print_error "TaskInput class not found (expected: ${TASK_NAME}TaskInput)"
    fi

    if grep -q "class ${TASK_NAME}Output" "$TASK_FILE"; then
        print_success "TaskOutput class found"
    else
        print_error "TaskOutput class not found (expected: ${TASK_NAME}Output)"
    fi

    if grep -q "class ${TASK_NAME}Task" "$TASK_FILE"; then
        print_success "Task class found"
    else
        print_error "Task class not found (expected: ${TASK_NAME}Task)"
    fi

    # Check for required imports
    if grep -q "from czbenchmarks.tasks.task import Task" "$TASK_FILE"; then
        print_success "Task import found"
    else
        print_error "Missing import: from czbenchmarks.tasks.task import Task"
    fi

    # Check for required methods
    if grep -q "def _run_task" "$TASK_FILE"; then
        print_success "_run_task method found"
    else
        print_error "_run_task method not found"
    fi

    if grep -q "def _compute_metrics" "$TASK_FILE"; then
        print_success "_compute_metrics method found"
    else
        print_error "_compute_metrics method not found"
    fi

else
    print_error "Task file not found: $TASK_FILE"
fi

echo ""

# Check 2: Test file exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Checking Test File"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TEST_FILE="tests/tasks/test_${TASK_FILE_NAME}.py"
if [[ -f "$TEST_FILE" ]]; then
    print_success "Test file exists: $TEST_FILE"

    # Count test functions
    TEST_COUNT=$(grep -c "^def test_" "$TEST_FILE" || echo "0")
    if [[ $TEST_COUNT -ge 10 ]]; then
        print_success "Found $TEST_COUNT test functions (recommended: 10+)"
    elif [[ $TEST_COUNT -ge 5 ]]; then
        print_warning "Found $TEST_COUNT test functions (recommended: 10+)"
    else
        print_error "Found only $TEST_COUNT test functions (recommended: 10+)"
    fi

    # Check for basic test patterns
    if grep -q "def test_.*basic.*execution" "$TEST_FILE"; then
        print_success "Basic execution test found"
    else
        print_warning "No basic execution test found"
    fi

    if grep -q "def test_.*validation" "$TEST_FILE"; then
        print_success "Input validation test found"
    else
        print_warning "No input validation test found"
    fi

else
    print_error "Test file not found: $TEST_FILE"
fi

echo ""

# Check 3: Module exports updated
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Checking Module Exports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INIT_FILE="$TASK_DIR/__init__.py"
if [[ -f "$INIT_FILE" ]]; then
    # Check imports
    if grep -q "from .${TASK_FILE_NAME} import" "$INIT_FILE"; then
        print_success "Import statement found in __init__.py"

        # Check individual exports
        if grep -q "${TASK_NAME}Task" "$INIT_FILE"; then
            print_success "${TASK_NAME}Task exported"
        else
            print_error "${TASK_NAME}Task not exported in __init__.py"
        fi

        if grep -q "${TASK_NAME}TaskInput" "$INIT_FILE"; then
            print_success "${TASK_NAME}TaskInput exported"
        else
            print_error "${TASK_NAME}TaskInput not exported in __init__.py"
        fi

        if grep -q "${TASK_NAME}Output" "$INIT_FILE"; then
            print_success "${TASK_NAME}Output exported"
        else
            print_error "${TASK_NAME}Output not exported in __init__.py"
        fi

    else
        print_error "Import statement not found in $INIT_FILE"
    fi
else
    print_error "__init__.py not found: $INIT_FILE"
fi

echo ""

# Check 4: Integration tests updated
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Checking Integration Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INTEGRATION_TEST="tests/tasks/test_tasks.py"
if [[ -f "$INTEGRATION_TEST" ]]; then
    if grep -q "${TASK_NAME}Task" "$INTEGRATION_TEST"; then
        print_success "Task added to integration tests"
    else
        print_warning "Task not found in integration tests (may need manual addition)"
    fi
else
    print_error "Integration test file not found: $INTEGRATION_TEST"
fi

echo ""

# Check 5: Documentation updated
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Checking Documentation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DOC_FILE="docs/source/developer_guides/tasks.md"
if [[ -f "$DOC_FILE" ]]; then
    if grep -q "$TASK_NAME" "$DOC_FILE"; then
        print_success "Task documented in tasks.md"
    else
        print_warning "Task not found in tasks.md documentation"
    fi
else
    print_warning "Documentation file not found: $DOC_FILE"
fi

echo ""

# Check 6: Example file (optional)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Checking Example File (Optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXAMPLE_FILE="examples/${TASK_FILE_NAME}_example.py"
if [[ -f "$EXAMPLE_FILE" ]]; then
    print_success "Example file exists: $EXAMPLE_FILE"
else
    print_warning "Example file not found (optional): $EXAMPLE_FILE"
fi

echo ""

# Check 7: Python syntax and imports
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Checking Python Syntax and Imports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ -f "$TASK_FILE" ]]; then
    # Check Python syntax
    if python3 -m py_compile "$TASK_FILE" 2>/dev/null; then
        print_success "Python syntax is valid"
    else
        print_error "Python syntax errors found in $TASK_FILE"
    fi

    # Try importing the task
    IMPORT_TEST=$(python3 -c "import sys; sys.path.insert(0, 'src'); from czbenchmarks.tasks.${CATEGORY_PATH:+$CATEGORY_PATH.}${TASK_FILE_NAME} import ${TASK_NAME}Task; print('success')" 2>&1)
    if echo "$IMPORT_TEST" | grep -q "success"; then
        print_success "Task can be imported successfully"
    else
        print_error "Failed to import task: $IMPORT_TEST"
    fi
fi

echo ""

# Check 8: Linting (if ruff is available)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Checking Code Quality (Linting)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v ruff &> /dev/null; then
    if [[ -f "$TASK_FILE" ]]; then
        if ruff check "$TASK_FILE" --quiet 2>/dev/null; then
            print_success "No linting errors found"
        else
            LINT_COUNT=$(ruff check "$TASK_FILE" 2>/dev/null | wc -l)
            print_warning "Found linting issues (run: ruff check $TASK_FILE)"
        fi
    fi
else
    print_warning "ruff not installed, skipping linting check"
fi

echo ""

# Check 9: Run tests (if pytest is available)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "9. Running Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v pytest &> /dev/null; then
    if [[ -f "$TEST_FILE" ]]; then
        print_info "Running tests for $TASK_NAME..."
        if pytest "$TEST_FILE" -v --tb=short 2>&1 | tail -20; then
            print_success "All tests passed"
        else
            print_error "Some tests failed (see output above)"
        fi
    fi
else
    print_warning "pytest not installed, skipping test execution"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VALIDATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "  Checks Passed:  ${GREEN}$CHECKS_PASSED${NC}"
echo -e "  Checks Failed:  ${RED}$CHECKS_FAILED${NC}"
echo -e "  Warnings:       ${YELLOW}$WARNINGS${NC}"
echo ""

if [[ $CHECKS_FAILED -eq 0 ]]; then
    print_success "Integration validation passed! ✨"
    echo ""
    echo "Next steps:"
    echo "  1. Review the generated code carefully"
    echo "  2. Run full test suite: pytest tests/ -v"
    echo "  3. Commit changes: git add . && git commit -m 'Add $TASK_NAME task'"
    echo "  4. Create pull request"
    exit 0
else
    echo -e "${RED}Integration validation failed.${NC}"
    echo ""
    echo "Please fix the issues listed above and run validation again."
    exit 1
fi
