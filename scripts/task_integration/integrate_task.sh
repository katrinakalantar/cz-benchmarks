#!/bin/bash
#
# integrate_task.sh - Automate task integration into cz-benchmarks framework
#
# This script converts a research-level task implementation into a framework-integrated
# branch using Claude Code for intelligent code generation.
#
# Usage:
#   ./integrate_task.sh --task-name "MyTask" --implementation path/to/implementation.py [options]
#
# Options:
#   --task-name NAME          Name of the task (e.g., "RareCellDetection")
#   --implementation PATH     Path to the research implementation.py file
#   --display-name NAME       Human-readable display name (default: same as task-name)
#   --category CATEGORY       Task category: single_cell, spatial, multimodal, or root (default: single_cell)
#   --baseline MODEL          Baseline model: PCA, LabelPrediction, None (default: PCA)
#   --config FILE             Optional YAML config file with additional parameters
#   --branch-name NAME        Custom branch name (default: feat/add-{task-name-lower})
#   --skip-branch-creation    Don't create a new branch (use current branch)
#   --dry-run                 Generate prompt but don't launch Claude Code
#   --help                    Show this help message

set -e  # Exit on error

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
IMPLEMENTATION_FILE=""
DISPLAY_NAME=""
CATEGORY="single_cell"
BASELINE_MODEL="PCA"
CONFIG_FILE=""
BRANCH_NAME=""
SKIP_BRANCH_CREATION=false
DRY_RUN=false

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
integrate_task.sh - Automate task integration into cz-benchmarks framework

USAGE:
    ./integrate_task.sh --task-name NAME --implementation PATH [OPTIONS]

REQUIRED ARGUMENTS:
    --task-name NAME          Name of the task (PascalCase, e.g., "RareCellDetection")
    --implementation PATH     Path to the research implementation.py file

OPTIONS:
    --display-name NAME       Human-readable display name (default: same as task-name)
    --category CATEGORY       Task category (default: single_cell)
                              Options: single_cell, spatial, multimodal, root
    --baseline MODEL          Baseline model (default: PCA)
                              Options: PCA, LabelPrediction, None, Custom
    --config FILE             Optional YAML config with additional parameters
    --branch-name NAME        Custom branch name (default: feat/add-{task-name-lower})
    --skip-branch-creation    Don't create a new branch (use current branch)
    --dry-run                 Generate prompt but don't launch Claude Code
    --help                    Show this help message

EXAMPLES:
    # Basic usage
    ./integrate_task.sh --task-name "RareCellDetection" --implementation ./my_task.py

    # Custom category and baseline
    ./integrate_task.sh --task-name "SpatialClustering" \\
        --implementation ./spatial_task.py \\
        --category spatial \\
        --baseline None

    # With configuration file
    ./integrate_task.sh --task-name "MyTask" \\
        --implementation ./task.py \\
        --config ./task_config.yaml

    # Dry run to see the generated prompt
    ./integrate_task.sh --task-name "TestTask" \\
        --implementation ./test.py \\
        --dry-run

For more information, see scripts/task_integration/README.md
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        --implementation)
            IMPLEMENTATION_FILE="$2"
            shift 2
            ;;
        --display-name)
            DISPLAY_NAME="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --baseline)
            BASELINE_MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --branch-name)
            BRANCH_NAME="$2"
            shift 2
            ;;
        --skip-branch-creation)
            SKIP_BRANCH_CREATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TASK_NAME" ]]; then
    print_error "Missing required argument: --task-name"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -z "$IMPLEMENTATION_FILE" ]]; then
    print_error "Missing required argument: --implementation"
    echo "Use --help for usage information"
    exit 1
fi

# Validate implementation file exists
if [[ ! -f "$IMPLEMENTATION_FILE" ]]; then
    print_error "Implementation file not found: $IMPLEMENTATION_FILE"
    exit 1
fi

# Convert to absolute path
IMPLEMENTATION_FILE="$(cd "$(dirname "$IMPLEMENTATION_FILE")" && pwd)/$(basename "$IMPLEMENTATION_FILE")"

# Set defaults
if [[ -z "$DISPLAY_NAME" ]]; then
    # Convert PascalCase to Title Case with spaces
    DISPLAY_NAME=$(echo "$TASK_NAME" | sed 's/\([A-Z]\)/ \1/g' | sed 's/^ //')
fi

if [[ -z "$BRANCH_NAME" ]]; then
    # Convert to lowercase with hyphens
    TASK_NAME_LOWER=$(echo "$TASK_NAME" | sed 's/\([A-Z]\)/-\1/g' | sed 's/^-//' | tr '[:upper:]' '[:lower:]')
    BRANCH_NAME="feat/add-${TASK_NAME_LOWER}-task"
fi

# Determine category path and file name
TASK_FILE_NAME=$(echo "$TASK_NAME" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')
TASK_FILE_NAME="${TASK_FILE_NAME//_task/}"  # Remove _task suffix if present

if [[ "$CATEGORY" == "root" ]]; then
    CATEGORY_PATH=""
else
    CATEGORY_PATH="$CATEGORY"
fi

print_info "Task Integration Configuration"
echo "  Task Name:         $TASK_NAME"
echo "  Display Name:      $DISPLAY_NAME"
echo "  Category:          $CATEGORY"
echo "  Category Path:     ${CATEGORY_PATH:-<root>}"
echo "  Task File Name:    $TASK_FILE_NAME"
echo "  Baseline Model:    $BASELINE_MODEL"
echo "  Implementation:    $IMPLEMENTATION_FILE"
echo "  Branch Name:       $BRANCH_NAME"
echo ""

# Check if we're in the repo root
cd "$REPO_ROOT"
if [[ ! -d ".git" ]]; then
    print_error "Not in a git repository. Please run from the cz-benchmarks root."
    exit 1
fi

# Check if Claude Code is available (unless dry run)
if [[ "$DRY_RUN" == false ]]; then
    if ! command -v claude &> /dev/null; then
        print_error "Claude Code CLI not found. Please install it first:"
        echo "  See: https://docs.claude.com/en/docs/claude-code"
        exit 1
    fi
fi

# Analyze the implementation file
print_info "Analyzing implementation file..."
ANALYSIS_JSON=$("$SCRIPT_DIR/analyze_implementation.py" "$IMPLEMENTATION_FILE")

if [[ $? -ne 0 ]]; then
    print_error "Failed to analyze implementation file"
    exit 1
fi

# Check if analysis found errors
if echo "$ANALYSIS_JSON" | grep -q '"error"'; then
    print_error "Analysis found errors in implementation file:"
    echo "$ANALYSIS_JSON" | grep '"error"'
    exit 1
fi

print_success "Analysis complete"

# Extract source code from analysis
SOURCE_CODE=$(echo "$ANALYSIS_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['source_code'])")

# Load config overrides if provided
CONFIG_OVERRIDES="No additional configuration provided."
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ -f "$CONFIG_FILE" ]]; then
        print_info "Loading configuration from $CONFIG_FILE"
        CONFIG_OVERRIDES="Additional configuration from $CONFIG_FILE:\n\`\`\`yaml\n$(cat "$CONFIG_FILE")\n\`\`\`"
    else
        print_warning "Config file not found: $CONFIG_FILE"
    fi
fi

# Determine baseline model class name
case "$BASELINE_MODEL" in
    PCA)
        BASELINE_CLASS="PCABaselineInput"
        ;;
    LabelPrediction)
        BASELINE_CLASS="LabelPredictionBaselineInput"
        ;;
    None)
        BASELINE_CLASS="NoBaselineInput"
        ;;
    Custom)
        BASELINE_CLASS="CustomBaselineInput  # TODO: Define custom baseline"
        ;;
    *)
        print_warning "Unknown baseline model: $BASELINE_MODEL, defaulting to PCABaselineInput"
        BASELINE_CLASS="PCABaselineInput"
        ;;
esac

# Generate the prompt from template
print_info "Generating Claude Code prompt..."

PROMPT_TEMPLATE="$SCRIPT_DIR/prompt_template.txt"
PROMPT_FILE=$(mktemp)

# Replace placeholders in template using Python for better handling
python3 << EOF > "$PROMPT_FILE"
# Read template
with open("$PROMPT_TEMPLATE") as f:
    template = f.read()

# Replace placeholders
replacements = {
    "{TASK_NAME}": "$TASK_NAME",
    "{DISPLAY_NAME}": "$DISPLAY_NAME",
    "{CATEGORY}": "$CATEGORY",
    "{CATEGORY_PATH}": "$CATEGORY_PATH",
    "{TASK_FILE_NAME}": "$TASK_FILE_NAME",
    "{BRANCH_NAME}": "$BRANCH_NAME",
    "{BASELINE_CLASS}": "$BASELINE_CLASS",
}

for placeholder, value in replacements.items():
    template = template.replace(placeholder, value)

print(template)
EOF

# Replace JSON placeholder (needs special handling)
python3 << EOF > "${PROMPT_FILE}.tmp"
import sys
prompt = open("$PROMPT_FILE").read()
analysis = '''$ANALYSIS_JSON'''
source = '''$SOURCE_CODE'''
config = '''$CONFIG_OVERRIDES'''

prompt = prompt.replace("{ANALYSIS_JSON}", analysis)
prompt = prompt.replace("{SOURCE_CODE}", source)
prompt = prompt.replace("{CONFIG_OVERRIDES}", config)

print(prompt)
EOF

mv "${PROMPT_FILE}.tmp" "$PROMPT_FILE"

print_success "Prompt generated: $PROMPT_FILE"

# Show prompt in dry run mode
if [[ "$DRY_RUN" == true ]]; then
    print_info "DRY RUN MODE - Generated prompt:"
    echo ""
    echo "=========================================="
    cat "$PROMPT_FILE"
    echo "=========================================="
    echo ""
    print_info "Prompt saved to: $PROMPT_FILE"
    print_info "To execute, run: claude --prompt \"\$(cat $PROMPT_FILE)\""
    exit 0
fi

# Create branch unless skipped
if [[ "$SKIP_BRANCH_CREATION" == false ]]; then
    print_info "Creating feature branch: $BRANCH_NAME"

    # Check if branch already exists
    if git rev-parse --verify "$BRANCH_NAME" &>/dev/null; then
        print_warning "Branch $BRANCH_NAME already exists"
        read -p "Do you want to switch to it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git checkout "$BRANCH_NAME"
        else
            print_error "Aborted"
            exit 1
        fi
    else
        # Get the main branch name
        MAIN_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
        git checkout -b "$BRANCH_NAME" "$MAIN_BRANCH"
        print_success "Created and switched to branch: $BRANCH_NAME"
    fi
else
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_info "Using current branch: $CURRENT_BRANCH"
fi

# Launch Claude Code with the prompt
print_info "Launching Claude Code for task integration..."
echo ""
print_warning "Claude Code will now guide you through the integration process."
print_warning "Follow the steps and verify each change carefully."
echo ""

# Give user a moment to read
sleep 2

# Launch Claude Code
claude --prompt "$(cat "$PROMPT_FILE")"

# Clean up temp file
rm -f "$PROMPT_FILE"

print_success "Claude Code session completed!"
echo ""
print_info "Next steps:"
echo "  1. Review all generated files carefully"
echo "  2. Run tests: pytest tests/tasks/test_${TASK_FILE_NAME}.py -v"
echo "  3. Run validation: ./scripts/task_integration/validate_integration.sh"
echo "  4. Commit changes: git add . && git commit -m 'Add $TASK_NAME task'"
echo "  5. Create pull request when ready"
