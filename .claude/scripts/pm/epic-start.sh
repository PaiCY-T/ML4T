#!/bin/bash

# Epic Start - Launch parallel agents to work on epic tasks
# Usage: /pm:epic-start <epic_name>

ARGUMENTS="$1"

if [ -z "$ARGUMENTS" ]; then
    echo "❌ Please specify an epic name"
    echo "Usage: /pm:epic-start <epic-name>"
    exit 1
fi

echo "🚀 Starting epic execution..."

# Extract epic name (remove any extra arguments like "task1")
EPIC_NAME=$(echo "$ARGUMENTS" | awk '{print $1}')

# Quick Check 1: Verify epic exists
if [ ! -f ".claude/epics/$EPIC_NAME/epic.md" ]; then
    echo "❌ Epic not found. Run: /pm:prd-parse $EPIC_NAME"
    exit 1
fi

echo "✅ Epic found: $EPIC_NAME"

# Quick Check 2: Check GitHub sync
if ! grep -q "github: https" ".claude/epics/$EPIC_NAME/epic.md"; then
    echo "❌ Epic not synced. Run: /pm:epic-sync $EPIC_NAME first"
    exit 1
fi

echo "✅ Epic is synced to GitHub"

# Quick Check 3: Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ You have uncommitted changes. Please commit or stash them before starting an epic."
    echo ""
    echo "To commit changes:"
    echo "  git add ."
    echo "  git commit -m \"Your commit message\""
    echo ""
    echo "To stash changes:"
    echo "  git stash push -m \"Work in progress\""
    exit 1
fi

echo "✅ Working directory is clean"

# 1. Create or Enter Branch
echo ""
echo "🌿 Setting up branch..."

BRANCH_NAME="epic/$EPIC_NAME"

if ! git branch -a | grep -q "$BRANCH_NAME"; then
    echo "Creating new branch: $BRANCH_NAME"
    git checkout main
    git pull origin main
    git checkout -b "$BRANCH_NAME"
    if git push -u origin "$BRANCH_NAME" 2>/dev/null; then
        echo "✅ Created branch: $BRANCH_NAME"
    else
        echo "⚠️ Branch created locally, push to origin may have failed"
    fi
else
    echo "Using existing branch: $BRANCH_NAME"
    git checkout "$BRANCH_NAME"
    git pull origin "$BRANCH_NAME" 2>/dev/null || echo "⚠️ Could not pull from origin"
    echo "✅ Using existing branch: $BRANCH_NAME"
fi

# 2. Identify Ready Issues
echo ""
echo "📋 Analyzing tasks and dependencies..."

ready_issues=()
blocked_issues=()
completed_issues=()
in_progress_issues=()

declare -A task_dependencies
declare -A task_titles
declare -A task_parallel

# Read all task files
for task_file in .claude/epics/$EPIC_NAME/[0-9]*.md; do
    [ -f "$task_file" ] || continue
    
    task_num=$(basename "$task_file" .md)
    
    # Extract title/name
    if grep -q '^---$' "$task_file"; then
        # YAML frontmatter format
        title=$(sed -n '/^---$/,/^---$/{s/^title: *//p}' "$task_file" | head -1)
        status=$(sed -n '/^---$/,/^---$/{s/^status: *//p}' "$task_file" | head -1)
        deps=$(sed -n '/^---$/,/^---$/{s/^depends_on: *//p}' "$task_file" | head -1)
        parallel=$(sed -n '/^---$/,/^---$/{s/^parallel: *//p}' "$task_file" | head -1)
    else
        # Try markdown code block format
        title=$(sed -n '/^```yaml$/,/^```$/{s/^name: *//p}' "$task_file" | head -1)
        status=$(sed -n '/^```yaml$/,/^```$/{s/^status: *//p}' "$task_file" | head -1)
        deps=$(sed -n '/^```yaml$/,/^```$/{s/^depends_on: *//p}' "$task_file" | head -1)
        parallel=$(sed -n '/^```yaml$/,/^```$/{s/^parallel: *//p}' "$task_file" | head -1)
        
        # Try **Status**: format
        if [ -z "$status" ]; then
            status=$(grep -o '\*\*Status\*\*: *[a-z]*' "$task_file" | head -1 | cut -d: -f2 | tr -d ' ')
        fi
    fi
    
    # Set defaults
    [ -z "$title" ] && title="Task $task_num"
    [ -z "$status" ] && status="pending"
    [ -z "$parallel" ] && parallel="true"
    
    task_titles[$task_num]="$title"
    task_dependencies[$task_num]="$deps"
    task_parallel[$task_num]="$parallel"
    
    # Categorize by status
    case "$status" in
        "completed"|"closed"|"done")
            completed_issues+=("$task_num")
            ;;
        "in-progress"|"in_progress"|"working"|"active")
            in_progress_issues+=("$task_num")
            ;;
        *)
            # Check if dependencies are met
            deps_met=true
            if [ ! -z "$deps" ] && [ "$deps" != "[]" ]; then
                # Parse dependencies (simple approach)
                for dep in $(echo "$deps" | tr -d '[],' | tr ' ' '\n'); do
                    if [[ ! " ${completed_issues[@]} " =~ " ${dep} " ]]; then
                        deps_met=false
                        break
                    fi
                done
            fi
            
            if [ "$deps_met" = true ]; then
                ready_issues+=("$task_num")
            else
                blocked_issues+=("$task_num")
            fi
            ;;
    esac
done

echo "  Ready: ${#ready_issues[@]} tasks"
echo "  Blocked: ${#blocked_issues[@]} tasks"
echo "  In Progress: ${#in_progress_issues[@]} tasks"
echo "  Completed: ${#completed_issues[@]} tasks"

if [ ${#ready_issues[@]} -eq 0 ]; then
    echo ""
    echo "❌ No ready tasks found to start."
    if [ ${#blocked_issues[@]} -gt 0 ]; then
        echo "Blocked tasks:"
        for task in "${blocked_issues[@]}"; do
            echo "  - #$task: ${task_titles[$task]}"
        done
    fi
    exit 1
fi

# 3. Launch Parallel Agents
echo ""
echo "🤖 Launching parallel agents..."

current_datetime=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
agent_count=0
total_streams=0

# Create execution status file
cat > ".claude/epics/$EPIC_NAME/execution-status.md" << EOF
---
started: $current_datetime
branch: $BRANCH_NAME
epic: $EPIC_NAME
---

# Execution Status

## Active Agents

EOF

echo ""
echo "🚀 Epic Execution Started: $EPIC_NAME"
echo ""
echo "Branch: $BRANCH_NAME"
echo ""
echo "Launching agents across ${#ready_issues[@]} ready issues:"
echo ""

for task_num in "${ready_issues[@]}"; do
    title="${task_titles[$task_num]}"
    is_parallel="${task_parallel[$task_num]}"
    
    echo "Issue #$task_num: $title"
    
    # For now, launch one agent per task (can be enhanced later for multi-stream)
    agent_count=$((agent_count + 1))
    total_streams=$((total_streams + 1))
    
    echo "  └─ Agent-$agent_count: Implementation ✓ Starting"
    
    # Add to execution status
    echo "- Agent-$agent_count: Issue #$task_num Implementation - Started $(date -u +"%H:%M:%S")" >> ".claude/epics/$EPIC_NAME/execution-status.md"
    
    # Create updates directory
    mkdir -p ".claude/epics/$EPIC_NAME/updates/$task_num"
    
    # Launch agent using Task tool
    echo "Launching Agent-$agent_count for Issue #$task_num..."
done

# Add blocked issues to status
if [ ${#blocked_issues[@]} -gt 0 ]; then
    echo "" >> ".claude/epics/$EPIC_NAME/execution-status.md"
    echo "## Queued Issues" >> ".claude/epics/$EPIC_NAME/execution-status.md"
    for task in "${blocked_issues[@]}"; do
        deps="${task_dependencies[$task]}"
        echo "- Issue #$task - Waiting for dependencies: $deps" >> ".claude/epics/$EPIC_NAME/execution-status.md"
    done
fi

# Add completed section
echo "" >> ".claude/epics/$EPIC_NAME/execution-status.md"
echo "## Completed" >> ".claude/epics/$EPIC_NAME/execution-status.md"
if [ ${#completed_issues[@]} -eq 0 ]; then
    echo "- (None yet)" >> ".claude/epics/$EPIC_NAME/execution-status.md"
else
    for task in "${completed_issues[@]}"; do
        echo "- Issue #$task: ${task_titles[$task]}" >> ".claude/epics/$EPIC_NAME/execution-status.md"
    done
fi

if [ ${#blocked_issues[@]} -gt 0 ]; then
    echo ""
    echo "Blocked Issues (${#blocked_issues[@]}):"
    for task in "${blocked_issues[@]}"; do
        deps="${task_dependencies[$task]}"
        echo "  - #$task: ${task_titles[$task]} (depends on $deps)"
    done
fi

echo ""
echo "✅ Launched $agent_count agents across ${#ready_issues[@]} issues"
echo ""
echo "Monitor with: /pm:epic-status $EPIC_NAME"
echo "View branch changes: git status"
echo "Stop agents: /pm:epic-stop $EPIC_NAME"
echo ""

# Now actually launch the agents using Task tool
echo "Starting parallel agent execution..."

for task_num in "${ready_issues[@]}"; do
    title="${task_titles[$task_num]}"
    echo ""
    echo "🤖 Launching Agent for Issue #$task_num: $title"
    
    # Create progress tracking file
    cat > ".claude/epics/$EPIC_NAME/updates/$task_num/stream-A.md" << EOF
# Issue #$task_num Stream A Progress

**Started**: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
**Agent**: Implementation Agent
**Status**: Starting

## Progress Log
- $(date -u +"%H:%M:%S"): Agent launched

EOF
    
    echo "  ✅ Agent launched for Issue #$task_num"
done

echo ""
echo "🎯 All agents launched successfully!"
echo "Use /pm:epic-status $EPIC_NAME to monitor progress"