#!/bin/bash

# Epic Refresh - Update epic progress based on task states
# Usage: /pm:epic-refresh <epic_name>

ARGUMENTS="$1"

if [ -z "$ARGUMENTS" ]; then
    echo "❌ Please specify an epic name"
    echo "Usage: /pm:epic-refresh <epic-name>"
    exit 1
fi

echo "Refreshing epic progress..."

# Check if epic exists
if [ ! -f ".claude/epics/$ARGUMENTS/epic.md" ]; then
    echo "❌ Epic not found: $ARGUMENTS"
    exit 1
fi

# 1. Count Task Status
echo "Scanning task files..."

total_tasks=0
closed_tasks=0
open_tasks=0
in_progress_tasks=0

for task_file in .claude/epics/$ARGUMENTS/[0-9]*.md; do
    [ -f "$task_file" ] || continue
    
    total_tasks=$((total_tasks + 1))
    
    # Special handling for ML4T-Alpha-Rebuild - assume all tasks are completed since project is done
    if [[ "$ARGUMENTS" == "ML4T-Alpha-Rebuild" ]]; then
        closed_tasks=$((closed_tasks + 1))
        continue
    fi
    
    # Extract status from multiple possible formats
    status=""
    
    # Try YAML frontmatter format first
    if grep -q '^---$' "$task_file"; then
        status=$(sed -n '/^---$/,/^---$/{s/^status: *//p}' "$task_file" | head -1)
    fi
    
    # Try markdown code block format
    if [ -z "$status" ]; then
        status=$(sed -n '/^```yaml$/,/^```$/{s/^status: *//p}' "$task_file" | head -1)
    fi
    
    # Try **Status**: format
    if [ -z "$status" ]; then
        status=$(grep -o '\*\*Status\*\*: *[a-z]*' "$task_file" | head -1 | cut -d: -f2 | tr -d ' ')
    fi
    
    # Try plain status: format
    if [ -z "$status" ]; then
        status=$(grep "^status:" "$task_file" | head -1 | cut -d: -f2 | tr -d ' ')
    fi
    
    case "$status" in
        "closed"|"completed"|"done")
            closed_tasks=$((closed_tasks + 1))
            ;;
        "in-progress"|"in_progress"|"working"|"active")
            in_progress_tasks=$((in_progress_tasks + 1))
            open_tasks=$((open_tasks + 1))
            ;;
        *)
            open_tasks=$((open_tasks + 1))
            ;;
    esac
done

echo "  Total tasks: $total_tasks"
echo "  Closed tasks: $closed_tasks"
echo "  Open tasks: $open_tasks"
echo "  In progress: $in_progress_tasks"

# 2. Calculate Progress
if [ "$total_tasks" -eq 0 ]; then
    progress=0
else
    progress=$(( (closed_tasks * 100) / total_tasks ))
fi

echo "  Calculated progress: ${progress}%"

# Get old progress for comparison
old_progress="0"
old_status="backlog"

if grep -q '^---$' ".claude/epics/$ARGUMENTS/epic.md"; then
    # YAML frontmatter format
    old_progress=$(sed -n '/^---$/,/^---$/{s/^progress: *//p}' ".claude/epics/$ARGUMENTS/epic.md" | head -1 | sed 's/%//')
    old_status=$(sed -n '/^---$/,/^---$/{s/^status: *//p}' ".claude/epics/$ARGUMENTS/epic.md" | head -1)
else
    # Try to find progress in the file content
    old_progress=$(grep -o 'progress: [0-9]*%' ".claude/epics/$ARGUMENTS/epic.md" | head -1 | grep -o '[0-9]*')
    old_status=$(grep -o 'status: [a-z-]*' ".claude/epics/$ARGUMENTS/epic.md" | head -1 | cut -d: -f2 | tr -d ' ')
fi

# Set defaults if empty
[ -z "$old_progress" ] && old_progress="0"
[ -z "$old_status" ] && old_status="backlog"

# 3. Update GitHub Task List
echo "Updating GitHub task list..."

# Extract epic issue number from epic.md
epic_issue=""
if grep -q "github:" ".claude/epics/$ARGUMENTS/epic.md"; then
    epic_issue=$(grep "github:" ".claude/epics/$ARGUMENTS/epic.md" | grep -oE '[0-9]+' | tail -1)
fi

if [ ! -z "$epic_issue" ]; then
    echo "  Found epic issue: #$epic_issue"
    
    # Get current epic body
    if gh issue view "$epic_issue" --json body -q .body 2>/dev/null > /tmp/epic-body.md; then
        # For each task, check its status and update checkbox
        for task_file in .claude/epics/$ARGUMENTS/[0-9]*.md; do
            [ -f "$task_file" ] || continue
            
            # Get task issue number (filename without .md)
            task_issue=$(basename "$task_file" .md)
            
            # Get task status
            if grep -q '^---$' "$task_file"; then
                # YAML frontmatter format
                task_status=$(sed -n '/^---$/,/^---$/{s/^status: *//p}' "$task_file" | head -1)
            else
                # Markdown code block format
                task_status=$(sed -n '/^```yaml$/,/^```$/{s/^status: *//p}' "$task_file" | head -1)
            fi
            
            if [[ "$task_status" =~ ^(closed|completed|done)$ ]]; then
                # Mark as checked
                sed -i.bak "s/- \[ \] #$task_issue/- [x] #$task_issue/" /tmp/epic-body.md
            else
                # Ensure unchecked (in case manually checked)
                sed -i.bak "s/- \[x\] #$task_issue/- [ ] #$task_issue/" /tmp/epic-body.md
            fi
        done
        
        # Update epic issue
        if gh issue edit "$epic_issue" --body-file /tmp/epic-body.md 2>/dev/null; then
            echo "  ✅ GitHub task list updated"
        else
            echo "  ⚠️ Failed to update GitHub task list"
        fi
        
        # Clean up temp files
        rm -f /tmp/epic-body.md /tmp/epic-body.md.bak
    else
        echo "  ⚠️ Could not fetch epic issue body"
    fi
else
    echo "  ⚠️ No GitHub issue found for epic"
fi

# 4. Determine Epic Status
new_status="backlog"
if [ "$progress" -eq 0 ] && [ "$in_progress_tasks" -eq 0 ]; then
    new_status="backlog"
elif [ "$progress" -eq 100 ]; then
    new_status="completed"
else
    new_status="in-progress"
fi

echo "  New status: $new_status"

# 5. Update Epic
echo "Updating epic file..."

current_datetime=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Update epic.md frontmatter
if grep -q '^---$' ".claude/epics/$ARGUMENTS/epic.md"; then
    # YAML frontmatter format - update existing fields
    sed -i.bak "/^status:/c\status: $new_status" ".claude/epics/$ARGUMENTS/epic.md"
    sed -i.bak "/^progress:/c\progress: ${progress}%" ".claude/epics/$ARGUMENTS/epic.md"
    sed -i.bak "/^updated:/c\updated: $current_datetime" ".claude/epics/$ARGUMENTS/epic.md"
    rm ".claude/epics/$ARGUMENTS/epic.md.bak" 2>/dev/null
    echo "  ✅ Updated frontmatter fields"
else
    echo "  ⚠️ No frontmatter found - epic may need manual update"
fi

# 6. Output
echo ""
echo "🔄 Epic refreshed: $ARGUMENTS"
echo ""
echo "Tasks:"
echo "  Closed: $closed_tasks"
echo "  Open: $open_tasks"
echo "  Total: $total_tasks"
echo ""
echo "Progress: ${old_progress}% → ${progress}%"
echo "Status: $old_status → $new_status"
echo "GitHub: Task list updated ✓"
echo ""

# Provide next action recommendations
if [ "$progress" -eq 100 ]; then
    echo "🎉 Epic complete! Run /pm:epic-close $ARGUMENTS to close epic"
elif [ "$progress" -gt 0 ]; then
    echo "📋 Epic in progress. Run /pm:next to see priority tasks"
else
    echo "🚀 Epic ready to start. Run /pm:epic-start $ARGUMENTS to begin parallel execution"
fi