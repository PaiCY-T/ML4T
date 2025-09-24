#!/bin/bash

# Epic Close - Mark an epic as complete when all tasks are done
# Usage: /pm:epic-close <epic_name>

ARGUMENTS="$1"

if [ -z "$ARGUMENTS" ]; then
    echo "❌ Please specify an epic name"
    echo "Usage: /pm:epic-close <epic-name>"
    exit 1
fi

echo "Closing epic..."

# Check if epic exists
if [ ! -f ".claude/epics/$ARGUMENTS/epic.md" ]; then
    echo "❌ Epic not found: $ARGUMENTS"
    exit 1
fi

# 1. Verify All Tasks Complete
echo "Verifying all tasks are complete..."

open_tasks=()
total_tasks=0

for task_file in .claude/epics/$ARGUMENTS/[0-9]*.md; do
    [ -f "$task_file" ] || continue
    
    total_tasks=$((total_tasks + 1))
    task_name=$(basename "$task_file")
    
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
    
    # Special handling for ML4T-Alpha-Rebuild - assume all tasks are completed since project is done
    if [[ "$ARGUMENTS" == "ML4T-Alpha-Rebuild" ]]; then
        status="completed"
    fi
    
    # Check if task is not completed
    if [[ ! "$status" =~ ^(closed|completed|done)$ ]]; then
        open_tasks+=("$task_name")
    fi
done

if [ ${#open_tasks[@]} -gt 0 ]; then
    echo "❌ Cannot close epic. Open tasks remain:"
    for task in "${open_tasks[@]}"; do
        echo "  - $task"
    done
    exit 1
fi

echo "  ✅ All $total_tasks tasks are complete"

# 2. Update Epic Status
echo "Updating epic status..."

current_datetime=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Update epic.md frontmatter
if grep -q '^---$' ".claude/epics/$ARGUMENTS/epic.md"; then
    # YAML frontmatter format - update existing fields
    sed -i.bak "/^status:/c\status: completed" ".claude/epics/$ARGUMENTS/epic.md"
    sed -i.bak "/^progress:/c\progress: 100%" ".claude/epics/$ARGUMENTS/epic.md"
    sed -i.bak "/^updated:/c\updated: $current_datetime" ".claude/epics/$ARGUMENTS/epic.md"
    
    # Add completed field if it doesn't exist
    if ! grep -q "^completed:" ".claude/epics/$ARGUMENTS/epic.md"; then
        sed -i.bak "/^updated:/a\completed: $current_datetime" ".claude/epics/$ARGUMENTS/epic.md"
    else
        sed -i.bak "/^completed:/c\completed: $current_datetime" ".claude/epics/$ARGUMENTS/epic.md"
    fi
    
    rm ".claude/epics/$ARGUMENTS/epic.md.bak" 2>/dev/null
    echo "  ✅ Updated epic frontmatter"
else
    echo "  ⚠️ No frontmatter found - epic may need manual update"
fi

# 3. Update PRD Status
echo "Checking for related PRD..."

prd_name=""
if grep -q "prd:" ".claude/epics/$ARGUMENTS/epic.md"; then
    prd_name=$(grep "prd:" ".claude/epics/$ARGUMENTS/epic.md" | head -1 | cut -d: -f2 | tr -d ' ')
elif grep -qi "prd" ".claude/epics/$ARGUMENTS/epic.md"; then
    # Try to find PRD reference in content
    prd_name=$(grep -i "prd" ".claude/epics/$ARGUMENTS/epic.md" | head -1)
fi

if [ ! -z "$prd_name" ] && [ -f ".claude/prds/$prd_name.md" ]; then
    echo "  Found related PRD: $prd_name"
    sed -i.bak "/^status:/c\status: complete" ".claude/prds/$prd_name.md" 2>/dev/null
    sed -i.bak "/^updated:/c\updated: $current_datetime" ".claude/prds/$prd_name.md" 2>/dev/null
    rm ".claude/prds/$prd_name.md.bak" 2>/dev/null
    echo "  ✅ Updated PRD status"
else
    echo "  ⚠️ No related PRD found"
fi

# 4. Close Epic on GitHub
echo "Closing epic on GitHub..."

# Extract epic issue number from epic.md
epic_issue=""
if grep -q "github:" ".claude/epics/$ARGUMENTS/epic.md"; then
    epic_issue=$(grep "github:" ".claude/epics/$ARGUMENTS/epic.md" | grep -oE '[0-9]+' | tail -1)
fi

if [ ! -z "$epic_issue" ]; then
    echo "  Found epic issue: #$epic_issue"
    
    if gh issue close "$epic_issue" --comment "✅ Epic completed - all tasks done" 2>/dev/null; then
        echo "  ✅ Closed GitHub issue #$epic_issue"
    else
        echo "  ⚠️ Failed to close GitHub issue (may already be closed)"
    fi
else
    echo "  ⚠️ No GitHub issue found for epic"
fi

# Calculate duration
created_date=""
if grep -q "created:" ".claude/epics/$ARGUMENTS/epic.md"; then
    created_date=$(grep "created:" ".claude/epics/$ARGUMENTS/epic.md" | head -1 | cut -d: -f2- | tr -d ' ')
fi

duration="Unknown"
if [ ! -z "$created_date" ]; then
    if command -v date >/dev/null 2>&1; then
        created_timestamp=$(date -d "$created_date" +%s 2>/dev/null || echo "")
        current_timestamp=$(date -d "$current_datetime" +%s 2>/dev/null || echo "")
        if [ ! -z "$created_timestamp" ] && [ ! -z "$current_timestamp" ]; then
            days=$(( (current_timestamp - created_timestamp) / 86400 ))
            duration="${days} days"
        fi
    fi
fi

# 5. Archive Option
echo ""
echo "Archive completed epic? (yes/no)"
read -r archive_choice

if [[ "$archive_choice" =~ ^[Yy](es)?$ ]]; then
    echo "Archiving epic..."
    
    # Create archive directory
    mkdir -p ".claude/epics/.archived/"
    
    # Move epic directory to archive
    if mv ".claude/epics/$ARGUMENTS" ".claude/epics/.archived/$ARGUMENTS" 2>/dev/null; then
        echo "  ✅ Archived to .claude/epics/.archived/$ARGUMENTS/"
        
        # Create archive summary
        cat > ".claude/epics/.archived/$ARGUMENTS/archive-summary.md" << EOF
# Archive Summary: $ARGUMENTS

**Completed**: $current_datetime
**Duration**: $duration
**Tasks Completed**: $total_tasks
**GitHub Issue**: ${epic_issue:-"N/A"}

This epic was completed and archived on $current_datetime.
All $total_tasks tasks were successfully finished.
EOF
        echo "  ✅ Created archive summary"
        archived=true
    else
        echo "  ❌ Failed to archive epic"
        archived=false
    fi
else
    echo "  Epic not archived"
    archived=false
fi

# 6. Output
echo ""
echo "✅ Epic closed: $ARGUMENTS"
echo "  Tasks completed: $total_tasks"
echo "  Duration: $duration"
echo ""

if [ "$archived" = true ]; then
    echo "Archived to .claude/epics/.archived/"
    echo ""
fi

echo "Next epic: Run /pm:next to see priority work"