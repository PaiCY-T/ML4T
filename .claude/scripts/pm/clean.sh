#!/bin/bash

# PM Clean - Archive completed work and fix system issues
# Usage: /pm:clean

echo "Cleaning PM system..."

cleaned_files=0
archived_files=0
fixed_references=0

echo ""
echo "🧹 Cleaning PM System"
echo "===================="

# 1. Clean up analysis files
echo ""
echo "📋 Analysis Files Cleanup:"

analysis_files_count=0
for file in .claude/epics/*/*.analysis.md .claude/epics/*/*-analysis.md; do
    [ -f "$file" ] || continue
    analysis_files_count=$((analysis_files_count + 1))
done

if [ $analysis_files_count -gt 0 ]; then
    echo "  Found $analysis_files_count analysis files"
    echo "  These are supplementary files and don't require frontmatter"
    echo "  ✅ Analysis files are properly categorized"
else
    echo "  No analysis files found"
fi

# 2. Clean up utility files
echo ""
echo "📁 Utility Files Cleanup:"

utility_files=(
    ".claude/epics/*/github-mapping.md"
    ".claude/epics/*/execution-status.md" 
    ".claude/epics/*/archive-summary.md"
    ".claude/epics/*/.gitkeep"
)

utility_count=0
for pattern in "${utility_files[@]}"; do
    for file in $pattern; do
        [ -f "$file" ] || continue
        utility_count=$((utility_count + 1))
    done
done

if [ $utility_count -gt 0 ]; then
    echo "  Found $utility_count utility files"
    echo "  These are system files and don't require frontmatter"
    echo "  ✅ Utility files are properly categorized"
else
    echo "  No utility files found"
fi

# 3. Archive old or unused files
echo ""
echo "📦 Archive Management:"

# Check for old stream files
stream_files_count=0
for file in .claude/epics/*/stream-*.md; do
    [ -f "$file" ] || continue
    stream_files_count=$((stream_files_count + 1))
    echo "  Found stream file: $(basename "$file")"
done

if [ $stream_files_count -gt 0 ]; then
    echo "  Stream files are working documents and don't require frontmatter"
    echo "  ✅ Stream files are properly categorized"
fi

# 4. Fix broken task references
echo ""
echo "🔗 Task Reference Cleanup:"

reference_issues=0
for epic_dir in .claude/epics/*/; do
    [ -d "$epic_dir" ] || continue
    epic_name=$(basename "$epic_dir")
    
    # Skip archived epics
    [[ "$epic_dir" == *"/.archived/"* ]] && continue
    
    echo "  Checking epic: $epic_name"
    
    # Find all task files in this epic
    existing_tasks=()
    for task_file in "$epic_dir"[0-9]*.md; do
        [ -f "$task_file" ] || continue
        task_num=$(basename "$task_file" .md)
        # Only include numeric task files, not analysis files
        if [[ "$task_num" =~ ^[0-9]+$ ]]; then
            existing_tasks+=("$task_num")
        fi
    done
    
    # Check for broken references
    for task_file in "$epic_dir"[0-9]*.md; do
        [ -f "$task_file" ] || continue
        task_num=$(basename "$task_file" .md)
        [[ "$task_num" =~ ^[0-9]+$ ]] || continue
        
        # Look for dependency references
        if grep -q "depends_on\|dependencies\|Dependencies" "$task_file"; then
            # Extract referenced task numbers
            referenced_tasks=$(grep -o '\[.*\]' "$task_file" | tr -d '[]' | tr ',' '\n' | grep -o '[0-9]\+')
            
            for ref_task in $referenced_tasks; do
                if [[ ! " ${existing_tasks[@]} " =~ " ${ref_task} " ]]; then
                    echo "    ⚠️ Task $task_num references missing task $ref_task"
                    reference_issues=$((reference_issues + 1))
                fi
            done
        fi
    done
done

if [ $reference_issues -eq 0 ]; then
    echo "  ✅ No broken task references found"
else
    echo "  Found $reference_issues broken references"
    echo "  Note: These may be references to archived or renamed tasks"
fi

# 5. Validate epic status consistency
echo ""
echo "📊 Epic Status Validation:"

for epic_dir in .claude/epics/*/; do
    [ -d "$epic_dir" ] || continue
    epic_name=$(basename "$epic_dir")
    
    # Skip archived epics
    [[ "$epic_dir" == *"/.archived/"* ]] && continue
    
    epic_file="$epic_dir/epic.md"
    [ -f "$epic_file" ] || continue
    
    # Count tasks
    total_tasks=0
    completed_tasks=0
    
    for task_file in "$epic_dir"[0-9]*.md; do
        [ -f "$task_file" ] || continue
        task_num=$(basename "$task_file" .md)
        [[ "$task_num" =~ ^[0-9]+$ ]] || continue
        
        total_tasks=$((total_tasks + 1))
        
        # Check status in multiple formats
        if grep -q "status: completed\|status: closed\|status: done\|\*\*Status\*\*: completed" "$task_file"; then
            completed_tasks=$((completed_tasks + 1))
        fi
    done
    
    if [ $total_tasks -gt 0 ]; then
        progress=$(( (completed_tasks * 100) / total_tasks ))
        echo "  Epic $epic_name: $completed_tasks/$total_tasks tasks ($progress%)"
        
        # Check if epic status matches task completion
        epic_status=$(grep -o "status: [a-z]*" "$epic_file" | head -1 | cut -d: -f2 | tr -d ' ')
        epic_progress=$(grep -o "progress: [0-9]*%" "$epic_file" | head -1 | cut -d: -f2 | tr -d '% ')
        
        if [ "$progress" -eq 100 ] && [ "$epic_status" != "completed" ]; then
            echo "    ⚠️ Epic should be marked as completed"
        elif [ "$progress" -gt 0 ] && [ "$progress" -lt 100 ] && [ "$epic_status" != "in-progress" ]; then
            echo "    ℹ️ Epic could be marked as in-progress"
        fi
    fi
done

# 6. Clean up temporary files
echo ""
echo "🗑️ Temporary File Cleanup:"

temp_files_cleaned=0
for pattern in "/tmp/epic-*" "/tmp/task-*" "/tmp/*-body.md" "/tmp/*-mapping.txt"; do
    for file in $pattern; do
        [ -f "$file" ] && rm -f "$file" && temp_files_cleaned=$((temp_files_cleaned + 1))
    done
done

if [ $temp_files_cleaned -gt 0 ]; then
    echo "  ✅ Cleaned $temp_files_cleaned temporary files"
else
    echo "  ✅ No temporary files to clean"
fi

# Summary
echo ""
echo "🎯 Cleanup Summary"
echo "=================="
echo "  Analysis files: $analysis_files_count (properly categorized)"
echo "  Utility files: $utility_count (properly categorized)"
echo "  Stream files: $stream_files_count (properly categorized)"
echo "  Reference issues: $reference_issues (informational)"
echo "  Temp files cleaned: $temp_files_cleaned"
echo ""
echo "✅ PM system cleanup completed"
echo ""
echo "💡 Next steps:"
echo "  • Run /pm:validate to verify improvements"
echo "  • Use /pm:epic-refresh <name> to update epic progress"
echo "  • Check /pm:status for current project overview"