#!/bin/bash
echo "Getting status..."
echo ""
echo ""

echo "📋 Next Available Tasks"
echo "======================="
echo ""

# Find tasks that are open and have no dependencies or whose dependencies are closed
found=0

for epic_dir in .claude/epics/*/; do
  [ -d "$epic_dir" ] || continue
  epic_name=$(basename "$epic_dir")

  for task_file in "$epic_dir"[0-9]*.md; do
    [ -f "$task_file" ] || continue

    # Check if task is open
    status=$(grep "^status:" "$task_file" | head -1 | sed 's/^status: *//' | tr -d '\r')
    [ "$status" != "open" ] && [ -n "$status" ] && continue

    # Check dependencies - try both formats
    deps=$(grep -E "^(dependencies|depends_on):" "$task_file" | head -1 | sed 's/^[^:]*: *\[//' | sed 's/\]//' | tr -d '\r')

    # Check if all dependencies are completed
    deps_ready=true
    if [ -n "$deps" ] && [ "$deps" != "dependencies:" ] && [ "$deps" != "depends_on:" ]; then
      # Parse comma-separated dependencies
      IFS=',' read -ra dep_array <<< "$deps"
      for dep in "${dep_array[@]}"; do
        dep=$(echo "$dep" | tr -d ' ')  # Remove spaces
        dep_file="${epic_dir}${dep}.md"
        if [ -f "$dep_file" ]; then
          dep_status=$(grep "^status:" "$dep_file" | head -1 | sed 's/^status: *//' | tr -d '\r')
          if [ "$dep_status" != "completed" ] && [ "$dep_status" != "closed" ] && [ "$dep_status" != "done" ]; then
            deps_ready=false
            break
          fi
        else
          deps_ready=false
          break
        fi
      done
    fi

    # If all dependencies are completed, task is available
    if [ "$deps_ready" = "true" ]; then
      task_name=$(grep "^name:" "$task_file" | head -1 | sed 's/^name: *//' | tr -d '\r')
      task_num=$(basename "$task_file" .md)
      parallel=$(grep "^parallel:" "$task_file" | head -1 | sed 's/^parallel: *//' | tr -d '\r')

      echo "✅ Ready: #$task_num - $task_name"
      echo "   Epic: $epic_name"
      [ "$parallel" = "true" ] && echo "   🔄 Can run in parallel"
      echo ""
      ((found++))
    fi
  done
done

if [ $found -eq 0 ]; then
  echo "No available tasks found."
  echo ""
  echo "💡 Suggestions:"
  echo "  • Check blocked tasks: /pm:blocked"
  echo "  • View all tasks: /pm:epic-list"
fi

echo ""
echo "📊 Summary: $found tasks ready to start"

exit 0
