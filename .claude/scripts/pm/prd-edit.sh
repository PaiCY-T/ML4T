#!/bin/bash

# PRD Edit Script
# Usage: /pm:prd-edit <feature_name>

FEATURE_NAME="$1"

if [ -z "$FEATURE_NAME" ]; then
  echo "❌ Error: Feature name required"
  echo "Usage: /pm:prd-edit <feature_name>"
  exit 1
fi

PRD_FILE=".claude/prds/${FEATURE_NAME}.md"

# Check if PRD exists
if [ ! -f "$PRD_FILE" ]; then
  echo "❌ Error: PRD not found: $PRD_FILE"
  echo "Create it first with: /pm:prd-new $FEATURE_NAME"
  exit 1
fi

echo "📋 Editing PRD: $FEATURE_NAME"
echo "========================="
echo ""

# Read current PRD content
echo "📖 Current PRD content:"
echo "----------------------"
cat "$PRD_FILE"
echo ""
echo "----------------------"
echo ""

# Parse frontmatter to check for epic
EPIC_NAME=$(grep "^epic:" "$PRD_FILE" 2>/dev/null | head -1 | sed 's/^epic: *//' || echo "")

# Interactive section selection
echo "📝 Which sections would you like to edit?"
echo ""
echo "Available sections:"
echo "1. Executive Summary"
echo "2. Problem Statement"
echo "3. User Stories"
echo "4. Requirements (Functional/Non-Functional)"
echo "5. Success Criteria"
echo "6. Constraints & Assumptions"
echo "7. Out of Scope"
echo "8. Dependencies"
echo "9. Custom section"
echo ""

read -p "Enter section numbers to edit (comma-separated, e.g., 1,3,5): " SECTIONS_INPUT

if [ -z "$SECTIONS_INPUT" ]; then
  echo "❌ No sections selected. Exiting."
  exit 0
fi

# Convert sections input to array
IFS=',' read -ra SECTIONS <<< "$SECTIONS_INPUT"

# Create temporary file for editing
TEMP_FILE=$(mktemp)
cp "$PRD_FILE" "$TEMP_FILE"

EDITED_SECTIONS=()

for SECTION in "${SECTIONS[@]}"; do
  SECTION=$(echo "$SECTION" | xargs)  # Trim whitespace

  case $SECTION in
    1)
      echo ""
      echo "📝 Editing Executive Summary"
      echo "=============================="
      echo "Current content:"
      sed -n '/## Executive Summary/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Executive Summary content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Executive Summary section
      sed -i '/## Executive Summary/,/## /{/## Executive Summary/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Executive Summary/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Executive Summary")
      ;;

    2)
      echo ""
      echo "📝 Editing Problem Statement"
      echo "============================="
      echo "Current content:"
      sed -n '/## Problem Statement/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Problem Statement content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Problem Statement section
      sed -i '/## Problem Statement/,/## /{/## Problem Statement/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Problem Statement/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Problem Statement")
      ;;

    3)
      echo ""
      echo "📝 Editing User Stories"
      echo "======================="
      echo "Current content:"
      sed -n '/## User Stories/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new User Stories content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace User Stories section
      sed -i '/## User Stories/,/## /{/## User Stories/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## User Stories/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("User Stories")
      ;;

    4)
      echo ""
      echo "📝 Editing Requirements"
      echo "======================="
      echo "Current content:"
      sed -n '/## Requirements/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Requirements content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Requirements section
      sed -i '/## Requirements/,/## /{/## Requirements/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Requirements/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Requirements")
      ;;

    5)
      echo ""
      echo "📝 Editing Success Criteria"
      echo "==========================="
      echo "Current content:"
      sed -n '/## Success Criteria/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Success Criteria content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Success Criteria section
      sed -i '/## Success Criteria/,/## /{/## Success Criteria/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Success Criteria/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Success Criteria")
      ;;

    6)
      echo ""
      echo "📝 Editing Constraints & Assumptions"
      echo "===================================="
      echo "Current content:"
      sed -n '/## Constraints & Assumptions/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Constraints & Assumptions content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Constraints & Assumptions section
      sed -i '/## Constraints & Assumptions/,/## /{/## Constraints & Assumptions/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Constraints & Assumptions/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Constraints & Assumptions")
      ;;

    7)
      echo ""
      echo "📝 Editing Out of Scope"
      echo "======================="
      echo "Current content:"
      sed -n '/## Out of Scope/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Out of Scope content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Out of Scope section
      sed -i '/## Out of Scope/,/## /{/## Out of Scope/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Out of Scope/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Out of Scope")
      ;;

    8)
      echo ""
      echo "📝 Editing Dependencies"
      echo "======================="
      echo "Current content:"
      sed -n '/## Dependencies/,/## /p' "$PRD_FILE" | head -n -1
      echo ""
      echo "Enter new Dependencies content (end with '---' on a new line):"
      CONTENT=""
      while IFS= read -r line; do
        if [ "$line" = "---" ]; then
          break
        fi
        CONTENT="$CONTENT$line"$'\n'
      done

      # Replace Dependencies section
      sed -i '/## Dependencies/,/## /{/## Dependencies/!{/## /!d}}' "$TEMP_FILE"
      sed -i '/## Dependencies/a\\n'"${CONTENT}" "$TEMP_FILE"
      EDITED_SECTIONS+=("Dependencies")
      ;;

    9)
      echo ""
      echo "📝 Custom Section"
      echo "================="
      read -p "Enter section name (e.g., 'Technical Architecture'): " SECTION_NAME
      if [ -n "$SECTION_NAME" ]; then
        echo "Enter content for '## $SECTION_NAME' (end with '---' on a new line):"
        CONTENT=""
        while IFS= read -r line; do
          if [ "$line" = "---" ]; then
            break
          fi
          CONTENT="$CONTENT$line"$'\n'
        done

        # Add custom section at the end before any closing sections
        echo "" >> "$TEMP_FILE"
        echo "## $SECTION_NAME" >> "$TEMP_FILE"
        echo "" >> "$TEMP_FILE"
        echo "$CONTENT" >> "$TEMP_FILE"
        EDITED_SECTIONS+=("$SECTION_NAME")
      fi
      ;;

    *)
      echo "❌ Invalid section number: $SECTION"
      ;;
  esac
done

if [ ${#EDITED_SECTIONS[@]} -eq 0 ]; then
  echo "❌ No valid sections were edited"
  rm "$TEMP_FILE"
  exit 1
fi

# Get current datetime for update
CURRENT_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Update the 'updated' field in frontmatter while preserving everything else
sed -i "s/^updated: .*/updated: $CURRENT_DATE/" "$TEMP_FILE"

# Replace original file with updated content
mv "$TEMP_FILE" "$PRD_FILE"

# Create sections list for output
SECTIONS_LIST=$(IFS=', '; echo "${EDITED_SECTIONS[*]}")

echo ""
echo "✅ Updated PRD: $FEATURE_NAME"
echo "  Sections edited: $SECTIONS_LIST"

# Check epic impact
if [ -n "$EPIC_NAME" ]; then
  echo ""
  echo "⚠️ This PRD has epic: $EPIC_NAME"
  read -p "Epic may need updating based on PRD changes. Review epic? (yes/no): " REVIEW_EPIC
  if [ "$REVIEW_EPIC" = "yes" ] || [ "$REVIEW_EPIC" = "y" ]; then
    echo ""
    echo "Review with: /pm:epic-edit $EPIC_NAME"
  fi
fi

echo ""
echo "Next: /pm:prd-parse $FEATURE_NAME to update epic"

exit 0