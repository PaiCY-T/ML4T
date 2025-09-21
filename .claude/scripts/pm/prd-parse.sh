#!/bin/bash

prd_name="$1"

if [ -z "$prd_name" ]; then
  echo "❌ Please provide a PRD name"
  echo "Usage: /pm:prd-parse <prd-name>"
  exit 1
fi

echo "Parsing PRD into epic..."
echo ""
echo ""

prd_file=".claude/prds/$prd_name.md"
epic_dir=".claude/epics/$prd_name"
epic_file="$epic_dir/epic.md"

# Validate PRD exists
if [ ! -f "$prd_file" ]; then
  echo "❌ PRD not found: $prd_name"
  echo ""
  echo "Available PRDs:"
  for file in .claude/prds/*.md; do
    [ -f "$file" ] && echo "  • $(basename "$file" .md)"
  done
  exit 1
fi

# Check if epic already exists
if [ -f "$epic_file" ]; then
  echo "⚠️  Epic '$prd_name' already exists. Do you want to overwrite it? (yes/no)"
  read -r response
  if [ "$response" != "yes" ] && [ "$response" != "y" ]; then
    echo "❌ Operation cancelled"
    exit 1
  fi
  echo "🔄 Overwriting existing epic..."
fi

# Create epic directory
mkdir -p "$epic_dir"

# Extract PRD metadata
prd_description=$(grep "^description:" "$prd_file" | head -1 | sed 's/^description: *//')
prd_created=$(grep "^created:" "$prd_file" | head -1 | sed 's/^created: *//')

# Get current datetime for epic creation
epic_created=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Extract key sections from PRD
executive_summary=$(sed -n '/^## Executive Summary/,/^## /p' "$prd_file" | sed '$d' | tail -n +2)
problem_statement=$(sed -n '/^## Problem Statement/,/^## /p' "$prd_file" | sed '$d' | tail -n +2)
functional_reqs=$(sed -n '/^#### Core Trading Engine/,/^#### /p' "$prd_file" | sed '$d' | tail -n +2)
success_criteria=$(sed -n '/^## Success Criteria/,/^## /p' "$prd_file" | sed '$d' | tail -n +2)
phases=$(sed -n '/^## Implementation Phases/,/^## /p' "$prd_file" | sed '$d' | tail -n +2)

# Create epic.md file
cat > "$epic_file" << EOF
---
name: $prd_name
description: $prd_description
status: backlog
created: $epic_created
prd_source: $prd_name
progress: 0
total_issues: 0
completed_issues: 0
---

# Epic: $prd_name

## Overview
$executive_summary

## Problem Context
$problem_statement

## Core Requirements
$functional_reqs

## Success Metrics
$success_criteria

## Implementation Roadmap
$phases

## Task Breakdown
This epic will be decomposed into specific implementation tasks using \`/pm:epic-decompose $prd_name\`.

## Dependencies
- PRD: $prd_name.md
- Implementation phases as defined in PRD
- Technical stack: Python 3.11, FinLab, Fubon API

## Acceptance Criteria
- [ ] All core trading engine components implemented
- [ ] Multi-strategy framework operational
- [ ] Risk management system active
- [ ] Fubon API integration complete
- [ ] Backtesting validation passed
- [ ] Live trading deployment ready

## Notes
Epic auto-generated from PRD: $prd_name.md on $epic_created
EOF

# Update PRD status to "in_epic"
if grep -q "^status:" "$prd_file"; then
  sed -i 's/^status:.*/status: in_epic/' "$prd_file"
else
  echo "status: in_epic" >> "$prd_file"
fi

echo "✅ Epic created successfully!"
echo ""
echo "📁 Epic location: $epic_dir/epic.md"
echo "📋 PRD status updated to: in_epic"
echo ""
echo "🎯 Next steps:"
echo "  1. Review epic: /pm:epic-show $prd_name"
echo "  2. Break into tasks: /pm:epic-decompose $prd_name"
echo "  3. Sync to GitHub: /pm:epic-sync $prd_name"
echo ""