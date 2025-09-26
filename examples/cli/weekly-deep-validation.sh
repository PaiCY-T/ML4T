#!/bin/bash
# Weekly Deep Validation Script for FinLab Data Pipeline
# Performs comprehensive validation and generates detailed reports

set -e

# Configuration
REPORT_DIR="reports/weekly"
LOG_DIR="logs"
VALIDATION_DAYS=30
TOP_SYMBOLS=("2330" "2317" "2454" "2412" "3008" "2303" "2002" "2886" "2207" "2308")

# Create directories
mkdir -p "$REPORT_DIR" "$LOG_DIR"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

echo "Starting weekly deep validation - $(date)"

# 1. Comprehensive system validation
log "Running comprehensive data validation..."
finlab-cli validation run \
    --days $VALIDATION_DAYS \
    --severity warning \
    --output "$REPORT_DIR/comprehensive-validation-$(date '+%Y-%m-%d').json" \
    --format json

# 2. Symbol-specific validation for top stocks
log "Running symbol-specific validation for top Taiwan stocks..."
for symbol in "${TOP_SYMBOLS[@]}"; do
    log "Validating symbol: $symbol"
    finlab-cli validation symbol "$symbol" \
        --days $VALIDATION_DAYS \
        --format json \
        --output "$REPORT_DIR/symbol-$symbol-validation-$(date '+%Y-%m-%d').json" || true
done

# 3. Validation history analysis
log "Generating validation history report..."
finlab-cli validation history \
    --days 30 \
    --format json > "$REPORT_DIR/validation-history-$(date '+%Y-%m-%d').json"

# 4. Data consistency checks across datasets
log "Running cross-dataset consistency checks..."

# Create a batch command file for parallel dataset validation
cat > "$REPORT_DIR/dataset-validation-commands.txt" << EOF
validation run --dataset fundamental --days $VALIDATION_DAYS --output $REPORT_DIR/fundamental-validation-$(date '+%Y-%m-%d').json --format json
validation run --dataset market --days $VALIDATION_DAYS --output $REPORT_DIR/market-validation-$(date '+%Y-%m-%d').json --format json
validation run --dataset technical --days $VALIDATION_DAYS --output $REPORT_DIR/technical-validation-$(date '+%Y-%m-%d').json --format json
EOF

# Run dataset validations in parallel
finlab-cli batch script "$REPORT_DIR/dataset-validation-commands.txt" --parallel

# 5. Generate summary report
log "Generating weekly validation summary..."

SUMMARY_FILE="$REPORT_DIR/weekly-summary-$(date '+%Y-%m-%d').json"

# Calculate summary statistics
TOTAL_ISSUES=0
CRITICAL_ISSUES=0

# Count issues from comprehensive validation
if [ -f "$REPORT_DIR/comprehensive-validation-$(date '+%Y-%m-%d').json" ]; then
    TOTAL_ISSUES=$(jq '.issues | length' "$REPORT_DIR/comprehensive-validation-$(date '+%Y-%m-%d').json" 2>/dev/null || echo 0)
    CRITICAL_ISSUES=$(jq '.issues | map(select(.severity == "CRITICAL")) | length' "$REPORT_DIR/comprehensive-validation-$(date '+%Y-%m-%d').json" 2>/dev/null || echo 0)
fi

# Create summary report
cat > "$SUMMARY_FILE" << EOF
{
    "report_date": "$(date -I)",
    "validation_period_days": $VALIDATION_DAYS,
    "summary": {
        "total_issues": $TOTAL_ISSUES,
        "critical_issues": $CRITICAL_ISSUES,
        "symbols_validated": ${#TOP_SYMBOLS[@]},
        "datasets_validated": ["fundamental", "market", "technical"]
    },
    "reports_generated": {
        "comprehensive_validation": "$REPORT_DIR/comprehensive-validation-$(date '+%Y-%m-%d').json",
        "validation_history": "$REPORT_DIR/validation-history-$(date '+%Y-%m-%d').json",
        "symbol_reports": [
$(printf '            "%s",\n' "${TOP_SYMBOLS[@]/#/$REPORT_DIR/symbol-}" | sed 's/,$//' | sed 's/$/-validation-$(date +%Y-%m-%d).json/')
        ],
        "dataset_reports": [
            "$REPORT_DIR/fundamental-validation-$(date '+%Y-%m-%d').json",
            "$REPORT_DIR/market-validation-$(date '+%Y-%m-%d').json",
            "$REPORT_DIR/technical-validation-$(date '+%Y-%m-%d').json"
        ]
    },
    "recommendations": []
}
EOF

# Add recommendations based on findings
if [ "$CRITICAL_ISSUES" -gt 0 ]; then
    jq '.recommendations += ["Critical issues found - immediate attention required"]' "$SUMMARY_FILE" > "$SUMMARY_FILE.tmp" && mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"
fi

if [ "$TOTAL_ISSUES" -gt 100 ]; then
    jq '.recommendations += ["High number of validation issues - consider data source review"]' "$SUMMARY_FILE" > "$SUMMARY_FILE.tmp" && mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"
fi

# 6. Cleanup old weekly reports (keep last 12 weeks)
log "Cleaning up old weekly reports..."
find "$REPORT_DIR" -name "weekly-summary-*.json" -mtime +84 -delete
find "$REPORT_DIR" -name "*-validation-*.json" -mtime +84 -delete

log "Weekly deep validation completed. Summary: $SUMMARY_FILE"

# Display summary
echo "=== Weekly Validation Summary ==="
echo "Total Issues: $TOTAL_ISSUES"
echo "Critical Issues: $CRITICAL_ISSUES"
echo "Symbols Validated: ${#TOP_SYMBOLS[@]}"
echo "Reports Location: $REPORT_DIR"

if [ "$CRITICAL_ISSUES" -gt 0 ]; then
    echo "WARNING: Critical issues found - review required!"
    exit 1
fi