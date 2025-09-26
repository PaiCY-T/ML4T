#!/bin/bash
# Daily Maintenance Script for FinLab Data Pipeline
# This script performs daily data synchronization and health checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="logs"
REPORT_DIR="reports"
MAX_RETRIES=3
SYNC_DAYS=2

# Create directories if they don't exist
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to run command with retries
run_with_retry() {
    local command="$1"
    local description="$2"
    local attempt=1

    log "Starting: $description"

    while [ $attempt -le $MAX_RETRIES ]; do
        if eval "$command"; then
            success "$description completed successfully"
            return 0
        else
            warning "$description failed (attempt $attempt/$MAX_RETRIES)"
            if [ $attempt -eq $MAX_RETRIES ]; then
                error "$description failed after $MAX_RETRIES attempts"
                return 1
            fi
            attempt=$((attempt + 1))
            sleep 10  # Wait before retry
        fi
    done
}

# Main execution
main() {
    log "Starting daily maintenance for FinLab Data Pipeline"

    # Step 1: System health check
    log "=== System Health Check ==="
    if ! finlab-cli monitoring health > "$LOG_DIR/health-check.log" 2>&1; then
        error "System health check failed - aborting maintenance"
        exit 1
    fi
    success "System health check passed"

    # Step 2: Check for critical alerts
    log "=== Checking for Critical Alerts ==="
    if finlab-cli monitoring alerts --hours 24 --level critical --format json > "$REPORT_DIR/critical-alerts.json"; then
        local critical_count=$(jq '.alerts | length' "$REPORT_DIR/critical-alerts.json" 2>/dev/null || echo "0")
        if [ "$critical_count" -gt 0 ]; then
            warning "Found $critical_count critical alerts - review required"
        else
            success "No critical alerts found"
        fi
    fi

    # Step 3: Data synchronization
    log "=== Data Synchronization ==="
    run_with_retry "finlab-cli data sync --days $SYNC_DAYS --mode incremental" "Data synchronization"

    # Step 4: Data validation
    log "=== Data Validation ==="
    if finlab-cli validation run --days 7 --output "$REPORT_DIR/validation-report.json" --format json; then
        success "Data validation completed"

        # Check validation results
        local error_count=$(jq '.issues | length' "$REPORT_DIR/validation-report.json" 2>/dev/null || echo "0")
        if [ "$error_count" -gt 0 ]; then
            warning "Found $error_count validation issues - review required"
        else
            success "No validation issues found"
        fi
    else
        error "Data validation failed"
    fi

    # Step 5: Performance metrics collection
    log "=== Collecting Performance Metrics ==="
    finlab-cli monitoring metrics --hours 24 --format json > "$REPORT_DIR/performance-metrics.json"

    # Step 6: Generate daily summary
    log "=== Generating Daily Summary ==="
    local summary_file="$REPORT_DIR/daily-summary-$(date '+%Y-%m-%d').json"

    cat > "$summary_file" << EOF
{
    "date": "$(date -I)",
    "timestamp": "$(date -Iseconds)",
    "maintenance_status": "completed",
    "health_check": "$([ -f "$LOG_DIR/health-check.log" ] && echo "passed" || echo "failed")",
    "sync_status": "completed",
    "validation_issues": $(jq '.issues | length' "$REPORT_DIR/validation-report.json" 2>/dev/null || echo "null"),
    "critical_alerts": $(jq '.alerts | length' "$REPORT_DIR/critical-alerts.json" 2>/dev/null || echo "null"),
    "reports_generated": [
        "$REPORT_DIR/validation-report.json",
        "$REPORT_DIR/critical-alerts.json",
        "$REPORT_DIR/performance-metrics.json"
    ]
}
EOF

    success "Daily summary saved to $summary_file"

    # Step 7: Cleanup old reports (keep last 30 days)
    log "=== Cleaning up old reports ==="
    find "$REPORT_DIR" -name "daily-summary-*.json" -mtime +30 -delete
    find "$LOG_DIR" -name "*.log" -mtime +7 -delete

    log "Daily maintenance completed successfully"
}

# Error handling
trap 'error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"