# FinLab Data Integration CLI

A comprehensive command-line interface for managing the FinLab data integration pipeline, providing tools for data validation, monitoring, and system control.

## Features

- **Pipeline Management**: Start, stop, and monitor the data pipeline
- **Data Validation**: Run comprehensive data quality checks
- **System Monitoring**: Monitor system health and performance metrics
- **Data Operations**: Manual data sync, querying, and cleanup
- **Configuration Management**: Manage authentication and settings
- **Batch Operations**: Schedule and run batch jobs
- **Troubleshooting**: Diagnostic tools and automated repair
- **WSL Compatible**: Optimized for Windows Subsystem for Linux

## Installation

### Quick Installation (WSL/Linux)

```bash
# Navigate to the project directory
cd /path/to/ML4T/new

# Run the installation script
./scripts/install.sh

# For WSL-specific setup with virtual environment
./scripts/setup-wsl.sh
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make CLI script executable
chmod +x scripts/finlab-cli

# Add to PATH or create symlink
ln -sf $(pwd)/scripts/finlab-cli /usr/local/bin/finlab-cli
```

### Windows Installation

```batch
REM Use the batch script for Windows
scripts\finlab-cli.bat --help
```

## Quick Start

1. **Initialize Configuration**
   ```bash
   finlab-cli config init --interactive
   ```

2. **Test Connection**
   ```bash
   finlab-cli config test
   ```

3. **Check System Status**
   ```bash
   finlab-cli pipeline status
   finlab-cli monitoring health
   ```

4. **Start the Pipeline**
   ```bash
   finlab-cli pipeline start
   ```

## Configuration

### Authentication Methods

The CLI supports two authentication methods:

#### Method 1: FinLab API Token
```bash
finlab-cli config set finlab.token YOUR_API_TOKEN
```

#### Method 2: Database Credentials
```bash
finlab-cli config set finlab.db.host YOUR_DB_HOST
finlab-cli config set finlab.db.username YOUR_USERNAME
finlab-cli config set finlab.db.password YOUR_PASSWORD
```

### Configuration File

Create a configuration file for persistent settings:

```bash
# Generate template
finlab-cli config template --file finlab-cli.yaml

# Edit the template with your settings
# Then use it
finlab-cli --config-file finlab-cli.yaml pipeline status
```

Example configuration file:
```yaml
# FinLab Authentication
finlab_token: "your_api_token_here"
finlab_db_host: "your_db_host"
finlab_db_port: 5432
finlab_db_name: "finlab"
finlab_db_username: "your_username"
finlab_db_password: "your_password"

# CLI Settings
default_batch_size: 1000
default_timeout: 300
default_output_format: "table"
log_dir: "logs"
```

## Command Reference

### Pipeline Management

```bash
# Check pipeline status
finlab-cli pipeline status

# Start the pipeline
finlab-cli pipeline start

# Start with specific mode and symbols
finlab-cli pipeline start --mode incremental --symbols 2330,2317

# Stop the pipeline
finlab-cli pipeline stop --graceful

# Restart the pipeline
finlab-cli pipeline restart

# View pipeline logs
finlab-cli pipeline logs --follow
finlab-cli pipeline logs --level error --component validation
```

### Data Validation

```bash
# Run comprehensive validation
finlab-cli validation run

# Validate specific datasets
finlab-cli validation run --dataset fundamental --dataset market

# Validate specific symbols
finlab-cli validation run --symbols 2330,2317,2454

# Validate date range
finlab-cli validation run --start-date 2023-01-01 --end-date 2023-12-31

# View validation history
finlab-cli validation history --days 30

# Validate specific symbol
finlab-cli validation symbol 2330 --days 30
```

### System Monitoring

```bash
# Check system health
finlab-cli monitoring health

# View recent alerts
finlab-cli monitoring alerts --hours 24 --level error

# Monitor performance metrics
finlab-cli monitoring metrics --hours 6

# Real-time monitoring
finlab-cli monitoring watch --interval 5

# Health check for specific component
finlab-cli monitoring check data_ingestion
```

### Data Operations

```bash
# Sync data manually
finlab-cli data sync --symbols 2330,2317 --days 30

# Sync specific datasets
finlab-cli data sync --datasets fundamental,market --mode full

# Query symbol data
finlab-cli data query 2330 --start-date 2023-01-01 --limit 100

# List available symbols
finlab-cli data symbols --market TSE --active-only

# List available datasets
finlab-cli data datasets

# Cleanup old data
finlab-cli data cleanup --days 365 --dry-run
```

### Configuration Management

```bash
# Show current configuration
finlab-cli config show

# Show configuration as JSON
finlab-cli config show --format json --show-sensitive

# Set configuration values
finlab-cli config set finlab.token YOUR_TOKEN
finlab-cli config set cli.batch_size 2000

# Get configuration value
finlab-cli config get finlab.db.host

# Test configuration
finlab-cli config test

# Initialize configuration interactively
finlab-cli config init --interactive

# Generate configuration template
finlab-cli config template --file config-template.yaml
```

### Batch Operations

```bash
# Run multiple commands in sequence
finlab-cli batch run "pipeline status" "monitoring health" "data symbols"

# Run commands in parallel
finlab-cli batch run --parallel "validation run --dataset fundamental" "data sync --symbols 2330"

# Run commands from script file
finlab-cli batch script commands.txt --parallel

# Schedule a command (Linux/WSL only)
finlab-cli batch schedule "0 2 * * *" "data sync --days 1" --name "daily_sync"

# List scheduled tasks
finlab-cli batch list-scheduled

# Remove scheduled tasks
finlab-cli batch remove-scheduled --pattern "daily_sync"
```

### Troubleshooting

```bash
# Run comprehensive diagnostics
finlab-cli troubleshoot diagnose

# Show recent error logs
finlab-cli troubleshoot logs --level error --lines 50

# Attempt system repair
finlab-cli troubleshoot repair --create-dirs --fix-permissions

# Get suggestions for fixing issues
finlab-cli troubleshoot suggest "Connection refused"
finlab-cli troubleshoot suggest --category authentication
```

## Output Formats

Most commands support multiple output formats:

```bash
# Table format (default)
finlab-cli pipeline status

# JSON format
finlab-cli pipeline status --format json

# Save output to file
finlab-cli validation run --output validation-report.json --format json
```

## Logging and Verbosity

Control logging output and verbosity:

```bash
# Increase verbosity
finlab-cli -v pipeline status      # Warning level
finlab-cli -vv pipeline status     # Info level
finlab-cli -vvv pipeline status    # Debug level

# Quiet mode (errors only)
finlab-cli -q pipeline status

# Save logs to file
finlab-cli --log-file finlab.log pipeline start
```

## WSL-Specific Features

When running in Windows Subsystem for Linux:

### Virtual Environment Setup
```bash
# The setup-wsl.sh script creates a Python virtual environment
source .venv/bin/activate  # Activate environment
finlab-cli --help          # CLI is available
```

### Windows Path Access
```bash
# Access Windows files and .env files
finlab-cli data query 2330 --output /mnt/c/Users/YourName/Documents/data.csv

# Windows environment variables work
export FINLAB_TOKEN=$(cat /mnt/c/Users/YourName/.env | grep FINLAB_TOKEN | cut -d= -f2)
```

### Scheduled Tasks
```bash
# Use Linux cron for scheduling
finlab-cli batch schedule "0 9 * * 1-5" "data sync --days 1"

# Check cron status
sudo service cron status
```

## Common Use Cases

### Daily Data Pipeline

```bash
#!/bin/bash
# daily-pipeline.sh

# Check system health
finlab-cli monitoring health || exit 1

# Sync recent data
finlab-cli data sync --days 2 --mode incremental

# Run validation
finlab-cli validation run --days 7

# Check for alerts
finlab-cli monitoring alerts --hours 24 --level warning
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly-maintenance.sh

# Run comprehensive validation
finlab-cli validation run --days 30

# Cleanup old data
finlab-cli data cleanup --days 90

# System diagnostics
finlab-cli troubleshoot diagnose --include-sensitive

# Generate status report
finlab-cli monitoring health --format json > weekly-health-report.json
```

### Batch Data Processing

```bash
#!/bin/bash
# batch-processing.sh

# Process multiple symbols in parallel
finlab-cli batch run --parallel \
  "data sync --symbols 2330 --days 30" \
  "data sync --symbols 2317 --days 30" \
  "data sync --symbols 2454 --days 30"

# Validate processed data
finlab-cli validation run --symbols 2330,2317,2454 --days 30
```

## Error Handling and Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   finlab-cli config test
   finlab-cli troubleshoot suggest --category authentication
   ```

2. **Connection Issues**
   ```bash
   finlab-cli troubleshoot diagnose
   finlab-cli troubleshoot suggest "Connection refused"
   ```

3. **Permission Issues**
   ```bash
   finlab-cli troubleshoot repair --fix-permissions --create-dirs
   ```

4. **Missing Dependencies**
   ```bash
   finlab-cli troubleshoot diagnose
   pip install -r requirements.txt
   ```

### Getting Help

```bash
# General help
finlab-cli --help

# Command-specific help
finlab-cli pipeline --help
finlab-cli validation run --help

# System information
finlab-cli info

# Comprehensive diagnostics
finlab-cli troubleshoot diagnose --include-sensitive
```

## Environment Variables

The CLI respects these environment variables:

```bash
# Authentication
export FINLAB_TOKEN="your_api_token"
export FINLAB_DB_HOST="your_db_host"
export FINLAB_DB_USERNAME="your_username"
export FINLAB_DB_PASSWORD="your_password"

# CLI Settings
export CLI_DEFAULT_BATCH_SIZE=1000
export CLI_DEFAULT_TIMEOUT=300
export CLI_LOG_DIR="logs"
export CLI_OUTPUT_FORMAT="table"
```

## Integration with Other Tools

### With Docker
```bash
# Run CLI in Docker container
docker run -it --rm \
  -v $(pwd):/workspace \
  -e FINLAB_TOKEN=$FINLAB_TOKEN \
  python:3.9 bash -c "cd /workspace && ./scripts/finlab-cli pipeline status"
```

### With GitHub Actions
```yaml
name: Daily Data Pipeline
on:
  schedule:
    - cron: '0 2 * * *'
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run pipeline
        env:
          FINLAB_TOKEN: ${{ secrets.FINLAB_TOKEN }}
        run: |
          ./scripts/finlab-cli pipeline start
          ./scripts/finlab-cli validation run --days 7
```

## Support and Contributing

For support, issues, or contributions:

1. Check the troubleshooting guide: `finlab-cli troubleshoot suggest`
2. Run diagnostics: `finlab-cli troubleshoot diagnose`
3. View system information: `finlab-cli info`
4. Check logs: `finlab-cli troubleshoot logs --level error`

## License

This project is part of the ML4T (Machine Learning for Trading) framework.