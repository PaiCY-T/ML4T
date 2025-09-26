#!/bin/bash
# WSL-specific setup script for FinLab CLI
# This script configures the CLI for optimal WSL usage

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}FinLab CLI WSL Setup${NC}"
echo "===================="
echo ""

# Check if we're in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be WSL${NC}"
    echo "This script is optimized for Windows Subsystem for Linux"
    echo ""
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
echo "WSL detected: $(grep -i microsoft /proc/version 2>/dev/null | cut -d' ' -f1-3 || echo 'Not detected')"
echo ""

# Install basic dependencies for WSL
echo -e "${BLUE}Installing WSL-specific dependencies...${NC}"

# Update package list
sudo apt-get update -y

# Install essential packages
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    cron

echo -e "${GREEN}✓ Basic dependencies installed${NC}"

# Setup Python virtual environment (recommended for WSL)
echo ""
echo -e "${BLUE}Setting up Python virtual environment...${NC}"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Created virtual environment${NC}"
else
    echo -e "${BLUE}Virtual environment already exists${NC}"
fi

# Activate virtual environment and install dependencies
source "$VENV_DIR/bin/activate"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
    echo -e "${GREEN}✓ Installed Python dependencies in virtual environment${NC}"
fi

# Create WSL-specific CLI wrapper
echo ""
echo -e "${BLUE}Creating WSL CLI wrapper...${NC}"

WSL_CLI_PATH="$SCRIPT_DIR/finlab-cli-wsl"
cat > "$WSL_CLI_PATH" << 'EOF'
#!/bin/bash
# WSL-specific FinLab CLI wrapper
# This script ensures proper Python environment activation

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Add project src to Python path
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Run the working CLI
python3 "$SCRIPT_DIR/finlab-cli-working" "$@"
EOF

chmod +x "$WSL_CLI_PATH"
echo -e "${GREEN}✓ Created WSL CLI wrapper${NC}"

# Setup cron service (needed for scheduled tasks)
echo ""
echo -e "${BLUE}Setting up cron service...${NC}"

# Start cron service
sudo service cron start

# Enable cron to start automatically
if command -v systemctl &> /dev/null; then
    sudo systemctl enable cron
    echo -e "${GREEN}✓ Cron service enabled${NC}"
else
    echo -e "${YELLOW}Note: You may need to start cron manually: sudo service cron start${NC}"
fi

# Create symlink in user's local bin
echo ""
echo -e "${BLUE}Creating user command aliases...${NC}"

# Create local bin directory if it doesn't exist
mkdir -p "$HOME/.local/bin"

# Create symlink
ln -sf "$WSL_CLI_PATH" "$HOME/.local/bin/finlab-cli"

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo -e "${GREEN}✓ Added ~/.local/bin to PATH in .bashrc${NC}"
    echo -e "${YELLOW}Please run 'source ~/.bashrc' or restart your terminal${NC}"
else
    echo -e "${BLUE}PATH already includes ~/.local/bin${NC}"
fi

# Setup Windows interop helpers
echo ""
echo -e "${BLUE}Setting up Windows interop helpers...${NC}"

# Create helper script for accessing Windows credentials
WINDOWS_HELPER="$SCRIPT_DIR/windows-helper.sh"
cat > "$WINDOWS_HELPER" << 'EOF'
#!/bin/bash
# Windows interop helper for FinLab CLI
# This script helps access Windows files and environment variables

# Function to get Windows environment variable
get_windows_env() {
    local var_name="$1"
    /mnt/c/Windows/System32/cmd.exe /c "echo %${var_name}%" 2>/dev/null | tr -d '\r'
}

# Function to find .env files in Windows directories
find_windows_env() {
    local search_path="${1:-/mnt/c/Users}"
    find "$search_path" -name ".env" -type f 2>/dev/null | head -5
}

# Function to convert Windows path to WSL path
win_to_wsl_path() {
    local win_path="$1"
    echo "$win_path" | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/mnt/\L\1|'
}

# Function to convert WSL path to Windows path
wsl_to_win_path() {
    local wsl_path="$1"
    echo "$wsl_path" | sed 's|^/mnt/\([a-z]\)/|\U\1:/|' | sed 's|/|\\|g'
}

# Export functions for use by CLI
export -f get_windows_env find_windows_env win_to_wsl_path wsl_to_win_path
EOF

chmod +x "$WINDOWS_HELPER"
echo -e "${GREEN}✓ Created Windows interop helper${NC}"

# Create default configuration for WSL
echo ""
echo -e "${BLUE}Creating WSL-optimized configuration...${NC}"

CONFIG_DIR="$HOME/.config/finlab-cli"
mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    # Run the CLI to create template (using full path to avoid PATH issues)
    "$WSL_CLI_PATH" config template --file "$CONFIG_DIR/config.yaml"
    echo -e "${GREEN}✓ Created configuration template${NC}"

    # Add WSL-specific notes to config
    cat >> "$CONFIG_DIR/config.yaml" << 'EOF'

# WSL-specific notes:
# - You can use Windows paths like /mnt/c/Users/YourName/Documents/
# - .env files in Windows directories work fine
# - Use the windows-helper.sh script for advanced Windows interop
# - Scheduled tasks use Linux cron, not Windows Task Scheduler
EOF

    echo -e "${YELLOW}Please edit $CONFIG_DIR/config.yaml with your FinLab credentials${NC}"
fi

# Test installation
echo ""
echo -e "${BLUE}Testing WSL installation...${NC}"

# Test basic functionality
if "$WSL_CLI_PATH" --help &> /dev/null; then
    echo -e "${GREEN}✓ CLI installation successful${NC}"
else
    echo -e "${RED}✗ CLI installation failed${NC}"
    exit 1
fi

# Test cron
if crontab -l &> /dev/null || [ $? -eq 1 ]; then  # Exit code 1 means empty crontab, which is OK
    echo -e "${GREEN}✓ Cron service accessible${NC}"
else
    echo -e "${YELLOW}Warning: Cron service may not be properly configured${NC}"
fi

echo ""
echo -e "${GREEN}WSL setup completed successfully!${NC}"
echo ""
echo "WSL-specific features available:"
echo "- Virtual environment: $VENV_DIR"
echo "- CLI command: finlab-cli (after sourcing ~/.bashrc)"
echo "- Configuration: $CONFIG_DIR/config.yaml"
echo "- Windows helper: $WINDOWS_HELPER"
echo "- Cron scheduling: Use 'finlab-cli batch schedule' commands"
echo ""
echo "Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Edit configuration: finlab-cli config init --interactive"
echo "3. Test connection: finlab-cli config test"
echo "4. Start using: finlab-cli pipeline status"
echo ""
echo "WSL Tips:"
echo "- Access Windows files via /mnt/c/..."
echo "- Windows .env files work fine for authentication"
echo "- Use Linux cron for scheduling, not Windows Task Scheduler"
echo "- Virtual environment keeps dependencies isolated"