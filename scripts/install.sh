#!/bin/bash
# FinLab CLI Installation Script for WSL/Linux
# This script sets up the FinLab CLI for easy command-line access

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}FinLab CLI Installation Script${NC}"
echo "=================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python $PYTHON_VERSION"

# Check if version is >= 3.8
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${GREEN}✓ Python version is compatible${NC}"
else
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
fi

# Install Python dependencies
echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    python3 -m pip install -r "$PROJECT_DIR/requirements.txt"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found${NC}"
fi

# Make CLI script executable
echo ""
echo -e "${BLUE}Setting up CLI executable...${NC}"
chmod +x "$SCRIPT_DIR/finlab-cli"
echo -e "${GREEN}✓ CLI script is executable${NC}"

# Create symlink in /usr/local/bin if possible
echo ""
echo -e "${BLUE}Setting up system-wide access...${NC}"

if [ -w "/usr/local/bin" ]; then
    ln -sf "$SCRIPT_DIR/finlab-cli" "/usr/local/bin/finlab-cli"
    echo -e "${GREEN}✓ Created symlink in /usr/local/bin${NC}"
    echo "You can now run 'finlab-cli' from anywhere"
else
    echo -e "${YELLOW}Warning: Cannot write to /usr/local/bin${NC}"
    echo "To enable system-wide access, run:"
    echo "  sudo ln -sf $SCRIPT_DIR/finlab-cli /usr/local/bin/finlab-cli"
    echo ""
    echo "Or add this to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"$SCRIPT_DIR:\$PATH\""
fi

# Test installation
echo ""
echo -e "${BLUE}Testing installation...${NC}"
if "$SCRIPT_DIR/finlab-cli" --help &> /dev/null; then
    echo -e "${GREEN}✓ CLI installation successful${NC}"
else
    echo -e "${RED}✗ CLI installation failed${NC}"
    echo "Please check the error messages above and try again"
    exit 1
fi

# Create default configuration
echo ""
echo -e "${BLUE}Creating default configuration...${NC}"
CONFIG_DIR="$HOME/.config/finlab-cli"
mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    "$SCRIPT_DIR/finlab-cli" config template --file "$CONFIG_DIR/config.yaml"
    echo -e "${GREEN}✓ Created configuration template at $CONFIG_DIR/config.yaml${NC}"
    echo -e "${YELLOW}Please edit $CONFIG_DIR/config.yaml with your FinLab credentials${NC}"
else
    echo -e "${BLUE}Configuration file already exists${NC}"
fi

echo ""
echo -e "${GREEN}Installation completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit your configuration: $CONFIG_DIR/config.yaml"
echo "2. Test the connection: finlab-cli config test"
echo "3. Check pipeline status: finlab-cli pipeline status"
echo "4. View help: finlab-cli --help"
echo ""
echo "For WSL users:"
echo "- The CLI is now available in your Linux environment"
echo "- You can access Windows files through /mnt/c/..."
echo "- Authentication via .env files in Windows directories works fine"