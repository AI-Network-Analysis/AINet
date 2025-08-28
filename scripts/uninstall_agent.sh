#!/bin/bash

# AINet Agent Uninstallation Script
# This script removes the AINet monitoring agent from Linux machines

set -e  # Exit on any error

# Configuration
SCRIPT_NAME="AINet Agent Uninstaller"
SCRIPT_VERSION="1.0.0"
AGENT_USER="ainet"
AGENT_HOME="/opt/ainet"
SERVICE_NAME="ainet-agent"
LOG_DIR="/var/log/ainet"
CONFIG_DIR="/etc/ainet"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    cat << "EOF"
    ___    ____     __     __
   /   |  /  _/    / /   / /
  / /| |  / /     / /   / /
 / ___ |_/ /     / /   / /___
/_/  |_/___/____/_/   /_____/
           /___/              
                             
AINet Network Monitoring Agent
Uninstaller v1.0.0
EOF
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Confirm uninstallation
confirm_uninstall() {
    echo -e "${YELLOW}WARNING: This will completely remove the AINet agent and all its data.${NC}"
    echo ""
    read -p "Are you sure you want to uninstall AINet agent? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Uninstallation cancelled."
        exit 0
    fi
}

# Stop and disable service
stop_service() {
    log_info "Stopping AINet agent service..."
    
    # Stop systemd service
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl stop "$SERVICE_NAME"
        log_info "Service stopped"
    fi
    
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl disable "$SERVICE_NAME"
        log_info "Service disabled"
    fi
    
    # Stop supervisor service
    if command -v supervisorctl >/dev/null 2>&1; then
        if supervisorctl status "$SERVICE_NAME" >/dev/null 2>&1; then
            supervisorctl stop "$SERVICE_NAME"
            log_info "Supervisor service stopped"
        fi
    fi
}

# Remove service files
remove_services() {
    log_info "Removing service files..."
    
    # Remove systemd service
    if [[ -f "/etc/systemd/system/$SERVICE_NAME.service" ]]; then
        rm -f "/etc/systemd/system/$SERVICE_NAME.service"
        systemctl daemon-reload
        log_info "Systemd service file removed"
    fi
    
    # Remove supervisor configuration
    if [[ -f "/etc/supervisor/conf.d/$SERVICE_NAME.conf" ]]; then
        rm -f "/etc/supervisor/conf.d/$SERVICE_NAME.conf"
        if command -v supervisorctl >/dev/null 2>&1; then
            supervisorctl reread
            supervisorctl update
        fi
        log_info "Supervisor configuration removed"
    fi
}

# Remove management scripts
remove_management_scripts() {
    log_info "Removing management scripts..."
    
    rm -f /usr/local/bin/ainet-*
    log_info "Management scripts removed"
}

# Remove log rotation
remove_logrotate() {
    log_info "Removing log rotation configuration..."
    
    if [[ -f "/etc/logrotate.d/ainet-agent" ]]; then
        rm -f "/etc/logrotate.d/ainet-agent"
        log_info "Log rotation configuration removed"
    fi
}

# Remove directories and files
remove_files() {
    log_info "Removing application files and directories..."
    
    # Remove agent home directory
    if [[ -d "$AGENT_HOME" ]]; then
        rm -rf "$AGENT_HOME"
        log_info "Agent home directory removed: $AGENT_HOME"
    fi
    
    # Remove configuration directory
    if [[ -d "$CONFIG_DIR" ]]; then
        rm -rf "$CONFIG_DIR"
        log_info "Configuration directory removed: $CONFIG_DIR"
    fi
    
    # Optionally remove logs (ask user)
    if [[ -d "$LOG_DIR" ]]; then
        read -p "Remove log files? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$LOG_DIR"
            log_info "Log directory removed: $LOG_DIR"
        else
            log_info "Log directory preserved: $LOG_DIR"
        fi
    fi
}

# Remove system user
remove_user() {
    log_info "Removing system user..."
    
    if id "$AGENT_USER" &>/dev/null; then
        # Kill any remaining processes owned by the user
        pkill -u "$AGENT_USER" || true
        
        # Remove user
        userdel "$AGENT_USER" 2>/dev/null || true
        log_info "User $AGENT_USER removed"
    else
        log_info "User $AGENT_USER does not exist"
    fi
}

# Clean up package dependencies (optional)
cleanup_dependencies() {
    read -p "Remove Python packages installed for AINet? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Note: Python packages were installed in virtual environment and have been removed with the agent directory"
    fi
}

# Print completion message
print_completion() {
    cat << EOF

${GREEN}═══════════════════════════════════════════════════════════════════════${NC}
${GREEN}                    Uninstallation Complete!                           ${NC}
${GREEN}═══════════════════════════════════════════════════════════════════════${NC}

${YELLOW}REMOVED COMPONENTS:${NC}
- AINet agent service
- Agent files and directories
- Configuration files
- Management scripts
- Service configurations
- Log rotation setup

${BLUE}PRESERVED (if selected):${NC}
- Log files in $LOG_DIR
- System packages (Python, dependencies)

${GREEN}The AINet agent has been completely removed from this system.${NC}

EOF
}

# Main uninstallation flow
main() {
    print_banner
    
    log_info "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    
    # Pre-flight checks
    check_root
    confirm_uninstall
    
    # Uninstallation steps
    stop_service
    remove_services
    remove_management_scripts
    remove_logrotate
    remove_files
    remove_user
    cleanup_dependencies
    
    # Completion
    print_completion
    
    log_info "Uninstallation completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--force] [--help]"
            echo ""
            echo "Options:"
            echo "  --force    Skip confirmation prompts"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Override confirmation if force flag is set
if [[ "$FORCE" == "1" ]]; then
    confirm_uninstall() {
        log_warn "Force mode: skipping confirmation"
    }
fi

# Run main uninstallation
main "$@"
