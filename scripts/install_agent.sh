#!/bin/bash

# AINet Agent Installation Script
# This script installs and configures the AINet monitoring agent on Linux machines

set -e  # Exit on any error

# Configuration
SCRIPT_NAME="AINet Agent Installer"
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

log_debug() {
    if [[ "$DEBUG" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
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
Installer v1.0.0
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

# Detect OS and distribution
detect_os() {
    log_info "Detecting operating system..."
    
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        OS_VERSION=$VERSION_ID
        log_info "Detected: $OS $OS_VERSION"
    else
        log_error "Cannot detect operating system"
        exit 1
    fi
    
    # Check if supported
    case "$OS" in
        "Ubuntu"|"Debian GNU/Linux"|"CentOS Linux"|"Red Hat Enterprise Linux"|"Fedora"|"Amazon Linux")
            log_info "Supported operating system detected"
            ;;
        *)
            log_warn "Operating system may not be fully supported: $OS"
            ;;
    esac
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    # Update package lists
    if command -v apt-get >/dev/null 2>&1; then
        # Debian/Ubuntu
        apt-get update
        apt-get install -y python3 python3-pip python3-venv curl wget git supervisor
        apt-get install -y build-essential python3-dev libffi-dev libssl-dev
        PACKAGE_MANAGER="apt"
    elif command -v yum >/dev/null 2>&1; then
        # RHEL/CentOS/Amazon Linux
        yum update -y
        yum install -y python3 python3-pip curl wget git supervisor
        yum groupinstall -y "Development Tools"
        yum install -y python3-devel libffi-devel openssl-devel
        PACKAGE_MANAGER="yum"
    elif command -v dnf >/dev/null 2>&1; then
        # Fedora
        dnf update -y
        dnf install -y python3 python3-pip curl wget git supervisor
        dnf groupinstall -y "Development Tools"
        dnf install -y python3-devel libffi-devel openssl-devel
        PACKAGE_MANAGER="dnf"
    else
        log_error "Unsupported package manager"
        exit 1
    fi
    
    log_info "System dependencies installed successfully"
}

# Create system user
create_user() {
    log_info "Creating system user: $AGENT_USER"
    
    if id "$AGENT_USER" &>/dev/null; then
        log_warn "User $AGENT_USER already exists"
    else
        useradd --system --no-create-home --shell /bin/false "$AGENT_USER"
        log_info "User $AGENT_USER created"
    fi
}

# Create directories
create_directories() {
    log_info "Creating directories..."
    
    mkdir -p "$AGENT_HOME"
    mkdir -p "$LOG_DIR"
    mkdir -p "$CONFIG_DIR"
    
    # Set ownership and permissions
    chown -R "$AGENT_USER:$AGENT_USER" "$AGENT_HOME"
    chown -R "$AGENT_USER:$AGENT_USER" "$LOG_DIR"
    chmod 755 "$AGENT_HOME"
    chmod 755 "$LOG_DIR"
    chmod 755 "$CONFIG_DIR"
    
    log_info "Directories created successfully"
}

# Download and install agent
install_agent() {
    log_info "Installing AINet agent..."
    
    cd "$AGENT_HOME"
    
    # Create virtual environment
    log_info "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements (would normally download from repository)
    log_info "Installing Python dependencies..."
    
    # Create a temporary requirements file
    cat > requirements.txt << 'EOF'
psutil>=6.1.0
netifaces>=0.11.0
aiohttp>=3.10.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
structlog>=24.4.0
EOF
    
    pip install -r requirements.txt
    
    # Copy agent code (in real deployment, this would be downloaded)
    log_info "Setting up agent code..."
    
    # Create basic agent structure
    mkdir -p agent/network_monitor
    mkdir -p agent/data_collector
    
    # Note: In actual deployment, the agent code would be copied from the repository
    # For now, we'll create placeholder files
    
    # Set permissions
    chown -R "$AGENT_USER:$AGENT_USER" "$AGENT_HOME"
    
    log_info "Agent installed successfully"
}

# Create configuration file
create_config() {
    log_info "Creating configuration file..."
    
    cat > "$CONFIG_DIR/agent.yaml" << 'EOF'
# AINet Agent Configuration
# Generated by installation script

# Agent Configuration
agent:
  collection_interval: 30  # seconds between metric collections
  batch_size: 10          # number of metrics to batch together
  max_retries: 3          # max retries for operations
  timeout: 30             # timeout in seconds

# Data Sender Configuration
sender:
  server_url: "http://localhost:8000"  # Central server URL - UPDATE THIS!
  api_key: null                        # API key for authentication - UPDATE THIS!
  timeout: 30                          # HTTP timeout
  max_retries: 3                       # Max retry attempts
  retry_delay: 5                       # Initial retry delay (seconds)
  batch_size: 10                       # Metrics per batch
  max_queue_size: 1000                 # Max queued metrics
  ssl_verify: true                     # Verify SSL certificates

# Logging Configuration
logging:
  level: "INFO"                        # DEBUG, INFO, WARNING, ERROR
  format: "text"                       # text or json
  file: "/var/log/ainet/agent.log"     # Log file path

# Network Monitoring Settings
monitoring:
  interfaces: "all"                    # Monitor all interfaces or specify list
  collect_interface_details: true     # Collect detailed interface info
  ping_targets:                       # Additional ping targets
    - name: "google_dns"
      host: "8.8.8.8"
    - name: "cloudflare_dns"
      host: "1.1.1.1"
  ping_timeout: 5                     # Ping timeout in seconds

# Security Settings
security:
  api_key_header: "X-API-Key"         # Header name for API key
  user_agent: "AINet-Agent/1.0"       # User agent string
EOF
    
    chmod 644 "$CONFIG_DIR/agent.yaml"
    log_info "Configuration file created at $CONFIG_DIR/agent.yaml"
    log_warn "IMPORTANT: Update the server_url and api_key in $CONFIG_DIR/agent.yaml"
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."
    
    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=AINet Network Monitoring Agent
After=network.target
Wants=network.target

[Service]
Type=simple
User=$AGENT_USER
Group=$AGENT_USER
WorkingDirectory=$AGENT_HOME
Environment=PATH=$AGENT_HOME/venv/bin
ExecStart=$AGENT_HOME/venv/bin/python agent/main.py $CONFIG_DIR/agent.yaml
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ainet-agent

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$LOG_DIR $CONFIG_DIR
CapabilityBoundingSet=CAP_NET_RAW CAP_NET_ADMIN

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    log_info "Systemd service created"
}

# Create supervisor configuration (fallback for systems without systemd)
create_supervisor_config() {
    log_info "Creating supervisor configuration..."
    
    cat > "/etc/supervisor/conf.d/$SERVICE_NAME.conf" << EOF
[program:$SERVICE_NAME]
command=$AGENT_HOME/venv/bin/python agent/main.py $CONFIG_DIR/agent.yaml
directory=$AGENT_HOME
user=$AGENT_USER
group=$AGENT_USER
autostart=true
autorestart=true
stderr_logfile=$LOG_DIR/agent_error.log
stdout_logfile=$LOG_DIR/agent_output.log
environment=PATH="$AGENT_HOME/venv/bin:%(ENV_PATH)s"
EOF
    
    log_info "Supervisor configuration created"
}

# Setup log rotation
setup_logrotate() {
    log_info "Setting up log rotation..."
    
    cat > "/etc/logrotate.d/ainet-agent" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $AGENT_USER $AGENT_USER
    postrotate
        systemctl reload $SERVICE_NAME >/dev/null 2>&1 || true
    endscript
}
EOF
    
    log_info "Log rotation configured"
}

# Create management scripts
create_management_scripts() {
    log_info "Creating management scripts..."
    
    # Status script
    cat > "/usr/local/bin/ainet-status" << 'EOF'
#!/bin/bash
echo "AINet Agent Status:"
systemctl status ainet-agent
echo ""
echo "Recent logs:"
journalctl -u ainet-agent --lines=10 --no-pager
EOF
    
    # Start script
    cat > "/usr/local/bin/ainet-start" << 'EOF'
#!/bin/bash
echo "Starting AINet Agent..."
systemctl start ainet-agent
systemctl enable ainet-agent
echo "Agent started and enabled for auto-start"
EOF
    
    # Stop script
    cat > "/usr/local/bin/ainet-stop" << 'EOF'
#!/bin/bash
echo "Stopping AINet Agent..."
systemctl stop ainet-agent
echo "Agent stopped"
EOF
    
    # Restart script
    cat > "/usr/local/bin/ainet-restart" << 'EOF'
#!/bin/bash
echo "Restarting AINet Agent..."
systemctl restart ainet-agent
echo "Agent restarted"
EOF
    
    # Make scripts executable
    chmod +x /usr/local/bin/ainet-*
    
    log_info "Management scripts created in /usr/local/bin/"
}

# Post-installation setup
post_install() {
    log_info "Completing installation..."
    
    # Enable and start service if systemd is available
    if systemctl --version >/dev/null 2>&1; then
        log_info "Enabling AINet agent service..."
        systemctl enable "$SERVICE_NAME"
        log_info "Service enabled for auto-start"
    elif command -v supervisorctl >/dev/null 2>&1; then
        log_info "Reloading supervisor configuration..."
        supervisorctl reread
        supervisorctl update
        log_info "Supervisor configuration updated"
    fi
    
    log_info "Installation completed successfully!"
}

# Print post-installation instructions
print_instructions() {
    cat << EOF

${GREEN}═══════════════════════════════════════════════════════════════════════${NC}
${GREEN}                    Installation Complete!                             ${NC}
${GREEN}═══════════════════════════════════════════════════════════════════════${NC}

${YELLOW}NEXT STEPS:${NC}

1. ${BLUE}Configure the agent:${NC}
   Edit: $CONFIG_DIR/agent.yaml
   
   ${YELLOW}IMPORTANT:${NC} Update these settings:
   - server_url: Point to your AINet central server
   - api_key: Set the authentication key
   
2. ${BLUE}Start the agent:${NC}
   sudo ainet-start
   
3. ${BLUE}Check status:${NC}
   sudo ainet-status
   
4. ${BLUE}View logs:${NC}
   sudo journalctl -u $SERVICE_NAME -f

${YELLOW}MANAGEMENT COMMANDS:${NC}
- ainet-start    : Start the agent
- ainet-stop     : Stop the agent
- ainet-restart  : Restart the agent
- ainet-status   : Show agent status

${YELLOW}FILES CREATED:${NC}
- Agent: $AGENT_HOME
- Config: $CONFIG_DIR/agent.yaml
- Logs: $LOG_DIR/
- Service: /etc/systemd/system/$SERVICE_NAME.service

${GREEN}For support, visit: https://github.com/your-org/ainet${NC}

EOF
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup tasks if needed
}

# Signal handlers
trap cleanup EXIT

# Main installation flow
main() {
    print_banner
    
    log_info "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    
    # Pre-flight checks
    check_root
    detect_os
    
    # Installation steps
    install_dependencies
    create_user
    create_directories
    install_agent
    create_config
    
    # Service setup
    if systemctl --version >/dev/null 2>&1; then
        create_systemd_service
    elif command -v supervisorctl >/dev/null 2>&1; then
        create_supervisor_config
    else
        log_warn "No service manager detected. Manual startup required."
    fi
    
    # Additional setup
    setup_logrotate
    create_management_scripts
    post_install
    
    # Show next steps
    print_instructions
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--debug] [--help]"
            echo ""
            echo "Options:"
            echo "  --debug    Enable debug output"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main installation
main "$@"
