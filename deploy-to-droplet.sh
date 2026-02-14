#!/bin/bash

set -e

DROPLET_IP="159.65.174.46"
DROPLET_USER="root"
DEPLOY_PATH="/opt/congruence"

# Configurable cleanup settings
SESSION_RETENTION_DAYS="${SESSION_RETENTION_DAYS:-7}"  # Keep sessions from last 7 days
MIN_SESSIONS_TO_KEEP="${MIN_SESSIONS_TO_KEEP:-10}"     # Always keep at least 10 most recent
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"                      # Set to 1 to skip cleanup
MIN_DISK_SPACE_GB="${MIN_DISK_SPACE_GB:-2}"            # Minimum free space required (GB)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

echo "=== Setting up DigitalOcean Droplet ==="
echo ""

# Function to check disk space
check_disk_space() {
    print_step "Checking disk space on server..."
    
    DISK_INFO=$(ssh ${DROPLET_USER}@${DROPLET_IP} "df -BG / | tail -1")
    AVAILABLE_GB=$(echo "$DISK_INFO" | awk '{print $4}' | sed 's/G//')
    USED_PERCENT=$(echo "$DISK_INFO" | awk '{print $5}' | sed 's/%//')
    
    print_info "Disk usage: ${USED_PERCENT}% used, ${AVAILABLE_GB}GB available"
    
    if [ "$AVAILABLE_GB" -lt "$MIN_DISK_SPACE_GB" ]; then
        print_error "Insufficient disk space! Only ${AVAILABLE_GB}GB available (need ${MIN_DISK_SPACE_GB}GB minimum)"
        print_warning "Run cleanup: ./cleanup-server.sh"
        exit 1
    fi
    
    if [ "$USED_PERCENT" -gt 80 ]; then
        print_warning "Disk usage is at ${USED_PERCENT}% - cleanup recommended"
    fi
}

# Function to cleanup server
cleanup_server() {
    if [ "$SKIP_CLEANUP" = "1" ]; then
        print_warning "Skipping cleanup (SKIP_CLEANUP=1)"
        return 0
    fi
    
    print_step "Running automatic cleanup..."
    
    ssh ${DROPLET_USER}@${DROPLET_IP} << ENDSSH
set -e

CLEANUP_LOG="/opt/congruence/cleanup.log"
echo "=== Cleanup started at \$(date) ===" >> \$CLEANUP_LOG

# 1. Docker cleanup
echo "[Cleanup] Removing unused Docker images and containers..." | tee -a \$CLEANUP_LOG
DOCKER_BEFORE=\$(docker system df -v 2>/dev/null | grep "Images space usage" || echo "0B")
docker system prune -af --volumes >> \$CLEANUP_LOG 2>&1 || true
DOCKER_AFTER=\$(docker system df -v 2>/dev/null | grep "Images space usage" || echo "0B")
echo "[Cleanup] Docker cleanup complete" | tee -a \$CLEANUP_LOG

# 2. Session data cleanup (keep last N sessions OR last X days, whichever is more)
if [ -d "${DEPLOY_PATH}/data/sessions" ]; then
    echo "[Cleanup] Cleaning old session data (retention: ${SESSION_RETENTION_DAYS} days, min keep: ${MIN_SESSIONS_TO_KEEP})..." | tee -a \$CLEANUP_LOG
    
    cd ${DEPLOY_PATH}/data/sessions
    TOTAL_SESSIONS=\$(ls -1 | wc -l)
    
    if [ "\$TOTAL_SESSIONS" -gt "${MIN_SESSIONS_TO_KEEP}" ]; then
        # Delete sessions older than retention days, but keep at least MIN_SESSIONS_TO_KEEP
        SESSIONS_TO_DELETE=\$(find . -maxdepth 1 -type d -mtime +${SESSION_RETENTION_DAYS} | wc -l)
        SESSIONS_AFTER_DELETE=\$((\$TOTAL_SESSIONS - \$SESSIONS_TO_DELETE))
        
        if [ "\$SESSIONS_AFTER_DELETE" -ge "${MIN_SESSIONS_TO_KEEP}" ]; then
            find . -maxdepth 1 -type d -mtime +${SESSION_RETENTION_DAYS} -exec rm -rf {} + 2>/dev/null || true
            echo "[Cleanup] Deleted \$SESSIONS_TO_DELETE old sessions" | tee -a \$CLEANUP_LOG
        else
            # Keep most recent MIN_SESSIONS_TO_KEEP sessions
            ls -t | tail -n +\$(( ${MIN_SESSIONS_TO_KEEP} + 1 )) | xargs -r rm -rf 2>/dev/null || true
            echo "[Cleanup] Kept ${MIN_SESSIONS_TO_KEEP} most recent sessions" | tee -a \$CLEANUP_LOG
        fi
    else
        echo "[Cleanup] Only \$TOTAL_SESSIONS sessions found, keeping all" | tee -a \$CLEANUP_LOG
    fi
fi

# 3. Python cache cleanup
echo "[Cleanup] Removing Python cache files..." | tee -a \$CLEANUP_LOG
cd ${DEPLOY_PATH}
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 4. Final disk space report
DISK_AFTER=\$(df -h / | tail -1 | awk '{print \$4 " available (" \$5 " used)"}')
echo "[Cleanup] Disk space after cleanup: \$DISK_AFTER" | tee -a \$CLEANUP_LOG
echo "=== Cleanup completed at \$(date) ===" >> \$CLEANUP_LOG
echo "" >> \$CLEANUP_LOG

ENDSSH
    
    print_info "Cleanup complete! Check logs: ${DEPLOY_PATH}/cleanup.log"
}

# Step 0: Pre-deployment checks and cleanup
check_disk_space
cleanup_server

# Step 1: Setup server (install Docker, etc)
echo "Step 1: Installing Docker on droplet..."
ssh ${DROPLET_USER}@${DROPLET_IP} << 'ENDSSH'
set -e

# Install Docker
apt-get update
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Setup firewall
ufw allow OpenSSH
ufw allow 8000/tcp
ufw --force enable

# Create app directory
mkdir -p /opt/congruence

echo "Docker installed successfully!"
docker --version
ENDSSH

echo "Step 2: Syncing application code..."
rsync -avz --exclude 'venv/' \
           --exclude '__pycache__/' \
           --exclude 'data/sessions/*' \
           --exclude '.git/' \
           --exclude '*.pyc' \
           ./ ${DROPLET_USER}@${DROPLET_IP}:${DEPLOY_PATH}/

echo "Step 3: Setting up environment..."
ssh ${DROPLET_USER}@${DROPLET_IP} << ENDSSH
cd ${DEPLOY_PATH}

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "WARNING: Please update .env with your actual OpenAI API key"
fi

# Create data directories
mkdir -p data/sessions data/media

echo "Setup complete!"
ENDSSH

echo ""
echo "=== Deployment Complete! ==="
echo ""
print_info "Automatic cleanup is ENABLED"
print_info "  - Session retention: ${SESSION_RETENTION_DAYS} days (min ${MIN_SESSIONS_TO_KEEP} sessions kept)"
print_info "  - Cleanup logs: ${DEPLOY_PATH}/cleanup.log"
print_info "  - To skip cleanup: SKIP_CLEANUP=1 ./deploy-to-droplet.sh"
print_info "  - To change retention: SESSION_RETENTION_DAYS=30 ./deploy-to-droplet.sh"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key:"
echo "   ssh ${DROPLET_USER}@${DROPLET_IP}"
echo "   cd ${DEPLOY_PATH}"
echo "   nano .env  # Edit and add your OPENAI_API_KEY"
echo ""
echo "2. Start the application:"
echo "   ./deploy.sh docker"
echo ""
echo "Your API will be available at: http://${DROPLET_IP}:8000"
echo ""
print_warning "Monitor disk space: ssh ${DROPLET_USER}@${DROPLET_IP} 'df -h'"



