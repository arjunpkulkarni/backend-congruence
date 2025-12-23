#!/bin/bash

set -e

DROPLET_IP="159.65.174.46"
DROPLET_USER="root"
DEPLOY_PATH="/opt/congruence"

echo "=== Setting up DigitalOcean Droplet ==="

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



