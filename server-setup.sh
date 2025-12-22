#!/bin/bash

set -e

echo "=== Installing Docker ==="
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

echo "=== Docker installed successfully ==="
docker --version
docker compose version

echo "=== Setting up firewall ==="
ufw allow OpenSSH
ufw allow 8000/tcp
ufw --force enable

echo "=== Creating app directory ==="
mkdir -p /opt/congruence
cd /opt/congruence

echo "=== Setup complete! ==="
echo "Next steps:"
echo "1. Upload your application code to /opt/congruence"
echo "2. Create .env file with OPENAI_API_KEY"
echo "3. Run: ./deploy.sh docker"

