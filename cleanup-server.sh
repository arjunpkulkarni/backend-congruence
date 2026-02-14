#!/bin/bash

# Server disk space cleanup script
DROPLET_IP="159.65.174.46"
DROPLET_USER="root"

echo "=== Checking disk space on server ==="

ssh ${DROPLET_USER}@${DROPLET_IP} << 'ENDSSH'
set -e

echo "1. Current disk usage:"
df -h

echo ""
echo "2. Largest directories in /opt/congruence:"
du -sh /opt/congruence/* 2>/dev/null | sort -hr | head -10

echo ""
echo "3. Docker disk usage:"
docker system df

echo ""
echo "=== CLEANING UP ==="

echo "4. Removing old Docker images and containers..."
docker system prune -af --volumes

echo ""
echo "5. Cleaning old session data (keeping last 10 sessions)..."
cd /opt/congruence/data/sessions
ls -t | tail -n +11 | xargs -r rm -rf

echo ""
echo "6. Removing Python cache files..."
cd /opt/congruence
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "7. Final disk usage:"
df -h

echo ""
echo "=== CLEANUP COMPLETE ==="
ENDSSH

echo ""
echo "Run deployment again: ./deploy-to-droplet.sh"
