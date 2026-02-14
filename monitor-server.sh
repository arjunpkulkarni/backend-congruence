#!/bin/bash

# Server monitoring script - check disk space and resource usage
DROPLET_IP="159.65.174.46"
DROPLET_USER="root"
DEPLOY_PATH="/opt/congruence"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

echo ""
print_header "Server Status Monitor"
echo "Server: ${DROPLET_IP}"
echo "Time: $(date)"
echo ""

ssh ${DROPLET_USER}@${DROPLET_IP} << 'ENDSSH'

# Colors for remote output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Disk Space
echo -e "${GREEN}[1] Disk Space:${NC}"
df -h / | tail -1 | awk '{
    used_pct = substr($5, 1, length($5)-1);
    if (used_pct > 90) 
        printf "\033[0;31m  WARNING: %s used (%s available)\033[0m\n", $5, $4;
    else if (used_pct > 80) 
        printf "\033[1;33m  CAUTION: %s used (%s available)\033[0m\n", $5, $4;
    else 
        printf "\033[0;32m  OK: %s used (%s available)\033[0m\n", $5, $4;
}'
echo ""

# 2. Docker Status
echo -e "${GREEN}[2] Docker Containers:${NC}"
if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | tail -n +2 | grep -q .; then
    docker ps --format "  {{.Names}}: {{.Status}}"
else
    echo -e "${YELLOW}  No running containers${NC}"
fi
echo ""

# 3. Docker Disk Usage
echo -e "${GREEN}[3] Docker Disk Usage:${NC}"
docker system df --format "  {{.Type}}: {{.Size}} ({{.Reclaimable}} reclaimable)"
echo ""

# 4. Session Data Size
echo -e "${GREEN}[4] Session Data:${NC}"
if [ -d "/opt/congruence/data/sessions" ]; then
    SESSION_COUNT=$(ls -1 /opt/congruence/data/sessions 2>/dev/null | wc -l)
    SESSION_SIZE=$(du -sh /opt/congruence/data/sessions 2>/dev/null | awk '{print $1}')
    echo "  Sessions: $SESSION_COUNT (total size: $SESSION_SIZE)"
    
    # Show oldest and newest
    if [ $SESSION_COUNT -gt 0 ]; then
        OLDEST=$(ls -t /opt/congruence/data/sessions | tail -1)
        NEWEST=$(ls -t /opt/congruence/data/sessions | head -1)
        OLDEST_DATE=$(stat -c %y /opt/congruence/data/sessions/$OLDEST 2>/dev/null | cut -d' ' -f1)
        NEWEST_DATE=$(stat -c %y /opt/congruence/data/sessions/$NEWEST 2>/dev/null | cut -d' ' -f1)
        echo "  Oldest: $OLDEST_DATE"
        echo "  Newest: $NEWEST_DATE"
    fi
else
    echo "  No session data found"
fi
echo ""

# 5. Memory Usage
echo -e "${GREEN}[5] Memory Usage:${NC}"
free -h | grep Mem | awk '{
    used_pct = ($3/$2)*100;
    if (used_pct > 90)
        printf "\033[0;31m  WARNING: %.0f%% used (%s / %s)\033[0m\n", used_pct, $3, $2;
    else if (used_pct > 80)
        printf "\033[1;33m  CAUTION: %.0f%% used (%s / %s)\033[0m\n", used_pct, $3, $2;
    else
        printf "\033[0;32m  OK: %.0f%% used (%s / %s)\033[0m\n", used_pct, $3, $2;
}'
echo ""

# 6. API Status
echo -e "${GREEN}[6] API Health Check:${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}  API is responding ✓${NC}"
else
    echo -e "${RED}  API is not responding ✗${NC}"
fi
echo ""

# 7. Recent cleanup log
if [ -f "/opt/congruence/cleanup.log" ]; then
    echo -e "${GREEN}[7] Last Cleanup:${NC}"
    tail -3 /opt/congruence/cleanup.log | grep "Cleanup" | tail -1 || echo "  No cleanup records"
else
    echo -e "${GREEN}[7] Cleanup Log:${NC}"
    echo "  No cleanup log found"
fi

ENDSSH

echo ""
print_header "Recommendations"

# Get disk usage percentage
DISK_USAGE=$(ssh ${DROPLET_USER}@${DROPLET_IP} "df / | tail -1 | awk '{print \$5}' | sed 's/%//'")

if [ "$DISK_USAGE" -gt 90 ]; then
    print_error "CRITICAL: Disk is ${DISK_USAGE}% full!"
    echo "  Run: ./cleanup-server.sh"
elif [ "$DISK_USAGE" -gt 80 ]; then
    print_warning "Disk is ${DISK_USAGE}% full - cleanup recommended soon"
    echo "  Run: ./cleanup-server.sh"
else
    print_info "System health looks good!"
fi

echo ""
