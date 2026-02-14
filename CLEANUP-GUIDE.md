# Quick Server Cleanup Guide

## Step 1: Run the automated cleanup script (Easiest)

```bash
./cleanup-server.sh
```

This will automatically SSH in and clean everything up.

---

## Step 2: Manual SSH (if you prefer to see what's happening)

### Connect to server:
```bash
ssh root@159.65.174.46
```

### Once connected, run these commands:

```bash
# 1. Check current disk space
df -h

# 2. Clean Docker (this frees the most space!)
docker system prune -af --volumes

# 3. Clean old session data (keeps last 10)
cd /opt/congruence/data/sessions
ls -t | tail -n +11 | xargs -r rm -rf

# 4. Clean Python cache
cd /opt/congruence
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 5. Check disk space again
df -h

# 6. Exit SSH
exit
```

---

## Step 3: Deploy with new ffmpeg fixes

```bash
./deploy-to-droplet.sh
```

This will now:
- ✅ Check disk space before deployment
- ✅ Auto-cleanup if needed
- ✅ Deploy your iPhone video fixes
- ✅ Start the server

---

## Monitoring (optional - run anytime)

```bash
./monitor-server.sh
```

Shows disk space, Docker status, sessions, memory, etc.

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `./cleanup-server.sh` | Clean up server disk space |
| `./monitor-server.sh` | Check server health |
| `./deploy-to-droplet.sh` | Deploy with auto-cleanup |
| `ssh root@159.65.174.46` | Manually connect to server |
