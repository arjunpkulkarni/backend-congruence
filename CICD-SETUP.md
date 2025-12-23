# CI/CD Setup Guide

## GitHub Actions Workflows Created

✅ **Test Workflow** (`.github/workflows/test.yml`)
- Runs on every push and PR
- Tests Python imports
- Validates Docker build
- Runs linting checks

✅ **Deploy Workflow** (`.github/workflows/deploy.yml`)
- Deploys to DigitalOcean on push to main/master
- Automatically syncs code and restarts services
- Includes health checks

## Required GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

### 1. `DROPLET_SSH_KEY`
Your private SSH key for the droplet:
```bash
cat ~/.ssh/id_ed25519
```
Copy the entire output (including `-----BEGIN OPENSSH PRIVATE KEY-----`)

### 2. `DROPLET_IP`
```
159.65.174.46
```

### 3. `OPENAI_API_KEY`
Your OpenAI API key for the application

## Setup Steps

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add CI/CD workflows"
   git push origin main
   ```

2. **Add GitHub Secrets** (see above)

3. **Test the workflow**:
   - Make a small change to any file
   - Commit and push
   - Check GitHub Actions tab to see deployment

## Manual Trigger

You can also manually trigger deployment:
- Go to GitHub → Actions → Deploy to DigitalOcean → Run workflow

## What happens on each push:

1. **Test Phase**: Code is tested and validated
2. **Deploy Phase**: 
   - Code synced to droplet
   - Docker containers rebuilt
   - Application restarted
   - Health check performed

## Monitoring

- **GitHub Actions**: See deployment status and logs
- **Application Health**: http://159.65.174.46:8000/health
- **Server Logs**: SSH into droplet and run `docker compose logs -f`


