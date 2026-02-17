# Fly.io Deployment Guide

This guide walks you through deploying the AI Engineering Tutorial to Fly.io.

## Prerequisites

1. **Fly.io Account** - Sign up at [fly.io](https://fly.io/app/sign-up)
2. **Flyctl CLI** - Install the Fly.io command-line tool

### Install Flyctl

**Windows (PowerShell):**
```powershell
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

**macOS/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

**Via Homebrew (macOS/Linux):**
```bash
brew install flyctl
```

## Step-by-Step Deployment

### 1. Authenticate with Fly.io

```bash
flyctl auth login
```

This will open your browser for authentication.

### 2. Create Your Fly.io App

```bash
# Launch the app (interactive setup)
flyctl launch

# When prompted:
# - App name: Choose a unique name (e.g., "my-ai-tutorial")
# - Region: Select closest to your users (e.g., iad for USA East)
# - PostgreSQL: No (we use SQLite)
# - Redis: No (not needed)
# - Deploy now: No (we need to set secrets first)
```

This will create/update your `fly.toml` configuration file.

### 3. Set Secret Environment Variables

Generate a secure SECRET_KEY:
```bash
# Windows (PowerShell)
$secret = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | % {[char]$_})
echo $secret

# macOS/Linux
python -c "import secrets; print(secrets.token_hex(32))"
```

Set the secret on Fly.io:
```bash
flyctl secrets set SECRET_KEY="your-generated-secret-key-here"
```

If you want to set default API keys (optional):
```bash
flyctl secrets set \
  OPENAI_API_KEY="sk-..." \
  ANTHROPIC_API_KEY="sk-ant-..." \
  GOOGLE_API_KEY="..." \
  XAI_API_KEY="..."
```

**Note:** Users typically provide their own API keys via the UI, so this is optional.

### 4. Create Persistent Storage Volume

The app needs persistent storage for SQLite database and ChromaDB:

```bash
# Create a 1GB volume (free tier includes 3GB)
flyctl volumes create ai_tutorial_data \
  --region iad \
  --size 1
```

**Important:** Update `fly.toml` to ensure the volume mounts to `/app/data`.

### 5. Update Production Settings

Update `fly.toml` if needed:

```toml
# Ensure BASE_URL is set correctly
[env]
  BASE_URL = "https://your-app-name.fly.dev"
```

### 6. Deploy Your Application

```bash
# Deploy to Fly.io
flyctl deploy

# Monitor the deployment
flyctl logs
```

### 7. Open Your Application

```bash
# Open in browser
flyctl open

# Or visit manually
# https://your-app-name.fly.dev
```

## Post-Deployment

### Check Application Status

```bash
# View app status
flyctl status

# View real-time logs
flyctl logs

# SSH into the machine (for debugging)
flyctl ssh console
```

### Update Your Deployment

After making changes:

```bash
# Deploy updates
flyctl deploy

# Or automatic deployment via GitHub Actions (see CI/CD section)
```

### Scale Your Application

**Scale resources up:**
```bash
# Increase memory
flyctl scale memory 512

# Increase CPU
flyctl scale vm shared-cpu-2x

# Set minimum running instances
flyctl scale count 1
```

**Scale to zero (free tier):**
```bash
# Auto scale to zero when idle (already configured in fly.toml)
flyctl scale count 0 --max-per-region 1
```

### Monitor Costs

```bash
# View current usage
flyctl dashboard
```

**Free tier includes:**
- Up to 3 shared-cpu-1x VMs (256MB RAM)
- 3GB persistent storage
- 160GB outbound data transfer

### Custom Domain (Optional)

```bash
# Add your custom domain
flyctl certs add yourdomain.com

# Follow DNS instructions provided
```

## Troubleshooting

### Logs Not Showing

```bash
# Force log output
flyctl logs --app your-app-name
```

### App Not Starting

```bash
# Check machine status
flyctl status --app your-app-name

# Restart the app
flyctl apps restart your-app-name

# SSH into machine to debug
flyctl ssh console
```

### Health Check Failures

Ensure `/health` endpoint is accessible:
```bash
curl https://your-app-name.fly.dev/health
```

### Database Issues

If SQLite database gets corrupted:
```bash
# SSH into machine
flyctl ssh console

# Remove database (will reset user data)
rm /app/data/ai_tutorial.db

# Restart app
flyctl apps restart
```

## CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Fly.io

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: superfly/flyctl-actions/setup-flyctl@master
      
      - name: Deploy to Fly.io
        run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

Get your API token:
```bash
flyctl auth token
```

Add it to GitHub repository secrets as `FLY_API_TOKEN`.

## Environment-Specific Configuration

### Development
```bash
# Run locally with Docker
docker compose up
```

### Staging
```bash
# Create staging app
flyctl launch --name ai-tutorial-staging --copy-config

# Deploy to staging
flyctl deploy --app ai-tutorial-staging
```

### Production
```bash
# Deploy to production
flyctl deploy --app ai-tutorial
```

## Cost Optimization

1. **Use free tier limits:** Keep memory at 256MB, scale to zero
2. **Minimize storage:** Use 1GB volume, clean old data
3. **CDN for static assets:** Consider Cloudflare for `/static`
4. **Monitor usage:** Check dashboard regularly

## Backup Strategy

### Backup Database

```bash
# SSH into machine
flyctl ssh console

# Copy database
cp /app/data/ai_tutorial.db /tmp/backup.db

# Download backup
flyctl ssh sftp get /tmp/backup.db ./ai_tutorial_backup.db
```

### Automated Backups

Add to cron or GitHub Actions:
```bash
flyctl ssh console -C "cp /app/data/ai_tutorial.db /app/data/backup-$(date +%Y%m%d).db"
```

## Security Checklist

- [ ] SECRET_KEY is set and unique
- [ ] DEBUG=false in production
- [ ] HTTPS is enforced (force_https = true in fly.toml)
- [ ] Health check endpoint is working
- [ ] Volume permissions are correct
- [ ] API rate limiting is configured (if needed)

## Support

- **Fly.io Docs:** https://fly.io/docs/
- **Community Forum:** https://community.fly.io/
- **Status Page:** https://status.flyio.net/

## Quick Reference

```bash
# Common commands
flyctl status              # Check app status
flyctl logs               # View logs
flyctl deploy             # Deploy changes
flyctl ssh console        # SSH into machine
flyctl scale memory 512   # Scale memory
flyctl apps restart       # Restart app
flyctl dashboard          # Open web dashboard
```
