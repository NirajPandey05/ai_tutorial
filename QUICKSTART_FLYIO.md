# 🚀 Quick Deploy to Fly.io

Deploy your AI Engineering Tutorial to Fly.io in less than 5 minutes!

## Prerequisites

- ✅ Fly.io account ([Sign up free](https://fly.io/app/sign-up))
- ✅ Flyctl CLI installed

## Install Flyctl

**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**macOS/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

## Deploy (3 Steps)

### 1️⃣ Authenticate
```bash
flyctl auth login
```

### 2️⃣ Setup (One-time)
**Windows:**
```powershell
pwsh scripts/deploy-flyio.ps1 setup
```

**macOS/Linux:**
```bash
bash scripts/deploy-flyio.sh setup
```

This will:
- ✅ Create your Fly.io app
- ✅ Generate a secure SECRET_KEY
- ✅ Create persistent storage volume
- ✅ Configure environment

### 3️⃣ Deploy
**Windows:**
```powershell
pwsh scripts/deploy-flyio.ps1 deploy
```

**macOS/Linux:**
```bash
bash scripts/deploy-flyio.sh deploy
```

That's it! 🎉 Your app will open automatically.

## Configuration

### Set API Keys (Optional)

Users typically provide their own API keys via the UI, but you can set defaults:

**Windows:**
```powershell
pwsh scripts/deploy-flyio.ps1 secrets
```

**macOS/Linux:**
```bash
bash scripts/deploy-flyio.sh secrets
```

### Custom Domain

```bash
flyctl certs add yourdomain.com
```

Then update your DNS as instructed.

## Useful Commands

```bash
# View logs
flyctl logs

# Check status
flyctl status

# Restart app
flyctl apps restart

# SSH into machine
flyctl ssh console

# Scale up memory
flyctl scale memory 512

# Open dashboard
flyctl dashboard
```

## Cost Estimate

**Free Tier Includes:**
- 3 shared-cpu-1x VMs (256MB RAM each)
- 3GB persistent storage
- 160GB outbound data transfer

**Your App (default config):**
- 1x shared-cpu-1x, 256MB RAM: **$0-5/month**
- 1GB storage: **Free (within 3GB limit)**
- Scales to zero when idle: **$0 when not in use**

**Expected Cost:** ~$5-15/month for light to moderate traffic

## Troubleshooting

### App won't start?
```bash
flyctl logs
```

### Need more resources?
```bash
flyctl scale memory 512    # Upgrade to 512MB
flyctl scale vm shared-cpu-2x  # Faster CPU
```

### Database issues?
```bash
flyctl ssh console
# Then:
rm /app/data/ai_tutorial.db
exit
flyctl apps restart
```

## Next Steps

- ✅ Set up custom domain
- ✅ Configure CI/CD (see [DEPLOYMENT.md](docs/DEPLOYMENT.md))
- ✅ Set up monitoring
- ✅ Configure backups

## Resources

- 📖 [Full Deployment Guide](docs/DEPLOYMENT.md)
- 🔧 [Fly.io Documentation](https://fly.io/docs/)
- 💬 [Community Forum](https://community.fly.io/)

---

**Need help?** Check the [full deployment guide](docs/DEPLOYMENT.md) or open an issue.
