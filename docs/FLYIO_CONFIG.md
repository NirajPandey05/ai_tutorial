# Fly.io Launch Configuration Guide

Quick reference for `flyctl launch` settings.

## 🚀 Launch Command Options

When you run `flyctl launch`, you'll be prompted with several questions:

### Interactive Launch (Recommended for First Time)
```bash
flyctl launch
```

### Non-Interactive Launch (Advanced)
```bash
flyctl launch \
  --name my-ai-tutorial \
  --region iad \
  --no-deploy \
  --copy-config \
  --dockerfile Dockerfile
```

## 📋 Launch Prompts & Answers

| Prompt | Answer | Reason |
|--------|--------|--------|
| **App name?** | `your-unique-name` or blank | Blank auto-generates unique name |
| **Region?** | `iad` (or nearest you) | Choose datacenter closest to users |
| **PostgreSQL database?** | `N` (No) | Using SQLite instead |
| **Redis database?** | `N` (No) | Not needed |
| **Deploy now?** | `N` (No) | Need to set secrets & volume first |

## 🌍 Available Regions

Choose the region closest to your target users:

### North America
- `iad` - **Ashburn, Virginia (US East)**
- `ord` - Chicago, Illinois
- `dfw` - Dallas, Texas
- `lax` - **Los Angeles, California (US West)**
- `sjc` - San Jose, California
- `sea` - Seattle, Washington
- `ewr` - Secaucus, New Jersey
- `yyz` - Toronto, Canada

### Europe
- `lhr` - **London, UK**
- `ams` - Amsterdam, Netherlands
- `fra` - **Frankfurt, Germany**
- `cdg` - Paris, France
- `mad` - Madrid, Spain
- `waw` - Warsaw, Poland

### Asia Pacific
- `nrt` - Tokyo, Japan
- `hkg` - Hong Kong
- `sin` - Singapore
- `syd` - **Sydney, Australia**

### South America
- `gru` - São Paulo, Brazil
- `scl` - Santiago, Chile

**Tip:** Use `flyctl platform regions` to see full list with latency.

## ⚙️ fly.toml Configuration

### Must Change Before Launch

Edit [fly.toml](../fly.toml):

```toml
# 1. CHANGE THIS - Must be globally unique
app = "your-unique-app-name"

# 2. CHANGE THIS - Region nearest your users
primary_region = "iad"
```

### Optional Adjustments

```toml
# Increase memory if needed (default: 256MB)
[vm]
  memory_mb = 512  # Options: 256, 512, 1024, 2048, 4096, 8192

# Keep more than one instance running (costs more)
[[services]]
  min_machines_running = 1  # Default: 0 (scale to zero)

# Change volume mount path (advanced)
[mounts]
  source = "ai_tutorial_data"
  destination = "/app/data"  # Where volume is mounted in container
```

## 🔐 Required Secrets (Set After Launch)

```bash
# Generate and set SECRET_KEY
flyctl secrets set SECRET_KEY="your-generated-secret-key"

# Optional: Set default API keys
flyctl secrets set \
  OPENAI_API_KEY="sk-..." \
  ANTHROPIC_API_KEY="sk-ant-..." \
  GOOGLE_API_KEY="..." \
  XAI_API_KEY="..."
```

**Generate SECRET_KEY:**
```powershell
# Windows
python -c "import secrets; print(secrets.token_hex(32))"

# Or
[System.Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

## 💾 Required Volume (Create After Launch)

```bash
# Create 1GB persistent volume
flyctl volumes create ai_tutorial_data \
  --region iad \
  --size 1

# List volumes
flyctl volumes list

# Volume name must match fly.toml [mounts] source
```

## 🎯 Complete Launch Workflow

### Method 1: Using Scripts (Easiest)

```powershell
# Windows - Does everything automatically
pwsh scripts/deploy-flyio.ps1 setup
pwsh scripts/deploy-flyio.ps1 deploy
```

### Method 2: Manual Step-by-Step

```bash
# 1. Update fly.toml (app name, region)
# Edit fly.toml manually

# 2. Launch (don't deploy yet)
flyctl launch --no-deploy

# 3. Generate SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# 4. Set secrets
flyctl secrets set SECRET_KEY="<generated-key>"

# 5. Create volume
flyctl volumes create ai_tutorial_data --region iad --size 1

# 6. Deploy
flyctl deploy

# 7. Verify
flyctl status
flyctl open
```

## 🔍 Verify Configuration

Before deploying, check:

```bash
# View current config
cat fly.toml

# Validate configuration
flyctl config validate

# Check app info
flyctl info

# List secrets (values hidden)
flyctl secrets list

# List volumes
flyctl volumes list
```

## 🐛 Common Issues

### "App name already taken"
```bash
# Try a different name or let Fly generate one
flyctl launch  # Leave name blank when prompted
```

### "Volume not found"
```bash
# Create the volume
flyctl volumes create ai_tutorial_data --size 1

# Ensure volume region matches app region
flyctl volumes create ai_tutorial_data --region <same-as-app> --size 1
```

### "SECRET_KEY validation failed"
```bash
# Set the secret
flyctl secrets set SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

### "Health check failed"
```bash
# Check logs
flyctl logs

# SSH into machine
flyctl ssh console

# Test health endpoint
curl http://localhost:8080/health
```

## 📊 Resource Sizing Guide

### Development/Testing
```toml
[vm]
  memory_mb = 256  # Minimal, free tier
  min_machines_running = 0  # Scale to zero
```
**Cost:** $0-5/month

### Light Production (< 1000 users/day)
```toml
[vm]
  memory_mb = 512
  min_machines_running = 0
```
**Cost:** $5-10/month

### Medium Production (1000-10k users/day)
```toml
[vm]
  memory_mb = 1024
  min_machines_running = 1
```
**Cost:** $15-30/month

### High Traffic (10k+ users/day)
```toml
[vm]
  memory_mb = 2048
  cpus = 2
  min_machines_running = 2
```
**Cost:** $50-100/month

## 🔗 Quick Reference

```bash
# Essential commands
flyctl status          # App status
flyctl logs            # View logs
flyctl ssh console     # SSH into machine
flyctl scale show      # Current resources
flyctl secrets list    # List secrets
flyctl volumes list    # List volumes
flyctl info            # App details
flyctl dashboard       # Open web dashboard

# Scaling
flyctl scale memory 512
flyctl scale count 1
flyctl scale vm shared-cpu-2x

# Management
flyctl deploy          # Deploy changes
flyctl apps restart    # Restart app
flyctl apps destroy    # Delete app (careful!)
```

## 📚 Additional Resources

- **Fly.io Docs:** https://fly.io/docs/
- **Regions:** https://fly.io/docs/reference/regions/
- **Pricing:** https://fly.io/docs/about/pricing/
- **Volumes:** https://fly.io/docs/reference/volumes/
- **Secrets:** https://fly.io/docs/reference/secrets/

---

**Ready?** Run: `pwsh scripts/deploy-flyio.ps1 setup`
