# Pre-Deployment Checklist for Fly.io

Use this checklist before deploying to production.

## ✅ Required Steps

### 1. Account & Authentication
- [ ] Created Fly.io account at [fly.io](https://fly.io/app/sign-up)
- [ ] Installed Flyctl CLI
- [ ] Authenticated with `flyctl auth login`
- [ ] Payment method added (required even for free tier)

### 2. Configuration Files
- [ ] `fly.toml` exists in project root
- [ ] App name in `fly.toml` is unique
- [ ] Region is set to closest user location
- [ ] Volume mount configuration is correct
- [ ] Health check endpoint is `/health`

### 3. Environment Variables
- [ ] Generated new SECRET_KEY (not using default)
- [ ] Set DEBUG=false for production
- [ ] BASE_URL updated to production URL
- [ ] Environment variables reviewed in `fly.toml`

### 4. Docker Configuration
- [ ] `Dockerfile` builds successfully locally
  ```bash
  docker build -t ai-tutorial-test .
  docker run -p 8080:8080 ai-tutorial-test
  ```
- [ ] `.dockerignore` excludes unnecessary files
- [ ] Health check endpoint works: `http://localhost:8080/health`

### 5. Application Code
- [ ] All dependencies in `pyproject.toml` are correct
- [ ] No hardcoded secrets or API keys in code
- [ ] Database paths use `/app/data` directory
- [ ] Static files are properly copied in Dockerfile
- [ ] Application starts without errors locally

### 6. Security
- [ ] SECRET_KEY is unique and secure (32+ chars)
- [ ] HTTPS is enforced (`force_https = true` in fly.toml)
- [ ] No sensitive data in logs
- [ ] CORS settings are appropriate
- [ ] Content Security Policy reviewed (if applicable)

### 7. Testing
- [ ] Application runs locally: `uv run uvicorn src.ai_tutorial.main:app`
- [ ] Docker container runs: `docker compose up`
- [ ] Health check returns 200: `curl http://localhost:8080/health`
- [ ] All routes accessible
- [ ] Labs work with test API keys

### 8. Monitoring & Observability
- [ ] Health check endpoint tested
- [ ] Logging configured appropriately
- [ ] Error handling in place
- [ ] Know how to access logs: `flyctl logs`

## 🚀 Deployment Steps

### Initial Deployment
```bash
# Windows
pwsh scripts/deploy-flyio.ps1 setup
pwsh scripts/deploy-flyio.ps1 deploy

# macOS/Linux
bash scripts/deploy-flyio.sh setup
bash scripts/deploy-flyio.sh deploy
```

### Verify Deployment
- [ ] App opens successfully: `flyctl open`
- [ ] Health check passes: `curl https://your-app.fly.dev/health`
- [ ] Homepage loads correctly
- [ ] Navigation works
- [ ] Settings page accessible
- [ ] Can save API keys in browser
- [ ] Labs execute correctly

## 📋 Post-Deployment

### Immediate Actions
- [ ] Verify app is running: `flyctl status`
- [ ] Check logs: `flyctl logs`
- [ ] Test all major features
- [ ] Verify custom domain (if configured)
- [ ] Set up monitoring/alerts

### Documentation
- [ ] Update README with production URL
- [ ] Document any custom configuration
- [ ] Share credentials securely (if needed)
- [ ] Update DNS records (if using custom domain)

### Optional Enhancements
- [ ] Set up CI/CD with GitHub Actions
- [ ] Configure custom domain
- [ ] Set up automated backups
- [ ] Configure monitoring alerts
- [ ] Set up staging environment
- [ ] Enable metrics collection

## 🔍 Troubleshooting

If deployment fails, check:

1. **Build errors:**
   ```bash
   flyctl logs
   ```

2. **Volume not mounting:**
   ```bash
   flyctl volumes list
   flyctl ssh console -C "ls -la /app/data"
   ```

3. **Secrets not set:**
   ```bash
   flyctl secrets list
   ```

4. **Health check failing:**
   ```bash
   flyctl ssh console -C "curl http://localhost:8080/health"
   ```

5. **Out of memory:**
   ```bash
   flyctl scale memory 512
   ```

## 💰 Cost Monitoring

### Free Tier Limits
- 3 shared-cpu-1x VMs (256MB RAM)
- 3GB persistent storage
- 160GB outbound transfer/month

### Monitor Usage
```bash
flyctl dashboard
```

### Expected Costs (configured defaults)
- **Light traffic (< 1000 users/month):** $0-5/month
- **Moderate traffic (1000-10k users/month):** $5-15/month
- **Heavy traffic (10k+ users/month):** $15-50/month

### Cost Optimization
- [ ] Auto-scale to zero enabled (default in fly.toml)
- [ ] Using minimal VM size (256MB RAM)
- [ ] Volume is minimal size (1GB)
- [ ] Static assets cached properly

## 📞 Support Resources

- **Fly.io Status:** https://status.flyio.net/
- **Documentation:** https://fly.io/docs/
- **Community:** https://community.fly.io/
- **Get help:** `flyctl doctor`

## 🎯 Go-Live Criteria

Only proceed to production when:
- ✅ All checklist items complete
- ✅ Application tested locally
- ✅ Docker build successful
- ✅ All core features working
- ✅ Security reviewed
- ✅ Monitoring in place
- ✅ Rollback plan documented

---

**Ready to deploy?** Run:
```bash
# Windows
pwsh scripts/deploy-flyio.ps1 setup

# macOS/Linux  
bash scripts/deploy-flyio.sh setup
```
