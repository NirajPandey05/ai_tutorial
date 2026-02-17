#!/usr/bin/env pwsh
# AI Engineering Tutorial - Fly.io Deployment Script (Windows PowerShell)
# Run with: pwsh scripts/deploy-flyio.ps1

param(
    [string]$Action = "deploy",
    [string]$AppName = "ai-tutorial"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Green
Write-Host "AI Tutorial - Fly.io Deployment" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if flyctl is installed
if (-not (Get-Command flyctl -ErrorAction SilentlyContinue)) {
    Write-Host "Error: flyctl is not installed" -ForegroundColor Red
    Write-Host "Install with: iwr https://fly.io/install.ps1 -useb | iex" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
try {
    flyctl auth whoami | Out-Null
} catch {
    Write-Host "Not authenticated with Fly.io" -ForegroundColor Yellow
    Write-Host "Running: flyctl auth login" -ForegroundColor Cyan
    flyctl auth login
}

switch ($Action) {
    "setup" {
        Write-Host "Setting up new Fly.io app..." -ForegroundColor Cyan
        Write-Host ""
        
        # Generate SECRET_KEY
        Write-Host "Generating SECRET_KEY..." -ForegroundColor Cyan
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($bytes)
        $secretKey = [BitConverter]::ToString($bytes).Replace("-", "").ToLower()
        
        Write-Host "Generated SECRET_KEY (save this!):" -ForegroundColor Yellow
        Write-Host $secretKey -ForegroundColor White
        Write-Host ""
        
        # Launch app
        Write-Host "Launching Fly.io app..." -ForegroundColor Cyan
        flyctl launch --no-deploy
        
        # Set secret
        Write-Host "Setting SECRET_KEY..." -ForegroundColor Cyan
        flyctl secrets set SECRET_KEY=$secretKey
        
        # Create volume
        Write-Host "Creating persistent volume..." -ForegroundColor Cyan
        flyctl volumes create ai_tutorial_data --region iad --size 1
        
        Write-Host ""
        Write-Host "Setup complete! Now run: pwsh scripts/deploy-flyio.ps1 deploy" -ForegroundColor Green
    }
    
    "deploy" {
        Write-Host "Deploying to Fly.io..." -ForegroundColor Cyan
        flyctl deploy
        
        Write-Host ""
        Write-Host "Deployment complete!" -ForegroundColor Green
        Write-Host "Opening app..." -ForegroundColor Cyan
        flyctl open
    }
    
    "logs" {
        Write-Host "Fetching logs..." -ForegroundColor Cyan
        flyctl logs
    }
    
    "status" {
        Write-Host "App Status:" -ForegroundColor Cyan
        flyctl status
    }
    
    "secrets" {
        Write-Host "Setting API key secrets..." -ForegroundColor Cyan
        Write-Host "Leave blank to skip any key" -ForegroundColor Yellow
        Write-Host ""
        
        $openai = Read-Host "OpenAI API Key (optional)"
        $anthropic = Read-Host "Anthropic API Key (optional)"
        $google = Read-Host "Google API Key (optional)"
        $xai = Read-Host "xAI API Key (optional)"
        
        $secrets = @()
        if ($openai) { $secrets += "OPENAI_API_KEY=$openai" }
        if ($anthropic) { $secrets += "ANTHROPIC_API_KEY=$anthropic" }
        if ($google) { $secrets += "GOOGLE_API_KEY=$google" }
        if ($xai) { $secrets += "XAI_API_KEY=$xai" }
        
        if ($secrets.Count -gt 0) {
            flyctl secrets set $secrets
            Write-Host "Secrets set successfully!" -ForegroundColor Green
        } else {
            Write-Host "No secrets to set" -ForegroundColor Yellow
        }
    }
    
    "scale" {
        Write-Host "Current scaling:" -ForegroundColor Cyan
        flyctl scale show
        Write-Host ""
        Write-Host "Scale options:" -ForegroundColor Yellow
        Write-Host "  memory: flyctl scale memory 512" -ForegroundColor White
        Write-Host "  vm: flyctl scale vm shared-cpu-2x" -ForegroundColor White
        Write-Host "  count: flyctl scale count 1" -ForegroundColor White
    }
    
    "restart" {
        Write-Host "Restarting app..." -ForegroundColor Cyan
        flyctl apps restart
        Write-Host "App restarted!" -ForegroundColor Green
    }
    
    "ssh" {
        Write-Host "Opening SSH console..." -ForegroundColor Cyan
        flyctl ssh console
    }
    
    "destroy" {
        Write-Host "WARNING: This will destroy the app!" -ForegroundColor Red
        $confirm = Read-Host "Type 'yes' to confirm"
        if ($confirm -eq "yes") {
            flyctl apps destroy $AppName
            Write-Host "App destroyed" -ForegroundColor Yellow
        } else {
            Write-Host "Cancelled" -ForegroundColor Green
        }
    }
    
    default {
        Write-Host "Usage: pwsh scripts/deploy-flyio.ps1 <action>" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Actions:" -ForegroundColor Cyan
        Write-Host "  setup     - Initial setup (create app, volume, secrets)" -ForegroundColor White
        Write-Host "  deploy    - Deploy the application" -ForegroundColor White
        Write-Host "  logs      - View application logs" -ForegroundColor White
        Write-Host "  status    - Show app status" -ForegroundColor White
        Write-Host "  secrets   - Set API key secrets" -ForegroundColor White
        Write-Host "  scale     - Show scaling options" -ForegroundColor White
        Write-Host "  restart   - Restart the application" -ForegroundColor White
        Write-Host "  ssh       - SSH into the machine" -ForegroundColor White
        Write-Host "  destroy   - Destroy the app (WARNING)" -ForegroundColor White
        Write-Host ""
        Write-Host "Example: pwsh scripts/deploy-flyio.ps1 setup" -ForegroundColor Green
    }
}
