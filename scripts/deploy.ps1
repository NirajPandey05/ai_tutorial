# AI Engineering Tutorial - Windows Deployment Script
# Production deployment using Docker Compose

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "up", "up-build", "down", "logs", "status", "restart", "pull-models", "clean", "help")]
    [string]$Action = "up"
)

$ErrorActionPreference = "Stop"

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "========================================"
Write-ColorOutput Green "AI Engineering Tutorial - Deployment"
Write-ColorOutput Green "========================================"

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-ColorOutput Red "Error: Docker is not installed"
    exit 1
}

# Check if docker compose is available
try {
    docker compose version | Out-Null
} catch {
    Write-ColorOutput Red "Error: Docker Compose is not installed"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-ColorOutput Yellow "Warning: .env file not found"
    Write-Output "Creating .env from .env.example..."
    Copy-Item ".env.example" ".env"
    Write-ColorOutput Yellow "Please edit .env with your API keys before continuing"
}

switch ($Action) {
    "build" {
        Write-ColorOutput Green "Building Docker images..."
        docker compose build --no-cache
    }
    "up" {
        Write-ColorOutput Green "Starting services..."
        docker compose up -d
        Write-ColorOutput Green "Services started!"
        Write-Output "Application: http://localhost:8080"
        Write-Output "Ollama API: http://localhost:11434"
    }
    "up-build" {
        Write-ColorOutput Green "Building and starting services..."
        docker compose up -d --build
        Write-ColorOutput Green "Services started!"
        Write-Output "Application: http://localhost:8080"
        Write-Output "Ollama API: http://localhost:11434"
    }
    "down" {
        Write-ColorOutput Green "Stopping services..."
        docker compose down
        Write-ColorOutput Green "Services stopped"
    }
    "logs" {
        docker compose logs -f
    }
    "status" {
        Write-ColorOutput Green "Service Status:"
        docker compose ps
    }
    "restart" {
        Write-ColorOutput Green "Restarting services..."
        docker compose restart
        Write-ColorOutput Green "Services restarted"
    }
    "pull-models" {
        Write-ColorOutput Green "Pulling Ollama models..."
        docker compose exec ollama ollama pull llama3.2
        docker compose exec ollama ollama pull nomic-embed-text
        Write-ColorOutput Green "Models pulled successfully"
    }
    "clean" {
        Write-ColorOutput Yellow "Cleaning up Docker resources..."
        docker compose down -v --rmi local
        Write-ColorOutput Green "Cleanup complete"
    }
    "help" {
        Write-Output "Usage: .\deploy.ps1 <command>"
        Write-Output ""
        Write-Output "Commands:"
        Write-Output "  build       - Build Docker images"
        Write-Output "  up          - Start all services"
        Write-Output "  up-build    - Build and start services"
        Write-Output "  down        - Stop all services"
        Write-Output "  logs        - Follow service logs"
        Write-Output "  status      - Show service status"
        Write-Output "  restart     - Restart all services"
        Write-Output "  pull-models - Pull Ollama models"
        Write-Output "  clean       - Remove containers, volumes, and images"
        Write-Output "  help        - Show this help message"
    }
}
