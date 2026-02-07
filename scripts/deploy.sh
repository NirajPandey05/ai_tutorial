#!/bin/bash
# AI Engineering Tutorial - Deployment Script
# Production deployment using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AI Engineering Tutorial - Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your API keys before continuing${NC}"
fi

# Parse arguments
ACTION=${1:-"up"}
BUILD_FLAG=""

case $ACTION in
    "build")
        echo -e "${GREEN}Building Docker images...${NC}"
        docker compose build --no-cache
        ;;
    "up")
        echo -e "${GREEN}Starting services...${NC}"
        docker compose up -d
        echo -e "${GREEN}Services started!${NC}"
        echo -e "Application: http://localhost:8080"
        echo -e "Ollama API: http://localhost:11434"
        ;;
    "up-build")
        echo -e "${GREEN}Building and starting services...${NC}"
        docker compose up -d --build
        echo -e "${GREEN}Services started!${NC}"
        echo -e "Application: http://localhost:8080"
        echo -e "Ollama API: http://localhost:11434"
        ;;
    "down")
        echo -e "${GREEN}Stopping services...${NC}"
        docker compose down
        echo -e "${GREEN}Services stopped${NC}"
        ;;
    "logs")
        docker compose logs -f
        ;;
    "status")
        echo -e "${GREEN}Service Status:${NC}"
        docker compose ps
        ;;
    "restart")
        echo -e "${GREEN}Restarting services...${NC}"
        docker compose restart
        echo -e "${GREEN}Services restarted${NC}"
        ;;
    "pull-models")
        echo -e "${GREEN}Pulling Ollama models...${NC}"
        docker compose exec ollama ollama pull llama3.2
        docker compose exec ollama ollama pull nomic-embed-text
        echo -e "${GREEN}Models pulled successfully${NC}"
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
        docker compose down -v --rmi local
        echo -e "${GREEN}Cleanup complete${NC}"
        ;;
    *)
        echo "Usage: $0 {build|up|up-build|down|logs|status|restart|pull-models|clean}"
        echo ""
        echo "Commands:"
        echo "  build       - Build Docker images"
        echo "  up          - Start all services"
        echo "  up-build    - Build and start services"
        echo "  down        - Stop all services"
        echo "  logs        - Follow service logs"
        echo "  status      - Show service status"
        echo "  restart     - Restart all services"
        echo "  pull-models - Pull Ollama models"
        echo "  clean       - Remove containers, volumes, and images"
        exit 1
        ;;
esac
