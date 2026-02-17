#!/bin/bash
# AI Engineering Tutorial - Fly.io Deployment Script (Linux/macOS)
# Run with: bash scripts/deploy-flyio.sh <action>

set -e

ACTION=${1:-"help"}
APP_NAME=${2:-"ai-tutorial"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AI Tutorial - Fly.io Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo -e "${RED}Error: flyctl is not installed${NC}"
    echo -e "${YELLOW}Install with: curl -L https://fly.io/install.sh | sh${NC}"
    exit 1
fi

# Check if authenticated
if ! flyctl auth whoami &> /dev/null; then
    echo -e "${YELLOW}Not authenticated with Fly.io${NC}"
    echo -e "${CYAN}Running: flyctl auth login${NC}"
    flyctl auth login
fi

case $ACTION in
    "setup")
        echo -e "${CYAN}Setting up new Fly.io app...${NC}"
        echo ""
        
        # Generate SECRET_KEY
        echo -e "${CYAN}Generating SECRET_KEY...${NC}"
        SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
        echo -e "${YELLOW}Generated SECRET_KEY (save this!):${NC}"
        echo -e "${NC}$SECRET_KEY${NC}"
        echo ""
        
        # Launch app
        echo -e "${CYAN}Launching Fly.io app...${NC}"
        flyctl launch --no-deploy
        
        # Set secret
        echo -e "${CYAN}Setting SECRET_KEY...${NC}"
        flyctl secrets set SECRET_KEY="$SECRET_KEY"
        
        # Create volume
        echo -e "${CYAN}Creating persistent volume...${NC}"
        flyctl volumes create ai_tutorial_data --region iad --size 1
        
        echo ""
        echo -e "${GREEN}Setup complete! Now run: bash scripts/deploy-flyio.sh deploy${NC}"
        ;;
    
    "deploy")
        echo -e "${CYAN}Deploying to Fly.io...${NC}"
        flyctl deploy
        
        echo ""
        echo -e "${GREEN}Deployment complete!${NC}"
        echo -e "${CYAN}Opening app...${NC}"
        flyctl open
        ;;
    
    "logs")
        echo -e "${CYAN}Fetching logs...${NC}"
        flyctl logs
        ;;
    
    "status")
        echo -e "${CYAN}App Status:${NC}"
        flyctl status
        ;;
    
    "secrets")
        echo -e "${CYAN}Setting API key secrets...${NC}"
        echo -e "${YELLOW}Leave blank to skip any key${NC}"
        echo ""
        
        read -p "OpenAI API Key (optional): " OPENAI_KEY
        read -p "Anthropic API Key (optional): " ANTHROPIC_KEY
        read -p "Google API Key (optional): " GOOGLE_KEY
        read -p "xAI API Key (optional): " XAI_KEY
        
        SECRETS=""
        [ ! -z "$OPENAI_KEY" ] && SECRETS="$SECRETS OPENAI_API_KEY=$OPENAI_KEY"
        [ ! -z "$ANTHROPIC_KEY" ] && SECRETS="$SECRETS ANTHROPIC_API_KEY=$ANTHROPIC_KEY"
        [ ! -z "$GOOGLE_KEY" ] && SECRETS="$SECRETS GOOGLE_API_KEY=$GOOGLE_KEY"
        [ ! -z "$XAI_KEY" ] && SECRETS="$SECRETS XAI_API_KEY=$XAI_KEY"
        
        if [ ! -z "$SECRETS" ]; then
            flyctl secrets set $SECRETS
            echo -e "${GREEN}Secrets set successfully!${NC}"
        else
            echo -e "${YELLOW}No secrets to set${NC}"
        fi
        ;;
    
    "scale")
        echo -e "${CYAN}Current scaling:${NC}"
        flyctl scale show
        echo ""
        echo -e "${YELLOW}Scale options:${NC}"
        echo "  memory: flyctl scale memory 512"
        echo "  vm: flyctl scale vm shared-cpu-2x"
        echo "  count: flyctl scale count 1"
        ;;
    
    "restart")
        echo -e "${CYAN}Restarting app...${NC}"
        flyctl apps restart
        echo -e "${GREEN}App restarted!${NC}"
        ;;
    
    "ssh")
        echo -e "${CYAN}Opening SSH console...${NC}"
        flyctl ssh console
        ;;
    
    "destroy")
        echo -e "${RED}WARNING: This will destroy the app!${NC}"
        read -p "Type 'yes' to confirm: " CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            flyctl apps destroy "$APP_NAME"
            echo -e "${YELLOW}App destroyed${NC}"
        else
            echo -e "${GREEN}Cancelled${NC}"
        fi
        ;;
    
    *)
        echo -e "${YELLOW}Usage: bash scripts/deploy-flyio.sh <action>${NC}"
        echo ""
        echo -e "${CYAN}Actions:${NC}"
        echo "  setup     - Initial setup (create app, volume, secrets)"
        echo "  deploy    - Deploy the application"
        echo "  logs      - View application logs"
        echo "  status    - Show app status"
        echo "  secrets   - Set API key secrets"
        echo "  scale     - Show scaling options"
        echo "  restart   - Restart the application"
        echo "  ssh       - SSH into the machine"
        echo "  destroy   - Destroy the app (WARNING)"
        echo ""
        echo -e "${GREEN}Example: bash scripts/deploy-flyio.sh setup${NC}"
        ;;
esac
