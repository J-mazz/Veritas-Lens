#!/bin/bash

# Digital Ocean Deployment Script for Veritas-Lens Backend
# Make sure to run this script with proper permissions: chmod +x deploy.sh

set -e

echo "🚀 Starting Digital Ocean deployment for Veritas-Lens Backend..."

# Configuration
DROPLET_NAME="veritas-lens-backend"
REGION="nyc3"
SIZE="s-2vcpu-4gb"
IMAGE="docker-20-04"
SSH_KEY_NAME="your-ssh-key-name"  # Replace with your SSH key name
DOMAIN="your-domain.com"          # Replace with your domain

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}❌ doctl could not be found. Please install it first.${NC}"
    echo "Visit: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check if doctl is authenticated
if ! doctl account get &> /dev/null; then
    echo -e "${RED}❌ doctl is not authenticated. Please run 'doctl auth init' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ doctl is installed and authenticated${NC}"

# Function to create droplet
create_droplet() {
    echo -e "${YELLOW}📦 Creating Digital Ocean droplet...${NC}"
    
    doctl compute droplet create $DROPLET_NAME \
        --image $IMAGE \
        --size $SIZE \
        --region $REGION \
        --ssh-keys $SSH_KEY_NAME \
        --enable-monitoring \
        --enable-private-networking \
        --tag-names veritas-lens,backend,production \
        --user-data-file cloud-init.yml \
        --wait
    
    echo -e "${GREEN}✅ Droplet created successfully${NC}"
}

# Function to get droplet IP
get_droplet_ip() {
    DROPLET_IP=$(doctl compute droplet get $DROPLET_NAME --format PublicIPv4 --no-header)
    echo -e "${GREEN}📍 Droplet IP: $DROPLET_IP${NC}"
}

# Function to setup DNS (optional)
setup_dns() {
    if [ "$DOMAIN" != "your-domain.com" ]; then
        echo -e "${YELLOW}🌐 Setting up DNS records...${NC}"
        
        # Create A record for domain
        doctl compute domain records create $DOMAIN \
            --record-type A \
            --record-name @ \
            --record-data $DROPLET_IP \
            --record-ttl 300
        
        # Create A record for www subdomain
        doctl compute domain records create $DOMAIN \
            --record-type A \
            --record-name www \
            --record-data $DROPLET_IP \
            --record-ttl 300
        
        echo -e "${GREEN}✅ DNS records created${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipping DNS setup. Update DOMAIN variable with your actual domain.${NC}"
    fi
}

# Function to deploy application
deploy_app() {
    echo -e "${YELLOW}🚀 Deploying application...${NC}"
    
    # Copy files to droplet
    scp -r . root@$DROPLET_IP:/opt/veritas-lens-backend/
    
    # Connect to droplet and setup application
    ssh root@$DROPLET_IP << 'EOF'
        cd /opt/veritas-lens-backend
        
        # Install dependencies
        npm ci --only=production
        
        # Build application
        npm run build
        
        # Create systemd service
        cat > /etc/systemd/system/veritas-lens.service << 'SERVICE'
[Unit]
Description=Veritas-Lens Backend API
After=network.target

[Service]
Type=simple
User=nodejs
WorkingDirectory=/opt/veritas-lens-backend
ExecStart=/usr/bin/node dist/server.js
Restart=on-failure
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
SERVICE
        
        # Enable and start service
        systemctl daemon-reload
        systemctl enable veritas-lens
        systemctl start veritas-lens
        
        # Setup nginx reverse proxy
        cat > /etc/nginx/sites-available/veritas-lens << 'NGINX'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
NGINX
        
        ln -sf /etc/nginx/sites-available/veritas-lens /etc/nginx/sites-enabled/
        systemctl reload nginx
EOF
    
    echo -e "${GREEN}✅ Application deployed successfully${NC}"
}

# Function to setup SSL with Let's Encrypt
setup_ssl() {
    if [ "$DOMAIN" != "your-domain.com" ]; then
        echo -e "${YELLOW}🔒 Setting up SSL certificate...${NC}"
        
        ssh root@$DROPLET_IP << EOF
            # Install certbot
            apt-get update
            apt-get install -y certbot python3-certbot-nginx
            
            # Get SSL certificate
            certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
            
            # Setup auto-renewal
            crontab -l | { cat; echo "0 12 * * * /usr/bin/certbot renew --quiet"; } | crontab -
EOF
        
        echo -e "${GREEN}✅ SSL certificate installed${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipping SSL setup. Update DOMAIN variable with your actual domain.${NC}"
    fi
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}📊 Setting up monitoring...${NC}"
    
    ssh root@$DROPLET_IP << 'EOF'
        # Install Node Exporter for Prometheus monitoring
        wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
        tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz
        mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
        
        # Create systemd service for node_exporter
        cat > /etc/systemd/system/node_exporter.service << 'SERVICE'
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=nobody
Group=nogroup
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
SERVICE
        
        systemctl daemon-reload
        systemctl enable node_exporter
        systemctl start node_exporter
EOF
    
    echo -e "${GREEN}✅ Monitoring setup complete${NC}"
}

# Main deployment flow
main() {
    echo -e "${GREEN}🎯 Starting deployment process...${NC}"
    
    # Check if droplet already exists
    if doctl compute droplet get $DROPLET_NAME &> /dev/null; then
        echo -e "${YELLOW}⚠️  Droplet $DROPLET_NAME already exists${NC}"
        get_droplet_ip
    else
        create_droplet
        get_droplet_ip
        
        # Wait for droplet to be ready
        echo -e "${YELLOW}⏳ Waiting for droplet to be ready...${NC}"
        sleep 60
    fi
    
    deploy_app
    setup_dns
    setup_ssl
    setup_monitoring
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${GREEN}🌐 Your application is now available at:${NC}"
    echo -e "${GREEN}   - IP: http://$DROPLET_IP${NC}"
    if [ "$DOMAIN" != "your-domain.com" ]; then
        echo -e "${GREEN}   - Domain: https://$DOMAIN${NC}"
    fi
    echo -e "${GREEN}📊 Health check: http://$DROPLET_IP/health${NC}"
}

# Run main function
main "$@"
