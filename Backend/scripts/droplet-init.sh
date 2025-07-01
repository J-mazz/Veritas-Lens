#!/bin/bash

# Veritas-Lens Digital Ocean H100 GPU Droplet Initialization Script
# This script sets up a complete ML environment# Install MongoDB
echo "🍃 Installing MongoDB..."
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] http://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" | tee /etc/apt/sources.list.d/mongodb-org-7.0.list
apt-get update
apt-get install -y mongodb-org
systemctl enable mongod
systemctl start mongod

# Configure MongoDB
echo "🔐 Configuring MongoDB..."
mongosh --eval 'db.adminCommand("listCollections")' || true
echo "📋 MongoDB installed and running. Database: veritas_lens"cal bias detection
# DEBIAN 12 BOOKWORM LTS - NO SNAP BULLSHIT!

set -e  # Exit on any error

echo "🚀 Starting Veritas-Lens H100 GPU Droplet Setup on DEBIAN 12..."
echo "================================================================"

# Log everything
exec > >(tee -a /var/log/droplet-init.log) 2>&1

# Update system
echo "📦 Updating Debian system packages..."
apt-get update && apt-get upgrade -y

# Install essential tools
echo "🔧 Installing essential tools..."
apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    python3-pip \
    python3-dev \
    python3-venv \
    dirmngr

# Install NVIDIA Container Toolkit and Docker
echo "🐳 Installing Docker and NVIDIA Container Toolkit..."
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Add NVIDIA package repository for Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo "🎮 Configuring NVIDIA Docker runtime..."
cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

systemctl restart docker
systemctl enable docker

# Install Node.js 20 (LTS)
echo "📱 Installing Node.js 20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Install Python ML dependencies
echo "🐍 Setting up Python ML environment..."
pip3 install --upgrade pip
pip3 install \
    torch \
    torchvision \
    torchaudio \
    transformers \
    datasets \
    accelerate \
    tensorboard \
    jupyter \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    requests \
    beautifulsoup4 \
    feedparser \
    celery \
    redis

# Install TensorFlow with GPU support
echo "🧠 Installing TensorFlow GPU..."
pip3 install tensorflow[and-cuda]

# Install CUDA toolkit (if not already installed)
echo "⚡ Installing CUDA toolkit for Debian..."
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-3

# Set up CUDA environment variables
echo "🔧 Setting up CUDA environment..."
cat >> /etc/environment <<EOF
CUDA_HOME=/usr/local/cuda
PATH=/usr/local/cuda/bin:\$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
EOF

# Install PM2 for process management
echo "⚙️ Installing PM2..."
npm install -g pm2
pm2 startup

# Install Redis for caching and job queues
echo "🔴 Installing Redis..."
apt-get install -y redis-server
systemctl enable redis-server
systemctl start redis-server

# Note: PostgreSQL installation commented out - using managed database cluster
# echo "🐘 Installing PostgreSQL..."
# apt-get install -y postgresql postgresql-contrib
# systemctl enable postgresql
# systemctl start postgresql

# Install PostgreSQL client only (for connecting to managed cluster)
echo "� Installing PostgreSQL client..."
apt-get install -y postgresql-client-15

echo "📋 PostgreSQL client installed. Configure managed database connection in .env"

# Create application directories
echo "📁 Setting up application directories..."
mkdir -p /opt/veritas-lens
mkdir -p /opt/veritas-lens/backend
mkdir -p /opt/veritas-lens/models
mkdir -p /opt/veritas-lens/data
mkdir -p /opt/veritas-lens/logs
mkdir -p /opt/veritas-lens/uploads

# Set up log rotation
echo "📝 Setting up log rotation..."
cat > /etc/logrotate.d/veritas-lens <<EOF
/opt/veritas-lens/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        systemctl reload veritas-lens || true
    endscript
}
EOF

# Create systemd service for the backend
echo "🚀 Creating systemd service..."
cat > /etc/systemd/system/veritas-lens.service <<EOF
[Unit]
Description=Veritas-Lens Backend API
After=network.target mongod.service redis.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/veritas-lens/backend
Environment=NODE_ENV=production
Environment=PORT=3000
ExecStart=/usr/bin/npm start
Restart=on-failure
RestartSec=10
KillMode=process

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration for reverse proxy
echo "🌐 Installing and configuring Nginx..."
apt-get install -y nginx

cat > /etc/nginx/sites-available/veritas-lens <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 10M;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }

    location /api/ {
        proxy_pass http://localhost:3000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

ln -s /etc/nginx/sites-available/veritas-lens /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx

# Set up SSL with Let's Encrypt (optional, requires domain)
echo "🔒 Installing Certbot for SSL..."
apt-get install -y certbot python3-certbot-nginx

# Create deployment script
echo "📋 Creating deployment script..."
cat > /opt/veritas-lens/deploy.sh <<'EOF'
#!/bin/bash
set -e

echo "🚀 Deploying Veritas-Lens Backend..."

# Navigate to backend directory
cd /opt/veritas-lens/backend

# Pull latest code (if using git)
if [ -d ".git" ]; then
    git pull origin main
fi

# Install dependencies
npm install --production

# Build TypeScript
npm run build

# Restart services
systemctl restart veritas-lens
systemctl reload nginx

echo "✅ Deployment complete!"
EOF

chmod +x /opt/veritas-lens/deploy.sh

# Create environment template
echo "🔧 Creating environment template..."
cat > /opt/veritas-lens/.env.template <<EOF
# Server Configuration
NODE_ENV=production
PORT=3000
CORS_ORIGINS=https://yourdomain.com

# Database Configuration (Managed Database Cluster)
DB_HOST=your-postgres-cluster-host.db.ondigitalocean.com
DB_PORT=25060
DB_NAME=veritas_lens
DB_USERNAME=veritas_admin
DB_PASSWORD=your-managed-db-password
DB_SSL=true
DB_CA_CERT=/opt/veritas-lens/certs/ca-certificate.crt

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=24h
JWT_REFRESH_EXPIRES_IN=7d

# External APIs
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
NEWS_API_KEY=your_news_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Machine Learning Configuration
ML_MODEL_PATH=/opt/veritas-lens/models
ML_BATCH_SIZE=32
ML_MAX_SEQUENCE_LENGTH=512
ML_CONFIDENCE_THRESHOLD=0.8

# Active Learning Configuration
AL_UNCERTAINTY_THRESHOLD=0.6
AL_BATCH_SIZE=10
AL_RETRAIN_INTERVAL_HOURS=24

# Logging Configuration
LOG_LEVEL=info
LOG_FORMAT=combined

# Digital Ocean Spaces (optional)
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_ENDPOINT=nyc3.digitaloceanspaces.com
DO_SPACES_BUCKET=veritas-lens-storage

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
EOF

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > /opt/veritas-lens/monitor.sh <<'EOF'
#!/bin/bash

echo "=== Veritas-Lens System Status ==="
echo "Date: $(date)"
echo ""

echo "🖥️  System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')"
echo "Memory Usage: $(free -m | awk 'NR==2{printf "%.1f%%\n", $3*100/$2}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"
echo ""

echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
echo ""

echo "🚀 Service Status:"
systemctl is-active veritas-lens && echo "✅ Veritas-Lens: Running" || echo "❌ Veritas-Lens: Stopped"
systemctl is-active nginx && echo "✅ Nginx: Running" || echo "❌ Nginx: Stopped"
systemctl is-active mongod && echo "✅ MongoDB: Running" || echo "❌ MongoDB: Stopped"
systemctl is-active redis && echo "✅ Redis: Running" || echo "❌ Redis: Stopped"
echo ""

echo "🔌 Network Status:"
netstat -tlnp | grep :3000 && echo "✅ API Port 3000: Listening" || echo "❌ API Port 3000: Not listening"
netstat -tlnp | grep :80 && echo "✅ HTTP Port 80: Listening" || echo "❌ HTTP Port 80: Not listening"
echo ""

echo "📝 Recent Logs:"
echo "--- Last 5 API logs ---"
tail -5 /opt/veritas-lens/logs/combined.log 2>/dev/null || echo "No API logs found"
echo ""
EOF

chmod +x /opt/veritas-lens/monitor.sh

# Create backup script
echo "💾 Creating backup script..."
cat > /opt/veritas-lens/backup.sh <<'EOF'
#!/bin/bash

BACKUP_DIR="/opt/veritas-lens/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "🗄️  Creating database backup..."
sudo -u postgres pg_dump veritas_lens > $BACKUP_DIR/db_backup_$DATE.sql

echo "📁 Creating application backup..."
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz -C /opt/veritas-lens backend models data

echo "🧹 Cleaning old backups (keeping last 7 days)..."
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "✅ Backup complete: $BACKUP_DIR"
EOF

chmod +x /opt/veritas-lens/backup.sh

# Set up cron jobs
echo "⏰ Setting up cron jobs..."
cat > /tmp/veritas-cron <<EOF
# Daily backup at 2 AM
0 2 * * * /opt/veritas-lens/backup.sh >> /opt/veritas-lens/logs/backup.log 2>&1

# System monitoring every 5 minutes
*/5 * * * * /opt/veritas-lens/monitor.sh >> /opt/veritas-lens/logs/monitor.log 2>&1
EOF

crontab /tmp/veritas-cron

# Set up firewall
echo "🔥 Configuring UFW firewall..."
ufw --force enable
ufw allow ssh
ufw allow 80
ufw allow 443
ufw allow 3000

# Install GPU monitoring tools
echo "📊 Installing GPU monitoring tools..."
pip3 install gpustat
npm install -g vtop

# Test GPU
echo "🧪 Testing GPU setup..."
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}');"

# Create quick start guide
echo "📚 Creating quick start guide..."
cat > /opt/veritas-lens/README.md <<EOF
# Veritas-Lens H100 GPU Droplet

## Quick Start

1. **Deploy your application:**
   \`\`\`bash
   cd /opt/veritas-lens/backend
   # Copy your source code here
   npm install
   npm run build
   systemctl start veritas-lens
   \`\`\`

2. **Configure environment:**
   \`\`\`bash
   cp /opt/veritas-lens/.env.template /opt/veritas-lens/backend/.env
   nano /opt/veritas-lens/backend/.env
   systemctl restart veritas-lens
   \`\`\`

3. **Monitor system:**
   \`\`\`bash
   /opt/veritas-lens/monitor.sh
   \`\`\`

## Useful Commands

- **Check logs:** \`journalctl -u veritas-lens -f\`
- **GPU status:** \`nvidia-smi\`
- **System status:** \`/opt/veritas-lens/monitor.sh\`
- **Backup data:** \`/opt/veritas-lens/backup.sh\`
- **Deploy updates:** \`/opt/veritas-lens/deploy.sh\`

## Service Management

- **Start:** \`systemctl start veritas-lens\`
- **Stop:** \`systemctl stop veritas-lens\`
- **Restart:** \`systemctl restart veritas-lens\`
- **Status:** \`systemctl status veritas-lens\`

## File Locations

- **Application:** \`/opt/veritas-lens/backend\`
- **Models:** \`/opt/veritas-lens/models\`
- **Data:** \`/opt/veritas-lens/data\`
- **Logs:** \`/opt/veritas-lens/logs\`
- **Backups:** \`/opt/veritas-lens/backups\`

## API Endpoints

- **Health:** http://your-droplet-ip/health
- **API:** http://your-droplet-ip/api
EOF

# Set permissions
echo "🔐 Setting permissions..."
chown -R root:root /opt/veritas-lens
chmod 755 /opt/veritas-lens
chmod 755 /opt/veritas-lens/backend
chmod 755 /opt/veritas-lens/models
chmod 755 /opt/veritas-lens/data
chmod 755 /opt/veritas-lens/logs

# Final GPU verification
echo "🎮 Final GPU verification..."
echo "NVIDIA Driver Version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo ""
echo "CUDA Version:"
nvcc --version 2>/dev/null || echo "CUDA compiler not in PATH"
echo ""
echo "PyTorch CUDA Test:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "🎉 H100 GPU Droplet Setup Complete!"
echo "================================================="
echo ""
echo "✅ System Components Installed:"
echo "   - NVIDIA H100 GPU drivers and CUDA"
echo "   - Docker with NVIDIA Container Toolkit"
echo "   - Node.js 20 with npm"
echo "   - Python 3 with ML libraries"
echo "   - PostgreSQL database"
echo "   - Redis cache"
echo "   - Nginx reverse proxy"
echo "   - PM2 process manager"
echo ""
echo "📁 Application Structure:"
echo "   - Application root: /opt/veritas-lens"
echo "   - Backend code: /opt/veritas-lens/backend"
echo "   - ML models: /opt/veritas-lens/models"
echo "   - Data storage: /opt/veritas-lens/data"
echo "   - Log files: /opt/veritas-lens/logs"
echo ""
echo "🚀 Next Steps:"
echo "   1. Copy your backend code to /opt/veritas-lens/backend"
echo "   2. Configure environment: cp .env.template backend/.env && nano backend/.env"
echo "   3. Deploy: cd backend && npm install && npm run build"
echo "   4. Start service: systemctl enable veritas-lens && systemctl start veritas-lens"
echo "   5. Monitor: /opt/veritas-lens/monitor.sh"
echo ""
echo "📊 Access your API at: http://$(curl -s ifconfig.me):80"
echo "📚 Read the full guide: /opt/veritas-lens/README.md"
echo ""
echo "💡 Tip: Run '/opt/veritas-lens/monitor.sh' to check system status anytime!"
EOF
