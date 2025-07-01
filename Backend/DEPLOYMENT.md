# 🚀 Digital Ocean H100 GPU Droplet Setup Guide

## Prerequisites

1. **Digital Ocean Account** with GPU droplet access
2. **SSH Key** added to your DO account
3. **Domain Name** (optional, for SSL)
4. **API Keys** for external services

## Step 1: Create the Droplet

### Option A: Using DigitalOcean Web Interface

1. Go to DigitalOcean control panel → Create → Droplets
2. Choose **Ubuntu 22.04 LTS**
3. Select **GPU Droplet** → **H100 80GB** (gpu-h100x1-80gb)
4. Choose **New York 3** region (has H100 availability)
5. Add your **SSH key**
6. Under **Advanced Options** → **Add initialization scripts**
7. Upload the `scripts/droplet-init.sh` file
8. Create droplet (takes ~10-15 minutes to fully initialize)

### Option B: Using doctl CLI

```bash
# Install doctl
curl -sL https://github.com/digitalocean/doctl/releases/download/v1.98.0/doctl-1.98.0-linux-amd64.tar.gz | tar -xzv
sudo mv doctl /usr/local/bin

# Authenticate
doctl auth init

# Create droplet (replace SSH_KEY_FINGERPRINT with yours)
doctl compute droplet create veritas-lens-h100 \
  --region nyc3 \
  --size gpu-h100x1-80gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys YOUR_SSH_KEY_FINGERPRINT \
  --enable-ipv6 \
  --enable-private-networking \
  --tag-names veritas-lens,ml,gpu,production \
  --user-data-file ./scripts/droplet-init.sh \
  --wait
```

## Step 2: Initial Connection

```bash
# Get droplet IP
doctl compute droplet list

# Connect via SSH
ssh root@YOUR_DROPLET_IP

# Check initialization progress
tail -f /var/log/droplet-init.log
```

## Step 3: Deploy Your Application

```bash
# On your local machine, copy the backend code
scp -r /home/joseph-mazzini/Veritas-Lens/Backend/* root@YOUR_DROPLET_IP:/opt/veritas-lens/backend/

# Or using rsync (recommended)
rsync -avz --exclude 'node_modules' --exclude 'dist' \
  /home/joseph-mazzini/Veritas-Lens/Backend/ \
  root@YOUR_DROPLET_IP:/opt/veritas-lens/backend/
```

## Step 4: Configure Environment

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Configure environment
cd /opt/veritas-lens/backend
cp ../.env.template .env
nano .env

# Update these critical values:
# JWT_SECRET=your-super-secret-jwt-key
# DB_PASSWORD=secure_password_change_this
# HUGGINGFACE_API_TOKEN=your_token
# NEWS_API_KEY=your_key
```

## Step 5: Build and Start

```bash
# Install dependencies and build
npm install
npm run build

# Start the service
systemctl enable veritas-lens
systemctl start veritas-lens

# Check status
systemctl status veritas-lens
```

## Step 6: Verify Installation

```bash
# Run system monitor
/opt/veritas-lens/monitor.sh

# Test API
curl http://localhost:3000/health

# Check GPU
nvidia-smi

# View logs
journalctl -u veritas-lens -f
```

## Step 7: Configure Domain (Optional)

```bash
# Install SSL certificate
certbot --nginx -d your-domain.com

# Update nginx config if needed
nano /etc/nginx/sites-available/veritas-lens
nginx -t
systemctl reload nginx
```

## Step 8: Set Up Monitoring

```bash
# Install additional monitoring (optional)
npm install -g pm2
pm2 install pm2-server-monit

# Set up external monitoring service
# (Uptime Robot, New Relic, etc.)
```

## File Structure on Server

```
/opt/veritas-lens/
├── backend/          # Your Node.js application
├── models/           # ML models and weights
├── data/            # Training data and datasets
├── logs/            # Application logs
├── backups/         # Database backups
├── .env.template    # Environment template
├── deploy.sh        # Deployment script
├── monitor.sh       # System monitoring
├── backup.sh        # Backup script
└── README.md        # Server documentation
```

## Useful Commands

### Service Management
```bash
systemctl start veritas-lens     # Start service
systemctl stop veritas-lens      # Stop service
systemctl restart veritas-lens   # Restart service
systemctl status veritas-lens    # Check status
journalctl -u veritas-lens -f    # View live logs
```

### System Monitoring
```bash
/opt/veritas-lens/monitor.sh     # Full system status
nvidia-smi                       # GPU status
htop                            # CPU/Memory usage
df -h                           # Disk usage
```

### Deployment
```bash
/opt/veritas-lens/deploy.sh      # Deploy updates
/opt/veritas-lens/backup.sh      # Create backup
```

### Database Management
```bash
sudo -u postgres psql veritas_lens  # Connect to DB
pg_dump veritas_lens > backup.sql   # Backup database
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   journalctl -u veritas-lens --no-pager
   ```

2. **GPU not detected**
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Out of memory**
   ```bash
   free -h
   # Reduce ML_BATCH_SIZE in .env
   ```

4. **Port conflicts**
   ```bash
   netstat -tlnp | grep :3000
   ```

### Performance Optimization

1. **Adjust batch sizes** in `.env`
2. **Enable GPU memory growth** in ML code
3. **Use Redis for caching**
4. **Set up log rotation**

## Security Checklist

- [ ] Change default database password
- [ ] Set strong JWT secret
- [ ] Configure firewall rules
- [ ] Set up SSL certificates
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Use strong SSH keys only

## Cost Optimization

- **H100 Droplet**: ~$3.50/hour
- **Auto-shutdown** when not training
- **Use Spot instances** for batch jobs
- **Monitor usage** regularly

## Next Steps

1. **Set up CI/CD pipeline**
2. **Configure monitoring alerts**
3. **Implement model versioning**
4. **Scale with load balancer**
5. **Add backup automation**

---

**Need help?** Check `/opt/veritas-lens/README.md` on the server or run `/opt/veritas-lens/monitor.sh` for system status.
