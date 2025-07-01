# Veritas-Lens Backend

A TypeScript/Express.js backend service for political bias detection in news articles with live data aggregation and active learning capabilities.

## 🚀 Features

- **Political Bias Detection**: ML-powered analysis of news articles
- **Live Data Aggregation**: RSS feeds, web scraping, and API integration
- **Active Learning**: Continuous model improvement with human feedback
- **RESTful API**: Comprehensive API for frontend integration
- **Digital Ocean Ready**: Pre-configured for cloud deployment
- **Security**: JWT authentication, rate limiting, and security middleware
- **Monitoring**: Health checks and logging infrastructure

## 🏗️ Architecture

```
├── src/
│   ├── config/          # Environment and logging configuration
│   ├── controllers/     # Request handlers
│   ├── middleware/      # Express middleware
│   ├── models/          # Data models and database schemas
│   ├── routes/          # API route definitions
│   ├── services/        # Business logic services
│   ├── types/           # TypeScript type definitions
│   └── utils/           # Utility functions
├── logs/                # Application logs
├── models/              # ML model files
└── deployment/          # Docker and deployment configs
```

## 🛠️ Quick Start

### Local Development

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd Backend
npm install
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start development server:**
```bash
npm run dev:ts
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🌊 Digital Ocean Deployment

### Prerequisites

1. **Install doctl CLI:**
```bash
# macOS
brew install doctl

# Linux
wget https://github.com/digitalocean/doctl/releases/download/v1.94.0/doctl-1.94.0-linux-amd64.tar.gz
tar xf doctl-1.94.0-linux-amd64.tar.gz
sudo mv doctl /usr/local/bin
```

2. **Authenticate with Digital Ocean:**
```bash
doctl auth init
```

3. **Upload your SSH key:**
```bash
doctl compute ssh-key import your-key-name --public-key-file ~/.ssh/id_rsa.pub
```

### Deployment Options

#### Option 1: Automated Deployment Script

```bash
# Configure deployment settings in deploy.sh
vim deploy.sh  # Update DROPLET_NAME, DOMAIN, SSH_KEY_NAME

# Run deployment
./deploy.sh
```

#### Option 2: Manual Droplet Setup

1. **Create a droplet:**
```bash
doctl compute droplet create veritas-lens-backend \
  --image docker-20-04 \
  --size s-2vcpu-4gb \
  --region nyc3 \
  --ssh-keys your-ssh-key-name \
  --user-data-file cloud-init.yml
```

2. **Deploy application:**
```bash
# Get droplet IP
DROPLET_IP=$(doctl compute droplet get veritas-lens-backend --format PublicIPv4 --no-header)

# Copy files to droplet
scp -r . root@$DROPLET_IP:/opt/veritas-lens-backend/

# SSH into droplet and start services
ssh root@$DROPLET_IP
cd /opt/veritas-lens-backend
npm ci --only=production
npm run build
npm start
```

#### Option 3: App Platform (Managed)

Create `app-platform.yaml`:
```yaml
name: veritas-lens-backend
services:
- name: api
  source_dir: /
  github:
    repo: your-username/veritas-lens
    branch: main
  run_command: npm start
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: NODE_ENV
    value: production
```

Deploy:
```bash
doctl apps create --spec app-platform.yaml
```

## 🗄️ Database Setup

### Digital Ocean Managed PostgreSQL

1. **Create database cluster:**
```bash
doctl databases create veritas-lens-db \
  --engine postgres \
  --version 15 \
  --size db-s-1vcpu-1gb \
  --region nyc3 \
  --num-nodes 1
```

2. **Get connection details:**
```bash
doctl databases connection veritas-lens-db
```

3. **Update environment variables:**
```bash
# Update .env with database connection details
DB_HOST=your-db-host.db.ondigitalocean.com
DB_PORT=25060
DB_NAME=veritas_lens
DB_USERNAME=doadmin
DB_PASSWORD=your-secure-password
DB_SSL=true
```

### Local PostgreSQL

```bash
# Using Docker
docker run --name veritas-postgres \
  -e POSTGRES_DB=veritas_lens \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d postgres:15-alpine
```

## 🔧 Configuration

### Environment Variables

Key environment variables for production:

```bash
# Server
PORT=3000
NODE_ENV=production

# Database
DB_HOST=your-db-host.db.ondigitalocean.com
DB_PASSWORD=your-secure-password

# Security
JWT_SECRET=your-super-secure-jwt-secret

# External APIs
HUGGINGFACE_API_TOKEN=your-token
NEWS_API_KEY=your-key

# Digital Ocean Spaces
DO_SPACES_KEY=your-spaces-key
DO_SPACES_SECRET=your-spaces-secret
```

### ML Model Configuration

```bash
# Place your trained models in the models/ directory
models/
├── bias_classifier.h5      # TensorFlow model
├── tokenizer.json          # BERT tokenizer
└── label_encoder.pkl       # Label encoder
```

## 📡 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `POST /api/auth/refresh` - Refresh token

### Articles
- `GET /api/articles` - List articles with filters
- `POST /api/articles` - Create article
- `GET /api/articles/:id` - Get article by ID
- `PUT /api/articles/:id` - Update article
- `DELETE /api/articles/:id` - Delete article

### Bias Analysis
- `POST /api/bias/analyze` - Analyze text for bias
- `POST /api/bias/batch` - Batch analyze multiple texts
- `GET /api/bias/predictions/:articleId` - Get predictions for article

### Data Aggregation
- `GET /api/sources` - List news sources
- `POST /api/sources` - Add news source
- `POST /api/scraping/start` - Start scraping job
- `GET /api/scraping/status` - Get scraping status

### Active Learning
- `GET /api/learning/queries` - Get labeling queries
- `POST /api/learning/label` - Submit label for query
- `POST /api/learning/retrain` - Trigger model retraining

### Health & Monitoring
- `GET /health` - Health check
- `GET /health/detailed` - Detailed system status

## 🔍 Monitoring

### Health Checks

The service provides multiple health check endpoints:

```bash
# Basic health check
curl http://your-domain.com/health

# Detailed health check
curl http://your-domain.com/health/detailed
```

### Logging

Logs are structured and written to:
- Console (development)
- `logs/combined.log` (all logs)
- `logs/error.log` (errors only)

### Metrics

The service exposes metrics for monitoring:
- Node.js metrics via Node Exporter (port 9100)
- Application metrics via custom endpoints
- Database connection health
- ML model performance metrics

## 🔒 Security

### Production Security Checklist

- [ ] Change default JWT secret
- [ ] Use strong database passwords
- [ ] Enable SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable CORS for specific domains
- [ ] Use environment variables for secrets
- [ ] Regular security updates

### SSL/TLS Setup

The deployment script automatically sets up Let's Encrypt SSL certificates:

```bash
# Manual SSL setup
certbot --nginx -d your-domain.com
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 3000 already in use:**
```bash
# Kill process using port 3000
sudo lsof -t -i tcp:3000 | xargs kill -9
```

2. **Database connection failed:**
```bash
# Check database status
doctl databases get veritas-lens-db
# Verify connection details in .env
```

3. **ML model loading errors:**
```bash
# Ensure model files are in the correct location
ls -la models/
# Check file permissions
chmod 644 models/*
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=debug
npm run dev:ts
```

## 📊 Performance

### Recommended Digital Ocean Setup

- **Droplet**: 2 vCPUs, 4GB RAM (s-2vcpu-4gb)
- **Database**: 1 vCPU, 1GB RAM (db-s-1vcpu-1gb)
- **Spaces**: For model and data storage
- **Load Balancer**: For high availability

### Scaling Considerations

- Use Redis for session storage and caching
- Implement horizontal scaling with multiple droplets
- Use Digital Ocean Load Balancer for traffic distribution
- Consider upgrading to larger database instances for heavy workloads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review Digital Ocean documentation

---

**Ready to deploy to Digital Ocean!** 🚀
