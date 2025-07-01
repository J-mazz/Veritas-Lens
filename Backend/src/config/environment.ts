import dotenv from 'dotenv';

dotenv.config();

export const config = {
  // Server configuration
  port: parseInt(process.env.PORT || '3000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  
  // CORS configuration
  corsOrigins: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000', 'http://localhost:3001'],
  
  // Database configuration - MongoDB
  database: {
    uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/veritas_lens',
    host: process.env.MONGODB_HOST || 'localhost',
    port: parseInt(process.env.MONGODB_PORT || '27017', 10),
    name: process.env.MONGODB_DATABASE || 'veritas_lens',
    username: process.env.MONGODB_USERNAME,
    password: process.env.MONGODB_PASSWORD,
  },
  
  // JWT configuration
  jwt: {
    secret: process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-this-in-production',
    expiresIn: process.env.JWT_EXPIRES_IN || '24h',
    refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
  },
  
  // External APIs
  apis: {
    huggingFace: {
      token: process.env.HUGGINGFACE_API_TOKEN,
      baseUrl: 'https://api-inference.huggingface.co',
    },
    kaggle: {
      username: process.env.KAGGLE_USERNAME,
      key: process.env.KAGGLE_KEY,
    },
    newsApi: {
      key: process.env.NEWS_API_KEY,
      baseUrl: 'https://newsapi.org/v2',
    },
  },
  
  // Machine Learning configuration
  ml: {
    modelPath: process.env.ML_MODEL_PATH || './models',
    batchSize: parseInt(process.env.ML_BATCH_SIZE || '32', 10),
    maxSequenceLength: parseInt(process.env.ML_MAX_SEQUENCE_LENGTH || '512', 10),
    confidenceThreshold: parseFloat(process.env.ML_CONFIDENCE_THRESHOLD || '0.8'),
  },
  
  // Data aggregation configuration
  dataAggregation: {
    rssFeeds: process.env.RSS_FEEDS?.split(',') || [
      'https://rss.cnn.com/rss/edition.rss',
      'https://feeds.foxnews.com/foxnews/latest',
      'https://www.npr.org/rss/rss.php?id=1001',
      'https://feeds.reuters.com/Reuters/domesticNews',
    ],
    scrapingInterval: parseInt(process.env.SCRAPING_INTERVAL_MINUTES || '60', 10) * 60 * 1000, // Convert to milliseconds
    maxArticlesPerSource: parseInt(process.env.MAX_ARTICLES_PER_SOURCE || '100', 10),
  },
  
  // Active learning configuration
  activeLearning: {
    uncertaintyThreshold: parseFloat(process.env.AL_UNCERTAINTY_THRESHOLD || '0.6'),
    batchSize: parseInt(process.env.AL_BATCH_SIZE || '10', 10),
    retrainInterval: parseInt(process.env.AL_RETRAIN_INTERVAL_HOURS || '24', 10) * 60 * 60 * 1000, // Convert to milliseconds
  },
  
  // File upload configuration
  upload: {
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760', 10), // 10MB default
    allowedMimeTypes: ['text/plain', 'application/json', 'text/csv'],
  },
  
  // Logging configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: process.env.LOG_FORMAT || 'combined',
  },
  
  // Digital Ocean configuration
  digitalOcean: {
    spacesKey: process.env.DO_SPACES_KEY,
    spacesSecret: process.env.DO_SPACES_SECRET,
    spacesEndpoint: process.env.DO_SPACES_ENDPOINT,
    spacesBucket: process.env.DO_SPACES_BUCKET || 'veritas-lens-storage',
  },
  
  // Redis configuration (for caching and job queues)
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    password: process.env.REDIS_PASSWORD,
    db: parseInt(process.env.REDIS_DB || '0', 10),
  },
} as const;

// Validate required environment variables in production
if (config.nodeEnv === 'production') {
  const requiredEnvVars = [
    'JWT_SECRET',
    'DB_HOST',
    'DB_PASSWORD',
  ];
  
  const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);
  
  if (missingVars.length > 0) {
    throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
  }
}
