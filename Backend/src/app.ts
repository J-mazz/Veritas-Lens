import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';

// Import routes
import apiRoutes from '@/routes/api';
import healthRoutes from '@/routes/health';

// Import middleware
import { errorHandler } from '@/middleware/errorHandler';
import { requestLogger } from '@/middleware/requestLogger';

// Import configuration
import { config } from '@/config/environment';
import { logger } from '@/config/logger';

// Load environment variables
dotenv.config();

class App {
  public app: express.Application;
  private readonly port: number;

  constructor() {
    this.app = express();
    this.port = config.port;
    this.initializeMiddleware();
    this.initializeRoutes();
    this.initializeErrorHandling();
  }

  private initializeMiddleware(): void {
    // Security middleware
    this.app.use(helmet());
    
    // CORS configuration
    this.app.use(cors({
      origin: config.corsOrigins,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization']
    }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use('/api/', limiter);

    // Compression and parsing
    this.app.use(compression());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Logging
    this.app.use(morgan('combined', { stream: { write: (message) => logger.info(message.trim()) } }));
    this.app.use(requestLogger);
  }

  private initializeRoutes(): void {
    // Health check routes
    this.app.use('/health', healthRoutes);
    
    // API routes
    this.app.use('/api', apiRoutes);
    
    // Default route
    this.app.get('/', (req, res) => {
      res.json({
        message: 'Veritas-Lens Backend API',
        version: '1.0.0',
        status: 'healthy',
        timestamp: new Date().toISOString()
      });
    });

    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        error: 'Route not found',
        message: `The requested route ${req.originalUrl} does not exist`
      });
    });
  }

  private initializeErrorHandling(): void {
    this.app.use(errorHandler);
  }

  public listen(): void {
    this.app.listen(this.port, () => {
      logger.info(`🚀 Veritas-Lens Backend server running on port ${this.port}`);
      logger.info(`📱 Environment: ${config.nodeEnv}`);
      logger.info(`🔗 API Base URL: http://localhost:${this.port}/api`);
    });
  }

  public getApp(): express.Application {
    return this.app;
  }
}

export default App;
