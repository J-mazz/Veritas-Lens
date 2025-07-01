import { Pool, PoolConfig } from 'pg';
import { config } from '@/config/environment';
import { logger } from '@/config/logger';
import fs from 'fs';
import path from 'path';

class DatabaseConnection {
  private static instance: DatabaseConnection;
  private pool: Pool;

  private constructor() {
    this.pool = this.createPool();
  }

  public static getInstance(): DatabaseConnection {
    if (!DatabaseConnection.instance) {
      DatabaseConnection.instance = new DatabaseConnection();
    }
    return DatabaseConnection.instance;
  }

  private createPool(): Pool {
    const poolConfig: PoolConfig = {
      host: config.database.host,
      port: config.database.port,
      database: config.database.name,
      user: config.database.username,
      password: config.database.password,
      ssl: config.database.ssl ? {
        rejectUnauthorized: true,
        ca: this.loadCACertificate(),
      } : false,
      max: 20, // Maximum number of clients in the pool
      idleTimeoutMillis: 30000, // How long a client is allowed to remain idle
      connectionTimeoutMillis: 10000, // How long to wait when connecting
      statement_timeout: 60000, // How long to wait for a query to complete
      query_timeout: 60000,
    };

    const pool = new Pool(poolConfig);

    // Handle pool errors
    pool.on('error', (err) => {
      logger.error('Unexpected database pool error:', err);
    });

    // Log successful connection
    pool.on('connect', () => {
      logger.info('New database connection established');
    });

    // Log connection removal
    pool.on('remove', () => {
      logger.info('Database connection removed from pool');
    });

    return pool;
  }

  private loadCACertificate(): string | undefined {
    try {
      const certPath = process.env.DB_CA_CERT || '/opt/veritas-lens/certs/ca-certificate.crt';
      if (fs.existsSync(certPath)) {
        return fs.readFileSync(certPath, 'utf8');
      }
      logger.warn(`CA certificate not found at ${certPath}. SSL connection may fail.`);
      return undefined;
    } catch (error) {
      logger.error('Error loading CA certificate:', error);
      return undefined;
    }
  }

  public getPool(): Pool {
    return this.pool;
  }

  public async query(text: string, params?: any[]): Promise<any> {
    const start = Date.now();
    try {
      const res = await this.pool.query(text, params);
      const duration = Date.now() - start;
      logger.debug(`Query executed in ${duration}ms:`, { text, params, rowCount: res.rowCount });
      return res;
    } catch (error) {
      logger.error('Database query error:', { error, text, params });
      throw error;
    }
  }

  public async getClient() {
    return await this.pool.connect();
  }

  public async testConnection(): Promise<boolean> {
    try {
      const result = await this.query('SELECT NOW() as current_time, version() as db_version');
      logger.info('Database connection test successful:', {
        currentTime: result.rows[0].current_time,
        version: result.rows[0].db_version
      });
      return true;
    } catch (error) {
      logger.error('Database connection test failed:', error);
      return false;
    }
  }

  public async close(): Promise<void> {
    try {
      await this.pool.end();
      logger.info('Database pool closed');
    } catch (error) {
      logger.error('Error closing database pool:', error);
    }
  }

  // Initialize database schema
  public async initializeSchema(): Promise<void> {
    try {
      const schemaPath = path.join(__dirname, '../sql/schema.sql');
      if (fs.existsSync(schemaPath)) {
        const schema = fs.readFileSync(schemaPath, 'utf8');
        await this.query(schema);
        logger.info('Database schema initialized successfully');
      } else {
        logger.warn('Schema file not found, skipping schema initialization');
      }
    } catch (error) {
      logger.error('Error initializing database schema:', error);
      throw error;
    }
  }

  // Health check for monitoring
  public async healthCheck(): Promise<{ status: string; details: any }> {
    try {
      const result = await this.query('SELECT 1 as health_check');
      return {
        status: 'healthy',
        details: {
          connected: true,
          poolSize: this.pool.totalCount,
          idleCount: this.pool.idleCount,
          waitingCount: this.pool.waitingCount
        }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          connected: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        }
      };
    }
  }
}

export const db = DatabaseConnection.getInstance();
export default db;
