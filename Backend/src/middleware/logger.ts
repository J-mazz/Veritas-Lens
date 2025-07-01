import { Request, Response, NextFunction } from 'express';
import { logger } from '@/config/logger';

export const requestLogger = (req: Request, res: Response, next: NextFunction): void => {
  const start = Date.now();
  
  // Log request
  logger.info(`Incoming ${req.method} ${req.url}`, {
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    referer: req.get('Referer'),
    contentType: req.get('Content-Type'),
  });

  // Override res.end to log response
  const originalEnd = res.end;
  res.end = function(chunk?: any, encoding?: any): Response {
    const duration = Date.now() - start;
    
    logger.info(`Response ${req.method} ${req.url}`, {
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
    });

    return originalEnd.call(this, chunk, encoding);
  };

  next();
};
