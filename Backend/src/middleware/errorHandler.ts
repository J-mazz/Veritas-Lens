import { Request, Response, NextFunction } from 'express';
import { logger } from '@/config/logger';

export interface CustomError extends Error {
  statusCode?: number;
  isOperational?: boolean;
}

export const errorHandler = (
  err: CustomError,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  let { statusCode = 500, message } = err;

  // Log error
  logger.error(`Error ${statusCode}: ${message}`, {
    error: err,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
  });

  // Handle specific error types
  if (err.name === 'ValidationError') {
    statusCode = 400;
    message = 'Validation Error';
  } else if (err.name === 'UnauthorizedError' || err.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Unauthorized';
  } else if (err.name === 'CastError') {
    statusCode = 400;
    message = 'Invalid ID format';
  } else if (err.name === 'MongoError' && (err as any).code === 11000) {
    statusCode = 409;
    message = 'Duplicate field value';
  }

  // Don't leak error details in production
  const response: any = {
    success: false,
    error: {
      message: statusCode === 500 && process.env.NODE_ENV === 'production' 
        ? 'Internal Server Error' 
        : message,
      ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
    }
  };

  res.status(statusCode).json(response);
};

export const notFound = (req: Request, res: Response, next: NextFunction): void => {
  const error = new Error(`Not Found - ${req.originalUrl}`) as CustomError;
  error.statusCode = 404;
  next(error);
};

export const asyncHandler = (fn: Function) => (req: Request, res: Response, next: NextFunction) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};
