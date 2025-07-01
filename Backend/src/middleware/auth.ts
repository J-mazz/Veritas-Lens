import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { config } from '@/config/environment';
import { logger } from '@/config/logger';
import { User } from '@/types';

export interface AuthenticatedRequest extends Request {
  user?: User;
}

export const authenticateToken = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

    if (!token) {
      res.status(401).json({
        success: false,
        error: { message: 'Access token required' }
      });
      return;
    }

    const decoded = jwt.verify(token, config.jwt.secret) as any;
    
    // In a real implementation, you would fetch the user from the database
    // For now, we'll use the decoded token data
    req.user = {
      id: decoded.id,
      email: decoded.email,
      role: decoded.role,
      isActive: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      password: '' // Never expose password
    };

    next();
  } catch (error) {
    logger.error('Authentication error:', error);
    res.status(403).json({
      success: false,
      error: { message: 'Invalid or expired token' }
    });
  }
};

export const requireRole = (roles: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        success: false,
        error: { message: 'Authentication required' }
      });
      return;
    }

    if (!roles.includes(req.user.role)) {
      res.status(403).json({
        success: false,
        error: { message: 'Insufficient permissions' }
      });
      return;
    }

    next();
  };
};

export const requireAdmin = requireRole(['admin']);
export const requireAnnotator = requireRole(['admin', 'annotator']);
