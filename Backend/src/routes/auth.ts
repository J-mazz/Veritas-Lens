import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { config } from '@/config/environment';
import { asyncHandler } from '@/middleware/errorHandler';
import { ApiResponse, User, AuthTokens } from '@/types';

const router = Router();

// Register new user
router.post('/register', asyncHandler(async (req: Request, res: Response) => {
  const { email, password, role = 'viewer' } = req.body;

  // TODO: Check if user already exists in database
  
  // Hash password
  const saltRounds = 12;
  const hashedPassword = await bcrypt.hash(password, saltRounds);

  // TODO: Save user to database
  const newUser: User = {
    id: Date.now().toString(),
    email,
    password: hashedPassword,
    role,
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  };

  // Generate tokens
  const accessToken = jwt.sign(
    { id: newUser.id, email: newUser.email, role: newUser.role },
    config.jwt.secret,
    { expiresIn: config.jwt.expiresIn }
  );

  const refreshToken = jwt.sign(
    { id: newUser.id },
    config.jwt.secret,
    { expiresIn: config.jwt.refreshExpiresIn }
  );

  const tokens: AuthTokens = {
    accessToken,
    refreshToken,
    expiresIn: 24 * 60 * 60 // 24 hours in seconds
  };

  const response: ApiResponse<{ user: Omit<User, 'password'>, tokens: AuthTokens }> = {
    success: true,
    data: {
      user: {
        id: newUser.id,
        email: newUser.email,
        role: newUser.role,
        isActive: newUser.isActive,
        createdAt: newUser.createdAt,
        updatedAt: newUser.updatedAt
      },
      tokens
    }
  };

  res.status(201).json(response);
}));

// Login user
router.post('/login', asyncHandler(async (req: Request, res: Response) => {
  const { email, password } = req.body;

  // TODO: Find user in database
  // For now, using mock user
  const mockUser: User = {
    id: '1',
    email: 'admin@veritaslens.com',
    password: await bcrypt.hash('admin123', 12), // In real app, this would come from DB
    role: 'admin',
    isActive: true,
    createdAt: new Date(),
    updatedAt: new Date()
  };

  // Check if user exists and password is correct
  const isValidPassword = await bcrypt.compare(password, mockUser.password);
  
  if (!isValidPassword || email !== mockUser.email) {
    res.status(401).json({
      success: false,
      error: { message: 'Invalid email or password' }
    });
    return;
  }

  if (!mockUser.isActive) {
    res.status(401).json({
      success: false,
      error: { message: 'Account is deactivated' }
    });
    return;
  }

  // Generate tokens
  const accessToken = jwt.sign(
    { id: mockUser.id, email: mockUser.email, role: mockUser.role },
    config.jwt.secret,
    { expiresIn: config.jwt.expiresIn }
  );

  const refreshToken = jwt.sign(
    { id: mockUser.id },
    config.jwt.secret,
    { expiresIn: config.jwt.refreshExpiresIn }
  );

  const tokens: AuthTokens = {
    accessToken,
    refreshToken,
    expiresIn: 24 * 60 * 60 // 24 hours in seconds
  };

  const response: ApiResponse<{ user: Omit<User, 'password'>, tokens: AuthTokens }> = {
    success: true,
    data: {
      user: {
        id: mockUser.id,
        email: mockUser.email,
        role: mockUser.role,
        isActive: mockUser.isActive,
        createdAt: mockUser.createdAt,
        updatedAt: mockUser.updatedAt
      },
      tokens
    }
  };

  res.json(response);
}));

// Refresh token
router.post('/refresh', asyncHandler(async (req: Request, res: Response) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    res.status(401).json({
      success: false,
      error: { message: 'Refresh token required' }
    });
    return;
  }

  try {
    const decoded = jwt.verify(refreshToken, config.jwt.secret) as any;
    
    // TODO: Verify refresh token exists in database and is valid
    
    // Generate new access token
    const newAccessToken = jwt.sign(
      { id: decoded.id, email: decoded.email, role: decoded.role },
      config.jwt.secret,
      { expiresIn: config.jwt.expiresIn }
    );

    const tokens: AuthTokens = {
      accessToken: newAccessToken,
      refreshToken, // Keep the same refresh token
      expiresIn: 24 * 60 * 60
    };

    const response: ApiResponse<AuthTokens> = {
      success: true,
      data: tokens
    };

    res.json(response);
  } catch (error) {
    res.status(403).json({
      success: false,
      error: { message: 'Invalid refresh token' }
    });
  }
}));

// Logout (invalidate refresh token)
router.post('/logout', asyncHandler(async (req: Request, res: Response) => {
  const { refreshToken } = req.body;

  // TODO: Remove refresh token from database/blacklist
  
  const response: ApiResponse = {
    success: true
  };

  res.json(response);
}));

// Get current user profile
router.get('/me', asyncHandler(async (req: Request, res: Response) => {
  const authHeader = req.headers.authorization;
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    res.status(401).json({
      success: false,
      error: { message: 'Access token required' }
    });
    return;
  }

  try {
    const decoded = jwt.verify(token, config.jwt.secret) as any;
    
    // TODO: Fetch user from database
    const user = {
      id: decoded.id,
      email: decoded.email,
      role: decoded.role,
      isActive: true,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    const response: ApiResponse<Omit<User, 'password'>> = {
      success: true,
      data: user
    };

    res.json(response);
  } catch (error) {
    res.status(403).json({
      success: false,
      error: { message: 'Invalid token' }
    });
  }
}));

export default router;
