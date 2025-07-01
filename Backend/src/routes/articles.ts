import { Router, Request, Response } from 'express';
import { authenticateToken, AuthenticatedRequest } from '@/middleware/auth';
import { asyncHandler } from '@/middleware/errorHandler';
import { ApiResponse, Article, ArticleFilters, PaginationQuery } from '@/types';

const router = Router();

// Get articles with filters and pagination
router.get('/', asyncHandler(async (req: Request, res: Response) => {
  const {
    page = 1,
    limit = 20,
    sortBy = 'publishedAt',
    sortOrder = 'desc',
    source,
    biasLabel,
    dateFrom,
    dateTo,
    isLabeled,
    minConfidence,
    search
  } = req.query;

  // TODO: Implement actual database query
  // This is a mock response for now
  const mockArticles: Article[] = [
    {
      id: '1',
      title: 'Sample Political Article',
      content: 'This is a sample article content...',
      url: 'https://example.com/article1',
      source: 'Example News',
      publishedAt: new Date(),
      biasScore: 0.2,
      biasLabel: 'center',
      confidence: 0.85,
      isLabeled: true,
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ];

  const response: ApiResponse<Article[]> = {
    success: true,
    data: mockArticles,
    meta: {
      page: Number(page),
      limit: Number(limit),
      total: 1,
      totalPages: 1
    }
  };

  res.json(response);
}));

// Get single article by ID
router.get('/:id', asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params;

  // TODO: Implement actual database query
  const mockArticle: Article = {
    id,
    title: 'Sample Political Article',
    content: 'This is a sample article content with more details...',
    url: 'https://example.com/article1',
    source: 'Example News',
    publishedAt: new Date(),
    biasScore: 0.2,
    biasLabel: 'center',
    confidence: 0.85,
    isLabeled: true,
    createdAt: new Date(),
    updatedAt: new Date()
  };

  const response: ApiResponse<Article> = {
    success: true,
    data: mockArticle
  };

  res.json(response);
}));

// Create new article (authenticated)
router.post('/', authenticateToken, asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { title, content, url, source, author, description } = req.body;

  // TODO: Implement actual article creation
  const newArticle: Article = {
    id: Date.now().toString(),
    title,
    content,
    url,
    source,
    author,
    description,
    publishedAt: new Date(),
    isLabeled: false,
    createdAt: new Date(),
    updatedAt: new Date()
  };

  const response: ApiResponse<Article> = {
    success: true,
    data: newArticle
  };

  res.status(201).json(response);
}));

// Update article (authenticated)
router.put('/:id', authenticateToken, asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const updates = req.body;

  // TODO: Implement actual article update
  const updatedArticle: Article = {
    id,
    ...updates,
    updatedAt: new Date()
  };

  const response: ApiResponse<Article> = {
    success: true,
    data: updatedArticle
  };

  res.json(response);
}));

// Delete article (authenticated, admin only)
router.delete('/:id', authenticateToken, asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;

  // TODO: Check user permissions and implement actual deletion
  
  const response: ApiResponse = {
    success: true
  };

  res.json(response);
}));

// Label article for training (authenticated, annotator+)
router.post('/:id/label', authenticateToken, asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  const { id } = req.params;
  const { biasLabel, confidence } = req.body;

  // TODO: Implement article labeling logic
  
  const response: ApiResponse = {
    success: true,
    data: {
      articleId: id,
      biasLabel,
      confidence,
      labeledBy: req.user?.id,
      labeledAt: new Date()
    }
  };

  res.json(response);
}));

// Get article statistics
router.get('/stats/overview', asyncHandler(async (req: Request, res: Response) => {
  // TODO: Implement actual statistics
  const stats = {
    totalArticles: 1000,
    labeledArticles: 750,
    unlabeledArticles: 250,
    biasDistribution: {
      left: 300,
      center: 400,
      right: 300
    },
    averageConfidence: 0.82,
    recentlyAdded: 45 // last 24 hours
  };

  const response: ApiResponse = {
    success: true,
    data: stats
  };

  res.json(response);
}));

export default router;
