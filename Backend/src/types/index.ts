// Article related types
export interface Article {
  id: string;
  title: string;
  content: string;
  url: string;
  source: string;
  publishedAt: Date;
  author?: string;
  description?: string;
  biasScore?: number; // -1 (left) to 1 (right), 0 being center
  biasLabel?: 'left' | 'center' | 'right';
  confidence?: number; // 0 to 1, how confident the model is
  isLabeled: boolean; // Whether this article has been manually labeled
  createdAt: Date;
  updatedAt: Date;
}

// News source configuration
export interface NewsSource {
  id: string;
  name: string;
  url: string;
  rssUrl?: string;
  type: 'rss' | 'scraping' | 'api';
  isActive: boolean;
  lastFetchedAt?: Date;
  configuration?: {
    selectors?: {
      title?: string;
      content?: string;
      author?: string;
      publishedAt?: string;
    };
    headers?: Record<string, string>;
  };
}

// Machine Learning related types
export interface Biasprediction {
  articleId: string;
  biasScore: number;
  biasLabel: 'left' | 'center' | 'right';
  confidence: number;
  modelVersion: string;
  predictedAt: Date;
}

export interface ModelTrainingData {
  articleId: string;
  text: string;
  label: 'left' | 'center' | 'right';
  source: 'manual' | 'active_learning' | 'initial_dataset';
  confidence?: number;
  createdAt: Date;
}

// Active Learning types
export interface ActiveLearningQuery {
  id: string;
  articleId: string;
  uncertainty: number;
  queryStrategy: 'uncertainty' | 'entropy' | 'margin';
  status: 'pending' | 'labeled' | 'skipped';
  createdAt: Date;
  labeledAt?: Date;
  labeledBy?: string;
}

// User and Authentication types
export interface User {
  id: string;
  email: string;
  password: string; // hashed
  role: 'admin' | 'annotator' | 'viewer';
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    code?: string;
    details?: any;
  };
  meta?: {
    page?: number;
    limit?: number;
    total?: number;
    totalPages?: number;
  };
}

// Pagination types
export interface PaginationQuery {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

// Search and Filter types
export interface ArticleFilters {
  source?: string;
  biasLabel?: 'left' | 'center' | 'right';
  dateFrom?: Date;
  dateTo?: Date;
  isLabeled?: boolean;
  minConfidence?: number;
  search?: string;
}

// Data aggregation types
export interface ScrapingJob {
  id: string;
  sourceId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startedAt?: Date;
  completedAt?: Date;
  articlesFound: number;
  articlesProcessed: number;
  errors?: string[];
}

// External API types
export interface HuggingFaceResponse {
  label: string;
  score: number;
}

export interface NewsApiResponse {
  status: string;
  totalResults: number;
  articles: {
    source: { id: string; name: string };
    author: string;
    title: string;
    description: string;
    url: string;
    urlToImage: string;
    publishedAt: string;
    content: string;
  }[];
}

// Database connection types
export interface DatabaseConfig {
  host: string;
  port: number;
  name: string;
  username: string;
  password: string;
  ssl?: boolean;
}

// Request types with custom properties
export interface AuthenticatedRequest extends Request {
  user?: User;
}

// WebSocket types
export interface WebSocketMessage {
  type: 'training_update' | 'new_article' | 'bias_prediction' | 'system_status';
  data: any;
  timestamp: Date;
}

export type BiasLabel = 'left' | 'center' | 'right';
export type UserRole = 'admin' | 'annotator' | 'viewer';
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';
export type QueryStrategy = 'uncertainty' | 'entropy' | 'margin';
