import { Router } from 'express';

// Import route modules
import articlesRouter from './articles';
import authRouter from './auth';
import mlRouter from './ml';
import dataRouter from './data';
import adminRouter from './admin';
import activeLearningRouter from './activeLearning';

const router = Router();

// Mount route modules
router.use('/auth', authRouter);
router.use('/articles', articlesRouter);
router.use('/ml', mlRouter);
router.use('/data', dataRouter);
router.use('/admin', adminRouter);
router.use('/active-learning', activeLearningRouter);

// API info endpoint
router.get('/', (req, res) => {
  res.json({
    name: 'Veritas-Lens Backend API',
    version: '1.0.0',
    description: 'Political bias detection API with live data aggregation and active learning',
    endpoints: {
      authentication: '/api/auth',
      articles: '/api/articles',
      machine_learning: '/api/ml',
      data_aggregation: '/api/data',
      administration: '/api/admin',
      active_learning: '/api/active-learning',
      health: '/health'
    },
    documentation: 'https://github.com/your-username/veritas-lens/blob/main/Backend/README.md'
  });
});

export default router;
