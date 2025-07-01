import { Router, Request, Response } from 'express';
import { ActiveLearningService, AnnotationFeedback, LabelingTask } from '@/services/activeLearningService';
import { MLService } from '@/services/mlService';
import { logger } from '@/config/logger';
import { body, param, query, validationResult } from 'express-validator';

const router = Router();
const activeLearningService = ActiveLearningService.getInstance();
const mlService = MLService.getInstance();

/**
 * @swagger
 * tags:
 *   name: Active Learning
 *   description: Active learning for model improvement
 */

/**
 * @swagger
 * /api/active-learning/tasks:
 *   get:
 *     summary: Get pending labeling tasks
 *     tags: [Active Learning]
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *         description: Maximum number of tasks to return
 *       - in: query
 *         name: annotator_id
 *         schema:
 *           type: string
 *         description: ID of the annotator (assigns tasks)
 *     responses:
 *       200:
 *         description: List of pending labeling tasks
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/LabelingTask'
 */
router.get('/tasks', 
  [
    query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
    query('annotator_id').optional().isString().trim()
  ],
  async (req: Request, res: Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          error: {
            message: 'Validation failed',
            details: errors.array()
          }
        });
      }

      const limit = req.query.limit as number || 20;
      const annotatorId = req.query.annotator_id as string;

      const tasks = await activeLearningService.getPendingTasks(annotatorId, limit);

      res.json({
        success: true,
        data: tasks,
        meta: {
          count: tasks.length,
          limit,
          annotator_id: annotatorId
        }
      });
    } catch (error) {
      logger.error('Error getting labeling tasks:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to get labeling tasks',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

/**
 * @swagger
 * /api/active-learning/annotate:
 *   post:
 *     summary: Submit annotation for a labeling task
 *     tags: [Active Learning]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - task_id
 *               - article_id
 *               - label
 *               - confidence
 *               - annotator_id
 *               - time_spent
 *             properties:
 *               task_id:
 *                 type: string
 *               article_id:
 *                 type: string
 *               label:
 *                 type: string
 *                 enum: [left, center, right]
 *               confidence:
 *                 type: number
 *                 minimum: 0
 *                 maximum: 1
 *               annotator_id:
 *                 type: string
 *               time_spent:
 *                 type: number
 *                 description: Time spent in seconds
 *               notes:
 *                 type: string
 *               is_correction:
 *                 type: boolean
 *     responses:
 *       201:
 *         description: Annotation submitted successfully
 *       400:
 *         description: Invalid annotation data
 *       500:
 *         description: Server error
 */
router.post('/annotate',
  [
    body('task_id').isString().trim().notEmpty(),
    body('article_id').isString().trim().notEmpty(),
    body('label').isIn(['left', 'center', 'right']),
    body('confidence').isFloat({ min: 0, max: 1 }),
    body('annotator_id').isString().trim().notEmpty(),
    body('time_spent').isInt({ min: 0 }),
    body('notes').optional().isString().trim(),
    body('is_correction').optional().isBoolean()
  ],
  async (req: Request, res: Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          error: {
            message: 'Validation failed',
            details: errors.array()
          }
        });
      }

      const feedback: AnnotationFeedback = {
        taskId: req.body.task_id,
        articleId: req.body.article_id,
        label: req.body.label,
        confidence: req.body.confidence,
        annotatorId: req.body.annotator_id,
        timeSpent: req.body.time_spent,
        notes: req.body.notes,
        isCorrection: req.body.is_correction || false
      };

      await activeLearningService.processAnnotationFeedback(feedback);

      logger.info(`Annotation submitted by ${feedback.annotatorId} for article ${feedback.articleId}`);

      res.status(201).json({
        success: true,
        data: {
          message: 'Annotation submitted successfully',
          task_id: feedback.taskId,
          article_id: feedback.articleId
        }
      });
    } catch (error) {
      logger.error('Error processing annotation:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to process annotation',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

/**
 * @swagger
 * /api/active-learning/generate-tasks:
 *   post:
 *     summary: Generate new labeling tasks from articles
 *     tags: [Active Learning]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - article_ids
 *             properties:
 *               article_ids:
 *                 type: array
 *                 items:
 *                   type: string
 *               strategy:
 *                 type: string
 *                 enum: [uncertainty, entropy, margin]
 *                 default: uncertainty
 *               count:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 50
 *                 default: 10
 *     responses:
 *       201:
 *         description: Tasks generated successfully
 *       400:
 *         description: Invalid request data
 *       500:
 *         description: Server error
 */
router.post('/generate-tasks',
  [
    body('article_ids').isArray({ min: 1 }),
    body('article_ids.*').isString().trim().notEmpty(),
    body('strategy').optional().isIn(['uncertainty', 'entropy', 'margin']),
    body('count').optional().isInt({ min: 1, max: 50 })
  ],
  async (req: Request, res: Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          error: {
            message: 'Validation failed',
            details: errors.array()
          }
        });
      }

      const articleIds = req.body.article_ids;
      const strategy = req.body.strategy || 'uncertainty';
      const count = req.body.count || 10;

      // TODO: Fetch articles from database using articleIds
      // For now, create mock articles
      const articles = articleIds.map((id: string) => ({
        id,
        title: `Article ${id}`,
        content: `Content for article ${id}`,
        url: `https://example.com/article/${id}`,
        source: 'mock',
        publishedAt: new Date(),
        isLabeled: false,
        createdAt: new Date(),
        updatedAt: new Date()
      }));

      const result = await activeLearningService.processNewArticles(articles);

      logger.info(`Generated ${result.labelingTasks.length} labeling tasks using ${strategy} strategy`);

      res.status(201).json({
        success: true,
        data: {
          predictions: result.predictions,
          labeling_tasks: result.labelingTasks,
          retraining_triggered: result.retrainingTriggered,
          strategy_used: strategy
        },
        meta: {
          total_articles: articles.length,
          total_predictions: result.predictions.length,
          total_tasks: result.labelingTasks.length
        }
      });
    } catch (error) {
      logger.error('Error generating labeling tasks:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to generate labeling tasks',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

/**
 * @swagger
 * /api/active-learning/retrain:
 *   post:
 *     summary: Trigger model retraining
 *     tags: [Active Learning]
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               force:
 *                 type: boolean
 *                 description: Force retraining even if not enough new data
 *                 default: false
 *     responses:
 *       202:
 *         description: Retraining job queued successfully
 *       400:
 *         description: Invalid request
 *       500:
 *         description: Server error
 */
router.post('/retrain',
  [
    body('force').optional().isBoolean()
  ],
  async (req: Request, res: Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          error: {
            message: 'Validation failed',
            details: errors.array()
          }
        });
      }

      const force = req.body.force || false;

      if (!force) {
        // Check if retraining is actually needed
        const needsRetraining = await mlService.checkRetrainingNeed();
        if (!needsRetraining) {
          return res.status(400).json({
            success: false,
            error: {
              message: 'Retraining not needed at this time',
              details: 'Use force=true to override this check'
            }
          });
        }
      }

      const jobId = await activeLearningService.triggerRetraining();

      logger.info(`Model retraining triggered: ${jobId}`);

      res.status(202).json({
        success: true,
        data: {
          job_id: jobId,
          message: 'Model retraining job queued successfully',
          forced: force
        }
      });
    } catch (error) {
      logger.error('Error triggering retraining:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to trigger retraining',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

/**
 * @swagger
 * /api/active-learning/stats:
 *   get:
 *     summary: Get active learning statistics
 *     tags: [Active Learning]
 *     responses:
 *       200:
 *         description: Active learning statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: object
 *                   properties:
 *                     total_predictions:
 *                       type: integer
 *                     total_annotations:
 *                       type: integer
 *                     pending_tasks:
 *                       type: integer
 *                     accuracy_improvement:
 *                       type: number
 *                     labeling_budget_used:
 *                       type: integer
 *                     labeling_budget_remaining:
 *                       type: integer
 *                     model_version:
 *                       type: string
 *                     last_retraining:
 *                       type: string
 *                       format: date-time
 */
router.get('/stats',
  async (req: Request, res: Response) => {
    try {
      const stats = await activeLearningService.getActiveLearningStats();

      res.json({
        success: true,
        data: stats
      });
    } catch (error) {
      logger.error('Error getting active learning stats:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to get active learning statistics',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

/**
 * @swagger
 * /api/active-learning/task/{taskId}:
 *   get:
 *     summary: Get details of a specific labeling task
 *     tags: [Active Learning]
 *     parameters:
 *       - in: path
 *         name: taskId
 *         required: true
 *         schema:
 *           type: string
 *         description: The task ID
 *     responses:
 *       200:
 *         description: Task details
 *       404:
 *         description: Task not found
 *       500:
 *         description: Server error
 */
router.get('/task/:taskId',
  [
    param('taskId').isString().trim().notEmpty()
  ],
  async (req: Request, res: Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          error: {
            message: 'Validation failed',
            details: errors.array()
          }
        });
      }

      const taskId = req.params.taskId;

      // TODO: Implement database query for specific task
      // For now, return a mock response
      const task = {
        id: taskId,
        article_id: `article-${taskId}`,
        article_title: 'Sample Article Title',
        article_content: 'Sample article content for labeling...',
        current_prediction: {
          bias_score: 0.2,
          bias_label: 'center',
          confidence: 0.6
        },
        priority: 0.4,
        created_at: new Date(),
        deadline: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
      };

      res.json({
        success: true,
        data: task
      });
    } catch (error) {
      logger.error('Error getting task details:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to get task details',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

/**
 * @swagger
 * /api/active-learning/task/{taskId}/skip:
 *   post:
 *     summary: Skip a labeling task
 *     tags: [Active Learning]
 *     parameters:
 *       - in: path
 *         name: taskId
 *         required: true
 *         schema:
 *           type: string
 *         description: The task ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - annotator_id
 *               - reason
 *             properties:
 *               annotator_id:
 *                 type: string
 *               reason:
 *                 type: string
 *                 enum: [unclear, insufficient_context, controversial, other]
 *               notes:
 *                 type: string
 *     responses:
 *       200:
 *         description: Task skipped successfully
 *       400:
 *         description: Invalid request
 *       404:
 *         description: Task not found
 *       500:
 *         description: Server error
 */
router.post('/task/:taskId/skip',
  [
    param('taskId').isString().trim().notEmpty(),
    body('annotator_id').isString().trim().notEmpty(),
    body('reason').isIn(['unclear', 'insufficient_context', 'controversial', 'other']),
    body('notes').optional().isString().trim()
  ],
  async (req: Request, res: Response) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          success: false,
          error: {
            message: 'Validation failed',
            details: errors.array()
          }
        });
      }

      const taskId = req.params.taskId;
      const annotatorId = req.body.annotator_id;
      const reason = req.body.reason;
      const notes = req.body.notes;

      // TODO: Implement task skipping logic
      // - Mark task as skipped
      // - Record reason and annotator
      // - Update task priority or reassign

      logger.info(`Task ${taskId} skipped by ${annotatorId}: ${reason}`);

      res.json({
        success: true,
        data: {
          message: 'Task skipped successfully',
          task_id: taskId,
          reason,
          skipped_by: annotatorId
        }
      });
    } catch (error) {
      logger.error('Error skipping task:', error);
      res.status(500).json({
        success: false,
        error: {
          message: 'Failed to skip task',
          details: error instanceof Error ? error.message : 'Unknown error'
        }
      });
    }
  }
);

export default router;
