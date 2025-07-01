import { logger } from '@/config/logger';
import { config } from '@/config/environment';
import { 
  Article, 
  ActiveLearningQuery, 
  ModelTrainingData, 
  Biasprediction 
} from '@/types';
import { MLService } from './mlService';
import { Queue, Worker, Job } from 'bullmq';
import Redis from 'ioredis';

export interface ActiveLearningStrategy {
  name: string;
  selectSamples(predictions: Biasprediction[], count: number): string[];
}

export interface LabelingTask {
  id: string;
  articleId: string;
  articleTitle: string;
  articleContent: string;
  currentPrediction: Biasprediction;
  priority: number;
  assignedTo?: string;
  createdAt: Date;
  deadline?: Date;
}

export interface AnnotationFeedback {
  taskId: string;
  articleId: string;
  label: 'left' | 'center' | 'right';
  confidence: number;
  annotatorId: string;
  timeSpent: number; // seconds
  notes?: string;
  isCorrection?: boolean; // true if correcting a previous prediction
}

export class ActiveLearningService {
  private static instance: ActiveLearningService;
  private mlService: MLService;
  private redis: Redis;
  private retrainingQueue: Queue;
  private labelingQueue: Queue;
  private strategies: Map<string, ActiveLearningStrategy>;
  
  // Active learning state
  private labelingBudget: number = 100; // max labels per day
  private currentLabelsToday: number = 0;
  private lastResetDate: Date = new Date();
  
  public static getInstance(): ActiveLearningService {
    if (!ActiveLearningService.instance) {
      ActiveLearningService.instance = new ActiveLearningService();
    }
    return ActiveLearningService.instance;
  }

  constructor() {
    this.mlService = MLService.getInstance();
    this.redis = new Redis({
      host: config.redis.host,
      port: config.redis.port,
      db: config.redis.db
    });
    
    // Initialize queues for background processing
    this.retrainingQueue = new Queue('model-retraining', { 
      connection: this.redis 
    });
    
    this.labelingQueue = new Queue('active-learning-labeling', { 
      connection: this.redis 
    });
    
    this.strategies = new Map();
    this.initializeStrategies();
    this.setupWorkers();
  }

  private initializeStrategies(): void {
    // Uncertainty Sampling Strategy
    this.strategies.set('uncertainty', {
      name: 'Uncertainty Sampling',
      selectSamples: (predictions: Biasprediction[], count: number): string[] => {
        return predictions
          .sort((a, b) => a.confidence - b.confidence) // lowest confidence first
          .slice(0, count)
          .map(p => p.articleId);
      }
    });

    // Entropy-based Strategy
    this.strategies.set('entropy', {
      name: 'Entropy Sampling',
      selectSamples: (predictions: Biasprediction[], count: number): string[] => {
        // Calculate entropy for each prediction
        const withEntropy = predictions.map(p => {
          // Simulate class probabilities from confidence and bias score
          const centerProb = Math.max(0.1, 1 - Math.abs(p.biasScore));
          const leftProb = p.biasScore < 0 ? p.confidence : (1 - p.confidence) / 2;
          const rightProb = p.biasScore > 0 ? p.confidence : (1 - p.confidence) / 2;
          
          // Normalize probabilities
          const total = centerProb + leftProb + rightProb;
          const probs = [centerProb / total, leftProb / total, rightProb / total];
          
          // Calculate entropy
          const entropy = -probs.reduce((sum, prob) => 
            sum + (prob > 0 ? prob * Math.log2(prob) : 0), 0
          );
          
          return { ...p, entropy };
        });

        return withEntropy
          .sort((a, b) => b.entropy - a.entropy) // highest entropy first
          .slice(0, count)
          .map(p => p.articleId);
      }
    });

    // Margin Sampling Strategy
    this.strategies.set('margin', {
      name: 'Margin Sampling',
      selectSamples: (predictions: Biasprediction[], count: number): string[] => {
        // Select samples where margin between top two predictions is smallest
        const withMargin = predictions.map(p => {
          // Simulate margin calculation from bias score and confidence
          const margin = Math.abs(p.biasScore) * p.confidence;
          return { ...p, margin };
        });

        return withMargin
          .sort((a, b) => a.margin - b.margin) // smallest margin first
          .slice(0, count)
          .map(p => p.articleId);
      }
    });
  }

  private setupWorkers(): void {
    // Worker for model retraining
    new Worker('model-retraining', async (job: Job) => {
      const { trainingData, modelVersion } = job.data;
      logger.info(`Starting model retraining job: ${job.id}`);
      
      try {
        await this.performModelRetraining(trainingData, modelVersion);
        logger.info(`Model retraining completed: ${job.id}`);
        return { success: true, modelVersion };
      } catch (error) {
        logger.error(`Model retraining failed: ${job.id}`, error);
        throw error;
      }
    }, { connection: this.redis });

    // Worker for processing labeling tasks
    new Worker('active-learning-labeling', async (job: Job) => {
      const { articles, strategy, count } = job.data;
      logger.info(`Processing active learning selection: ${job.id}`);
      
      try {
        const tasks = await this.generateLabelingTasks(articles, strategy, count);
        logger.info(`Generated ${tasks.length} labeling tasks: ${job.id}`);
        return { success: true, tasks };
      } catch (error) {
        logger.error(`Active learning selection failed: ${job.id}`, error);
        throw error;
      }
    }, { connection: this.redis });
  }

  /**
   * Main active learning pipeline
   */
  async processNewArticles(articles: Article[]): Promise<{
    predictions: Biasprediction[];
    labelingTasks: LabelingTask[];
    retrainingTriggered: boolean;
  }> {
    try {
      logger.info(`Processing ${articles.length} new articles for active learning`);
      
      // Step 1: Get predictions for all articles
      const predictions = await Promise.all(
        articles.map(article => 
          this.mlService.predictBias(article.content, article.id)
        )
      );

      // Step 2: Identify uncertain samples for labeling
      const uncertainPredictions = predictions.filter(
        p => p.confidence < config.activeLearning.uncertaintyThreshold
      );

      // Step 3: Generate labeling tasks using active learning strategy
      const labelingTasks = await this.selectSamplesForLabeling(
        uncertainPredictions,
        articles,
        'uncertainty', // default strategy
        Math.min(uncertainPredictions.length, this.getRemainingLabelingBudget())
      );

      // Step 4: Check if we need to trigger retraining
      const retrainingTriggered = await this.checkAndTriggerRetraining();

      // Step 5: Store results
      await this.storePredictions(predictions);
      await this.storeLabelingTasks(labelingTasks);

      logger.info(`Active learning processing complete: ${predictions.length} predictions, ${labelingTasks.length} labeling tasks`);
      
      return {
        predictions,
        labelingTasks,
        retrainingTriggered
      };
    } catch (error) {
      logger.error('Error in active learning pipeline:', error);
      throw new Error(`Active learning processing failed: ${error}`);
    }
  }

  /**
   * Select samples for labeling using specified strategy
   */
  async selectSamplesForLabeling(
    predictions: Biasprediction[],
    articles: Article[],
    strategyName: string = 'uncertainty',
    count: number = 10
  ): Promise<LabelingTask[]> {
    const strategy = this.strategies.get(strategyName);
    if (!strategy) {
      throw new Error(`Unknown active learning strategy: ${strategyName}`);
    }

    // Get selected article IDs
    const selectedIds = strategy.selectSamples(predictions, count);
    
    // Create labeling tasks
    const tasks: LabelingTask[] = [];
    
    for (const articleId of selectedIds) {
      const article = articles.find(a => a.id === articleId);
      const prediction = predictions.find(p => p.articleId === articleId);
      
      if (article && prediction) {
        const task: LabelingTask = {
          id: `task-${Date.now()}-${articleId}`,
          articleId: article.id,
          articleTitle: article.title,
          articleContent: article.content,
          currentPrediction: prediction,
          priority: 1 - prediction.confidence, // higher priority for lower confidence
          createdAt: new Date(),
          deadline: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days deadline
        };
        tasks.push(task);
      }
    }

    return tasks.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Process human annotation feedback
   */
  async processAnnotationFeedback(feedback: AnnotationFeedback): Promise<void> {
    try {
      logger.info(`Processing annotation feedback for task: ${feedback.taskId}`);
      
      // Store the annotation
      await this.storeAnnotation(feedback);
      
      // Update labeling budget
      this.currentLabelsToday++;
      
      // Create training data entry
      const trainingData: ModelTrainingData = {
        articleId: feedback.articleId,
        text: '', // Will be filled from article content
        label: feedback.label,
        source: 'active_learning',
        confidence: feedback.confidence,
        createdAt: new Date()
      };
      
      await this.addToTrainingDataset(trainingData);
      
      // Calculate disagreement score for model evaluation
      const currentPrediction = await this.getCurrentPrediction(feedback.articleId);
      if (currentPrediction) {
        const disagreement = this.calculateDisagreement(
          currentPrediction.biasLabel,
          feedback.label
        );
        
        await this.recordModelPerformance(feedback.articleId, disagreement);
      }
      
      // Check if we have enough new labels to trigger retraining
      const newLabelsCount = await this.getNewLabelsCount();
      if (newLabelsCount >= config.activeLearning.batchSize) {
        await this.triggerRetraining();
      }
      
      logger.info(`Annotation feedback processed successfully: ${feedback.taskId}`);
    } catch (error) {
      logger.error('Error processing annotation feedback:', error);
      throw new Error(`Failed to process annotation: ${error}`);
    }
  }

  /**
   * Get pending labeling tasks for an annotator
   */
  async getPendingTasks(annotatorId?: string, limit: number = 20): Promise<LabelingTask[]> {
    try {
      // TODO: Implement database query for pending tasks
      // For now, return cached tasks from Redis
      const tasksJson = await this.redis.get('pending-labeling-tasks');
      const allTasks: LabelingTask[] = tasksJson ? JSON.parse(tasksJson) : [];
      
      let filteredTasks = allTasks.filter(task => !task.assignedTo || task.assignedTo === annotatorId);
      
      if (annotatorId) {
        // Assign tasks to this annotator
        filteredTasks = filteredTasks.slice(0, limit).map(task => ({
          ...task,
          assignedTo: annotatorId
        }));
        
        // Update cache
        await this.redis.set('pending-labeling-tasks', JSON.stringify(allTasks));
      }
      
      return filteredTasks.slice(0, limit);
    } catch (error) {
      logger.error('Error getting pending tasks:', error);
      throw new Error(`Failed to get pending tasks: ${error}`);
    }
  }

  /**
   * Trigger model retraining with new data
   */
  async triggerRetraining(): Promise<string> {
    try {
      logger.info('Triggering model retraining with active learning data');
      
      // Get all training data
      const trainingData = await this.getTrainingDataset();
      const modelVersion = `v${Date.now()}`;
      
      // Queue retraining job
      const job = await this.retrainingQueue.add('retrain-model', {
        trainingData,
        modelVersion,
        useActiveLearning: true
      });
      
      logger.info(`Model retraining job queued: ${job.id}`);
      return job.id as string;
    } catch (error) {
      logger.error('Error triggering retraining:', error);
      throw new Error(`Failed to trigger retraining: ${error}`);
    }
  }

  /**
   * Get active learning statistics
   */
  async getActiveLearningStats(): Promise<{
    totalPredictions: number;
    totalAnnotations: number;
    pendingTasks: number;
    accuracyImprovement: number;
    labelingBudgetUsed: number;
    labelingBudgetRemaining: number;
    modelVersion: string;
    lastRetraining: Date;
  }> {
    try {
      // TODO: Implement actual database queries
      return {
        totalPredictions: await this.getTotalPredictionsCount(),
        totalAnnotations: await this.getTotalAnnotationsCount(),
        pendingTasks: await this.getPendingTasksCount(),
        accuracyImprovement: await this.calculateAccuracyImprovement(),
        labelingBudgetUsed: this.currentLabelsToday,
        labelingBudgetRemaining: this.getRemainingLabelingBudget(),
        modelVersion: await this.getCurrentModelVersion(),
        lastRetraining: await this.getLastRetrainingDate()
      };
    } catch (error) {
      logger.error('Error getting active learning stats:', error);
      throw new Error(`Failed to get stats: ${error}`);
    }
  }

  // Private helper methods
  private async performModelRetraining(trainingData: ModelTrainingData[], modelVersion: string): Promise<void> {
    // TODO: Implement actual model retraining
    // This would involve:
    // 1. Preparing training data in the correct format
    // 2. Running the Python training script
    // 3. Evaluating the new model
    // 4. Updating the model in production if performance improves
    
    logger.info(`Model retraining simulation for version: ${modelVersion}`);
    await new Promise(resolve => setTimeout(resolve, 5000)); // Simulate training time
  }

  private async generateLabelingTasks(articles: Article[], strategy: string, count: number): Promise<LabelingTask[]> {
    // This is called by the worker queue
    const predictions = await Promise.all(
      articles.map(article => this.mlService.predictBias(article.content, article.id))
    );
    
    return this.selectSamplesForLabeling(predictions, articles, strategy, count);
  }

  private getRemainingLabelingBudget(): number {
    // Reset daily budget if needed
    const today = new Date();
    if (today.toDateString() !== this.lastResetDate.toDateString()) {
      this.currentLabelsToday = 0;
      this.lastResetDate = today;
    }
    
    return Math.max(0, this.labelingBudget - this.currentLabelsToday);
  }

  private calculateDisagreement(predicted: string, actual: string): number {
    if (predicted === actual) return 0;
    
    // Calculate disagreement severity
    const labels = ['left', 'center', 'right'];
    const predIndex = labels.indexOf(predicted);
    const actualIndex = labels.indexOf(actual);
    
    return Math.abs(predIndex - actualIndex) / (labels.length - 1);
  }

  private async checkAndTriggerRetraining(): Promise<boolean> {
    const newLabelsCount = await this.getNewLabelsCount();
    const timeSinceLastRetraining = await this.getTimeSinceLastRetraining();
    
    const shouldRetrain = 
      newLabelsCount >= config.activeLearning.batchSize ||
      timeSinceLastRetraining > config.activeLearning.retrainInterval;
    
    if (shouldRetrain) {
      await this.triggerRetraining();
      return true;
    }
    
    return false;
  }

  // Placeholder methods for database operations
  private async storePredictions(predictions: Biasprediction[]): Promise<void> {
    await this.redis.lpush('predictions', ...predictions.map(p => JSON.stringify(p)));
  }

  private async storeLabelingTasks(tasks: LabelingTask[]): Promise<void> {
    const existing = await this.redis.get('pending-labeling-tasks');
    const allTasks = existing ? JSON.parse(existing) : [];
    allTasks.push(...tasks);
    await this.redis.set('pending-labeling-tasks', JSON.stringify(allTasks));
  }

  private async storeAnnotation(feedback: AnnotationFeedback): Promise<void> {
    await this.redis.lpush('annotations', JSON.stringify(feedback));
  }

  private async addToTrainingDataset(data: ModelTrainingData): Promise<void> {
    await this.redis.lpush('training-data', JSON.stringify(data));
  }

  private async getCurrentPrediction(articleId: string): Promise<Biasprediction | null> {
    // TODO: Implement database query
    return null;
  }

  private async recordModelPerformance(articleId: string, disagreement: number): Promise<void> {
    await this.redis.lpush('model-performance', JSON.stringify({ articleId, disagreement, timestamp: new Date() }));
  }

  private async getNewLabelsCount(): Promise<number> {
    return await this.redis.llen('annotations');
  }

  private async getTrainingDataset(): Promise<ModelTrainingData[]> {
    const data = await this.redis.lrange('training-data', 0, -1);
    return data.map((item: string) => JSON.parse(item));
  }

  private async getTotalPredictionsCount(): Promise<number> {
    return await this.redis.llen('predictions');
  }

  private async getTotalAnnotationsCount(): Promise<number> {
    return await this.redis.llen('annotations');
  }

  private async getPendingTasksCount(): Promise<number> {
    const tasks = await this.redis.get('pending-labeling-tasks');
    return tasks ? JSON.parse(tasks).length : 0;
  }

  private async calculateAccuracyImprovement(): Promise<number> {
    // TODO: Calculate actual accuracy improvement
    return 0.05; // 5% improvement placeholder
  }

  private async getCurrentModelVersion(): Promise<string> {
    return await this.redis.get('current-model-version') || 'v1.0.0';
  }

  private async getLastRetrainingDate(): Promise<Date> {
    const timestamp = await this.redis.get('last-retraining-date');
    return timestamp ? new Date(timestamp) : new Date('2025-01-01');
  }

  private async getTimeSinceLastRetraining(): Promise<number> {
    const lastRetraining = await this.getLastRetrainingDate();
    return Date.now() - lastRetraining.getTime();
  }
}
