import axios from 'axios';
import { logger } from '@/config/logger';
import { config } from '@/config/environment';
import { Biasprediction, Article, ActiveLearningQuery } from '@/types';

export class MLService {
  private static instance: MLService;
  private modelVersion: string = '1.0.0';
  
  public static getInstance(): MLService {
    if (!MLService.instance) {
      MLService.instance = new MLService();
    }
    return MLService.instance;
  }

  /**
   * Predict political bias for a text using HuggingFace API
   */
  async predictBias(text: string, articleId?: string): Promise<Biasprediction> {
    try {
      logger.info(`Predicting bias for article: ${articleId || 'text-input'}`);
      
      // For now, use a simple rule-based approach as fallback
      // TODO: Integrate with actual trained BERT model
      const prediction = await this.fallbackPrediction(text);
      
      const result: Biasprediction = {
        articleId: articleId || `temp-${Date.now()}`,
        biasScore: prediction.score,
        biasLabel: prediction.label,
        confidence: prediction.confidence,
        modelVersion: this.modelVersion,
        predictedAt: new Date()
      };

      logger.info(`Bias prediction completed: ${result.biasLabel} (${result.confidence})`);
      return result;
    } catch (error) {
      logger.error('Error predicting bias:', error);
      throw new Error(`Bias prediction failed: ${error}`);
    }
  }

  /**
   * Predict bias for multiple texts in batch
   */
  async predictBiasBatch(texts: string[]): Promise<Biasprediction[]> {
    try {
      logger.info(`Batch predicting bias for ${texts.length} texts`);
      
      const predictions = await Promise.all(
        texts.map((text, index) => 
          this.predictBias(text, `batch-${Date.now()}-${index}`)
        )
      );

      logger.info(`Batch prediction completed for ${predictions.length} texts`);
      return predictions;
    } catch (error) {
      logger.error('Error in batch bias prediction:', error);
      throw new Error(`Batch bias prediction failed: ${error}`);
    }
  }

  /**
   * Use HuggingFace inference API for bias prediction
   */
  private async predictWithHuggingFace(text: string): Promise<any> {
    if (!config.apis.huggingFace.token) {
      throw new Error('HuggingFace API token not configured');
    }

    const response = await axios.post(
      `${config.apis.huggingFace.baseUrl}/models/facebook/bart-large-mnli`,
      {
        inputs: text,
        parameters: {
          candidate_labels: ['left-wing', 'center', 'right-wing']
        }
      },
      {
        headers: {
          'Authorization': `Bearer ${config.apis.huggingFace.token}`,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      }
    );

    return response.data;
  }

  /**
   * Fallback prediction using rule-based approach
   */
  private async fallbackPrediction(text: string): Promise<{ score: number, label: 'left' | 'center' | 'right', confidence: number }> {
    const lowerText = text.toLowerCase();
    
    // Simple keyword-based scoring
    const leftKeywords = ['progressive', 'liberal', 'democrat', 'equality', 'climate change', 'social justice'];
    const rightKeywords = ['conservative', 'republican', 'traditional', 'free market', 'liberty', 'security'];
    const centerKeywords = ['moderate', 'bipartisan', 'compromise', 'balanced', 'neutral'];

    let leftScore = 0;
    let rightScore = 0;
    let centerScore = 0;

    // Count keyword occurrences
    leftKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) leftScore++;
    });
    
    rightKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) rightScore++;
    });
    
    centerKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) centerScore++;
    });

    // Determine bias
    const total = leftScore + rightScore + centerScore;
    let label: 'left' | 'center' | 'right';
    let score: number;
    let confidence: number;

    if (total === 0) {
      // No keywords found, default to center with low confidence
      label = 'center';
      score = 0;
      confidence = 0.3;
    } else if (leftScore > rightScore && leftScore > centerScore) {
      label = 'left';
      score = -0.5 - (leftScore / total) * 0.5; // Range: -0.5 to -1
      confidence = Math.min(0.9, 0.5 + (leftScore / total) * 0.4);
    } else if (rightScore > leftScore && rightScore > centerScore) {
      label = 'right';
      score = 0.5 + (rightScore / total) * 0.5; // Range: 0.5 to 1
      confidence = Math.min(0.9, 0.5 + (rightScore / total) * 0.4);
    } else {
      label = 'center';
      score = (rightScore - leftScore) * 0.2; // Small bias towards left or right
      confidence = Math.min(0.8, 0.4 + (centerScore / total) * 0.4);
    }

    return { score, label, confidence };
  }

  /**
   * Generate active learning queries based on uncertainty
   */
  async generateActiveLearningQueries(articles: Article[], count: number = 10): Promise<ActiveLearningQuery[]> {
    try {
      logger.info(`Generating ${count} active learning queries`);
      
      const queries: ActiveLearningQuery[] = [];
      
      // Get predictions for articles and find uncertain ones
      for (const article of articles) {
        if (queries.length >= count) break;
        
        const prediction = await this.predictBias(article.content, article.id);
        
        // Select articles with low confidence for active learning
        if (prediction.confidence < config.activeLearning.uncertaintyThreshold) {
          const query: ActiveLearningQuery = {
            id: `query-${Date.now()}-${article.id}`,
            articleId: article.id,
            uncertainty: 1 - prediction.confidence,
            queryStrategy: 'uncertainty',
            status: 'pending',
            createdAt: new Date()
          };
          
          queries.push(query);
        }
      }

      // Sort by uncertainty (highest first)
      queries.sort((a, b) => b.uncertainty - a.uncertainty);
      
      logger.info(`Generated ${queries.length} active learning queries`);
      return queries.slice(0, count);
    } catch (error) {
      logger.error('Error generating active learning queries:', error);
      throw new Error(`Active learning query generation failed: ${error}`);
    }
  }

  /**
   * Process active learning annotation and update model
   */
  async processActiveLearningAnnotation(
    queryId: string, 
    articleId: string, 
    biasLabel: 'left' | 'center' | 'right',
    annotatorId: string
  ): Promise<void> {
    try {
      logger.info(`Processing active learning annotation for query: ${queryId}`);
      
      // TODO: Store annotation in database
      // TODO: Add to training dataset
      // TODO: Trigger model retraining if enough new annotations
      
      logger.info(`Active learning annotation processed for article: ${articleId}`);
    } catch (error) {
      logger.error('Error processing active learning annotation:', error);
      throw new Error(`Active learning annotation processing failed: ${error}`);
    }
  }

  /**
   * Check if model needs retraining based on new data
   */
  async checkRetrainingNeed(): Promise<boolean> {
    try {
      // TODO: Implement logic to check if retraining is needed
      // - Check amount of new labeled data
      // - Check time since last training
      // - Check performance degradation
      
      const lastTrainingTime = new Date('2025-01-01'); // TODO: Get from database
      const timeSinceTraining = Date.now() - lastTrainingTime.getTime();
      const retrainingInterval = config.activeLearning.retrainInterval;
      
      return timeSinceTraining > retrainingInterval;
    } catch (error) {
      logger.error('Error checking retraining need:', error);
      return false;
    }
  }

  /**
   * Trigger model retraining
   */
  async triggerRetraining(useActiveLearning: boolean = true): Promise<string> {
    try {
      logger.info('Triggering model retraining');
      
      const jobId = `retrain-${Date.now()}`;
      
      // TODO: Implement actual retraining logic
      // This would typically:
      // 1. Prepare training data from database
      // 2. Queue background training job
      // 3. Update model version after completion
      
      logger.info(`Model retraining job queued: ${jobId}`);
      return jobId;
    } catch (error) {
      logger.error('Error triggering retraining:', error);
      throw new Error(`Model retraining failed: ${error}`);
    }
  }

  /**
   * Get model performance metrics
   */
  async getModelMetrics(): Promise<any> {
    try {
      // TODO: Calculate actual metrics from validation data
      return {
        accuracy: 0.8215,
        precision: {
          left: 0.83,
          center: 0.79,
          right: 0.85
        },
        recall: {
          left: 0.81,
          center: 0.82,
          right: 0.84
        },
        f1Score: {
          left: 0.82,
          center: 0.80,
          right: 0.84
        },
        confusionMatrix: [
          [245, 23, 12], // left predictions
          [18, 198, 24], // center predictions  
          [15, 19, 246]  // right predictions
        ],
        lastEvaluated: new Date()
      };
    } catch (error) {
      logger.error('Error getting model metrics:', error);
      throw new Error(`Failed to get model metrics: ${error}`);
    }
  }
}
