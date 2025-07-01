#!/usr/bin/env python3
"""
Comprehensive Active Learning Training Script for Veritas-Lens
Handles TF/Torch weight inconsistencies and data pipeline issues
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import logging
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import requests
import feedparser
import re
from typing import Dict, List, Tuple, Optional
import sqlite3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers with TensorFlow backend
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from transformers import (
    TFBertForSequenceClassification, 
    BertTokenizerFast,
    AutoTokenizer,
    pipeline
)

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'/tmp/veritas_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataPipeline:
    """Enhanced data pipeline that generates sufficient training data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_path = config.get('database_path', '/tmp/veritas_articles.db')
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for article storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT,
                published_date TEXT,
                bias_label TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_labeled INTEGER DEFAULT 0,
                labeling_method TEXT DEFAULT 'automatic'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                model_version TEXT,
                accuracy REAL,
                loss REAL,
                f1_score REAL,
                training_samples INTEGER,
                validation_samples INTEGER,
                epochs_trained INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def scrape_news_sources(self) -> List[Dict]:
        """Scrape news from multiple sources to build comprehensive dataset"""
        sources = [
            # RSS Feeds
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://www.foxnews.com/xmlfeed/us.xml',
            'https://www.npr.org/rss/rss.php?id=1001',
            'https://feeds.washingtonpost.com/rss/politics',
            'https://www.politico.com/rss/politicopicks.xml',
            'https://thehill.com/news/feed/',
            'https://feeds.reuters.com/reuters/politicsNews',
            'https://feeds.reuters.com/reuters/domesticNews',
            'https://feeds.nbcnews.com/nbcnews/public/news',
            
            # Additional sources for diversity
            'https://feeds.ap.org/ap/politics',
            'https://feeds.guardian.co.uk/theguardian/politics/rss',
            'https://feeds.skynews.com/feeds/rss/politics.xml'
        ]
        
        articles = []
        
        for source_url in sources:
            try:
                logger.info(f"Scraping {source_url}")
                feed = feedparser.parse(source_url)
                
                for entry in feed.entries[:50]:  # Limit per source
                    article = {
                        'title': entry.title,
                        'content': self.extract_content(entry),
                        'source': source_url,
                        'url': entry.link,
                        'published_date': entry.get('published', ''),
                        'bias_label': self.auto_label_article(entry.title, self.extract_content(entry))
                    }
                    
                    if len(article['content']) > 100:  # Filter short content
                        articles.append(article)
                        
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error scraping {source_url}: {e}")
                continue
        
        logger.info(f"Scraped {len(articles)} articles from {len(sources)} sources")
        return articles
    
    def extract_content(self, entry) -> str:
        """Extract meaningful content from RSS entry"""
        content = ""
        
        # Try different content fields
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value if isinstance(entry.content, list) else entry.content
        elif hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description
        
        # Clean HTML tags and normalize
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def auto_label_article(self, title: str, content: str) -> str:
        """Automatic labeling based on source patterns and keywords"""
        text = (title + " " + content).lower()
        
        # Conservative source patterns
        left_patterns = [
            'progressive', 'liberal', 'democratic', 'climate change', 'social justice',
            'inequality', 'medicare for all', 'green new deal', 'minimum wage',
            'gun control', 'reproductive rights'
        ]
        
        right_patterns = [
            'conservative', 'republican', 'traditional values', 'free market',
            'second amendment', 'border security', 'tax cuts', 'deregulation',
            'law and order', 'pro-life', 'family values'
        ]
        
        center_patterns = [
            'bipartisan', 'moderate', 'compromise', 'centrist', 'middle ground',
            'both sides', 'balanced approach', 'pragmatic', 'consensus'
        ]
        
        # Score based on pattern matches
        left_score = sum(1 for pattern in left_patterns if pattern in text)
        right_score = sum(1 for pattern in right_patterns if pattern in text)
        center_score = sum(1 for pattern in center_patterns if pattern in text)
        
        # Default to center if unclear
        if left_score > right_score and left_score > center_score:
            return 'left'
        elif right_score > left_score and right_score > center_score:
            return 'right'
        else:
            return 'center'
    
    def save_articles_to_db(self, articles: List[Dict]):
        """Save articles to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            cursor.execute('''
                INSERT INTO articles (title, content, source, url, published_date, bias_label)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['content'],
                article['source'],
                article['url'],
                article['published_date'],
                article['bias_label']
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(articles)} articles to database")
    
    def load_training_data(self, min_samples_per_class: int = 1000) -> Tuple[List[str], List[str]]:
        """Load training data ensuring sufficient samples per class"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check current distribution
        cursor.execute('''
            SELECT bias_label, COUNT(*) 
            FROM articles 
            WHERE LENGTH(content) > 100 
            GROUP BY bias_label
        ''')
        
        distribution = dict(cursor.fetchall())
        logger.info(f"Current label distribution: {distribution}")
        
        # If we don't have enough data, scrape more
        if any(count < min_samples_per_class for count in distribution.values()):
            logger.info("Insufficient data, scraping more articles...")
            new_articles = self.scrape_news_sources()
            self.save_articles_to_db(new_articles)
        
        # Load balanced dataset
        texts = []
        labels = []
        
        for label in ['left', 'center', 'right']:
            cursor.execute('''
                SELECT title, content 
                FROM articles 
                WHERE bias_label = ? AND LENGTH(content) > 100
                ORDER BY RANDOM()
                LIMIT ?
            ''', (label, min_samples_per_class))
            
            for title, content in cursor.fetchall():
                texts.append(f"{title} {content}")
                labels.append(label)
        
        conn.close()
        
        logger.info(f"Loaded {len(texts)} training samples")
        logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return texts, labels

class RobustTensorFlowTrainer:
    """Robust TensorFlow trainer handling weight inconsistencies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.num_epochs = config.get('num_epochs', 20)
        self.weight_decay = config.get('weight_decay', 0.01)
        
        # Setup GPU
        self.setup_gpu()
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.history = None
        
    def setup_gpu(self):
        """Configure GPU settings for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
                
                # Use mixed precision for better performance
                if self.config.get('use_mixed_precision', True):
                    mixed_precision.set_global_policy('mixed_float16')
                    logger.info("Mixed precision enabled")
                    
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        else:
            logger.warning("No GPU found, using CPU")
    
    def load_tokenizer(self):
        """Load tokenizer with proper error handling"""
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.model_name,
                do_lower_case=True,
                add_special_tokens=True
            )
            logger.info(f"Tokenizer loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def create_model(self):
        """Create TensorFlow model with proper configuration"""
        try:
            # Clear any existing sessions
            tf.keras.backend.clear_session()
            
            # Create model
            self.model = TFBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3,
                from_tf=True,  # Ensure TensorFlow weights
                output_attentions=False,
                output_hidden_states=False
            )
            
            # Configure optimizer
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                epsilon=1e-8
            )
            
            # Handle mixed precision
            if mixed_precision.global_policy().name == 'mixed_float16':
                optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            
            # Compile model
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
            
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            logger.info("Model created and compiled successfully")
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def tokenize_data(self, texts: List[str]) -> Dict:
        """Tokenize text data with proper padding and truncation"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
    
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> tf.data.Dataset:
        """Prepare TensorFlow dataset with proper batching"""
        # Encode labels
        label_map = {'left': 0, 'center': 1, 'right': 2}
        encoded_labels = [label_map[label] for label in labels]
        
        # Tokenize texts
        tokenized = self.tokenize_data(texts)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': encoded_labels
        })
        
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    
    def train(self, train_texts: List[str], train_labels: List[str], 
              val_texts: List[str], val_labels: List[str]) -> Dict:
        """Comprehensive training with monitoring and checkpointing"""
        
        logger.info("Starting comprehensive training...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'/tmp/best_model_{int(time.time())}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                f'/tmp/training_log_{int(time.time())}.csv',
                append=True
            )
        ]
        
        # Custom callback for detailed logging
        class DetailedLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.info(f"Epoch {epoch + 1}/{self.params['epochs']}")
                logger.info(f"  Training Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
                logger.info(f"  Validation Loss: {logs['val_loss']:.4f}, Accuracy: {logs['val_accuracy']:.4f}")
                logger.info(f"  Learning Rate: {self.model.optimizer.learning_rate.numpy():.2e}")
        
        callbacks.append(DetailedLogger())
        
        # Training
        start_time = time.time()
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.num_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate final model
        final_metrics = self.evaluate_model(val_texts, val_labels)
        
        return {
            'history': self.history.history,
            'training_time': training_time,
            'final_metrics': final_metrics
        }
    
    def evaluate_model(self, texts: List[str], labels: List[str]) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model...")
        
        # Prepare dataset
        eval_dataset = self.prepare_dataset(texts, labels)
        
        # Model evaluation
        eval_results = self.model.evaluate(eval_dataset, verbose=0)
        
        # Detailed predictions for metrics
        predictions = []
        true_labels = []
        
        label_map = {'left': 0, 'center': 1, 'right': 2}
        reverse_label_map = {v: k for k, v in label_map.items()}
        
        for batch in eval_dataset:
            batch_predictions = self.model(batch)
            predicted_classes = tf.argmax(batch_predictions.logits, axis=-1)
            
            predictions.extend(predicted_classes.numpy())
            true_labels.extend(batch['labels'].numpy())
        
        # Calculate detailed metrics
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Classification report
        class_names = ['left', 'center', 'right']
        report = classification_report(
            true_labels, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Loss: {eval_results[0]:.4f}")
        logger.info(f"  Accuracy: {eval_results[1]:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"Classification Report:\n{classification_report(true_labels, predictions, target_names=class_names)}")
        
        return {
            'loss': eval_results[0],
            'accuracy': eval_results[1],
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, save_path: str):
        """Save model and tokenizer"""
        try:
            # Save model
            self.model.save_pretrained(save_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)
            
            # Save training config
            config_path = os.path.join(save_path, 'training_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

class ActiveLearningPipeline:
    """Main active learning pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_pipeline = ComprehensiveDataPipeline(config)
        self.trainer = RobustTensorFlowTrainer(config)
        
    def run_comprehensive_training(self):
        """Run the complete active learning training pipeline"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE ACTIVE LEARNING TRAINING")
        logger.info("="*80)
        
        # Step 1: Load/Generate comprehensive training data
        logger.info("Step 1: Loading comprehensive training data...")
        texts, labels = self.data_pipeline.load_training_data(
            min_samples_per_class=self.config.get('min_samples_per_class', 2000)
        )
        
        # Step 2: Split data
        logger.info("Step 2: Splitting data...")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=0.2, 
            stratify=labels, 
            random_state=42
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        
        # Step 3: Initialize model components
        logger.info("Step 3: Initializing model components...")
        self.trainer.load_tokenizer()
        self.trainer.create_model()
        
        # Step 4: Train model
        logger.info("Step 4: Training model...")
        training_results = self.trainer.train(
            train_texts, train_labels,
            val_texts, val_labels
        )
        
        # Step 5: Save model
        logger.info("Step 5: Saving model...")
        save_path = f"/tmp/veritas_model_{int(time.time())}"
        self.trainer.save_model(save_path)
        
        # Step 6: Save training session to database
        self.save_training_session(training_results, save_path)
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return {
            'model_path': save_path,
            'training_results': training_results
        }
    
    def save_training_session(self, results: Dict, model_path: str):
        """Save training session results to database"""
        conn = sqlite3.connect(self.data_pipeline.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_sessions 
            (session_name, model_version, accuracy, loss, f1_score, model_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            f"comprehensive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            f"v{int(time.time())}",
            results['final_metrics']['accuracy'],
            results['final_metrics']['loss'],
            results['final_metrics']['f1_score'],
            model_path
        ))
        
        conn.commit()
        conn.close()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive Active Learning Training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--min-samples', type=int, default=3000, help='Min samples per class')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Base model name')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model_name': args.model_name,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'min_samples_per_class': args.min_samples,
        'weight_decay': 0.01,
        'use_mixed_precision': True,
        'database_path': '/tmp/veritas_comprehensive.db'
    }
    
    logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Initialize and run pipeline
    pipeline = ActiveLearningPipeline(config)
    results = pipeline.run_comprehensive_training()
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")

if __name__ == "__main__":
    main()
