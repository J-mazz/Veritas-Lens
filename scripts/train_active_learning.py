#!/usr/bin/env python3
"""
Active Learning Training Script for Veritas-Lens
Integrates with existing BERT model and implements active learning strategies
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from typing import List, Tuple, Dict, Optional
import redis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import existing model definition
try:
    from model_definition import create_and_compile_model, load_tokenizer, PRE_TRAINED_MODEL_NAME
    from train_model import encode_directory_examples  # Import encoding function
except ImportError as e:
    print(f"Error importing existing model modules: {e}")
    print("Make sure model_definition.py and train_model.py are in the scripts directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActiveLearningTrainer:
    """
    Active Learning Trainer that extends the existing BERT model
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = self.setup_redis()
        self.model = None
        self.tokenizer = None
        self.class_names = ['left', 'center', 'right']
        self.setup_paths()
        
    def setup_redis(self) -> redis.Redis:
        """Setup Redis connection for coordination with backend"""
        try:
            r = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            r.ping()  # Test connection
            logger.info("Connected to Redis")
            return r
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Return None to continue without Redis (standalone mode)
            return None
    
    def setup_paths(self):
        """Setup directory paths"""
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models" / "active_learning"
        self.data_dir = self.project_root / "data" / "active_learning"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_existing_model(self, model_path: str = None) -> bool:
        """Load existing trained model"""
        try:
            logger.info("Loading existing BERT model...")
            
            # Load tokenizer
            self.tokenizer = load_tokenizer()
            
            # Create model architecture
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.get('learning_rate', 2e-5),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
            
            self.model = create_and_compile_model(
                num_labels=len(self.class_names),
                optimizer=optimizer
            )
            
            # Load weights if available
            if model_path and os.path.exists(model_path):
                # Build model first
                sample_input = self.tokenizer(
                    "Sample text",
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors='tf'
                )
                _ = self.model(sample_input)
                
                self.model.load_weights(model_path)
                logger.info(f"Loaded model weights from {model_path}")
            else:
                logger.info("No existing weights found, using pre-trained BERT")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_training_data_from_redis(self) -> Tuple[List[str], List[str]]:
        """Get training data accumulated through active learning"""
        texts, labels = [], []
        
        if not self.redis_client:
            logger.warning("No Redis connection, using fallback data")
            return self.get_fallback_training_data()
        
        try:
            # Get training data from Redis
            training_data_raw = self.redis_client.lrange('training-data', 0, -1)
            
            for item in training_data_raw:
                data = json.loads(item)
                texts.append(data['text'])
                labels.append(data['label'])
            
            if len(texts) == 0:
                logger.warning("No training data in Redis, using fallback")
                return self.get_fallback_training_data()
            
            logger.info(f"Loaded {len(texts)} training samples from Redis")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error getting training data from Redis: {e}")
            return self.get_fallback_training_data()
    
    def get_fallback_training_data(self) -> Tuple[List[str], List[str]]:
        """Get fallback training data if Redis is not available"""
        try:
            # Load from existing processed data directory
            processed_data_dir = self.project_root / "data" / "processed_combined"
            
            texts, labels = [], []
            
            for split in ['train', 'valid']:
                split_dir = processed_data_dir / split
                if split_dir.exists():
                    for label_dir in split_dir.iterdir():
                        if label_dir.is_dir() and label_dir.name in self.class_names:
                            for text_file in label_dir.glob('*.txt'):
                                with open(text_file, 'r', encoding='utf-8') as f:
                                    text = f.read().strip()
                                    texts.append(text)
                                    labels.append(label_dir.name)
            
            logger.info(f"Loaded {len(texts)} fallback training samples")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error loading fallback data: {e}")
            # Create minimal synthetic data
            return self.create_synthetic_data()
    
    def create_synthetic_data(self) -> Tuple[List[str], List[str]]:
        """Create minimal synthetic data for testing"""
        texts = [
            "Conservative policies promote economic growth through tax cuts",
            "Liberal social programs help reduce inequality in society", 
            "Independent analysis shows mixed results for both approaches",
            "Traditional values guide our understanding of social issues",
            "Progressive reforms address systemic inequalities",
            "Balanced reporting presents multiple perspectives on issues"
        ]
        labels = ['right', 'left', 'center', 'right', 'left', 'center']
        
        logger.info("Using synthetic training data")
        return texts, labels
    
    def uncertainty_sampling(self, texts: List[str], n_samples: int = 50) -> List[int]:
        """Select most uncertain samples for labeling"""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded for uncertainty sampling")
            return []
        
        try:
            # Get predictions for all texts
            uncertainties = []
            
            for i, text in enumerate(texts):
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors='tf'
                )
                
                # Predict
                predictions = self.model(inputs)
                probs = tf.nn.softmax(predictions.logits, axis=-1).numpy()[0]
                
                # Calculate uncertainty (entropy)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                uncertainties.append((i, entropy))
            
            # Sort by uncertainty (highest first)
            uncertainties.sort(key=lambda x: x[1], reverse=True)
            
            # Return indices of most uncertain samples
            return [idx for idx, _ in uncertainties[:n_samples]]
            
        except Exception as e:
            logger.error(f"Error in uncertainty sampling: {e}")
            return []
    
    def train_with_active_learning(self, 
                                   initial_texts: List[str], 
                                   initial_labels: List[str],
                                   epochs: int = 3,
                                   batch_size: int = 16) -> Dict:
        """Train model with active learning approach"""
        try:
            logger.info("Starting active learning training...")
            
            # Prepare training data
            texts_train, texts_val, labels_train, labels_val = train_test_split(
                initial_texts, initial_labels, test_size=0.2, random_state=42,
                stratify=initial_labels
            )
            
            # Convert labels to integers
            label_map = {'left': 0, 'center': 1, 'right': 2}
            y_train = [label_map[label] for label in labels_train]
            y_val = [label_map[label] for label in labels_val]
            
            # Prepare datasets
            train_dataset = self.prepare_dataset(texts_train, y_train, batch_size)
            val_dataset = self.prepare_dataset(texts_val, y_val, batch_size, shuffle=False)
            
            # Training callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=2,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=1,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_accuracy = self.model.evaluate(val_dataset, verbose=0)
            
            # Save model
            model_version = f"v{int(datetime.now().timestamp())}"
            model_path = self.models_dir / f"active_learning_model_{model_version}.weights.h5"
            self.model.save_weights(str(model_path))
            
            # Update Redis with new model info
            if self.redis_client:
                self.redis_client.set('current-model-version', model_version)
                self.redis_client.set('last-retraining-date', datetime.now().isoformat())
            
            results = {
                'model_version': model_version,
                'val_accuracy': float(val_accuracy),
                'val_loss': float(val_loss),
                'training_samples': len(texts_train),
                'model_path': str(model_path),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in active learning training: {e}")
            raise
    
    def prepare_dataset(self, texts: List[str], labels: List[int], batch_size: int, shuffle: bool = True):
        """Prepare TensorFlow dataset"""
        try:
            # Tokenize all texts
            encoded = self.tokenizer(
                texts,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors='tf'
            )
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': labels
            })
            
            if shuffle:
                dataset = dataset.shuffle(1000)
            
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict bias for a batch of texts"""
        if not self.model or not self.tokenizer:
            raise Exception("Model not loaded")
        
        try:
            results = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors='tf'
                )
                
                # Predict
                predictions = self.model(inputs)
                probs = tf.nn.softmax(predictions.logits, axis=-1).numpy()[0]
                
                # Get prediction
                predicted_idx = np.argmax(probs)
                predicted_label = self.class_names[predicted_idx]
                confidence = float(probs[predicted_idx])
                
                # Calculate bias score
                bias_score = float(probs[2] - probs[0])  # right - left
                
                result = {
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'bias_score': bias_score,
                    'probabilities': {
                        'left': float(probs[0]),
                        'center': float(probs[1]),
                        'right': float(probs[2])
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Active Learning Training for Veritas-Lens')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model-path', type=str, help='Path to existing model weights')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--redis-host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Load config
    config = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'redis_db': 0
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    try:
        # Initialize trainer
        trainer = ActiveLearningTrainer(config)
        
        # Load model
        if not trainer.load_existing_model(args.model_path):
            logger.error("Failed to load model")
            return
        
        # Get training data
        texts, labels = trainer.get_training_data_from_redis()
        
        if len(texts) == 0:
            logger.error("No training data available")
            return
        
        logger.info(f"Training with {len(texts)} samples")
        
        # Train model
        results = trainer.train_with_active_learning(
            texts, labels, 
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save results
        results_path = trainer.logs_dir / f"training_results_{results['model_version']}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model version: {results['model_version']}")
        logger.info(f"Validation accuracy: {results['val_accuracy']:.4f}")
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
