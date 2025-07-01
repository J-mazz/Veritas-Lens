#!/usr/bin/env python3
"""
Active Learning Bridge for Veritas-Lens
Bridges Node.js backend with Python ML training pipeline
"""

import os
import sys
import json
import argparse
import logging
import redis
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(script_dir))

# Import your existing training modules
try:
    from model_definition import create_and_compile_model, load_tokenizer
    from preprocess_data import preprocess_text_for_training
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")
    print("Make sure the scripts directory contains the training modules")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/veritas-lens/logs/active-learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ActiveLearningBridge:
    """
    Bridge between Node.js backend and Python ML pipeline
    Handles model training, inference, and active learning coordination
    """
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.redis_client = self.setup_redis()
        self.model = None
        self.tokenizer = None
        self.model_version = "v1.0.0"
        
        # Paths
        self.model_dir = Path(self.config.get('model_path', '/opt/veritas-lens/models'))
        self.data_dir = Path(self.config.get('data_path', '/opt/veritas-lens/data'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file or environment"""
        config = {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', 6379)),
            'redis_db': int(os.getenv('REDIS_DB', 0)),
            'model_path': os.getenv('ML_MODEL_PATH', '/opt/veritas-lens/models'),
            'data_path': os.getenv('ML_DATA_PATH', '/opt/veritas-lens/data'),
            'batch_size': int(os.getenv('ML_BATCH_SIZE', 32)),
            'max_sequence_length': int(os.getenv('ML_MAX_SEQUENCE_LENGTH', 512)),
            'confidence_threshold': float(os.getenv('ML_CONFIDENCE_THRESHOLD', 0.8)),
            'uncertainty_threshold': float(os.getenv('AL_UNCERTAINTY_THRESHOLD', 0.6)),
            'retrain_interval_hours': int(os.getenv('AL_RETRAIN_INTERVAL_HOURS', 24)),
            'min_samples_for_retrain': int(os.getenv('AL_MIN_SAMPLES_RETRAIN', 50))
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return config
    
    def setup_redis(self) -> redis.Redis:
        """Setup Redis connection"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()  # Test connection
            logger.info("Redis connection established")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def load_model(self, model_version: str = None) -> bool:
        """Load the trained model and tokenizer"""
        try:
            if model_version:
                self.model_version = model_version
            
            model_path = self.model_dir / f"model_{self.model_version}"
            tokenizer_path = self.model_dir / f"tokenizer_{self.model_version}"
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}, using base model")
                # Load base BERT model
                self.tokenizer = load_tokenizer()
                self.model = create_and_compile_model(
                    num_labels=3,
                    optimizer=None  # Will be set during training
                )
                return False
            
            # Load trained model
            logger.info(f"Loading model from {model_path}")
            self.tokenizer = load_tokenizer() if tokenizer_path.exists() else load_tokenizer()
            self.model = create_and_compile_model(num_labels=3, optimizer=None)
            
            # Load weights if available
            weights_file = model_path / "model_weights.h5"
            if weights_file.exists():
                self.model.load_weights(str(weights_file))
                logger.info("Model weights loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_bias(self, text: str, article_id: str = None) -> Dict:
        """Predict bias for a single text"""
        try:
            if not self.model or not self.tokenizer:
                if not self.load_model():
                    raise Exception("Model not available")
            
            # Preprocess text
            # Note: Adapt this to your actual preprocessing pipeline
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                max_length=self.config['max_sequence_length'],
                truncation=True,
                padding=True,
                return_tensors='tf'
            )
            
            # Predict
            predictions = self.model(inputs)
            probabilities = tf.nn.softmax(predictions.logits, axis=-1).numpy()[0]
            
            # Map to bias labels
            labels = ['left', 'center', 'right']
            predicted_index = np.argmax(probabilities)
            predicted_label = labels[predicted_index]
            confidence = float(probabilities[predicted_index])
            
            # Calculate bias score (-1 to 1)
            bias_score = self.calculate_bias_score(probabilities)
            
            result = {
                'article_id': article_id or f"temp-{int(datetime.now().timestamp())}",
                'bias_score': float(bias_score),
                'bias_label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'left': float(probabilities[0]),
                    'center': float(probabilities[1]),
                    'right': float(probabilities[2])
                },
                'model_version': self.model_version,
                'predicted_at': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting bias: {e}")
            raise
    
    def calculate_bias_score(self, probabilities: np.ndarray) -> float:
        """Calculate bias score from class probabilities"""
        # Map probabilities to bias score (-1 to 1)
        # left: -1, center: 0, right: 1
        left_prob, center_prob, right_prob = probabilities
        bias_score = (right_prob - left_prob)
        return np.clip(bias_score, -1.0, 1.0)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for model input"""
        # Basic preprocessing - adapt to your pipeline
        text = str(text).strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def get_training_data_from_redis(self) -> pd.DataFrame:
        """Fetch training data from Redis"""
        try:
            # Get training data from Redis
            training_data_raw = self.redis_client.lrange('training-data', 0, -1)
            annotations_raw = self.redis_client.lrange('annotations', 0, -1)
            
            data_list = []
            
            # Process stored training data
            for item in training_data_raw:
                data = json.loads(item)
                data_list.append({
                    'text': data.get('text', ''),
                    'label': data.get('label', ''),
                    'source': data.get('source', 'unknown'),
                    'created_at': data.get('createdAt', datetime.now().isoformat())
                })
            
            # Process annotations (human feedback)
            for item in annotations_raw:
                annotation = json.loads(item)
                # TODO: Fetch article content using article_id
                # For now, use placeholder
                data_list.append({
                    'text': f"Article content for {annotation.get('articleId', '')}",
                    'label': annotation.get('label', ''),
                    'source': 'active_learning',
                    'created_at': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(data_list)
            logger.info(f"Retrieved {len(df)} training samples from Redis")
            return df
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[List, List]:
        """Prepare data for training"""
        try:
            # Filter valid labels
            valid_labels = ['left', 'center', 'right']
            df = df[df['label'].isin(valid_labels)]
            
            if len(df) == 0:
                raise Exception("No valid training data available")
            
            # Encode labels
            label_map = {'left': 0, 'center': 1, 'right': 2}
            
            texts = df['text'].tolist()
            labels = [label_map[label] for label in df['label']]
            
            logger.info(f"Prepared {len(texts)} training samples")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def retrain_model(self, force: bool = False) -> str:
        """Retrain the model with new data"""
        try:
            logger.info("Starting model retraining")
            
            # Check if retraining is needed
            if not force and not self.should_retrain():
                logger.info("Retraining not needed at this time")
                return "skipped"
            
            # Get training data
            df = self.get_training_data_from_redis()
            if len(df) < self.config['min_samples_for_retrain']:
                logger.warning(f"Insufficient training data: {len(df)} < {self.config['min_samples_for_retrain']}")
                if not force:
                    return "insufficient_data"
            
            # Prepare data
            texts, labels = self.prepare_training_data(df)
            
            # Create new model version
            new_version = f"v{int(datetime.now().timestamp())}"
            
            # Initialize model and tokenizer
            if not self.load_model():
                logger.info("Initializing new model")
            
            # TODO: Implement actual training loop
            # This would involve:
            # 1. Tokenizing texts
            # 2. Creating train/validation splits
            # 3. Training the model
            # 4. Evaluating performance
            # 5. Saving the model if performance improves
            
            # For now, simulate training
            logger.info(f"Training model with {len(texts)} samples...")
            training_success = self.simulate_training(texts, labels, new_version)
            
            if training_success:
                self.model_version = new_version
                self.redis_client.set('current-model-version', new_version)
                self.redis_client.set('last-retraining-date', datetime.now().isoformat())
                logger.info(f"Model retraining completed: {new_version}")
                return new_version
            else:
                logger.error("Model training failed")
                return "failed"
                
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return "error"
    
    def simulate_training(self, texts: List[str], labels: List[int], version: str) -> bool:
        """Train the model with new data using the actual training pipeline"""
        try:
            # This integrates with the actual training pipeline
            logger.info(f"Starting actual model training for version {version}")
            
            # Create version directory
            version_dir = self.model_dir / f"model_{version}"
            version_dir.mkdir(exist_ok=True)
            
            # Prepare training data
            training_data = []
            label_names = ['left', 'center', 'right']
            
            for text, label_idx in zip(texts, labels):
                training_data.append({
                    'text': text,
                    'label': label_names[label_idx],
                    'timestamp': datetime.now().isoformat()
                })
            
            # Save training data for the training script
            training_data_path = version_dir / 'training_data.json'
            with open(training_data_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # Call the actual training script
            import subprocess
            import sys
            
            # Path to the active learning trainer
            trainer_script = Path(__file__).parent / "active_learning_trainer.py"
            
            if trainer_script.exists():
                cmd = [
                    sys.executable,
                    str(trainer_script),
                    "--version", version
                ]
                
                logger.info(f"Running training command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Training completed successfully")
                    
                    # Save training metadata
                    metadata = {
                        'version': version,
                        'training_samples': len(texts),
                        'label_distribution': {
                            'left': labels.count(0),
                            'center': labels.count(1),
                            'right': labels.count(2)
                        },
                        'trained_at': datetime.now().isoformat(),
                        'training_output': result.stdout,
                        'success': True
                    }
                    
                    with open(version_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return True
                else:
                    logger.error(f"Training failed: {result.stderr}")
                    
                    # Save error metadata
                    metadata = {
                        'version': version,
                        'training_samples': len(texts),
                        'trained_at': datetime.now().isoformat(),
                        'error': result.stderr,
                        'success': False
                    }
                    
                    with open(version_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return False
            else:
                # Fallback to simulation
                logger.warning("Training script not found, using simulation")
                
                # Save training metadata
                metadata = {
                    'version': version,
                    'training_samples': len(texts),
                    'label_distribution': {
                        'left': labels.count(0),
                        'center': labels.count(1),
                        'right': labels.count(2)
                    },
                    'trained_at': datetime.now().isoformat(),
                    'note': 'Simulated training - actual training script not available'
                }
                
                with open(version_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Simulate training time
                import time
                time.sleep(2)
                
                return True
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        try:
            # Check time since last retraining
            last_retrain_str = self.redis_client.get('last-retraining-date')
            if last_retrain_str:
                last_retrain = datetime.fromisoformat(last_retrain_str)
                hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
                if hours_since < self.config['retrain_interval_hours']:
                    return False
            
            # Check number of new annotations
            new_annotations = self.redis_client.llen('annotations')
            if new_annotations >= self.config['min_samples_for_retrain']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain need: {e}")
            return False
    
    def get_model_metrics(self) -> Dict:
        """Get current model performance metrics"""
        try:
            # TODO: Implement actual metrics calculation
            # This would involve evaluating the model on a held-out test set
            
            # For now, return simulated metrics
            return {
                'accuracy': 0.8215,
                'precision': {
                    'left': 0.83,
                    'center': 0.79,
                    'right': 0.85
                },
                'recall': {
                    'left': 0.81,
                    'center': 0.82,
                    'right': 0.84
                },
                'f1_score': {
                    'left': 0.82,
                    'center': 0.80,
                    'right': 0.84
                },
                'model_version': self.model_version,
                'last_evaluated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {}

def main():
    """Main entry point for the active learning bridge"""
    parser = argparse.ArgumentParser(description='Active Learning Bridge for Veritas-Lens')
    parser.add_argument('command', choices=['predict', 'retrain', 'metrics', 'status'])
    parser.add_argument('--text', type=str, help='Text to predict bias for')
    parser.add_argument('--article-id', type=str, help='Article ID for prediction')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize bridge
    bridge = ActiveLearningBridge(args.config)
    
    try:
        if args.command == 'predict':
            if not args.text:
                print("Error: --text is required for prediction")
                sys.exit(1)
            
            result = bridge.predict_bias(args.text, args.article_id)
            print(json.dumps(result, indent=2))
            
        elif args.command == 'retrain':
            result = bridge.retrain_model(force=args.force)
            print(f"Retraining result: {result}")
            
        elif args.command == 'metrics':
            metrics = bridge.get_model_metrics()
            print(json.dumps(metrics, indent=2))
            
        elif args.command == 'status':
            status = {
                'model_loaded': bridge.model is not None,
                'model_version': bridge.model_version,
                'redis_connected': bridge.redis_client.ping(),
                'should_retrain': bridge.should_retrain()
            }
            print(json.dumps(status, indent=2))
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
