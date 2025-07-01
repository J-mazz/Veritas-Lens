#!/usr/bin/env python3
"""
Active Learning Training Script for Veritas-Lens
Integrates with existing training pipeline for iterative model improvement
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# Add scripts directory to path
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(script_dir))

try:
    from model_definition import create_and_compile_model, load_tokenizer
    import tensorflow as tf
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActiveLearningTrainer:
    """
    Active Learning Training Pipeline for Veritas-Lens
    """
    
    def __init__(self, config_path: str = "/opt/veritas-lens/active-learning/config.json"):
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.current_version = "v1.0"
        
        # Paths
        self.model_dir = Path(self.config['paths']['models'])
        self.data_dir = Path(self.config['paths']['data'])
        self.annotations_dir = Path(self.config['paths']['annotations'])
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {
                "model": {
                    "name": "bert-base-uncased",
                    "num_labels": 3,
                    "max_sequence_length": 256,
                    "batch_size": 32,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01
                },
                "training": {
                    "epochs_per_iteration": 3,
                    "validation_split": 0.1,
                    "early_stopping_patience": 2
                },
                "paths": {
                    "models": "/opt/veritas-lens/models",
                    "data": "/opt/veritas-lens/data",
                    "annotations": "/opt/veritas-lens/annotations"
                }
            }
    
    def load_model(self, version: str = None) -> bool:
        """Load or initialize the model"""
        try:
            if version:
                self.current_version = version
            
            # Load tokenizer
            self.tokenizer = load_tokenizer()
            
            # Create model
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config['model']['learning_rate'],
                weight_decay=self.config['model']['weight_decay']
            )
            
            self.model = create_and_compile_model(
                num_labels=self.config['model']['num_labels'],
                optimizer=optimizer
            )
            
            # Load existing weights if available
            model_path = self.model_dir / f"model_{self.current_version}"
            weights_path = model_path / "model_weights.h5"
            
            if weights_path.exists():
                logger.info(f"Loading model weights from {weights_path}")
                
                # Build model with sample input first
                sample_input = {
                    'input_ids': tf.constant([[1, 2, 3]], dtype=tf.int32),
                    'attention_mask': tf.constant([[1, 1, 1]], dtype=tf.int32)
                }
                _ = self.model(sample_input)
                
                # Load weights
                self.model.load_weights(str(weights_path))
                logger.info("Model weights loaded successfully")
            else:
                logger.info("No existing weights found, using fresh model")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from various sources"""
        try:
            all_data = []
            
            # Load initial labeled data
            initial_data_path = self.data_dir / "initial_labeled"
            if initial_data_path.exists():
                for label_dir in initial_data_path.iterdir():
                    if label_dir.is_dir():
                        label = label_dir.name
                        for text_file in label_dir.glob("*.txt"):
                            with open(text_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                all_data.append({
                                    'text': content,
                                    'label': label,
                                    'source': 'initial',
                                    'file_path': str(text_file)
                                })
            
            # Load annotations from active learning
            for annotation_file in self.annotations_dir.glob("*.json"):
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                    if isinstance(annotations, list):
                        for ann in annotations:
                            all_data.append({
                                'text': ann.get('text', ''),
                                'label': ann.get('label', ''),
                                'source': 'active_learning',
                                'annotator': ann.get('annotator', ''),
                                'confidence': ann.get('confidence', 1.0),
                                'timestamp': ann.get('timestamp', '')
                            })
            
            df = pd.DataFrame(all_data)
            logger.info(f"Loaded {len(df)} training samples")
            
            # Filter valid labels
            valid_labels = ['left', 'center', 'right']
            df = df[df['label'].isin(valid_labels)]
            logger.info(f"After filtering: {len(df)} valid samples")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Preprocess data for training"""
        try:
            # Prepare texts and labels
            texts = df['text'].tolist()
            label_map = {'left': 0, 'center': 1, 'right': 2}
            labels = [label_map[label] for label in df['label']]
            
            # Split into train/validation
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, 
                test_size=self.config['training']['validation_split'],
                stratify=labels,
                random_state=42
            )
            
            # Tokenization function
            def tokenize_function(texts, labels):
                inputs = self.tokenizer(
                    texts,
                    max_length=self.config['model']['max_sequence_length'],
                    truncation=True,
                    padding=True,
                    return_tensors='tf'
                )
                return inputs, tf.constant(labels)
            
            # Create datasets
            batch_size = self.config['model']['batch_size']
            
            train_inputs, train_labels_tensor = tokenize_function(train_texts, train_labels)
            val_inputs, val_labels_tensor = tokenize_function(val_texts, val_labels)
            
            train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_inputs), train_labels_tensor
            )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((
                dict(val_inputs), val_labels_tensor
            )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            logger.info(f"Created training dataset with {len(train_texts)} samples")
            logger.info(f"Created validation dataset with {len(val_texts)} samples")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train_model(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> Dict:
        """Train the model"""
        try:
            logger.info("Starting model training...")
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config['training']['early_stopping_patience'],
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-7
                )
            ]
            
            # Train the model
            epochs = self.config['training']['epochs_per_iteration']
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get final metrics
            final_metrics = {
                'train_loss': float(history.history['loss'][-1]),
                'train_accuracy': float(history.history['accuracy'][-1]),
                'val_loss': float(history.history['val_loss'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'epochs_trained': len(history.history['loss']),
                'training_time': datetime.now().isoformat()
            }
            
            logger.info(f"Training completed. Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def save_model(self, version: str, metrics: Dict) -> bool:
        """Save the trained model"""
        try:
            model_dir = self.model_dir / f"model_{version}"
            model_dir.mkdir(exist_ok=True)
            
            # Save model weights
            weights_path = model_dir / "model_weights.h5"
            self.model.save_weights(str(weights_path))
            
            # Save tokenizer
            tokenizer_path = model_dir / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            
            # Save metadata
            metadata = {
                'version': version,
                'model_config': self.config['model'],
                'training_metrics': metrics,
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def run_training_iteration(self, version: str = None) -> str:
        """Run one iteration of active learning training"""
        try:
            if not version:
                version = f"v{int(datetime.now().timestamp())}"
            
            logger.info(f"Starting training iteration: {version}")
            
            # Load model
            if not self.load_model():
                logger.error("Failed to load model")
                return "failed"
            
            # Load and preprocess data
            df = self.load_training_data()
            if len(df) == 0:
                logger.error("No training data available")
                return "no_data"
            
            train_dataset, val_dataset = self.preprocess_data(df)
            
            # Train model
            metrics = self.train_model(train_dataset, val_dataset)
            
            # Save model
            if self.save_model(version, metrics):
                logger.info(f"Training iteration {version} completed successfully")
                return version
            else:
                logger.error("Failed to save model")
                return "save_failed"
                
        except Exception as e:
            logger.error(f"Error in training iteration: {e}")
            return "error"

def main():
    parser = argparse.ArgumentParser(description="Active Learning Training for Veritas-Lens")
    parser.add_argument("--config", default="/opt/veritas-lens/active-learning/config.json", 
                       help="Path to configuration file")
    parser.add_argument("--version", help="Model version to train")
    parser.add_argument("--force", action="store_true", help="Force training even with limited data")
    
    args = parser.parse_args()
    
    trainer = ActiveLearningTrainer(args.config)
    result = trainer.run_training_iteration(args.version)
    
    if result in ["failed", "no_data", "save_failed", "error"]:
        logger.error(f"Training failed: {result}")
        sys.exit(1)
    else:
        logger.info(f"Training completed successfully: {result}")
        sys.exit(0)

if __name__ == "__main__":
    main()
