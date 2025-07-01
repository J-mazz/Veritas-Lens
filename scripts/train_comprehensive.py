#!/usr/bin/env python3
"""
Comprehensive Veritas-Lens Training Script
Production-ready training with extensive monitoring, checkpointing, and validation
Designed for long-running training sessions with GPU optimization

Features:
- Advanced learning rate scheduling
- Comprehensive metrics tracking
- Model checkpointing and recovery
- GPU memory optimization
- Data augmentation
- Early stopping with patience
- TensorBoard integration
- Model comparison and selection
- Automated hyperparameter logging
"""

import os
import sys
import json
import time
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFBertForSequenceClassification, BertTokenizerFast, AutoTokenizer
from datasets import load_dataset

# Enable mixed precision for better GPU utilization
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveTrainer:
    """
    Production-ready trainer for Veritas-Lens bias detection model
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_environment()
        self.setup_directories()
        self.setup_gpu()
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_times': [],
            'gpu_memory': []
        }
        
    def setup_environment(self):
        """Configure training environment"""
        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
    def setup_directories(self):
        """Create necessary directories"""
        self.base_dir = Path(self.config['output_dir'])
        self.checkpoint_dir = self.base_dir / 'checkpoints'
        self.logs_dir = self.base_dir / 'logs'
        self.models_dir = self.base_dir / 'models'
        self.plots_dir = self.base_dir / 'plots'
        self.metrics_dir = self.base_dir / 'metrics'
        
        for directory in [self.checkpoint_dir, self.logs_dir, self.models_dir, 
                         self.plots_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created directories in {self.base_dir}")
        
    def setup_gpu(self):
        """Configure GPU settings"""
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU available - enabling mixed precision training")
            self.strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")
        else:
            logger.warning("No GPU detected - falling back to CPU")
            self.strategy = tf.distribute.get_strategy()
            
    def load_and_prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare training, validation, and test datasets"""
        logger.info("Loading and preparing datasets...")
        
        # Load datasets from multiple sources for robustness
        datasets = []
        
        # Dataset 1: Political bias dataset
        try:
            dataset1 = load_dataset("siddharthmb/article-bias-prediction-media-splits")
            df1 = pd.DataFrame(dataset1['train'])
            df1 = self.preprocess_dataset(df1, 'content', 'bias_text')
            datasets.append(df1)
            logger.info(f"Loaded dataset 1: {len(df1)} samples")
        except Exception as e:
            logger.warning(f"Could not load dataset 1: {e}")
            
        # Dataset 2: Additional bias dataset
        try:
            dataset2 = load_dataset("valurank/SemEval23-Task3-EN")
            df2 = pd.DataFrame(dataset2['train'])
            df2 = self.preprocess_dataset(df2, 'text', 'label')
            datasets.append(df2)
            logger.info(f"Loaded dataset 2: {len(df2)} samples")
        except Exception as e:
            logger.warning(f"Could not load dataset 2: {e}")
            
        # Combine datasets
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
        else:
            # Fallback to synthetic data if no datasets available
            logger.warning("No datasets loaded, creating synthetic data")
            combined_df = self.create_synthetic_data()
            
        # Balance the dataset
        combined_df = self.balance_dataset(combined_df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(combined_df)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create TensorFlow datasets
        train_ds = self.create_tf_dataset(train_df, is_training=True)
        val_ds = self.create_tf_dataset(val_df, is_training=False)
        test_ds = self.create_tf_dataset(test_df, is_training=False)
        
        return train_ds, val_ds, test_ds
        
    def preprocess_dataset(self, df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
        """Preprocess individual dataset"""
        # Rename columns
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
        
        # Map labels to standard format
        label_mapping = {
            'left': 'left', 'liberal': 'left', 'LEFT': 'left',
            'center': 'center', 'neutral': 'center', 'CENTER': 'center',
            'right': 'right', 'conservative': 'right', 'RIGHT': 'right',
            0: 'left', 1: 'center', 2: 'right'
        }
        
        df['label'] = df['label'].map(label_mapping)
        df = df.dropna(subset=['text', 'label'])
        df = df[df['label'].isin(['left', 'center', 'right'])]
        
        # Clean text
        df['text'] = df['text'].astype(str)
        df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
        df['text'] = df['text'].str.replace(r'[^\w\s]', ' ', regex=True)  # Remove special chars
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)     # Normalize whitespace
        df['text'] = df['text'].str.strip()
        
        # Filter by length
        df = df[df['text'].str.len() >= 50]  # Minimum 50 characters
        df = df[df['text'].str.len() <= 5000]  # Maximum 5000 characters
        
        return df.reset_index(drop=True)
        
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset across classes"""
        min_count = df['label'].value_counts().min()
        balanced_dfs = []
        
        for label in ['left', 'center', 'right']:
            label_df = df[df['label'] == label].sample(n=min_count, random_state=42)
            balanced_dfs.append(label_df)
            
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test with stratification"""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=0.15, stratify=df['label'], random_state=42
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15, stratify=train_val_df['label'], random_state=42
        )
        
        return train_df, val_df, test_df
        
    def create_tf_dataset(self, df: pd.DataFrame, is_training: bool = False) -> tf.data.Dataset:
        """Create TensorFlow dataset with proper preprocessing"""
        # Convert to TensorFlow dataset
        texts = df['text'].values
        labels = df['label'].map({'left': 0, 'center': 1, 'right': 2}).values
        
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        
        # Apply tokenization
        dataset = dataset.map(
            lambda text, label: self.tokenize_function(text, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            # Data augmentation for training
            dataset = dataset.map(
                self.augment_data,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.shuffle(10000)
            
        # Batch and prefetch
        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def tokenize_function(self, text, label):
        """Tokenize text for BERT"""
        # Load tokenizer if not already loaded
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            
        # Tokenize
        encoded = self.tokenizer(
            text.numpy().decode('utf-8'),
            truncation=True,
            padding='max_length',
            max_length=self.config['max_sequence_length'],
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0]
        }, label
        
    def augment_data(self, features, label):
        """Apply data augmentation"""
        # For text data, we can apply techniques like:
        # - Random token masking
        # - Synonym replacement
        # For now, we'll implement simple token dropout
        
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        
        # Random token dropout (5% chance)
        if tf.random.uniform([]) < 0.05:
            mask_positions = tf.random.uniform(tf.shape(input_ids)) < 0.1
            input_ids = tf.where(mask_positions, 103, input_ids)  # 103 is [MASK] token
            
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, label
        
    def create_model(self) -> tf.keras.Model:
        """Create the BERT-based model"""
        logger.info("Creating BERT model for sequence classification...")
        
        with self.strategy.scope():
            # Load pre-trained BERT
            bert_model = TFBertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=3,
                classifier_dropout=self.config['dropout_rate']
            )
            
            # Create inputs
            input_ids = tf.keras.Input(shape=(self.config['max_sequence_length'],), dtype=tf.int32, name='input_ids')
            attention_mask = tf.keras.Input(shape=(self.config['max_sequence_length'],), dtype=tf.int32, name='attention_mask')
            
            # Get BERT outputs
            outputs = bert_model([input_ids, attention_mask])
            
            # Add additional layers if specified
            if self.config.get('add_custom_head', False):
                x = tf.keras.layers.Dropout(self.config['dropout_rate'])(outputs.last_hidden_state[:, 0, :])
                x = tf.keras.layers.Dense(512, activation='relu')(x)
                x = tf.keras.layers.Dropout(self.config['dropout_rate'])(x)
                predictions = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
                
                model = tf.keras.Model(
                    inputs=[input_ids, attention_mask],
                    outputs=predictions
                )
            else:
                model = tf.keras.Model(
                    inputs=[input_ids, attention_mask],
                    outputs=outputs.logits
                )
            
            # Compile model
            optimizer = self.create_optimizer()
            
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'sparse_categorical_crossentropy']
            )
            
        logger.info(f"Model created with {model.count_params():,} parameters")
        return model
        
    def create_optimizer(self):
        """Create optimizer with learning rate schedule"""
        initial_lr = self.config['learning_rate']
        
        # Learning rate schedule
        if self.config['lr_schedule'] == 'cosine':
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=self.config['epochs'] * self.config['steps_per_epoch'],
                alpha=0.1
            )
        elif self.config['lr_schedule'] == 'exponential':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_lr,
                decay_steps=self.config['steps_per_epoch'],
                decay_rate=0.95
            )
        else:
            lr_schedule = initial_lr
            
        # Create optimizer
        if self.config['optimizer'] == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=self.config['weight_decay'],
                beta_1=0.9,
                beta_2=0.999
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999
            )
            
        # Wrap with mixed precision optimizer
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        return optimizer
        
    def create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks"""
        callbacks_list = []
        
        # Model checkpointing
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(self.logs_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks_list.append(tensorboard)
        
        # Custom callback for detailed logging
        custom_callback = CustomLoggingCallback(
            self.metrics_dir,
            self.training_history
        )
        callbacks_list.append(custom_callback)
        
        return callbacks_list
        
    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> tf.keras.Model:
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create model
        model = self.create_model()
        
        # Print model summary
        model.summary()
        
        # Create callbacks
        callbacks_list = self.create_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = None  # Let Keras determine automatically
        
        # Start training
        start_time = time.time()
        
        try:
            history = model.fit(
                train_ds,
                epochs=self.config['epochs'],
                validation_data=val_ds,
                callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch,
                verbose=1
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time/3600:.2f} hours")
            
            # Save training history
            self.save_training_history(history, training_time)
            
            # Save final model
            final_model_path = self.models_dir / 'final_model.h5'
            model.save(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")
            
            return model
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save current state
            interrupted_model_path = self.models_dir / 'interrupted_model.h5'
            model.save(str(interrupted_model_path))
            logger.info(f"Interrupted model saved to {interrupted_model_path}")
            return model
            
    def evaluate_model(self, model: tf.keras.Model, test_ds: tf.data.Dataset) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for batch_x, batch_y in test_ds:
            predictions = model(batch_x, training=False)
            if hasattr(predictions, 'logits'):
                predictions = predictions.logits
                
            proba = tf.nn.softmax(predictions)
            pred_classes = tf.argmax(proba, axis=1)
            
            y_true.extend(batch_y.numpy())
            y_pred.extend(pred_classes.numpy())
            y_pred_proba.extend(proba.numpy())
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Classification report
        class_names = ['left', 'center', 'right']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save evaluation results
        eval_results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Save to file
        eval_path = self.metrics_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
            
        # Create visualizations
        self.create_evaluation_plots(y_true, y_pred, y_pred_proba, cm)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 (Macro): {f1_macro:.4f}")
        logger.info(f"Test F1 (Weighted): {f1_weighted:.4f}")
        
        return eval_results
        
    def create_evaluation_plots(self, y_true, y_pred, y_pred_proba, cm):
        """Create evaluation plots"""
        plt.style.use('seaborn-v0_8')
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['left', 'center', 'right'],
                   yticklabels=['left', 'center', 'right'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # Classification Report Heatmap
        class_names = ['left', 'center', 'right']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        metrics_df = pd.DataFrame({
            'precision': [report[cls]['precision'] for cls in class_names],
            'recall': [report[cls]['recall'] for cls in class_names],
            'f1-score': [report[cls]['f1-score'] for cls in class_names]
        }, index=class_names)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics_df, annot=True, cmap='RdYlBu_r', center=0.5)
        plt.title('Classification Metrics by Class')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'classification_metrics.png', dpi=300)
        plt.close()
        
    def save_training_history(self, history, training_time):
        """Save training history and create plots"""
        # Save history to JSON
        history_dict = {
            'history': history.history,
            'training_time_hours': training_time / 3600,
            'config': self.config
        }
        
        history_path = self.metrics_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2, default=str)
            
        # Create training plots
        self.create_training_plots(history.history)
        
    def create_training_plots(self, history):
        """Create training visualization plots"""
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss and Accuracy plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300)
        plt.close()
        
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for testing"""
        logger.warning("Creating synthetic data for testing")
        
        synthetic_data = []
        templates = {
            'left': [
                "The progressive policies will help working families and reduce inequality",
                "We need more government investment in social programs and healthcare",
                "Climate change requires immediate action and green energy investment"
            ],
            'center': [
                "Both sides have valid points that need to be considered carefully",
                "A balanced approach considering all stakeholders is necessary",
                "We should evaluate policies based on evidence and practical outcomes"
            ],
            'right': [
                "Free market solutions and reduced regulation will drive economic growth",
                "Traditional values and personal responsibility are important foundations",
                "Lower taxes and smaller government enable individual freedom and prosperity"
            ]
        }
        
        for label, texts in templates.items():
            for i, text in enumerate(texts):
                for j in range(500):  # 500 variations per template
                    synthetic_data.append({
                        'text': f"{text} This is variation {j} with additional context about current events and policy implications.",
                        'label': label
                    })
                    
        return pd.DataFrame(synthetic_data)


class CustomLoggingCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed logging"""
    
    def __init__(self, metrics_dir, training_history):
        super().__init__()
        self.metrics_dir = Path(metrics_dir)
        self.training_history = training_history
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch + 1}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {logs['loss']:.4f}, Train Acc: {logs['accuracy']:.4f}")
        logger.info(f"Val Loss: {logs['val_loss']:.4f}, Val Acc: {logs['val_accuracy']:.4f}")
        
        # Store in history
        self.training_history['train_loss'].append(logs['loss'])
        self.training_history['train_accuracy'].append(logs['accuracy'])
        self.training_history['val_loss'].append(logs['val_loss'])
        self.training_history['val_accuracy'].append(logs['val_accuracy'])
        self.training_history['epoch_times'].append(epoch_time)
        
        # Get GPU memory usage
        if tf.config.list_physical_devices('GPU'):
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1024**3
            self.training_history['gpu_memory'].append(gpu_memory)
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
            
        # Save checkpoint metrics
        checkpoint_data = {
            'epoch': epoch + 1,
            'train_loss': logs['loss'],
            'train_accuracy': logs['accuracy'],
            'val_loss': logs['val_loss'],
            'val_accuracy': logs['val_accuracy'],
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.metrics_dir / f'epoch_{epoch + 1}_metrics.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Comprehensive Veritas-Lens Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./training_output', help='Output directory')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'max_sequence_length': args.max_seq_length,
        'early_stopping_patience': args.early_stopping_patience,
        'random_seed': 42,
        'dropout_rate': 0.1,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'lr_schedule': 'cosine',
        'add_custom_head': False,
        'steps_per_epoch': None  # Auto-calculate
    }
    
    logger.info("Starting Comprehensive Veritas-Lens Training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize trainer
    trainer = ComprehensiveTrainer(config)
    
    # Load data
    train_ds, val_ds, test_ds = trainer.load_and_prepare_data()
    
    # Train model
    model = trainer.train(train_ds, val_ds)
    
    # Evaluate model
    eval_results = trainer.evaluate_model(model, test_ds)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final Test Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"Results saved to: {trainer.base_dir}")


if __name__ == "__main__":
    main()
