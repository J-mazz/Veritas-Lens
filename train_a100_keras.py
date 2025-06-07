"""
A100 GPU Training Script for Veritas-Lens Political Bias Detection
Pure TensorFlow/Keras implementation for consistency
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VeritasLensA100Trainer:
    def __init__(self, model_name="distilbert-base-uncased", max_length=512):
        """Initialize the trainer with A100 optimizations"""
        self.model_name = model_name
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
        # A100 GPU configuration
        self.setup_gpu()
        self.setup_mixed_precision()
        
    def setup_gpu(self):
        """Configure GPU settings for A100"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            except RuntimeError as e:
                logger.error(f"GPU setup error: {e}")
        else:
            logger.warning("No GPUs found. Using CPU.")
    
    def setup_mixed_precision(self):
        """Enable mixed precision for A100 performance"""
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled for A100 optimization")
    
    def load_and_prepare_data(self):
        """Load and prepare the political bias dataset"""
        logger.info("Loading dataset...")
        
        try:
            # Load the AllSides dataset for political bias
            dataset = load_dataset("mteb/allsides")
            
            # Convert to pandas for easier manipulation
            train_df = pd.DataFrame(dataset['train'])
            test_df = pd.DataFrame(dataset['test'])
            
            # Combine for preprocessing
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Clean and prepare data
            df = df.dropna(subset=['text', 'label'])
            df['text'] = df['text'].astype(str)
            
            # Encode labels
            df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
            
            logger.info(f"Dataset loaded: {len(df)} samples")
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Fallback to synthetic data for testing
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic data for testing"""
        logger.info("Creating synthetic dataset for testing...")
        
        texts = [
            "Conservative policies promote economic growth",
            "Liberal approaches focus on social equality",
            "Centrist views balance multiple perspectives",
            "Right-wing ideology emphasizes traditional values",
            "Left-wing politics prioritizes social justice"
        ] * 200
        
        labels = ['Right', 'Left', 'Center', 'Right', 'Left'] * 200
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        
        return df
    
    def tokenize_data(self, texts):
        """Tokenize texts using TensorFlow Text operations"""
        # Simple tokenization - in production, use TensorFlow Text
        # For now, use basic preprocessing
        processed_texts = []
        for text in texts:
            # Basic text cleaning
            text = str(text).lower()
            # Truncate to max_length characters (simple approach)
            text = text[:self.max_length]
            processed_texts.append(text)
        
        return processed_texts
    
    def create_model(self, vocab_size=10000, num_classes=3):
        """Create TF/Keras model for political bias classification"""
        
        model = keras.Sequential([
            # Text preprocessing layers
            layers.TextVectorization(
                max_tokens=vocab_size,
                output_sequence_length=self.max_length,
                name="text_vectorization"
            ),
            
            # Embedding layer
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=128,
                mask_zero=True,
                name="embedding"
            ),
            
            # Transformer-like architecture
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.1),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer (float32 for numerical stability)
            layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=2e-5):
        """Compile model with A100-optimized settings"""
        
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=1e-7,  # Stability for mixed precision
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
    
    def create_callbacks(self, model_dir="./models"):
        """Create training callbacks for A100 optimization"""
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, "logs"),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train_model(self, batch_size=32, epochs=10):
        """Main training function optimized for A100"""
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].values,
            df['label_encoded'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label_encoded'].values
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.25,
            random_state=42,
            stratify=y_train
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create and compile model
        num_classes = len(self.label_encoder.classes_)
        model = self.create_model(num_classes=num_classes)
        
        # Adapt text vectorization layer
        text_vectorize_layer = model.get_layer("text_vectorization")
        text_vectorize_layer.adapt(X_train)
        
        model = self.compile_model(model)
        
        # Display model architecture
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        logger.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=1)
        
        # Generate predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names)
        logger.info(f"Classification Report:\n{report}")
        
        # Save results
        self.save_results(model, history, test_accuracy, report)
        
        return model, history
    
    def save_results(self, model, history, test_accuracy, report):
        """Save training results and model"""
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model
        model.save(os.path.join(results_dir, "veritas_lens_model.h5"))
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(results_dir, "training_history.csv"))
        
        # Save label encoder
        import pickle
        with open(os.path.join(results_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save results summary
        results = {
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        
        with open(os.path.join(results_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")

def main():
    """Main training execution"""
    logger.info("Starting Veritas-Lens A100 Training")
    
    # Initialize trainer
    trainer = VeritasLensA100Trainer()
    
    # Train model
    model, history = trainer.train_model(
        batch_size=64,  # Increased for A100
        epochs=15
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
