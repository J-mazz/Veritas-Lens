#!/usr/bin/env python3
"""
Integration Script for Existing Models with Active Learning
Connects pre-trained Veritas-Lens models to the active learning pipeline
"""

import os
import sys
import json
import shutil
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add script directories to path
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent / "scripts"
sys.path.append(str(scripts_dir))

try:
    # Try to import existing training modules
    # These imports might fail if not in the right environment
    import tensorflow as tf
    from transformers import AutoTokenizer, TFBertForSequenceClassification
    print("TensorFlow and Transformers imported successfully")
except ImportError as e:
    print(f"Warning: Could not import ML libraries: {e}")
    print("Make sure you're in the Python environment with ML dependencies")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelIntegrator:
    """
    Integrates existing supervised learning models with active learning
    """
    
    def __init__(self):
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.backend_dir = self.project_root / "Backend"
        self.active_learning_dir = Path("/opt/veritas-lens/active-learning")
        
        # Existing model paths
        self.existing_model_dir = self.project_root / "tokenizer_bert_combined_bs64_adamw_no_mp"
        self.existing_weights_pattern = "best_model_bert_combined_bs64_adamw_no_mp.weights.h5"
        
        # Active learning paths
        self.al_models_dir = self.active_learning_dir / "models"
        self.al_data_dir = self.active_learning_dir / "data"
        
        # Create directories
        self.active_learning_dir.mkdir(parents=True, exist_ok=True)
        self.al_models_dir.mkdir(exist_ok=True)
        self.al_data_dir.mkdir(exist_ok=True)
    
    def find_existing_models(self) -> List[Path]:
        """Find existing trained models"""
        model_paths = []
        
        # Check for tokenizer directory (indicates trained model)
        if self.existing_model_dir.exists():
            model_paths.append(self.existing_model_dir)
            logger.info(f"Found existing model: {self.existing_model_dir}")
        
        # Check for weight files in parent directory
        for weight_file in self.project_root.glob("**/*bert*weights*.h5"):
            logger.info(f"Found weight file: {weight_file}")
        
        # Check saved_model directory if it exists
        saved_model_dir = self.project_root / "saved_model"
        if saved_model_dir.exists():
            for model_dir in saved_model_dir.iterdir():
                if model_dir.is_dir() and "bert" in model_dir.name.lower():
                    model_paths.append(model_dir)
                    logger.info(f"Found saved model: {model_dir}")
        
        return model_paths
    
    def extract_model_info(self, model_path: Path) -> Dict:
        """Extract information about an existing model"""
        info = {
            'path': str(model_path),
            'name': model_path.name,
            'type': 'unknown',
            'has_tokenizer': False,
            'has_weights': False,
            'config_file': None,
            'created_date': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        }
        
        # Check for tokenizer files
        tokenizer_files = ['tokenizer.json', 'vocab.txt', 'tokenizer_config.json']
        if any((model_path / f).exists() for f in tokenizer_files):
            info['has_tokenizer'] = True
            info['type'] = 'tokenizer_dir'
        
        # Check for weight files
        weight_files = list(model_path.glob("*.h5")) + list(model_path.glob("**/*.h5"))
        if weight_files:
            info['has_weights'] = True
            info['weight_files'] = [str(w) for w in weight_files]
        
        # Check for config files
        config_files = list(model_path.glob("config.json")) + list(model_path.glob("metadata.json"))
        if config_files:
            info['config_file'] = str(config_files[0])
        
        return info
    
    def copy_model_to_active_learning(self, model_info: Dict, version_name: str) -> bool:
        """Copy existing model to active learning directory"""
        try:
            source_path = Path(model_info['path'])
            target_path = self.al_models_dir / f"model_{version_name}"
            
            logger.info(f"Copying model from {source_path} to {target_path}")
            
            # Create target directory
            target_path.mkdir(exist_ok=True)
            
            # Copy tokenizer if available
            if model_info['has_tokenizer']:
                tokenizer_target = target_path / "tokenizer"
                tokenizer_target.mkdir(exist_ok=True)
                
                tokenizer_files = ['tokenizer.json', 'vocab.txt', 'tokenizer_config.json']
                for file_name in tokenizer_files:
                    source_file = source_path / file_name
                    if source_file.exists():
                        shutil.copy2(source_file, tokenizer_target / file_name)
                        logger.info(f"Copied {file_name}")
            
            # Copy weight files
            if model_info.get('weight_files'):
                for weight_file_path in model_info['weight_files']:
                    weight_file = Path(weight_file_path)
                    if weight_file.exists():
                        target_weight_file = target_path / "model_weights.h5"
                        shutil.copy2(weight_file, target_weight_file)
                        logger.info(f"Copied weights: {weight_file.name}")
                        break  # Use first weight file found
            
            # Look for weights in parent directory with pattern
            if not model_info.get('weight_files'):
                weight_pattern_files = list(self.project_root.glob(f"**/{self.existing_weights_pattern}"))
                if weight_pattern_files:
                    weight_file = weight_pattern_files[0]
                    target_weight_file = target_path / "model_weights.h5"
                    shutil.copy2(weight_file, target_weight_file)
                    logger.info(f"Copied pattern-matched weights: {weight_file.name}")
            
            # Create metadata for active learning
            al_metadata = {
                'version': version_name,
                'original_path': str(source_path),
                'model_type': 'bert-base-uncased',
                'num_labels': 3,
                'max_sequence_length': 256,
                'imported_at': datetime.now().isoformat(),
                'source': 'supervised_training',
                'original_info': model_info
            }
            
            metadata_file = target_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(al_metadata, f, indent=2)
            
            logger.info(f"Model {version_name} imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error copying model: {e}")
            return False
    
    def create_data_bridge(self) -> bool:
        """Create data bridge for existing datasets"""
        try:
            logger.info("Creating data bridge...")
            
            # Look for existing processed data
            data_sources = []
            
            # Check for processed_combined directory
            processed_combined = self.project_root / "data" / "processed_combined"
            if processed_combined.exists():
                data_sources.append(processed_combined)
            
            # Check for other data directories
            data_dirs = ["data", "datasets", "processed_data"]
            for data_dir_name in data_dirs:
                data_dir = self.project_root / data_dir_name
                if data_dir.exists():
                    data_sources.append(data_dir)
            
            # Create symbolic links or copy small datasets
            for data_source in data_sources:
                target_name = f"imported_{data_source.name}"
                target_path = self.al_data_dir / target_name
                
                if not target_path.exists():
                    try:
                        # Try to create symbolic link first
                        target_path.symlink_to(data_source.absolute())
                        logger.info(f"Created symlink: {target_name} -> {data_source}")
                    except OSError:
                        # If symlink fails, copy for small datasets
                        if self.get_dir_size(data_source) < 100 * 1024 * 1024:  # 100MB
                            shutil.copytree(data_source, target_path)
                            logger.info(f"Copied small dataset: {target_name}")
                        else:
                            logger.warning(f"Dataset too large to copy: {data_source}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating data bridge: {e}")
            return False
    
    def get_dir_size(self, path: Path) -> int:
        """Get directory size in bytes"""
        try:
            total = 0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
            return total
        except:
            return float('inf')  # Return large number if can't calculate
    
    def create_integration_config(self) -> Dict:
        """Create configuration for the integrated system"""
        config = {
            "integration": {
                "created_at": datetime.now().isoformat(),
                "source_project": "veritas-lens-supervised",
                "integration_method": "model_copy_and_bridge"
            },
            "model": {
                "base_architecture": "bert-base-uncased",
                "num_labels": 3,
                "label_mapping": {
                    "0": "left",
                    "1": "center", 
                    "2": "right"
                },
                "max_sequence_length": 256,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "weight_decay": 0.01
            },
            "active_learning": {
                "strategy": "uncertainty_sampling",
                "initial_model": "imported_v1",
                "batch_size": 20,
                "confidence_threshold": 0.8,
                "retrain_threshold": 100,
                "max_iterations": 10
            },
            "paths": {
                "models": str(self.al_models_dir),
                "data": str(self.al_data_dir),
                "annotations": str(self.active_learning_dir / "annotations"),
                "logs": str(self.active_learning_dir / "logs")
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "queues": {
                    "labeling": "labeling-tasks",
                    "retraining": "retraining-jobs",
                    "predictions": "prediction-requests"
                }
            }
        }
        
        # Save config
        config_path = self.active_learning_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Integration config saved to {config_path}")
        return config
    
    def run_integration(self) -> bool:
        """Run the complete integration process"""
        logger.info("Starting model integration process...")
        
        try:
            # Find existing models
            existing_models = self.find_existing_models()
            
            if not existing_models:
                logger.warning("No existing models found")
                logger.info("Creating placeholder for new active learning setup")
                
                # Create empty model structure
                placeholder_model = self.al_models_dir / "model_base"
                placeholder_model.mkdir(exist_ok=True)
                
                placeholder_metadata = {
                    'version': 'base',
                    'model_type': 'bert-base-uncased', 
                    'num_labels': 3,
                    'status': 'placeholder',
                    'created_at': datetime.now().isoformat(),
                    'note': 'Placeholder model - train with active learning'
                }
                
                with open(placeholder_model / "metadata.json", 'w') as f:
                    json.dump(placeholder_metadata, f, indent=2)
            else:
                # Process each existing model
                for i, model_path in enumerate(existing_models):
                    model_info = self.extract_model_info(model_path)
                    version_name = f"imported_v{i+1}"
                    
                    logger.info(f"Processing model: {model_info['name']}")
                    
                    if self.copy_model_to_active_learning(model_info, version_name):
                        logger.info(f"Successfully imported model as {version_name}")
                    else:
                        logger.error(f"Failed to import model: {model_info['name']}")
            
            # Create data bridge
            if self.create_data_bridge():
                logger.info("Data bridge created successfully")
            else:
                logger.warning("Data bridge creation had issues")
            
            # Create integration config
            config = self.create_integration_config()
            
            logger.info("Integration completed successfully!")
            
            # Print summary
            self.print_integration_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return False
    
    def print_integration_summary(self):
        """Print integration summary"""
        print("\n" + "="*60)
        print("MODEL INTEGRATION COMPLETED")
        print("="*60)
        print(f"Active learning directory: {self.active_learning_dir}")
        print(f"Models directory: {self.al_models_dir}")
        print(f"Data directory: {self.al_data_dir}")
        
        print("\nImported models:")
        for model_dir in self.al_models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    print(f"  - {model_dir.name}: {metadata.get('model_type', 'unknown')}")
                else:
                    print(f"  - {model_dir.name}: (no metadata)")
        
        print("\nNext steps:")
        print("1. Start Redis server: redis-server")
        print("2. Start the Backend API server")
        print("3. Run the active learning bridge:")
        print("   python3 Backend/scripts/active_learning_bridge.py --mode daemon")
        print("4. Test the system with new articles")
        print("5. Begin the active learning workflow")

def main():
    parser = argparse.ArgumentParser(description="Integrate existing models with active learning")
    parser.add_argument("--force", action="store_true", help="Force integration even if AL directory exists")
    
    args = parser.parse_args()
    
    integrator = ModelIntegrator()
    
    # Check if already integrated
    if integrator.active_learning_dir.exists() and not args.force:
        config_file = integrator.active_learning_dir / "config.json"
        if config_file.exists():
            print(f"Active learning already set up at {integrator.active_learning_dir}")
            print("Use --force to re-integrate")
            sys.exit(0)
    
    success = integrator.run_integration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
