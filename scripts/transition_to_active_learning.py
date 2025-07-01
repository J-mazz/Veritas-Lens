#!/usr/bin/env python3
"""
Transition Script for Veritas-Lens Active Learning
Converts existing supervised learning setup to active learning workflow
"""

import os
import sys
import json
import shutil
import argparse
import logging
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from model_definition import create_and_compile_model, load_tokenizer
    from train_model import VeritasLensTrainer  # Assuming we'll refactor the training script
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActiveLearningTransition:
    """
    Handles the transition from supervised to active learning
    """
    
    def __init__(self, base_model_path: str = None):
        self.base_model_path = base_model_path or "/opt/veritas-lens/models"
        self.active_learning_dir = Path("/opt/veritas-lens/active-learning")
        self.scripts_dir = Path(__file__).parent
        
        # Create necessary directories
        self.active_learning_dir.mkdir(parents=True, exist_ok=True)
        (self.active_learning_dir / "models").mkdir(exist_ok=True)
        (self.active_learning_dir / "data").mkdir(exist_ok=True)
        (self.active_learning_dir / "annotations").mkdir(exist_ok=True)
        (self.active_learning_dir / "logs").mkdir(exist_ok=True)
    
    def setup_base_model(self) -> bool:
        """
        Setup the initial model for active learning
        """
        try:
            logger.info("Setting up base model for active learning...")
            
            # Create base model
            optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=0.01)
            model = create_and_compile_model(num_labels=3, optimizer=optimizer)
            
            # Load existing weights if available
            weights_path = self.scripts_dir.parent / "tokenizer_bert_combined_bs64_adamw_no_mp"
            if weights_path.exists():
                logger.info(f"Loading existing model weights from {weights_path}")
                # Note: Adjust this path based on your actual model weights location
                model_weights_path = weights_path.parent / "best_model_bert_combined_bs64_adamw_no_mp.weights.h5"
                if model_weights_path.exists():
                    model.load_weights(str(model_weights_path))
                    logger.info("Model weights loaded successfully")
            
            # Save model for active learning
            model_save_path = self.active_learning_dir / "models" / "base_model"
            model.save(str(model_save_path))
            logger.info(f"Base model saved to {model_save_path}")
            
            # Setup tokenizer
            tokenizer = load_tokenizer()
            tokenizer_save_path = self.active_learning_dir / "models" / "tokenizer"
            tokenizer.save_pretrained(str(tokenizer_save_path))
            logger.info(f"Tokenizer saved to {tokenizer_save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up base model: {e}")
            return False
    
    def create_active_learning_config(self) -> Dict:
        """
        Create configuration for active learning
        """
        config = {
            "model": {
                "name": "bert-base-uncased",
                "num_labels": 3,
                "max_sequence_length": 256,
                "batch_size": 32,  # Smaller batch for active learning
                "learning_rate": 2e-5,
                "weight_decay": 0.01
            },
            "active_learning": {
                "strategy": "uncertainty_sampling",
                "batch_size": 20,  # Number of samples to label per batch
                "max_iterations": 10,
                "confidence_threshold": 0.8,
                "diversity_weight": 0.3,
                "budget_per_iteration": 100
            },
            "training": {
                "epochs_per_iteration": 3,
                "validation_split": 0.1,
                "early_stopping_patience": 2,
                "checkpoint_frequency": 1
            },
            "data": {
                "initial_labeled_size": 1000,
                "unlabeled_pool_size": 10000,
                "validation_size": 500
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "paths": {
                "models": str(self.active_learning_dir / "models"),
                "data": str(self.active_learning_dir / "data"),
                "annotations": str(self.active_learning_dir / "annotations"),
                "logs": str(self.active_learning_dir / "logs")
            },
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        # Save config
        config_path = self.active_learning_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Active learning configuration saved to {config_path}")
        return config
    
    def prepare_initial_dataset(self, labeled_data_path: str = None) -> bool:
        """
        Prepare initial labeled dataset for active learning
        """
        try:
            logger.info("Preparing initial dataset...")
            
            # If labeled data path is provided, use it
            if labeled_data_path and Path(labeled_data_path).exists():
                shutil.copytree(labeled_data_path, self.active_learning_dir / "data" / "initial_labeled")
                logger.info(f"Copied initial labeled data from {labeled_data_path}")
            else:
                # Create placeholder structure
                initial_data_dir = self.active_learning_dir / "data" / "initial_labeled"
                initial_data_dir.mkdir(exist_ok=True)
                
                # Create sample structure
                for label in ["left", "center", "right"]:
                    (initial_data_dir / label).mkdir(exist_ok=True)
                
                logger.info("Created placeholder initial labeled data structure")
            
            # Create unlabeled pool directory
            unlabeled_dir = self.active_learning_dir / "data" / "unlabeled_pool"
            unlabeled_dir.mkdir(exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing initial dataset: {e}")
            return False
    
    def create_active_learning_workflow(self) -> bool:
        """
        Create the active learning workflow scripts
        """
        try:
            logger.info("Creating active learning workflow...")
            
            # Create workflow script
            workflow_script = """#!/usr/bin/env python3
'''
Active Learning Workflow for Veritas-Lens
Manages the iterative active learning process
'''

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "Backend" / "scripts"))

from active_learning_bridge import ActiveLearningBridge

def run_active_learning_iteration():
    '''Run one iteration of active learning'''
    bridge = ActiveLearningBridge()
    
    print("Starting active learning iteration...")
    
    # Step 1: Get unlabeled data and make predictions
    # Step 2: Select most informative samples
    # Step 3: Create labeling tasks
    # Step 4: Wait for annotations
    # Step 5: Retrain model with new annotations
    
    print("Active learning iteration completed")

if __name__ == "__main__":
    run_active_learning_iteration()
"""
            
            workflow_path = self.active_learning_dir / "run_workflow.py"
            with open(workflow_path, 'w') as f:
                f.write(workflow_script)
            
            # Make executable
            workflow_path.chmod(0o755)
            
            logger.info(f"Active learning workflow created at {workflow_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            return False
    
    def setup_cron_jobs(self) -> bool:
        """
        Setup cron jobs for automated active learning
        """
        try:
            logger.info("Setting up cron jobs...")
            
            cron_script = f"""#!/bin/bash
# Veritas-Lens Active Learning Cron Jobs

# Run data aggregation every 6 hours
0 */6 * * * cd {self.scripts_dir} && python3 scrape_rss_feeds.py >> {self.active_learning_dir}/logs/aggregation.log 2>&1

# Run active learning iteration once daily
0 2 * * * cd {self.active_learning_dir} && python3 run_workflow.py >> {self.active_learning_dir}/logs/workflow.log 2>&1

# Model health check every 12 hours
0 */12 * * * cd {self.scripts_dir.parent}/Backend/scripts && python3 active_learning_bridge.py --health-check >> {self.active_learning_dir}/logs/health.log 2>&1
"""
            
            cron_path = self.active_learning_dir / "crontab_setup.sh"
            with open(cron_path, 'w') as f:
                f.write(cron_script)
            
            cron_path.chmod(0o755)
            
            logger.info(f"Cron setup script created at {cron_path}")
            logger.info("Run 'bash crontab_setup.sh' to install cron jobs")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up cron jobs: {e}")
            return False
    
    def run_transition(self, labeled_data_path: str = None) -> bool:
        """
        Run the complete transition to active learning
        """
        logger.info("Starting transition to active learning...")
        
        steps = [
            ("Setting up base model", self.setup_base_model),
            ("Creating configuration", self.create_active_learning_config),
            ("Preparing initial dataset", lambda: self.prepare_initial_dataset(labeled_data_path)),
            ("Creating workflow", self.create_active_learning_workflow),
            ("Setting up automation", self.setup_cron_jobs)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Failed at step: {step_name}")
                return False
            logger.info(f"Completed: {step_name}")
        
        logger.info("Transition to active learning completed successfully!")
        
        # Print next steps
        print("\n" + "="*60)
        print("ACTIVE LEARNING TRANSITION COMPLETED")
        print("="*60)
        print(f"Active learning directory: {self.active_learning_dir}")
        print("\nNext steps:")
        print("1. Review configuration in config.json")
        print("2. Add initial labeled data to data/initial_labeled/")
        print("3. Start Redis server for task queuing")
        print("4. Run the Backend API server")
        print("5. Execute: python3 active_learning_bridge.py --mode daemon")
        print("6. Set up cron jobs: bash crontab_setup.sh")
        print("\nThe system is now ready for active learning!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Transition Veritas-Lens to Active Learning")
    parser.add_argument("--labeled-data", help="Path to initial labeled data directory")
    parser.add_argument("--model-path", help="Path to existing model weights")
    
    args = parser.parse_args()
    
    transition = ActiveLearningTransition(args.model_path)
    success = transition.run_transition(args.labeled_data)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
