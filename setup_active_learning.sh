#!/bin/bash

# Setup Active Learning Environment for Veritas-Lens
# This script sets up the Python environment and installs all dependencies

set -e

echo "🚀 Setting up Active Learning Environment for Veritas-Lens..."

# Check if we're in the right directory
if [[ ! -f "scripts/train_active_learning.py" ]]; then
    echo "❌ Error: Please run this script from the Veritas-Lens root directory"
    exit 1
fi

# Create Python virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv_active_learning

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_active_learning/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core ML dependencies
echo "🧠 Installing TensorFlow and core ML libraries..."
pip install tensorflow==2.15.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.36.0
pip install datasets==2.14.0

# Install data processing libraries
echo "📊 Installing data processing libraries..."
pip install pandas==2.1.0
pip install numpy==1.24.0
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.0
pip install seaborn==0.12.0

# Install Redis and job queue libraries
echo "🔴 Installing Redis and background job libraries..."
pip install redis==5.0.0
pip install celery==5.3.0

# Install web scraping and data collection
echo "🌐 Installing web scraping libraries..."
pip install requests==2.31.0
pip install beautifulsoup4==4.12.0
pip install feedparser==6.0.0

# Install additional utilities
echo "🛠️ Installing utility libraries..."
pip install python-dotenv==1.0.0
pip install tqdm==4.66.0
pip install click==8.1.0

# Create requirements file
echo "📝 Creating requirements.txt for active learning..."
cat > requirements_active_learning.txt << EOF
# Active Learning Requirements for Veritas-Lens
tensorflow==2.15.0
torch
torchvision
torchaudio
transformers==4.36.0
datasets==2.14.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
redis==5.0.0
celery==5.3.0
requests==2.31.0
beautifulsoup4==4.12.0
feedparser==6.0.0
python-dotenv==1.0.0
tqdm==4.66.0
click==8.1.0
EOF

# Test imports
echo "🧪 Testing Python imports..."
python3 -c "
import tensorflow as tf
import transformers
import pandas as pd
import sklearn
import redis
print('✅ All core libraries imported successfully!')
print(f'TensorFlow version: {tf.__version__}')
print(f'Transformers version: {transformers.__version__}')
"

# Create environment activation script
echo "📋 Creating environment activation script..."
cat > activate_active_learning.sh << 'EOF'
#!/bin/bash
# Activate Active Learning Environment
echo "🔄 Activating Veritas-Lens Active Learning Environment..."
source venv_active_learning/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"
echo "✅ Environment activated! Python path includes scripts directory."
echo "To deactivate, run: deactivate"
EOF

chmod +x activate_active_learning.sh

# Create training launcher script
echo "🚀 Creating training launcher script..."
cat > train_model_active.sh << 'EOF'
#!/bin/bash
# Launch Active Learning Training

# Activate environment
source venv_active_learning/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 if available
export TF_CPP_MIN_LOG_LEVEL=1  # Reduce TensorFlow logging

echo "🚀 Starting Active Learning Training..."

# Run training script
python3 scripts/train_active_learning.py \
    --epochs 5 \
    --batch-size 16 \
    --redis-host localhost \
    --redis-port 6379 \
    "$@"

echo "✅ Training completed!"
EOF

chmod +x train_model_active.sh

# Create data preparation script
echo "📊 Creating data preparation script..."
cat > prepare_active_data.sh << 'EOF'
#!/bin/bash
# Prepare data for active learning

source venv_active_learning/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

echo "📊 Preparing data for active learning..."

# Create active learning data directory
mkdir -p data/active_learning

# Check if we have existing processed data
if [[ -d "data/processed_combined" ]]; then
    echo "✅ Found existing processed data, using for active learning initialization"
else
    echo "⚠️ No existing processed data found. Running data preprocessing..."
    
    # Run preprocessing if needed
    if [[ -f "scripts/preprocess_combine_datasets.py" ]]; then
        python3 scripts/preprocess_combine_datasets.py
    else
        echo "❌ Preprocessing script not found. Please run data preprocessing first."
        exit 1
    fi
fi

echo "✅ Data preparation completed!"
EOF

chmod +x prepare_active_data.sh

# Create model evaluation script
echo "📈 Creating model evaluation script..."
cat > evaluate_active_model.sh << 'EOF'
#!/bin/bash
# Evaluate active learning model

source venv_active_learning/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

echo "📈 Evaluating active learning model..."

python3 -c "
import sys
sys.path.append('scripts')
from train_active_learning import ActiveLearningTrainer
import json

# Initialize trainer
config = {
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0
}

trainer = ActiveLearningTrainer(config)

# Load model
if trainer.load_existing_model():
    print('✅ Model loaded successfully')
    
    # Test prediction
    test_texts = [
        'Conservative economic policies promote business growth',
        'Liberal social programs help reduce inequality',
        'Independent analysis shows mixed results'
    ]
    
    results = trainer.predict_batch(test_texts)
    
    print('🧪 Test Predictions:')
    for text, result in zip(test_texts, results):
        print(f'Text: {text[:50]}...')
        print(f'Prediction: {result[\"predicted_label\"]} (confidence: {result[\"confidence\"]:.3f})')
        print(f'Bias Score: {result[\"bias_score\"]:.3f}')
        print('---')
else:
    print('❌ Failed to load model')
"

echo "✅ Evaluation completed!"
EOF

chmod +x evaluate_active_model.sh

echo ""
echo "🎉 Active Learning Environment Setup Complete!"
echo ""
echo "📋 Available Scripts:"
echo "  ./activate_active_learning.sh    - Activate the Python environment"
echo "  ./prepare_active_data.sh         - Prepare data for active learning"
echo "  ./train_model_active.sh          - Train the active learning model"
echo "  ./evaluate_active_model.sh       - Evaluate the trained model"
echo ""
echo "🚀 Quick Start:"
echo "  1. ./prepare_active_data.sh"
echo "  2. ./train_model_active.sh"
echo "  3. ./evaluate_active_model.sh"
echo ""
echo "💡 To manually activate the environment:"
echo "  source ./activate_active_learning.sh"
echo ""
