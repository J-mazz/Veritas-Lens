#!/bin/bash

# Comprehensive Python Environment Setup for Veritas-Lens Active Learning
# This script sets up a complete ML environment for training and active learning

set -e

echo "🚀 Setting up Veritas-Lens Python Environment..."
echo "================================================================"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv_veritas
source venv_veritas/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core ML dependencies
echo "🧠 Installing TensorFlow and ML dependencies..."
pip install \
    tensorflow==2.15.0 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    transformers==4.35.0 \
    datasets==2.14.0 \
    accelerate==0.24.0

# Install data science libraries
echo "📊 Installing data science libraries..."
pip install \
    numpy==1.24.3 \
    pandas==2.1.0 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.17.0

# Install text processing and NLP
echo "📝 Installing NLP libraries..."
pip install \
    nltk==3.8.1 \
    spacy==3.6.1 \
    textblob==0.17.1 \
    wordcloud==1.9.2

# Install Redis and job queue libraries
echo "🔴 Installing Redis and queue libraries..."
pip install \
    redis==5.0.1 \
    celery==5.3.0 \
    kombu==5.3.0

# Install web scraping and API libraries
echo "🌐 Installing web scraping libraries..."
pip install \
    requests==2.31.0 \
    beautifulsoup4==4.12.2 \
    feedparser==6.0.10 \
    newspaper3k==0.2.8

# Install monitoring and logging
echo "📈 Installing monitoring libraries..."
pip install \
    tensorboard==2.15.0 \
    wandb==0.15.12 \
    mlflow==2.7.1

# Install Jupyter and development tools
echo "🔧 Installing development tools..."
pip install \
    jupyter==1.0.0 \
    jupyterlab==4.0.7 \
    ipykernel==6.25.2 \
    black==23.9.1 \
    flake8==6.1.0 \
    pytest==7.4.2

# Install additional utilities
echo "🛠️ Installing utility libraries..."
pip install \
    tqdm==4.66.1 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    jsonlines==3.1.0 \
    schedule==1.2.0

# Download spaCy model
echo "📚 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create requirements.txt
echo "💾 Creating requirements.txt..."
pip freeze > requirements_ml.txt

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "🔮 Activating Veritas-Lens ML Environment..."
source venv_veritas/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts
export TF_CPP_MIN_LOG_LEVEL=2
echo "✅ Environment activated!"
echo "📍 Python path: $(which python)"
echo "📦 TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "🤖 Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
EOF

chmod +x activate_env.sh

# Create training script runner
cat > run_training.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Veritas-Lens Comprehensive Training..."

# Activate environment
source activate_env.sh

# Set GPU memory growth (prevents OOM)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run training with monitoring
python scripts/train_comprehensive.py \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --output-dir ./training_results \
    --max-seq-length 512 \
    --early-stopping-patience 10 \
    2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

echo "✅ Training completed! Check training_results/ for outputs."
EOF

chmod +x run_training.sh

# Create active learning training script
cat > run_active_learning.sh << 'EOF'
#!/bin/bash
echo "🎯 Starting Veritas-Lens Active Learning Training..."

# Activate environment
source activate_env.sh

# Run active learning training
python Backend/scripts/active_learning_comprehensive.py \
    --initial-epochs 20 \
    --active-learning-cycles 5 \
    --samples-per-cycle 100 \
    --output-dir ./active_learning_results \
    2>&1 | tee active_learning_$(date +%Y%m%d_%H%M%S).log

echo "✅ Active learning completed! Check active_learning_results/ for outputs."
EOF

chmod +x run_active_learning.sh

# Create testing script
cat > test_environment.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Veritas-Lens Environment..."

source activate_env.sh

python << 'PYTHON'
import sys
print(f"Python version: {sys.version}")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} - GPU available: {tf.config.list_physical_devices('GPU')}")
except Exception as e:
    print(f"❌ TensorFlow error: {e}")

# Test Transformers
try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers error: {e}")

# Test datasets
try:
    import datasets
    print(f"✅ Datasets {datasets.__version__}")
except Exception as e:
    print(f"❌ Datasets error: {e}")

# Test Redis
try:
    import redis
    print(f"✅ Redis {redis.__version__}")
except Exception as e:
    print(f"❌ Redis error: {e}")

# Test data science libraries
try:
    import pandas as pd
    import numpy as np
    import sklearn
    print(f"✅ Data science stack ready - pandas {pd.__version__}, numpy {np.__version__}, sklearn {sklearn.__version__}")
except Exception as e:
    print(f"❌ Data science error: {e}")

print("🎉 Environment test completed!")
PYTHON
EOF

chmod +x test_environment.sh

echo ""
echo "🎉 Python Environment Setup Complete!"
echo "================================================================"
echo ""
echo "📁 Created files:"
echo "   - venv_veritas/           (Python virtual environment)"
echo "   - requirements_ml.txt     (All installed packages)"
echo "   - activate_env.sh        (Environment activation script)"
echo "   - run_training.sh        (Comprehensive training runner)"
echo "   - run_active_learning.sh (Active learning training runner)"
echo "   - test_environment.sh    (Environment testing script)"
echo ""
echo "🚀 Quick start:"
echo "   1. Test environment:     ./test_environment.sh"
echo "   2. Run training:         ./run_training.sh"
echo "   3. Run active learning:  ./run_active_learning.sh"
echo ""
echo "💡 To manually activate environment: source activate_env.sh"
echo ""
