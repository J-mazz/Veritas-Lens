# Veritas-Lens A100 Training Setup

This directory contains the TensorFlow/Keras implementation for training Veritas-Lens on A100 GPUs.

## Quick Start for Google Colab A100

1. **Clone the repository:**
```bash
!git clone https://github.com/J-mazz/Veritas-Lens.git
%cd Veritas-Lens
```

2. **Install dependencies:**
```bash
!pip install -r requirements_a100.txt
```

3. **Run training:**
```bash
!python train_a100_keras.py
```

## Features

- **Pure TensorFlow/Keras**: Avoids library inconsistencies
- **A100 Optimized**: Mixed precision, memory management
- **Political Bias Detection**: Left/Center/Right classification
- **Robust Pipeline**: Data loading, preprocessing, training, evaluation

## Files

- `train_a100_keras.py`: Main training script optimized for A100
- `requirements_a100.txt`: TensorFlow-focused dependencies
- `README_A100.md`: This file

## Training Configuration

- **Model**: Text classification with embedding layers
- **Mixed Precision**: Enabled for A100 performance
- **Batch Size**: 64 (optimized for A100 memory)
- **Learning Rate**: 2e-5 with scheduling
- **Callbacks**: Early stopping, checkpointing, TensorBoard

## Results

After training, check the `./results/` directory for:
- Trained model (`veritas_lens_model.h5`)
- Training history (`training_history.csv`)
- Classification report (`results.json`)
- Label encoder (`label_encoder.pkl`)

## Memory and Performance

The script includes:
- GPU memory growth to prevent OOM
- Mixed precision for faster training
- Efficient data loading and preprocessing
- TensorBoard logging for monitoring

Ready for A100 training in Google Colab!
