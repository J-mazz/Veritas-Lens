import tensorflow as tf
from transformers import AutoTokenizer
import os
import sys
import numpy as np # For creating dummy data

# --- Configuration ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "political_bias_detector"
SCRIPTS_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "scripts")
SAVED_MODEL_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "saved_model")
TFLITE_OUTPUT_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "tflite_model") # Directory to save the .tflite file

# Model/Tokenizer specific details (should match the trained model)
BATCH_SIZE = 64 # Batch size used during training (needed to find correct saved files)
MAX_SEQUENCE_LENGTH = 256
NUM_CLASSES = 3 # Number of output classes (left, right, center)

# Filenames for the saved weights and tokenizer (adjust if different)
TOKENIZER_DIR_NAME = f'tokenizer_bert_combined_bs{BATCH_SIZE}_adamw_no_mp'
WEIGHTS_FILENAME = f'best_model_bert_combined_bs{BATCH_SIZE}_adamw_no_mp.weights.h5'

# Output TFLite filename
TFLITE_FILENAME = f'bert_classifier_bs{BATCH_SIZE}_seq{MAX_SEQUENCE_LENGTH}.tflite'

# Add scripts directory to Python path to import model definition
if SCRIPTS_DIR not in sys.path:
    print(f"Appending {SCRIPTS_DIR} to sys.path")
    sys.path.append(SCRIPTS_DIR)
else:
     print(f"{SCRIPTS_DIR} already in sys.path")

# Import model creation function
try:
    # Ensure using the model_definition.py that loads TFBertForSequenceClassification
    print("Attempting import from model_definition...")
    from model_definition import create_and_compile_model, load_tokenizer, PRE_TRAINED_MODEL_NAME
    print("Import successful.")
    if PRE_TRAINED_MODEL_NAME != 'bert-base-uncased':
        print(f"Warning: Imported PRE_TRAINED_MODEL_NAME is '{PRE_TRAINED_MODEL_NAME}'. Ensure this matches the trained model.")
except ImportError as e:
    print(f"\nCaught ImportError: {e}")
    print(f"Error: Could not import from model_definition.py in {SCRIPTS_DIR}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- Main Conversion Logic ---

# 1. Load Tokenizer
print("\n--- Loading Tokenizer ---")
tokenizer_load_path = os.path.join(SAVED_MODEL_DIR, TOKENIZER_DIR_NAME)
if not os.path.isdir(tokenizer_load_path):
    print(f"Error: Saved tokenizer not found at {tokenizer_load_path}")
    exit()
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
    print(f"Tokenizer loaded successfully from {tokenizer_load_path}")
except Exception as e:
    print(f"Failed to load saved tokenizer: {e}")
    exit()

# 2. Load Model Architecture and Weights
print("\n--- Loading Model Architecture and Weights ---")
# Create a dummy optimizer (needed for compile step in create_and_compile_model)
# The optimizer state itself isn't used for conversion/inference
try:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=0.01) # Use AdamW as it was used in training
except AttributeError:
     optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=2e-5, weight_decay=0.01)

# Create the model structure
try:
    model = create_and_compile_model(num_labels=NUM_CLASSES, optimizer=optimizer)
    print("Model architecture created successfully.")
except Exception as e:
    print(f"Failed to create model architecture: {e}")
    exit()

# Load the trained weights
checkpoint_filepath = os.path.join(SAVED_MODEL_DIR, WEIGHTS_FILENAME)
if not os.path.exists(checkpoint_filepath):
    print(f"Error: Weights file not found at {checkpoint_filepath}")
    exit()
try:
    # Build the model first by calling it with dummy data matching expected input spec
    print("Building model with dummy input...")
    dummy_input_ids = tf.zeros((1, MAX_SEQUENCE_LENGTH), dtype=tf.int32)
    dummy_attention_mask = tf.zeros((1, MAX_SEQUENCE_LENGTH), dtype=tf.int32)
    dummy_inputs = {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask}
    _ = model(dummy_inputs, training=False)
    print("Model built.")
    # Now load weights
    model.load_weights(checkpoint_filepath)
    print(f"Successfully loaded weights from {checkpoint_filepath}")
except Exception as e:
    print(f"Failed to load weights: {e}")
    exit()

# 3. Create Concrete Function for TFLite Conversion
# TFLite converter works best with concrete functions that define specific input signatures.
print("\n--- Creating Concrete Function for Conversion ---")
# Define the inference function
@tf.function(input_signature=[{
    # Define the expected input dictionary structure and shapes/types
    'input_ids': tf.TensorSpec(shape=[1, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='input_ids'),
    'attention_mask': tf.TensorSpec(shape=[1, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='attention_mask')
}])
def serving_function(input_dict):
    """Runs inference with the model."""
    return model(input_dict, training=False)

# Get the concrete function
try:
    concrete_func = serving_function.get_concrete_function()
    print("Concrete function created successfully.")
except Exception as e:
    print(f"Failed to create concrete function: {e}")
    exit()

# 4. Convert to TFLite
print("\n--- Converting Model to TensorFlow Lite ---")
try:
    # Initialize the converter using the concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model) # Pass original model too

    # --- Configure the Converter ---
    # Apply standard optimizations (includes quantization like dynamic range)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # *** Crucial: Enable TensorFlow ops support ***
    # Many transformer ops are not standard TFLite built-ins
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable default TFLite ops.
        tf.lite.OpsSet.SELECT_TF_OPS    # Enable necessary TensorFlow ops.
    ]

    # Optional: Lower float precision further if needed (can impact accuracy)
    # converter._experimental_lower_tensor_list_ops = False # Keep default for compatibility

    print("Converter configured with DEFAULT optimizations and SELECT_TF_OPS.")

    # Perform the conversion
    print("Starting conversion process...")
    tflite_model = converter.convert()
    print("Conversion successful!")

except Exception as e:
    print(f"Error during TFLite conversion: {e}")
    traceback.print_exc()
    exit()

# 5. Save the TFLite Model
print("\n--- Saving TFLite Model ---")
os.makedirs(TFLITE_OUTPUT_DIR, exist_ok=True)
tflite_model_path = os.path.join(TFLITE_OUTPUT_DIR, TFLITE_FILENAME)
try:
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved successfully to: {tflite_model_path}")
    # Print size comparison
    keras_weights_size = os.path.getsize(checkpoint_filepath) / (1024 * 1024) # MB
    tflite_size = len(tflite_model) / (1024 * 1024) # MB
    print(f"  Original Keras weights size: {keras_weights_size:.2f} MB")
    print(f"  Converted TFLite model size: {tflite_size:.2f} MB")
except Exception as e:
    print(f"Error saving TFLite model: {e}")
    exit()

print("\n--- TFLite Conversion Script Finished ---")

