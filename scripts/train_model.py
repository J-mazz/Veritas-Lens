import tensorflow as tf
from transformers import AutoTokenizer
import os
import sys
import time
import math
import glob # Import glob
import datetime # For logging timestamps
import traceback # Import traceback
# *** Import for classification report and confusion matrix ***
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# *** Add a version marker ***
print("--- RUNNING SCRIPT VERSION: train_model_final_combined.py (10 Epochs, AdamW, Combined Data) ---")

# --- Mixed Precision Setup ---
# Keep mixed precision disabled
use_mixed_precision = False
if use_mixed_precision:
    # This branch will not be hit
    print("Enabling Mixed Precision (mixed_float16)...")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Compute dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}")
    print(f"Variable dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}")
else:
    print("Mixed Precision DISABLED.")

# --- Configuration ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "political_bias_detector"
SCRIPTS_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "scripts")
SAVED_MODEL_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "saved_model")
# *** POINT TO THE COMBINED DATA DIRECTORY ***
PROCESSED_DATA_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed_combined")

# Add scripts directory to Python path
if SCRIPTS_DIR not in sys.path:
    print("Appending SCRIPTS_DIR to sys.path...")
    sys.path.append(SCRIPTS_DIR)
else:
    print("SCRIPTS_DIR already in sys.path.")


# Import model creation function from the updated model_definition.py
try:
    # *** Ensure using model_definition.py that loads TFBertForSequenceClassification ***
    print("Attempting import from model_definition...")
    from model_definition import create_and_compile_model, load_tokenizer, PRE_TRAINED_MODEL_NAME
    print("Import successful.")
    # Verify the imported model name matches expectation
    if PRE_TRAINED_MODEL_NAME != 'bert-base-uncased':
        print(f"Warning: Imported PRE_TRAINED_MODEL_NAME is '{PRE_TRAINED_MODEL_NAME}', expected 'bert-base-uncased'.")

except ImportError as e: # Catch specific error
    print(f"\nCaught ImportError: {e}")
    print(f"Error: Could not import from model_definition.py.")
    print(f"Ensure model_definition.py (with create_and_compile_model function) is in the '{SCRIPTS_DIR}' directory and contains the required definitions.")
    exit()
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     traceback.print_exc() # Print traceback for unexpected errors
     exit()

# --- Dataset/Model Config ---
CLASS_NAMES = ['left', 'right', 'center']
NUM_CLASSES = len(CLASS_NAMES)

# Training Hyperparameters
BATCH_SIZE = 64 # Keep increased batch size for A100
MAX_SEQUENCE_LENGTH = 256
# *** INCREASED EPOCHS ***
EPOCHS = 10 # Increased epochs for potentially better convergence
LEARNING_RATE = 2e-5 # Standard LR for full BERT fine-tuning
# *** Add weight decay for AdamW ***
WEIGHT_DECAY_RATE = 0.01 # Common value for AdamW

# --- Google Drive Mounting (Commented Out) ---
# ...

# --- Verify Project Folders ---
print("Verifying project structure...")
project_path = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER)
if not os.path.isdir(project_path):
    print(f"Error: Project folder '{PROJECT_FOLDER}' not found in '{DRIVE_BASE_PATH}'.")
    exit()
if not os.path.isdir(PROCESSED_DATA_DIR):
     print(f"Error: Processed data directory '{PROCESSED_DATA_DIR}' not found.")
     print("Please ensure the preprocessing scripts (including combine) were run successfully.")
     exit()
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
print("Project structure verified.")


# --- Load Tokenizer ---
print("Loading tokenizer...")
try:
    # *** Load the tokenizer saved from the previous successful training run ***
    # Make sure the corresponding tokenizer exists
    tokenizer_load_path = os.path.join(SAVED_MODEL_DIR, f'tokenizer_bert_combined_bs{BATCH_SIZE}_adamw_no_mp')
    if not os.path.isdir(tokenizer_load_path):
         print(f"Warning: Tokenizer from previous run not found at {tokenizer_load_path}. Loading default.")
         tokenizer = load_tokenizer() # Fallback to loading default
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
        print(f"Tokenizer loaded successfully from {tokenizer_load_path}")
except Exception as e:
    print(f"Failed to load tokenizer. Exiting. Error: {e}")
    exit()

# --- Data Loading and Preprocessing Functions for Directory Loading ---
def encode_directory_examples(text_batch, label_batch):
    """Tokenizes a batch of text loaded from directory using tf.py_function."""
    def _encode_py(text_tensor, label_tensor):
        text_list = [t.decode('utf-8') for t in text_tensor.numpy()]
        tokenized_inputs = tokenizer(
            text_list,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        input_ids = tokenized_inputs['input_ids'].astype('int32')
        attention_mask = tokenized_inputs['attention_mask'].astype('int32')
        labels = label_tensor.numpy().astype('int32')
        return input_ids, attention_mask, labels

    input_ids, attention_mask, labels = tf.py_function(
        func=_encode_py,
        inp=[text_batch, label_batch],
        Tout=[tf.int32, tf.int32, tf.int32]
    )
    input_ids.set_shape([None, MAX_SEQUENCE_LENGTH])
    attention_mask.set_shape([None, MAX_SEQUENCE_LENGTH])
    labels.set_shape([None])
    features = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return features, labels


# --- Load and Prepare Datasets from Directory ---
print(f"Loading datasets from directory: {PROCESSED_DATA_DIR}")
try:
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
    val_dir = os.path.join(PROCESSED_DATA_DIR, "valid")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test") # Keep test_dir for potential evaluation later

    # *** Use tf.data.experimental.cardinality for faster dataset size calculation ***
    print("Loading raw text datasets to determine size...")
    # Load with the NEW batch size to get correct cardinality
    temp_raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, label_mode='int', class_names=CLASS_NAMES, batch_size=BATCH_SIZE, shuffle=False)
    temp_raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        val_dir, label_mode='int', class_names=CLASS_NAMES, batch_size=BATCH_SIZE, shuffle=False)

    steps_per_epoch = tf.data.experimental.cardinality(temp_raw_train_ds).numpy()
    validation_steps = tf.data.experimental.cardinality(temp_raw_val_ds).numpy()

    # Clean up temporary datasets to free memory
    del temp_raw_train_ds
    del temp_raw_val_ds

    if steps_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY or steps_per_epoch <= 0:
        raise ValueError("Could not determine the number of batches in the training dataset. Check data loading.")
    if validation_steps == tf.data.experimental.UNKNOWN_CARDINALITY or validation_steps <= 0:
         raise ValueError("Could not determine the number of batches in the validation dataset. Check data loading.")

    print(f"Using BATCH_SIZE: {BATCH_SIZE}") # Log updated batch size
    print(f"Determined steps_per_epoch: {steps_per_epoch}")
    print(f"Determined validation_steps: {validation_steps}")

    # Now load the actual datasets for training (with shuffling if desired)
    print("Loading actual datasets for training...")
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, label_mode='int', class_names=CLASS_NAMES, batch_size=BATCH_SIZE, seed=42) # Re-add seed for shuffling
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        val_dir, label_mode='int', class_names=CLASS_NAMES, batch_size=BATCH_SIZE, seed=42)
    print("Raw datasets loaded.")

    print(f"Loaded classes from train_ds: {raw_train_ds.class_names}")
    if raw_train_ds.class_names != CLASS_NAMES:
         print("Warning: Class names loaded from directory do not match expected CLASS_NAMES order.")

    print("Encoding datasets...")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.map(encode_directory_examples, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.map(encode_directory_examples, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
    print("TensorFlow datasets prepared and encoded.")

except Exception as e:
    print(f"Error loading or processing dataset from directory: {e}")
    # *** Use traceback now that it's imported ***
    traceback.print_exc()
    print(f"Ensure the directory structure under {PROCESSED_DATA_DIR} is correct (e.g., train/left/*.txt)")
    exit()


# --- Setup Model, Optimizer, Loss, Metrics ---
print("Setting up model, optimizer, loss, and metrics...")
# Create optimizer
# *** Use AdamW Optimizer ***
if use_mixed_precision:
    # This block won't run now
    print(f"Creating LossScaleOptimizer wrapping AdamW with LR={LEARNING_RATE}, WD={WEIGHT_DECAY_RATE}...")
    base_optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY_RATE)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
    print("Mixed precision enabled. Optimizer wrapped.")
else:
    print(f"Creating standard AdamW optimizer with LR={LEARNING_RATE}, WD={WEIGHT_DECAY_RATE}...")
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY_RATE)
    except AttributeError:
        print("tf.keras.optimizers.AdamW not found, trying experimental path...")
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY_RATE)
    print("Mixed precision disabled. Using standard AdamW.")

# Create model instance using the standard HF model function
try:
    # *** Call create_and_compile_model from model_definition ***
    # Pass the AdamW optimizer
    model = create_and_compile_model(num_labels=NUM_CLASSES, optimizer=optimizer)

    # Build the model to ensure weights are created before checking
    print("Building model...")
    sample_features, _ = next(iter(train_ds))
    _ = model(sample_features, training=False) # Build step
    print("Model built successfully.")

    # Check trainable weights AFTER compile
    print(f"Direct check - Number of trainable weights after compile: {len(model.trainable_weights)}")
    if len(model.trainable_weights) < 100: # Check if fine-tuning seems active
        print("ERROR: Model reports very few trainable weights AFTER compile. Check model_definition.py.")
        model.summary() # Print summary again for inspection
        exit()
    else:
        print("Trainable weights check passed (using standard HF model).")
        model.summary() # Print summary to confirm

except Exception as e:
    print(f"Failed to create, build, or compile model: {e}")
    # *** Use traceback now that it's imported ***
    traceback.print_exc()
    exit()

# Define Loss function (must match model output type: logits)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
print("Setup complete.")

# --- Custom Training Step ---
@tf.function # Compile function for better performance
def train_step(features, labels):
    with tf.GradientTape() as tape:
        # Get model predictions (logits)
        predictions = model(features, training=True) # Pass features dict
        # Calculate loss using ground truth labels and predictions
        loss = loss_fn(labels, predictions.logits) # Access logits from output object
        scaled_loss = loss # Use original loss as mixed precision is disabled

    # Calculate gradients
    gradients = tape.gradient(scaled_loss, model.trainable_variables) # Use original loss

    # Check if gradients list itself is None or empty
    if gradients is None or not gradients:
         tf.print("Error: No gradients computed. Check model trainability and loss connection.")
         return # Skip applying gradients if none were computed

    # Check for None gradients within the list (more thorough)
    filtered_grads_and_vars = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
    if len(filtered_grads_and_vars) < len(model.trainable_variables):
        tf.print("Warning: Some gradients were None. Applying gradients only for non-None pairs.")
        if not filtered_grads_and_vars:
             tf.print("Error: All gradients were None. No weights updated.")
             return # Skip if all gradients are None

    # Apply gradients to update model weights
    optimizer.apply_gradients(filtered_grads_and_vars) # Apply potentially filtered list

    # Update training metrics
    train_loss(loss)
    train_accuracy(labels, predictions.logits) # Use logits for accuracy metric

# --- Custom Validation Step ---
@tf.function # Compile function for better performance
def validation_step(features, labels):
    # Get model predictions (logits) - training=False disables dropout etc.
    predictions = model(features, training=False)
    # Calculate loss
    loss = loss_fn(labels, predictions.logits)

    # Update validation metrics
    val_loss(loss)
    val_accuracy(labels, predictions.logits)

# --- Training Loop ---
print(f"\nStarting custom training loop for {EPOCHS} epochs...")
best_val_accuracy = -1.0
# *** Adjusted checkpoint filename for combined data + AdamW ***
checkpoint_filepath = os.path.join(SAVED_MODEL_DIR, f'best_model_bert_combined_bs{BATCH_SIZE}_adamw_no_mp.weights.h5') # Added combined

# *** Add EarlyStopping Callback parameters ***
early_stopping_patience = 3
epochs_no_improve = 0

for epoch in range(EPOCHS):
    start_time_epoch = time.time()
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Reset metrics at the start of each epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    val_loss.reset_state()
    val_accuracy.reset_state()

    # --- Training Phase ---
    print("Training...")
    # *** Record start time for ETA calculation ***
    epoch_train_start_time = time.time()
    for step, (features, labels) in enumerate(train_ds):
        # Add try-except block around train_step for debugging potential data issues
        try:
            train_step(features, labels)
        except tf.errors.ResourceExhaustedError as e: # Catch OOM errors specifically
            print(f"\n--- GPU OUT OF MEMORY during training step {step}! ---")
            print(f"Reduce BATCH_SIZE from {BATCH_SIZE} and try again.")
            print(f"Error details: {e}")
            exit()
        except Exception as e:
            print(f"\n!!! Error during training step {step} !!!")
            print(f"Error: {e}")
            print("Features:", features)
            print("Labels:", labels)
            # Optionally: exit() or raise e
            traceback.print_exc() # Use traceback
            raise e # Re-raise the exception to stop execution

        # Log progress and ETA every N steps
        current_step = step + 1 # Use 1-based indexing for calculations
        if current_step % 100 == 0: # Adjust log frequency if needed
            elapsed_time = time.time() - epoch_train_start_time
            time_per_step = elapsed_time / current_step if current_step > 0 else 0
            remaining_steps = steps_per_epoch - current_step
            eta_seconds = remaining_steps * time_per_step if time_per_step > 0 else 0

            # Format ETA
            eta_str = "Calculating..."
            if eta_seconds > 0:
                eta_minutes, eta_secs = divmod(int(eta_seconds), 60)
                eta_hours, eta_minutes = divmod(eta_minutes, 60) # Add hours for longer epochs
                if eta_hours > 0:
                     eta_str = f"{eta_hours:d}h {eta_minutes:02d}m {eta_secs:02d}s"
                else:
                     eta_str = f"{eta_minutes:02d}m {eta_secs:02d}s"


            print(f"  Step {current_step}/{steps_per_epoch} - "
                  f"Loss: {train_loss.result():.4f}, "
                  f"Accuracy: {train_accuracy.result():.4f} - "
                  f"ETA: {eta_str}") # Added ETA

    print(f"Epoch {epoch + 1} Training complete.")
    print(f"  Training Loss: {train_loss.result():.4f}, "
          f"Training Accuracy: {train_accuracy.result():.4f}")

    # --- Validation Phase ---
    print("Validating...")
    for step, (features, labels) in enumerate(val_ds):
         # Add try-except block around validation_step for debugging potential data issues
        try:
            validation_step(features, labels)
        except tf.errors.ResourceExhaustedError as e: # Catch OOM errors specifically
            print(f"\n--- GPU OUT OF MEMORY during validation step {step}! ---")
            print(f"Reduce BATCH_SIZE from {BATCH_SIZE} and try again.")
            print(f"Error details: {e}")
            exit()
        except Exception as e:
            print(f"\n!!! Error during validation step {step} !!!")
            print(f"Error: {e}")
            print("Features:", features)
            print("Labels:", labels)
            # Optionally: exit() or raise e
            traceback.print_exc() # Use traceback
            raise e # Re-raise the exception to stop execution

        if (step + 1) % 50 == 0: # Log every 50 steps
             print(f"  Validation Step {step + 1}/{validation_steps}")

    current_val_accuracy = val_accuracy.result()
    print(f"Epoch {epoch + 1} Validation complete.")
    print(f"  Validation Loss: {val_loss.result():.4f}, "
          f"Validation Accuracy: {current_val_accuracy:.4f}")

    # --- Checkpoint Saving & Early Stopping ---
    if current_val_accuracy > best_val_accuracy:
        print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {current_val_accuracy:.4f}. Saving model weights to {checkpoint_filepath}")
        best_val_accuracy = current_val_accuracy
        # Save only the weights
        model.save_weights(checkpoint_filepath)
        epochs_no_improve = 0 # Reset counter
    else:
        epochs_no_improve += 1
        print(f"Validation accuracy did not improve from {best_val_accuracy:.4f}. ({epochs_no_improve}/{early_stopping_patience})")

    epoch_time = time.time() - start_time_epoch
    print(f"Epoch {epoch + 1} finished in {epoch_time:.2f} seconds.")

    # Early stopping check
    if epochs_no_improve >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epochs_no_improve} epochs with no improvement.")
        break # Exit the epoch loop


print("\nCustom training loop finished.")
print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
print(f"Best model weights saved to: {checkpoint_filepath}")

# --- Final Evaluation (Optional) ---
# print("\nLoading best weights for final evaluation...")
# try:
#     # Load test dataset
#     print("Loading test dataset...")
#     raw_test_ds = tf.keras.utils.text_dataset_from_directory(
#         test_dir, label_mode='int', class_names=CLASS_NAMES, batch_size=BATCH_SIZE, shuffle=False)
#     test_ds = raw_test_ds.map(encode_directory_examples, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
#     test_steps = tf.data.experimental.cardinality(test_ds).numpy()
#     print(f"Evaluating on test set with {test_steps} steps...")
#
#     if os.path.exists(checkpoint_filepath):
#         # Re-create the model architecture before loading weights
#         print("Re-creating model structure for evaluation...")
#         eval_optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY_RATE) # Need an optimizer instance
#         eval_model = create_and_compile_model(num_labels=NUM_CLASSES, optimizer=eval_optimizer)
#         print("Building evaluation model...")
#         sample_features_eval, _ = next(iter(test_ds)) # Need a sample from test_ds
#         _ = eval_model(sample_features_eval, training=False) # Build step
#         print("Loading weights into evaluation model...")
#         eval_model.load_weights(checkpoint_filepath)
#         print("Best weights loaded into evaluation model.")
#
#         test_loss = tf.keras.metrics.Mean(name='test_loss')
#         test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#         all_predictions_eval = []
#         all_labels_eval = []
#
#         print("Running evaluation loop...")
#         for features, labels in test_ds:
#             predictions = eval_model(features, training=False) # Use eval_model
#             loss = loss_fn(labels, predictions.logits) # Use logits
#             test_loss(loss)
#             test_accuracy(labels, predictions.logits) # Use logits
#             all_predictions_eval.extend(tf.argmax(predictions.logits, axis=-1).numpy())
#             all_labels_eval.extend(labels.numpy())
#
#
#         print("-" * 40)
#         print("Final Test Set Evaluation (Best Model):")
#         print(f"Test Loss: {test_loss.result():.4f}")
#         print(f"Test Accuracy: {test_accuracy.result():.4f}")
#         print("-" * 40)
#
#         # Optional detailed metrics
#         print("\nCalculating additional metrics for test set...")
#         try:
#             all_labels_eval_np = np.array(all_labels_eval)
#             all_predictions_eval_np = np.array(all_predictions_eval)
#             print("\nClassification Report (Test Set):")
#             print(classification_report(all_labels_eval_np, all_predictions_eval_np, target_names=CLASS_NAMES, digits=4))
#             print("\nConfusion Matrix (Test Set):")
#             cm_test = confusion_matrix(all_labels_eval_np, all_predictions_eval_np)
#             print(cm_test)
#         except Exception as e_metrics:
#              print(f"Could not calculate detailed test metrics: {e_metrics}")
#
#     else:
#         print("Checkpoint file not found. Cannot evaluate.")
# except Exception as e_eval:
#     print(f"Error during final evaluation: {e_eval}")
#     traceback.print_exc()


# --- Save Tokenizer ---
# *** Adjusted tokenizer save path for combined data + AdamW ***
tokenizer_save_path = os.path.join(SAVED_MODEL_DIR, f'tokenizer_bert_combined_bs{BATCH_SIZE}_adamw_no_mp') # Added combined
try:
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer saved to {tokenizer_save_path}")
except Exception as e:
    print(f"Error saving tokenizer: {e}")

print("\nCustom training script complete.")

