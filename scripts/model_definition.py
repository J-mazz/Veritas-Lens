import tensorflow as tf
# *** Import TFBertForSequenceClassification ***
from transformers import TFBertForSequenceClassification, AutoTokenizer

# --- Configuration ---
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# --- Model Creation & Compilation Function ---
def create_and_compile_model(num_labels, optimizer):
    """
    Loads a pre-trained TFBertForSequenceClassification model, sets it to
    be fully trainable, and compiles it.
    """
    print(f"Loading TFBertForSequenceClassification model: {PRE_TRAINED_MODEL_NAME}")
    print("Full fine-tuning enabled.")

    try:
        # *** Load the sequence classification model directly ***
        # Pass num_labels to initialize the classification head correctly
        model = TFBertForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=num_labels
        )
        # *** Ensure the entire model is trainable ***
        # This is generally the default for from_pretrained sequence classification models,
        # but setting it explicitly ensures our intent.
        model.trainable = True
        print(f"Model '{PRE_TRAINED_MODEL_NAME}' loaded and set to trainable.")

    except Exception as e:
        print(f"Error loading pre-trained sequence classification model '{PRE_TRAINED_MODEL_NAME}': {e}")
        raise

    # *** Compile the loaded model ***
    # TFBertForSequenceClassification outputs LOGITS by default
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print("Using SparseCategoricalCrossentropy(from_logits=True)") # Important!

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("TFBertForSequenceClassification model compiled successfully.")
    return model

# --- Tokenizer Loading --- (No changes needed)
def load_tokenizer():
    """Loads the tokenizer corresponding to the pre-trained model."""
    print(f"Loading tokenizer for: {PRE_TRAINED_MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        print("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer '{PRE_TRAINED_MODEL_NAME}': {e}")
        raise

# Example usage block
if __name__ == '__main__':
    print("Running model definition script example (TFBertForSequenceClassification)...")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    num_classes = 3
    example_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == 'mixed_float16':
         example_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(example_optimizer)

    # Create model using the updated function
    example_model = create_and_compile_model(num_labels=num_classes, optimizer=example_optimizer)
    example_tokenizer = load_tokenizer()
    print("Example model and tokenizer created.")

    # Build the model by calling it (or use model.summary())
    dummy_input = {
        'input_ids': tf.constant([[101, 102]]),
        'attention_mask': tf.constant([[1, 1]])
    }
    # The model call might need specific input shapes depending on TF/Keras version
    # Or just rely on model.summary() which should build it
    # _ = example_model(dummy_input)

    print("Model Summary (Should show ~110M trainable params):")
    example_model.summary() # Summary should now correctly show trainable params
    print(f"\nDirect check - Number of trainable weights: {len(example_model.trainable_weights)}")
    tf.keras.mixed_precision.set_global_policy('float32') # Reset policy
