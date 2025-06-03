# Add this import at the top
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import os
from transformers import BertTokenizerFast, TFBertModel
from google.colab import drive

# Assuming Drive is already mounted from a previous cell
# drive.mount('/content/drive')

# 2. TPU and mixed precision setup (Keep as is)
def setup_tpu_and_mixed_precision():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("✅ TPU v6e-1 initialized and strategy set.")
        mixed_precision.set_global_policy('mixed_bfloat16')
        print(f"✅ Mixed precision policy set: {mixed_precision.global_policy()}")
        return strategy
    except Exception as e:
        print("❌ TPU initialization failed:", e)
        print("Falling back to GPU/CPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            mixed_precision.set_global_policy('mixed_float16')
            strategy = tf.distribute.MirroredStrategy()
            print("✅ Running on GPU with mixed_float16.")
        else:
            mixed_precision.set_global_policy('float32')
            strategy = tf.distribute.get_strategy()
            print("✅ Running on CPU with float32.")
        print(f"Mixed precision policy: {mixed_precision.global_policy()}")
        return strategy

# 3. Load Hugging Face tokenizer from Google Drive (Keep as is)
TOKENIZER_DIR = '/content/drive/MyDrive/Veritas-lens/saved_model/tokenizer_bert_combined_bs64_adamw_no_mp'
if not os.path.exists(TOKENIZER_DIR):
    print(f"❌ Tokenizer directory not found at {TOKENIZER_DIR}")
    tokenizer = None # Or load a default one
else:
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print("✅ Tokenizer loaded from directory!")

# 4. Path to your weights file in Google Drive (Keep as is)
WEIGHTS_PATH = '/content/drive/MyDrive/Veritas-lens/saved_model/best_model_bert_combined_bs64_adamw_no_mp.weights.h5'

# 5. Build your BERT-based model
def build_model():
    input_ids = keras.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = keras.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    token_type_ids = keras.Input(shape=(128,), dtype=tf.int32, name='token_type_ids')

    # Load the TFBertModel instance with default settings (should include pooler)
    bert_model = TFBertModel.from_pretrained("bert-base-uncased", return_dict=True)

    bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # Use the [CLS] token output
    cls_output = bert_outputs.last_hidden_state[:, 0, :]

    # Add your classification head
    x = keras.layers.Dense(128, activation='relu', name='dense_relu')(cls_output)
    output = keras.layers.Dense(1, activation='sigmoid', dtype='float32', name='dense_output')(x)

    model = keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

    return model

# Function to recursively print H5 file structure
def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}/")
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")

# 6. Main logic
def main():
    strategy = setup_tpu_and_mixed_precision()

    with strategy.scope():
        model = build_model()
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=2e-5, weight_decay=1e-2
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        print("\n--- Model Summary ---")
        model.summary()
        print("\n--- Model Layers and Variables ---")
        for layer in model.layers:
            print(f"Layer: {layer.name} ({layer.__class__.__name__})")
            for var in layer.variables:
                 print(f"  - {var.name}")
        print("----------------------")


        # Load weights from Drive
        if os.path.exists(WEIGHTS_PATH):
            print(f"\nAttempting to load weights from {WEIGHTS_PATH}")
            try:
                # Try inspecting the H5 file first
                print("\n--- H5 File Structure ---")
                with h5py.File(WEIGHTS_PATH, 'r') as f:
                    f.visititems(print_h5_structure)
                print("-------------------------")

                # Attempt to load weights directly
                # This is expected to fail if the structure doesn't match
                model.load_weights(WEIGHTS_PATH)
                print(f"✅ Weights loaded from {WEIGHTS_PATH}")
            except Exception as e:
                 print(f"❌ Failed to load weights from {WEIGHTS_PATH}: {e}")
                 print("Ensure the model architecture and layer names match the saved weights.")
                 print("Consider checking the exact library versions used when saving the weights.")
        else:
            print(f"❌ Weights file not found at {WEIGHTS_PATH}")

    # TODO: Add inference or further training here

if __name__ == "__main__":
    main()