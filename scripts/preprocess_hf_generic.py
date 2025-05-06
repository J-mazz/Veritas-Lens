import pandas as pd
import os
import re
from datasets import load_dataset, ClassLabel # Import ClassLabel
import argparse # To accept arguments from command line (optional)
import traceback # Import traceback

# --- Configuration ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "political_bias_detector"
# Output directory for the standardized CSV files
PROCESSED_CSV_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed_csv")

# --- Dataset Specific Settings ---
# You will change these settings for each dataset you run this script for.

# === Settings for: siddharthmb/article-bias-prediction-media-splits ===
HF_DATASET_NAME = "siddharthmb/article-bias-prediction-media-splits"
SOURCE_TEXT_COLUMN = 'content' # Column containing the main text
SOURCE_LABEL_COLUMN = 'bias_text' # Column containing the label (might be int or string)
OUTPUT_CSV_BASENAME = "siddharthmb_article-bias-prediction-media-splits" # Name for the output CSV

# === Settings for: Faith1712/Allsides_political_bias_proper ===
#HF_DATASET_NAME = "Faith1712/Allsides_political_bias_proper"
# *** CORRECTED COLUMN NAMES ***
#SOURCE_TEXT_COLUMN = 'text' # Actual text column name
#SOURCE_LABEL_COLUMN = 'label' # Actual label column name (likely ClassLabel int)
#OUTPUT_CSV_BASENAME = "Faith1712_Allsides_political_bias_proper"

# === Settings for: cajcodes/political-bias ===
# HF_DATASET_NAME = "cajcodes/political-bias"
# SOURCE_TEXT_COLUMN = 'text' # Use the actual column name 'text'
# SOURCE_LABEL_COLUMN = 'label' # Use the actual column name 'label' (likely ClassLabel int)
# OUTPUT_CSV_BASENAME = "cajcodes_political-bias"

# --- Generic Settings ---
TARGET_TEXT_COLUMN = 'text' # Standardized output column name
TARGET_LABEL_COLUMN = 'label' # Standardized output column name
# Define the final labels we want to keep and standardize to
FINAL_LABELS_MAP = {
    # Map source labels (lowercase) to standardized labels
    'left': 'left',
    'center': 'center',
    'right': 'right',
    # Add other potential mappings if needed, e.g.:
    # 'left-center': 'center', # Or decide how to handle these
    # 'right-center': 'center',
    # 'least biased': 'center',
    # 'mixed': 'center', # Example mapping
}
FINAL_LABELS_LIST = ['left', 'right', 'center'] # The labels to actually keep

# --- Helper Functions ---
def clean_text(text):
    """Basic text cleaning function."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    # Keep basic punctuation that might be relevant
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Preprocessing Logic ---
print(f"--- Starting Generic HF Dataset Processing ---")
print(f"Processing Dataset: {HF_DATASET_NAME}")
print(f"Source Text Column: {SOURCE_TEXT_COLUMN}")
print(f"Source Label Column: {SOURCE_LABEL_COLUMN}") # Will now print 'label'
print(f"Output CSV Basename: {OUTPUT_CSV_BASENAME}")

# 1. Create Output Directory
try:
    os.makedirs(PROCESSED_CSV_DIR, exist_ok=True)
    print(f"Output directory check/creation successful: {PROCESSED_CSV_DIR}")
except Exception as e:
    print(f"Error creating directory {PROCESSED_CSV_DIR}: {e}")
    exit()


# 2. Load Dataset
try:
    print(f"Loading dataset '{HF_DATASET_NAME}' from Hugging Face...")
    # Load all available splits, common ones are 'train', 'test', maybe 'validation'
    raw_datasets = load_dataset(HF_DATASET_NAME)
    print("Dataset loaded successfully.")
    print(f"Available splits: {list(raw_datasets.keys())}")

    # Determine if label column is ClassLabel or string
    # Check the first available split (assuming features are consistent)
    first_split_key = list(raw_datasets.keys())[0]
    print(f"Dataset features: {raw_datasets[first_split_key].features}") # Print features for debugging
    # *** Access the correct label column name ***
    label_feature = raw_datasets[first_split_key].features[SOURCE_LABEL_COLUMN]
    is_class_label = isinstance(label_feature, ClassLabel)
    map_int_to_str = None
    if is_class_label:
        print(f"Label column '{SOURCE_LABEL_COLUMN}' is ClassLabel. Mapping integers to strings.")
        map_int_to_str = label_feature.int2str # This will get the mapping from 'label'
    else:
        # Check if it's integer type even if not ClassLabel
        if pd.api.types.is_integer_dtype(label_feature.dtype):
             print(f"Warning: Label column '{SOURCE_LABEL_COLUMN}' is integer but not ClassLabel. Assuming 0=left, 1=center, 2=right.")
             # Define a manual mapping if necessary (adjust if assumption is wrong)
             manual_map = {0: 'left', 1: 'center', 2: 'right'}
             # We'll apply this later if needed
        else:
            print(f"Label column '{SOURCE_LABEL_COLUMN}' is likely string type.")


except KeyError as e:
    print(f"Error: Column '{SOURCE_LABEL_COLUMN}' not found in dataset features.")
    print(f"Available features: {raw_datasets[first_split_key].features if 'raw_datasets' in locals() else 'Could not load features.'}")
    exit()
except Exception as e:
    print(f"Error loading dataset or getting label info: {e}")
    traceback.print_exc()
    exit()

# 3. Process and Combine All Available Splits
all_processed_dfs = []
for split_name in raw_datasets.keys():
    print(f"\n--- Processing split: {split_name} ---")
    try:
        current_split_ds = raw_datasets[split_name]
        print(f"Original {split_name} size: {len(current_split_ds)}")
        if len(current_split_ds) == 0:
            print(f"Split '{split_name}' is empty. Skipping.")
            continue

        # Convert to pandas
        print(f"Converting {split_name} split to Pandas DataFrame...")
        df = current_split_ds.to_pandas()

        # Check if required columns exist
        if SOURCE_TEXT_COLUMN not in df.columns or SOURCE_LABEL_COLUMN not in df.columns:
             raise ValueError(f"DataFrame missing required source columns '{SOURCE_TEXT_COLUMN}' or '{SOURCE_LABEL_COLUMN}'.")

        # --- Label Standardization ---
        # Apply int-to-string mapping if needed
        if is_class_label and map_int_to_str:
            print("Applying ClassLabel int-to-string mapping...")
            df[TARGET_LABEL_COLUMN] = df[SOURCE_LABEL_COLUMN].apply(map_int_to_str) # Map 0,1,2 to 'left','center','right'
        elif 'manual_map' in locals(): # Check if manual map was defined
             print("Applying manual integer-to-string mapping...")
             df[TARGET_LABEL_COLUMN] = df[SOURCE_LABEL_COLUMN].map(manual_map)
        else:
             # Assume it's already string or convert just in case
             print("Assuming label column is already string type...")
             df[TARGET_LABEL_COLUMN] = df[SOURCE_LABEL_COLUMN].astype(str)

        # Convert source labels to lowercase and apply mapping
        print("Standardizing labels using FINAL_LABELS_MAP...")
        df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].str.strip().str.lower()
        df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].map(FINAL_LABELS_MAP)

        # Filter out rows where label didn't map or is not in the final list
        initial_len = len(df)
        df.dropna(subset=[TARGET_LABEL_COLUMN], inplace=True) # Remove rows that didn't map
        df = df[df[TARGET_LABEL_COLUMN].isin(FINAL_LABELS_LIST)] # Keep only desired final labels
        if len(df) < initial_len:
            print(f"  Removed {initial_len - len(df)} rows due to unmapped or undesired labels.")

        # Select and rename text column
        df_processed = df[[SOURCE_TEXT_COLUMN, TARGET_LABEL_COLUMN]].copy()
        df_processed.rename(columns={SOURCE_TEXT_COLUMN: TARGET_TEXT_COLUMN}, inplace=True)

        # Clean text data
        print("Applying text cleaning function...")
        df_processed[TARGET_TEXT_COLUMN] = df_processed[TARGET_TEXT_COLUMN].apply(clean_text)
        print("Text cleaning applied.")

        # Drop rows with empty text OR empty/NaN labels after processing
        print("Dropping rows with NaN or empty text/labels...")
        initial_len = len(df_processed)
        df_processed.dropna(subset=[TARGET_TEXT_COLUMN, TARGET_LABEL_COLUMN], inplace=True)
        df_processed = df_processed[df_processed[TARGET_TEXT_COLUMN].astype(str).str.strip().astype(bool)]
        if len(df_processed) < initial_len:
             print(f"Removed {initial_len - len(df_processed)} rows with empty text or labels after cleaning.")

        print(f"Processed {len(df_processed)} rows from split '{split_name}'.")
        if not df_processed.empty:
            all_processed_dfs.append(df_processed)

    except Exception as e:
        print(f"!!! Error processing split {split_name}: {e}")
        traceback.print_exc() # Use traceback if imported

# 4. Combine processed splits and save
if not all_processed_dfs:
    print("\nError: No data processed from any split. No CSV file saved.")
    exit()

print("\nCombining processed data from all splits...")
final_df = pd.concat(all_processed_dfs, ignore_index=True)
print(f"Total processed rows for {HF_DATASET_NAME}: {len(final_df)}")

if final_df.empty:
    print("Warning: Final DataFrame is empty after combining splits. No CSV will be saved.")
else:
    # Save to CSV
    output_csv_path = os.path.join(PROCESSED_CSV_DIR, f"{OUTPUT_CSV_BASENAME}.csv")
    print(f"\nAttempting to save standardized data to: {output_csv_path}...")
    try:
        final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        if os.path.exists(output_csv_path):
             print(f"Successfully saved CSV to: {output_csv_path}")
        else:
             print(f"Error: CSV file was NOT saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving final CSV: {e}")

print(f"\n--- Finished processing {HF_DATASET_NAME} ---")

