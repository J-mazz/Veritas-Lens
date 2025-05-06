import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split # For splitting data
import numpy as np # For shuffling indices
import traceback # For error details

# --- Configuration ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "political_bias_detector"
# Directory where individual processed CSVs (from Step 1 for each dataset) are stored
PROCESSED_CSV_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed_csv")
# Directory where the final combined/split TXT files will be saved
FINAL_TXT_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed_combined") # Use a new output dir name

# List of CSV filenames (without extension) in PROCESSED_CSV_DIR to combine
# *** VERIFY THIS LIST matches the CSV files you generated ***
SOURCE_CSV_BASENAMES = [
    'siddharthmb_article-bias-prediction-media-splits',
    'Faith1712_Allsides_political_bias_proper',
    'cajcodes_political-bias'
]

# Define the columns expected in the source CSVs
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'

# Define the final labels we care about (should match labels in source CSVs)
FINAL_LABELS = ['left', 'right', 'center']

# Define split ratios
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10 # Must sum to 1.0 with TRAIN_RATIO and VALIDATION_RATIO

# --- Helper Function to Save TXT Files ---
def save_data_to_txt_files(dataframe, base_output_path):
    """Saves dataframe rows into text files within class-specific folders."""
    print(f"Saving data to {base_output_path}...")
    os.makedirs(base_output_path, exist_ok=True)

    # Create subdirectories for each label
    for label_dir_name in FINAL_LABELS:
        label_dir = os.path.join(base_output_path, label_dir_name)
        os.makedirs(label_dir, exist_ok=True)

    saved_count = 0
    error_count = 0
    skipped_label_count = 0
    skipped_empty_count = 0

    # Use index from the dataframe for unique filenames
    for index, row in dataframe.iterrows():
        try:
            label_raw = row[LABEL_COLUMN]
            if pd.isna(label_raw):
                 skipped_label_count += 1
                 continue
            # Assume labels are already cleaned/standardized in source CSVs
            label = str(label_raw).strip().lower()

            if label not in FINAL_LABELS:
                # This shouldn't happen if source CSVs are pre-filtered
                print(f"Warning: Skipping row {index} with unexpected label: {label}")
                skipped_label_count += 1
                continue

            text_content = row[TEXT_COLUMN]
            if pd.isna(text_content):
                skipped_empty_count += 1
                continue
            text_content = str(text_content)

            if text_content.strip():
                # Use the dataframe's original index in the filename for uniqueness
                # Prepend split name to index to avoid potential collisions if indices overlap across sources
                # (though ignore_index=True on concat should prevent this)
                file_path = os.path.join(base_output_path, label, f"text_{index}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                saved_count += 1
            else:
                skipped_empty_count += 1

        except Exception as e:
            print(f"Error processing row {index} from dataframe: {e}")
            error_count += 1

    print(f"Saved {saved_count} files to {base_output_path}")
    if skipped_label_count > 0:
        print(f"Skipped {skipped_label_count} rows due to unexpected/missing labels.")
    if skipped_empty_count > 0:
         print(f"Skipped {skipped_empty_count} rows due to empty text content.")
    if error_count > 0:
        print(f"Encountered {error_count} errors during processing.")

# --- Main Processing Logic ---
print("Starting Dataset Combination and Splitting...")

all_dataframes = []
print(f"Looking for source CSVs in: {PROCESSED_CSV_DIR}")

# 1. Load and Concatenate DataFrames
for basename in SOURCE_CSV_BASENAMES:
    csv_path = os.path.join(PROCESSED_CSV_DIR, f"{basename}.csv")
    print(f"Attempting to load: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # Basic validation
        if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
             print(f"Warning: CSV {csv_path} is missing '{TEXT_COLUMN}' or '{LABEL_COLUMN}'. Skipping.")
             continue
        # Ensure labels are strings and lowercase for consistency before filtering
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
        # Filter for the desired labels ONLY
        initial_len = len(df)
        df = df[df[LABEL_COLUMN].isin(FINAL_LABELS)]
        filtered_len = len(df)
        if filtered_len < initial_len:
            print(f"  Filtered {initial_len - filtered_len} rows with labels outside {FINAL_LABELS}.")
        # Drop rows with missing text
        df.dropna(subset=[TEXT_COLUMN], inplace=True)
        df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
        # Drop rows with empty text after conversion
        df = df[df[TEXT_COLUMN].str.strip().astype(bool)]

        if not df.empty:
            print(f"  Loaded and filtered {len(df)} rows from {basename}.csv")
            all_dataframes.append(df[[TEXT_COLUMN, LABEL_COLUMN]]) # Keep only necessary columns
        else:
            print(f"  No valid rows found in {basename}.csv after filtering. Skipping.")

    except FileNotFoundError:
        print(f"Error: Source CSV file not found at {csv_path}. Skipping.")
    except Exception as e:
        print(f"Error loading or processing {csv_path}: {e}. Skipping.")
        traceback.print_exc() # Print traceback for other errors

if not all_dataframes:
    print("\nError: No valid data loaded from any source CSV. Exiting.")
    exit()

print("\nConcatenating all loaded dataframes...")
combined_df = pd.concat(all_dataframes, ignore_index=True) # ignore_index=True re-indexes the combined df
print(f"Total combined rows: {len(combined_df)}")
print("Value counts for combined labels:")
print(combined_df[LABEL_COLUMN].value_counts())

# 2. Shuffle the Combined DataFrame
print("\nShuffling combined data...")
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle and reset index

# 3. Split into Train, Validation, Test
print("\nSplitting data into train, validation, and test sets...")
if not (np.isclose(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0)):
    print("Error: Split ratios must sum to 1.0")
    exit()

# First split: separate test set
try:
    train_val_df, test_df = train_test_split(
        combined_df,
        test_size=TEST_RATIO,
        random_state=42,
        stratify=combined_df[LABEL_COLUMN] # Stratify to keep label proportions similar
    )
except ValueError as e:
    print(f"\nWarning: Error during stratified test split (potentially too few samples for stratification): {e}")
    print("Consider adjusting split ratios or checking label distribution.")
    # Fallback to non-stratified split if needed, though less ideal
    print("Attempting non-stratified split...")
    train_val_df, test_df = train_test_split(
        combined_df,
        test_size=TEST_RATIO,
        random_state=42
    )


# Second split: separate train and validation from the remaining data
# Adjust validation ratio relative to the remaining train_val data
if TRAIN_RATIO + VALIDATION_RATIO > 0: # Avoid division by zero
    relative_val_ratio = VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO)
    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_ratio,
            random_state=42,
            stratify=train_val_df[LABEL_COLUMN] # Stratify again
        )
    except ValueError as e:
        print(f"\nWarning: Error during stratified train/validation split (potentially too few samples for stratification): {e}")
        print("Consider adjusting split ratios or checking label distribution.")
        # Fallback to non-stratified split if needed
        print("Attempting non-stratified split...")
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_ratio,
            random_state=42
        )
else:
    # Handle edge case where only test split is defined
    train_df = pd.DataFrame(columns=[TEXT_COLUMN, LABEL_COLUMN])
    val_df = pd.DataFrame(columns=[TEXT_COLUMN, LABEL_COLUMN])


print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# 4. Save Split DataFrames to TXT Directory Structure
print(f"\nCreating final output directory: {FINAL_TXT_DIR}")
os.makedirs(FINAL_TXT_DIR, exist_ok=True)

save_data_to_txt_files(train_df, os.path.join(FINAL_TXT_DIR, "train"))
save_data_to_txt_files(val_df, os.path.join(FINAL_TXT_DIR, "valid"))
save_data_to_txt_files(test_df, os.path.join(FINAL_TXT_DIR, "test"))

print("\nDataset combination, splitting, and saving complete.")
print(f"Final data structure for training saved in: {FINAL_TXT_DIR}")
print(f"IMPORTANT: Update PROCESSED_DATA_DIR in train_model_final.py to point to '{FINAL_TXT_DIR}' before training.")

