import pandas as pd
import os

# --- Configuration ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "political_bias_detector"
PROCESSED_CSV_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed_csv")
FINAL_TXT_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed")

# Define the columns in the intermediate CSVs
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'

# *** Define the bias labels in LOWERCASE for robust comparison ***
LABELS = ['left', 'right', 'center']

# --- Helper Functions ---
def save_data_to_txt_files(dataframe, base_output_path):
    """Saves dataframe rows into text files within class-specific folders."""
    print(f"Saving data to {base_output_path}...")
    os.makedirs(base_output_path, exist_ok=True)

    # Create subdirectories for each label (using the lowercase defined LABELS)
    for label_dir_name in LABELS:
        label_dir = os.path.join(base_output_path, label_dir_name)
        os.makedirs(label_dir, exist_ok=True)

    saved_count = 0
    error_count = 0
    skipped_label_count = 0
    skipped_empty_count = 0

    for index, row in dataframe.iterrows():
        try:
            # *** Read label, convert to string, strip whitespace, convert to lowercase ***
            label_raw = row[LABEL_COLUMN]
            if pd.isna(label_raw): # Check for NaN/None
                 skipped_label_count += 1
                 continue
            label = str(label_raw).strip().lower() # Convert to lowercase string

            # *** Check if the lowercase label is in the lowercase LABELS list ***
            if label not in LABELS:
                # print(f"Skipping row {index} with unexpected label: {label_raw} -> {label}") # Optional warning
                skipped_label_count += 1
                continue

            text_content = row[TEXT_COLUMN]
            # Ensure text_content is a string
            if pd.isna(text_content):
                skipped_empty_count += 1
                continue
            text_content = str(text_content)

            # Avoid writing empty files if text became empty after cleaning
            if text_content.strip():
                # *** Use the matched lowercase label for the directory path ***
                file_path = os.path.join(base_output_path, label, f"text_{index}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                saved_count += 1
            else:
                # print(f"Skipping empty text content for index {index}") # Optional: Log skipped empty files
                skipped_empty_count += 1

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            error_count += 1

    print(f"Saved {saved_count} files to {base_output_path}")
    if skipped_label_count > 0:
        print(f"Skipped {skipped_label_count} rows due to unexpected/missing labels.")
    if skipped_empty_count > 0:
         print(f"Skipped {skipped_empty_count} rows due to empty text content.")
    if error_count > 0:
        print(f"Encountered {error_count} errors during processing.")

# --- Main Processing Logic ---
print("Starting Step 2: Processed CSV -> TXT File Structure...")
os.makedirs(FINAL_TXT_DIR, exist_ok=True)

for split_name in ['train', 'valid', 'test']:
    print(f"\nProcessing split: {split_name}...")
    input_csv_path = os.path.join(PROCESSED_CSV_DIR, f"{split_name}.csv")
    output_split_path = os.path.join(FINAL_TXT_DIR, split_name)

    try:
        print(f"Loading data from {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        # Handle potential NaN values that might result from cleaning/loading
        df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True) # Keep initial dropna
        print(f"Loaded {len(df)} rows from {split_name}.csv.")
        if len(df) == 0:
            print(f"Warning: No data loaded from {input_csv_path}. Skipping.")
            continue
        if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
            raise ValueError(f"CSV {input_csv_path} must contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns.")

    except FileNotFoundError:
        print(f"Error: Processed CSV file not found at {input_csv_path}")
        print("Please ensure Step 1 (preprocess_hf_to_csv.py) was run successfully.")
        continue
    except Exception as e:
        print(f"Error loading or validating CSV {input_csv_path}: {e}")
        continue

    save_data_to_txt_files(df, output_split_path)

print("\nStep 2: Processed CSV -> TXT File Structure complete.")
print(f"Final data structure for training saved in: {FINAL_TXT_DIR}")
