import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Adjust these paths based on your Google Drive structure
DRIVE_BASE_PATH = "/content/drive/MyDrive/"
PROJECT_FOLDER = "political_bias_detector"
RAW_DATA_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER, "data", "processed")
RAW_CSV_FILE = os.path.join(RAW_DATA_DIR, "news_bias.csv") # Change if your CSV has a different name

# Define the columns in your CSV that contain the text and the label
TEXT_COLUMN = 'text' # Adjust if your text column is named differently
LABEL_COLUMN = 'bias_label' # Adjust if your label column is named differently

# Define the bias labels (ensure these match your CSV labels exactly)
LABELS = ['left', 'left_center', 'objective', 'right_center', 'right']

# Define split ratios
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15 # Validation split from the remaining training data

# --- Helper Functions ---
def clean_text(text):
    """Basic text cleaning function."""
    if not isinstance(text, str):
        text = str(text) # Ensure text is string
    text = text.lower() # Lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Add more cleaning steps if needed (e.g., remove URLs, special chars)
    return text

def save_data_to_files(dataframe, base_path):
    """Saves dataframe rows into text files within class-specific folders."""
    print(f"Saving data to {base_path}...")
    # Ensure base directory exists
    os.makedirs(base_path, exist_ok=True)

    # Create subdirectories for each label if they don't exist
    for label in LABELS:
        label_dir = os.path.join(base_path, label)
        os.makedirs(label_dir, exist_ok=True)

    # Save each text entry as a separate .txt file in its corresponding label folder
    saved_count = 0
    for index, row in dataframe.iterrows():
        label = row[LABEL_COLUMN]
        text_content = row[TEXT_COLUMN]
        # Use index as part of the filename to ensure uniqueness
        file_path = os.path.join(base_path, label, f"text_{index}.txt")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            saved_count += 1
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")

    print(f"Saved {saved_count} files to {base_path}")

# --- Main Preprocessing Logic ---
print("Starting data preprocessing...")

# 1. Load Data
try:
    print(f"Loading data from {RAW_CSV_FILE}...")
    df = pd.read_csv(RAW_CSV_FILE)
    print(f"Loaded {len(df)} rows.")
    # Basic validation
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"CSV must contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns.")
    # Filter out rows with missing text or labels
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    # Filter out rows with labels not in our defined LABELS list
    df = df[df[LABEL_COLUMN].isin(LABELS)]
    print(f"Using {len(df)} rows after filtering.")
    if len(df) == 0:
        raise ValueError("No valid data found after filtering. Check CSV content and LABELS.")
except FileNotFoundError:
    print(f"Error: Raw CSV file not found at {RAW_CSV_FILE}")
    print("Please ensure the file exists and the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading or validating CSV: {e}")
    exit()

# 2. Clean Text Data
print("Cleaning text data...")
df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

# 3. Split Data (Train/Val/Test)
print("Splitting data into train, validation, and test sets...")
# First split: Separate test set
train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=42,  # for reproducibility
    stratify=df[LABEL_COLUMN] # Ensure proportional representation of labels
)

# Second split: Separate validation set from the remaining data
# Adjust validation size relative to the remaining train_val data
val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_size_adjusted,
    random_state=42, # for reproducibility
    stratify=train_val_df[LABEL_COLUMN] # Ensure proportional representation
)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# 4. Save Processed Data
train_path = os.path.join(PROCESSED_DATA_DIR, "train")
val_path = os.path.join(PROCESSED_DATA_DIR, "val")
test_path = os.path.join(PROCESSED_DATA_DIR, "test")

save_data_to_files(train_df, train_path)
save_data_to_files(val_df, val_pat
save_data_to_files(test_df, test_path)

print("Data preprocessing complete.")
print(f"Processed data saved in: {PROCESSED_DATA_DIR}")
