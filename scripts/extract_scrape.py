"""
Generic News/Text Data Collection and Preparation Script

- Downloads from Hugging Face Datasets (public or with token)
- Downloads from Kaggle (requires kaggle.json)
- Processes all .csv/.json files in data/raw/
- Cleans, deduplicates, and outputs a single CSV for ML use

Usage in Colab:
  1. (Optional) Set HF_TOKEN in your environment for Hugging Face gated datasets.
  2. (Optional) Upload kaggle.json and place in ~/.kaggle/
  3. Edit HF_DATASETS and KAGGLE_DATASETS as needed.
  4. Run this script!

Author: Copilot & J-mazz
"""

import os
import glob
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

HF_TOKEN = os.environ.get("HF_TOKEN")  # Set in Colab or .env if needed

# ========== CONFIGURE DATASETS HERE ==========

# Hugging Face datasets (repo_id, optional subset)
HF_DATASETS = [
    ("ag_news", None),
    ("cc_news", None),
    # ("dataset/repo", "subset")  # Add more as needed
]

# Kaggle datasets (owner/dataset-name)
KAGGLE_DATASETS = [
    "therohk/million-headlines",
    "crowdflower/bbc-news",
    # Add more as needed
]

RAW_DIR = "data/raw"
CLEANED_OUT = "data/cleaned/all_cleaned_news.csv"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEANED_OUT), exist_ok=True)

# ========== Download from Hugging Face ==========

def download_hf_dataset(dataset_name, subset):
    try:
        ds = load_dataset(dataset_name, subset, split='train', token=HF_TOKEN)
        safe_name = dataset_name.replace('/', '_') + (f"_{subset}" if subset else "")
        out_file = f"{RAW_DIR}/{safe_name}.csv"
        ds.to_csv(out_file)
        print(f"Downloaded {dataset_name} to {out_file}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")

# ========== Download from Kaggle ==========

def download_kaggle_dataset(dataset):
    try:
        os.system(f"kaggle datasets download -d {dataset} -p {RAW_DIR} --unzip")
        print(f"Downloaded {dataset}")
    except Exception as e:
        print(f"Failed to download {dataset}: {e}")

# ========== Run Downloads in Parallel ==========

with ThreadPoolExecutor(max_workers=4) as exec:
    # Hugging Face
    exec.map(lambda args: download_hf_dataset(*args), HF_DATASETS)
    # Kaggle
    exec.map(download_kaggle_dataset, KAGGLE_DATASETS)

# ========== Preprocess: Clean, Deduplicate, Standardize ==========

def clean_text(text):
    # Generic cleaning: strip whitespace, lower, basic unicode fix, etc.
    if not isinstance(text, str): return ""
    return text.strip().replace('\n', ' ').replace('\r', '').lower()

def extract_text_fields(df):
    # Try to extract a main text column (adjust as needed)
    for col in ['text', 'article', 'content', 'headline', 'body']:
        if col in df.columns:
            return df[col].astype(str)
    # Fallback to all columns joined
    return df.astype(str).agg(' '.join, axis=1)

all_texts = []

for path in glob.glob(f"{RAW_DIR}/*.csv") + glob.glob(f"{RAW_DIR}/*.json"):
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_json(path, lines=True)
        texts = extract_text_fields(df).map(clean_text)
        all_texts.extend(texts.tolist())
        print(f"Loaded {len(texts)} records from {path}")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

# Deduplicate and save
cleaned = pd.DataFrame({"cleaned_text": pd.Series(list(set(all_texts)))})
cleaned = cleaned[cleaned["cleaned_text"].str.len() > 30]  # Remove trivially short texts
cleaned.to_csv(CLEANED_OUT, index=False)
print(f"\nDone! Final cleaned dataset at {CLEANED_OUT} with {len(cleaned)} unique articles.")
