import pandas as pd
import glob
import re
import os

def clean_text(text):
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", str(text))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_file(file_path):
    ext = os.path.splitext(file_path)[-1]
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.json':
        df = pd.read_json(file_path, lines=True)
    else:
        print(f"Skipping unsupported file: {file_path}")
        return None

    # Try to find a text/content column
    text_col = next((col for col in df.columns if col.lower() in ["text", "content", "article", "body", "news_content"]), None)
    if not text_col:
        print(f"No text column found in {file_path}, skipping.")
        return None

    # Clean text
    df['cleaned_text'] = df[text_col].apply(clean_text)

    # Keep only cleaned text and maybe date/source if wanted
    keep_cols = ['cleaned_text']
    for meta in ['date', 'source', 'url']:
        if meta in df.columns:
            keep_cols.append(meta)

    cleaned_df = df[keep_cols].drop_duplicates(subset=['cleaned_text']).dropna(subset=['cleaned_text'])
    return cleaned_df

def main(input_dir, output_file):
    files = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.json"))
    all_cleaned = []
    for f in files:
        cleaned = process_file(f)
        if cleaned is not None:
            all_cleaned.append(cleaned)
    if all_cleaned:
        df_out = pd.concat(all_cleaned, ignore_index=True)
        df_out.to_csv(output_file, index=False)
        print(f"Saved cleaned dataset to {output_file}")
    else:
        print("No valid files processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing raw CSV/JSON news datasets")
    parser.add_argument("--output_file", required=True, help="CSV file to write cleaned news stories")
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
