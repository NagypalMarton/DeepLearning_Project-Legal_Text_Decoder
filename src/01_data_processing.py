import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import os
import unicodedata
from pathlib import Path

def load_json_data(file_path):
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        raise

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to string if not already
    text = str(text)
    # Normalize Unicode to preserve and standardize accented characters
    # NFC keeps composed forms (recommended for display and matching)
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep common punctuation and symbols used in HU legal texts
    # Keep: word chars (incl. accented letters), whitespace, . , ! ? ; : ( ) - – — quotes, percent, slash, €, $, ellipsis
    text = re.sub(r'[^\w\s\.,!\?;:\-–—\(\)"\'„”%/€$…]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def stratified_split(df, target_column, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train/validation/test sets with stratification"""
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_column], 
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, 
        test_size=val_size_adjusted, 
        stratify=train_val[target_column], 
        random_state=random_state
    )
    
    return train, val, test

def process_legal_data(json_file_path, output_dir=None):
    """Main function to process legal text data"""
    
    # Use environment variables for Docker compatibility
    if output_dir is None:
        output_dir = os.getenv('OUTPUT_DIR', '/app/output/processed')
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    print("Loading JSON data...")
    data = load_json_data(json_file_path)

    # Speciális feldolgozás a mintád szerkezetéhez
    records = []
    for item in data:
        # Szöveg
        text = item.get("data", {}).get("text", "")
        # Címke (ha van annotáció és choices)
        label = None
        try:
            label = item["annotations"][0]["result"][0]["value"]["choices"][0]
        except (KeyError, IndexError, TypeError):
            label = None
        records.append({"text": text, "label": label})

    df = pd.DataFrame(records)

    # Clean text column
    print("Cleaning text column...")
    df["text"] = df["text"].apply(clean_text)
    # Remove rows with empty text or missing label
    df = df[(df["text"] != "") & (df["label"].notna())]
    df = df.reset_index(drop=True)

    print(f"Data shape after cleaning: {df.shape}")

    target_column = "label"
    if target_column in df.columns:
        # Stratified split
        print("Performing stratified split...")
        train_df, val_df, test_df = stratified_split(df, target_column)
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        # Save to CSV files
        # Use UTF-8 with BOM for better Windows/Excel compatibility
        train_df.to_csv(f"{output_dir}/train.csv", index=False, encoding='utf-8-sig')
        val_df.to_csv(f"{output_dir}/val.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv(f"{output_dir}/test.csv", index=False, encoding='utf-8-sig')
        print(f"Data saved to {output_dir}/")
    else:
        print(f"Target column '{target_column}' not found. Saving full dataset.")
        df.to_csv(f"{output_dir}/processed_data.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    # Use environment variables for Docker compatibility
    data_dir = os.getenv('DATA_DIR', '/app/data')
    json_file_path = f"{data_dir}/legal_data.json"  # Most már csak a data mappában
    print(f"Looking for JSON file at: {json_file_path}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {os.getenv('OUTPUT_DIR', '/app/output/processed')}")
    process_legal_data(json_file_path)