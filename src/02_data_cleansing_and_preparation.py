import os
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_PREFIX = '02-preparation'

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
	SentenceTransformer = None  # embeddings optional


def clean_text(text):
   """Clean and preprocess text data for Hungarian legal texts (lowercase, normalize, strip)."""
   if pd.isna(text) or text == "":
	   return ""
   text = str(text)
   text = text.lower()  # lowercase for deduplication and normalization
   text = unicodedata.normalize('NFC', text)
   text = re.sub(r'\s+', ' ', text)
   text = re.sub(r'[^\w\s\.,!\?;:\-–—\(\)"\'„"%/€$…]', '', text)
   return text.strip()


def stratified_split(df, target_column, test_size=0.2, val_size=0.2, random_state=42):
	"""Split data into train/validation/test sets with stratification."""
	train_val, test = train_test_split(
		df, test_size=test_size, stratify=df[target_column], random_state=random_state
	)
	val_size_adjusted = val_size / (1 - test_size)
	train, val = train_test_split(
		train_val, test_size=val_size_adjusted, stratify=train_val[target_column], random_state=random_state
	)
	return train, val, test


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
	"""Add basic text stats: word_count, avg_word_len."""
	if 'text' not in df.columns:
		return df
	texts = df['text'].astype(str)
	df = df.copy()
	df['word_count'] = texts.apply(lambda t: len(t.split()))
	df['avg_word_len'] = texts.apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0.0)
	return df


def save_histogram(series: pd.Series, title: str, path: str, bins: int = 50):
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.hist(series.values, bins=bins)
	ax.set_title(title)
	ax.set_xlabel('Value')
	ax.set_ylabel('Frequency')
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def maybe_compute_embeddings(df_list, features_dir: str, model_name: str, batch_size: int = 32):
	"""Optionally compute Sentence-BERT embeddings for train/val/test splits."""
	if SentenceTransformer is None:
		print("sentence-transformers not available; skipping embeddings.")
		return None
	print(f"Loading embedding model: {model_name}")
	model = SentenceTransformer(model_name)
	split_names = ['train', 'val', 'test']
	meta = {"model": model_name}
	for name, df in zip(split_names, df_list):
		if df is None:
			continue
		texts = df['text'].astype(str).tolist()
		print(f"Encoding {name} split with {len(texts)} texts...")
		emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
		out_path = os.path.join(features_dir, f'{STEP_PREFIX}_embeddings_{name}.npy')
		np.save(out_path, emb)
		meta[f'{name}_embeddings'] = out_path
	meta_path = os.path.join(features_dir, f'{STEP_PREFIX}_embeddings_meta.json')
	with open(meta_path, 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print(f"Saved embeddings metadata to {meta_path}")
	return meta


def main():

   base_output = os.getenv('OUTPUT_DIR', '/app/output')
   raw_dir = os.path.join(base_output, 'raw')
   processed_dir = os.path.join(base_output, 'processed')
   features_dir = os.path.join(base_output, 'reports')
   Path(processed_dir).mkdir(parents=True, exist_ok=True)
   Path(features_dir).mkdir(parents=True, exist_ok=True)

   print(f"RAW input: {raw_dir}")
   print(f"Processed output: {processed_dir}")
   print(f"Reports dir (metrics/plots): {features_dir}")

   # Load raw dataset (not EDA-filtered, so we do all cleaning here)
   raw_csv = os.path.join(raw_dir, 'raw_dataset.csv')
   if not os.path.exists(raw_csv):
	   raise FileNotFoundError(f"Missing {raw_csv}. Run 01_data_acquisition_and_analysis.py first.")
   df_raw = pd.read_csv(raw_csv)
   print(f"Loaded {len(df_raw)} rows from {raw_csv}")

   # Extract label if needed (if label_raw not present)
   if 'label_raw' not in df_raw.columns:
	   def extract_label_from_annotations(annotations_raw):
		   if not annotations_raw:
			   return None
		   try:
			   annotations = json.loads(annotations_raw)
			   if annotations and isinstance(annotations, list) and len(annotations) > 0:
				   return annotations[0]['result'][0]['value']['choices'][0]
		   except (KeyError, IndexError, TypeError, json.JSONDecodeError):
			   return None
		   return None
	   df_raw['label_raw'] = df_raw.get('annotations_raw', '').apply(extract_label_from_annotations)

   # Build working DataFrame
   df = pd.DataFrame({
	   'text': df_raw['text_raw'],
	   'label': df_raw['label_raw']
   })
   print(f"Rows before cleaning: {len(df)}")

   # Clean text (lowercase, normalize, etc.)
   df['text'] = df['text'].apply(clean_text)

   # Remove empty text or label
   before = len(df)
   missing_mask = (df['text'].isna() | (df['text'] == '')) | (df['label'].isna() | (df['label'] == ''))
   missing_rows = df[missing_mask].copy()
   df = df[~missing_mask].reset_index(drop=True)
   after_missing = len(df)
   print(f"Rows after removing empty text/label: {after_missing} (removed {before - after_missing})")

   # Deduplicate by lowercase-cleaned text
   df['text_dedup'] = df['text'].str.lower()
   before_dedup = len(df)
   df = df.drop_duplicates(subset=['text_dedup'], keep='first').reset_index(drop=True)
   after_dedup = len(df)
   print(f"Rows after deduplication: {after_dedup} (removed {before_dedup - after_dedup})")

   # Save removed rows
   if before - after_missing > 0:
	   missing_file = os.path.join(raw_dir, f'{STEP_PREFIX}_removed_missing_labels_or_text.csv')
	   missing_rows.to_csv(missing_file, index=False, encoding='utf-8-sig')
	   print(f"Saved removed missing rows to {missing_file}")
   if before_dedup - after_dedup > 0:
	   dup_file = os.path.join(raw_dir, f'{STEP_PREFIX}_removed_duplicates.csv')
	   # Save only the dropped duplicates
	   # Mark which rows were dropped
	   deduped = df['text_dedup'].tolist()
	   dropped = df_raw[~df_raw['text_raw'].str.lower().isin(deduped)]
	   dropped.to_csv(dup_file, index=False, encoding='utf-8-sig')
	   print(f"Saved removed duplicates to {dup_file}")

   # Drop helper column
   df = df.drop(columns=['text_dedup'])


	   # Stratified split
	   target_column = 'label'
	   if target_column in df.columns and len(df) > 10:
		   print("Performing stratified split...")
		   train_df, val_df, test_df = stratified_split(df, target_column)
		   print(f"Train set: {len(train_df)}, Val set: {len(val_df)}, Test set: {len(test_df)}")

		   # Add text stats to each split
		   for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
			   aug = add_text_stats(split_df)
			   out_csv = os.path.join(processed_dir, f'{name}.csv')
			   aug.to_csv(out_csv, index=False, encoding='utf-8-sig')
			   print(f"Saved {name}.csv with stats -> {out_csv}")

		   # Clean EDA histograms (from train split)
		   if len(train_df) > 0 and 'text' in train_df.columns:
			   temp = add_text_stats(train_df)
			   if 'word_count' in temp.columns:
				   save_histogram(temp['word_count'], 'CLEAN Word Count Distribution (Train)',
								  os.path.join(features_dir, f'{STEP_PREFIX}_clean_word_count_hist.png'))
			   if 'avg_word_len' in temp.columns:
				   save_histogram(temp['avg_word_len'], 'CLEAN Average Word Length Distribution (Train)',
								  os.path.join(features_dir, f'{STEP_PREFIX}_clean_avg_word_len_hist.png'))

		   # Optional embeddings
		   enable_embeddings = os.getenv('ENABLE_EMBEDDINGS', 'false').lower() in {'1', 'true', 'yes'}
		   if enable_embeddings:
			   emb_model = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
			   maybe_compute_embeddings([train_df, val_df, test_df], features_dir, emb_model)
		   else:
			   print("Embeddings disabled (set ENABLE_EMBEDDINGS=true to enable).")
	   else:
		   print(f"Insufficient data for stratified split. Saving single file.")
		   aug = add_text_stats(df)
		   aug.to_csv(os.path.join(processed_dir, 'processed_data.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
	main()