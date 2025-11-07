import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
	SentenceTransformer = None  # embeddings optional


def load_split(processed_dir: str):
	train = os.path.join(processed_dir, 'train.csv')
	val = os.path.join(processed_dir, 'val.csv')
	test = os.path.join(processed_dir, 'test.csv')
	single = os.path.join(processed_dir, 'processed_data.csv')

	if os.path.exists(train):
		train_df = pd.read_csv(train)
		val_df = pd.read_csv(val) if os.path.exists(val) else None
		test_df = pd.read_csv(test) if os.path.exists(test) else None
		return train_df, val_df, test_df
	elif os.path.exists(single):
		df = pd.read_csv(single)
		return df, None, None
	else:
		raise FileNotFoundError(
			f"No processed CSVs found in {processed_dir}. Run 01_data_processing.py first.")


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
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
		out_path = os.path.join(features_dir, f'embeddings_{name}.npy')
		np.save(out_path, emb)
		meta[f'{name}_embeddings'] = out_path
	meta_path = os.path.join(features_dir, 'embeddings_meta.json')
	with open(meta_path, 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print(f"Saved embeddings metadata to {meta_path}")
	return meta


def main():
	base_output = os.getenv('OUTPUT_DIR', '/app/output')
	processed_dir = os.path.join(base_output, 'processed')
	features_dir = os.path.join(base_output, 'features')
	Path(features_dir).mkdir(parents=True, exist_ok=True)

	print(f"Processed dir: {processed_dir}")
	print(f"Features dir: {features_dir}")

	train_df, val_df, test_df = load_split(processed_dir)

	# Add simple text stats and save augmented CSVs next to originals
	for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
		if df is None:
			continue
		aug = add_text_stats(df)
		out_csv = os.path.join(processed_dir, f'{name}.csv')
		aug.to_csv(out_csv, index=False, encoding='utf-8-sig')
		print(f"Saved augmented {name}.csv with text stats -> {out_csv}")

	# Save histograms for EDA
	if train_df is not None and 'text' in train_df.columns:
		temp = add_text_stats(train_df)
		if 'word_count' in temp.columns:
			save_histogram(temp['word_count'], 'Word Count Distribution (Train)',
						   os.path.join(features_dir, 'train_word_count_hist.png'))
		if 'avg_word_len' in temp.columns:
			save_histogram(temp['avg_word_len'], 'Average Word Length Distribution (Train)',
						   os.path.join(features_dir, 'train_avg_word_len_hist.png'))

	# Optional embeddings (can be large and require internet)
	enable_embeddings = os.getenv('ENABLE_EMBEDDINGS', 'false').lower() in {'1', 'true', 'yes'}
	if enable_embeddings:
		emb_model = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
		maybe_compute_embeddings([train_df, val_df, test_df], features_dir, emb_model)
	else:
		print("Embeddings disabled (set ENABLE_EMBEDDINGS=true to enable).")


if __name__ == '__main__':
	main()

# import os
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
# from sentence_transformers import SentenceTransformer

# import matplotlib.pyplot as plt

# # Resolve input/output directories using environment variables for Docker
# OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/app/output/processed')
# INPUT_DIR = OUTPUT_DIR  # processed CSVs are saved by step 01 into OUTPUT_DIR

# # Ensure output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load data from previous step (01): prefer train.csv; fallback to processed_data.csv
# train_csv = os.path.join(INPUT_DIR, 'train.csv')
# processed_csv = os.path.join(INPUT_DIR, 'processed_data.csv')

# if os.path.exists(train_csv):
#     df = pd.read_csv(train_csv)
# elif os.path.exists(processed_csv):
#     df = pd.read_csv(processed_csv)
# else:
#     raise FileNotFoundError(
#         f"No processed data found. Expected one of: {train_csv} or {processed_csv}. "
#         "Run 01_data_processing.py first to generate processed CSVs."
#     )

# # texts = df['text'].astype(str).tolist()

# # TF-IDF and n-gram features
# tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=1000)
# tfidf_features = tfidf.fit_transform(texts).toarray()
# tfidf_feature_names = tfidf.get_feature_names_out().tolist()

# # Word length and sentence length features
# word_lengths = [np.mean([len(word) for word in text.split()]) for text in texts]
# sentence_lengths = [len(text.split()) for text in texts]
# stat_feature_names = ["avg_word_len", "word_count"]

# # Embedding-based features (Sentence-BERT)
# # Prefer higher-accuracy multilingual model by default; allow override via env var
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
# print(f"Using embedding model: {EMBEDDING_MODEL}")
# model = SentenceTransformer(EMBEDDING_MODEL)
# embeddings = model.encode(texts, show_progress_bar=True)
# emb_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
# embedding_feature_names = [f"emb_{i}" for i in range(emb_dim)]

# # Combine all features
# features = np.hstack([
#     tfidf_features,
#     np.array(word_lengths).reshape(-1, 1),
#     np.array(sentence_lengths).reshape(-1, 1),
#     embeddings
# ])
# feature_names = tfidf_feature_names + stat_feature_names + embedding_feature_names

# # Save features
# features_path = os.path.join(OUTPUT_DIR, '02_features.npy')
# np.save(features_path, features)

# # Feature importance visualization (using RandomForest)
# if 'label' in df.columns:
#     y = df['label']
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(features, y)
#     # Compute permutation importance; handle NaNs and tiny values
#     result = permutation_importance(rf, features, y, n_repeats=15, random_state=42)
#     importances = result.importances_mean
#     # Sanitize importances
#     importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)

#     # Fallback to model-based importances if all zeros
#     if not np.any(importances):
#         if hasattr(rf, 'feature_importances_'):
#             importances = rf.feature_importances_
#         importances = np.nan_to_num(importances, nan=0.0)

#     # Visualize top 20 features
#     plt.figure(figsize=(10,6))
#     # Select top indices by absolute importance and filter zeros
#     nonzero_mask = importances != 0
#     if np.any(nonzero_mask):
#         ranked_idx = np.argsort(np.abs(importances))
#         top_idx = ranked_idx[-20:]
#         top_vals = importances[top_idx]
#         top_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}' for i in top_idx]
#         plt.barh(range(len(top_idx)), top_vals)
#         plt.yticks(range(len(top_idx)), top_names)
#     else:
#         # No informative features; annotate plot
#         plt.text(0.5, 0.5, 'All feature importances are zero/NaN', ha='center', va='center')
#         top_idx, top_vals, top_names = [], [], []
#     plt.xlabel("Feature Importance")
#     plt.title("Top 20 Feature Importances")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, '02_feature_importance.png'))
#     plt.close()

#     # Save importances to CSV for debugging/inspection
#     try:
#         all_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}' for i in range(len(importances))]
#         imp_df = pd.DataFrame({
#             'feature': all_names,
#             'importance': importances
#         }).sort_values('importance', ascending=False)
#         imp_df.to_csv(os.path.join(OUTPUT_DIR, '02_feature_importance.csv'), index=False)
#     except Exception:
#         pass