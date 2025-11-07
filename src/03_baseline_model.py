import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless environments (Docker)
import matplotlib.pyplot as plt


def load_split_csv(processed_dir: str):
	"""Load train/val/test CSVs from processed_dir. Returns (train, val, test).

	If val/test are missing, returns None for them. Raises if train is missing.
	"""
	train_path = os.path.join(processed_dir, "train.csv")
	val_path = os.path.join(processed_dir, "val.csv")
	test_path = os.path.join(processed_dir, "test.csv")

	if not os.path.exists(train_path):
		# Fallback to a single processed file
		single_path = os.path.join(processed_dir, "processed_data.csv")
		if os.path.exists(single_path):
			df = pd.read_csv(single_path)
			return df, None, None
		raise FileNotFoundError(
			f"No training data found in {processed_dir}. Expected train.csv or processed_data.csv."
		)

	train_df = pd.read_csv(train_path)
	val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
	test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
	return train_df, val_df, test_df


def ensure_dirs(base_output: str):
	models_dir = os.path.join(base_output, "models")
	reports_dir = os.path.join(base_output, "reports")
	Path(models_dir).mkdir(parents=True, exist_ok=True)
	Path(reports_dir).mkdir(parents=True, exist_ok=True)
	return models_dir, reports_dir


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=labels, yticklabels=labels,
		   ylabel='True label', xlabel='Predicted label',
		   title='Confusion Matrix (Test)')
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	thresh = cm.max() / 2.0 if cm.size else 0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], 'd'),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")

	fig.tight_layout()
	fig.savefig(save_path, bbox_inches='tight')
	plt.close(fig)


def main():
	# Base output directory; processed data lives under processed/
	base_output = os.getenv('OUTPUT_DIR', '/app/output')
	processed_dir = os.path.join(base_output, 'processed')
	models_dir, reports_dir = ensure_dirs(base_output)

	print(f"Processed data dir: {processed_dir}")
	print(f"Models dir: {models_dir}")
	print(f"Reports dir: {reports_dir}")

	train_df, val_df, test_df = load_split_csv(processed_dir)

	# Basic sanity checks
	if 'text' not in train_df.columns or 'label' not in train_df.columns:
		raise ValueError("Train CSV must contain 'text' and 'label' columns")

	X_train = train_df['text'].astype(str).tolist()
	y_train = train_df['label'].astype(str).tolist()

	X_val, y_val = None, None
	if val_df is not None and {'text', 'label'}.issubset(val_df.columns):
		X_val = val_df['text'].astype(str).tolist()
		y_val = val_df['label'].astype(str).tolist()

	X_test, y_test = None, None
	if test_df is not None and {'text', 'label'}.issubset(test_df.columns):
		X_test = test_df['text'].astype(str).tolist()
		y_test = test_df['label'].astype(str).tolist()

	# Hyperparameters via env vars
	max_features = int(os.getenv('TFIDF_MAX_FEATURES', '20000'))
	ngram_max = int(os.getenv('TFIDF_NGRAM_MAX', '2'))
	C = float(os.getenv('LR_C', '1.0'))

	print(f"Training baseline model with TF-IDF(max_features={max_features}, ngram_range=(1,{ngram_max})) + LogisticRegression(C={C})")

	clf = Pipeline([
		('tfidf', TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features)) ,
		('lr', LogisticRegression(max_iter=1000, n_jobs=None, C=C))
	])

	clf.fit(X_train, y_train)

	# Save model
	model_path = os.path.join(models_dir, 'baseline_model.pkl')
	with open(model_path, 'wb') as f:
		pickle.dump(clf, f)
	print(f"Saved model to {model_path}")

	# Evaluate on val and test if available
	def evaluate_and_save(split_name, X, y):
		if X is None or y is None:
			print(f"No {split_name} split available; skipping evaluation.")
			return None
		y_pred = clf.predict(X)
		labels = sorted(list(set(y) | set(y_pred)))
		report = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
		# Save JSON report
		report_path = os.path.join(reports_dir, f'baseline_{split_name}_report.json')
		with open(report_path, 'w', encoding='utf-8') as f:
			json.dump(report, f, ensure_ascii=False, indent=2)
		print(f"Saved {split_name} report to {report_path}")
		# Confusion matrix only for test
		if split_name == 'test':
			cm_path = os.path.join(reports_dir, 'baseline_test_confusion_matrix.png')
			plot_confusion_matrix(y, y_pred, labels, cm_path)
			print(f"Saved test confusion matrix to {cm_path}")
		return report

	evaluate_and_save('val', X_val, y_val)
	evaluate_and_save('test', X_test, y_test)


if __name__ == "__main__":
	main()

