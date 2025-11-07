import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_eval_dir(base_output: str):
	eval_dir = os.path.join(base_output, 'evaluation')
	Path(eval_dir).mkdir(parents=True, exist_ok=True)
	return eval_dir


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
		   xticklabels=labels, yticklabels=labels,
		   ylabel='True label', xlabel='Predicted label',
		   title='Confusion Matrix (Test)')
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	thresh = cm.max() / 2.0 if cm.size else 0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	fig.savefig(save_path, bbox_inches='tight')
	plt.close(fig)


def main():
	base_output = os.getenv('OUTPUT_DIR', '/app/output')
	processed_dir = os.path.join(base_output, 'processed')
	models_dir = os.path.join(base_output, 'models')
	eval_dir = ensure_eval_dir(base_output)

	model_path = os.path.join(models_dir, 'baseline_model.pkl')
	test_path = os.path.join(processed_dir, 'test.csv')

	if not os.path.exists(model_path):
		print(f"Model not found at {model_path}. Train the baseline first (03_train_baseline.py).")
		return

	if not os.path.exists(test_path):
		print(f"Test CSV not found at {test_path}. Run data processing first (01_data_processing.py).")
		return

	with open(model_path, 'rb') as f:
		clf = pickle.load(f)

	test_df = pd.read_csv(test_path)
	if not {'text', 'label'}.issubset(test_df.columns):
		raise ValueError("Test CSV must contain 'text' and 'label' columns")

	X_test = test_df['text'].astype(str).tolist()
	y_test = test_df['label'].astype(str).tolist()

	y_pred = clf.predict(X_test)
	labels = sorted(list(set(y_test) | set(y_pred)))
	report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

	report_path = os.path.join(eval_dir, 'baseline_test_report.json')
	with open(report_path, 'w', encoding='utf-8') as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	print(f"Saved test report to {report_path}")

	cm_path = os.path.join(eval_dir, 'baseline_test_confusion_matrix.png')
	plot_confusion_matrix(y_test, y_pred, labels, cm_path)
	print(f"Saved test confusion matrix to {cm_path}")


if __name__ == '__main__':
	main()

