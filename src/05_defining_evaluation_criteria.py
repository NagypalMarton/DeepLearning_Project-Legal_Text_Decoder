import os
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import textstat

# Suppress UserWarning about newly initialized classifier weights
warnings.filterwarnings('ignore', message='.*Some weights of.*were not initialized from the model checkpoint.*')
warnings.filterwarnings('ignore', message='.*You should probably TRAIN this model on a down-stream task.*')

# Import helper functions from incremental development script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from importlib import import_module
    inc_dev = import_module('04_incremental_model_development')
    normalize_label = inc_dev.normalize_label
except Exception as e:
    print(f"Warning: Could not import normalize_label from 04_incremental_model_development: {e}")
    normalize_label = None


def find_best_model(models_dir: str):
	"""Resolve the best model checkpoint from outputs of 04_incremental_model_development.

	Priority:
	1) models/best_overall.json → {"best_checkpoint": "/path/to/best.pt"}
	2) models/best_model.pt
	3) first matching models/best_*.pt (prefer Final_Balanced if present, else latest)
	"""
	models_path = Path(models_dir)
	# 1) best_overall.json
	best_overall = models_path / 'best_overall.json'
	if best_overall.exists():
		try:
			with open(best_overall, 'r', encoding='utf-8') as f:
				data = json.load(f)
			best_checkpoint = data.get('best_checkpoint')
			if best_checkpoint and Path(best_checkpoint).exists():
				return str(best_checkpoint)
		except Exception:
			pass

	# 2) generic best_model.pt
	generic_best = models_path / 'best_model.pt'
	if generic_best.exists():
		return str(generic_best)

	# 3) any best_*.pt, prefer Final_Balanced
	best_models = list(models_path.glob('best_*.pt'))
	if not best_models:
		raise FileNotFoundError(f"No best_*.pt checkpoints found in {models_dir}")
	final_balanced = [m for m in best_models if 'Final_Balanced' in m.name or 'final_balanced' in m.name.lower()]
	return str(final_balanced[0] if final_balanced else best_models[-1])


def ensure_eval_dir(base_output: str):
	reports_dir = os.path.join(base_output, 'reports')
	Path(reports_dir).mkdir(parents=True, exist_ok=True)
	return reports_dir


class TransformerDataset(Dataset):
	"""Dataset for transformer inference with optional readability features."""
	def __init__(self, texts, tokenizer, max_length=384, use_features=False, stats=None):
		self.texts = texts
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.use_features = use_features
		self._precomputed_features = None
		
		if self.use_features and extract_readability_features is not None:
			raw_feats = [extract_readability_features(str(t)).numpy() for t in self.texts]
			if stats:
				mean = np.array(stats['mean'])
				std = np.array(stats['std'])
				norm_feats = (np.stack(raw_feats) - mean) / std
				self._precomputed_features = torch.tensor(norm_feats, dtype=torch.float32)
			else:
				self._precomputed_features = torch.stack([extract_readability_features(str(t)) for t in self.texts])
	
	def __len__(self):
		return len(self.texts)
	
	def __getitem__(self, idx):
		text = str(self.texts[idx])
		encoding = self.tokenizer(
			text,
			add_special_tokens=True,
			max_length=self.max_length,
			padding='max_length',
			truncation=True,
			return_tensors='pt'
		)
		item = {
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': encoding['attention_mask'].flatten()
		}
		if self.use_features and self._precomputed_features is not None:
			item['readability_features'] = self._precomputed_features[idx]
		return item


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
		   xticklabels=labels, yticklabels=labels,
		   ylabel='True label', xlabel='Predicted label',
		   title='Confusion Matrix (Test) - Transformer')
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

	test_path = os.path.join(processed_dir, 'test.csv')

	if not os.path.exists(test_path):
		print(f"Test CSV not found at {test_path}. Run data processing first (02_data_cleansing_and_preparation.py).")
		return

	# Load label mapping (supports baseline or generic file) and ensure full labels
	label_map_path_generic = os.path.join(models_dir, 'label_mapping.json')
	label_map_path_baseline = os.path.join(models_dir, 'baseline_label_mapping.json')
	label_mapping = None
	if os.path.exists(label_map_path_generic):
		with open(label_map_path_generic, 'r', encoding='utf-8') as f:
			label_mapping = json.load(f)
	elif os.path.exists(label_map_path_baseline):
		with open(label_map_path_baseline, 'r', encoding='utf-8') as f:
			label_mapping = json.load(f)
	else:
		print(f"No label mapping file found (looked for {label_map_path_generic} and baseline).")
		return

	id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
	label2id = label_mapping['label2id']

	def _expand_numeric_labels(id2label_local, sample_labels):
		# Build mapping from leading digit to full Hungarian label string
		num_to_full = {}
		for lbl in sample_labels:
			m = re.match(r'^([1-5])', str(lbl).strip())
			if m:
				n = m.group(1)
				if n not in num_to_full:
					num_to_full[n] = lbl
		# Replace plain digit labels with full form if available
		for k in list(id2label_local.keys()):
			v = id2label_local[k]
			if re.fullmatch(r'[1-5]', str(v)) and v in num_to_full:
				id2label_local[k] = num_to_full[v]
		return id2label_local


	# Setup device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Find and load best model checkpoint
	models_dir = os.path.join(base_output, 'models')
	best_checkpoint_path = find_best_model(models_dir)
	print(f"Loading best model checkpoint from {best_checkpoint_path}...")
	
	# Load checkpoint
	checkpoint = torch.load(best_checkpoint_path, map_location=device)
	label2id = checkpoint['label2id']
	id2label = {int(k): v for k, v in checkpoint['id2label'].items()}

	# Instantiate the exact trained architecture from 04_incremental_model_development
	transformer_model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')
	base_transformer = AutoModel.from_pretrained(transformer_model_name)

	from importlib import import_module
	inc = import_module('04_incremental_model_development')
	fname = Path(best_checkpoint_path).name.lower()
	if 'final_balanced' in fname:
		model = inc.BalancedFinalModel(base_transformer, num_classes=len(id2label))
	elif 'step3_advanced' in fname:
		model = inc.Step3_AdvancedModel(base_transformer, num_classes=len(id2label))
	elif 'step2_extended' in fname:
		model = inc.Step2_ExtendedModel(base_transformer, num_classes=len(id2label))
	else:
		model = inc.Step1_BaselineModel(base_transformer, num_classes=len(id2label))

	model.load_state_dict(checkpoint['model_state_dict'], strict=False)
	model.to(device)
	model.eval()

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

	# Load test data
	test_df = pd.read_csv(test_path)
	if not {'text', 'label'}.issubset(test_df.columns):
		raise ValueError("Test CSV must contain 'text' and 'label' columns")

	X_test = test_df['text'].astype(str).tolist()
	y_test = test_df['label'].astype(str).tolist()

	# Expand numeric labels to full Hungarian forms if needed
	if '_expand_numeric_labels' in locals():
		id2label = _expand_numeric_labels(id2label, y_test)

	# Create dataset and dataloader
	batch_size = int(os.getenv('BATCH_SIZE', '8'))
	max_length = int(os.getenv('MAX_LENGTH', '384'))
	# Evaluation uses transformer-only path; disable fusion features by default
	use_fusion = False
	stats = None
	test_dataset = TransformerDataset(X_test, tokenizer, max_length, use_features=use_fusion, stats=stats)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	# Run inference
	print("Running evaluation on test set...")
	y_pred = []
	disable_tqdm = not sys.stdout.isatty()
	
	with torch.no_grad():
		for batch in tqdm(test_loader, desc="Evaluating", disable=disable_tqdm):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			logits = outputs.logits
			preds = torch.argmax(logits, dim=1)
			
			y_pred.extend([id2label[int(p)] for p in preds.cpu().numpy()])

	# Generate classification report
	labels = sorted(list(set(y_test) | set(y_pred)))
	report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
	
	# Add ordinal regression metrics (MAE, RMSE)
	def labels_to_numeric(labels):
		out = []
		for l in labels:
			m = str(l).strip()
			if m and m[0].isdigit():
				out.append(int(m[0]))
			else:
				out.append(0)
		return np.array(out)
	
	y_true_num = labels_to_numeric(y_test)
	y_pred_num = labels_to_numeric(y_pred)
	mae = mean_absolute_error(y_true_num, y_pred_num)
	rmse = np.sqrt(mean_squared_error(y_true_num, y_pred_num))
	
	report['mae'] = float(mae)
	report['rmse'] = float(rmse)
	weighted_f1 = report.get('weighted avg', {}).get('f1-score', 0)
	
	print(f"Test Accuracy: {report['accuracy']:.4f}, Weighted F1: {weighted_f1:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

	report_path = os.path.join(eval_dir, '05-evaluation_test_report.json')
	with open(report_path, 'w', encoding='utf-8') as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	print(f"Saved test report to {report_path}")

	cm_path = os.path.join(eval_dir, '05-evaluation_test_confusion_matrix.png')
	plot_confusion_matrix(y_test, y_pred, labels, cm_path)
	print(f"Saved test confusion matrix to {cm_path}")

	# Additional visualizations
	def label_to_numeric(labels):
		out = []
		for l in labels:
			m = str(l).strip()
			if m and m[0].isdigit():
				out.append(int(m[0]))
			else:
				out.append(0)
		return np.array(out)

	def plot_metrics_summary(report, save_path, split_name='Test'):
		metrics = {
			'Accuracy': report.get('accuracy', 0),
			'Weighted F1': report.get('weighted avg', {}).get('f1-score', 0),
			'MAE': report.get('mae', 0),
			'RMSE': report.get('rmse', 0)
		}
		fig, ax = plt.subplots(figsize=(10, 6))
		colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
		bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
		ax.set_ylabel('Score / Error', fontsize=12)
		ax.set_title(f'Test Metrics Summary ({split_name})', fontsize=14, fontweight='bold')
		ax.set_ylim([0, max(max(metrics.values()) * 1.2, 1.0)])
		ax.grid(axis='y', alpha=0.3, linestyle='--')
		for bar, (name, value) in zip(bars, metrics.items()):
			h = bar.get_height()
			ax.text(bar.get_x() + bar.get_width()/2., h, f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
		fig.tight_layout()
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		plt.close(fig)

	def plot_classwise_bars(report, save_dir, split_name='Test'):
		reserved = {"accuracy", "macro avg", "weighted avg"}
		class_keys = [k for k in report.keys() if k not in reserved]
		class_keys = [k for k in class_keys if isinstance(report[k], dict) and 'precision' in report[k]]
		if not class_keys:
			return
		class_keys_sorted = class_keys

		def _plot_metric(metric_name, filename, color):
			values = [report[c].get(metric_name, 0) for c in class_keys_sorted]
			fig, ax = plt.subplots(figsize=(max(8, len(values)*0.9), 5))
			ax.bar(class_keys_sorted, values, color=color, alpha=0.85)
			ax.set_title(f'{metric_name.title()} by Class ({split_name})', fontsize=14, fontweight='bold')
			ax.set_ylabel(metric_name.title())
			ax.set_ylim([0, 1])
			plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
			for i, v in enumerate(values):
				ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
			fig.tight_layout()
			fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
			plt.close(fig)

		_plot_metric('precision', f'05-evaluation_test_class_precision.png', '#5dade2')
		_plot_metric('recall', f'05-evaluation_test_class_recall.png', '#58d68d')
		_plot_metric('f1-score', f'05-evaluation_test_class_f1.png', '#f4d03f')

		supports = [report[c].get('support', 0) for c in class_keys_sorted]
		fig, ax = plt.subplots(figsize=(max(8, len(supports)*0.9), 5))
		ax.bar(class_keys_sorted, supports, color='#a569bd', alpha=0.85)
		ax.set_title(f'Support by Class ({split_name})', fontsize=14, fontweight='bold')
		ax.set_ylabel('Support (count)')
		plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
		for i, v in enumerate(supports):
			ax.text(i, v, f"{int(v)}", ha='center', va='bottom', fontsize=9)
		fig.tight_layout()
		fig.savefig(os.path.join(save_dir, f'05-evaluation_test_class_support.png'), dpi=150, bbox_inches='tight')
		plt.close(fig)

	def plot_average_metrics(report, save_path, split_name='Test'):
		metrics = ['precision', 'recall', 'f1-score']
		macro = [report.get('macro avg', {}).get(m, 0) for m in metrics]
		weighted = [report.get('weighted avg', {}).get(m, 0) for m in metrics]
		x = np.arange(len(metrics))
		width = 0.35
		fig, ax = plt.subplots(figsize=(8, 5))
		ax.bar(x - width/2, macro, width, label='Macro', color='#7fb3d5')
		ax.bar(x + width/2, weighted, width, label='Weighted', color='#76d7c4')
		ax.set_xticks(x)
		ax.set_xticklabels([m.title() for m in metrics])
		ax.set_ylim([0, 1])
		ax.set_ylabel('Score')
		ax.set_title(f'Average Metrics ({split_name})', fontsize=14, fontweight='bold')
		ax.legend()
		for i, v in enumerate(macro):
			ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
		for i, v in enumerate(weighted):
			ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
		fig.tight_layout()
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		plt.close(fig)

	def plot_error_metrics(mae, rmse, save_path, split_name='Test'):
		labels_err = ['MAE', 'RMSE']
		values = [mae, rmse]
		fig, ax = plt.subplots(figsize=(6, 5))
		colors = ['#e67e22', '#c0392b']
		ax.bar(labels_err, values, color=colors, alpha=0.85)
		ax.set_title(f'Error Metrics ({split_name})', fontsize=14, fontweight='bold')
		ax.set_ylabel('Error')
		for i, v in enumerate(values):
			ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
		fig.tight_layout()
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		plt.close(fig)

	# Generate all plots
	metrics_plot_path = os.path.join(eval_dir, '05-evaluation_test_metrics_summary.png')
	plot_metrics_summary(report, metrics_plot_path, split_name='Test')
	print(f"Saved test metrics summary to {metrics_plot_path}")

	plot_classwise_bars(report, eval_dir, split_name='Test')
	print(f"Saved test classwise metric plots to {eval_dir}")

	avg_metrics_path = os.path.join(eval_dir, '05-evaluation_test_avg_metrics.png')
	plot_average_metrics(report, avg_metrics_path, split_name='Test')
	print(f"Saved test avg metrics to {avg_metrics_path}")

	error_metrics_path = os.path.join(eval_dir, '05-evaluation_test_errors.png')
	plot_error_metrics(mae, rmse, error_metrics_path, split_name='Test')
	print(f"Saved test error metrics to {error_metrics_path}")


if __name__ == '__main__':
	main()

