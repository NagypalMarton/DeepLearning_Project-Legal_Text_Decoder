import os
import json
from pathlib import Path
import re

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

# Import helper functions from incremental development script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from importlib import import_module
    inc_dev = import_module('04_incremental_model_development')
    FusionModel = inc_dev.FusionModel
    coral_predict = inc_dev.coral_predict
    extract_readability_features = inc_dev.extract_readability_features
    normalize_label = inc_dev.normalize_label
except Exception as e:
    print(f"Warning: Could not import from 04_incremental_model_development: {e}")
    FusionModel = None
    coral_predict = None
    extract_readability_features = None
    normalize_label = None


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

	model_path = os.path.join(models_dir, 'best_transformer_model')
	test_path = os.path.join(processed_dir, 'test.csv')

	if not os.path.isdir(model_path):
		print(f"Transformer model not found at {model_path}. Train the transformer first (04_incremental_model_development.py).")
		return

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

	# Load model and tokenizer
	print(f"Loading transformer model from {model_path}...")
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	
	# Check if this is a FusionModel checkpoint
	checkpoint_path = os.path.join(model_path, 'pytorch_model.bin')
	use_fusion = False
	use_coral = False
	stats = None
	
	if os.path.exists(checkpoint_path) and FusionModel is not None:
		try:
			checkpoint = torch.load(checkpoint_path, map_location=device)
			if isinstance(checkpoint, dict) and 'use_feature_fusion' in checkpoint:
				use_fusion = checkpoint.get('use_feature_fusion', False)
				use_coral = checkpoint.get('use_coral', False)
				num_labels = checkpoint.get('num_labels', len(id2label))
				
				if use_fusion:
					print(f"Loading FusionModel (CORAL={use_coral})...")
					base_model = AutoModel.from_pretrained(model_path.replace('/best_transformer_model', '/baseline_transformer_model') if 'baseline' not in model_path else model_path)
					if base_model is None:
						model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')
						base_model = AutoModel.from_pretrained(model_name)
					model = FusionModel(base_model, num_classes=num_labels, use_coral=use_coral)
					model.load_state_dict(checkpoint['model_state_dict'])
					model.to(device)
					
					# Load feature stats
					metadata_path = os.path.join(model_path, 'metadata.json')
					if os.path.exists(metadata_path):
						with open(metadata_path, 'r') as f:
							metadata = json.load(f)
					else:
						print("Warning: metadata.json not found, features may not be normalized correctly")
				else:
					raise ValueError("Checkpoint indicates fusion model but FusionModel class not available")
		except Exception as e:
			print(f"Could not load as FusionModel: {e}, falling back to standard model")
			use_fusion = False
	
	if not use_fusion:
		model = AutoModelForSequenceClassification.from_pretrained(model_path)
		model.to(device)
	
	model.eval()

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
			readability_features = batch.get('readability_features')
			if readability_features is not None:
				readability_features = readability_features.to(device)
			
			if use_fusion:
				outputs = model(input_ids=input_ids, attention_mask=attention_mask, readability_features=readability_features)
			else:
				outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			
			logits = outputs.logits
			
			if use_coral and coral_predict is not None:
				preds = coral_predict(logits)
			else:
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


if __name__ == '__main__':
	main()

