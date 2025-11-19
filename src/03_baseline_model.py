import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")  # headless environments (Docker)
import matplotlib.pyplot as plt
import math
from tqdm.auto import tqdm
import sys


def load_split_csv(processed_dir: str):
	"""Load train/val/test CSVs from processed_dir. Returns (train, val, test)."""
	train_path = os.path.join(processed_dir, "train.csv")
	val_path = os.path.join(processed_dir, "val.csv")
	test_path = os.path.join(processed_dir, "test.csv")

	if not os.path.exists(train_path):
		single_path = os.path.join(processed_dir, "processed_data.csv")
		if os.path.exists(single_path):
			df = pd.read_csv(single_path)
			return df, None, None
		raise FileNotFoundError(f"Missing training data in {processed_dir}")

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


def plot_confusion_matrix(y_true, y_pred, labels, save_path, split_name='Test'):
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=labels, yticklabels=labels,
		   ylabel='True label', xlabel='Predicted label',
		   title=f'Confusion Matrix ({split_name})')
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	thresh = cm.max() / 2.0 if cm.size else 0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	fig.savefig(save_path, bbox_inches='tight')
	plt.close(fig)


def label_to_numeric(labels):
	numeric = []
	for label in labels:
		s = str(label).strip()
		if s and s[0].isdigit():
			numeric.append(int(s[0]))
		else:
			try:
				numeric.append(int(s))
			except ValueError:
				numeric.append(0)
	return np.array(numeric)


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
	ax.set_title(f'Baseline Transformer Metrics Summary ({split_name})', fontsize=14, fontweight='bold')
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
	# Filter only dict-like entries (skip floats like accuracy)
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

	# Precision / Recall / F1 per class
	_plot_metric('precision', f'03-baseline_{split_name}_class_precision.png', '#5dade2')
	_plot_metric('recall', f'03-baseline_{split_name}_class_recall.png', '#58d68d')
	_plot_metric('f1-score', f'03-baseline_{split_name}_class_f1.png', '#f4d03f')

	# Support per class
	supports = [report[c].get('support', 0) for c in class_keys_sorted]
	fig, ax = plt.subplots(figsize=(max(8, len(supports)*0.9), 5))
	ax.bar(class_keys_sorted, supports, color='#a569bd', alpha=0.85)
	ax.set_title(f'Support by Class ({split_name})', fontsize=14, fontweight='bold')
	ax.set_ylabel('Support (count)')
	plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
	for i, v in enumerate(supports):
		ax.text(i, v, f"{int(v)}", ha='center', va='bottom', fontsize=9)
	fig.tight_layout()
	fig.savefig(os.path.join(save_dir, f'03-baseline_{split_name}_class_support.png'), dpi=150, bbox_inches='tight')
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
	labels = ['MAE', 'RMSE']
	values = [mae, rmse]
	fig, ax = plt.subplots(figsize=(6, 5))
	colors = ['#e67e22', '#c0392b']
	ax.bar(labels, values, color=colors, alpha=0.85)
	ax.set_title(f'Error Metrics ({split_name})', fontsize=14, fontweight='bold')
	ax.set_ylabel('Error')
	for i, v in enumerate(values):
		ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
	fig.tight_layout()
	fig.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close(fig)


class BaselineTransformerDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_length=320):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		text = str(self.texts[idx])
		label = int(self.labels[idx])
		enc = self.tokenizer(text,
							 add_special_tokens=True,
							 max_length=self.max_length,
							 padding='max_length',
							 truncation=True,
							 return_tensors='pt')
		return {
			'input_ids': enc['input_ids'].squeeze(0),
			'attention_mask': enc['attention_mask'].squeeze(0),
			'label': torch.tensor(label, dtype=torch.long)
		}


def create_label_mapping(labels):
	unique = sorted(list(set(labels)), key=lambda x: int(str(x)[0]) if str(x)[0].isdigit() else x)
	label2id = {lab: idx for idx, lab in enumerate(unique)}
	id2label = {idx: lab for lab, idx in label2id.items()}
	return label2id, id2label


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	disable_tqdm = not sys.stdout.isatty()
	for batch in tqdm(dataloader, desc="Train", disable=disable_tqdm):
		optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['label'].to(device)
		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		logits = outputs.logits
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()
		if scheduler:
			scheduler.step()
		running_loss += loss.item() * input_ids.size(0)
		preds = torch.argmax(logits, dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)
	return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, dataloader, device, criterion, id2label):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	all_preds = []
	all_trues = []
	disable_tqdm = not sys.stdout.isatty()
	for batch in tqdm(dataloader, desc="Eval", disable=disable_tqdm):
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['label'].to(device)
		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		logits = outputs.logits
		loss = criterion(logits, labels)
		running_loss += loss.item() * input_ids.size(0)
		preds = torch.argmax(logits, dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)
		all_preds.extend([id2label[int(p)] for p in preds.cpu().numpy()])
		all_trues.extend([id2label[int(t)] for t in labels.cpu().numpy()])
	avg_loss = running_loss / max(1, total)
	acc = correct / max(1, total)
	return avg_loss, acc, all_preds, all_trues

def main():
	base_output = os.getenv('OUTPUT_DIR', '/app/output')
	processed_dir = os.path.join(base_output, 'processed')
	models_dir, reports_dir = ensure_dirs(base_output)
	print(f"Processed data dir: {processed_dir}")
	train_df, val_df, test_df = load_split_csv(processed_dir)
	if 'text' not in train_df.columns or 'label' not in train_df.columns:
		raise ValueError("Train CSV must contain 'text' and 'label'")

	# Extract splits
	X_train = train_df['text'].astype(str).tolist()
	y_train = train_df['label'].astype(str).tolist()
	X_val = val_df['text'].astype(str).tolist() if val_df is not None else None
	y_val = val_df['label'].astype(str).tolist() if val_df is not None else None
	X_test = test_df['text'].astype(str).tolist() if test_df is not None else None
	y_test = test_df['label'].astype(str).tolist() if test_df is not None else None

	# Label mapping
	label2id, id2label = create_label_mapping(y_train)
	mapping_path = os.path.join(models_dir, 'baseline_label_mapping.json')
	with open(mapping_path, 'w', encoding='utf-8') as f:
		json.dump({'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}, f, ensure_ascii=False, indent=2)
	print(f"Saved baseline label mapping to {mapping_path}")

	# Convert labels to ids
	y_train_ids = [label2id[l] for l in y_train]
	y_val_ids = [label2id.get(l, 0) for l in y_val] if y_val else None
	y_test_ids = [label2id.get(l, 0) for l in y_test] if y_test else None

	# Hyperparameters
	model_name = os.getenv('BASELINE_TRANSFORMER_MODEL', os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc'))
	epochs = int(os.getenv('BASELINE_EPOCHS', '3'))
	batch_size = int(os.getenv('BATCH_SIZE', '8'))
	lr = float(os.getenv('BASELINE_LR', os.getenv('LEARNING_RATE', '2e-5')))
	max_length = int(os.getenv('BASELINE_MAX_LENGTH', os.getenv('MAX_LENGTH', '320')))
	weight_decay = float(os.getenv('BASELINE_WEIGHT_DECAY', os.getenv('WEIGHT_DECAY', '0.01')))
	label_smoothing = float(os.getenv('BASELINE_LABEL_SMOOTHING', os.getenv('LABEL_SMOOTHING', '0.15')))

	print(f"Baseline Transformer: {model_name} | epochs={epochs} | batch_size={batch_size} | lr={lr} | max_length={max_length}")

	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))
	model.to(device)

	# Datasets / Loaders
	train_ds = BaselineTransformerDataset(X_train, y_train_ids, tokenizer, max_length)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = None
	test_loader = None
	if X_val and y_val_ids:
		val_ds = BaselineTransformerDataset(X_val, y_val_ids, tokenizer, max_length)
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
	if X_test and y_test_ids:
		test_ds = BaselineTransformerDataset(X_test, y_test_ids, tokenizer, max_length)
		test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

	# Optimizer & Scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	total_steps = epochs * math.ceil(len(train_loader))
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
	criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

	history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

	for epoch in range(epochs):
		print(f"\nEpoch {epoch+1}/{epochs}")
		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
		history['train_loss'].append(train_loss)
		history['train_acc'].append(train_acc)
		print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
		if val_loader:
			val_loss, val_acc, val_preds, val_trues = evaluate(model, val_loader, device, criterion, id2label)
			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)
			print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

	# Plot training curves
	try:
		fig, ax = plt.subplots(1, 2, figsize=(12, 4))
		ax[0].plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', marker='o')
		if history['val_loss']:
			ax[0].plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', marker='o')
		ax[0].set_title('Loss per Epoch')
		ax[0].set_xlabel('Epoch')
		ax[0].set_ylabel('Loss')
		ax[0].grid(alpha=0.3, linestyle='--')
		ax[0].legend()

		ax[1].plot(range(1, len(history['train_acc'])+1), history['train_acc'], label='Train Acc', marker='o')
		if history['val_acc']:
			ax[1].plot(range(1, len(history['val_acc'])+1), history['val_acc'], label='Val Acc', marker='o')
		ax[1].set_title('Accuracy per Epoch')
		ax[1].set_xlabel('Epoch')
		ax[1].set_ylabel('Accuracy')
		ax[1].set_ylim([0, 1])
		ax[1].grid(alpha=0.3, linestyle='--')
		ax[1].legend()

		fig.tight_layout()
		training_plot_path = os.path.join(reports_dir, '03-baseline_training_curves.png')
		fig.savefig(training_plot_path, dpi=150, bbox_inches='tight')
		plt.close(fig)
		print(f"Saved training curves to {training_plot_path}")
	except Exception as e:
		print(f"Warning: failed to plot training curves: {e}")

	# Save baseline transformer model
	baseline_model_dir = os.path.join(models_dir, 'baseline_transformer_model')
	Path(baseline_model_dir).mkdir(parents=True, exist_ok=True)
	model.save_pretrained(baseline_model_dir)
	tokenizer.save_pretrained(baseline_model_dir)
	print(f"Saved baseline transformer to {baseline_model_dir}")

	def evaluate_and_save(split_name, loader, original_labels):
		if loader is None or original_labels is None:
			print(f"No {split_name} split; skipping")
			return
		loss, acc, preds, trues = evaluate(model, loader, device, criterion, id2label)
		all_labels = sorted(list(set(trues) | set(preds)))
		report = classification_report(trues, preds, labels=all_labels, output_dict=True, zero_division=0)
		# Regression metrics
		y_true_num = label_to_numeric(trues)
		y_pred_num = label_to_numeric(preds)
		mae = mean_absolute_error(y_true_num, y_pred_num)
		rmse = np.sqrt(mean_squared_error(y_true_num, y_pred_num))
		report['mae'] = float(mae)
		report['rmse'] = float(rmse)
		report_path = os.path.join(reports_dir, f'03-baseline_{split_name}_report.json')
		with open(report_path, 'w', encoding='utf-8') as f:
			json.dump(report, f, ensure_ascii=False, indent=2)
		print(f"Saved {split_name} report to {report_path}")
		print(f"  Accuracy: {report['accuracy']:.4f}, Weighted F1: {report.get('weighted avg', {}).get('f1-score', 0):.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
		cm_path = os.path.join(reports_dir, f'03-baseline_{split_name}_confusion_matrix.png')
		plot_confusion_matrix(trues, preds, all_labels, cm_path, split_name=split_name.capitalize())
		print(f"Saved {split_name} confusion matrix to {cm_path}")
		metrics_plot_path = os.path.join(reports_dir, f'03-baseline_{split_name}_metrics_summary.png')
		plot_metrics_summary(report, metrics_plot_path, split_name=split_name.capitalize())
		print(f"Saved {split_name} metrics summary to {metrics_plot_path}")

		# Additional visualizations: per-class and averages, plus error bars
		plot_classwise_bars(report, reports_dir, split_name=split_name)
		avg_metrics_path = os.path.join(reports_dir, f'03-baseline_{split_name}_avg_metrics.png')
		plot_average_metrics(report, avg_metrics_path, split_name=split_name.capitalize())
		error_metrics_path = os.path.join(reports_dir, f'03-baseline_{split_name}_errors.png')
		plot_error_metrics(mae, rmse, error_metrics_path, split_name=split_name.capitalize())
		print(f"Saved {split_name} classwise/avg/error metric plots to reports")

	evaluate_and_save('val', val_loader, y_val)
	evaluate_and_save('test', test_loader, y_test)

if __name__ == '__main__':
	main()
	