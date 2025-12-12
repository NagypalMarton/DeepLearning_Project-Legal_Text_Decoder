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
from utils import setup_logger

logger = setup_logger(__name__)


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

def _plot_overfitting_convergence(losses, accuracies, final_iteration):
	"""Plot Loss and Accuracy curves for overfitting test convergence."""
	base_output = os.getenv('OUTPUT_DIR', '/app/output')
	reports_dir = os.path.join(base_output, 'reports')
	Path(reports_dir).mkdir(parents=True, exist_ok=True)
	
	iterations = range(1, len(losses) + 1)
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	# Loss plot
	ax1.plot(iterations, losses, linewidth=2, color='#e74c3c', alpha=0.8)
	ax1.axhline(y=0.001, color='green', linestyle='--', linewidth=2, label='Target (loss < 0.001)', alpha=0.7)
	ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
	ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
	ax1.set_title('Overfitting Test: Loss Convergence', fontsize=13, fontweight='bold')
	ax1.grid(True, alpha=0.3, linestyle='--')
	ax1.legend(fontsize=11)
	ax1.set_yscale('log')
	
	# Accuracy plot
	ax2.plot(iterations, accuracies, linewidth=2, color='#27ae60', alpha=0.8)
	ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target (acc = 100%)', alpha=0.7)
	ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
	ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
	ax2.set_title('Overfitting Test: Accuracy Convergence', fontsize=13, fontweight='bold')
	ax2.set_ylim([0, 1.1])
	ax2.grid(True, alpha=0.3, linestyle='--')
	ax2.legend(fontsize=11)
	
	fig.tight_layout()
	
	save_path = os.path.join(reports_dir, '03-baseline_overfitting_test_convergence.png')
	fig.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	
	logger.info(f"Saved overfitting test convergence plot to {save_path}")


def overfitting_test(model, tokenizer, device, X_data, y_data_ids, label2id, id2label, max_length=320, max_iterations=1000):
	"""
	Overfitting test: train on a single batch (32 samples) until 100% accuracy and loss < 0.001.
	All regularization disabled (no dropout, no weight decay).
	"""
	logger.info("\n" + "="*80)
	logger.info("OVERFITTING TEST: Training on single batch (32 samples)")
	logger.info("Target: 100% accuracy and loss < 0.001")
	logger.info("="*80)
	
	# Take only first 32 samples
	test_batch_size = min(32, len(X_data))
	X_test = X_data[:test_batch_size]
	y_test = y_data_ids[:test_batch_size]
	
	logger.info(f"Batch size: {test_batch_size}")
	logger.info(f"Unique labels in batch: {set(y_test)}")
	logger.info(f"Label distribution: {[(label, y_test.count(label)) for label in sorted(set(y_test))]}")
	
	# Create dataset
	test_ds = BaselineTransformerDataset(X_test, y_test, tokenizer, max_length=max_length)
	test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=0)
	
	# Disable regularization
	for module in model.modules():
		if isinstance(module, nn.Dropout):
			module.p = 0.0
	
	# Optimizer with NO weight decay
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
	criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
	
	model.train()
	iteration = 0
	best_loss = float('inf')
	patience_counter = 0
	max_patience = 50
	
	# Track metrics for visualization
	history_losses = []
	history_accuracies = []
	
	while iteration < max_iterations:
		iteration += 1
		
		for batch in test_loader:
			optimizer.zero_grad()
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['label'].to(device)
			
			# Forward pass
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			logits = outputs.logits
			loss = criterion(logits, labels)
			
			# Backward pass
			loss.backward()
			optimizer.step()
			
			# Calculate accuracy
			preds = torch.argmax(logits, dim=1)
			accuracy = (preds == labels).sum().item() / len(labels)
			
			# Store metrics for plotting
			history_losses.append(loss.item())
			history_accuracies.append(accuracy)
			
			if iteration % 50 == 0 or iteration == 1:
				logger.info(f"Iteration {iteration:4d} | Loss: {loss.item():.6f} | Accuracy: {accuracy:.4f} ({int(accuracy*100)}%)")
			
			# Check success criteria
			if loss.item() < 0.001 and accuracy == 1.0:
				logger.info("\n" + "="*80)
				logger.info("SUCCESS! Overfitting test passed!")
				logger.info(f"Final Loss: {loss.item():.8f} | Final Accuracy: {accuracy:.4f} (100%)")
				logger.info(f"Converged in {iteration} iterations")
				logger.info("="*80 + "\n")
				
				# Plot convergence curve
				_plot_overfitting_convergence(history_losses, history_accuracies, iteration)
				
				return True
			
			# Check for improvement (if loss not improving, might indicate an issue)
			if loss.item() < best_loss:
				best_loss = loss.item()
				patience_counter = 0
			else:
				patience_counter += 1
	
	# If we reach here, test failed
	logger.error("\n" + "="*80)
	logger.error("FAILURE! Overfitting test did NOT converge to 100% accuracy + loss < 0.001")
	logger.error(f"Reached iteration {iteration} without meeting criteria")
	logger.error(f"Best loss achieved: {best_loss:.6f}")
	logger.error("Possible issues:")
	logger.error("  1. Model is too small/weak for the task")
	logger.error("  2. Data corruption or label mismatch")
	logger.error("  3. Tokenization issues")
	logger.error("  4. Optimizer/learning rate problems")
	logger.error("="*80)
	
	# Detailed debug info
	logger.error("\nDEBUG INFO:")
	logger.error(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	logger.error(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
	logger.error(f"  - Batch size: {test_batch_size}")
	logger.error(f"  - Max sequence length: {max_length}")
	
	# Print first sample details
	logger.error("\nFirst sample tokenization check:")
	first_text = X_test[0]
	first_label = y_test[0]
	logger.error(f"  Text (first 200 chars): {first_text[:200]}")
	logger.error(f"  Label ID: {first_label} -> {id2label.get(first_label, 'UNKNOWN')}")
	
	enc = tokenizer(first_text, add_special_tokens=True, max_length=max_length, 
					padding='max_length', truncation=True, return_tensors='pt')
	logger.error(f"  Tokenized length: {enc['input_ids'].shape}")
	logger.error(f"  Non-padding tokens: {(enc['attention_mask'].sum()).item()}")
	
	raise RuntimeError("OVERFITTING TEST FAILED - Model cannot achieve 100% accuracy on single batch. "
					  "Fix data/model/tokenization before proceeding.")


def main():

	base_output = os.getenv('OUTPUT_DIR', '/app/output')
	processed_dir = os.path.join(base_output, 'processed')
	models_dir, reports_dir = ensure_dirs(base_output)
	logger.info(f"Processed data dir: {processed_dir}")
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
	logger.info(f"Saved baseline label mapping to {mapping_path}")

	# Convert labels to ids
	y_train_ids = [label2id[l] for l in y_train]
	y_val_ids = [label2id.get(l, 0) for l in y_val] if y_val else None
	y_test_ids = [label2id.get(l, 0) for l in y_test] if y_test else None

	# Hyperparameters
	model_name = os.getenv('BASELINE_TRANSFORMER_MODEL', os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc'))
	batch_size = int(os.getenv('BATCH_SIZE', '8'))
	max_length = int(os.getenv('BASELINE_MAX_LENGTH', os.getenv('MAX_LENGTH', '320')))

	logger.info(f"Baseline Transformer: {model_name} | batch_size={batch_size} | max_length={max_length}")

	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info(f"Using device: {device}")


	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))
	model.to(device)

	# Log model parameter counts
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logger.info(f"Model Architecture: {total_params:,} total | {trainable_params:,} trainable parameters")

	# RUN OVERFITTING TEST (ONLY THIS)
	logger.info("\n" + "="*80)
	logger.info("BASELINE MODEL: Overfitting Test Mode")
	logger.info("="*80)
	
	overfitting_test(model, tokenizer, device, X_train, y_train_ids, label2id, id2label, max_length=max_length)
	logger.info("Overfitting test PASSED! Model validated successfully.\n")
	
	# Save baseline transformer model after successful overfitting test
	baseline_model_dir = os.path.join(models_dir, 'baseline_transformer_model')
	Path(baseline_model_dir).mkdir(parents=True, exist_ok=True)
	model.save_pretrained(baseline_model_dir)
	tokenizer.save_pretrained(baseline_model_dir)
	logger.info(f"Saved baseline transformer to {baseline_model_dir}")

	# Datasets / Loaders for evaluation/reporting
	val_loader = None
	test_loader = None
	criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
	
	if X_val and y_val_ids:
		val_ds = BaselineTransformerDataset(X_val, y_val_ids, tokenizer, max_length)
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
	if X_test and y_test_ids:
		test_ds = BaselineTransformerDataset(X_test, y_test_ids, tokenizer, max_length)
		test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

	def evaluate_and_save(split_name, loader, original_labels):
		if loader is None or original_labels is None:
			logger.warning(f"No {split_name} split; skipping")
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
		logger.info(f"Saved {split_name} report to {report_path}")
		logger.info(f"  Accuracy: {report['accuracy']:.4f}, Weighted F1: {report.get('weighted avg', {}).get('f1-score', 0):.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
		cm_path = os.path.join(reports_dir, f'03-baseline_{split_name}_confusion_matrix.png')
		plot_confusion_matrix(trues, preds, all_labels, cm_path, split_name=split_name.capitalize())
		logger.info(f"Saved {split_name} confusion matrix to {cm_path}")
		metrics_plot_path = os.path.join(reports_dir, f'03-baseline_{split_name}_metrics_summary.png')
		plot_metrics_summary(report, metrics_plot_path, split_name=split_name.capitalize())
		logger.info(f"Saved {split_name} metrics summary to {metrics_plot_path}")

		# Additional visualizations: per-class and averages, plus error bars
		plot_classwise_bars(report, reports_dir, split_name=split_name)
		avg_metrics_path = os.path.join(reports_dir, f'03-baseline_{split_name}_avg_metrics.png')
		plot_average_metrics(report, avg_metrics_path, split_name=split_name.capitalize())
		error_metrics_path = os.path.join(reports_dir, f'03-baseline_{split_name}_errors.png')
		plot_error_metrics(mae, rmse, error_metrics_path, split_name=split_name.capitalize())
		logger.info(f"Saved {split_name} classwise/avg/error metric plots to reports")

	evaluate_and_save('val', val_loader, y_val)
	evaluate_and_save('test', test_loader, y_test)

if __name__ == '__main__':
	main()
	