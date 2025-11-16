import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys


def ensure_eval_dir(base_output: str):
	eval_dir = os.path.join(base_output, 'evaluation')
	Path(eval_dir).mkdir(parents=True, exist_ok=True)
	return eval_dir


class TransformerDataset(Dataset):
	"""Dataset for transformer inference."""
	def __init__(self, texts, tokenizer, max_length=384):
		self.texts = texts
		self.tokenizer = tokenizer
		self.max_length = max_length
	
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
		return {
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': encoding['attention_mask'].flatten()
		}


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

	# Load label mapping
	label_map_path = os.path.join(models_dir, 'label_mapping.json')
	if not os.path.exists(label_map_path):
		print(f"Label mapping not found at {label_map_path}.")
		return
	
	with open(label_map_path, 'r', encoding='utf-8') as f:
		label_mapping = json.load(f)
		id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
		label2id = label_mapping['label2id']

	# Setup device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Load model and tokenizer
	print(f"Loading transformer model from {model_path}...")
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForSequenceClassification.from_pretrained(model_path)
	model.to(device)
	model.eval()

	# Load test data
	test_df = pd.read_csv(test_path)
	if not {'text', 'label'}.issubset(test_df.columns):
		raise ValueError("Test CSV must contain 'text' and 'label' columns")

	X_test = test_df['text'].astype(str).tolist()
	y_test = test_df['label'].astype(str).tolist()

	# Create dataset and dataloader
	batch_size = int(os.getenv('BATCH_SIZE', '8'))
	max_length = int(os.getenv('MAX_LENGTH', '384'))
	test_dataset = TransformerDataset(X_test, tokenizer, max_length)
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

	report_path = os.path.join(eval_dir, '05-evaluation_test_report.json')
	with open(report_path, 'w', encoding='utf-8') as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	print(f"Saved test report to {report_path}")

	cm_path = os.path.join(eval_dir, '05-evaluation_test_confusion_matrix.png')
	plot_confusion_matrix(y_test, y_pred, labels, cm_path)
	print(f"Saved test confusion matrix to {cm_path}")


if __name__ == '__main__':
	main()

