import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


class LegalTextDataset(Dataset):
    """Dataset for legal text classification with transformers."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_split_csv(processed_dir: str):
    """Load train/val/test CSVs from processed_dir."""
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    return train_df, val_df, test_df


def encode_labels(labels):
    """Convert string labels to numeric indices."""
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    encoded = [label2id[label] for label in labels]
    
    return encoded, label2id, id2label


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels


def plot_training_history(history, save_path):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    # Configuration from environment variables
    base_output = os.getenv('OUTPUT_DIR', '/app/output')
    processed_dir = os.path.join(base_output, 'processed')
    models_dir = os.path.join(base_output, 'models')
    reports_dir = os.path.join(base_output, 'reports')
    
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')  # Hungarian BERT
    batch_size = int(os.getenv('BATCH_SIZE', '8'))
    epochs = int(os.getenv('EPOCHS', '3'))
    learning_rate = float(os.getenv('LEARNING_RATE', '2e-5'))
    max_length = int(os.getenv('MAX_LENGTH', '512'))
    
    print(f"Loading data from {processed_dir}")
    train_df, val_df, test_df = load_split_csv(processed_dir)
    
    # Prepare labels
    y_train_str = train_df['label'].astype(str).tolist()
    y_train, label2id, id2label = encode_labels(y_train_str)
    
    # Save label mapping
    label_map_path = os.path.join(models_dir, 'label_mapping.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, ensure_ascii=False, indent=2)
    
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")
    print(f"Label mapping: {label2id}")
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading transformer model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Prepare datasets
    X_train = train_df['text'].astype(str).tolist()
    train_dataset = LegalTextDataset(X_train, y_train, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_df is not None:
        y_val_str = val_df['label'].astype(str).tolist()
        y_val = [label2id[label] for label in y_val_str]
        X_val = val_df['text'].astype(str).tolist()
        val_dataset = LegalTextDataset(X_val, y_val, tokenizer, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    # Save model
    model_path = os.path.join(models_dir, 'transformer_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    history_plot_path = os.path.join(reports_dir, 'transformer_training_history.png')
    plot_training_history(history, history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    
    # Evaluate on test set if available
    if test_df is not None:
        y_test_str = test_df['label'].astype(str).tolist()
        y_test = [label2id[label] for label in y_test_str]
        X_test = test_df['text'].astype(str).tolist()
        test_dataset = LegalTextDataset(X_test, y_test, tokenizer, max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, device)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        # Generate classification report
        pred_labels_str = [id2label[str(pred)] for pred in predictions]
        true_labels_str = [id2label[str(label)] for label in true_labels]
        
        report = classification_report(
            true_labels_str,
            pred_labels_str,
            output_dict=True,
            zero_division=0
        )
        
        report_path = os.path.join(reports_dir, 'transformer_test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Test report saved to {report_path}")


if __name__ == '__main__':
    main()
