import os
import json
from pathlib import Path
import random

import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import torch.nn.functional as F
import re
from datetime import datetime


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LegalTextDataset(Dataset):
    """Dataset for legal text classification."""

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


def normalize_label(raw):
    s = str(raw).strip()
    m = re.match(r'^([1-5])', s)
    return int(m.group(1)) if m else 0


def build_ordinal_mapping(labels):
    numeric = [normalize_label(l) for l in labels]
    unique = sorted(set(numeric))
    label2id = {u: i for i, u in enumerate(unique)}
    # Fix: enumerate was missing; map each id back to string form
    id2label = {label2id[u]: str(u) for u in unique}
    encoded = [label2id[n] for n in numeric]
    return encoded, label2id, id2label





def train_epoch(model, dataloader, optimizer, scheduler, device, criterion=None, grad_acc_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    disable_tqdm = not sys.stdout.isatty()
    progress_bar = tqdm(dataloader, desc="Training", disable=disable_tqdm)
    scaler = amp.GradScaler('cuda', enabled=device.type == 'cuda')
    step_count = 0
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        with amp.autocast('cuda', enabled=scaler.is_enabled()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            if criterion is None:
                outputs_with_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs_with_labels.loss / grad_acc_steps
            else:
                loss = criterion(logits, labels) / grad_acc_steps
        
        scaler.scale(loss).backward()
        step_count += 1
        
        if step_count % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        total_loss += loss.item() * grad_acc_steps
        preds = torch.argmax(logits.detach(), dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': (total_loss / step_count)})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device, criterion=None):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        disable_tqdm = not sys.stdout.isatty()
        progress_bar = tqdm(dataloader, desc="Evaluating", disable=disable_tqdm)
        mixed = device.type == 'cuda'
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            with amp.autocast('cuda', enabled=mixed):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                if criterion is None:
                    outputs_with_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs_with_labels.loss
                else:
                    loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits.detach(), dim=1)
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
    # Set random seed for reproducibility
    seed = int(os.getenv('RANDOM_SEED', '42'))
    set_seed(seed)
    print(f"Random seed set to {seed} for reproducibility")
    
    # Configuration from environment variables
    base_output = os.getenv('OUTPUT_DIR', '/app/output')
    processed_dir = os.path.join(base_output, 'processed')
    models_dir = os.path.join(base_output, 'models')
    reports_dir = os.path.join(base_output, 'reports')
    
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')
    batch_size = int(os.getenv('BATCH_SIZE', '8'))
    epochs = int(os.getenv('EPOCHS', '15'))
    learning_rate = float(os.getenv('LEARNING_RATE', '1.5e-5'))
    weight_decay = float(os.getenv('WEIGHT_DECAY', '0.01'))
    max_length = int(os.getenv('MAX_LENGTH', '320'))
    label_smoothing = float(os.getenv('LABEL_SMOOTHING', '0.02'))
    early_stopping_patience = int(os.getenv('EARLY_STOPPING_PATIENCE', '3'))
    num_workers = int(os.getenv('NUM_WORKERS', '2'))
    grad_acc_steps = int(os.getenv('GRAD_ACC_STEPS', '2'))
    
    print(f"Loading data from {processed_dir}")
    train_df, val_df, test_df = load_split_csv(processed_dir)
    
    # Prepare labels
    y_train_str = train_df['label'].astype(str).tolist()
    y_train, label2id, id2label = build_ordinal_mapping(y_train_str)
    
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
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} | Total VRAM: {total_mem:.2f} GB")
    
    # Load tokenizer and model
    baseline_dir = os.path.join(models_dir, 'baseline_transformer_model')
    if os.path.isdir(baseline_dir):
        print(f"Loading baseline checkpoint from {baseline_dir} for fine-tuning")
        try:
            tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                baseline_dir,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
        except Exception as e:
            print(f"Warning: failed to load baseline model, falling back to {model_name}: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
    else:
        print(f"Loading transformer model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
    model.to(device)

    # Compute class weights for handling class imbalance
    class_weights_raw = compute_class_weight('balanced', classes=np.unique(y_train), y=np.array(y_train))
    class_weights = np.sqrt(class_weights_raw)  # sqrt scaling to reduce aggressiveness
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights (sqrt-scaled): {class_weights}")
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
    print(f"Using CrossEntropy with class weights (label_smoothing={label_smoothing})")
    
    # Prepare datasets
    X_train = train_df['text'].astype(str).tolist()
    train_dataset = LegalTextDataset(X_train, y_train, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type=='cuda',
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = None
    if val_df is not None:
        y_val_str = val_df['label'].astype(str).tolist()
        y_val_numeric = [normalize_label(label) for label in y_val_str]
        y_val = [label2id[n] for n in y_val_numeric]
        X_val = val_df['text'].astype(str).tolist()
        val_dataset = LegalTextDataset(X_val, y_val, tokenizer, max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type=='cuda',
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    effective_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_acc_steps))
    total_steps = effective_steps_per_epoch * epochs
    warmup_steps = int(0.15 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print('Using linear scheduler with warmup')
    
    # Log key hyperparameters
    run_metadata = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'model_name': model_name,
        'device': str(device),
        'epochs': epochs,
        'batch_size': batch_size,
        'grad_acc_steps': grad_acc_steps,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_length': max_length,
        'label_smoothing': label_smoothing,
        'early_stopping_patience': early_stopping_patience,
        'num_workers': num_workers,
        'seed': seed,
    }
    if device.type == 'cuda':
        run_metadata['gpu_name'] = torch.cuda.get_device_name(0)
        run_metadata['gpu_total_vram_gb'] = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    with open(os.path.join(models_dir, 'run_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    print(f"Mixed precision: {'ON' if device.type=='cuda' else 'OFF'} | Grad Acc Steps: {grad_acc_steps}")
    print(f"Weight decay: {weight_decay}")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': [], 'val_weighted_f1': []}

    # Early stopping & best checkpoint tracking
    best_metric_val = -float('inf')  # tracking val_weighted_f1 (higher is better)
    no_improve_epochs = 0
    best_model_path = os.path.join(models_dir, 'best_transformer_model')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, grad_acc_steps)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        if val_loader is not None:
            val_loss, val_acc, val_preds, val_trues = evaluate(model, val_loader, device, criterion)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            val_macro_f1 = f1_score(val_trues, val_preds, average='macro') if len(val_trues) > 0 else 0.0
            val_weighted_f1 = f1_score(val_trues, val_preds, average='weighted') if len(val_trues) > 0 else 0.0
            history['val_macro_f1'].append(val_macro_f1)
            history['val_weighted_f1'].append(val_weighted_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}, Val Weighted-F1: {val_weighted_f1:.4f}")

            # Early stopping logic based on val_weighted_f1
            current = val_weighted_f1
            improved = current > best_metric_val + 1e-4
            
            if improved:
                best_metric_val = current
                no_improve_epochs = 0
                Path(best_model_path).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                with open(os.path.join(best_model_path, 'metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(run_metadata, f, ensure_ascii=False, indent=2)
                print(f"Saved BEST model to {best_model_path} (val_weighted_f1 = {current:.4f})")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_patience:
                    print(f"Early stopping triggered (no improvement in {early_stopping_patience} epochs)")
                    break
    
    # Load best checkpoint as the final model
    if os.path.isdir(best_model_path):
        print(f"\nLoading BEST checkpoint from {best_model_path} as final model...")
        model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        print(f"Best checkpoint loaded (val_weighted_f1 = {best_metric_val:.4f})")
    else:
        print(f"\nWarning: Best checkpoint not found, using current model state")
    
    # Plot training history
    history_plot_path = os.path.join(reports_dir, '04-transformer_training_history.png')
    plot_training_history(history, history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    
    # Evaluate on test set if available
    if test_df is not None:
        
        y_test_str = test_df['label'].astype(str).tolist()
        y_test_numeric = [normalize_label(label) for label in y_test_str]
        y_test = [label2id[n] for n in y_test_numeric]
        X_test = test_df['text'].astype(str).tolist()
        test_dataset = LegalTextDataset(X_test, y_test, tokenizer, max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type=='cuda',
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )
        test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, device, criterion)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        test_macro_f1 = f1_score(true_labels, predictions, average='macro')
        print(f"Test Macro-F1: {test_macro_f1:.4f}")
        
        # Generate classification report
        pred_labels_str = [id2label[int(pred)] for pred in predictions]
        true_labels_str = [id2label[int(label)] for label in true_labels]
        report = classification_report(
            true_labels_str,
            pred_labels_str,
            output_dict=True,
            zero_division=0
        )
        
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
        
        y_true_num = labels_to_numeric(true_labels_str)
        y_pred_num = labels_to_numeric(pred_labels_str)
        mae = mean_absolute_error(y_true_num, y_pred_num)
        rmse = np.sqrt(mean_squared_error(y_true_num, y_pred_num))
        
        report['mae'] = float(mae)
        report['rmse'] = float(rmse)
        
        print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
        
        report_path = os.path.join(reports_dir, '04-transformer_test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Test report saved to {report_path}")

        # Confusion matrix plot
        cm = confusion_matrix(true_labels_str, pred_labels_str)
        labels = sorted(set(true_labels_str) | set(pred_labels_str))
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels,
               ylabel='True', xlabel='Predicted', title='Confusion Matrix (Test)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black')
        
        fig.tight_layout()
        cm_path = os.path.join(reports_dir, '04-transformer_test_confusion_matrix.png')
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix saved to {cm_path}")


if __name__ == '__main__':
    main()
