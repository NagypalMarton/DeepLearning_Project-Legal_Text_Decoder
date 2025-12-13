import os
import json
from pathlib import Path
import random
import shutil
import argparse

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


# ============================================================================
# Progressive Model Architectures
# ============================================================================

class Step1_BaselineModel(nn.Module):
    """Step 1: Minimal baseline - Transformer + single linear classifier.
    
    Architecture:
    - Pretrained transformer (frozen or fine-tuned)
    - CLS pooling
    - Single linear layer
    - Minimal regularization (only weight decay from optimizer)
    
    Purpose: Validate basic capacity without overfitting.
    """
    def __init__(self, transformer_model, num_classes=5):
        super().__init__()
        self.transformer = transformer_model
        self.num_classes = num_classes
        hidden_size = transformer_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        # CLS token pooling
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        
        output = type('Output', (), {'logits': logits})()
        if labels is not None:
            output.loss = nn.CrossEntropyLoss()(logits, labels)
        return output


class Step2_ExtendedModel(nn.Module):
    """Step 2: Extended - 2-layer adapter + BatchNorm + Dropout.
    
    Architecture:
    - Pretrained transformer
    - CLS pooling
    - 2-layer adapter network (hidden_dim=256)
    - BatchNorm for stabilization
    - Dropout(0.3) for regularization
    - GELU activation
    
    Purpose: Add moderate capacity with basic regularization.
    """
    def __init__(self, transformer_model, num_classes=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.transformer = transformer_model
        self.num_classes = num_classes
        trans_hidden = transformer_model.config.hidden_size
        
        self.adapter = nn.Sequential(
            nn.Linear(trans_hidden, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        adapted = self.adapter(pooled)
        logits = self.classifier(adapted)
        
        output = type('Output', (), {'logits': logits})()
        if labels is not None:
            output.loss = nn.CrossEntropyLoss()(logits, labels)
        return output


class Step3_AdvancedModel(nn.Module):
    """Step 3: Advanced - Attention pooling + 3-layer adapter + gating.
    
    Architecture:
    - Pretrained transformer
    - Multi-head attention pooling (learns important tokens)
    - 3-layer deep adapter network
    - Gating mechanism (residual connection)
    - LayerNorm + Dropout(0.4)
    - GELU activation
    
    Purpose: Maximum capacity with advanced regularization.
    """
    def __init__(self, transformer_model, num_classes=5, hidden_dim=256, dropout=0.4, num_heads=4):
        super().__init__()
        self.transformer = transformer_model
        self.num_classes = num_classes
        trans_hidden = transformer_model.config.hidden_size
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=trans_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, trans_hidden))
        
        # 3-layer adapter with residual gating
        self.adapter_1 = nn.Sequential(
            nn.Linear(trans_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.adapter_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.adapter_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gating for residual connection
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = outputs.last_hidden_state  # [B, L, H]
        
        # Attention pooling
        batch_size = hidden.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pool(query, hidden, hidden, key_padding_mask=~attention_mask.bool())
        pooled = pooled.squeeze(1)  # [B, H]
        
        # 3-layer adapter with gating
        x = self.adapter_1(pooled)
        residual = x
        x = self.adapter_2(x)
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        x = gate * x + (1 - gate) * residual
        x = self.adapter_3(x)
        
        logits = self.classifier(x)
        
        output = type('Output', (), {'logits': logits})()
        if labels is not None:
            output.loss = nn.CrossEntropyLoss()(logits, labels)
        return output


class BalancedFinalModel(nn.Module):
    """Final: Balanced - Production-ready model with best practices.
    
    Architecture:
    - Pretrained transformer
    - Mean pooling (more robust than CLS)
    - 2-layer adapter (balanced complexity)
    - LayerNorm (better than BatchNorm for variable batch sizes)
    - Moderate Dropout(0.3)
    - GELU activation
    - Gradient clipping in training
    
    Purpose: PRODUCTION RECOMMENDED - Best balance of performance and robustness.
    """
    def __init__(self, transformer_model, num_classes=5, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.transformer = transformer_model
        self.num_classes = num_classes
        trans_hidden = transformer_model.config.hidden_size
        
        self.adapter = nn.Sequential(
            nn.Linear(trans_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = outputs.last_hidden_state  # [B, L, H]
        
        # Mean pooling (more robust than CLS)
        mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        summed = (hidden * mask).sum(1)  # [B, H]
        counts = mask.sum(1).clamp(min=1)  # [B, 1]
        pooled = summed / counts
        
        adapted = self.adapter(pooled)
        logits = self.classifier(adapted)
        
        output = type('Output', (), {'logits': logits})()
        if labels is not None:
            output.loss = nn.CrossEntropyLoss()(logits, labels)
        return output




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


def plot_model_comparison(all_results, save_path):
    """Plot comparison of all 4 models' performance."""
    model_names = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Val Accuracy Comparison
    val_accs = [all_results[name]['val_acc'] for name in model_names]
    train_accs = [all_results[name]['train_acc'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_accs, width, label='Train Acc', color='#3498db')
    axes[0, 0].bar(x + width/2, val_accs, width, label='Val Acc', color='#e74c3c')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Train vs Val Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Train-Val Gap (Overfitting Indicator)
    gaps = [all_results[name]['train_acc'] - all_results[name]['val_acc'] for name in model_names]
    colors = ['#2ecc71' if g < 0.05 else '#f39c12' if g < 0.10 else '#e74c3c' for g in gaps]
    axes[0, 1].bar(model_names, gaps, color=colors)
    axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', label='5% threshold')
    axes[0, 1].axhline(y=0.10, color='red', linestyle='--', label='10% threshold')
    axes[0, 1].set_ylabel('Train - Val Accuracy Gap')
    axes[0, 1].set_title('Overfitting Analysis (Lower is Better)')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Convergence Speed (Epochs to Best)
    epochs_to_best = [all_results[name]['epochs_trained'] for name in model_names]
    axes[1, 0].bar(model_names, epochs_to_best, color='#9b59b6')
    axes[1, 0].set_ylabel('Epochs')
    axes[1, 0].set_title('Training Epochs (Convergence Speed)')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: F1 Scores Comparison
    macro_f1s = [all_results[name]['val_macro_f1'] for name in model_names]
    weighted_f1s = [all_results[name]['val_weighted_f1'] for name in model_names]
    
    x = np.arange(len(model_names))
    axes[1, 1].bar(x - width/2, macro_f1s, width, label='Macro F1', color='#1abc9c')
    axes[1, 1].bar(x + width/2, weighted_f1s, width, label='Weighted F1', color='#34495e')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Model comparison plot saved to {save_path}")


def train_single_model(model_config, base_transformer, tokenizer, train_loader, val_loader, 
                      device, criterion, label2id, id2label, models_dir, reports_dir, 
                      learning_rate, weight_decay, epochs, early_stopping_patience, grad_acc_steps):
    """Train a single model configuration and return results."""
    
    model_name = model_config['name']
    model_class = model_config['class']
    num_classes = len(label2id)
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"Architecture: {model_config['description']}")
    print(f"{'='*80}\n")
    
    # Create model
    model = model_class(base_transformer, num_classes=num_classes)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    effective_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_acc_steps))
    total_steps = effective_steps_per_epoch * epochs
    warmup_steps = int(0.15 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': [], 'val_weighted_f1': []}
    best_metric_val = -float('inf')
    no_improve_epochs = 0
    best_model_path = os.path.join(models_dir, f'best_{model_name.lower().replace(" ", "_")}.pt')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, grad_acc_steps)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        if val_loader is not None:
            val_loss, val_acc, val_preds, val_trues = evaluate(model, val_loader, device, criterion)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            val_macro_f1 = f1_score(val_trues, val_preds, average='macro')
            val_weighted_f1 = f1_score(val_trues, val_preds, average='weighted')
            history['val_macro_f1'].append(val_macro_f1)
            history['val_weighted_f1'].append(val_weighted_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}, Val Weighted-F1: {val_weighted_f1:.4f}")
            
            # Early stopping based on val_weighted_f1
            current = val_weighted_f1
            if current > best_metric_val + 1e-4:
                best_metric_val = current
                no_improve_epochs = 0
                # Save best checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'val_weighted_f1': best_metric_val,
                    'label2id': label2id,
                    'id2label': id2label,
                    'epoch': epoch + 1
                }
                torch.save(checkpoint, best_model_path)
                print(f"✓ Saved BEST checkpoint (val_weighted_f1 = {current:.4f})")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_patience:
                    print(f"Early stopping triggered (no improvement in {early_stopping_patience} epochs)")
                    break
    
    # Load best checkpoint
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Loaded best checkpoint (val_weighted_f1 = {checkpoint['val_weighted_f1']:.4f})")
    
    # Plot training history
    history_plot_path = os.path.join(reports_dir, f'04-{model_name.lower().replace(" ", "_")}_history.png')
    plot_training_history(history, history_plot_path)
    
    # Return final metrics
    return {
        'name': model_name,
        'train_acc': history['train_acc'][-1],
        'val_acc': history['val_acc'][-1] if history['val_acc'] else 0.0,
        'val_macro_f1': history['val_macro_f1'][-1] if history['val_macro_f1'] else 0.0,
        'val_weighted_f1': best_metric_val,
        'epochs_trained': len(history['train_acc']),
        'best_checkpoint': best_model_path
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive model training with optional subset exploration")
    parser.add_argument("--transformer_model", default="SZTAKI-HLT/hubert-base-cc", help="Base transformer model name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_acc_steps", type=int, default=2)
    parser.add_argument("--output_dir", default="/app/output", help="Base output directory")
    parser.add_argument("--subset_fraction", type=float, default=0.2, help="Fraction of train/val used for exploration")
    parser.add_argument("--final_train_winner_only", action="store_true", default=True, help="Retrain only winner on full data (default True)")
    parser.add_argument("--balanced_subset", action="store_true", default=True, help="Use equal per-class sampling for subset (default True)")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    # Set random seed
    seed = args.random_seed
    set_seed(seed)
    print(f"Random seed set to {seed}")
    
    # Configuration
    base_output = args.output_dir
    processed_dir = os.path.join(base_output, 'processed')
    models_dir = os.path.join(base_output, 'models')
    reports_dir = os.path.join(base_output, 'reports')
    
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    model_name = args.transformer_model
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    max_length = args.max_length
    label_smoothing = args.label_smoothing
    early_stopping_patience = args.early_stopping_patience
    num_workers = args.num_workers
    grad_acc_steps = args.grad_acc_steps
    # Subset exploration + final winner training controls
    subset_fraction = args.subset_fraction  # fraction of train/val used for exploration
    final_train_winner_only = args.final_train_winner_only
    balanced_subset = args.balanced_subset  # if true, use equal class counts in subset
    
    print(f"\n{'='*80}")
    print(f"PROGRESSIVE MODEL EXPANSION TRAINING")
    print(f"{'='*80}")
    print(f"Base Model: {model_name}")
    print(f"Batch Size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}")
    print(f"Weight Decay: {weight_decay} | Label Smoothing: {label_smoothing}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Subset Fraction: {subset_fraction} | Final Train Winner Only: {final_train_winner_only} | Balanced Subset: {balanced_subset}")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"Loading data from {processed_dir}")
    train_df, val_df, test_df = load_split_csv(processed_dir)
    
    # Optional stratified subset for fast exploration
    if subset_fraction < 1.0:
        print(f"\n{'='*80}")
        print(f"SUBSET MODE: Using {subset_fraction*100:.1f}% of training/validation data for exploration")
        print(f"{'='*80}\n")
        def _stratified_sample(df, label_col, frac, seed, balanced=False):
            if df is None or len(df) == 0:
                return df
            
            # Manually sample each group to avoid groupby().apply() column issues
            unique_labels = df[label_col].unique()
            samples = []
            
            if not balanced:
                # Standard stratified sampling by fraction
                for label_val in unique_labels:
                    group = df[df[label_col] == label_val]
                    if len(group) > 0:
                        sample = group.sample(frac=frac, random_state=seed)
                        samples.append(sample)
            else:
                # Balanced sampling: equal count per class based on the smallest class size
                class_counts = df[label_col].value_counts()
                min_count = int(class_counts.min())
                target_per_class = max(1, int(min_count * frac))
                
                for label_val in unique_labels:
                    group = df[df[label_col] == label_val]
                    if len(group) > 0:
                        sample_size = min(target_per_class, len(group))
                        sample = group.sample(n=sample_size, random_state=seed)
                        samples.append(sample)
            
            # Concatenate all samples and reset index
            if samples:
                result = pd.concat(samples, ignore_index=True)
                return result
            else:
                return df.iloc[:0]  # Return empty dataframe with same structure

        train_df = _stratified_sample(train_df, 'label', subset_fraction, 42, balanced_subset)
        if val_df is not None and len(val_df) > 0:
            val_df = _stratified_sample(val_df, 'label', subset_fraction, 42, balanced_subset)
        print(f"Train size after subset: {len(train_df)}")
        if val_df is not None:
            print(f"Val size after subset: {len(val_df)}")

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
    
    # Load tokenizer and BASE transformer (will be reused for all 4 models)
    print(f"\nLoading base transformer: {model_name}")
    from transformers import AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
    
    # Compute class weights
    class_weights_raw = compute_class_weight('balanced', classes=np.unique(y_train), y=np.array(y_train))
    class_weights = np.sqrt(class_weights_raw)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights (sqrt-scaled): {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
    
    # ============================================================================
    # PROGRESSIVE MODEL CONFIGS
    # ============================================================================
    model_configs = [
        {
            'name': 'Step1_Baseline',
            'class': Step1_BaselineModel,
            'description': 'Minimal baseline: Transformer + single linear classifier'
        },
        {
            'name': 'Step2_Extended',
            'class': Step2_ExtendedModel,
            'description': '2-layer adapter + BatchNorm + Dropout(0.3)'
        },
        {
            'name': 'Step3_Advanced',
            'class': Step3_AdvancedModel,
            'description': 'Attention pooling + 3-layer adapter + gating mechanism'
        },
        {
            'name': 'Final_Balanced',
            'class': BalancedFinalModel,
            'description': 'PRODUCTION RECOMMENDED: Mean pooling + balanced architecture'
        }
    ]
    
    # ============================================================================
    # TRAIN ALL 4 MODELS AUTOMATICALLY
    # ============================================================================
    all_results = {}
    
    for config in model_configs:
        # Load fresh base transformer for each model
        from transformers import AutoModel
        base_transformer = AutoModel.from_pretrained(model_name)
        
        result = train_single_model(
            model_config=config,
            base_transformer=base_transformer,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            label2id=label2id,
            id2label=id2label,
            models_dir=models_dir,
            reports_dir=reports_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            grad_acc_steps=grad_acc_steps
        )
        
        all_results[config['name']] = result
        
        # Clean up GPU memory
        del base_transformer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ============================================================================
    # GENERATE COMPARISON REPORT
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    for name, result in all_results.items():
        print(f"{name}:")
        print(f"  Train Acc: {result['train_acc']:.4f} | Val Acc: {result['val_acc']:.4f}")
        print(f"  Val Weighted F1: {result['val_weighted_f1']:.4f} | Val Macro F1: {result['val_macro_f1']:.4f}")
        print(f"  Epochs Trained: {result['epochs_trained']}")
        print(f"  Gap (Train-Val): {result['train_acc'] - result['val_acc']:.4f}")
        print()
    
    # Save summary JSON
    summary_path = os.path.join(reports_dir, '04-expansion_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # Generate comparison plot
    comparison_plot_path = os.path.join(reports_dir, '04-model_expansion_comparison.png')
    plot_model_comparison(all_results, comparison_plot_path)
    
    # Identify best model from exploration
    best_name, best_result = max(all_results.items(), key=lambda kv: kv[1]['val_weighted_f1'])
    print(f"\n{'='*80}")
    print(f"BEST MODEL IDENTIFIED (exploration): {best_name}")
    print(f"Val Weighted F1: {best_result['val_weighted_f1']:.4f}")
    print(f"{'='*80}")

    # Export generic pointers for downstream pipeline (05/06)
    best_overall_path = os.path.join(models_dir, 'best_overall.json')
    with open(best_overall_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_model_name': best_name,
            'best_checkpoint': best_result['best_checkpoint'],
            'val_weighted_f1': best_result['val_weighted_f1']
        }, f, ensure_ascii=False, indent=2)
    try:
        shutil.copyfile(best_result['best_checkpoint'], os.path.join(models_dir, 'best_model.pt'))
    except Exception as _copy_exc:
        print(f"Warning: could not copy generic best_model.pt: {_copy_exc}")

    # Optional: retrain winner on FULL dataset for final production checkpoint
    if subset_fraction < 1.0 and final_train_winner_only:
        print(f"\n{'='*80}")
        print(f"FINAL TRAINING: Retraining winner '{best_name}' on FULL dataset")
        print(f"{'='*80}\n")

        # Reload full data
        train_df_full, val_df_full, test_df_full = load_split_csv(processed_dir)

        # Rebuild label mapping on full train
        y_train_str_full = train_df_full['label'].astype(str).tolist()
        y_train_full, label2id_full, id2label_full = build_ordinal_mapping(y_train_str_full)

        X_train_full = train_df_full['text'].astype(str).tolist()
        train_dataset_full = LegalTextDataset(X_train_full, y_train_full, tokenizer, max_length)
        train_loader_full = DataLoader(
            train_dataset_full,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type=='cuda',
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )

        val_loader_full = None
        if val_df_full is not None:
            y_val_str_full = val_df_full['label'].astype(str).tolist()
            y_val_numeric_full = [normalize_label(label) for label in y_val_str_full]
            y_val_full = [label2id_full[n] for n in y_val_numeric_full]
            X_val_full = val_df_full['text'].astype(str).tolist()
            val_dataset_full = LegalTextDataset(X_val_full, y_val_full, tokenizer, max_length)
            val_loader_full = DataLoader(
                val_dataset_full,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=device.type=='cuda',
                num_workers=num_workers,
                persistent_workers=num_workers > 0
            )

        # Fresh base transformer and retrain only the winner
        from transformers import AutoModel
        base_transformer_final = AutoModel.from_pretrained(model_name)
        winner_config = None
        for cfg in model_configs:
            if cfg['name'] == best_name:
                winner_config = cfg
                break
        final_result = train_single_model(
            model_config=winner_config,
            base_transformer=base_transformer_final,
            tokenizer=tokenizer,
            train_loader=train_loader_full,
            val_loader=val_loader_full,
            device=device,
            criterion=criterion,
            label2id=label2id_full,
            id2label=id2label_full,
            models_dir=models_dir,
            reports_dir=reports_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            grad_acc_steps=grad_acc_steps
        )

        # Update generic best pointers to the full-data checkpoint
        with open(best_overall_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_model_name': best_name,
                'best_checkpoint': final_result['best_checkpoint'],
                'val_weighted_f1': final_result['val_weighted_f1']
            }, f, ensure_ascii=False, indent=2)
        try:
            shutil.copyfile(final_result['best_checkpoint'], os.path.join(models_dir, 'best_model.pt'))
        except Exception as _copy_exc2:
            print(f"Warning: could not copy generic best_model.pt: {_copy_exc2}")

        # Clean up
        del base_transformer_final
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*80}")
    print(f"✓ PROGRESSIVE MODEL EXPANSION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest model checkpoints:")
    for name, result in all_results.items():
        print(f"  - {name}: {result['best_checkpoint']}")
    print(f"\n✓ PRODUCTION RECOMMENDED: {best_name}")
    print(f"  Val Weighted F1: {max(all_results.items(), key=lambda kv: kv[1]['val_weighted_f1'])[1]['val_weighted_f1']:.4f}")


if __name__ == '__main__':
    main()
