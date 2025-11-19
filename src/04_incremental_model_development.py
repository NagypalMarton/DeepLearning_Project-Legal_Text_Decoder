import os
import json
import pickle
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
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import torch.nn.functional as F
import textstat
import re


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LegalTextDataset(Dataset):
    """Dataset for legal text classification with transformers."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, use_features=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = use_features
    
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
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        if self.use_features:
            features = extract_readability_features(text)
            item['readability_features'] = features
        
        return item


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


def extract_readability_features(text):
    """Extract readability and complexity features from text.
    
    Returns 8-dimensional feature vector:
    - Flesch Reading Ease (0-100, higher = easier)
    - Flesch-Kincaid Grade Level (US school grade)
    - Average word length (characters)
    - Average sentence length (words)
    - Lexical diversity (unique/total words ratio)
    - Complex words ratio (3+ syllables)
    - Sentence count
    - Word count (log-scaled)
    """
    if not text or not text.strip():
        return torch.zeros(8, dtype=torch.float32)
    
    try:
        # Basic stats
        words = text.split()
        word_count = max(1, len(words))
        sentences = re.split(r'[.!?]+', text)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        
        # Flesch metrics (textstat handles edge cases)
        flesch_ease = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
        
        # Average word length
        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
        
        # Average sentence length
        avg_sent_len = word_count / sentence_count
        
        # Lexical diversity
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0.0
        
        # Complex words (textstat syllable count)
        complex_words = sum(1 for w in words if textstat.syllable_count(w) >= 3)
        complex_ratio = complex_words / word_count if word_count > 0 else 0.0
        
        # Log-scaled word count (normalize very long texts)
        log_word_count = np.log1p(word_count)
        
        features = [
            flesch_ease / 100.0,  # normalize to [0,1]
            flesch_grade / 20.0,  # normalize (grade levels ~0-20)
            avg_word_len / 15.0,  # normalize (typical max ~12-15)
            avg_sent_len / 50.0,  # normalize (typical max ~30-50)
            lexical_diversity,    # already [0,1]
            complex_ratio,        # already [0,1]
            sentence_count / 100.0,  # normalize
            log_word_count / 10.0    # normalize (log(~20k words) ≈ 10)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    except Exception as e:
        # Fallback to zeros if extraction fails
        print(f"Warning: feature extraction failed: {e}")
        return torch.zeros(8, dtype=torch.float32)


class FusionModel(nn.Module):
    """Transformer + Readability Features Fusion Model.
    
    Combines BERT-like embeddings with handcrafted readability metrics.
    """
    
    def __init__(self, transformer_model, num_classes=5, feature_dim=8, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.transformer = transformer_model
        self.feature_dim = feature_dim
        
        # Get transformer hidden size (768 for BERT-base)
        transformer_hidden_size = transformer_model.config.hidden_size
        
        # Fusion layers
        self.fusion_fc = nn.Linear(transformer_hidden_size + feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, readability_features=None, labels=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract [CLS] token embedding from last hidden state
        cls_embedding = outputs.hidden_states[-1][:, 0, :]  # [batch, hidden_size]
        
        if readability_features is not None:
            # Concatenate transformer embedding + readability features
            fused = torch.cat([cls_embedding, readability_features], dim=1)
        else:
            # Fallback: pad with zeros if features not provided
            batch_size = cls_embedding.size(0)
            zero_features = torch.zeros(batch_size, self.feature_dim, device=cls_embedding.device)
            fused = torch.cat([cls_embedding, zero_features], dim=1)
        
        # Classification head
        x = self.dropout(F.relu(self.fusion_fc(fused)))
        logits = self.classifier(x)
        
        # Return in format compatible with transformers
        output = type('Output', (), {'logits': logits})()
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            output.loss = loss_fct(logits, labels)
        
        return output


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the correct class.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights (tensor or None)
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes)
        # targets: (batch_size,)
        
        # Compute cross entropy with label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Get probabilities
        p = torch.exp(-ce_loss)  # p_t
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, dataloader, optimizer, scheduler, device, criterion=None, use_fusion=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    # Disable progress bar if output is redirected to file
    disable_tqdm = not sys.stdout.isatty()
    progress_bar = tqdm(dataloader, desc="Training", disable=disable_tqdm)
    scaler = amp.GradScaler('cuda', enabled=os.getenv('MIXED_PRECISION', '1') == '1' and device.type == 'cuda')
    grad_acc_steps = int(os.getenv('GRAD_ACC_STEPS', '1'))
    step_count = 0
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        readability_features = batch.get('readability_features', None)
        if readability_features is not None:
            readability_features = readability_features.to(device, non_blocking=True)
        
        with amp.autocast('cuda', enabled=scaler.is_enabled()):
            if use_fusion:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                              readability_features=readability_features)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if criterion is None:
                # Fallback to model's internal loss if no custom criterion provided
                if use_fusion:
                    outputs_with_labels = model(input_ids=input_ids, attention_mask=attention_mask,
                                               readability_features=readability_features, labels=labels)
                else:
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
        
        total_loss += loss.item() * grad_acc_steps  # accumulate real loss value
        preds = torch.argmax(logits.detach(), dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': (total_loss / (step_count))})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device, criterion=None, use_fusion=False):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        # Disable progress bar if output is redirected to file
        disable_tqdm = not sys.stdout.isatty()
        progress_bar = tqdm(dataloader, desc="Evaluating", disable=disable_tqdm)
        mixed = os.getenv('MIXED_PRECISION', '1') == '1' and device.type == 'cuda'
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            readability_features = batch.get('readability_features', None)
            if readability_features is not None:
                readability_features = readability_features.to(device, non_blocking=True)
            
            with amp.autocast('cuda', enabled=mixed):
                if use_fusion:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                  readability_features=readability_features)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if criterion is None:
                    if use_fusion:
                        outputs_with_labels = model(input_ids=input_ids, attention_mask=attention_mask,
                                                   readability_features=readability_features, labels=labels)
                    else:
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
    model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')  # Hungarian BERT
    batch_size = int(os.getenv('BATCH_SIZE', '8'))  # increased default; still safe for 4GB VRAM
    epochs = int(os.getenv('EPOCHS', '10'))  # increased default epochs for better convergence
    learning_rate = float(os.getenv('LEARNING_RATE', '2e-5'))
    weight_decay = float(os.getenv('WEIGHT_DECAY', '0.01'))  # L2 regularization
    max_length = int(os.getenv('MAX_LENGTH', '320'))  # optimized for 4GB VRAM
    label_smoothing = float(os.getenv('LABEL_SMOOTHING', '0.05'))  # OPTIMIZED: reduced from 0.15
    early_stopping_enabled = os.getenv('EARLY_STOPPING', '1') == '1'
    early_stopping_patience = int(os.getenv('EARLY_STOPPING_PATIENCE', '4'))  # OPTIMIZED: increased from 2
    save_best_metric = os.getenv('SAVE_BEST_METRIC', 'val_weighted_f1')  # OPTIMIZED: weighted F1 for imbalanced data
    use_class_weights = os.getenv('USE_CLASS_WEIGHTS', '1') == '1'  # sqrt-scaled class weights
    use_focal_loss = os.getenv('USE_FOCAL_LOSS', '0') == '1'  # Focal Loss toggle
    focal_gamma = float(os.getenv('FOCAL_GAMMA', '2.0'))  # Focal Loss gamma parameter
    use_feature_fusion = os.getenv('USE_FEATURE_FUSION', '1') == '1'  # OPTIMIZED: Feature Fusion enabled
    
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
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} | Total VRAM: {total_mem:.2f} GB")
    
    # Load tokenizer and model
    print(f"Loading transformer model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create fusion or standard model based on configuration
    if use_feature_fusion:
        print("Creating FusionModel with readability features (8-dim)")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        model = FusionModel(base_model, num_classes=num_labels, feature_dim=8)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
    model.to(device)

    # Compute class weights for handling class imbalance
    class_weights_tensor = None
    if use_class_weights:
        class_weights_raw = compute_class_weight('balanced', classes=np.unique(y_train), y=np.array(y_train))
        # Apply sqrt scaling to reduce aggressiveness (less extreme weights)
        class_weights = np.sqrt(class_weights_raw)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class weights (sqrt-scaled): {class_weights}")
        print(f"Class weights (original balanced): {class_weights_raw}")
    
    # Loss function selection
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    elif use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
        print(f"Using CrossEntropy with class weights (label_smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"Using CrossEntropy (label_smoothing={label_smoothing})")
    
    # Prepare datasets
    X_train = train_df['text'].astype(str).tolist()
    train_dataset = LegalTextDataset(X_train, y_train, tokenizer, max_length, use_features=use_feature_fusion)
    num_workers = int(os.getenv('NUM_WORKERS', '2'))
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
        y_val = [label2id[label] for label in y_val_str]
        X_val = val_df['text'].astype(str).tolist()
        val_dataset = LegalTextDataset(X_val, y_val, tokenizer, max_length, use_features=use_feature_fusion)
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
    grad_acc_steps = int(os.getenv('GRAD_ACC_STEPS', '2'))  # gradient accumulation for effective larger batch
    effective_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_acc_steps))
    total_steps = effective_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    print(f"Mixed precision: {'ON' if (os.getenv('MIXED_PRECISION','1')=='1' and device.type=='cuda') else 'OFF'} | Grad Acc Steps: {grad_acc_steps}")
    print(f"Weight decay: {weight_decay}")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': [], 'val_weighted_f1': []}

    # Early stopping & best checkpoint tracking
    best_metric_val = float('inf') if save_best_metric == 'val_loss' else -float('inf')
    no_improve_epochs = 0
    best_model_path = os.path.join(models_dir, 'best_transformer_model')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, use_fusion=use_feature_fusion)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        if val_loader is not None:
            val_loss, val_acc, val_preds, val_trues = evaluate(model, val_loader, device, criterion, use_fusion=use_feature_fusion)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            val_macro_f1 = f1_score(val_trues, val_preds, average='macro') if len(val_trues) > 0 else 0.0
            val_weighted_f1 = f1_score(val_trues, val_preds, average='weighted') if len(val_trues) > 0 else 0.0
            history['val_macro_f1'].append(val_macro_f1)
            history['val_weighted_f1'].append(val_weighted_f1)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}, Val Weighted-F1: {val_weighted_f1:.4f}")

            # Early stopping logic - support multiple metrics
            if save_best_metric == 'val_loss':
                current = val_loss
            elif save_best_metric == 'val_weighted_f1':
                current = val_weighted_f1
            else:  # val_macro_f1
                current = val_macro_f1
            
            improved = (current < best_metric_val - 1e-3) if save_best_metric == 'val_loss' else (current > best_metric_val + 1e-4)
            if improved:
                best_metric_val = current
                no_improve_epochs = 0
                # Save best checkpoint - handle both FusionModel and standard transformers
                Path(best_model_path).mkdir(parents=True, exist_ok=True)
                
                if use_feature_fusion:
                    # For FusionModel: save using torch.save
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'best_metric_value': best_metric_val,
                        'save_best_metric': save_best_metric,
                        'label2id': label2id,
                        'id2label': id2label,
                        'use_feature_fusion': True,
                        'num_labels': num_labels
                    }
                    torch.save(checkpoint, os.path.join(best_model_path, 'pytorch_model.bin'))
                    tokenizer.save_pretrained(best_model_path)
                    # Save config for reference
                    with open(os.path.join(best_model_path, 'fusion_config.json'), 'w') as f:
                        json.dump({'num_classes': num_labels, 'feature_dim': 8, 'hidden_dim': 256, 'dropout': 0.3}, f)
                else:
                    # For standard transformer: use save_pretrained
                    model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
                
                print(f"Saved BEST model to {best_model_path} (metric {save_best_metric} = {current:.4f})")
            else:
                no_improve_epochs += 1
                if early_stopping_enabled and no_improve_epochs >= early_stopping_patience:
                    print(f"Early stopping triggered (no improvement in {early_stopping_patience} epochs)")
                    break
    
    # Load best checkpoint as the final model
    if os.path.isdir(best_model_path):
        print(f"\nLoading BEST checkpoint from {best_model_path} as final model...")
        
        if use_feature_fusion:
            # Load FusionModel checkpoint
            checkpoint_path = os.path.join(best_model_path, 'pytorch_model.bin')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                tokenizer = AutoTokenizer.from_pretrained(best_model_path)
                print(f"Best FusionModel checkpoint loaded (metric {save_best_metric} = {best_metric_val:.4f})")
            else:
                print(f"Warning: FusionModel checkpoint not found at {checkpoint_path}")
        else:
            # Load standard transformer
            model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(best_model_path)
            print(f"Best checkpoint loaded (metric {save_best_metric} = {best_metric_val:.4f})")
    else:
        print(f"\nWarning: Best checkpoint not found, using current model state")
    
    # Plot training history
    history_plot_path = os.path.join(reports_dir, '04-transformer_training_history.png')
    plot_training_history(history, history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    
    # Evaluate on test set if available
    if test_df is not None:
        
        y_test_str = test_df['label'].astype(str).tolist()
        y_test = [label2id[label] for label in y_test_str]
        X_test = test_df['text'].astype(str).tolist()
        test_dataset = LegalTextDataset(X_test, y_test, tokenizer, max_length, use_features=use_feature_fusion)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type=='cuda',
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )
        test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, device, criterion, use_fusion=use_feature_fusion)
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


if __name__ == '__main__':
    main()
