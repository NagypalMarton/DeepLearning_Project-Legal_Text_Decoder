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
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }
        if self.use_features and self._precomputed_features is not None:
            item['readability_features'] = self._precomputed_features[idx]
        return item


def predict_with_transformer(model, tokenizer, texts, device, batch_size=8, id2label=None, return_probs=False, use_fusion=False, use_coral=False, stats=None):
    """Run batch inference with transformer model."""
    dataset = TransformerDataset(texts, tokenizer, max_length=384, use_features=use_fusion, stats=stats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = []
    probabilities = []
    disable_tqdm = not sys.stdout.isatty()
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", disable=disable_tqdm):
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
            
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
            
            if id2label:
                predictions.extend([id2label[int(p)] for p in preds.cpu().numpy()])
            else:
                predictions.extend(preds.cpu().numpy().tolist())
    
    if return_probs:
        return predictions, probabilities
    return predictions


def get_attention_based_importance(model, tokenizer, texts, labels, device, n_examples=5, use_fusion=False, use_coral=False, stats=None):
    """Extract attention-based token importance for sample texts.
    
    Note: This is a simplified version. Full attention visualization would require
    extracting attention weights from intermediate layers.
    """
    results = []
    
    model.eval()
    for i, text in enumerate(texts[:n_examples]):
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=384,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask_tensor = encoding['attention_mask'].to(device)
        
        # Prepare readability features if fusion
        readability_features = None
        if use_fusion and extract_readability_features is not None:
            feat = extract_readability_features(text).unsqueeze(0)
            if stats:
                mean = torch.tensor(stats['mean'], dtype=torch.float32)
                std = torch.tensor(stats['std'], dtype=torch.float32)
                feat = (feat - mean) / std
            readability_features = feat.to(device)
        
        with torch.no_grad():
            # Extract attention from the transformer backbone
            if use_fusion:
                # For FusionModel, get attentions from model.transformer
                transformer_outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask_tensor, output_attentions=True, return_dict=True)
                attentions = transformer_outputs.attentions
                hidden = transformer_outputs.last_hidden_state
                pooled = model._pool(hidden, attention_mask_tensor)
                if readability_features is not None:
                    feat_out = model.feature_branch(readability_features)
                else:
                    feat_out = torch.zeros(pooled.size(0), 32, device=pooled.device)
                fused = torch.cat([pooled, feat_out], dim=1)
                x = model.dropout(F.gelu(model.fusion_fc(fused)))
                if use_coral:
                    logits = model.coral_head(x)
                else:
                    logits = model.classifier(x)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask_tensor, output_attentions=True)
                logits = outputs.logits
                attentions = outputs.attentions  # tuple of attention weights per layer
            
            # Get prediction
            if use_coral and coral_predict is not None:
                pred_class = int(coral_predict(logits)[0])
                probs = torch.sigmoid(logits)[0]  # CORAL uses sigmoid per threshold
            else:
                pred_class = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1)[0]
            
            # Average attention across all layers and heads
            avg_attention = torch.stack([att.mean(dim=1) for att in attentions]).mean(dim=0)[0]
            
            # Get tokens and their attention scores
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_scores = avg_attention.mean(dim=0).cpu().numpy()  # average across attention to all positions
            
            # Filter out padding tokens
            valid_tokens = []
            valid_scores = []
            for token, score in zip(tokens, attention_scores):
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    valid_tokens.append(token)
                    valid_scores.append(float(score))
            
            # Get top 10 attended tokens
            top_indices = np.argsort(valid_scores)[-10:][::-1]
            top_tokens = [(valid_tokens[idx], valid_scores[idx]) for idx in top_indices if idx < len(valid_tokens)]
            
            results.append({
                'example_id': i,
                'text_preview': text[:200] + ('...' if len(text) > 200 else ''),
                'true_label': str(labels[i]),
                'predicted_class_id': pred_class,
                'prediction_probabilities': probs.cpu().numpy().tolist(),
                'top_attended_tokens': top_tokens
            })
    
    return results


def plot_confusion_pairs(confusion_pairs, save_path, top_n=10):
    """Plot top confusion pairs."""
    if not confusion_pairs:
        print("No confusion pairs to plot")
        return
    
    pairs = confusion_pairs[:top_n]
    pair_labels = [f"{p['true_label']} → {p['predicted_label']}" for p in pairs]
    counts = [p['count'] for p in pairs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pair_labels, counts, color='coral')
    ax.set_xlabel('Count')
    ax.set_title('Top Confusion Pairs - Transformer')
    ax.invert_yaxis()
    
    for i, (pair, count) in enumerate(zip(pair_labels, counts)):
        ax.text(count, i, f' {count}', va='center')
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Confusion pairs plot saved to {save_path}")


def analyze_misclassifications(y_test, y_pred, X_test):
    """Analyze common misclassification patterns."""
    # Find misclassified examples
    misclassified = []
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        if pred != true:
            misclassified.append({
                'index': i,
                'text': X_test[i][:200] + ('...' if len(X_test[i]) > 200 else ''),
                'true_label': str(true),
                'predicted_label': str(pred)
            })
    
    # Count misclassification patterns
    confusion_pairs = {}
    for item in misclassified:
        pair = (item['true_label'], item['predicted_label'])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'total_misclassified': len(misclassified),
        'total_examples': len(X_test),
        'error_rate': len(misclassified) / len(X_test) if len(X_test) > 0 else 0,
        'confusion_pairs': [
            {
                'true_label': pair[0],
                'predicted_label': pair[1],
                'count': count
            }
            for pair, count in sorted_pairs[:10]
        ],
        'examples': misclassified[:10]
    }


def summarize_attention(results):
    """Compute average attention score per label."""
    by_label = {}
    for r in results:
        tl = r['true_label']
        score_mean = np.mean([s for _, s in r['top_attended_tokens']]) if r['top_attended_tokens'] else 0
        by_label.setdefault(tl, []).append(score_mean)
    return {k: float(np.mean(v)) for k, v in by_label.items()}


def main():
    base_output = os.getenv('OUTPUT_DIR', '/app/output')
    processed_dir = os.path.join(base_output, 'processed')
    models_dir = os.path.join(base_output, 'models')
    explainability_dir = os.path.join(base_output, 'reports')
    
    Path(explainability_dir).mkdir(parents=True, exist_ok=True)
    
    # Load transformer model
    model_path = os.path.join(models_dir, 'best_transformer_model')
    if not os.path.isdir(model_path):
        print(f"Transformer model not found at {model_path}. Train the transformer first.")
        return
    
    # Load label mapping with fallback + expansion
    label_map_path_generic = os.path.join(models_dir, 'label_mapping.json')
    label_map_path_baseline = os.path.join(models_dir, 'baseline_label_mapping.json')
    if os.path.exists(label_map_path_generic):
        with open(label_map_path_generic, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
    elif os.path.exists(label_map_path_baseline):
        with open(label_map_path_baseline, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
    else:
        print(f"No label mapping file found (searched both generic and baseline).")
        return
    id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
    
    def _expand_numeric_labels(id2label_local, sample_labels):
        num_to_full = {}
        for lbl in sample_labels:
            m = re.match(r'^([1-5])', str(lbl).strip())
            if m:
                n = m.group(1)
                if n not in num_to_full:
                    num_to_full[n] = lbl
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
                    model_name = os.getenv('TRANSFORMER_MODEL', 'SZTAKI-HLT/hubert-base-cc')
                    base_model = AutoModel.from_pretrained(model_name)
                    model = FusionModel(base_model, num_classes=num_labels, use_coral=use_coral)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    
                    # Load feature stats
                    metadata_path = os.path.join(model_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
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
    test_path = os.path.join(processed_dir, 'test.csv')
    if not os.path.exists(test_path):
        print(f"Test data not found at {test_path}.")
        return
    
    test_df = pd.read_csv(test_path)
    X_test = test_df['text'].astype(str).tolist()
    y_test = test_df['label'].astype(str).tolist()
    
    # Expand numeric labels if mapping numeric-only
    id2label = _expand_numeric_labels(id2label, y_test)
    
    batch_size = int(os.getenv('BATCH_SIZE', '8'))
    
    print("Analyzing model explainability...")
    
    # Get predictions for all test samples
    print("Running predictions on test set...")
    y_pred = predict_with_transformer(model, tokenizer, X_test, device, batch_size, id2label, use_fusion=use_fusion, use_coral=use_coral, stats=stats)
    
    # Attention-based importance for sample texts
    print("Extracting attention-based token importance...")
    try:
        attention_results = get_attention_based_importance(model, tokenizer, X_test, y_test, device, n_examples=10, use_fusion=use_fusion, use_coral=use_coral, stats=stats)
        
        attention_path = os.path.join(explainability_dir, '07-explainability_attention_importance.json')
        with open(attention_path, 'w', encoding='utf-8') as f:
            json.dump(attention_results, f, ensure_ascii=False, indent=2)
        print(f"Attention importance saved to {attention_path}")
        
        # Save attention summary
        summary = summarize_attention(attention_results)
        summary_path = os.path.join(explainability_dir, '07-explainability_attention_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Attention summary saved to {summary_path}")
    except Exception as e:
        print(f"Attention extraction failed: {e}")
        attention_results = []
    
    # Analyze misclassifications
    print("Analyzing misclassifications...")
    misclass_analysis = analyze_misclassifications(y_test, y_pred, X_test)
    
    misclass_path = os.path.join(explainability_dir, '07-explainability_misclassification_analysis.json')
    with open(misclass_path, 'w', encoding='utf-8') as f:
        json.dump(misclass_analysis, f, ensure_ascii=False, indent=2)
    print(f"Misclassification analysis saved to {misclass_path}")
    
    # Plot confusion pairs
    if misclass_analysis['confusion_pairs']:
        confusion_plot_path = os.path.join(explainability_dir, '07-explainability_top_confusion_pairs.png')
        plot_confusion_pairs(misclass_analysis['confusion_pairs'], confusion_plot_path)
    
    # Print summary
    print("\n=== Explainability Summary ===")
    print(f"Total test examples: {misclass_analysis['total_examples']}")
    print(f"Misclassified: {misclass_analysis['total_misclassified']}")
    print(f"Error rate: {misclass_analysis['error_rate']:.4f}")
    
    if attention_results:
        print(f"\nAnalyzed attention for {len(attention_results)} example(s)")
    
    print("Top confusion pairs:")
    for pair in misclass_analysis['confusion_pairs'][:5]:
        print(f"  {pair['true_label']} -> {pair['predicted_label']}: {pair['count']} times")


if __name__ == '__main__':
    main()
