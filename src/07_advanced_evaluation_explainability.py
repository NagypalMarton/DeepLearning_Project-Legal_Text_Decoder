import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys


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
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }


def predict_with_transformer(model, tokenizer, texts, device, batch_size=8, id2label=None, return_probs=False):
    """Run batch inference with transformer model."""
    dataset = TransformerDataset(texts, tokenizer, max_length=384)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = []
    probabilities = []
    disable_tqdm = not sys.stdout.isatty()
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", disable=disable_tqdm):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
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


def get_attention_based_importance(model, tokenizer, texts, labels, device, n_examples=5):
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
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions  # tuple of attention weights per layer
            
            # Get prediction
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
    
    # Load label mapping
    label_map_path = os.path.join(models_dir, 'label_mapping.json')
    if not os.path.exists(label_map_path):
        print(f"Label mapping not found at {label_map_path}.")
        return
    
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
        id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
    
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
    test_path = os.path.join(processed_dir, 'test.csv')
    if not os.path.exists(test_path):
        print(f"Test data not found at {test_path}.")
        return
    
    test_df = pd.read_csv(test_path)
    X_test = test_df['text'].astype(str).tolist()
    y_test = test_df['label'].astype(str).tolist()
    
    batch_size = int(os.getenv('BATCH_SIZE', '8'))
    
    print("Analyzing model explainability...")
    
    # Get predictions for all test samples
    print("Running predictions on test set...")
    y_pred = predict_with_transformer(model, tokenizer, X_test, device, batch_size, id2label)
    
    # Attention-based importance for sample texts
    print("Extracting attention-based token importance...")
    try:
        attention_results = get_attention_based_importance(model, tokenizer, X_test, y_test, device, n_examples=10)
        
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
