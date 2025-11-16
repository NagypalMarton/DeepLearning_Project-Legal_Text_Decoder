import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys


def add_noise_to_text(text, noise_level=0.1):
    """Add character-level noise to text by randomly modifying characters."""
    if not text or noise_level == 0:
        return text
    
    text_list = list(text)
    n_chars_to_modify = max(1, int(len(text_list) * noise_level))
    
    for _ in range(n_chars_to_modify):
        if len(text_list) > 0:
            idx = np.random.randint(0, len(text_list))
            # Random modification: delete, duplicate, or replace with space
            action = np.random.choice(['delete', 'duplicate', 'space'])
            if action == 'delete':
                text_list.pop(idx)
            elif action == 'duplicate' and idx < len(text_list):
                text_list.insert(idx, text_list[idx])
            else:
                text_list[idx] = ' '
    
    return ''.join(text_list)


def truncate_text(text, ratio=0.5):
    """Truncate text to a given ratio of its original length."""
    if not text:
        return text
    words = text.split()
    n_words = max(1, int(len(words) * ratio))
    return ' '.join(words[:n_words])


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


def predict_with_transformer(model, tokenizer, texts, device, batch_size=8, id2label=None):
    """Run batch inference with transformer model."""
    dataset = TransformerDataset(texts, tokenizer, max_length=384)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    predictions = []
    disable_tqdm = not sys.stdout.isatty()
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", disable=disable_tqdm, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            if id2label:
                predictions.extend([id2label[int(p)] for p in preds.cpu().numpy()])
            else:
                predictions.extend(preds.cpu().numpy().tolist())
    
    return predictions


def test_robustness(model, tokenizer, X_test, y_test, device, test_name, transformation_func, transformation_params, id2label, batch_size=8):
    """Test model robustness with a specific text transformation."""
    X_transformed = [transformation_func(text, **transformation_params) for text in X_test]
    y_pred = predict_with_transformer(model, tokenizer, X_transformed, device, batch_size, id2label)
    
    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(list(set(y_test) | set(y_pred)))
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # Compute label indices for macro/weighted F1
    label2id_local = {label: idx for idx, label in enumerate(labels)}
    y_test_idx = [label2id_local[y] for y in y_test]
    y_pred_idx = [label2id_local[y] for y in y_pred]
    
    macro_f1 = f1_score(y_test_idx, y_pred_idx, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test_idx, y_pred_idx, average='weighted', zero_division=0)
    
    return {
        'test_name': test_name,
        'accuracy': accuracy,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'classification_report': report,
        'transformation': transformation_params
    }


def plot_robustness_results(results, save_path):
    """Plot robustness test results."""
    test_names = [r['test_name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(test_names, accuracies, color='steelblue')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Robustness Tests')
    ax.set_ylim([0, 1])
    ax.axhline(y=accuracies[0], color='red', linestyle='--', label='Baseline (Original)')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    base_output = os.getenv('OUTPUT_DIR', '/app/output')
    processed_dir = os.path.join(base_output, 'processed')
    models_dir = os.path.join(base_output, 'models')
    robustness_dir = os.path.join(base_output, 'robustness')
    
    Path(robustness_dir).mkdir(parents=True, exist_ok=True)
    
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
    
    print("Running robustness tests...")
    
    # Define robustness tests
    tests = [
        {
            'name': 'Original',
            'func': lambda x: x,
            'params': {}
        },
        {
            'name': 'Noise 5%',
            'func': add_noise_to_text,
            'params': {'noise_level': 0.05}
        },
        {
            'name': 'Noise 10%',
            'func': add_noise_to_text,
            'params': {'noise_level': 0.10}
        },
        {
            'name': 'Noise 20%',
            'func': add_noise_to_text,
            'params': {'noise_level': 0.20}
        },
        {
            'name': 'Truncate 75%',
            'func': truncate_text,
            'params': {'ratio': 0.75}
        },
        {
            'name': 'Truncate 50%',
            'func': truncate_text,
            'params': {'ratio': 0.50}
        },
        {
            'name': 'Truncate 25%',
            'func': truncate_text,
            'params': {'ratio': 0.25}
        }
    ]
    
    results = []
    for test in tests:
        print(f"Running test: {test['name']}")
        result = test_robustness(
            model, tokenizer, X_test, y_test, device,
            test['name'],
            test['func'],
            test['params'],
            id2label,
            batch_size
        )
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}, Macro-F1: {result['macro_f1']:.4f}, Weighted-F1: {result['weighted_f1']:.4f}")
    
    # Save results
    results_path = os.path.join(robustness_dir, '06-robustness_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRobustness results saved to {results_path}")
    
    # Plot results
    plot_path = os.path.join(robustness_dir, '06-robustness_comparison.png')
    plot_robustness_results(results, plot_path)
    print(f"Robustness plot saved to {plot_path}")
    
    # Print summary
    print("\n=== Robustness Test Summary ===")
    for result in results:
        print(f"{result['test_name']:20s}: Accuracy = {result['accuracy']:.4f}, Macro-F1 = {result['macro_f1']:.4f}, Weighted-F1 = {result['weighted_f1']:.4f}")


if __name__ == '__main__':
    main()
