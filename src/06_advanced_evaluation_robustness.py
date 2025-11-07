import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def test_robustness(model, X_test, y_test, test_name, transformation_func, transformation_params):
    """Test model robustness with a specific text transformation."""
    X_transformed = [transformation_func(text, **transformation_params) for text in X_test]
    y_pred = model.predict(X_transformed)
    
    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(list(set(y_test) | set(y_pred)))
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    return {
        'test_name': test_name,
        'accuracy': accuracy,
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
    
    # Load model
    model_path = os.path.join(models_dir, 'baseline_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the baseline first.")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    test_path = os.path.join(processed_dir, 'test.csv')
    if not os.path.exists(test_path):
        print(f"Test data not found at {test_path}.")
        return
    
    test_df = pd.read_csv(test_path)
    X_test = test_df['text'].astype(str).tolist()
    y_test = test_df['label'].astype(str).tolist()
    
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
            model, X_test, y_test,
            test['name'],
            test['func'],
            test['params']
        )
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")
    
    # Save results
    results_path = os.path.join(robustness_dir, 'robustness_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRobustness results saved to {results_path}")
    
    # Plot results
    plot_path = os.path.join(robustness_dir, 'robustness_comparison.png')
    plot_robustness_results(results, plot_path)
    print(f"Robustness plot saved to {plot_path}")
    
    # Print summary
    print("\n=== Robustness Test Summary ===")
    for result in results:
        print(f"{result['test_name']:20s}: Accuracy = {result['accuracy']:.4f}")


if __name__ == '__main__':
    main()
