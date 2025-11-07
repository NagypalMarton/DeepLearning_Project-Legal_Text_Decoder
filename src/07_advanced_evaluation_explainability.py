import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_top_features_per_class(model, feature_names, top_n=20):
    """Extract top features (words) for each class from a linear model."""
    results = {}
    
    try:
        # For Pipeline with LogisticRegression
        if hasattr(model, 'named_steps'):
            lr_model = model.named_steps.get('lr')
            if lr_model is None:
                print("No 'lr' step found in pipeline")
                return results
        else:
            lr_model = model
        
        if not hasattr(lr_model, 'coef_'):
            print("Model does not have coefficients (not a linear model)")
            return results
        
        coef = lr_model.coef_
        classes = lr_model.classes_ if hasattr(lr_model, 'classes_') else range(coef.shape[0])
        
        for idx, class_label in enumerate(classes):
            # Get coefficients for this class
            if coef.ndim == 1:
                # Binary classification
                class_coef = coef
            else:
                # Multi-class classification
                class_coef = coef[idx]
            
            # Get top positive coefficients (most indicative)
            top_indices = np.argsort(class_coef)[-top_n:][::-1]
            top_features = [(feature_names[i], class_coef[i]) for i in top_indices]
            
            results[str(class_label)] = {
                'top_features': top_features,
                'top_indices': top_indices.tolist()
            }
    
    except Exception as e:
        print(f"Error extracting features: {e}")
    
    return results


def plot_top_features(feature_importance, save_path, max_classes=5):
    """Plot top features for each class."""
    n_classes = min(len(feature_importance), max_classes)
    
    if n_classes == 0:
        print("No feature importance data to plot")
        return
    
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 4 * n_classes))
    
    if n_classes == 1:
        axes = [axes]
    
    for idx, (class_label, data) in enumerate(list(feature_importance.items())[:max_classes]):
        ax = axes[idx]
        features = data['top_features'][:15]  # Top 15
        
        words = [f[0] for f in features]
        scores = [f[1] for f in features]
        
        y_pos = np.arange(len(words))
        ax.barh(y_pos, scores, align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'Top Features for Class: {class_label}')
        ax.grid(axis='x', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Feature importance plot saved to {save_path}")


def explain_predictions(model, texts, labels, feature_names, n_examples=5):
    """Explain predictions for sample texts."""
    explanations = []
    
    for i in range(min(n_examples, len(texts))):
        text = texts[i]
        true_label = labels[i]
        
        # Get prediction
        pred_label = model.predict([text])[0]
        
        # Get prediction probabilities if available
        try:
            proba = model.predict_proba([text])[0]
            classes = model.classes_ if hasattr(model, 'classes_') else list(range(len(proba)))
            
            # Get top 3 predicted classes with probabilities
            top_indices = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {'class': str(classes[idx]), 'probability': float(proba[idx])}
                for idx in top_indices
            ]
        except Exception:
            top_predictions = [{'class': str(pred_label), 'probability': 1.0}]
        
        explanation = {
            'example_id': i,
            'text': text[:200] + ('...' if len(text) > 200 else ''),
            'true_label': str(true_label),
            'predicted_label': str(pred_label),
            'top_predictions': top_predictions,
            'correct': str(pred_label) == str(true_label)
        }
        
        explanations.append(explanation)
    
    return explanations


def analyze_misclassifications(model, X_test, y_test, feature_importance):
    """Analyze common misclassification patterns."""
    predictions = model.predict(X_test)
    
    # Find misclassified examples
    misclassified = []
    for i, (pred, true) in enumerate(zip(predictions, y_test)):
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


def main():
    base_output = os.getenv('OUTPUT_DIR', '/app/output')
    processed_dir = os.path.join(base_output, 'processed')
    models_dir = os.path.join(base_output, 'models')
    explainability_dir = os.path.join(base_output, 'explainability')
    
    Path(explainability_dir).mkdir(parents=True, exist_ok=True)
    
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
    
    print("Analyzing model explainability...")
    
    # Extract feature names from TF-IDF vectorizer
    try:
        if hasattr(model, 'named_steps'):
            tfidf = model.named_steps.get('tfidf')
            if tfidf is not None:
                feature_names = tfidf.get_feature_names_out().tolist()
            else:
                print("No TF-IDF vectorizer found in pipeline")
                feature_names = []
        else:
            feature_names = []
    except Exception as e:
        print(f"Error extracting feature names: {e}")
        feature_names = []
    
    # Get top features per class
    print("Extracting top features per class...")
    feature_importance = get_top_features_per_class(model, feature_names, top_n=20)
    
    # Save feature importance
    importance_path = os.path.join(explainability_dir, 'feature_importance.json')
    with open(importance_path, 'w', encoding='utf-8') as f:
        # Convert to serializable format
        serializable_importance = {}
        for class_label, data in feature_importance.items():
            serializable_importance[class_label] = {
                'top_features': [(word, float(score)) for word, score in data['top_features']],
                'top_indices': data['top_indices']
            }
        json.dump(serializable_importance, f, ensure_ascii=False, indent=2)
    print(f"Feature importance saved to {importance_path}")
    
    # Plot top features
    if feature_importance:
        plot_path = os.path.join(explainability_dir, 'top_features_per_class.png')
        plot_top_features(feature_importance, plot_path)
    
    # Explain sample predictions
    print("Generating prediction explanations...")
    explanations = explain_predictions(model, X_test, y_test, feature_names, n_examples=10)
    
    explanations_path = os.path.join(explainability_dir, 'prediction_explanations.json')
    with open(explanations_path, 'w', encoding='utf-8') as f:
        json.dump(explanations, f, ensure_ascii=False, indent=2)
    print(f"Prediction explanations saved to {explanations_path}")
    
    # Analyze misclassifications
    print("Analyzing misclassifications...")
    misclass_analysis = analyze_misclassifications(model, X_test, y_test, feature_importance)
    
    misclass_path = os.path.join(explainability_dir, 'misclassification_analysis.json')
    with open(misclass_path, 'w', encoding='utf-8') as f:
        json.dump(misclass_analysis, f, ensure_ascii=False, indent=2)
    print(f"Misclassification analysis saved to {misclass_path}")
    
    # Print summary
    print("\n=== Explainability Summary ===")
    print(f"Total test examples: {misclass_analysis['total_examples']}")
    print(f"Misclassified: {misclass_analysis['total_misclassified']}")
    print(f"Error rate: {misclass_analysis['error_rate']:.4f}")
    
    if feature_importance:
        print(f"\nAnalyzed {len(feature_importance)} classes")
        print("Top confusion pairs:")
        for pair in misclass_analysis['confusion_pairs'][:5]:
            print(f"  {pair['true_label']} -> {pair['predicted_label']}: {pair['count']} times")


if __name__ == '__main__':
    main()
