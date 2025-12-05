import json
import os
import glob
import re
from pathlib import Path
from typing import List, Union
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


def load_json_data(file_path: str):
    """Load data from a single JSON file and return its parsed content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        raise


def load_json_items(input_path: str) -> List[Union[dict, list]]:
    """Load items from a JSON file or from all JSON files in a directory.

    - If input_path is a directory: loads all *.json files and concatenates lists.
    - If input_path is a file: loads that JSON.
    Returns a list of items (dicts) with source file annotated.
    """
    if os.path.isdir(input_path):
        print(f"Loading all JSON files from directory: {input_path}")
        items: List[dict] = []
        json_files = sorted(glob.glob(os.path.join(input_path, '*.json')))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {input_path}")
        for fp in json_files:
            data = load_json_data(fp)
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict):
                        it = {**it, "__source_file__": os.path.basename(fp)}
                        items.append(it)
            elif isinstance(data, dict):
                data["__source_file__"] = os.path.basename(fp)
                items.append(data)
            else:
                print(f"Warning: Unsupported JSON root type in {fp}: {type(data)} — skipping")
        print(f"Total items loaded from directory: {len(items)}")
        return items
    elif os.path.isfile(input_path):
        print(f"Loading JSON data from file: {input_path}")
        data = load_json_data(input_path)
        if isinstance(data, list):
            items = []
            for it in data:
                if isinstance(it, dict):
                    it = {**it, "__source_file__": os.path.basename(input_path)}
                    items.append(it)
            return items
        elif isinstance(data, dict):
            data["__source_file__"] = os.path.basename(input_path)
            return [data]
        else:
            raise ValueError(f"Unsupported JSON root type in {input_path}: {type(data)}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


STEP_PREFIX = '01-acquisition'

def save_histogram(series: pd.Series, title: str, path: str, bins: int = 50):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series.values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ===== READABILITY METRICS =====

def count_syllables_hu(word: str) -> int:
    """Approximate syllable count for Hungarian words based on vowels."""
    vowels = 'aáeéiíoóöőuúüű'
    word = word.lower()
    syllable_count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    return max(1, syllable_count)


def flesch_reading_ease_hu(text: str) -> float:
    """Flesch Reading Ease adapted for Hungarian.
    Higher score = easier to read (0-100 scale).
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    total_syllables = sum(count_syllables_hu(w) for w in words)
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Adapted formula for Hungarian (similar to English but adjusted)
    score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    return max(0.0, min(100.0, score))


def gunning_fog_index(text: str) -> float:
    """Gunning Fog Index: years of education needed to understand text."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    # Complex words = 3+ syllables
    complex_words = sum(1 for w in words if count_syllables_hu(w) >= 3)
    avg_sentence_length = len(words) / len(sentences)
    percent_complex = (complex_words / len(words)) * 100
    
    fog = 0.4 * (avg_sentence_length + percent_complex)
    return fog


def smog_index(text: str) -> float:
    """SMOG Index: years of education needed."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    complex_words = sum(1 for w in words if count_syllables_hu(w) >= 3)
    smog = 1.0430 * np.sqrt(complex_words * (30 / len(sentences))) + 3.1291
    return smog


# ===== LEXICAL DIVERSITY METRICS =====

def type_token_ratio(text: str) -> float:
    """TTR: unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def moving_average_ttr(text: str, window_size: int = 100) -> float:
    """MATTR: average TTR over moving windows."""
    words = text.lower().split()
    if len(words) < window_size:
        return type_token_ratio(text)
    
    ttrs = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i+window_size]
        ttrs.append(len(set(window)) / len(window))
    
    return np.mean(ttrs) if ttrs else 0.0


def hapax_legomena_ratio(text: str) -> float:
    """Ratio of words that appear only once."""
    words = text.lower().split()
    if not words:
        return 0.0
    word_counts = Counter(words)
    hapax = sum(1 for count in word_counts.values() if count == 1)
    return hapax / len(words)


# ===== ADVANCED EDA FUNCTIONS =====

def compute_readability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add readability metrics to dataframe."""
    df = df.copy()
    texts = df['text_raw'].astype(str)
    
    print("Computing readability metrics...")
    df['flesch_score'] = texts.apply(flesch_reading_ease_hu)
    df['fog_index'] = texts.apply(gunning_fog_index)
    df['smog_index'] = texts.apply(smog_index)
    
    return df


def compute_lexical_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Add lexical diversity metrics to dataframe."""
    df = df.copy()
    texts = df['text_raw'].astype(str)
    
    print("Computing lexical diversity metrics...")
    df['ttr'] = texts.apply(type_token_ratio)
    df['mattr'] = texts.apply(moving_average_ttr)
    df['hapax_ratio'] = texts.apply(hapax_legomena_ratio)
    
    return df


def plot_metrics_by_label(df: pd.DataFrame, features_dir: str):
    """Plot readability and lexical metrics grouped by label."""
    if 'label_raw' not in df.columns:
        print("Skipping label-based plots: no label_raw column")
        return
    
    metrics = ['flesch_score', 'fog_index', 'smog_index', 'ttr', 'mattr', 'hapax_ratio']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return
    
    # Box plots for each metric by label
    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot = df[df[metric].notna()].copy()
        labels_sorted = sorted(df_plot['label_raw'].unique())
        
        data_to_plot = [df_plot[df_plot['label_raw'] == label][metric].values 
                        for label in labels_sorted]
        
        ax.boxplot(data_to_plot, tick_labels=labels_sorted, patch_artist=True)
        ax.set_title(f'{metric.replace("_", " ").title()} by Label')
        ax.set_xlabel('Label')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        
        out_path = os.path.join(features_dir, f'{STEP_PREFIX}_{metric}_by_label.png')
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {metric} by label plot -> {out_path}")


def compute_tfidf_top_words(df: pd.DataFrame, features_dir: str, top_n: int = 20):
    """Compute and save top TF-IDF words per label."""
    if 'label_raw' not in df.columns:
        print("Skipping TF-IDF: no label_raw column")
        return
    
    print("Computing TF-IDF top words per label...")
    
    labels = df['label_raw'].unique()
    results = []
    
    for label in sorted(labels):
        label_texts = df[df['label_raw'] == label]['text_raw'].astype(str).tolist()
        if not label_texts:
            continue
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), 
                                     min_df=2, max_df=0.8)
        try:
            tfidf_matrix = vectorizer.fit_transform(label_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            
            top_words = [(feature_names[i], avg_scores[i]) for i in top_indices]
            results.append({
                'label': label,
                'top_words': ', '.join([f"{word}({score:.3f})" for word, score in top_words[:10]])
            })
        except Exception as e:
            print(f"TF-IDF failed for label {label}: {e}")
    
    if results:
        df_tfidf = pd.DataFrame(results)
        out_csv = os.path.join(features_dir, f'{STEP_PREFIX}_tfidf_top_words_by_label.csv')
        df_tfidf.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"Saved TF-IDF results -> {out_csv}")


def plot_correlation_matrix(df: pd.DataFrame, features_dir: str):
    """Plot correlation heatmap of numerical features including label."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Encode label as numeric if present
    if 'label_raw' in df.columns:
        df_corr = df.copy()
        # Extract numeric part from label (e.g., "1-Nagyon nehezen érthető" -> 1)
        df_corr['label_numeric'] = df_corr['label_raw'].astype(str).str.extract(r'(\d+)')[0].astype(float)
        numeric_cols.append('label_numeric')
    else:
        df_corr = df.copy()
    
    # Filter to relevant metrics
    relevant = [col for col in numeric_cols if col in df_corr.columns]
    if len(relevant) < 2:
        print("Not enough numeric columns for correlation matrix")
        return
    
    corr_matrix = df_corr[relevant].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, linewidths=0.5)
    except:
        # Fallback if seaborn not available
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(relevant)))
        ax.set_yticks(range(len(relevant)))
        ax.set_xticklabels(relevant, rotation=45, ha='right')
        ax.set_yticklabels(relevant)
        plt.colorbar(im, ax=ax)
    
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    
    out_path = os.path.join(features_dir, f'{STEP_PREFIX}_correlation_matrix.png')
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved correlation matrix -> {out_path}")


def plot_top_confusion_pairs(df: pd.DataFrame, features_dir: str):
    """Plot top confusion pairs: szomszédos címkék közötti potenciális összekeverés.
    
    A tisztított adatokon alapul (szűrés után), és a címkék numerikus
    távolsága alapján azonosítja a leggyakoribb szomszédos párokat.
    Pl. '4-Érthető' vs '5-Könnyen érthető' gyakori keveredés lehet.
    """
    if 'label_raw' not in df.columns:
        print("Skipping confusion pairs: no label_raw column")
        return
    
    # Numerikus címke értékek kinyerése (1-5 skála)
    df_analysis = df.copy()
    df_analysis['label_numeric'] = df_analysis['label_raw'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    
    # Címke párok számlálása: szomszédos címkék (távolság = 1)
    label_pairs = []
    for label_num in sorted(df_analysis['label_numeric'].dropna().unique()):
        neighbor = label_num + 1
        if neighbor in df_analysis['label_numeric'].values:
            # Mindkét címke szövegének megtalálása
            label_1 = df_analysis[df_analysis['label_numeric'] == label_num]['label_raw'].iloc[0]
            label_2 = df_analysis[df_analysis['label_numeric'] == neighbor]['label_raw'].iloc[0]
            count_1 = len(df_analysis[df_analysis['label_numeric'] == label_num])
            count_2 = len(df_analysis[df_analysis['label_numeric'] == neighbor])
            # Potenciális confusion: mindkét címke gyakorisága
            avg_count = (count_1 + count_2) / 2
            label_pairs.append((f"{label_1} ↔ {label_2}", avg_count))
    
    if not label_pairs:
        print("No confusion pairs to plot")
        return
    
    # Top 5 leggyakoribb szomszédos pár
    label_pairs = sorted(label_pairs, key=lambda x: x[1], reverse=True)[:5]
    pair_names, pair_counts = zip(*label_pairs)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(pair_names, pair_counts, color='#F58518')
    ax.set_xlabel('Átlagos mintaszám (potenciális összekeverés)')
    ax.set_title('Top Confusion Pairs (szomszédos címkék)')
    fig.tight_layout()
    
    out_path = os.path.join(features_dir, f'{STEP_PREFIX}_top_confusion_pairs.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved top confusion pairs -> {out_path}")


def raw_eda(df: pd.DataFrame, features_dir: str):
    """Compute and save simple EDA plots on RAW text (no cleaning)."""
    texts = df.get('text_raw', pd.Series([], dtype=str)).astype(str)
    word_counts = texts.apply(lambda t: len(t.split()))
    avg_word_len = texts.apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0.0)

    Path(features_dir).mkdir(parents=True, exist_ok=True)
    save_histogram(word_counts, 'RAW Word Count Distribution', os.path.join(features_dir, f'{STEP_PREFIX}_raw_word_count_hist.png'))
    save_histogram(avg_word_len, 'RAW Average Word Length Distribution', os.path.join(features_dir, f'{STEP_PREFIX}_raw_avg_word_len_hist.png'))


def extract_label_from_annotations(annotations_raw: str):
    """Best-effort extraction of label from Label Studio-like annotations JSON.

    Mirrors step 02 logic so that EDA can compute label distribution without
    changing the downstream raw CSV. Returns a string label or None.
    """
    if not annotations_raw:
        return None
    try:
        annotations = json.loads(annotations_raw)
        if annotations and isinstance(annotations, list) and len(annotations) > 0:
            return annotations[0]['result'][0]['value']['choices'][0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    return None


def eda_label_analysis(df: pd.DataFrame, raw_dir: str, features_dir: str):
    """Perform EDA-only label checks:

    - kinyeri a labelt az annotációból (label_raw)
    - üres értékeket felderíti (nem törli a sort)
    - label eloszlást, metrikákat számol
    - csak EDA: nem tisztít, nem deduplikál, nem módosítja a RAW állományt
    """
    if 'text_raw' not in df.columns:
        print("EDA label analysis skipped: 'text_raw' column missing.")
        return

    # Create label column for EDA from raw annotations
    df = df.copy()
    df['label_raw'] = df.get('annotations_raw', '').apply(extract_label_from_annotations)

    total_rows = len(df)
    missing_mask = (df['label_raw'].isna() | (df['label_raw'] == '')) | (df['text_raw'].isna() | (df['text_raw'] == ''))
    missing_count = int(missing_mask.sum())

    print(f"EDA label analysis: rows={total_rows}, missing_label_or_text={missing_count}")

    # Save statistics to file (informational, no rows are dropped)
    stats_file = os.path.join(features_dir, f'{STEP_PREFIX}_raw_eda_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("RAW EDA Statisztikák\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Összes sor: {total_rows}\n")
        f.write(f"Hiányzó label VAGY üres text: {missing_count}\n")
    print(f"Saved EDA statistics -> {stats_file}")

    # Plot label distribution
    try:
        counts = df['label_raw'].value_counts().sort_values(ascending=False)
        if len(counts) == 0:
            print("No labels available for distribution plot.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind='bar', ax=ax, color='#4C78A8')
            ax.set_title('Besorolás eloszlás (RAW EDA)')
            ax.set_xlabel('Besorolás (label)')
            ax.set_ylabel('Darab')
            fig.tight_layout()
            out_fig = os.path.join(features_dir, f'{STEP_PREFIX}_raw_label_distribution.png')
            fig.savefig(out_fig)
            plt.close(fig)
            print(f"Saved label distribution plot -> {out_fig}")
    except Exception as e:
        print(f"Failed to generate label distribution plot: {e}")

    # ===== ADVANCED STATISTICAL ANALYSIS =====
    print("\n" + "="*50)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("="*50)
    
    # Compute readability metrics
    try:
        df = compute_readability_metrics(df)
        print("✓ Readability metrics computed")
    except Exception as e:
        print(f"✗ Readability metrics failed: {e}")
    
    # Compute lexical diversity
    try:
        df = compute_lexical_diversity(df)
        print("✓ Lexical diversity metrics computed")
    except Exception as e:
        print(f"✗ Lexical diversity failed: {e}")
    
    # Plot metrics by label
    try:
        plot_metrics_by_label(df, features_dir)
        print("✓ Metrics by label plots saved")
    except Exception as e:
        print(f"✗ Metrics by label plots failed: {e}")
    
    # TF-IDF top words per label
    try:
        compute_tfidf_top_words(df, features_dir)
        print("✓ TF-IDF analysis completed")
    except Exception as e:
        print(f"✗ TF-IDF analysis failed: {e}")
    
    # Correlation matrix
    try:
        plot_correlation_matrix(df, features_dir)
        print("✓ Correlation matrix saved")
    except Exception as e:
        print(f"✗ Correlation matrix failed: {e}")
    
    # Top confusion pairs analysis (label co-occurrence)
    try:
        plot_top_confusion_pairs(df, features_dir)
        print("✓ Top confusion pairs plot saved")
    except Exception as e:
        print(f"✗ Top confusion pairs failed: {e}")
    
    # Save enhanced EDA dataset with all metrics
    try:
        out_csv_enhanced = os.path.join(raw_dir, 'raw_dataset_eda_enhanced.csv')  # keep original name for downstream compatibility
        df_enhanced = df.drop(columns=['row_id'], errors='ignore')
        df_enhanced.to_csv(out_csv_enhanced, index=False, encoding='utf-8-sig')
        print(f"✓ Saved enhanced EDA dataset with all metrics -> {out_csv_enhanced}")
    except Exception as e:
        print(f"✗ Failed to save enhanced dataset: {e}")
    
    print("="*50)
    print("ADVANCED ANALYSIS COMPLETE")
    print("="*50 + "\n")


def process_raw_data(input_path: str, raw_dir: str, features_dir: str):
    """Aggregate RAW JSON items and persist a lightweight snapshot for downstream steps.

    - No cleaning, label extraction or splitting here.
    - Writes: raw_dir/raw_dataset.csv with columns: text_raw, annotations_raw (JSON), source_file
    - Creates basic RAW EDA plots under features_dir
    """
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(features_dir).mkdir(parents=True, exist_ok=True)

    print("Loading JSON data...")
    data_items = load_json_items(input_path)

    records = []
    for item in data_items:
        text = ''
        annotations = None
        source = ''
        if isinstance(item, dict):
            text = item.get('data', {}).get('text', '')
            annotations = item.get('annotations')
            source = item.get('__source_file__', '')
        records.append({
            'text_raw': text if text is not None else '',
            'annotations_raw': json.dumps(annotations, ensure_ascii=False) if annotations is not None else '',
            'source_file': source
        })

    df = pd.DataFrame(records)
    print(f"RAW rows aggregated: {len(df)}")

    raw_csv = os.path.join(raw_dir, 'raw_dataset.csv')  # keep canonical name
    df.to_csv(raw_csv, index=False, encoding='utf-8-sig')
    print(f"Saved RAW dataset -> {raw_csv}")

    # RAW EDA
    try:
        raw_eda(df, features_dir)
        print(f"Saved RAW EDA plots to {features_dir}")
    except Exception as e:
        print(f"RAW EDA failed (continuing): {e}")

    # Additional RAW EDA: duplicate removal, missing-label filtering, label distribution
    try:
        eda_label_analysis(df, raw_dir, features_dir)
    except Exception as e:
        print(f"EDA label analysis failed (continuing): {e}")


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR', '/app/data')
    base_output = os.getenv('OUTPUT_DIR', '/app/output')

    raw_dir = os.path.join(base_output, 'raw')
    features_dir = os.path.join(base_output, 'reports')

    print(f"Input path: {data_dir}")
    print(f"RAW output: {raw_dir}")
    process_raw_data(data_dir, raw_dir, features_dir)