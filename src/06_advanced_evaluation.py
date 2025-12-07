import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure we can import helper utilities from incremental development
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from importlib import import_module

    inc_dev = import_module("04_incremental_model_development")
    FusionModel = inc_dev.FusionModel
    coral_predict = inc_dev.coral_predict
    extract_readability_features = inc_dev.extract_readability_features
except Exception as exc:  # pragma: no cover - defensive
    print(f"Warning: could not import from 04_incremental_model_development: {exc}")
    FusionModel = None
    coral_predict = None
    extract_readability_features = None


# ---------------------------------------------------------------------------
# Shared dataset + inference
# ---------------------------------------------------------------------------
class TransformerDataset(Dataset):
    """Dataset for transformer inference with optional readability features."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        max_length: int = 384,
        use_features: bool = False,
        stats: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = use_features
        self._precomputed_features = None

        if self.use_features and extract_readability_features is not None:
            raw_feats = [extract_readability_features(str(t)).numpy() for t in self.texts]
            if stats:
                mean = np.array(stats.get("mean", []))
                std = np.array(stats.get("std", [])) + 1e-6
                norm_feats = (np.stack(raw_feats) - mean) / std
                self._precomputed_features = torch.tensor(norm_feats, dtype=torch.float32)
            else:
                self._precomputed_features = torch.stack(
                    [extract_readability_features(str(t)) for t in self.texts]
                )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "text": text,
        }
        if self.use_features and self._precomputed_features is not None:
            item["readability_features"] = self._precomputed_features[idx]
        return item


def predict_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int = 8,
    id2label: Optional[Dict[int, str]] = None,
    use_fusion: bool = False,
    use_coral: bool = False,
    stats: Optional[Dict[str, List[float]]] = None,
    max_length: int = 384,
    return_probs: bool = False,
):
    """Run batched inference and optionally return probabilities."""

    dataset = TransformerDataset(
        texts, tokenizer, max_length=max_length, use_features=use_fusion, stats=stats
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    predictions: List[str] = []
    probabilities: List[List[float]] = []
    disable_tqdm = not sys.stdout.isatty()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            readability_features = batch.get("readability_features")
            if readability_features is not None:
                readability_features = readability_features.to(device)

            if use_fusion:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    readability_features=readability_features,
                )
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits

            if use_coral and coral_predict is not None:
                preds = coral_predict(logits)
                probs = torch.sigmoid(logits)
            else:
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)

            if id2label:
                predictions.extend([id2label[int(p)] for p in preds.cpu().numpy()])
            else:
                predictions.extend(preds.cpu().numpy().tolist())

            if return_probs:
                probabilities.extend(probs.cpu().numpy().tolist())

    if return_probs:
        return predictions, probabilities
    return predictions


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def _expand_numeric_labels(id2label: Dict[int, str], sample_labels: Sequence[str]):
    """Map numeric-only labels ("1".."5") back to original textual labels."""

    num_to_full: Dict[str, str] = {}
    for lbl in sample_labels:
        match = re.match(r"^([1-5])", str(lbl).strip())
        if match:
            n = match.group(1)
            num_to_full.setdefault(n, lbl)

    for k, v in list(id2label.items()):
        if re.fullmatch(r"[1-5]", str(v)) and v in num_to_full:
            id2label[k] = num_to_full[v]
    return id2label


def load_label_mapping(models_dir: Path) -> Dict[int, str]:
    label_map_path = models_dir / "label_mapping.json"
    baseline_map_path = models_dir / "baseline_label_mapping.json"
    mapping = None
    if label_map_path.exists():
        mapping = label_map_path
    elif baseline_map_path.exists():
        mapping = baseline_map_path
    if mapping is None:
        raise FileNotFoundError("No label mapping file found (label_mapping.json or baseline_label_mapping.json)")
    with open(mapping, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["id2label"].items()}


def load_model_bundle(models_dir: Path, device: torch.device):
    """Load tokenizer + model (+ fusion metadata) from disk."""

    model_dir = models_dir / "best_transformer_model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Transformer model not found at {model_dir}")

    id2label = load_label_mapping(models_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    checkpoint_path = model_dir / "pytorch_model.bin"
    use_fusion = False
    use_coral = False
    stats = None

    if checkpoint_path.exists() and FusionModel is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "use_feature_fusion" in checkpoint:
                use_fusion = bool(checkpoint.get("use_feature_fusion", False))
                use_coral = bool(checkpoint.get("use_coral", False))
                num_labels = int(checkpoint.get("num_labels", len(id2label)))

                if use_fusion:
                    print(f"Loading FusionModel (CORAL={use_coral})...")
                    base_model = AutoModel.from_pretrained(
                        os.getenv("TRANSFORMER_MODEL", "SZTAKI-HLT/hubert-base-cc")
                    )
                    model = FusionModel(base_model, num_classes=num_labels, use_coral=use_coral)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(device)

                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        stats = metadata.get("feature_stats") or metadata.get("stats")
                else:
                    raise ValueError("Checkpoint indicates fusion model but FusionModel class not available")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Could not load fusion checkpoint: {exc}; falling back to standard model")
            use_fusion = False

    if not use_fusion:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)

    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "id2label": id2label,
        "use_fusion": use_fusion,
        "use_coral": use_coral,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Robustness utilities
# ---------------------------------------------------------------------------
def add_noise_to_text(text: str, noise_level: float = 0.1) -> str:
    """Add simple character-level noise (delete/duplicate/space)."""
    if not text or noise_level <= 0:
        return text

    text_list = list(text)
    n_chars_to_modify = max(1, int(len(text_list) * noise_level))
    rng = np.random.default_rng()

    for _ in range(n_chars_to_modify):
        if not text_list:
            break
        idx = int(rng.integers(0, len(text_list)))
        action = rng.choice(["delete", "duplicate", "space"])
        if action == "delete":
            text_list.pop(idx)
        elif action == "duplicate" and idx < len(text_list):
            text_list.insert(idx, text_list[idx])
        else:
            text_list[idx] = " "

    return "".join(text_list)


def truncate_text(text: str, ratio: float = 0.5) -> str:
    """Keep the first `ratio` portion of the words."""
    if not text:
        return text
    words = text.split()
    n_words = max(1, int(len(words) * ratio))
    return " ".join(words[:n_words])


def evaluate_robustness(
    *,
    bundle: Dict,
    texts: Sequence[str],
    labels: Sequence[str],
    perturbations: Sequence[Dict],
    device: torch.device,
    batch_size: int,
    max_length: int,
    output_dir: Path,
):
    """Run multiple perturbation tests and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    id2label = bundle["id2label"]
    use_fusion = bundle["use_fusion"]
    use_coral = bundle["use_coral"]
    stats = bundle["stats"]

    results = []
    for test in perturbations:
        name = test["name"]
        func: Callable = test["func"]
        params = test.get("params", {})
        print(f"Running robustness test: {name} ({params})")
        transformed = [func(t, **params) for t in texts]
        y_pred = predict_batch(
            model,
            tokenizer,
            transformed,
            device,
            batch_size=batch_size,
            id2label=id2label,
            use_fusion=use_fusion,
            use_coral=use_coral,
            stats=stats,
            max_length=max_length,
        )

        labels_unique = sorted(list(set(labels) | set(y_pred)))
        report = classification_report(
            labels,
            y_pred,
            labels=labels_unique,
            output_dict=True,
            zero_division=0,
        )
        label2id_local = {label: idx for idx, label in enumerate(labels_unique)}
        y_true_idx = [label2id_local[l] for l in labels]
        y_pred_idx = [label2id_local[l] for l in y_pred]
        macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)

        results.append(
            {
                "test_name": name,
                "accuracy": float(accuracy_score(labels, y_pred)),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "classification_report": report,
                "transformation": params,
            }
        )

    json_path = output_dir / "robustness_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Robustness results saved to {json_path}")

    plot_robustness_results(results, output_dir / "robustness_accuracy.png")
    return results


def plot_robustness_results(results: Sequence[Dict], save_path: Path):
    """Plot bar chart of accuracies for perturbations."""
    if not results:
        return

    test_names = [r["test_name"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(test_names, accuracies, color="steelblue")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Robustness Tests")
    ax.set_ylim([0, 1])
    ax.axhline(y=accuracies[0], color="red", linestyle="--", label="Baseline (Original)")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha="center", va="bottom")

    plt.xticks(rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Robustness plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Explainability utilities
# ---------------------------------------------------------------------------
def get_attention_based_importance(
    *,
    model,
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[str],
    device: torch.device,
    n_examples: int,
    use_fusion: bool,
    use_coral: bool,
    stats: Optional[Dict[str, List[float]]],
    max_length: int,
):
    """Extract top attended tokens for a handful of examples."""

    results = []
    model.eval()
    for i, text in enumerate(texts[:n_examples]):
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask_tensor = encoding["attention_mask"].to(device)

        readability_features = None
        if use_fusion and extract_readability_features is not None:
            feat = extract_readability_features(text).unsqueeze(0)
            if stats:
                mean = torch.tensor(stats.get("mean", []), dtype=torch.float32)
                std = torch.tensor(stats.get("std", []), dtype=torch.float32)
                feat = (feat - mean) / (std + 1e-6)
            readability_features = feat.to(device)

        with torch.no_grad():
            if use_fusion:
                transformer_outputs = model.transformer(
                    input_ids=input_ids, attention_mask=attention_mask_tensor, output_attentions=True, return_dict=True
                )
                attentions = transformer_outputs.attentions
                hidden = transformer_outputs.last_hidden_state
                pooled = model._pool(hidden, attention_mask_tensor)
                if readability_features is not None:
                    feat_out = model.feature_branch(readability_features)
                else:
                    feat_out = torch.zeros(pooled.size(0), 32, device=pooled.device)
                fused = torch.cat([pooled, feat_out], dim=1)
                x = model.dropout(F.gelu(model.fusion_fc(fused)))
                logits = model.coral_head(x) if use_coral else model.classifier(x)
            else:
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask_tensor, output_attentions=True, return_dict=True
                )
                logits = outputs.logits
                attentions = outputs.attentions

            if use_coral and coral_predict is not None:
                pred_class = int(coral_predict(logits)[0])
                probs = torch.sigmoid(logits)[0]
            else:
                pred_class = int(torch.argmax(logits, dim=1))
                probs = torch.softmax(logits, dim=1)[0]

            avg_attention = torch.stack([att.mean(dim=1) for att in attentions]).mean(dim=0)[0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_scores = avg_attention.mean(dim=0).cpu().numpy()

            valid_tokens: List[str] = []
            valid_scores: List[float] = []
            for token, score in zip(tokens, attention_scores):
                if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                    valid_tokens.append(token)
                    valid_scores.append(float(score))

            top_indices = np.argsort(valid_scores)[-10:][::-1]
            top_tokens = [
                (valid_tokens[idx], valid_scores[idx]) for idx in top_indices if idx < len(valid_tokens)
            ]

            results.append(
                {
                    "example_id": i,
                    "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
                    "true_label": str(labels[i]),
                    "predicted_class_id": pred_class,
                    "prediction_probabilities": probs.cpu().numpy().tolist(),
                    "top_attended_tokens": top_tokens,
                }
            )

    return results


def analyze_misclassifications(y_true: Sequence[str], y_pred: Sequence[str], texts: Sequence[str]):
    misclassified = []
    for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
        if pred != true:
            misclassified.append(
                {
                    "index": idx,
                    "text": texts[idx][:200] + ("..." if len(texts[idx]) > 200 else ""),
                    "true_label": str(true),
                    "predicted_label": str(pred),
                }
            )

    confusion_pairs: Dict[Tuple[str, str], int] = {}
    for item in misclassified:
        pair = (item["true_label"], item["predicted_label"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    return {
        "total_misclassified": len(misclassified),
        "total_examples": len(texts),
        "error_rate": (len(misclassified) / len(texts)) if texts else 0.0,
        "confusion_pairs": [
            {"true_label": p[0], "predicted_label": p[1], "count": c}
            for p, c in sorted_pairs[:10]
        ],
        "examples": misclassified[:10],
    }


def summarize_attention(results: Sequence[Dict]):
    by_label: Dict[str, List[float]] = {}
    for r in results:
        tl = r.get("true_label")
        scores = [s for _, s in r.get("top_attended_tokens", [])]
        mean_score = float(np.mean(scores)) if scores else 0.0
        by_label.setdefault(tl, []).append(mean_score)
    return {k: float(np.mean(v)) for k, v in by_label.items()}


def plot_confusion_pairs(confusion_pairs: Sequence[Dict], save_path: Path, top_n: int = 10):
    if not confusion_pairs:
        return

    pairs = confusion_pairs[:top_n]
    pair_labels = [f"{p['true_label']} → {p['predicted_label']}" for p in pairs]
    counts = [p["count"] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pair_labels, counts, color="coral")
    ax.set_xlabel("Count")
    ax.set_title("Top Confusion Pairs - Transformer")
    ax.invert_yaxis()

    for i, count in enumerate(counts):
        ax.text(count, i, f" {count}", va="center")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Confusion pairs plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_robustness_flow(args, bundle, texts, labels, output_root: Path):
    perturbations = [
        {"name": "Original", "func": lambda x, **_: x, "params": {}},
        {"name": "Noise 5%", "func": add_noise_to_text, "params": {"noise_level": 0.05}},
        {"name": "Noise 10%", "func": add_noise_to_text, "params": {"noise_level": 0.10}},
        {"name": "Noise 20%", "func": add_noise_to_text, "params": {"noise_level": 0.20}},
        {"name": "Truncate 75%", "func": truncate_text, "params": {"ratio": 0.75}},
        {"name": "Truncate 50%", "func": truncate_text, "params": {"ratio": 0.50}},
        {"name": "Truncate 25%", "func": truncate_text, "params": {"ratio": 0.25}},
    ]

    robustness_dir = output_root / "advanced" / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # expand numeric labels if needed
    bundle["id2label"] = _expand_numeric_labels(bundle["id2label"], labels)

    evaluate_robustness(
        bundle=bundle,
        texts=texts,
        labels=labels,
        perturbations=perturbations,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=robustness_dir,
    )


def run_explainability_flow(args, bundle, texts, labels, output_root: Path):
    explain_dir = output_root / "advanced" / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)

    bundle["id2label"] = _expand_numeric_labels(bundle["id2label"], labels)

    print("Running predictions for explainability...")
    y_pred, _probs = predict_batch(
        bundle["model"],
        bundle["tokenizer"],
        texts,
        args.device,
        batch_size=args.batch_size,
        id2label=bundle["id2label"],
        use_fusion=bundle["use_fusion"],
        use_coral=bundle["use_coral"],
        stats=bundle["stats"],
        max_length=args.max_length,
        return_probs=False,
    ), None
    y_pred = y_pred  # for clarity

    print("Extracting attention-based importance...")
    try:
        attention_results = get_attention_based_importance(
            model=bundle["model"],
            tokenizer=bundle["tokenizer"],
            texts=texts,
            labels=labels,
            device=args.device,
            n_examples=min(args.sample_count, len(texts)),
            use_fusion=bundle["use_fusion"],
            use_coral=bundle["use_coral"],
            stats=bundle["stats"],
            max_length=args.max_length,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Attention extraction failed: {exc}")
        attention_results = []

    attention_path = explain_dir / "attention_importance.json"
    with open(attention_path, "w", encoding="utf-8") as f:
        json.dump(attention_results, f, ensure_ascii=False, indent=2)
    print(f"Attention importance saved to {attention_path}")

    summary = summarize_attention(attention_results)
    summary_path = explain_dir / "attention_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Attention summary saved to {summary_path}")

    print("Analyzing misclassifications...")
    misclass = analyze_misclassifications(labels, y_pred, texts)
    misclass_path = explain_dir / "misclassification_analysis.json"
    with open(misclass_path, "w", encoding="utf-8") as f:
        json.dump(misclass, f, ensure_ascii=False, indent=2)
    print(f"Misclassification analysis saved to {misclass_path}")

    if misclass.get("confusion_pairs"):
        plot_confusion_pairs(misclass["confusion_pairs"], explain_dir / "confusion_pairs.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced evaluation (robustness + explainability)")
    parser.add_argument(
        "--mode",
        choices=["robustness", "explainability", "both"],
        default=os.getenv("ADV_EVAL_MODE", "both"),
        help="Which evaluation to run",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "/app/output"),
        help="Base output directory",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing trained models (default: <output-dir>/models)",
    )
    parser.add_argument(
        "--data-csv",
        default=os.getenv("DATA_CSV", None),
        help="CSV with columns 'text' and 'label' (default: <output-dir>/processed/test.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "384")))
    parser.add_argument(
        "--sample-count",
        type=int,
        default=int(os.getenv("EXPLAIN_SAMPLES", "10")),
        help="How many samples to use for attention explanations",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force device (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_dir)
    models_dir = Path(args.models_dir) if args.models_dir else output_root / "models"
    data_csv = args.data_csv or str(output_root / "processed" / "test.csv")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.device = device
    print(f"Using device: {device}")

    print(f"Loading data from {data_csv}...")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Data CSV not found at {data_csv}")
    df = pd.read_csv(data_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Data CSV must contain 'text' and 'label' columns")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    print(f"Loading model bundle from {models_dir}...")
    bundle = load_model_bundle(models_dir, device)

    if args.mode in {"robustness", "both"}:
        run_robustness_flow(args, bundle, texts, labels, output_root)

    if args.mode in {"explainability", "both"}:
        run_explainability_flow(args, bundle, texts, labels, output_root)


if __name__ == "__main__":
    main()
