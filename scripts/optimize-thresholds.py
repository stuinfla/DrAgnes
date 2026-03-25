#!/usr/bin/env python3
"""
ADR-123: Per-Class Threshold Optimization for Mela v2 Combined Model
========================================================================

Loads the v2 ViT classifier, runs inference on the ISIC 2019 15% stratified
holdout (seed=42, matching train-combined.py), then computes per-class optimal
decision thresholds using ROC-curve analysis with clinically-motivated
constraints:

  - Melanoma (mel):  maximize sensitivity subject to specificity >= 85%
  - BCC, akiec:      maximize sensitivity subject to specificity >= 90%
  - Benign classes:  maximize specificity subject to sensitivity >= 70%

Outputs:
  scripts/threshold-optimization-results.json  -- full results + ROC data
  scripts/optimal-thresholds.json              -- thresholds for TS classifier

Usage:
    python scripts/optimize-thresholds.py

Hardware: Apple M3 Max (MPS backend).
"""

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTImageProcessor

# ---------------------------------------------------------------------------
# Suppress noisy warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tokenizer.*")

# ---------------------------------------------------------------------------
# Configuration (matches train-combined.py exactly)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "mela-classifier-v2" / "best"
RESULTS_PATH = SCRIPT_DIR / "threshold-optimization-results.json"
THRESHOLDS_PATH = SCRIPT_DIR / "optimal-thresholds.json"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
MEL_IDX = CLASS_NAMES.index("mel")  # 4
CANCER_INDICES = {0, 1, 4}  # akiec, bcc, mel

# ISIC 2019 label mapping (matches train-combined.py)
ISIC_INT_TO_HAM = {
    0: 0,  # AK -> akiec
    1: 1,  # BCC -> bcc
    2: 2,  # BKL -> bkl
    3: 3,  # DF -> df
    4: 4,  # MEL -> mel
    5: 5,  # NV -> nv
    6: 0,  # SCC -> akiec
    7: 6,  # VASC -> vasc
}
ISIC_LABEL_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

TEST_SIZE = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 32

# Threshold optimization constraints
CANCER_HIGH_RISK = {"mel"}           # spec >= 85%, maximize sens
CANCER_MODERATE = {"bcc", "akiec"}   # spec >= 90%, maximize sens
BENIGN_CLASSES = {"bkl", "df", "nv", "vasc"}  # sens >= 70%, maximize spec

MIN_SPEC_MEL = 0.85
MIN_SPEC_CANCER = 0.90
MIN_SENS_BENIGN = 0.70


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load v2 model and processor, move to MPS if available."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model from {MODEL_DIR}")
    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(device)
    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model.eval()
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor, device


# ---------------------------------------------------------------------------
# Data loading — reproduce exact ISIC 2019 test split from train-combined.py
# ---------------------------------------------------------------------------

def load_isic_test_set():
    """
    Load the ISIC 2019 dataset, map labels to HAM10000 7-class taxonomy,
    and reproduce the exact 15% stratified holdout used during training.
    Returns (images, labels) for the test split only.
    """
    print("\nLoading ISIC 2019 dataset...")
    ds = load_dataset("akinsanyaayomide/skin_cancer_dataset_balanced_labels_2")
    print(f"  Splits: {list(ds.keys())}")

    # Combine all splits (matching train-combined.py)
    all_splits = []
    for split_name in ds:
        all_splits.append(ds[split_name])
        print(f"    {split_name}: {len(ds[split_name])} examples")
    isic_full = concatenate_datasets(all_splits)
    print(f"  Total ISIC samples: {len(isic_full)}")

    # Extract labels and map to HAM taxonomy
    labels_raw = isic_full["label"]
    try:
        label_feature = isic_full.features["label"]
        if hasattr(label_feature, "names"):
            label_names = label_feature.names
            print(f"  Label names from dataset: {label_names}")
        else:
            label_names = None
    except Exception:
        label_names = None

    isic_labels = []
    for raw_lbl in labels_raw:
        if isinstance(raw_lbl, (int, np.integer)):
            ham_idx = ISIC_INT_TO_HAM.get(int(raw_lbl), -1)
        else:
            ham_idx = -1
        isic_labels.append(ham_idx)

    valid = sum(1 for l in isic_labels if l >= 0)
    print(f"  Mapped {valid}/{len(isic_labels)} labels to HAM taxonomy")

    # Reproduce the exact stratified split (seed=42, test_size=0.15)
    isic_indices = list(range(len(isic_full)))
    _, isic_test_idx = train_test_split(
        isic_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=isic_labels,
    )

    print(f"  Test set size: {len(isic_test_idx)} images")

    # Extract test images and labels
    test_labels = [isic_labels[i] for i in isic_test_idx]

    # Print class distribution
    dist = Counter(test_labels)
    print("\n  Test set class distribution:")
    for cls_idx in range(NUM_CLASSES):
        count = dist.get(cls_idx, 0)
        print(f"    {CLASS_NAMES[cls_idx]:>6}: {count:>4} ({100*count/len(test_labels):5.1f}%)")

    # Load images (batch-friendly — extract from dataset by index)
    print("\n  Loading test images...")
    test_images = []
    for i, idx in enumerate(isic_test_idx):
        row = isic_full[idx]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        test_images.append(img)
        if (i + 1) % 500 == 0:
            print(f"    Loaded {i+1}/{len(isic_test_idx)} images...")

    print(f"  Loaded all {len(test_images)} test images")
    return test_images, test_labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, device, images):
    """Run batch inference, return softmax probability matrix (N, 7)."""
    print(f"\nRunning inference on {len(images)} images (batch_size={BATCH_SIZE})...")
    all_probs = []
    t0 = time.time()

    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i:i + BATCH_SIZE]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

        done = min(i + BATCH_SIZE, len(images))
        if done % 500 < BATCH_SIZE or done == len(images):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done:>5}/{len(images)}  ({rate:.1f} img/s)")

    elapsed = time.time() - t0
    print(f"  Inference complete: {elapsed:.1f}s ({len(images)/elapsed:.1f} img/s)")
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Default argmax metrics
# ---------------------------------------------------------------------------

def compute_argmax_metrics(y_true, probs):
    """Compute metrics using standard argmax classification."""
    y_pred = np.argmax(probs, axis=1)
    y_true_arr = np.array(y_true)

    metrics = {"overall_accuracy": float(np.mean(y_pred == y_true_arr))}

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(np.sum((y_pred == c) & (y_true_arr == c)))
        fn = int(np.sum((y_pred != c) & (y_true_arr == c)))
        fp = int(np.sum((y_pred == c) & (y_true_arr != c)))
        tn = int(np.sum((y_pred != c) & (y_true_arr != c)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "support": int(tp + fn),
        }

    metrics["per_class"] = per_class
    metrics["melanoma_sensitivity"] = per_class["mel"]["sensitivity"]
    metrics["melanoma_specificity"] = per_class["mel"]["specificity"]

    # All-cancer sensitivity: fraction of any-cancer samples correctly called cancer
    cancer_true = np.isin(y_true_arr, list(CANCER_INDICES))
    cancer_pred = np.isin(y_pred, list(CANCER_INDICES))
    cancer_tp = int(np.sum(cancer_true & cancer_pred))
    cancer_fn = int(np.sum(cancer_true & ~cancer_pred))
    metrics["all_cancer_sensitivity"] = round(
        cancer_tp / (cancer_tp + cancer_fn) if (cancer_tp + cancer_fn) > 0 else 0.0, 4
    )

    return metrics


# ---------------------------------------------------------------------------
# Per-class ROC computation and threshold optimization
# ---------------------------------------------------------------------------

def compute_roc_curves(y_true, probs):
    """Compute ROC curves for each class (one-vs-rest)."""
    y_true_arr = np.array(y_true)
    roc_data = {}

    for c in range(NUM_CLASSES):
        binary_true = (y_true_arr == c).astype(int)
        fpr, tpr, thresholds = roc_curve(binary_true, probs[:, c])
        roc_auc = auc(fpr, tpr)
        roc_data[CLASS_NAMES[c]] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
        }
    return roc_data


def find_optimal_threshold(roc_data_class, class_name):
    """
    Find the optimal threshold for a given class based on clinical constraints.

    Melanoma: maximize sensitivity s.t. specificity >= 85%
    BCC, akiec: maximize sensitivity s.t. specificity >= 90%
    Benign: maximize specificity s.t. sensitivity >= 70%
    """
    fpr = roc_data_class["fpr"]
    tpr = roc_data_class["tpr"]
    thresholds = roc_data_class["thresholds"]

    # specificity = 1 - fpr, sensitivity = tpr
    spec = 1.0 - fpr
    sens = tpr

    if class_name in CANCER_HIGH_RISK:
        # Melanoma: maximize sensitivity where specificity >= MIN_SPEC_MEL
        constraint = "specificity >= {:.0f}%".format(MIN_SPEC_MEL * 100)
        mask = spec >= MIN_SPEC_MEL
        if not np.any(mask):
            # Fall back: pick the point closest to the constraint
            best_idx = np.argmin(np.abs(spec - MIN_SPEC_MEL))
        else:
            # Among points meeting constraint, pick highest sensitivity
            candidates = np.where(mask)[0]
            best_idx = candidates[np.argmax(sens[candidates])]

    elif class_name in CANCER_MODERATE:
        # BCC/akiec: maximize sensitivity where specificity >= MIN_SPEC_CANCER
        constraint = "specificity >= {:.0f}%".format(MIN_SPEC_CANCER * 100)
        mask = spec >= MIN_SPEC_CANCER
        if not np.any(mask):
            best_idx = np.argmin(np.abs(spec - MIN_SPEC_CANCER))
        else:
            candidates = np.where(mask)[0]
            best_idx = candidates[np.argmax(sens[candidates])]

    else:
        # Benign: maximize specificity where sensitivity >= MIN_SENS_BENIGN
        constraint = "sensitivity >= {:.0f}%".format(MIN_SENS_BENIGN * 100)
        mask = sens >= MIN_SENS_BENIGN
        if not np.any(mask):
            best_idx = np.argmin(np.abs(sens - MIN_SENS_BENIGN))
        else:
            candidates = np.where(mask)[0]
            best_idx = candidates[np.argmax(spec[candidates])]

    # Clamp threshold to [0, 1] — roc_curve can return values slightly outside
    optimal_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))

    return {
        "threshold": round(optimal_threshold, 4),
        "sensitivity_at_threshold": round(float(sens[best_idx]), 4),
        "specificity_at_threshold": round(float(spec[best_idx]), 4),
        "constraint": constraint,
        "auc": round(roc_data_class["auc"], 4),
    }


# ---------------------------------------------------------------------------
# Apply optimized thresholds for classification
# ---------------------------------------------------------------------------

def classify_with_thresholds(probs, thresholds_dict):
    """
    Classify using per-class thresholds with a priority-based decision scheme.

    For each sample:
    1. Compute score = prob / threshold for each class
    2. Among classes where prob >= threshold, pick the one with highest score
    3. If no class meets its threshold, fall back to argmax
    """
    n = probs.shape[0]
    y_pred = np.zeros(n, dtype=int)
    thresholds_arr = np.array([thresholds_dict[CLASS_NAMES[c]] for c in range(NUM_CLASSES)])

    for i in range(n):
        p = probs[i]
        # Compute ratio of probability to threshold
        ratios = p / thresholds_arr

        # Which classes exceed their threshold?
        above = p >= thresholds_arr
        if np.any(above):
            # Among those above threshold, pick highest ratio
            masked_ratios = np.where(above, ratios, -1.0)
            y_pred[i] = int(np.argmax(masked_ratios))
        else:
            # Fallback: argmax
            y_pred[i] = int(np.argmax(p))

    return y_pred


def compute_threshold_metrics(y_true, probs, thresholds_dict):
    """Compute metrics using per-class optimized thresholds."""
    y_pred = classify_with_thresholds(probs, thresholds_dict)
    y_true_arr = np.array(y_true)

    metrics = {"overall_accuracy": float(np.mean(y_pred == y_true_arr))}

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(np.sum((y_pred == c) & (y_true_arr == c)))
        fn = int(np.sum((y_pred != c) & (y_true_arr == c)))
        fp = int(np.sum((y_pred == c) & (y_true_arr != c)))
        tn = int(np.sum((y_pred != c) & (y_true_arr != c)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "support": int(tp + fn),
        }

    metrics["per_class"] = per_class
    metrics["melanoma_sensitivity"] = per_class["mel"]["sensitivity"]
    metrics["melanoma_specificity"] = per_class["mel"]["specificity"]

    cancer_true = np.isin(y_true_arr, list(CANCER_INDICES))
    cancer_pred = np.isin(y_pred, list(CANCER_INDICES))
    cancer_tp = int(np.sum(cancer_true & cancer_pred))
    cancer_fn = int(np.sum(cancer_true & ~cancer_pred))
    metrics["all_cancer_sensitivity"] = round(
        cancer_tp / (cancer_tp + cancer_fn) if (cancer_tp + cancer_fn) > 0 else 0.0, 4
    )

    return metrics


# ---------------------------------------------------------------------------
# Display and comparison
# ---------------------------------------------------------------------------

def print_comparison(default_metrics, optimized_metrics, thresholds_info):
    """Print a before/after comparison table."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  ADR-123: THRESHOLD OPTIMIZATION RESULTS")
    print(f"{sep}\n")

    # Per-class thresholds
    print("Per-Class Optimal Thresholds:")
    print(f"  {'Class':>8}  {'Threshold':>10}  {'Constraint':>28}  {'Sens':>6}  {'Spec':>6}  {'AUC':>6}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cls in CLASS_NAMES:
        info = thresholds_info[cls]
        print(
            f"  {cls:>8}  {info['threshold']:>10.4f}  {info['constraint']:>28}  "
            f"{info['sensitivity_at_threshold']:>6.4f}  "
            f"{info['specificity_at_threshold']:>6.4f}  "
            f"{info['auc']:>6.4f}"
        )

    # Overall comparison
    print(f"\n{'─' * 80}")
    print(f"  {'Metric':<30} {'Default (argmax)':>18} {'Optimized':>18} {'Delta':>10}")
    print(f"  {'─'*30} {'─'*18} {'─'*18} {'─'*10}")

    def row(name, d, o):
        delta = o - d
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<30} {d:>18.4f} {o:>18.4f} {sign}{delta:>9.4f}")

    row("Overall accuracy", default_metrics["overall_accuracy"], optimized_metrics["overall_accuracy"])
    row("Melanoma sensitivity", default_metrics["melanoma_sensitivity"], optimized_metrics["melanoma_sensitivity"])
    row("Melanoma specificity", default_metrics["melanoma_specificity"], optimized_metrics["melanoma_specificity"])
    row("All-cancer sensitivity", default_metrics["all_cancer_sensitivity"], optimized_metrics["all_cancer_sensitivity"])

    # Per-class F1 comparison
    print(f"\n{'─' * 80}")
    print(f"  {'Class F1':<12} {'Default':>10} {'Optimized':>10} {'Delta':>10}  "
          f"{'Def Sens':>10} {'Opt Sens':>10} {'Def Spec':>10} {'Opt Spec':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for cls in CLASS_NAMES:
        df1 = default_metrics["per_class"][cls]["f1"]
        of1 = optimized_metrics["per_class"][cls]["f1"]
        delta = of1 - df1
        sign = "+" if delta >= 0 else ""
        ds = default_metrics["per_class"][cls]["sensitivity"]
        os_ = optimized_metrics["per_class"][cls]["sensitivity"]
        dsp = default_metrics["per_class"][cls]["specificity"]
        osp = optimized_metrics["per_class"][cls]["specificity"]
        cancer_marker = " *" if cls in ("mel", "bcc", "akiec") else ""
        print(
            f"  {cls+cancer_marker:<12} {df1:>10.4f} {of1:>10.4f} {sign}{delta:>9.4f}  "
            f"{ds:>10.4f} {os_:>10.4f} {dsp:>10.4f} {osp:>10.4f}"
        )

    print(f"\n  * = cancer class")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 60)
    print("  ADR-123: Per-Class Threshold Optimization")
    print("  Model: mela-classifier-v2")
    print("  Dataset: ISIC 2019 (15% stratified holdout, seed=42)")
    print("=" * 60)

    # Step 1: Load model
    model, processor, device = load_model()

    # Step 2: Load ISIC 2019 test set (exact same split as training)
    test_images, test_labels = load_isic_test_set()
    assert len(test_images) == len(test_labels)
    print(f"\n  Test set: {len(test_images)} images")

    # Step 3: Run inference
    probs = run_inference(model, processor, device, test_images)
    assert probs.shape == (len(test_images), NUM_CLASSES)
    print(f"  Probability matrix shape: {probs.shape}")

    # Step 4: Compute default (argmax) metrics
    print("\n" + "=" * 60)
    print("  Computing default (argmax) metrics...")
    print("=" * 60)
    default_metrics = compute_argmax_metrics(test_labels, probs)
    print(f"  Overall accuracy: {default_metrics['overall_accuracy']:.4f}")
    print(f"  Melanoma sensitivity: {default_metrics['melanoma_sensitivity']:.4f}")
    print(f"  Melanoma specificity: {default_metrics['melanoma_specificity']:.4f}")
    print(f"  All-cancer sensitivity: {default_metrics['all_cancer_sensitivity']:.4f}")

    # Step 5: Compute ROC curves for each class
    print("\n" + "=" * 60)
    print("  Computing per-class ROC curves...")
    print("=" * 60)
    roc_data = compute_roc_curves(test_labels, probs)
    for cls in CLASS_NAMES:
        print(f"  {cls:>6} AUC: {roc_data[cls]['auc']:.4f}")

    # Step 6: Find optimal thresholds per class
    print("\n" + "=" * 60)
    print("  Finding optimal thresholds per class...")
    print("=" * 60)
    thresholds_info = {}
    thresholds_simple = {}
    for cls in CLASS_NAMES:
        result = find_optimal_threshold(roc_data[cls], cls)
        thresholds_info[cls] = result
        thresholds_simple[cls] = result["threshold"]
        print(
            f"  {cls:>6}: threshold={result['threshold']:.4f}  "
            f"sens={result['sensitivity_at_threshold']:.4f}  "
            f"spec={result['specificity_at_threshold']:.4f}  "
            f"({result['constraint']})"
        )

    # Step 7: Compute metrics with optimized thresholds
    print("\n" + "=" * 60)
    print("  Computing metrics with optimized thresholds...")
    print("=" * 60)
    optimized_metrics = compute_threshold_metrics(test_labels, probs, thresholds_simple)

    # Step 8: Print comparison
    print_comparison(default_metrics, optimized_metrics, thresholds_info)

    # Step 9: Save results
    # Prepare ROC curve data for melanoma (subsample for reasonable JSON size)
    mel_roc = roc_data["mel"]
    n_points = len(mel_roc["fpr"])
    if n_points > 500:
        # Subsample to ~500 points evenly spaced
        indices = np.linspace(0, n_points - 1, 500, dtype=int)
    else:
        indices = np.arange(n_points)
    mel_roc_data = {
        "fpr": [round(float(mel_roc["fpr"][i]), 6) for i in indices],
        "tpr": [round(float(mel_roc["tpr"][i]), 6) for i in indices],
        "thresholds": [round(float(mel_roc["thresholds"][i]), 6) for i in indices],
        "auc": round(mel_roc["auc"], 4),
    }

    full_results = {
        "adr": "ADR-123",
        "description": "Per-class threshold optimization for Mela v2 combined model",
        "model": str(MODEL_DIR),
        "dataset": "akinsanyaayomide/skin_cancer_dataset_balanced_labels_2",
        "test_set_size": len(test_images),
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_SEED},
        "optimal_thresholds": thresholds_simple,
        "threshold_details": thresholds_info,
        "constraints": {
            "melanoma": f"maximize sensitivity, specificity >= {MIN_SPEC_MEL*100:.0f}%",
            "bcc_akiec": f"maximize sensitivity, specificity >= {MIN_SPEC_CANCER*100:.0f}%",
            "benign": f"maximize specificity, sensitivity >= {MIN_SENS_BENIGN*100:.0f}%",
        },
        "default_argmax_metrics": default_metrics,
        "optimized_threshold_metrics": optimized_metrics,
        "comparison": {
            "overall_accuracy_delta": round(
                optimized_metrics["overall_accuracy"] - default_metrics["overall_accuracy"], 4
            ),
            "melanoma_sensitivity_delta": round(
                optimized_metrics["melanoma_sensitivity"] - default_metrics["melanoma_sensitivity"], 4
            ),
            "melanoma_specificity_delta": round(
                optimized_metrics["melanoma_specificity"] - default_metrics["melanoma_specificity"], 4
            ),
            "all_cancer_sensitivity_delta": round(
                optimized_metrics["all_cancer_sensitivity"] - default_metrics["all_cancer_sensitivity"], 4
            ),
            "per_class_f1_delta": {
                cls: round(
                    optimized_metrics["per_class"][cls]["f1"] - default_metrics["per_class"][cls]["f1"], 4
                )
                for cls in CLASS_NAMES
            },
        },
        "melanoma_roc_curve": mel_roc_data,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_PATH}")

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds_simple, f, indent=2)
    print(f"Simple thresholds saved to: {THRESHOLDS_PATH}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
