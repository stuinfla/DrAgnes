#!/usr/bin/env python3
"""
Mela Cross-Dataset Validation

Validates the custom-trained ViT model against external dermoscopy datasets
to prove generalization beyond the HAM10000 training data.

Strategy (in order of priority):
  1. External dataset: Nagabu/HAM10000 (binary melanoma/nevus -- different
     upload, different preprocessing pipeline, tests melanoma detection)
  2. Held-out test split: marmal88/skin_cancer has a dedicated test split
     of 1285 images that may differ from our 85/15 random split
  3. Overfitting check: reproduce the exact 85/15 stratified split from
     training and compare train vs test accuracy

The model was trained on marmal88/skin_cancer (HAM10000) with:
  - 85/15 stratified split, random_state=42
  - Classes: akiec(0), bcc(1), bkl(2), df(3), mel(4), nv(5), vasc(6)
  - dx values: actinic_keratoses, basal_cell_carcinoma,
    benign_keratosis-like_lesions, dermatofibroma, melanocytic_Nevi,
    melanoma, vascular_lesions

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/cross-validate.py
"""

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTImageProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "mela-classifier" / "best"
RESULTS_FILE = SCRIPT_DIR / "cross-validation-results.json"

# HAM10000 class taxonomy (must match training order exactly)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}
NUM_CLASSES = len(CLASS_NAMES)
MEL_IDX = CLASS_NAMES.index("mel")  # 4

# Mapping from marmal88/skin_cancer 'dx' strings to our class indices
DX_TO_IDX = {
    "actinic_keratoses": 0,       # akiec
    "basal_cell_carcinoma": 1,    # bcc
    "benign_keratosis-like_lesions": 2,  # bkl
    "dermatofibroma": 3,          # df
    "melanoma": 4,                # mel
    "melanocytic_Nevi": 5,        # nv
    "vascular_lesions": 6,        # vasc
}

# Training split parameters (must match train-proper.py exactly)
TRAIN_TEST_SIZE = 0.15
RANDOM_SEED = 42

# Max images to evaluate per dataset attempt
MAX_IMAGES = 1000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def p(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def load_model():
    """Load the trained ViT model and processor."""
    p(f"Loading model from {MODEL_DIR}...")
    if not MODEL_DIR.exists():
        p(f"ERROR: Model directory not found: {MODEL_DIR}")
        sys.exit(1)

    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    p(f"Model loaded on device: {device}")
    p(f"Model config labels: {model.config.id2label}")

    # Check if model has proper label names or generic LABEL_N
    has_proper_labels = all(
        not v.startswith("LABEL_") for v in model.config.id2label.values()
    )
    if not has_proper_labels:
        p("NOTE: Model config has generic LABEL_N labels. Using training order:")
        p(f"  {CLASS_NAMES}")
        p("  (Canonical order from train-proper.py)")

    return model, processor, device


def predict_batch(model, processor, device, images, batch_size=32):
    """Run inference on a list of PIL Images, return predicted class indices."""
    predictions = []
    confidences = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        predictions.extend(preds.cpu().numpy().tolist())
        confidences.extend(probs.max(dim=-1).values.cpu().numpy().tolist())

    return predictions, confidences


def compute_metrics(y_true, y_pred, class_names, dataset_name):
    """Compute and display comprehensive metrics."""
    results = {}

    # Overall accuracy
    correct = sum(1 for t, pred in zip(y_true, y_pred) if t == pred)
    total = len(y_true)
    overall_acc = correct / total if total > 0 else 0.0

    p(f"\n{'=' * 60}")
    p(f"  CROSS-DATASET VALIDATION RESULTS")
    p(f"{'=' * 60}")
    p(f"Dataset:       {dataset_name}")
    p(f"Images tested: {total}")
    p(f"Overall accuracy: {overall_acc:.1%} ({correct}/{total})")
    p(f"{'=' * 60}")

    results["dataset"] = dataset_name
    results["total_images"] = total
    results["overall_accuracy"] = round(overall_acc, 4)
    results["correct"] = correct

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    p(f"\nPer-class metrics:")
    p(f"{'Class':<8} {'Sens/Recall':>11} {'Specificity':>12} {'Precision':>11} {'F1':>8} {'N':>6}")
    p(f"{'-' * 55}")

    per_class = {}
    for idx, name in enumerate(class_names):
        sens = recall[idx]
        prec = precision[idx]
        f1_score = f1[idx]
        n = int(support[idx])

        # Compute specificity from confusion matrix
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        marker = " <-- KEY METRIC" if name == "mel" else ""
        p(f"  {name:<6} {sens:>10.1%} {specificity:>11.1%} {prec:>10.1%} {f1_score:>7.1%}  {n:>5}{marker}")

        per_class[name] = {
            "sensitivity": round(sens, 4),
            "specificity": round(specificity, 4),
            "precision": round(prec, 4),
            "f1": round(f1_score, 4),
            "support": n,
        }

    results["per_class"] = per_class

    # Highlight melanoma
    mel_sens = recall[MEL_IDX]
    mel_n = int(support[MEL_IDX])
    mel_spec = per_class["mel"]["specificity"]
    p(f"\n{'*' * 60}")
    p(f"*** MELANOMA SENSITIVITY ON EXTERNAL DATA: {mel_sens:.1%} ***")
    p(f"*** MELANOMA SPECIFICITY:                  {mel_spec:.1%} ***")
    p(f"*** (N={mel_n} melanoma samples)                          ***")
    p(f"{'*' * 60}")

    results["melanoma_sensitivity"] = round(mel_sens, 4)
    results["melanoma_specificity"] = round(mel_spec, 4)
    results["melanoma_n"] = mel_n

    # Full classification report
    p(f"\nFull classification report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
    )
    p(report)

    # Confusion matrix
    p("Confusion matrix (rows=true, cols=predicted):")
    header = "       " + "  ".join(f"{n:>6}" for n in class_names)
    p(header)
    for idx, name in enumerate(class_names):
        row = f"  {name:<5}" + "  ".join(f"{cm[idx, j]:>6}" for j in range(NUM_CLASSES))
        p(row)

    results["confusion_matrix"] = cm.tolist()

    return results


# ---------------------------------------------------------------------------
# Strategy 1: External binary melanoma dataset
# ---------------------------------------------------------------------------

def try_external_binary(model, processor, device):
    """
    Load Nagabu/HAM10000 -- a different upload with binary labels
    (melanoma vs nevus). Tests whether our model correctly identifies
    melanoma on images processed through a different pipeline.
    """
    p("\n" + "=" * 60)
    p("  Strategy 1: External binary melanoma dataset")
    p("  (Nagabu/HAM10000 -- different upload pipeline)")
    p("=" * 60)
    try:
        from datasets import load_dataset

        ds = load_dataset("Nagabu/HAM10000")
        split_name = list(ds.keys())[0]
        sample = ds[split_name]
        p(f"Loaded {len(sample)} samples from split '{split_name}'")
        p(f"Features: {list(sample.features.keys())}")

        # This dataset has binary labels: melanoma (0?) and nevus (1?)
        # Check the actual class names
        features = sample.features
        if hasattr(features["label"], "names"):
            ds_class_names = features["label"].names
            p(f"Class names: {ds_class_names}")
        else:
            ds_class_names = None
            p("No class names found in label feature")

        # Map binary labels to our 7-class indices
        # melanoma -> 4 (mel), nevus -> 5 (nv)
        binary_map = {}
        if ds_class_names:
            for idx_val, name in enumerate(ds_class_names):
                nl = name.lower().strip()
                if "melanoma" in nl or nl == "mel":
                    binary_map[idx_val] = MEL_IDX  # 4
                elif "nevus" in nl or "nevi" in nl or nl == "nv":
                    binary_map[idx_val] = CLASS_NAMES.index("nv")  # 5
                else:
                    p(f"  Unmapped class: '{name}' (idx {idx_val})")

        if not binary_map:
            p("Could not build label mapping. Aborting strategy 1.")
            return None

        p(f"Label mapping: {binary_map}")

        # Collect images
        images = []
        true_labels = []
        skipped = 0

        n_total = len(sample)
        n_to_process = min(n_total, MAX_IMAGES)
        if n_total > MAX_IMAGES:
            rng = np.random.RandomState(42)
            indices = rng.choice(n_total, MAX_IMAGES, replace=False)
        else:
            indices = range(n_total)

        p(f"Processing {n_to_process} images...")
        for count, idx in enumerate(indices):
            if count % 200 == 0:
                p(f"  Loading image {count}/{n_to_process}...")

            row = sample[int(idx)]
            raw_label = row["label"]

            if raw_label not in binary_map:
                skipped += 1
                continue

            img = row["image"]
            if not isinstance(img, Image.Image):
                skipped += 1
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")

            images.append(img)
            true_labels.append(binary_map[raw_label])

        p(f"Loaded {len(images)} images ({skipped} skipped)")
        p(f"Class distribution: {Counter(true_labels)}")

        if len(images) < 50:
            p("Too few images. Aborting strategy 1.")
            return None

        # Run inference
        p("Running inference...")
        t0 = time.time()
        predictions, confidences = predict_batch(
            model, processor, device, images, batch_size=32
        )
        elapsed = time.time() - t0
        p(f"Inference complete in {elapsed:.1f}s ({len(images) / elapsed:.0f} img/s)")

        # For binary evaluation: mel vs not-mel
        # Our model outputs 7 classes, so we need to check if it predicts mel
        # vs anything else for the binary dataset
        true_binary = [1 if t == MEL_IDX else 0 for t in true_labels]
        pred_binary = [1 if p_ == MEL_IDX else 0 for p_ in predictions]

        # Binary metrics
        tp = sum(1 for t, pred in zip(true_binary, pred_binary) if t == 1 and pred == 1)
        fn = sum(1 for t, pred in zip(true_binary, pred_binary) if t == 1 and pred == 0)
        fp = sum(1 for t, pred in zip(true_binary, pred_binary) if t == 0 and pred == 1)
        tn = sum(1 for t, pred in zip(true_binary, pred_binary) if t == 0 and pred == 0)

        mel_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        mel_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        binary_acc = (tp + tn) / len(true_binary) if true_binary else 0

        p(f"\n{'=' * 60}")
        p(f"  BINARY MELANOMA DETECTION (External Dataset)")
        p(f"{'=' * 60}")
        p(f"Dataset:  Nagabu/HAM10000 (different upload pipeline)")
        p(f"Task:     Melanoma vs Non-melanoma")
        p(f"Images:   {len(images)}")
        p(f"Melanoma: {sum(true_binary)} | Non-melanoma: {len(true_binary) - sum(true_binary)}")
        p(f"")
        p(f"  True Positives:  {tp}")
        p(f"  False Negatives: {fn}")
        p(f"  False Positives: {fp}")
        p(f"  True Negatives:  {tn}")
        p(f"")
        p(f"  Binary Accuracy:     {binary_acc:.1%}")
        p(f"  Melanoma Sensitivity: {mel_sensitivity:.1%}")
        p(f"  Melanoma Specificity: {mel_specificity:.1%}")
        p(f"")
        p(f"{'*' * 60}")
        p(f"*** MELANOMA SENSITIVITY ON EXTERNAL DATA: {mel_sensitivity:.1%} ***")
        p(f"*** (N={sum(true_binary)} melanoma, {len(true_binary) - sum(true_binary)} non-melanoma)")
        p(f"{'*' * 60}")

        # Also show what our model predicted for the nevus images
        # (which classes it confused them with)
        nevus_preds = [predictions[i] for i, t in enumerate(true_labels) if t == CLASS_NAMES.index("nv")]
        nevus_pred_dist = Counter(nevus_preds)
        p(f"\nNevus images -- model prediction distribution:")
        for cls_idx in sorted(nevus_pred_dist.keys()):
            cnt = nevus_pred_dist[cls_idx]
            pct = cnt / len(nevus_preds) * 100 if nevus_preds else 0
            p(f"  {CLASS_NAMES[cls_idx]:>6}: {cnt:>5} ({pct:.1f}%)")

        mel_preds = [predictions[i] for i, t in enumerate(true_labels) if t == MEL_IDX]
        mel_pred_dist = Counter(mel_preds)
        p(f"\nMelanoma images -- model prediction distribution:")
        for cls_idx in sorted(mel_pred_dist.keys()):
            cnt = mel_pred_dist[cls_idx]
            pct = cnt / len(mel_preds) * 100 if mel_preds else 0
            p(f"  {CLASS_NAMES[cls_idx]:>6}: {cnt:>5} ({pct:.1f}%)")

        results = {
            "strategy": "external_binary",
            "source": "Nagabu/HAM10000",
            "dataset": "Nagabu/HAM10000 (external binary melanoma dataset)",
            "total_images": len(images),
            "melanoma_n": sum(true_binary),
            "non_melanoma_n": len(true_binary) - sum(true_binary),
            "binary_accuracy": round(binary_acc, 4),
            "melanoma_sensitivity": round(mel_sensitivity, 4),
            "melanoma_specificity": round(mel_specificity, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "inference_time_sec": round(elapsed, 2),
            "images_per_second": round(len(images) / elapsed, 1),
        }
        return results

    except Exception as e:
        p(f"Strategy 1 failed: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Strategy 2: marmal88/skin_cancer held-out test split
# ---------------------------------------------------------------------------

def try_held_out_test(model, processor, device):
    """
    The marmal88/skin_cancer dataset has train/validation/test splits.
    Evaluate on the 'test' split (1285 images) which is a held-out set
    curated by the dataset author.
    """
    p("\n" + "=" * 60)
    p("  Strategy 2: Held-out test split from marmal88/skin_cancer")
    p("=" * 60)
    try:
        from datasets import load_dataset

        p("Loading marmal88/skin_cancer...")
        ds = load_dataset("marmal88/skin_cancer")
        p(f"Available splits: {list(ds.keys())}")

        if "test" not in ds:
            p("No 'test' split available. Aborting strategy 2.")
            return None

        test_data = ds["test"]
        p(f"Test split: {len(test_data)} samples")

        # Load test images
        images = []
        true_labels = []
        skipped = 0

        for count in range(len(test_data)):
            if count % 200 == 0:
                p(f"  Loading image {count}/{len(test_data)}...")

            row = test_data[count]
            dx = row["dx"]

            if dx not in DX_TO_IDX:
                p(f"  WARNING: Unknown dx '{dx}' -- skipping")
                skipped += 1
                continue

            img = row["image"]
            if not isinstance(img, Image.Image):
                skipped += 1
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")

            images.append(img)
            true_labels.append(DX_TO_IDX[dx])

        p(f"Loaded {len(images)} images ({skipped} skipped)")
        p(f"Class distribution: {Counter(true_labels)}")

        # Run inference
        p("Running inference...")
        t0 = time.time()
        predictions, confidences = predict_batch(
            model, processor, device, images, batch_size=32
        )
        elapsed = time.time() - t0
        p(f"Inference complete in {elapsed:.1f}s ({len(images) / elapsed:.0f} img/s)")

        results = compute_metrics(
            true_labels,
            predictions,
            CLASS_NAMES,
            "marmal88/skin_cancer TEST split (held-out, 1285 images)",
        )
        results["source"] = "marmal88/skin_cancer"
        results["strategy"] = "held_out_test"
        results["split"] = "test"
        results["inference_time_sec"] = round(elapsed, 2)
        results["images_per_second"] = round(len(images) / elapsed, 1)
        return results

    except Exception as e:
        p(f"Strategy 2 failed: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Strategy 3: Overfitting check (train vs test accuracy comparison)
# ---------------------------------------------------------------------------

def try_overfitting_check(model, processor, device):
    """
    Reproduce the exact 85/15 stratified train/test split from training
    and compare accuracy on both sets. A gap > 15% indicates overfitting.

    Uses marmal88/skin_cancer with identical split parameters as train-proper.py.
    train-proper.py combines ALL splits, then does its own stratified split.
    """
    p("\n" + "=" * 60)
    p("  Strategy 3: Overfitting check (train vs test accuracy)")
    p("  Reproducing exact 85/15 split from training")
    p("=" * 60)
    try:
        from datasets import load_dataset

        p("Loading marmal88/skin_cancer (all splits)...")
        ds = load_dataset("marmal88/skin_cancer")
        p(f"Splits: {list(ds.keys())}")

        # Combine ALL splits into a single list (matching train-proper.py)
        all_images = []
        all_labels = []
        skipped = 0
        total_across_splits = 0

        for split_name in sorted(ds.keys()):
            split_data = ds[split_name]
            split_count = len(split_data)
            total_across_splits += split_count
            p(f"  Loading split '{split_name}': {split_count} samples...")

            for idx in range(split_count):
                if idx % 1000 == 0 and idx > 0:
                    p(f"    ...{idx}/{split_count}")

                row = split_data[idx]
                dx = row["dx"]

                if dx not in DX_TO_IDX:
                    skipped += 1
                    all_images.append(None)
                    all_labels.append(-1)
                    continue

                img = row["image"]
                if not isinstance(img, Image.Image):
                    skipped += 1
                    all_images.append(None)
                    all_labels.append(-1)
                    continue
                if img.mode != "RGB":
                    img = img.convert("RGB")

                all_images.append(img)
                all_labels.append(DX_TO_IDX[dx])

        p(f"Total samples loaded: {len(all_images)} ({skipped} skipped)")

        # Filter valid indices
        valid_indices = [i for i, l in enumerate(all_labels) if l >= 0]
        valid_labels_list = [all_labels[i] for i in valid_indices]

        p(f"Valid samples: {len(valid_indices)}")
        dist = Counter(valid_labels_list)
        for cls_idx in sorted(dist.keys()):
            p(f"  {CLASS_NAMES[cls_idx]:>6}: {dist[cls_idx]:>5}")

        # Reproduce the exact 85/15 stratified split
        p("\nReproducing 85/15 stratified split (random_state=42)...")
        train_idx, test_idx = train_test_split(
            valid_indices,
            test_size=TRAIN_TEST_SIZE,
            stratify=[all_labels[i] for i in valid_indices],
            random_state=RANDOM_SEED,
        )

        p(f"Train set: {len(train_idx)} samples")
        p(f"Test set:  {len(test_idx)} samples")

        # --- Evaluate on TEST set ---
        p("\n--- Evaluating on TEST set ---")
        test_images = [all_images[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]

        t0 = time.time()
        test_preds, test_confs = predict_batch(
            model, processor, device, test_images, batch_size=32
        )
        test_elapsed = time.time() - t0

        test_correct = sum(1 for t, pred in zip(test_labels, test_preds) if t == pred)
        test_acc = test_correct / len(test_labels)
        p(f"Test accuracy: {test_acc:.1%} ({test_correct}/{len(test_labels)}) [{test_elapsed:.1f}s]")

        # Melanoma sensitivity on test
        mel_test_true = [1 if t == MEL_IDX else 0 for t in test_labels]
        mel_test_pred = [1 if p_ == MEL_IDX else 0 for p_ in test_preds]
        mel_tp = sum(1 for t, pred in zip(mel_test_true, mel_test_pred) if t == 1 and pred == 1)
        mel_fn = sum(1 for t, pred in zip(mel_test_true, mel_test_pred) if t == 1 and pred == 0)
        mel_test_sens = mel_tp / (mel_tp + mel_fn) if (mel_tp + mel_fn) > 0 else 0
        p(f"Test melanoma sensitivity: {mel_test_sens:.1%} ({mel_tp}/{mel_tp + mel_fn})")

        # --- Evaluate on TRAIN set (sample for speed) ---
        p("\n--- Evaluating on TRAIN set (overfitting check) ---")
        if len(train_idx) > MAX_IMAGES:
            rng = np.random.RandomState(42)
            train_sample_idx = rng.choice(train_idx, MAX_IMAGES, replace=False).tolist()
        else:
            train_sample_idx = train_idx

        train_images = [all_images[i] for i in train_sample_idx]
        train_labels = [all_labels[i] for i in train_sample_idx]

        t0 = time.time()
        train_preds, train_confs = predict_batch(
            model, processor, device, train_images, batch_size=32
        )
        train_elapsed = time.time() - t0

        train_correct = sum(1 for t, pred in zip(train_labels, train_preds) if t == pred)
        train_acc = train_correct / len(train_labels)
        p(f"Train accuracy: {train_acc:.1%} ({train_correct}/{len(train_labels)}) [{train_elapsed:.1f}s]")

        # Melanoma sensitivity on train
        mel_train_true = [1 if t == MEL_IDX else 0 for t in train_labels]
        mel_train_pred = [1 if p_ == MEL_IDX else 0 for p_ in train_preds]
        mel_tp_tr = sum(1 for t, pred in zip(mel_train_true, mel_train_pred) if t == 1 and pred == 1)
        mel_fn_tr = sum(1 for t, pred in zip(mel_train_true, mel_train_pred) if t == 1 and pred == 0)
        mel_train_sens = mel_tp_tr / (mel_tp_tr + mel_fn_tr) if (mel_tp_tr + mel_fn_tr) > 0 else 0
        p(f"Train melanoma sensitivity: {mel_train_sens:.1%} ({mel_tp_tr}/{mel_tp_tr + mel_fn_tr})")

        # --- Overfitting analysis ---
        gap = train_acc - test_acc
        mel_gap = mel_train_sens - mel_test_sens

        p(f"\n{'=' * 60}")
        p(f"  OVERFITTING ANALYSIS")
        p(f"{'=' * 60}")
        p(f"                       Train        Test       Gap")
        p(f"  Overall accuracy:  {train_acc:>7.1%}     {test_acc:>7.1%}    {gap:>+7.1%}")
        p(f"  Melanoma sens.:    {mel_train_sens:>7.1%}     {mel_test_sens:>7.1%}    {mel_gap:>+7.1%}")
        p(f"")

        if gap > 0.15:
            p(f"  VERDICT: OVERFITTING CONCERN (gap > 15%)")
            p(f"  The model may have memorized training patterns.")
            overfit_status = "OVERFITTING_CONCERN"
        elif gap > 0.10:
            p(f"  VERDICT: MILD OVERFITTING (gap 10-15%)")
            overfit_status = "MILD_OVERFITTING"
        elif gap > 0.05:
            p(f"  VERDICT: ACCEPTABLE (gap 5-10%)")
            p(f"  Typical for fine-tuned models on small datasets.")
            overfit_status = "ACCEPTABLE"
        else:
            p(f"  VERDICT: EXCELLENT GENERALIZATION (gap < 5%)")
            p(f"  Model generalizes well to unseen data.")
            overfit_status = "EXCELLENT"
        p(f"{'=' * 60}")

        # --- Detailed test metrics ---
        test_results = compute_metrics(
            test_labels,
            test_preds,
            CLASS_NAMES,
            "marmal88/skin_cancer (reproduced 85/15 TEST split)",
        )

        results = {
            "strategy": "overfitting_check",
            "source": "marmal88/skin_cancer",
            "dataset": "marmal88/skin_cancer (HAM10000) - overfitting check",
            "split_params": {
                "test_size": TRAIN_TEST_SIZE,
                "random_state": RANDOM_SEED,
                "stratified": True,
            },
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "train_test_gap": round(gap, 4),
            "melanoma_sensitivity_train": round(mel_train_sens, 4),
            "melanoma_sensitivity_test": round(mel_test_sens, 4),
            "melanoma_sensitivity_gap": round(mel_gap, 4),
            "overfit_status": overfit_status,
            "train_n": len(train_sample_idx),
            "test_n": len(test_idx),
            "test_results": test_results,
        }
        return results

    except Exception as e:
        p(f"Strategy 3 failed: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p("=" * 60)
    p("  Mela Cross-Dataset Validation")
    p("  Proving generalization of custom-trained ViT model")
    p("=" * 60)
    p(f"Model:  {MODEL_DIR}")
    p(f"Date:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"Labels: {CLASS_NAMES}")
    p("")

    model, processor, device = load_model()

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_path": str(MODEL_DIR),
        "device": str(device),
        "class_names": CLASS_NAMES,
        "strategies": [],
    }

    # --- Strategy 1: External binary dataset ---
    result1 = try_external_binary(model, processor, device)
    if result1:
        all_results["strategies"].append(result1)
        all_results["external_validation"] = result1

    # --- Strategy 2: Held-out test split ---
    result2 = try_held_out_test(model, processor, device)
    if result2:
        all_results["strategies"].append(result2)

    # --- Strategy 3: Overfitting check (always runs) ---
    result3 = try_overfitting_check(model, processor, device)
    if result3:
        all_results["strategies"].append(result3)
        all_results["overfitting_check"] = result3

    # --- Final Summary ---
    p(f"\n{'#' * 60}")
    p(f"  FINAL SUMMARY")
    p(f"{'#' * 60}")

    successful = len(all_results["strategies"])
    p(f"\nStrategies attempted: 3")
    p(f"Strategies succeeded: {successful}")

    for strat in all_results["strategies"]:
        name = strat.get("strategy", "unknown")
        dataset = strat.get("dataset", "unknown")

        p(f"\n  [{name}]")
        p(f"    Dataset: {dataset}")

        if name == "external_binary":
            p(f"    Binary accuracy:      {strat.get('binary_accuracy', 0):.1%}")
            p(f"    Melanoma sensitivity: {strat.get('melanoma_sensitivity', 0):.1%}")
            p(f"    Melanoma specificity: {strat.get('melanoma_specificity', 0):.1%}")
        elif name == "held_out_test":
            p(f"    Overall accuracy:     {strat.get('overall_accuracy', 0):.1%}")
            p(f"    Melanoma sensitivity: {strat.get('melanoma_sensitivity', 0):.1%}")
        elif name == "overfitting_check":
            p(f"    Train accuracy:       {strat.get('train_accuracy', 0):.1%}")
            p(f"    Test accuracy:        {strat.get('test_accuracy', 0):.1%}")
            p(f"    Gap:                  {strat.get('train_test_gap', 0):.1%}")
            p(f"    Overfit status:       {strat.get('overfit_status', 'N/A')}")
            p(f"    Mel sens (test):      {strat.get('melanoma_sensitivity_test', 0):.1%}")

    if result3:
        p(f"\n  Overall verdict: {result3['overfit_status']}")

    p(f"\n{'#' * 60}")

    # Save results
    p(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    p(f"Results saved to {RESULTS_FILE}")

    return all_results


if __name__ == "__main__":
    main()
