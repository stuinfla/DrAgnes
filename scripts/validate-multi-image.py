#!/usr/bin/env python3
"""
Multi-Image Consensus Validation for DrAgnes ViT Classifier
============================================================
Measures actual accuracy improvement from multi-image voting on HAM10000.

Three methods compared:
  1. Single-image (unaugmented baseline)
  2. 3-image majority vote (augmented views)
  3. 3-image quality-weighted vote (weighted by Laplacian sharpness)

Dataset: kuchikihater/HAM10000 (10,015 images, 7 classes)
Model:   ViT-Base fine-tuned on HAM10000 (scripts/dragnes-classifier/best/)
"""

import json
import time
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "dragnes-classifier" / "best"
RESULTS_PATH = SCRIPT_DIR / "multi-image-validation-results.json"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX = 4
HOLDOUT_FRAC = 0.15
N_VIEWS = 3
SEED = 42
PROGRESS_EVERY = 100


def log(msg: str) -> None:
    print(msg, flush=True)


def augment(img: Image.Image) -> Image.Image:
    """Simulate a second photo: slight crop, rotation, brightness shift."""
    w, h = img.size
    crop_frac = random.uniform(0.95, 1.0)
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    img = TF.crop(img, top, left, ch, cw)
    img = TF.resize(img, [h, w])
    angle = random.uniform(-10, 10)
    img = TF.rotate(img, angle)
    brightness = random.uniform(0.9, 1.1)
    img = TF.adjust_brightness(img, brightness)
    return img


def sharpness(img: Image.Image) -> float:
    """Laplacian variance as a sharpness proxy."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify(model, processor, images, device):
    """Return (predicted_class_indices, probability_matrices) for a batch."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs.argmax(axis=-1).tolist(), probs


def f1_per_class(y_true, y_pred, n_classes=7):
    """Return dict of per-class precision, recall, F1."""
    result = {}
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result[CLASS_NAMES[c]] = {"precision": round(prec, 4),
                                   "recall": round(rec, 4),
                                   "f1": round(f1, 4),
                                   "support": tp + fn}
    return result


def mel_specificity(y_true, y_pred):
    """Specificity for melanoma = TN / (TN + FP)."""
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != MEL_IDX and p != MEL_IDX)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != MEL_IDX and p == MEL_IDX)
    return tn / (tn + fp) if (tn + fp) else 0.0


def report(tag, y_true, y_pred):
    """Compute and print metrics for one method."""
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    f1s = f1_per_class(y_true, y_pred)
    mel_rec = f1s["mel"]["recall"]
    mel_spec = mel_specificity(y_true, y_pred)
    log(f"\n--- {tag} ---")
    log(f"  Overall accuracy:     {acc:.4f}")
    log(f"  Melanoma sensitivity: {mel_rec:.4f}")
    log(f"  Melanoma specificity: {mel_spec:.4f}")
    log(f"  Per-class F1:")
    for name in CLASS_NAMES:
        m = f1s[name]
        log(f"    {name:>6}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  (n={m['support']})")
    return {"accuracy": round(acc, 4),
            "melanoma_sensitivity": round(mel_rec, 4),
            "melanoma_specificity": round(mel_spec, 4),
            "per_class_f1": f1s}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    log("=" * 70)
    log("Multi-Image Consensus Validation — DrAgnes ViT")
    log("=" * 70)
    log(f"Timestamp: {datetime.now().isoformat()}")

    # --- Load model ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(device)
    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model.eval()
    log(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Load dataset and create 15% stratified holdout ---
    log(f"\nLoading kuchikihater/HAM10000 ...")
    ds = load_dataset("kuchikihater/HAM10000", split="train")
    log(f"Total images: {len(ds)}")

    by_class = {c: [] for c in range(7)}
    for i in range(len(ds)):
        by_class[ds[i]["label"]].append(i)

    test_indices = []
    for c in range(7):
        idxs = by_class[c]
        random.shuffle(idxs)
        n_test = max(1, int(len(idxs) * HOLDOUT_FRAC))
        test_indices.extend(idxs[:n_test])
    random.shuffle(test_indices)

    dist = Counter(ds[i]["label"] for i in test_indices)
    log(f"Test set: {len(test_indices)} images (15% stratified holdout)")
    for c in range(7):
        log(f"  {CLASS_NAMES[c]:>6}: {dist.get(c, 0)}")

    # --- Inference ---
    log(f"\nRunning {N_VIEWS}-view inference on {len(test_indices)} images ...")
    t0 = time.time()
    true_labels, single_preds, vote_preds, weighted_preds = [], [], [], []

    for step, idx in enumerate(test_indices):
        row = ds[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        true_label = row["label"]
        true_labels.append(true_label)

        # View 0: original (unaugmented)
        preds_0, probs_0 = classify(model, processor, [img], device)
        single_preds.append(preds_0[0])
        sharp_0 = sharpness(img)

        # Views 1..N-1: augmented
        all_preds = [preds_0[0]]
        all_probs = [probs_0[0]]
        all_sharps = [sharp_0]

        for _ in range(N_VIEWS - 1):
            aug = augment(img)
            p, pr = classify(model, processor, [aug], device)
            all_preds.append(p[0])
            all_probs.append(pr[0])
            all_sharps.append(sharpness(aug))

        # Majority vote
        counts = Counter(all_preds)
        vote_preds.append(counts.most_common(1)[0][0])

        # Quality-weighted vote: sum probability vectors weighted by sharpness
        weights = np.array(all_sharps)
        weights = weights / (weights.sum() + 1e-12)
        blended = sum(w * p for w, p in zip(weights, all_probs))
        weighted_preds.append(int(np.argmax(blended)))

        if (step + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            log(f"  [{step+1:>5}/{len(test_indices)}] "
                f"{elapsed:.1f}s  ({(step+1)/elapsed:.1f} img/s)")

    elapsed = time.time() - t0
    log(f"\nDone: {len(true_labels)} images in {elapsed:.1f}s "
        f"({len(true_labels)/elapsed:.1f} img/s)")

    # --- Report ---
    log("\n" + "=" * 70)
    r_single = report("Single-image (baseline)", true_labels, single_preds)
    r_vote = report("3-image majority vote", true_labels, vote_preds)
    r_weight = report("3-image quality-weighted vote", true_labels, weighted_preds)

    # --- Deltas ---
    log("\n" + "=" * 70)
    log("IMPROVEMENT SUMMARY")
    log("=" * 70)
    for tag, r in [("Majority vote", r_vote), ("Weighted vote", r_weight)]:
        da = r["accuracy"] - r_single["accuracy"]
        ds_ = r["melanoma_sensitivity"] - r_single["melanoma_sensitivity"]
        log(f"  {tag} vs single:")
        log(f"    Accuracy:     {da:+.4f}  ({r_single['accuracy']:.4f} -> {r['accuracy']:.4f})")
        log(f"    Mel sens:     {ds_:+.4f}  ({r_single['melanoma_sensitivity']:.4f} -> {r['melanoma_sensitivity']:.4f})")

    # --- Save JSON ---
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": str(MODEL_DIR),
        "device": str(device),
        "dataset": "kuchikihater/HAM10000",
        "test_set_size": len(true_labels),
        "holdout_fraction": HOLDOUT_FRAC,
        "n_views": N_VIEWS,
        "augmentations": "crop_pm5pct, rotation_pm10deg, brightness_pm10pct",
        "inference_seconds": round(elapsed, 1),
        "single_image": r_single,
        "majority_vote_3img": r_vote,
        "quality_weighted_3img": r_weight,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
