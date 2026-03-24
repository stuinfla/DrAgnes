#!/usr/bin/env python3
"""
ADR-125 Ensemble Validation: V1-only vs V2-only vs 0.7*V2 + 0.3*V1 ensemble.

Loads both ViT models, runs them on the HAM10000 15% stratified holdout
(seed=42 -- identical to training split), and reports melanoma sensitivity,
specificity, and overall accuracy for all three configurations.

Results saved to scripts/ensemble-validation-results.json.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from transformers import ViTForImageClassification, ViTImageProcessor

# ── Constants ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
V1_DIR = SCRIPT_DIR / "dragnes-classifier" / "best"
V2_DIR = SCRIPT_DIR / "dragnes-classifier-v2" / "best"
OUTPUT_PATH = SCRIPT_DIR / "ensemble-validation-results.json"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX = CLASS_NAMES.index("mel")

V2_WEIGHT = 0.7
V1_WEIGHT = 0.3
MEL_SAFETY_THRESHOLD = 0.15


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Model loading ─────────────────────────────────────────────────────

def load_model(model_dir: Path, device: torch.device):
    """Load a ViT model + processor from a local checkpoint."""
    log(f"  Loading model from {model_dir.name}/ ...")
    processor = ViTImageProcessor.from_pretrained(str(model_dir))
    model = ViTForImageClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"  {model_dir.name}: {n_params:.1f}M params")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, processor, images, device, batch_size=32):
    """Run inference on a list of PIL images; return (N, 7) softmax array."""
    all_probs = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start : start + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ── Ensemble ──────────────────────────────────────────────────────────

def ensemble_probs(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Weighted average ensemble with melanoma safety override.
    v1, v2: (N, 7) probability arrays.
    Returns: (N, 7) ensembled probabilities.
    """
    merged = V1_WEIGHT * v1 + V2_WEIGHT * v2

    # Melanoma safety override: if EITHER model flags mel above the
    # threshold, take the MAX mel probability instead of the average.
    v1_mel = v1[:, MEL_IDX]
    v2_mel = v2[:, MEL_IDX]
    safety_mask = (v1_mel > MEL_SAFETY_THRESHOLD) | (v2_mel > MEL_SAFETY_THRESHOLD)
    merged[safety_mask, MEL_IDX] = np.maximum(v1_mel[safety_mask], v2_mel[safety_mask])

    # Renormalize rows to sum to 1
    row_sums = merged.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    merged /= row_sums
    return merged


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, probs: np.ndarray):
    """Compute mel sensitivity, mel specificity, and overall accuracy."""
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)

    # Melanoma binary: mel vs everything else
    mel_true = (y_true == MEL_IDX).astype(int)
    mel_pred = (y_pred == MEL_IDX).astype(int)

    # Sensitivity = TP / (TP + FN)
    tp = ((mel_pred == 1) & (mel_true == 1)).sum()
    fn = ((mel_pred == 0) & (mel_true == 1)).sum()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    tn = ((mel_pred == 0) & (mel_true == 0)).sum()
    fp = ((mel_pred == 1) & (mel_true == 0)).sum()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "overall_accuracy": round(acc, 4),
        "mel_sensitivity": round(sensitivity, 4),
        "mel_specificity": round(specificity, 4),
        "mel_tp": int(tp),
        "mel_fn": int(fn),
        "mel_fp": int(fp),
        "mel_tn": int(tn),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Device: {device}")

    # 1. Load models
    log("\n--- Loading models ---")
    v1_model, v1_proc = load_model(V1_DIR, device)
    v2_model, v2_proc = load_model(V2_DIR, device)

    # 2. Load dataset and create holdout
    log("\n--- Loading kuchikihater/HAM10000 ---")
    ds = load_dataset("kuchikihater/HAM10000")["train"]
    log(f"  Total images: {len(ds)}")

    labels = [ex["label"] for ex in ds]
    _, test_idx = train_test_split(
        range(len(ds)), test_size=0.15, stratify=labels, random_state=42
    )
    log(f"  Holdout size: {len(test_idx)}")

    test_images = [ds[i]["image"].convert("RGB") for i in test_idx]
    y_true = np.array([labels[i] for i in test_idx])

    mel_count = (y_true == MEL_IDX).sum()
    log(f"  Melanoma cases in holdout: {mel_count}")

    # 3. Run V1
    log("\n--- Running V1 (Anwarkh1 ViT) ---")
    t0 = time.time()
    v1_probs = predict_batch(v1_model, v1_proc, test_images, device)
    v1_time = time.time() - t0
    log(f"  V1 inference: {v1_time:.1f}s ({len(test_images)/v1_time:.0f} img/s)")

    # 4. Run V2
    log("\n--- Running V2 ---")
    t0 = time.time()
    v2_probs = predict_batch(v2_model, v2_proc, test_images, device)
    v2_time = time.time() - t0
    log(f"  V2 inference: {v2_time:.1f}s ({len(test_images)/v2_time:.0f} img/s)")

    # 5. Compute ensemble
    log("\n--- Computing ensemble (0.7*V2 + 0.3*V1 + mel safety override) ---")
    ens_probs = ensemble_probs(v1_probs, v2_probs)

    # 6. Compute metrics
    v1_metrics = compute_metrics(y_true, v1_probs)
    v2_metrics = compute_metrics(y_true, v2_probs)
    ens_metrics = compute_metrics(y_true, ens_probs)

    # 7. Report
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    for name, m in [("V1-only", v1_metrics), ("V2-only", v2_metrics), ("Ensemble", ens_metrics)]:
        log(f"\n  {name}:")
        log(f"    Overall accuracy:   {m['overall_accuracy']*100:.1f}%")
        log(f"    Mel sensitivity:    {m['mel_sensitivity']*100:.1f}%")
        log(f"    Mel specificity:    {m['mel_specificity']*100:.1f}%")
        log(f"    Mel TP={m['mel_tp']}  FN={m['mel_fn']}  FP={m['mel_fp']}  TN={m['mel_tn']}")

    # Improvement summary
    log(f"\n  Ensemble vs V1 accuracy delta: {(ens_metrics['overall_accuracy'] - v1_metrics['overall_accuracy'])*100:+.2f}pp")
    log(f"  Ensemble vs V2 accuracy delta: {(ens_metrics['overall_accuracy'] - v2_metrics['overall_accuracy'])*100:+.2f}pp")
    log(f"  Ensemble mel sensitivity delta vs V1: {(ens_metrics['mel_sensitivity'] - v1_metrics['mel_sensitivity'])*100:+.2f}pp")
    log(f"  Ensemble mel sensitivity delta vs V2: {(ens_metrics['mel_sensitivity'] - v2_metrics['mel_sensitivity'])*100:+.2f}pp")

    # 8. Save results
    results = {
        "metadata": {
            "dataset": "kuchikihater/HAM10000",
            "holdout": "15% stratified, seed=42",
            "holdout_size": len(test_idx),
            "mel_count": int(mel_count),
            "ensemble_weights": {"v1": V1_WEIGHT, "v2": V2_WEIGHT},
            "mel_safety_threshold": MEL_SAFETY_THRESHOLD,
            "device": str(device),
            "v1_model": str(V1_DIR),
            "v2_model": str(V2_DIR),
        },
        "v1_only": v1_metrics,
        "v2_only": v2_metrics,
        "ensemble": ens_metrics,
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    log(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
