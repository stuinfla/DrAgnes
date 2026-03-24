#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for DrAgnes v2 on ISIC 2019 external holdout."""
import json, time, warnings, os
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTImageProcessor

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "dragnes-classifier-v2" / "best"
OUT = SCRIPT_DIR / "confidence-intervals.json"
CLS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL, BS, CANCER = 4, 32, {0, 1, 4}
ISIC_INT_TO_HAM = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:0, 7:6}

def load_model():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(dev)
    p = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    m.eval()
    return m, p, dev

def infer(model, proc, dev, imgs):
    chunks = []
    for i in range(0, len(imgs), BS):
        inp = proc(images=imgs[i:i+BS], return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**inp).logits
        chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks)

def load_test_set():
    """Reproduce the exact 15% stratified ISIC 2019 holdout from train-combined.py."""
    ds = load_dataset("akinsanyaayomide/skin_cancer_dataset_balanced_labels_2")
    isic_full = concatenate_datasets([ds[s] for s in ds])
    raw = isic_full["label"]
    labels = [ISIC_INT_TO_HAM.get(int(r), -1) for r in raw]
    indices = list(range(len(isic_full)))
    _, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    imgs, lbls = [], []
    for i in test_idx:
        if labels[i] < 0:
            continue
        img = isic_full[i]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        imgs.append(img)
        lbls.append(labels[i])
    return imgs, np.array(lbls)

def bootstrap_ci(metric_fn, n_samples, n_boot=1000, seed=42):
    """Return (point, ci_lower, ci_upper) via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    point = metric_fn(np.arange(n_samples))
    vals = []
    for _ in range(n_boot):
        ix = rng.randint(0, n_samples, n_samples)
        try:
            vals.append(metric_fn(ix))
        except (ValueError, ZeroDivisionError):
            continue
    lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    return round(point, 4), round(lo, 4), round(hi, 4)

def main():
    print("=" * 60)
    print("DrAgnes v2 — Bootstrap 95% Confidence Intervals")
    print("=" * 60)

    model, proc, dev = load_model()
    print(f"Device: {dev}")

    print("Loading ISIC 2019 test set (15% holdout, seed=42)...")
    imgs, y_true = load_test_set()
    n = len(imgs)
    print(f"Test images: {n}")

    print("Running inference...", flush=True)
    t0 = time.time()
    probs = infer(model, proc, dev, imgs)
    preds = probs.argmax(axis=1)
    print(f"Inference: {time.time()-t0:.1f}s")

    # Precompute binary arrays
    is_mel = (y_true == MEL).astype(int)
    pred_mel = (preds == MEL).astype(int)
    is_cancer = np.isin(y_true, list(CANCER)).astype(int)
    pred_cancer = np.isin(preds, list(CANCER)).astype(int)

    # Metric functions operating on index arrays
    def mel_sens(ix):
        m, p = is_mel[ix], pred_mel[ix]
        return p[m == 1].sum() / m.sum()

    def mel_spec(ix):
        m, p = is_mel[ix], pred_mel[ix]
        neg = (m == 0)
        return (1 - p[neg]).sum() / neg.sum()

    def cancer_sens(ix):
        c, p = is_cancer[ix], pred_cancer[ix]
        return p[c == 1].sum() / c.sum()

    def accuracy(ix):
        return (preds[ix] == y_true[ix]).mean()

    def mel_auroc(ix):
        return roc_auc_score(is_mel[ix], probs[ix, MEL])

    print(f"\nBootstrapping (1000 iterations)...", flush=True)
    t0 = time.time()
    results = {}
    for name, fn in [
        ("melanoma_sensitivity", mel_sens),
        ("melanoma_specificity", mel_spec),
        ("all_cancer_sensitivity", cancer_sens),
        ("overall_accuracy", accuracy),
        ("melanoma_auroc", mel_auroc),
    ]:
        pt, lo, hi = bootstrap_ci(fn, n)
        results[name] = {"point": pt, "ci_lower": lo, "ci_upper": hi}
        print(f"  {name:30s}  {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"Bootstrap time: {time.time()-t0:.1f}s")

    out = {
        "dataset": "ISIC 2019 external holdout",
        "n_images": n,
        "n_bootstrap": 1000,
        "metrics": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")

if __name__ == "__main__":
    main()
