#!/usr/bin/env python3
"""Temperature scaling calibration for DrAgnes v2.

Finds optimal temperature T that minimises NLL on the ISIC 2019 holdout,
producing well-calibrated probabilities: calibrated = softmax(logits / T).

Usage: python scripts/calibrate-temperature.py
"""
import json, time, warnings, os
from pathlib import Path
from collections import Counter
import numpy as np, torch
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from datasets import load_dataset, concatenate_datasets
from transformers import ViTForImageClassification, ViTImageProcessor

warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
SD = Path(__file__).resolve().parent
MD = SD / "dragnes-classifier-v2" / "best"
NC, BS = 7, 32
I2H = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:0, 7:6}

def img(row):
    for k in ("image", "img"):
        if k in row:
            v = row[k]
            if isinstance(v, Image.Image): return v.convert("RGB")
            if isinstance(v, dict) and "bytes" in v:
                import io; return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
            return v.convert("RGB")
    raise ValueError(f"no image column: {list(row.keys())}")

def ece(probs, labels, n_bins=10):
    mx, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (mx > lo) & (mx <= hi) if i else (mx >= lo) & (mx <= hi)
        n = m.sum()
        if n: total += abs(correct[m].mean() - mx[m].mean()) * n
    return total / len(labels)

def nll(logits, labels, T):
    scaled = logits / T
    log_p = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    return -log_p[np.arange(len(labels)), labels].mean()

def main():
    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[1/4] Loading model on {dev}")
    model = ViTForImageClassification.from_pretrained(str(MD)).to(dev).eval()
    proc = ViTImageProcessor.from_pretrained(str(MD))

    print("[2/4] Loading ISIC 2019 holdout (seed=42, 15%)")
    ds = load_dataset("akinsanyaayomide/skin_cancer_dataset_balanced_labels_2")
    full = concatenate_datasets([ds[s] for s in ds])
    labels = [I2H.get(int(r), -1) for r in full["label"]]
    _, tidx = train_test_split(range(len(full)), test_size=0.15, random_state=42, stratify=labels)
    items = [(i, labels[i]) for i in tidx if labels[i] >= 0]
    yt = np.array([l for _, l in items])
    print(f"   N={len(items)}  dist={dict(Counter(yt))}")

    print("[3/4] Collecting logits ...")
    all_logits = []
    for s in range(0, len(items), BS):
        b = items[s:s+BS]
        inp = proc(images=[img(full[i]) for i, _ in b], return_tensors="pt").to(dev)
        with torch.no_grad(): lo = model(**inp).logits
        all_logits.append(lo.cpu().numpy())
        if (s // BS) % 30 == 0: print(f"   {s+len(b)}/{len(items)}", end="\r")
    logits = np.concatenate(all_logits)
    print(f"   Collected {logits.shape} logits in {time.time()-t0:.0f}s")

    # Before calibration
    probs_before = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ece_before = ece(probs_before, yt)
    nll_before = nll(logits, yt, T=1.0)

    print("[4/4] Optimising temperature T ...")
    res = minimize_scalar(lambda T: nll(logits, yt, T), bounds=(0.1, 10.0), method="bounded")
    T_opt = res.x

    # After calibration
    scaled_logits = logits / T_opt
    probs_after = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
    ece_after = ece(probs_after, yt)
    nll_after = nll(logits, yt, T_opt)

    results = {
        "optimal_temperature": round(float(T_opt), 4),
        "ece_before": round(float(ece_before), 4),
        "ece_after": round(float(ece_after), 4),
        "nll_before": round(float(nll_before), 4),
        "nll_after": round(float(nll_after), 4),
    }
    r1 = SD / "calibration-results.json"
    r2 = SD / "calibration-temperature.json"
    with open(r1, "w") as f: json.dump(results, f, indent=2)
    with open(r2, "w") as f: json.dump({"temperature": round(float(T_opt), 4)}, f, indent=2)

    el = time.time() - t0
    print(f"\n{'='*56}")
    print(f"  Temperature Scaling Calibration Results")
    print(f"{'='*56}")
    print(f"  Optimal T       = {T_opt:.4f}")
    print(f"  ECE  before     = {ece_before:.4f}")
    print(f"  ECE  after      = {ece_after:.4f}  ({(1-ece_after/ece_before)*100:+.1f}%)")
    print(f"  NLL  before     = {nll_before:.4f}")
    print(f"  NLL  after      = {nll_after:.4f}")
    print(f"  Completed in {el:.0f}s")
    print(f"  Saved: {r1.name}, {r2.name}")
    print(f"{'='*56}")

if __name__ == "__main__": main()
