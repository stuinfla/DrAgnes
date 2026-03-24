#!/usr/bin/env python3
"""Compute AUROC for the DrAgnes ViT classifier on HAM10000 + ISIC 2019."""
import json, time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from transformers import ViTForImageClassification, ViTImageProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "dragnes-classifier" / "best"
OUT = SCRIPT_DIR / "auroc-results.json"
CLS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NC, MEL, BS = 7, 4, 32
DX = {"actinic_keratoses":0,"basal_cell_carcinoma":1,"benign_keratosis-like_lesions":2,
      "dermatofibroma":3,"melanoma":4,"melanocytic_Nevi":5,"vascular_lesions":6}
ISIC = {"AK":0,"BCC":1,"BKL":2,"DF":3,"MEL":4,"NV":5,"SCC":0,"VASC":6}

def load_model():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(dev)
    p = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    m.eval(); return m, p, dev

def infer(model, proc, dev, imgs):
    chunks = []
    for i in range(0, len(imgs), BS):
        inp = proc(images=imgs[i:i+BS], return_tensors="pt").to(dev)
        with torch.no_grad(): logits = model(**inp).logits
        chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks)

def boot_ci(yt, yp, n=1000, seed=42):
    base = roc_auc_score(yt, yp) if yt.ndim == 1 else roc_auc_score(yt, yp, average=None)
    rng, vals = np.random.RandomState(seed), []
    for _ in range(n):
        ix = rng.randint(0, len(yt), len(yt))
        a, b = yt[ix], yp[ix]
        skip = (a.ndim == 1 and len(np.unique(a)) < 2) or \
               (a.ndim > 1 and any(a[:,c].sum() in (0,len(a)) for c in range(a.shape[1])))
        if skip: continue
        try: vals.append(roc_auc_score(a, b) if a.ndim == 1 else roc_auc_score(a, b, average=None))
        except ValueError: pass
    if not vals: return (base.tolist() if hasattr(base,'tolist') else float(base)), None, None
    v = np.array(vals)
    lo, hi = np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0)
    r = lambda x: round(float(x),4) if np.ndim(x)==0 else [round(float(i),4) for i in x]
    return r(base), r(lo), r(hi)

def aurocs(y_true, y_prob):
    yb = label_binarize(y_true, classes=list(range(NC)))
    pc = {}
    for c in range(NC):
        if yb[:,c].sum() in (0, len(yb)):
            pc[CLS[c]] = {"auroc":None,"ci_lo":None,"ci_hi":None}; continue
        a, lo, hi = boot_ci(yb[:,c], y_prob[:,c])
        pc[CLS[c]] = {"auroc":a, "ci_lo":lo, "ci_hi":hi}
    wa = round(roc_auc_score(yb, y_prob, average="weighted", multi_class="ovr"), 4)
    ma = round(roc_auc_score(yb, y_prob, average="macro", multi_class="ovr"), 4)
    _, wlo, whi = boot_ci(yb, y_prob)
    mt = (np.array(y_true)==MEL).astype(int)
    mela, mlo, mhi = boot_ci(mt, y_prob[:,MEL])
    return {"per_class":pc, "weighted_auroc":wa, "weighted_ci":[wlo,whi],
            "macro_auroc":ma, "melanoma_binary":{"auroc":mela,"ci_lo":mlo,"ci_hi":mhi}}

def run_ham(model, proc, dev):
    from datasets import load_dataset
    print("Loading kuchikihater/HAM10000...", flush=True)
    ds = load_dataset("kuchikihater/HAM10000"); sp = ds["train"]
    imgs, labels = [], []
    for row in sp:
        lbl = row["label"]  # int 0-6, matches CLS order exactly
        img = row["image"]
        if img.mode != "RGB": img = img.convert("RGB")
        imgs.append(img); labels.append(lbl)
    print(f"  Total: {len(imgs)}", flush=True)
    _, ix, _, yt = train_test_split(list(range(len(imgs))), labels,
                                    test_size=0.15, stratify=labels, random_state=42)
    ti = [imgs[i] for i in ix]
    print(f"  Holdout: {len(ti)}", flush=True)
    t0 = time.time(); probs = infer(model, proc, dev, ti)
    print(f"  Inference: {time.time()-t0:.1f}s", flush=True)
    return aurocs(yt, probs)

def run_isic(model, proc, dev):
    from datasets import load_dataset
    print("Loading ISIC 2019...", flush=True)
    ds = load_dataset("akinsanyaayomide/skin_cancer_dataset_balanced_labels_2")
    imgs, labels, ln = [], [], None
    for sn in ds:
        sp = ds[sn]
        if ln is None and hasattr(sp.features["label"], "names"): ln = sp.features["label"].names
        for row in sp:
            if len(imgs) >= 4998: break
            ls = ln[row["label"]] if ln else str(row["label"])
            if ls not in ISIC: continue
            img = row["image"]
            if img.mode != "RGB": img = img.convert("RGB")
            imgs.append(img); labels.append(ISIC[ls])
        if len(imgs) >= 4998: break
    print(f"  Loaded: {len(imgs)}", flush=True)
    t0 = time.time(); probs = infer(model, proc, dev, imgs)
    print(f"  Inference: {time.time()-t0:.1f}s", flush=True)
    return aurocs(labels, probs)

def main():
    print("="*60+"\nDrAgnes AUROC Computation\n"+"="*60, flush=True)
    model, proc, dev = load_model()
    print(f"Device: {dev}", flush=True)
    res = {"ham10000": run_ham(model, proc, dev), "isic2019": run_isic(model, proc, dev)}
    for dn, r in res.items():
        print(f"\n--- {dn} ---", flush=True)
        print(f"  Weighted AUROC: {r['weighted_auroc']}", flush=True)
        print(f"  Macro AUROC:    {r['macro_auroc']}", flush=True)
        m = r["melanoma_binary"]
        print(f"  Melanoma AUROC: {m['auroc']} [{m['ci_lo']}, {m['ci_hi']}]", flush=True)
        for c, v in r["per_class"].items():
            if v["auroc"] is not None:
                print(f"    {c:>6}: {v['auroc']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]", flush=True)
    with open(OUT, "w") as f: json.dump(res, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)

if __name__ == "__main__":
    main()
