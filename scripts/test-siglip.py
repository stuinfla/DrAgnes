#!/usr/bin/env python3
"""
Mela Model Benchmark: SigLIP / SwinV2 / ViT-Melanoma

Tests three dermatology models against HAM10000 test images (last 15%, 30/class):
  1. skintaglabs/siglip-skin-lesion-classifier  (SigLIP 400M + XGBoost head, binary)
  2. TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019  (8-class ISIC)
  3. UnipaPolitoUnimore/vit-large-patch32-384-melanoma  (3-class melanoma-focused)

Usage: python3 scripts/test-siglip.py
"""

import os
import sys
import json
import time
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# ---- Paths ----
PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / ".cache" / "ham10000"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MAX_PER_CLASS = 30


def find_hf_token():
    """Find HuggingFace token from env or .env files."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]

    for env_path in [PROJECT_DIR / ".env", PROJECT_DIR.parent.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("\"'")
                if key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HuggingFace_Key") and val:
                    return val

    hf_cache = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache.exists():
        return hf_cache.read_text().strip()
    return None


def collect_test_images():
    """Collect test images: last 15% of each class from train/ or test/ split."""
    images = []
    for cls in CLASSES:
        test_dir = CACHE_DIR / "test" / cls
        train_dir = CACHE_DIR / "train" / cls

        if test_dir.exists():
            files = sorted(f for f in test_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        elif train_dir.exists():
            all_files = sorted(f for f in train_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            start = int(len(all_files) * 0.85)
            files = all_files[start:]
        else:
            flat_dir = CACHE_DIR / cls
            if flat_dir.exists():
                all_files = sorted(f for f in flat_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
                start = int(len(all_files) * 0.85)
                files = all_files[start:]
            else:
                files = []

        for f in files[:MAX_PER_CLASS]:
            images.append((f, cls))

    return images


def compute_metrics(confusion, classes, total_valid):
    """Compute sensitivity, specificity, PPV, NPV per class."""
    metrics = {}
    for cls in classes:
        tp = confusion.get(cls, {}).get(cls, 0)
        fn = sum(v for k, v in confusion.get(cls, {}).items()) - tp
        fp = sum(confusion.get(other, {}).get(cls, 0) for other in classes if other != cls)
        tn = total_valid - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metrics[cls] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                        "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
                        "n": tp + fn}
    return metrics


def print_confusion(confusion, classes):
    header = "         " + "".join(c.ljust(8) for c in classes)
    print(header)
    for actual in classes:
        row = actual.ljust(8) + " "
        for pred in classes:
            row += str(confusion.get(actual, {}).get(pred, 0)).ljust(8)
        print(row)


# ==================================================================
# Model 1: SigLIP SkinTag (SigLIP embeddings + XGBoost binary head)
# ==================================================================
def run_siglip_skintaglabs(images, hf_token):
    """
    skintaglabs/siglip-skin-lesion-classifier
    Binary: benign vs malignant.
    Uses frozen SigLIP embeddings + XGBoost classifier from Misc/.
    """
    print("\n  Loading skintaglabs/siglip-skin-lesion-classifier...")

    from huggingface_hub import hf_hub_download
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image

    # Download the XGBoost classifier (finetuned version is best: AUC 0.960)
    clf_path = hf_hub_download(
        "skintaglabs/siglip-skin-lesion-classifier",
        "Misc/xgboost_finetuned_binary.pkl",
        token=hf_token,
    )

    # Also try the condition classifier for 10-class estimation
    cond_clf_path = None
    try:
        cond_clf_path = hf_hub_download(
            "skintaglabs/siglip-skin-lesion-classifier",
            "Misc/xgboost_finetuned_condition.pkl",
            token=hf_token,
        )
    except Exception as e:
        print(f"    Could not download condition classifier: {e}")

    with open(clf_path, "rb") as f:
        binary_bundle = pickle.load(f)
    # The pickle is a dict with 'classifier' (XGBClassifier) and 'scaler' (StandardScaler)
    if isinstance(binary_bundle, dict):
        binary_clf = binary_bundle["classifier"]
        binary_scaler = binary_bundle.get("scaler")
        print(f"    Binary classifier loaded: {type(binary_clf).__name__} (classes={binary_clf.classes_})")
        if binary_scaler:
            print(f"    Binary scaler loaded: {type(binary_scaler).__name__} (n_features={binary_scaler.n_features_in_})")
    else:
        binary_clf = binary_bundle
        binary_scaler = None
        print(f"    Binary classifier loaded: {type(binary_clf).__name__}")

    cond_clf = None
    cond_scaler = None
    if cond_clf_path:
        with open(cond_clf_path, "rb") as f:
            cond_bundle = pickle.load(f)
        if isinstance(cond_bundle, dict):
            cond_clf = cond_bundle["classifier"]
            cond_scaler = cond_bundle.get("scaler")
            print(f"    Condition classifier loaded: {type(cond_clf).__name__} (classes={cond_clf.classes_})")
        else:
            cond_clf = cond_bundle
            print(f"    Condition classifier loaded: {type(cond_clf).__name__}")

    # Load the SigLIP model for embeddings
    # Check if fine-tuned model_state.pt gives us embeddings, or use base
    processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384", token=hf_token)

    # Try loading the fine-tuned state dict
    try:
        finetuned_path = hf_hub_download(
            "skintaglabs/siglip-skin-lesion-classifier",
            "model_state.pt",
            token=hf_token,
        )
        print("    Fine-tuned model_state.pt downloaded (3.5GB), loading...")
        base_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", token=hf_token)

        # The state dict uses 'backbone.' prefix: backbone.vision_model.xxx -> vision_model.xxx
        state = torch.load(finetuned_path, map_location="cpu", weights_only=False)

        # Strip 'backbone.' prefix to match AutoModel structure
        remapped = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                remapped[k[len("backbone."):]] = v
            # Skip 'head.*' keys -- those are the classification head, not needed for embeddings

        matched, total = 0, len(base_model.state_dict())
        missing = []
        for k in base_model.state_dict():
            if k in remapped:
                matched += 1
            else:
                missing.append(k)

        if matched > 0:
            load_result = base_model.load_state_dict(remapped, strict=False)
            print(f"    Fine-tuned weights loaded: {matched}/{total} keys matched")
            if load_result.unexpected_keys:
                print(f"    Unexpected keys (ignored): {len(load_result.unexpected_keys)}")
        else:
            print(f"    WARNING: No keys matched after remapping. Using base model.")

        model = base_model
    except Exception as e:
        print(f"    Could not load fine-tuned model: {e}")
        print("    Using base SigLIP model")
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", token=hf_token)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"    SigLIP model on device: {device}")

    # Run inference
    results = []
    t0 = time.time()

    for i, (img_path, label) in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                emb = outputs.pooler_output.cpu().numpy()

            # Scale embeddings before classification
            emb_scaled = binary_scaler.transform(emb) if binary_scaler is not None else emb

            # Binary prediction
            binary_proba = binary_clf.predict_proba(emb_scaled)[0]  # [benign, malignant]
            malignant_prob = float(binary_proba[1]) if len(binary_proba) > 1 else float(binary_proba[0])

            # Condition prediction if available
            condition = None
            if cond_clf is not None:
                try:
                    emb_cond = cond_scaler.transform(emb) if cond_scaler is not None else emb_scaled
                    cond_proba = cond_clf.predict_proba(emb_cond)[0]
                    cond_pred = int(cond_clf.predict(emb_cond)[0])
                    condition = {"pred": cond_pred, "proba": cond_proba.tolist()}
                except Exception:
                    pass

            results.append({
                "label": label,
                "malignant_prob": malignant_prob,
                "binary_pred": "malignant" if malignant_prob >= 0.5 else "benign",
                "condition": condition,
            })

        except Exception as e:
            results.append({"label": label, "malignant_prob": None, "binary_pred": None, "error": str(e)})

        if (i + 1) % 30 == 0 or i == len(images) - 1:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(images)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return results


def print_siglip_report(results):
    """Print binary triage report for SigLIP model."""
    # 10-class condition mapping from README
    # 0: Melanoma, 1: BCC, 2: SCC, 3: Actinic Keratosis,
    # 4: Melanocytic Nevus, 5: Seborrheic Keratosis, 6: Dermatofibroma,
    # 7: Vascular Lesion, 8: Non-Neoplastic, 9: Other/Unknown
    COND_TO_HAM = {
        0: "mel", 1: "bcc", 2: "akiec", 3: "akiec",
        4: "nv", 5: "bkl", 6: "df", 7: "vasc",
        8: None, 9: None,  # no direct HAM mapping
    }
    COND_NAMES = {
        0: "Melanoma", 1: "BCC", 2: "SCC", 3: "Actinic Keratosis",
        4: "Melanocytic Nevus", 5: "Seborrheic Keratosis", 6: "Dermatofibroma",
        7: "Vascular Lesion", 8: "Non-Neoplastic", 9: "Other/Unknown",
    }

    print(f"\n--- skintaglabs/siglip-skin-lesion-classifier ---")
    print(f"  (Binary triage: benign vs malignant + 10-class condition)")

    valid = [r for r in results if r.get("malignant_prob") is not None]
    print(f"  Valid predictions: {len(valid)}/{len(results)}")

    if not valid:
        print("  NO VALID PREDICTIONS")
        print(f"\n  *** MELANOMA SENSITIVITY: N/A ***")
        print(f"  Status: FAIL (no data)")
        return {"mel_sens": None, "mel_spec": None}

    # For the binary model: malignant = mel + bcc + akiec; benign = nv + bkl + df + vasc
    # But the key metric is: does it flag melanomas?
    # Actually their training treats: MEL, BCC, SCC, AK as malignant
    malignant_classes = {"mel", "bcc", "akiec"}  # SCC maps to akiec in HAM
    benign_classes = {"nv", "bkl", "df", "vasc"}

    # Melanoma-specific sensitivity
    mel_results = [r for r in valid if r["label"] == "mel"]
    mel_detected = sum(1 for r in mel_results if r["malignant_prob"] >= 0.5)
    mel_n = len(mel_results)
    mel_sens = mel_detected / mel_n if mel_n > 0 else 0

    # Overall malignant sensitivity
    mal_results = [r for r in valid if r["label"] in malignant_classes]
    mal_detected = sum(1 for r in mal_results if r["malignant_prob"] >= 0.5)
    mal_n = len(mal_results)
    mal_sens = mal_detected / mal_n if mal_n > 0 else 0

    # Benign specificity
    ben_results = [r for r in valid if r["label"] in benign_classes]
    ben_correct = sum(1 for r in ben_results if r["malignant_prob"] < 0.5)
    ben_n = len(ben_results)
    ben_spec = ben_correct / ben_n if ben_n > 0 else 0

    # Per-class predicted-as-malignant rate
    print(f"\n  Per-class predicted-as-malignant rate:")
    print(f"  {'Class':<8} {'PredMal':<10} {'PredBen':<10} {'MalRate':<10} {'AvgProb':<10} {'N':<6}")
    print(f"  {'-'*54}")
    for cls in CLASSES:
        cls_r = [r for r in valid if r["label"] == cls]
        pred_mal = sum(1 for r in cls_r if r["malignant_prob"] >= 0.5)
        pred_ben = len(cls_r) - pred_mal
        avg_prob = np.mean([r["malignant_prob"] for r in cls_r]) if cls_r else 0
        mal_rate = (pred_mal / len(cls_r) * 100) if cls_r else 0
        print(f"  {cls:<8} {pred_mal:<10} {pred_ben:<10} {mal_rate:>5.1f}%    {avg_prob:>5.3f}     {len(cls_r)}")

    # Thresholded analysis
    print(f"\n  Threshold analysis for melanoma:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mel_det = sum(1 for r in mel_results if r["malignant_prob"] >= thresh)
        ben_corr = sum(1 for r in ben_results if r["malignant_prob"] < thresh)
        s = mel_det / mel_n if mel_n > 0 else 0
        sp = ben_corr / ben_n if ben_n > 0 else 0
        print(f"    threshold={thresh:.1f}: mel_sens={s*100:.1f}% benign_spec={sp*100:.1f}%")

    status = "PASS (>=90%)" if mel_sens >= 0.90 else "MARGINAL (80-90%)" if mel_sens >= 0.80 else "FAIL (<80%)"
    print(f"\n  Overall malignant sensitivity: {mal_sens*100:.1f}% ({mal_detected}/{mal_n})")
    print(f"  Benign specificity: {ben_spec*100:.1f}% ({ben_correct}/{ben_n})")
    print(f"\n  *** MELANOMA SENSITIVITY: {mel_sens*100:.1f}% ({mel_detected}/{mel_n}) ***")
    print(f"  *** MELANOMA SPECIFICITY (benign spec): {ben_spec*100:.1f}% ***")
    print(f"  Status: {status}")

    # 10-class condition analysis
    cond_results = [r for r in valid if r.get("condition")]
    if cond_results:
        print(f"\n  --- 10-class condition analysis ({len(cond_results)} images) ---")
        # Build confusion matrix: actual HAM class -> predicted condition -> HAM
        cond_confusion = defaultdict(lambda: defaultdict(int))
        cond_correct = 0
        cond_valid = 0
        for r in cond_results:
            pred_cond = r["condition"]["pred"]
            ham_pred = COND_TO_HAM.get(pred_cond)
            if ham_pred is not None:
                cond_confusion[r["label"]][ham_pred] += 1
                if ham_pred == r["label"]:
                    cond_correct += 1
                cond_valid += 1

        if cond_valid > 0:
            cond_accuracy = cond_correct / cond_valid
            print(f"  Condition accuracy (mapped to 7-class): {cond_accuracy*100:.1f}% ({cond_correct}/{cond_valid})")

            cond_metrics = compute_metrics(dict(cond_confusion), CLASSES, cond_valid)
            print(f"\n  Per-class metrics (condition head):")
            print(f"  {'Class':<8} {'Sens':<8} {'Spec':<8} {'N':<6}")
            print(f"  {'-'*28}")
            for cls in CLASSES:
                m = cond_metrics[cls]
                print(f"  {cls:<8} {m['sens']*100:>5.1f}%  {m['spec']*100:>5.1f}%  {m['n']:<6}")

            cond_mel = cond_metrics["mel"]
            print(f"\n  *** CONDITION-HEAD MELANOMA SENSITIVITY: {cond_mel['sens']*100:.1f}% ({cond_mel['tp']}/{cond_mel['n']}) ***")

    return {"mel_sens": mel_sens, "mel_spec": ben_spec, "mal_sens": mal_sens}


# ==================================================================
# Model 2: SwinV2 ISIC-2019 (8-class, standard transformers)
# ==================================================================
def run_swinv2(images, hf_token):
    """TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019"""
    print("\n  Loading SwinV2-ISIC2019...")

    import torch
    from transformers import pipeline as hf_pipeline
    from PIL import Image

    device = 0 if torch.backends.mps.is_available() else -1  # 0 = first GPU/MPS
    # SwinV2 may not work well on MPS; use CPU if needed
    try:
        pipe = hf_pipeline(
            "image-classification",
            model="TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019",
            token=hf_token,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )
        print(f"    Model loaded on MPS")
    except Exception:
        pipe = hf_pipeline(
            "image-classification",
            model="TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019",
            token=hf_token,
            device="cpu",
        )
        print(f"    Model loaded on CPU (MPS fallback)")

    # id2label: {0: AK, 1: BCC, 2: BKL, 3: DF, 4: MEL, 5: NV, 6: SCC, 7: VASC}
    label_map = {
        "AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df",
        "MEL": "mel", "NV": "nv", "SCC": "akiec", "VASC": "vasc",
    }

    results = []
    t0 = time.time()

    for i, (img_path, label) in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            preds = pipe(img, top_k=8)
            # Map to HAM classes
            mapped = {}
            for p in preds:
                ham_cls = label_map.get(p["label"])
                if ham_cls:
                    mapped[ham_cls] = mapped.get(ham_cls, 0) + p["score"]

            if mapped:
                top = max(mapped, key=mapped.get)
            else:
                top = None

            results.append({"label": label, "predicted": top, "probs": mapped, "raw": preds[:3]})
        except Exception as e:
            results.append({"label": label, "predicted": None, "error": str(e)})

        if (i + 1) % 30 == 0 or i == len(images) - 1:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(images)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return results


# ==================================================================
# Model 3: ViT-Large Melanoma (3-class)
# ==================================================================
def run_vit_melanoma(images, hf_token):
    """UnipaPolitoUnimore/vit-large-patch32-384-melanoma"""
    print("\n  Loading ViT-Large-Melanoma...")

    import torch
    from transformers import pipeline as hf_pipeline
    from PIL import Image

    try:
        pipe = hf_pipeline(
            "image-classification",
            model="UnipaPolitoUnimore/vit-large-patch32-384-melanoma",
            token=hf_token,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )
        print(f"    Model loaded on MPS")
    except Exception:
        pipe = hf_pipeline(
            "image-classification",
            model="UnipaPolitoUnimore/vit-large-patch32-384-melanoma",
            token=hf_token,
            device="cpu",
        )
        print(f"    Model loaded on CPU (MPS fallback)")

    # id2label: {0: Melanoma, 1: Nevus, 2: Seborrheic_keratosis}
    # This only covers 3 of 7 HAM classes
    label_map = {
        "Melanoma": "mel",
        "Nevus": "nv",
        "Seborrheic_keratosis": "bkl",
    }

    results = []
    t0 = time.time()

    for i, (img_path, label) in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            preds = pipe(img, top_k=3)

            mapped = {}
            for p in preds:
                ham_cls = label_map.get(p["label"])
                if ham_cls:
                    mapped[ham_cls] = mapped.get(ham_cls, 0) + p["score"]

            # Also extract melanoma probability directly
            mel_prob = mapped.get("mel", 0.0)

            if mapped:
                top = max(mapped, key=mapped.get)
            else:
                top = None

            results.append({
                "label": label,
                "predicted": top,
                "mel_prob": mel_prob,
                "probs": mapped,
                "raw": preds[:3],
            })
        except Exception as e:
            results.append({"label": label, "predicted": None, "mel_prob": None, "error": str(e)})

        if (i + 1) % 30 == 0 or i == len(images) - 1:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(images)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return results


# ==================================================================
# Report printers
# ==================================================================
def print_multiclass_report(model_name, results, n_model_classes=None):
    """Print standard 7-class report for a multi-class model."""
    print(f"\n--- {model_name} ---")

    valid = [r for r in results if r.get("predicted") and r["predicted"] in CLASSES]
    print(f"  Valid predictions: {len(valid)}/{len(results)}")

    if not valid:
        print("  NO VALID PREDICTIONS")
        err = next((r.get("error") for r in results if r.get("error")), None)
        if err:
            print(f"  Sample error: {err[:200]}")
        print(f"\n  *** MELANOMA SENSITIVITY: N/A ***")
        print(f"  Status: FAIL (no data)")
        return {"mel_sens": None, "mel_spec": None, "accuracy": None}

    # Sample raw labels
    sample = next((r for r in results if r.get("raw")), None)
    if sample:
        print(f"  Sample raw labels:")
        for p in sample["raw"][:5]:
            print(f"    \"{p.get('label', '?')}\" -> score={p.get('score', 0):.4f}")

    # Overall accuracy
    correct = sum(1 for r in valid if r["predicted"] == r["label"])
    accuracy = correct / len(valid)
    print(f"  Overall accuracy: {accuracy*100:.1f}%")

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for r in valid:
        confusion[r["label"]][r["predicted"]] += 1

    metrics = compute_metrics(dict(confusion), CLASSES, len(valid))

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<8} {'Sens':<8} {'Spec':<8} {'N':<6}")
    print(f"  {'-'*28}")
    for cls in CLASSES:
        m = metrics[cls]
        print(f"  {cls:<8} {m['sens']*100:>5.1f}%  {m['spec']*100:>5.1f}%  {m['n']:<6}")

    mel = metrics["mel"]
    mel_sens = mel["sens"]
    mel_spec = mel["spec"]
    status = "PASS (>=90%)" if mel_sens >= 0.90 else "MARGINAL (80-90%)" if mel_sens >= 0.80 else "FAIL (<80%)"
    print(f"\n  *** MELANOMA SENSITIVITY: {mel_sens*100:.1f}% ({mel['tp']}/{mel['n']}) ***")
    print(f"  *** MELANOMA SPECIFICITY: {mel_spec*100:.1f}% ***")
    print(f"  Status: {status}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print_confusion(dict(confusion), CLASSES)

    return {"mel_sens": mel_sens, "mel_spec": mel_spec, "accuracy": accuracy}


def print_vit_melanoma_report(results):
    """
    Print report for ViT-Melanoma model (3-class: Melanoma, Nevus, Seb. Keratosis).
    Since it only has 3 classes, we show both the 3-class mapping AND
    the melanoma-as-binary-triage view.
    """
    print(f"\n--- UnipaPolitoUnimore/vit-large-patch32-384-melanoma ---")
    print(f"  (3-class: Melanoma, Nevus, Seborrheic keratosis)")

    valid = [r for r in results if r.get("predicted") is not None]
    print(f"  Valid predictions: {len(valid)}/{len(results)}")

    if not valid:
        print("  NO VALID PREDICTIONS")
        print(f"\n  *** MELANOMA SENSITIVITY: N/A ***")
        print(f"  Status: FAIL (no data)")
        return {"mel_sens": None, "mel_spec": None}

    # Show raw labels
    sample = next((r for r in results if r.get("raw")), None)
    if sample:
        print(f"  Sample raw labels:")
        for p in sample["raw"][:3]:
            print(f"    \"{p.get('label', '?')}\" -> score={p.get('score', 0):.4f}")

    # Model only outputs mel, nv, bkl - other classes get forced into these
    # Show the mapping confusion
    print(f"\n  Per-class prediction distribution (3-class model):")
    print(f"  {'Actual':<8} {'->mel':<8} {'->nv':<8} {'->bkl':<8} {'N':<6}")
    print(f"  {'-'*38}")
    for cls in CLASSES:
        cls_r = [r for r in valid if r["label"] == cls]
        pred_mel = sum(1 for r in cls_r if r["predicted"] == "mel")
        pred_nv = sum(1 for r in cls_r if r["predicted"] == "nv")
        pred_bkl = sum(1 for r in cls_r if r["predicted"] == "bkl")
        print(f"  {cls:<8} {pred_mel:<8} {pred_nv:<8} {pred_bkl:<8} {len(cls_r)}")

    # Melanoma sensitivity (binary: did it predict mel for actual mel?)
    mel_results = [r for r in valid if r["label"] == "mel"]
    mel_detected = sum(1 for r in mel_results if r["predicted"] == "mel")
    mel_n = len(mel_results)
    mel_sens = mel_detected / mel_n if mel_n > 0 else 0

    # Melanoma specificity (did it NOT predict mel for non-mel?)
    nonmel_results = [r for r in valid if r["label"] != "mel"]
    nonmel_correct = sum(1 for r in nonmel_results if r["predicted"] != "mel")
    nonmel_n = len(nonmel_results)
    mel_spec = nonmel_correct / nonmel_n if nonmel_n > 0 else 0

    # Threshold analysis on mel_prob
    print(f"\n  Threshold analysis for melanoma (using mel_prob):")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        mel_det = sum(1 for r in mel_results if (r.get("mel_prob") or 0) >= thresh)
        nonmel_corr = sum(1 for r in nonmel_results if (r.get("mel_prob") or 0) < thresh)
        s = mel_det / mel_n if mel_n > 0 else 0
        sp = nonmel_corr / nonmel_n if nonmel_n > 0 else 0
        print(f"    threshold={thresh:.1f}: mel_sens={s*100:.1f}% mel_spec={sp*100:.1f}%")

    status = "PASS (>=90%)" if mel_sens >= 0.90 else "MARGINAL (80-90%)" if mel_sens >= 0.80 else "FAIL (<80%)"
    print(f"\n  *** MELANOMA SENSITIVITY: {mel_sens*100:.1f}% ({mel_detected}/{mel_n}) ***")
    print(f"  *** MELANOMA SPECIFICITY: {mel_spec*100:.1f}% ***")
    print(f"  Status: {status}")

    return {"mel_sens": mel_sens, "mel_spec": mel_spec}


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 70)
    print("  Mela Model Benchmark: SigLIP / SwinV2 / ViT-Melanoma")
    print("  Running LOCALLY with transformers (not via HF Inference API)")
    print("=" * 70)

    hf_token = find_hf_token()
    if not hf_token:
        print("\nERROR: No HuggingFace API key found.")
        print("Set HF_TOKEN or put HuggingFace_Key in .env")
        sys.exit(1)
    print(f"\nHF Token: {hf_token[:5]}...{hf_token[-4:]} ({len(hf_token)} chars)")

    # Collect images
    images = collect_test_images()
    print(f"\nCollected {len(images)} test images (last 15%, max {MAX_PER_CLASS}/class)")
    per_class = defaultdict(int)
    for _, cls in images:
        per_class[cls] += 1
    print("Per class:", ", ".join(f"{c}: {per_class[c]}" for c in CLASSES))

    if not images:
        print("No test images found!")
        sys.exit(1)

    total_start = time.time()
    all_metrics = {}

    # ---- Model 1: SigLIP SkinTagLabs ----
    print("\n" + "=" * 70)
    print("  MODEL 1: skintaglabs/siglip-skin-lesion-classifier")
    print("=" * 70)
    try:
        siglip_results = run_siglip_skintaglabs(images, hf_token)
        all_metrics["siglip"] = print_siglip_report(siglip_results)
    except Exception as e:
        print(f"\n  FATAL ERROR running SigLIP: {e}")
        import traceback; traceback.print_exc()
        all_metrics["siglip"] = {"mel_sens": None, "mel_spec": None}

    # ---- Model 2: SwinV2 ISIC-2019 ----
    print("\n" + "=" * 70)
    print("  MODEL 2: TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019")
    print("=" * 70)
    try:
        swinv2_results = run_swinv2(images, hf_token)
        all_metrics["swinv2"] = print_multiclass_report(
            "TriDat/swinv2-base-patch4-window12-192-22k-finetuned-lora-ISIC-2019",
            swinv2_results
        )
    except Exception as e:
        print(f"\n  FATAL ERROR running SwinV2: {e}")
        import traceback; traceback.print_exc()
        all_metrics["swinv2"] = {"mel_sens": None, "mel_spec": None}

    # ---- Model 3: ViT-Large Melanoma ----
    print("\n" + "=" * 70)
    print("  MODEL 3: UnipaPolitoUnimore/vit-large-patch32-384-melanoma")
    print("=" * 70)
    try:
        vit_results = run_vit_melanoma(images, hf_token)
        all_metrics["vit_mel"] = print_vit_melanoma_report(vit_results)
    except Exception as e:
        print(f"\n  FATAL ERROR running ViT-Melanoma: {e}")
        import traceback; traceback.print_exc()
        all_metrics["vit_mel"] = {"mel_sens": None, "mel_spec": None}

    # ---- Summary ----
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n  {'Model':<35} {'MelSens':<10} {'MelSpec':<10} {'Type':<12}")
    print(f"  {'-'*67}")

    for name, key in [
        ("SigLIP-SkinTag (binary)", "siglip"),
        ("SwinV2-ISIC2019 (8-class)", "swinv2"),
        ("ViT-L-Melanoma (3-class)", "vit_mel"),
    ]:
        m = all_metrics.get(key, {})
        sens = f"{m['mel_sens']*100:.1f}%" if m.get("mel_sens") is not None else "N/A"
        spec = f"{m['mel_spec']*100:.1f}%" if m.get("mel_spec") is not None else "N/A"
        acc = f"{m['accuracy']*100:.1f}%" if m.get("accuracy") is not None else "binary" if "binary" in name else "3-class"
        print(f"  {name:<35} {sens:<10} {spec:<10} {acc:<12}")

    print(f"  {'DermaSensor (benchmark)':<35} {'95.5%':<10} {'32.5%':<10} {'device':<12}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 70)

    # Save results
    output_path = PROJECT_DIR / "scripts" / "siglip-test-results.json"
    output_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_images": len(images),
        "metrics": {k: {kk: (float(vv) if isinstance(vv, (int, float)) and vv is not None else vv)
                        for kk, vv in v.items()} for k, v in all_metrics.items()},
    }, indent=2, default=str))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
