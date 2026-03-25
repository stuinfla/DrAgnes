#!/usr/bin/env python3
"""
ISIC 2019 Cross-Dataset Validation for Mela Classifier
=========================================================
Validates the Mela ViT model (trained on HAM10000) against a large
independent dermoscopy dataset with ISIC 2019-style 8-class labels
(MEL, NV, BCC, AK, BKL, DF, VASC, SCC) mapped to the HAM10000 7-class
scheme used by the model.

Key metrics: melanoma sensitivity, BCC sensitivity, all-cancer sensitivity,
per-class sensitivity/specificity/PPV/NPV, and multi-image voting improvement.

Dataset: akinsanyaayomide/skin_cancer_dataset_balanced_labels_2
  - 26,006 dermoscopy images (21,814 train + 4,192 test)
  - 8 ISIC-style classes: AK, BCC, BKL, DF, MEL, NV, SCC, VASC
  - Pre-resized to 224x224, parquet format
"""

import json
import sys
import time
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "mela-classifier" / "best"
RESULTS_PATH = SCRIPT_DIR / "isic2019-validation-results.json"

# Primary dataset (confirmed working: 26K images, 8 ISIC classes)
PRIMARY_DATASET = "akinsanyaayomide/skin_cancer_dataset_balanced_labels_2"

MAX_IMAGES = 5000
BATCH_SIZE = 32
PROGRESS_EVERY = 500
MULTI_IMAGE_N = 100
MULTI_IMAGE_VIEWS = 3

# HAM10000 7-class scheme (model output indices)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CANCER_CLASSES = {0, 1, 4}  # akiec, bcc, mel

# ISIC 2019 label names from the dataset feature metadata:
# 0=AK, 1=BCC, 2=BKL, 3=DF, 4=MEL, 5=NV, 6=SCC, 7=VASC
# Mapping: ISIC integer label -> HAM10000 class index
ISIC_INT_TO_HAM = {
    0: 0,  # AK -> akiec
    1: 1,  # BCC -> bcc
    2: 2,  # BKL -> bkl
    3: 3,  # DF -> df
    4: 4,  # MEL -> mel
    5: 5,  # NV -> nv
    6: 0,  # SCC -> akiec (closest HAM class: keratoses / carcinoma)
    7: 6,  # VASC -> vasc
}

ISIC_LABEL_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def compute_metrics(y_true, y_pred, num_classes=7):
    """Compute per-class sensitivity, specificity, PPV, NPV and confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    metrics = {}
    for c in range(num_classes):
        tp = cm[c][c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metrics[CLASS_NAMES[c]] = {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "support": int(cm[c].sum()),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
        }

    return metrics, cm


def apply_random_augmentation(image):
    """Apply random augmentation to simulate a different photo of the same lesion."""
    import torchvision.transforms.functional as TF

    # Random rotation (-30 to +30 degrees)
    angle = random.uniform(-30, 30)
    image = TF.rotate(image, angle)

    # Random horizontal flip (50%)
    if random.random() > 0.5:
        image = TF.hflip(image)

    # Random vertical flip (30%)
    if random.random() > 0.7:
        image = TF.vflip(image)

    # Random brightness adjustment (0.7 to 1.3)
    brightness = random.uniform(0.7, 1.3)
    image = TF.adjust_brightness(image, brightness)

    # Random contrast adjustment (0.8 to 1.2)
    contrast = random.uniform(0.8, 1.2)
    image = TF.adjust_contrast(image, contrast)

    # Random small crop and resize back
    if random.random() > 0.5:
        w, h = image.size
        crop_frac = random.uniform(0.85, 0.95)
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        image = TF.crop(image, top, left, ch, cw)
        image = TF.resize(image, [h, w])

    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 70)
    log("Mela ISIC 2019 Cross-Dataset Validation")
    log("=" * 70)
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"Model: {MODEL_DIR}")
    log(f"Max images: {MAX_IMAGES}")
    log(f"Batch size: {BATCH_SIZE}")
    log("")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    log("Loading model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"  Device: {device}")

    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(device)
    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model.eval()
    log(f"  Model loaded: ViT-base, 7 classes, {sum(p.numel() for p in model.parameters()):,} parameters")
    log("")

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    log(f"Loading dataset: {PRIMARY_DATASET} ...")
    from datasets import load_dataset

    ds = load_dataset(PRIMARY_DATASET)
    log(f"  Available splits: {list(ds.keys())}")
    for s in ds:
        log(f"    {s}: {len(ds[s])} examples")

    # Use test split for validation (independent from any training)
    # Then supplement with train split if needed to reach MAX_IMAGES
    test_data = ds["test"]
    train_data = ds["train"]

    # Verify label names match our expectation
    label_names = test_data.features["label"].names
    log(f"  Label names: {label_names}")
    assert label_names == ISIC_LABEL_NAMES, f"Unexpected labels: {label_names}"
    log(f"  Label mapping (ISIC -> HAM): {dict(zip(ISIC_LABEL_NAMES, [CLASS_NAMES[ISIC_INT_TO_HAM[i]] for i in range(8)]))}")
    log("")

    # ------------------------------------------------------------------
    # 3. Sample evenly across classes
    # ------------------------------------------------------------------
    log("Preparing sample...")

    # Group all images by HAM class across both splits
    # Prefer test split images first (truly independent), then train
    class_indices_test = defaultdict(list)
    class_indices_train = defaultdict(list)

    for i in range(len(test_data)):
        isic_label = test_data[i]["label"]
        ham_idx = ISIC_INT_TO_HAM[isic_label]
        class_indices_test[ham_idx].append(("test", i))

    for i in range(len(train_data)):
        isic_label = train_data[i]["label"]
        ham_idx = ISIC_INT_TO_HAM[isic_label]
        class_indices_train[ham_idx].append(("train", i))

    # Report full distribution
    log("  Full dataset distribution:")
    for c in range(7):
        n_test = len(class_indices_test.get(c, []))
        n_train = len(class_indices_train.get(c, []))
        log(f"    {CLASS_NAMES[c]:>6}: {n_test:>5} test + {n_train:>6} train = {n_test + n_train:>6} total")

    # Build combined pool per class, test first
    class_pool = defaultdict(list)
    for c in range(7):
        class_pool[c] = class_indices_test.get(c, []) + class_indices_train.get(c, [])

    total_available = sum(len(v) for v in class_pool.values())
    log(f"  Total mappable: {total_available}")

    # Even sampling across classes
    if total_available <= MAX_IMAGES:
        selected = []
        for c in range(7):
            selected.extend(class_pool[c])
        log(f"  Using all {len(selected)} images (under {MAX_IMAGES} limit)")
    else:
        n_classes = len(class_pool)
        per_class = MAX_IMAGES // n_classes
        selected = []
        remaining_budget = MAX_IMAGES

        # First pass: take min(per_class, available) from each class
        leftover = {}
        for c in range(7):
            avail = class_pool[c]
            take = min(per_class, len(avail))
            sampled = random.sample(avail, take)
            selected.extend(sampled)
            remaining_budget -= take
            if len(avail) > take:
                leftover[c] = [x for x in avail if x not in set(sampled)]

        # Second pass: distribute remaining proportionally
        if remaining_budget > 0 and leftover:
            total_left = sum(len(v) for v in leftover.values())
            for c, leftovers in leftover.items():
                extra = min(len(leftovers), int(remaining_budget * len(leftovers) / total_left))
                selected.extend(random.sample(leftovers, extra))

        log(f"  Sampled {len(selected)} images (even across classes)")

    random.shuffle(selected)

    # Count final distribution
    final_dist = Counter()
    for split, idx in selected:
        data = test_data if split == "test" else train_data
        isic_label = data[idx]["label"]
        ham_idx = ISIC_INT_TO_HAM[isic_label]
        final_dist[CLASS_NAMES[ham_idx]] += 1
    log(f"  Final distribution: {dict(sorted(final_dist.items()))}")

    test_count = sum(1 for s, _ in selected if s == "test")
    train_count = sum(1 for s, _ in selected if s == "train")
    log(f"  From test split: {test_count}, from train split: {train_count}")
    log("")

    # ------------------------------------------------------------------
    # 4. Run inference in batches
    # ------------------------------------------------------------------
    log(f"Running inference on {len(selected)} images (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    y_true = []
    y_pred = []
    y_probs = []
    errors = 0

    # Store images for multi-image test
    multi_image_candidates = []

    for batch_start in range(0, len(selected), BATCH_SIZE):
        batch_items = selected[batch_start:batch_start + BATCH_SIZE]
        batch_images = []
        batch_labels = []

        for split, idx in batch_items:
            try:
                data = test_data if split == "test" else train_data
                row = data[idx]
                img = row["image"]
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                isic_label = row["label"]
                ham_idx = ISIC_INT_TO_HAM[isic_label]

                batch_images.append(img)
                batch_labels.append(ham_idx)

                # Collect candidates for multi-image test
                if len(multi_image_candidates) < MULTI_IMAGE_N * 3:
                    multi_image_candidates.append((img.copy(), ham_idx))

            except Exception as e:
                errors += 1
                if errors <= 5:
                    log(f"  WARNING: Error loading index ({split},{idx}): {e}")
                continue

        if not batch_images:
            continue

        try:
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1).cpu().numpy()
                probs_np = probs.cpu().numpy()

            y_true.extend(batch_labels)
            y_pred.extend(preds.tolist())
            y_probs.extend(probs_np.tolist())

        except Exception as e:
            errors += 1
            log(f"  WARNING: Batch error at {batch_start}: {e}")
            # Fall back to single-image
            for img, lbl in zip(batch_images, batch_labels):
                try:
                    inputs = processor(images=[img], return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        pred = probs.argmax(dim=-1).cpu().item()
                        probs_np = probs.cpu().numpy()[0]
                    y_true.append(lbl)
                    y_pred.append(pred)
                    y_probs.append(probs_np.tolist())
                except Exception:
                    errors += 1

        processed = len(y_true)
        if processed % PROGRESS_EVERY < BATCH_SIZE:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            log(f"  [{processed:>5}/{len(selected)}] {elapsed:.1f}s, {rate:.1f} img/s, {errors} errors")

    t_end = time.time()
    inference_time = t_end - t_start
    rate = len(y_true) / inference_time if inference_time > 0 else 0

    log(f"\nInference complete:")
    log(f"  Total processed: {len(y_true)}")
    log(f"  Errors: {errors}")
    log(f"  Time: {inference_time:.1f}s ({rate:.1f} img/s)")
    log("")

    if len(y_true) == 0:
        log("ERROR: No images were successfully processed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Overall accuracy
    correct = int((y_true_np == y_pred_np).sum())
    accuracy = correct / len(y_true_np)
    log(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{len(y_true_np)})")

    # Per-class metrics
    per_class, cm = compute_metrics(y_true, y_pred, num_classes=7)

    log(f"\n{'Class':<10} {'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} {'Support':>8}")
    log("-" * 50)
    for c_name in CLASS_NAMES:
        m = per_class[c_name]
        if m["support"] > 0:
            log(f"{c_name:<10} {m['sensitivity']:>7.4f} {m['specificity']:>7.4f} "
                f"{m['ppv']:>7.4f} {m['npv']:>7.4f} {m['support']:>8d}")
        else:
            log(f"{c_name:<10} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {0:>8d}")

    # Key cancer metrics
    mel_m = per_class["mel"]
    bcc_m = per_class["bcc"]
    akiec_m = per_class["akiec"]

    log(f"\n--- KEY CANCER METRICS ---")
    log(f"Melanoma sensitivity:     {mel_m['sensitivity']:.4f} ({mel_m['tp']}/{mel_m['tp']+mel_m['fn']})")
    log(f"BCC sensitivity:          {bcc_m['sensitivity']:.4f} ({bcc_m['tp']}/{bcc_m['tp']+bcc_m['fn']})")
    log(f"AKIEC sensitivity:        {akiec_m['sensitivity']:.4f} ({akiec_m['tp']}/{akiec_m['tp']+akiec_m['fn']})")

    # All-cancer sensitivity: cancer case predicted as ANY cancer class
    cancer_true = y_true_np[np.isin(y_true_np, list(CANCER_CLASSES))]
    cancer_pred_for_cancer = y_pred_np[np.isin(y_true_np, list(CANCER_CLASSES))]
    cancer_detected = np.isin(cancer_pred_for_cancer, list(CANCER_CLASSES))
    all_cancer_sens = float(cancer_detected.sum() / len(cancer_true)) if len(cancer_true) > 0 else 0.0
    log(f"All-cancer sensitivity:   {all_cancer_sens:.4f} ({int(cancer_detected.sum())}/{len(cancer_true)})")

    # Confusion matrix
    log(f"\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':>10} " + " ".join(f"{c:>7}" for c in CLASS_NAMES)
    log(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>7d}" for v in row)
        log(f"{CLASS_NAMES[i]:>10} {row_str}")

    # ------------------------------------------------------------------
    # 6. Comparison with HAM10000 baseline
    # ------------------------------------------------------------------
    log(f"\n--- COMPARISON WITH HAM10000 BASELINE ---")
    ham_results = {
        "mel_sensitivity": 0.9822,
        "bcc_sensitivity": 0.9706,
        "akiec_sensitivity": 0.8939,
        "overall_accuracy": 0.7809,
        "source": "marmal88/skin_cancer (HAM10000 test split, 2004 images)"
    }

    log(f"{'Metric':<30} {'HAM10000':>10} {'External':>10} {'Delta':>10}")
    log("-" * 65)
    log(f"{'Overall accuracy':<30} {ham_results['overall_accuracy']:>10.4f} {accuracy:>10.4f} "
        f"{accuracy - ham_results['overall_accuracy']:>+10.4f}")
    log(f"{'Melanoma sensitivity':<30} {ham_results['mel_sensitivity']:>10.4f} {mel_m['sensitivity']:>10.4f} "
        f"{mel_m['sensitivity'] - ham_results['mel_sensitivity']:>+10.4f}")
    log(f"{'BCC sensitivity':<30} {ham_results['bcc_sensitivity']:>10.4f} {bcc_m['sensitivity']:>10.4f} "
        f"{bcc_m['sensitivity'] - ham_results['bcc_sensitivity']:>+10.4f}")
    log(f"{'AKIEC sensitivity':<30} {ham_results['akiec_sensitivity']:>10.4f} {akiec_m['sensitivity']:>10.4f} "
        f"{akiec_m['sensitivity'] - ham_results['akiec_sensitivity']:>+10.4f}")

    # ------------------------------------------------------------------
    # 7. Multi-image voting test
    # ------------------------------------------------------------------
    log(f"\n{'=' * 70}")
    log("MULTI-IMAGE VOTING TEST")
    log(f"{'=' * 70}")
    log(f"Simulating {MULTI_IMAGE_VIEWS} photos per lesion for {MULTI_IMAGE_N} test images")
    log(f"(Random augmentations: rotation, flip, brightness, contrast, crop)")
    log("")

    # Select diverse candidates: ensure melanoma representation
    mel_cands = [(img, lbl) for img, lbl in multi_image_candidates if lbl == 4]
    non_mel_cands = [(img, lbl) for img, lbl in multi_image_candidates if lbl != 4]

    n_mel = min(len(mel_cands), int(MULTI_IMAGE_N * 0.4))
    n_other = MULTI_IMAGE_N - n_mel

    multi_test_set = random.sample(mel_cands, n_mel) if n_mel > 0 else []
    if len(non_mel_cands) >= n_other:
        multi_test_set.extend(random.sample(non_mel_cands, n_other))
    else:
        multi_test_set.extend(non_mel_cands)

    log(f"Multi-image test set: {len(multi_test_set)} images "
        f"({n_mel} melanoma, {len(multi_test_set) - n_mel} other)")

    single_preds = []
    multi_preds = []
    multi_true = []

    for i, (img, true_label) in enumerate(multi_test_set):
        try:
            # Single-image prediction
            inputs = processor(images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                single_pred = outputs.logits.argmax(dim=-1).cpu().item()
            single_preds.append(single_pred)

            # Multi-image: classify original + augmented views
            view_preds = [single_pred]
            for _ in range(MULTI_IMAGE_VIEWS - 1):
                aug_img = apply_random_augmentation(img)
                inputs = processor(images=[aug_img], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    aug_pred = outputs.logits.argmax(dim=-1).cpu().item()
                view_preds.append(aug_pred)

            # Majority vote
            vote_counts = Counter(view_preds)
            majority_pred = vote_counts.most_common(1)[0][0]
            multi_preds.append(majority_pred)
            multi_true.append(true_label)

        except Exception as e:
            log(f"  WARNING: Multi-image error for image {i}: {e}")
            continue

    multi_true_np = np.array(multi_true)
    single_preds_np = np.array(single_preds)
    multi_preds_np = np.array(multi_preds)

    # Overall accuracy comparison
    single_acc = float((single_preds_np == multi_true_np).sum() / len(multi_true_np))
    multi_acc = float((multi_preds_np == multi_true_np).sum() / len(multi_true_np))

    log(f"\n{'Metric':<40} {'Single':>10} {'3-Image':>10} {'Delta':>10}")
    log("-" * 75)
    log(f"{'Overall accuracy':<40} {single_acc:>10.4f} {multi_acc:>10.4f} "
        f"{multi_acc - single_acc:>+10.4f}")

    # Melanoma sensitivity
    mel_mask = multi_true_np == 4
    single_mel_sens = None
    multi_mel_sens = None
    single_mel_cancer = None
    multi_mel_cancer = None

    if mel_mask.sum() > 0:
        single_mel_sens = float((single_preds_np[mel_mask] == 4).sum() / mel_mask.sum())
        multi_mel_sens = float((multi_preds_np[mel_mask] == 4).sum() / mel_mask.sum())
        log(f"{'Melanoma sensitivity':<40} {single_mel_sens:>10.4f} {multi_mel_sens:>10.4f} "
            f"{multi_mel_sens - single_mel_sens:>+10.4f}")

        single_mel_cancer = float(np.isin(single_preds_np[mel_mask], list(CANCER_CLASSES)).sum() / mel_mask.sum())
        multi_mel_cancer = float(np.isin(multi_preds_np[mel_mask], list(CANCER_CLASSES)).sum() / mel_mask.sum())
        log(f"{'Melanoma -> any cancer':<40} {single_mel_cancer:>10.4f} {multi_mel_cancer:>10.4f} "
            f"{multi_mel_cancer - single_mel_cancer:>+10.4f}")
    else:
        log("  No melanoma cases in multi-image test set")

    # BCC sensitivity
    bcc_mask = multi_true_np == 1
    single_bcc_sens = None
    multi_bcc_sens = None
    if bcc_mask.sum() > 0:
        single_bcc_sens = float((single_preds_np[bcc_mask] == 1).sum() / bcc_mask.sum())
        multi_bcc_sens = float((multi_preds_np[bcc_mask] == 1).sum() / bcc_mask.sum())
        log(f"{'BCC sensitivity':<40} {single_bcc_sens:>10.4f} {multi_bcc_sens:>10.4f} "
            f"{multi_bcc_sens - single_bcc_sens:>+10.4f}")

    # All-cancer sensitivity for multi-image
    cancer_mask = np.isin(multi_true_np, list(CANCER_CLASSES))
    single_cancer_sens = None
    multi_cancer_sens = None
    if cancer_mask.sum() > 0:
        single_cancer_sens = float(np.isin(single_preds_np[cancer_mask], list(CANCER_CLASSES)).sum() / cancer_mask.sum())
        multi_cancer_sens = float(np.isin(multi_preds_np[cancer_mask], list(CANCER_CLASSES)).sum() / cancer_mask.sum())
        log(f"{'All-cancer sensitivity':<40} {single_cancer_sens:>10.4f} {multi_cancer_sens:>10.4f} "
            f"{multi_cancer_sens - single_cancer_sens:>+10.4f}")

    does_multi_improve = multi_acc > single_acc
    log(f"\nDoes multi-image improve accuracy? {'YES' if does_multi_improve else 'NO'} "
        f"({multi_acc:.4f} vs {single_acc:.4f})")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(MODEL_DIR),
        "device": device,
        "dataset": {
            "name": PRIMARY_DATASET,
            "split_used": "test + train (test preferred)",
            "total_available": total_available,
            "total_processed": len(y_true),
            "from_test_split": test_count,
            "from_train_split": train_count,
            "errors": errors,
            "isic_label_names": ISIC_LABEL_NAMES,
            "class_mapping": {ISIC_LABEL_NAMES[k]: CLASS_NAMES[v] for k, v in ISIC_INT_TO_HAM.items()},
            "note": "SCC mapped to akiec (closest HAM10000 class for squamous cell carcinoma)",
        },
        "class_names": CLASS_NAMES,
        "inference": {
            "total_images": len(y_true),
            "batch_size": BATCH_SIZE,
            "time_seconds": round(inference_time, 2),
            "images_per_second": round(rate, 1),
        },
        "overall_accuracy": round(accuracy, 4),
        "per_class": per_class,
        "cancer_metrics": {
            "melanoma_sensitivity": mel_m["sensitivity"],
            "bcc_sensitivity": bcc_m["sensitivity"],
            "akiec_sensitivity": akiec_m["sensitivity"],
            "all_cancer_sensitivity": round(all_cancer_sens, 4),
            "cancer_support": int(len(cancer_true)),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "comparison_with_ham10000": {
            "ham10000_source": ham_results["source"],
            "ham10000_accuracy": ham_results["overall_accuracy"],
            "external_accuracy": round(accuracy, 4),
            "accuracy_delta": round(accuracy - ham_results["overall_accuracy"], 4),
            "ham10000_mel_sensitivity": ham_results["mel_sensitivity"],
            "external_mel_sensitivity": mel_m["sensitivity"],
            "mel_sensitivity_delta": round(mel_m["sensitivity"] - ham_results["mel_sensitivity"], 4),
            "ham10000_bcc_sensitivity": ham_results["bcc_sensitivity"],
            "external_bcc_sensitivity": bcc_m["sensitivity"],
            "bcc_sensitivity_delta": round(bcc_m["sensitivity"] - ham_results["bcc_sensitivity"], 4),
            "ham10000_akiec_sensitivity": ham_results["akiec_sensitivity"],
            "external_akiec_sensitivity": akiec_m["sensitivity"],
            "akiec_sensitivity_delta": round(akiec_m["sensitivity"] - ham_results["akiec_sensitivity"], 4),
        },
        "multi_image_voting": {
            "n_test_images": len(multi_true),
            "n_views": MULTI_IMAGE_VIEWS,
            "augmentations": ["rotation_-30_30", "horizontal_flip", "vertical_flip",
                              "brightness_0.7_1.3", "contrast_0.8_1.2", "random_crop_85_95pct"],
            "single_image_accuracy": round(single_acc, 4),
            "multi_image_accuracy": round(multi_acc, 4),
            "accuracy_improvement": round(multi_acc - single_acc, 4),
            "does_multi_improve": bool(does_multi_improve),
            "melanoma_n": int(mel_mask.sum()),
            "single_melanoma_sensitivity": round(single_mel_sens, 4) if single_mel_sens is not None else None,
            "multi_melanoma_sensitivity": round(multi_mel_sens, 4) if multi_mel_sens is not None else None,
            "melanoma_sensitivity_improvement": round(multi_mel_sens - single_mel_sens, 4) if single_mel_sens is not None and multi_mel_sens is not None else None,
            "single_melanoma_as_cancer": round(single_mel_cancer, 4) if single_mel_cancer is not None else None,
            "multi_melanoma_as_cancer": round(multi_mel_cancer, 4) if multi_mel_cancer is not None else None,
            "single_bcc_sensitivity": round(single_bcc_sens, 4) if single_bcc_sens is not None else None,
            "multi_bcc_sensitivity": round(multi_bcc_sens, 4) if multi_bcc_sens is not None else None,
            "single_cancer_sensitivity": round(single_cancer_sens, 4) if single_cancer_sens is not None else None,
            "multi_cancer_sensitivity": round(multi_cancer_sens, 4) if multi_cancer_sens is not None else None,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    log(f"\n{'=' * 70}")
    log("SUMMARY")
    log(f"{'=' * 70}")
    log(f"Dataset:                  {PRIMARY_DATASET} ({len(y_true)} images)")
    log(f"Overall accuracy:         {accuracy:.4f}")
    log(f"Melanoma sensitivity:     {mel_m['sensitivity']:.4f}  (HAM10000: {ham_results['mel_sensitivity']:.4f})")
    log(f"BCC sensitivity:          {bcc_m['sensitivity']:.4f}  (HAM10000: {ham_results['bcc_sensitivity']:.4f})")
    log(f"AKIEC sensitivity:        {akiec_m['sensitivity']:.4f}  (HAM10000: {ham_results['akiec_sensitivity']:.4f})")
    log(f"All-cancer sensitivity:   {all_cancer_sens:.4f}")
    if multi_mel_sens is not None:
        log(f"3-image mel sensitivity:  {multi_mel_sens:.4f}  (single: {single_mel_sens:.4f})")
    log(f"Multi-image improvement:  {'YES' if does_multi_improve else 'NO'} "
        f"(accuracy {single_acc:.4f} -> {multi_acc:.4f})")
    log(f"Inference speed:          {rate:.1f} img/s on {device}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
