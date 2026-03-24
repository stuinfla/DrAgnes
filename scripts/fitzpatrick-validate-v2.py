#!/usr/bin/env python3
"""
ADR-124: Fitzpatrick Skin Tone Equity Validation — DrAgnes v2 Combined Model
=============================================================================
Validates the DrAgnes v2 ViT-Base classifier across all six Fitzpatrick skin
types (I-VI) using a stratified sample from Fitzpatrick17k.

Steps:
  1. Download Fitzpatrick17k CSV from GitHub (mattgroh/fitzpatrick17k)
  2. Filter to the ~3,413 images that map to our 7-class scheme
  3. Download a stratified sample: up to 50 images per class per skin type
  4. Run v2 model inference on each downloaded image (MPS device)
  5. Report performance BY Fitzpatrick type (I-VI):
     - Per-type: melanoma sensitivity, specificity, overall accuracy
     - Equity gap: max difference between any two skin types
     - Flag CRITICAL if gap > 5%, DANGEROUS if gap > 10%
  6. Save results to scripts/fitzpatrick-v2-validation.json

Usage:
    python3 examples/dragnes/scripts/fitzpatrick-validate-v2.py
"""

import csv
import io
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "dragnes-classifier-v2" / "best"
MANIFEST_PATH = SCRIPT_DIR / "dataset-manifests" / "fitzpatrick17k.json"
RESULTS_PATH = SCRIPT_DIR / "fitzpatrick-v2-validation.json"
IMAGE_CACHE_DIR = SCRIPT_DIR / ".fitzpatrick-image-cache"

CSV_URL = (
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/"
    "master/fitzpatrick17k.csv"
)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic Keratosis / SCC",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesion",
}
CANCER_CLASSES = {"akiec", "bcc", "mel"}

SKIN_TYPE_LABELS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
VALID_SKIN_TYPES = [1, 2, 3, 4, 5, 6]

MAX_PER_CLASS_PER_TYPE = 50
DOWNLOAD_TIMEOUT = 15  # seconds per image
DOWNLOAD_WORKERS = 8
PROGRESS_EVERY = 50
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def download_image(url: str, cache_path: Path) -> Image.Image | None:
    """Download a single image, using a file-based cache to avoid re-downloads."""
    if cache_path.exists():
        try:
            img = Image.open(cache_path).convert("RGB")
            return img
        except Exception:
            cache_path.unlink(missing_ok=True)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "DrAgnes-Research/1.0"
                )
            },
        )
        resp = urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT)
        data = resp.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception:
        return None


def compute_per_type_metrics(
    y_true: list[int],
    y_pred: list[int],
    skin_types: list[int],
    num_classes: int = 7,
) -> dict[str, Any]:
    """Compute per-Fitzpatrick-type classification metrics."""
    results_by_type: dict[str, Any] = {}
    mel_idx = CLASS_NAMES.index("mel")

    for st in VALID_SKIN_TYPES:
        mask = [i for i, s in enumerate(skin_types) if s == st]
        if not mask:
            results_by_type[SKIN_TYPE_LABELS[st]] = {
                "n_images": 0,
                "accuracy": None,
                "melanoma_sensitivity": None,
                "melanoma_specificity": None,
            }
            continue

        yt = [y_true[i] for i in mask]
        yp = [y_pred[i] for i in mask]
        n = len(yt)
        correct = sum(1 for t, p in zip(yt, yp) if t == p)
        accuracy = correct / n

        # Melanoma sensitivity: TP_mel / (TP_mel + FN_mel)
        mel_true_pos = sum(1 for t, p in zip(yt, yp) if t == mel_idx and p == mel_idx)
        mel_false_neg = sum(1 for t, p in zip(yt, yp) if t == mel_idx and p != mel_idx)
        mel_support = mel_true_pos + mel_false_neg
        mel_sensitivity = mel_true_pos / mel_support if mel_support > 0 else None

        # Melanoma specificity: TN_mel / (TN_mel + FP_mel)
        mel_false_pos = sum(1 for t, p in zip(yt, yp) if t != mel_idx and p == mel_idx)
        mel_true_neg = sum(1 for t, p in zip(yt, yp) if t != mel_idx and p != mel_idx)
        mel_spec_denom = mel_true_neg + mel_false_pos
        mel_specificity = mel_true_neg / mel_spec_denom if mel_spec_denom > 0 else None

        # Per-class breakdown for this skin type
        per_class = {}
        for c in range(num_classes):
            tp = sum(1 for t, p in zip(yt, yp) if t == c and p == c)
            fn = sum(1 for t, p in zip(yt, yp) if t == c and p != c)
            fp = sum(1 for t, p in zip(yt, yp) if t != c and p == c)
            tn = sum(1 for t, p in zip(yt, yp) if t != c and p != c)
            support = tp + fn
            per_class[CLASS_NAMES[c]] = {
                "sensitivity": round(tp / support, 4) if support > 0 else None,
                "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
                "support": support,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }

        # Cancer sensitivity for this skin type
        cancer_idxs = {CLASS_NAMES.index(c) for c in CANCER_CLASSES}
        cancer_true = sum(1 for t in yt if t in cancer_idxs)
        cancer_detected = sum(
            1 for t, p in zip(yt, yp) if t in cancer_idxs and p in cancer_idxs
        )
        cancer_sensitivity = cancer_detected / cancer_true if cancer_true > 0 else None

        results_by_type[SKIN_TYPE_LABELS[st]] = {
            "n_images": n,
            "accuracy": round(accuracy, 4),
            "melanoma_sensitivity": round(mel_sensitivity, 4) if mel_sensitivity is not None else None,
            "melanoma_specificity": round(mel_specificity, 4) if mel_specificity is not None else None,
            "melanoma_support": mel_support,
            "cancer_sensitivity": round(cancer_sensitivity, 4) if cancer_sensitivity is not None else None,
            "cancer_support": cancer_true,
            "per_class": per_class,
        }

    return results_by_type


def compute_equity_gaps(results_by_type: dict[str, Any]) -> dict[str, Any]:
    """Compute equity gaps across skin types."""
    gaps: dict[str, Any] = {}

    # Accuracy gap
    accuracies = {
        st: r["accuracy"]
        for st, r in results_by_type.items()
        if r["accuracy"] is not None and r["n_images"] >= 10
    }
    if len(accuracies) >= 2:
        max_acc = max(accuracies.values())
        min_acc = min(accuracies.values())
        gap = max_acc - min_acc
        gaps["accuracy"] = {
            "max": {"type": max(accuracies, key=accuracies.get), "value": max_acc},
            "min": {"type": min(accuracies, key=accuracies.get), "value": min_acc},
            "gap": round(gap, 4),
            "gap_pct": round(gap * 100, 2),
            "severity": _severity(gap),
        }

    # Melanoma sensitivity gap
    mel_sens = {
        st: r["melanoma_sensitivity"]
        for st, r in results_by_type.items()
        if r["melanoma_sensitivity"] is not None and r.get("melanoma_support", 0) >= 5
    }
    if len(mel_sens) >= 2:
        max_s = max(mel_sens.values())
        min_s = min(mel_sens.values())
        gap = max_s - min_s
        gaps["melanoma_sensitivity"] = {
            "max": {"type": max(mel_sens, key=mel_sens.get), "value": max_s},
            "min": {"type": min(mel_sens, key=mel_sens.get), "value": min_s},
            "gap": round(gap, 4),
            "gap_pct": round(gap * 100, 2),
            "severity": _severity(gap),
        }

    # Cancer sensitivity gap
    cancer_sens = {
        st: r["cancer_sensitivity"]
        for st, r in results_by_type.items()
        if r["cancer_sensitivity"] is not None and r.get("cancer_support", 0) >= 5
    }
    if len(cancer_sens) >= 2:
        max_s = max(cancer_sens.values())
        min_s = min(cancer_sens.values())
        gap = max_s - min_s
        gaps["cancer_sensitivity"] = {
            "max": {"type": max(cancer_sens, key=cancer_sens.get), "value": max_s},
            "min": {"type": min(cancer_sens, key=cancer_sens.get), "value": min_s},
            "gap": round(gap, 4),
            "gap_pct": round(gap * 100, 2),
            "severity": _severity(gap),
        }

    # Melanoma specificity gap
    mel_spec = {
        st: r["melanoma_specificity"]
        for st, r in results_by_type.items()
        if r["melanoma_specificity"] is not None and r["n_images"] >= 10
    }
    if len(mel_spec) >= 2:
        max_s = max(mel_spec.values())
        min_s = min(mel_spec.values())
        gap = max_s - min_s
        gaps["melanoma_specificity"] = {
            "max": {"type": max(mel_spec, key=mel_spec.get), "value": max_s},
            "min": {"type": min(mel_spec, key=mel_spec.get), "value": min_s},
            "gap": round(gap, 4),
            "gap_pct": round(gap * 100, 2),
            "severity": _severity(gap),
        }

    return gaps


def _severity(gap: float) -> str:
    if gap > 0.10:
        return "DANGEROUS"
    elif gap > 0.05:
        return "CRITICAL"
    elif gap > 0.03:
        return "WARNING"
    else:
        return "ACCEPTABLE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 72)
    log("ADR-124: Fitzpatrick Skin Tone Equity Validation — DrAgnes v2")
    log("=" * 72)
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log(f"Model:     {MODEL_DIR}")
    log(f"Manifest:  {MANIFEST_PATH}")
    log(f"Strategy:  up to {MAX_PER_CLASS_PER_TYPE} images per class per skin type")
    log("")

    # ------------------------------------------------------------------
    # 1. Load class mapping from manifest
    # ------------------------------------------------------------------
    log("Loading class mapping from manifest...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    class_mapping = manifest["class_mapping"]
    # Normalize keys to lowercase for matching
    class_mapping = {k.lower(): v for k, v in class_mapping.items()}
    log(f"  {len(class_mapping)} raw labels mapped to {len(set(class_mapping.values()))} classes")
    log("")

    # ------------------------------------------------------------------
    # 2. Download and parse Fitzpatrick17k CSV
    # ------------------------------------------------------------------
    log(f"Downloading Fitzpatrick17k CSV from GitHub...")
    try:
        resp = urllib.request.urlopen(CSV_URL, timeout=30)
        csv_text = resp.read().decode("utf-8")
    except Exception as e:
        log(f"  FATAL: Failed to download CSV: {e}")
        sys.exit(1)

    reader = csv.DictReader(io.StringIO(csv_text))
    all_rows = list(reader)
    log(f"  Total rows in CSV: {len(all_rows)}")

    # ------------------------------------------------------------------
    # 3. Filter to our 7 classes and valid skin types
    # ------------------------------------------------------------------
    log("Filtering to DrAgnes 7 classes with valid Fitzpatrick types...")
    eligible: list[dict[str, Any]] = []
    unmapped_count = 0
    invalid_skin_count = 0

    for row in all_rows:
        label = row.get("label", "").strip().lower()
        dragnes_class = class_mapping.get(label)
        if dragnes_class is None:
            unmapped_count += 1
            continue

        # Parse Fitzpatrick scale
        try:
            ft_raw = row.get("fitzpatrick_scale") or row.get("fitzpatrick") or ""
            ft = int(float(ft_raw))
        except (ValueError, TypeError):
            ft = -1

        if ft not in VALID_SKIN_TYPES:
            invalid_skin_count += 1
            continue

        url = row.get("url", "").strip()
        if not url:
            continue

        md5 = row.get("md5hash", "").strip()
        eligible.append({
            "label": label,
            "dragnes_class": dragnes_class,
            "fitzpatrick": ft,
            "url": url,
            "md5": md5,
        })

    log(f"  Eligible (mapped + valid skin type + has URL): {len(eligible)}")
    log(f"  Unmapped labels: {unmapped_count}")
    log(f"  Invalid/unknown skin type: {invalid_skin_count}")

    # Distribution before sampling
    dist = defaultdict(lambda: defaultdict(int))
    for row in eligible:
        dist[row["dragnes_class"]][row["fitzpatrick"]] += 1

    log("\n  Available distribution (class x skin type):")
    header = f"  {'Class':<8}" + "".join(f"{'T'+SKIN_TYPE_LABELS[s]:>6}" for s in VALID_SKIN_TYPES) + f"{'Total':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for cls in CLASS_NAMES:
        counts = [dist[cls][s] for s in VALID_SKIN_TYPES]
        total = sum(counts)
        line = f"  {cls:<8}" + "".join(f"{c:>6}" for c in counts) + f"{total:>8}"
        log(line)
    log("")

    # ------------------------------------------------------------------
    # 4. Stratified sampling: up to 50 per class per skin type
    # ------------------------------------------------------------------
    log(f"Stratified sampling (max {MAX_PER_CLASS_PER_TYPE} per class per type)...")
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in eligible:
        grouped[(row["dragnes_class"], row["fitzpatrick"])].append(row)

    sample: list[dict[str, Any]] = []
    for (cls, st), rows in grouped.items():
        take = min(len(rows), MAX_PER_CLASS_PER_TYPE)
        # Deterministic sampling for reproducibility
        import random
        rng = random.Random(42)
        sampled = rng.sample(rows, take) if take < len(rows) else rows
        sample.extend(sampled)

    log(f"  Total sampled: {len(sample)}")

    # Sampled distribution
    sample_dist = defaultdict(lambda: defaultdict(int))
    for row in sample:
        sample_dist[row["dragnes_class"]][row["fitzpatrick"]] += 1

    log("\n  Sampled distribution:")
    header = f"  {'Class':<8}" + "".join(f"{'T'+SKIN_TYPE_LABELS[s]:>6}" for s in VALID_SKIN_TYPES) + f"{'Total':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for cls in CLASS_NAMES:
        counts = [sample_dist[cls][s] for s in VALID_SKIN_TYPES]
        total = sum(counts)
        line = f"  {cls:<8}" + "".join(f"{c:>6}" for c in counts) + f"{total:>8}"
        log(line)
    log("")

    # ------------------------------------------------------------------
    # 5. Download images (parallel, with caching)
    # ------------------------------------------------------------------
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {len(sample)} images (cache: {IMAGE_CACHE_DIR.name})...")
    log(f"  Workers: {DOWNLOAD_WORKERS}, timeout: {DOWNLOAD_TIMEOUT}s per image")
    t_dl_start = time.time()

    download_results: list[dict[str, Any] | None] = [None] * len(sample)
    download_failures = 0
    download_cached = 0

    def _download_one(idx: int) -> tuple[int, Image.Image | None]:
        row = sample[idx]
        md5 = row["md5"] or f"img_{idx}"
        cache_path = IMAGE_CACHE_DIR / f"{md5}.jpg"
        was_cached = cache_path.exists()
        img = download_image(row["url"], cache_path)
        return idx, img, was_cached

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download_one, i): i for i in range(len(sample))}
        completed = 0
        for future in as_completed(futures):
            idx, img, was_cached = future.result()
            completed += 1
            if img is not None:
                download_results[idx] = {
                    "image": img,
                    "dragnes_class": sample[idx]["dragnes_class"],
                    "fitzpatrick": sample[idx]["fitzpatrick"],
                    "url": sample[idx]["url"],
                }
                if was_cached:
                    download_cached += 1
            else:
                download_failures += 1

            if completed % PROGRESS_EVERY == 0:
                elapsed = time.time() - t_dl_start
                log(f"  Downloaded {completed}/{len(sample)} "
                    f"({download_failures} failures, {elapsed:.1f}s)")

    t_dl_end = time.time()
    valid_images = [r for r in download_results if r is not None]
    log(f"\n  Download complete in {t_dl_end - t_dl_start:.1f}s")
    log(f"  Successfully loaded: {len(valid_images)}")
    log(f"  From cache: {download_cached}")
    log(f"  Failed: {download_failures}")
    failure_rate = download_failures / len(sample) if sample else 0
    log(f"  Failure rate: {failure_rate:.1%}")

    stale_urls_warning = None
    if failure_rate > 0.50:
        stale_urls_warning = (
            f"WARNING: {failure_rate:.0%} of image URLs failed to download. "
            "The Fitzpatrick17k URLs are likely stale (images hosted on dermatology "
            "atlas sites that rotate URLs). Recommended alternatives: "
            "(1) Use the DDI (Diverse Dermatology Images) dataset from Stanford; "
            "(2) Use ISIC Archive with self-reported Fitzpatrick metadata; "
            "(3) Contact dataset authors for updated URLs; "
            "(4) Use a local Fitzpatrick17k image archive if available."
        )
        log(f"\n  *** {stale_urls_warning}")

    if len(valid_images) < 20:
        log("\n  FATAL: Too few images downloaded for meaningful validation.")
        # Still save a partial result
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "FAILED",
            "reason": f"Only {len(valid_images)} images downloaded (need >= 20)",
            "download_failure_rate": round(failure_rate, 4),
            "stale_urls_warning": stale_urls_warning,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  Partial results saved to {RESULTS_PATH}")
        sys.exit(1)

    log("")

    # ------------------------------------------------------------------
    # 6. Load model
    # ------------------------------------------------------------------
    log("Loading v2 model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"  Device: {device}")

    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR)).to(device)
    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: ViT-Base, 7 classes, {n_params:,} parameters")
    log("")

    # ------------------------------------------------------------------
    # 7. Run inference in batches
    # ------------------------------------------------------------------
    log(f"Running inference on {len(valid_images)} images (batch_size={BATCH_SIZE})...")
    t_infer_start = time.time()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_skin: list[int] = []
    y_probs: list[list[float]] = []
    inference_errors = 0

    for batch_start in range(0, len(valid_images), BATCH_SIZE):
        batch = valid_images[batch_start : batch_start + BATCH_SIZE]
        batch_images = []
        batch_labels = []
        batch_skins = []

        for item in batch:
            try:
                img = item["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                cls_idx = CLASS_NAMES.index(item["dragnes_class"])
                batch_images.append(img)
                batch_labels.append(cls_idx)
                batch_skins.append(item["fitzpatrick"])
            except Exception:
                inference_errors += 1
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
            y_skin.extend(batch_skins)
            y_probs.extend(probs_np.tolist())

        except Exception as e:
            # Fall back to single-image inference
            for img, lbl, skin in zip(batch_images, batch_labels, batch_skins):
                try:
                    inputs = processor(images=[img], return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        pred = probs.argmax(dim=-1).cpu().item()
                        probs_np = probs.cpu().numpy()[0]
                    y_true.append(lbl)
                    y_pred.append(pred)
                    y_skin.append(skin)
                    y_probs.append(probs_np.tolist())
                except Exception:
                    inference_errors += 1

        processed = len(y_true)
        if processed % PROGRESS_EVERY < BATCH_SIZE:
            elapsed = time.time() - t_infer_start
            rate = processed / elapsed if elapsed > 0 else 0
            log(f"  [{processed:>5}/{len(valid_images)}] {elapsed:.1f}s, "
                f"{rate:.1f} img/s, {inference_errors} errors")

    t_infer_end = time.time()
    inference_time = t_infer_end - t_infer_start
    rate = len(y_true) / inference_time if inference_time > 0 else 0

    log(f"\n  Inference complete:")
    log(f"    Processed: {len(y_true)}")
    log(f"    Errors: {inference_errors}")
    log(f"    Time: {inference_time:.1f}s ({rate:.1f} img/s)")
    log("")

    if len(y_true) == 0:
        log("FATAL: No images successfully processed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 8. Overall metrics
    # ------------------------------------------------------------------
    log("=" * 72)
    log("OVERALL RESULTS")
    log("=" * 72)

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    correct = int((y_true_np == y_pred_np).sum())
    accuracy = correct / len(y_true_np)
    log(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{len(y_true_np)})")

    # Per-class metrics (global)
    mel_idx = CLASS_NAMES.index("mel")
    mel_tp = int(((y_true_np == mel_idx) & (y_pred_np == mel_idx)).sum())
    mel_fn = int(((y_true_np == mel_idx) & (y_pred_np != mel_idx)).sum())
    mel_fp = int(((y_true_np != mel_idx) & (y_pred_np == mel_idx)).sum())
    mel_tn = int(((y_true_np != mel_idx) & (y_pred_np != mel_idx)).sum())
    mel_support = mel_tp + mel_fn
    mel_sensitivity = mel_tp / mel_support if mel_support > 0 else 0.0
    mel_specificity = mel_tn / (mel_tn + mel_fp) if (mel_tn + mel_fp) > 0 else 0.0

    log(f"Melanoma sensitivity: {mel_sensitivity:.4f} ({mel_tp}/{mel_support})")
    log(f"Melanoma specificity: {mel_specificity:.4f}")

    # ------------------------------------------------------------------
    # 9. Per-Fitzpatrick-type metrics
    # ------------------------------------------------------------------
    log(f"\n{'=' * 72}")
    log("PER-FITZPATRICK-TYPE PERFORMANCE")
    log(f"{'=' * 72}")

    results_by_type = compute_per_type_metrics(y_true, y_pred, y_skin)

    log(f"\n{'Type':<6} {'N':>5} {'Accuracy':>10} {'Mel Sens':>10} {'Mel Spec':>10} "
        f"{'Cancer Sens':>12} {'Mel N':>6}")
    log("-" * 65)
    for st in VALID_SKIN_TYPES:
        label = SKIN_TYPE_LABELS[st]
        r = results_by_type[label]
        n = r["n_images"]
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A"
        ms = f"{r['melanoma_sensitivity']:.4f}" if r["melanoma_sensitivity"] is not None else "N/A"
        msp = f"{r['melanoma_specificity']:.4f}" if r["melanoma_specificity"] is not None else "N/A"
        cs = f"{r['cancer_sensitivity']:.4f}" if r["cancer_sensitivity"] is not None else "N/A"
        mn = r.get("melanoma_support", 0)
        log(f"{label:<6} {n:>5} {acc:>10} {ms:>10} {msp:>10} {cs:>12} {mn:>6}")

    # ------------------------------------------------------------------
    # 10. Equity gap analysis
    # ------------------------------------------------------------------
    log(f"\n{'=' * 72}")
    log("EQUITY GAP ANALYSIS")
    log(f"{'=' * 72}")
    log("Gap = max difference between any two skin types for a metric")
    log("ACCEPTABLE < 3%  |  WARNING 3-5%  |  CRITICAL 5-10%  |  DANGEROUS > 10%\n")

    gaps = compute_equity_gaps(results_by_type)

    any_critical = False
    any_dangerous = False

    for metric_name, gap_info in gaps.items():
        sev = gap_info["severity"]
        flag = ""
        if sev == "CRITICAL":
            flag = " *** CRITICAL ***"
            any_critical = True
        elif sev == "DANGEROUS":
            flag = " *** DANGEROUS ***"
            any_dangerous = True
        elif sev == "WARNING":
            flag = " (warning)"

        log(f"  {metric_name}:")
        log(f"    Best:  Type {gap_info['max']['type']} = {gap_info['max']['value']:.4f}")
        log(f"    Worst: Type {gap_info['min']['type']} = {gap_info['min']['value']:.4f}")
        log(f"    Gap:   {gap_info['gap_pct']:.2f}%  [{sev}]{flag}")
        log("")

    # Per-class accuracy gap across skin types
    log("  Per-class sensitivity gaps across skin types:")
    per_class_gaps = {}
    for cls in CLASS_NAMES:
        cls_sens: dict[str, float] = {}
        for st_label, r in results_by_type.items():
            pc = r.get("per_class", {}).get(cls, {})
            s = pc.get("sensitivity")
            sup = pc.get("support", 0)
            if s is not None and sup >= 3:
                cls_sens[st_label] = s
        if len(cls_sens) >= 2:
            max_s = max(cls_sens.values())
            min_s = min(cls_sens.values())
            gap = max_s - min_s
            sev = _severity(gap)
            per_class_gaps[cls] = {
                "max_type": max(cls_sens, key=cls_sens.get),
                "max_value": max_s,
                "min_type": min(cls_sens, key=cls_sens.get),
                "min_value": min_s,
                "gap": round(gap, 4),
                "gap_pct": round(gap * 100, 2),
                "severity": sev,
            }
            flag = ""
            if sev in ("CRITICAL", "DANGEROUS"):
                flag = f" *** {sev} ***"
                if sev == "CRITICAL":
                    any_critical = True
                elif sev == "DANGEROUS":
                    any_dangerous = True
            log(f"    {cls:<8}: {gap*100:.2f}% gap  "
                f"(best: Type {max(cls_sens, key=cls_sens.get)}={max_s:.4f}, "
                f"worst: Type {min(cls_sens, key=cls_sens.get)}={min_s:.4f})  "
                f"[{sev}]{flag}")

    # ------------------------------------------------------------------
    # 11. Summary verdict
    # ------------------------------------------------------------------
    log(f"\n{'=' * 72}")
    log("VERDICT")
    log(f"{'=' * 72}")

    if any_dangerous:
        verdict = "DANGEROUS"
        log("RESULT: DANGEROUS equity gaps detected (>10% difference between skin types)")
        log("ACTION: Model requires retraining with balanced skin-type representation")
    elif any_critical:
        verdict = "CRITICAL"
        log("RESULT: CRITICAL equity gaps detected (5-10% difference between skin types)")
        log("ACTION: Model should be retrained or augmented for underrepresented skin types")
    else:
        verdict = "ACCEPTABLE"
        log("RESULT: Equity gaps within acceptable range (<5%)")
        log("ACTION: Monitor with ongoing validation; continue targeted data collection")

    # ------------------------------------------------------------------
    # 12. Save results
    # ------------------------------------------------------------------
    results = {
        "adr": "ADR-124",
        "title": "Fitzpatrick Skin Tone Equity Validation — DrAgnes v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_DIR),
        "model_type": "ViT-Base (google/vit-base-patch16-224)",
        "device": device,
        "dataset": {
            "name": "Fitzpatrick17k",
            "csv_url": CSV_URL,
            "total_csv_rows": len(all_rows),
            "eligible_with_skin_type": len(eligible),
            "sampled": len(sample),
            "downloaded_successfully": len(valid_images),
            "download_failures": download_failures,
            "download_failure_rate": round(failure_rate, 4),
            "download_cached": download_cached,
            "inference_errors": inference_errors,
            "total_evaluated": len(y_true),
            "sampling_strategy": f"up to {MAX_PER_CLASS_PER_TYPE} per class per skin type",
        },
        "stale_urls_warning": stale_urls_warning,
        "overall": {
            "accuracy": round(accuracy, 4),
            "melanoma_sensitivity": round(mel_sensitivity, 4),
            "melanoma_specificity": round(mel_specificity, 4),
            "melanoma_support": mel_support,
            "total_images": len(y_true),
        },
        "per_fitzpatrick_type": results_by_type,
        "equity_gaps": gaps,
        "per_class_equity_gaps": per_class_gaps,
        "verdict": verdict,
        "verdict_criteria": {
            "ACCEPTABLE": "all gaps < 5%",
            "CRITICAL": "any gap 5-10%",
            "DANGEROUS": "any gap > 10%",
        },
        "inference": {
            "batch_size": BATCH_SIZE,
            "time_seconds": round(inference_time, 2),
            "images_per_second": round(rate, 1),
        },
        "sample_distribution": {
            cls: {SKIN_TYPE_LABELS[s]: sample_dist[cls][s] for s in VALID_SKIN_TYPES}
            for cls in CLASS_NAMES
        },
        "recommendations": [],
    }

    # Build recommendations based on findings
    recs = results["recommendations"]
    if stale_urls_warning:
        recs.append(stale_urls_warning)
    if any_dangerous:
        recs.append(
            "DANGEROUS gaps detected. Prioritize: (1) collect more training images "
            "for underperforming skin types, (2) apply skin-type-aware data augmentation, "
            "(3) consider Fitzpatrick-stratified training with class weights."
        )
    if any_critical:
        recs.append(
            "CRITICAL gaps detected. Consider: (1) synthetic augmentation for dark skin "
            "types (StyleGAN-based or diffusion-based), (2) transfer learning from "
            "DDI dataset, (3) ensemble with a specialist model for dark skin."
        )
    # Always recommend ongoing monitoring
    recs.append(
        "Continuously validate on diverse datasets. Consider adding DDI "
        "(Diverse Dermatology Images) and PAD-UFES-20 to the validation pipeline."
    )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_PATH}")
    log(f"{'=' * 72}")


if __name__ == "__main__":
    main()
