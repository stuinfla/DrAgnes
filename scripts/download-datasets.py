#!/usr/bin/env python3
"""
DrAgnes Dataset Downloader & Verifier
======================================
Downloads and verifies access to dermatology training datasets:
  1. BCN20000 / ISIC 2019  -- dermoscopy images with histopathology labels
  2. Fitzpatrick17k         -- clinical images with Fitzpatrick skin type labels
  3. PAD-UFES-20            -- smartphone clinical images

For each dataset we:
  - Stream metadata (no bulk image download to disk)
  - Report total images, class distribution, available metadata columns
  - Save a manifest JSON to scripts/dataset-manifests/
  - Map source labels to the DrAgnes 7-class system where possible

Usage:
    python3 scripts/download-datasets.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# DrAgnes 7-class taxonomy
# ---------------------------------------------------------------------------
DRAGNES_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ---------------------------------------------------------------------------
# ISIC / HAM10000 label mapping (dx field or class label -> DrAgnes class)
# Covers abbreviations, full names, and underscore variants from various
# HuggingFace dataset uploads.
# ---------------------------------------------------------------------------
ISIC_MAP: dict[str, str] = {
    # Melanoma
    "mel": "mel",
    "melanoma": "mel",
    # Melanocytic nevi
    "nv": "nv",
    "nevus": "nv",
    "melanocytic nevi": "nv",
    "melanocytic_nevi": "nv",
    # BCC
    "bcc": "bcc",
    "basal cell carcinoma": "bcc",
    "basal_cell_carcinoma": "bcc",
    # Actinic keratoses / intraepithelial carcinoma (akiec)
    "akiec": "akiec",
    "actinic keratosis": "akiec",
    "actinic keratoses": "akiec",
    "actinic_keratoses": "akiec",
    "ak": "akiec",
    "scc": "akiec",
    "squamous cell carcinoma": "akiec",
    "squamous_cell_carcinoma": "akiec",
    "intraepithelial carcinoma": "akiec",
    # Benign keratosis-like lesions (bkl)
    "bkl": "bkl",
    "benign keratosis": "bkl",
    "benign keratosis-like lesions": "bkl",
    "benign_keratosis-like_lesions": "bkl",
    "benign keratosis-like": "bkl",
    "seborrheic keratosis": "bkl",
    "solar lentigo": "bkl",
    "lentigo": "bkl",
    # Dermatofibroma
    "df": "df",
    "dermatofibroma": "df",
    # Vascular lesions
    "vasc": "vasc",
    "vascular": "vasc",
    "vascular lesion": "vasc",
    "vascular lesions": "vasc",
    "vascular_lesions": "vasc",
}

# ---------------------------------------------------------------------------
# Fitzpatrick17k: mapping of 114 disease categories to DrAgnes 7-class
# Only conditions that clearly map are included; rest tagged "other".
# ---------------------------------------------------------------------------
FITZ17K_MAP: dict[str, str] = {
    # Melanoma
    "malignant melanoma": "mel",
    "lentigo maligna": "mel",
    "melanoma": "mel",
    "acral lentiginous melanoma": "mel",
    "melanoma in situ": "mel",
    "superficial spreading melanoma ssm": "mel",
    # Nevi
    "melanocytic nevi": "nv",
    "dysplastic nevus": "nv",
    "blue nevus": "nv",
    "spitz nevus": "nv",
    "nevus": "nv",
    "atypical melanocytic proliferation": "nv",
    "halo nevus": "nv",
    "congenital nevus": "nv",
    "recurrent nevus": "nv",
    "combined nevus": "nv",
    "melanocytic nevus": "nv",
    "nevocytic nevus": "nv",
    "becker nevus": "nv",
    "epidermal nevus": "nv",
    # BCC
    "basal cell carcinoma": "bcc",
    "superficial basal cell carcinoma": "bcc",
    "nodular basal cell carcinoma": "bcc",
    "morpheaform basal cell carcinoma": "bcc",
    "pigmented basal cell carcinoma": "bcc",
    "solid cystic basal cell carcinoma": "bcc",
    "basal cell carcinoma morpheiform": "bcc",
    # Actinic keratosis / SCC -> akiec
    "actinic keratosis": "akiec",
    "squamous cell carcinoma": "akiec",
    "squamous cell carcinoma in situ": "akiec",
    "bowen's disease": "akiec",
    "bowens disease": "akiec",
    "keratoacanthoma": "akiec",
    "actinic cheilitis": "akiec",
    "porokeratosis actinic": "akiec",
    "disseminated actinic porokeratosis": "akiec",
    # BKL
    "seborrheic keratosis": "bkl",
    "solar lentigo": "bkl",
    "lichenoid keratosis": "bkl",
    "lentigo simplex": "bkl",
    "large cell acanthoma": "bkl",
    "keratosis pilaris": "bkl",
    # Dermatofibroma
    "dermatofibroma": "df",
    # Vascular
    "hemangioma": "vasc",
    "cherry angioma": "vasc",
    "pyogenic granuloma": "vasc",
    "granuloma pyogenic": "vasc",
    "angiokeratoma": "vasc",
    "angioma": "vasc",
    "spider angioma": "vasc",
    "port wine stain": "vasc",
    "kaposi sarcoma": "vasc",
    "lymphangioma": "vasc",
    "telangiectases": "vasc",
}

# ---------------------------------------------------------------------------
# PAD-UFES-20 label mapping
# ---------------------------------------------------------------------------
PAD_UFES_MAP: dict[str, str] = {
    "mel": "mel",
    "bcc": "bcc",
    "scc": "akiec",
    "ack": "akiec",
    "actinic keratosis": "akiec",
    "sek": "bkl",
    "seborrheic keratosis": "bkl",
    "nev": "nv",
    "nevus": "nv",
}

MANIFEST_DIR = Path(__file__).resolve().parent / "dataset-manifests"


def ensure_manifest_dir() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def save_manifest(name: str, data: dict[str, Any]) -> Path:
    ensure_manifest_dir()
    path = MANIFEST_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [manifest] saved -> {path}")
    return path


def pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def print_distribution(dist: dict[str, int], title: str = "Class Distribution") -> None:
    total = sum(dist.values())
    print(f"\n  {title} (total: {total:,})")
    print(f"  {'Class':<40} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8}")
    for cls in sorted(dist.keys(), key=lambda k: dist[k], reverse=True):
        print(f"  {cls:<40} {dist[cls]:>8,} {pct(dist[cls], total):>8}")


def try_load_hf_dataset(ds_id: str, config: str | None = None, split: str = "train"):
    """Attempt to load a HuggingFace dataset with streaming, handling version compat."""
    from datasets import load_dataset, get_dataset_config_names

    # Check configs
    try:
        configs = get_dataset_config_names(ds_id)
        print(f"    Available configs: {configs}")
        if config is None and configs:
            config = configs[0]
    except Exception:
        pass

    # Try without trust_remote_code first (newer datasets lib rejects it)
    try:
        ds = load_dataset(ds_id, config, split=split, streaming=True)
        return ds, config
    except Exception:
        pass

    # Fallback: try with trust_remote_code for older datasets
    try:
        ds = load_dataset(ds_id, config, split=split, streaming=True, trust_remote_code=True)
        return ds, config
    except Exception as e:
        raise e


def scan_streaming_dataset(
    ds,
    label_col_candidates: list[str],
    label_map: dict[str, str],
    fitz_col_candidates: list[str] | None = None,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Stream through a HF dataset and collect statistics."""
    raw_counts: Counter = Counter()
    mapped_counts: Counter = Counter()
    unmapped_labels: Counter = Counter()
    fitzpatrick_dist: Counter = Counter()
    total = 0
    has_image = False
    metadata_cols: set[str] = set()
    label_col = None
    fitz_col = None
    t0 = time.time()

    for row in ds:
        total += 1
        metadata_cols.update(row.keys())

        if "image" in row and row["image"] is not None:
            has_image = True

        # Auto-detect label column on first row
        if label_col is None:
            for candidate in label_col_candidates:
                if candidate in row:
                    label_col = candidate
                    break

        if fitz_col is None and fitz_col_candidates:
            for candidate in fitz_col_candidates:
                if candidate in row:
                    fitz_col = candidate
                    break

        # Count labels
        if label_col and label_col in row:
            raw_label = row[label_col]
            if isinstance(raw_label, int):
                raw_label_str = str(raw_label)
            else:
                raw_label_str = str(raw_label).strip().lower()
            raw_counts[raw_label_str] += 1

            mapped = label_map.get(raw_label_str)
            if mapped:
                mapped_counts[mapped] += 1
            else:
                unmapped_labels[raw_label_str] += 1

        # Count Fitzpatrick skin types
        if fitz_col and fitz_col in row:
            fst = str(row[fitz_col]).strip()
            if fst and fst.lower() not in ("none", "nan", ""):
                fitzpatrick_dist[fst] += 1

        if total % progress_every == 0:
            elapsed = time.time() - t0
            print(f"    ... {total:,} rows scanned ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    return {
        "total": total,
        "has_image": has_image,
        "metadata_cols": sorted(metadata_cols),
        "label_col": label_col,
        "fitz_col": fitz_col,
        "raw_counts": dict(raw_counts.most_common()),
        "mapped_counts": dict(mapped_counts.most_common()),
        "unmapped_labels": dict(unmapped_labels.most_common()),
        "fitzpatrick_dist": dict(fitzpatrick_dist.most_common()),
        "elapsed_sec": round(elapsed, 1),
    }


# ============================================================================
# 1. ISIC 2019 / BCN20000 / HAM10000 via HuggingFace
# ============================================================================
def download_isic_datasets() -> None:
    print("\n" + "=" * 72)
    print("DATASET 1: ISIC Dermoscopy (HAM10000 + ISIC 2019 search)")
    print("=" * 72)

    from datasets import load_dataset

    # We will try multiple known HuggingFace dataset IDs
    candidates = [
        ("marmal88/skin_cancer", "HAM10000 (skin_cancer)"),
        ("NicolasRR/isic-2019", "ISIC 2019"),
        ("imranraad/ISIC2019", "ISIC 2019 (imranraad)"),
    ]

    found_datasets: list[dict[str, Any]] = []

    for ds_id, description in candidates:
        print(f"\n  Trying: {ds_id} ({description}) ...")
        try:
            ds, config = try_load_hf_dataset(ds_id)
            # Peek
            sample = next(iter(ds))
            columns = list(sample.keys())
            print(f"    SUCCESS -- Columns: {columns}")
            print(f"    Types: { {k: type(v).__name__ for k, v in sample.items()} }")

            # Full scan
            print(f"    Streaming full dataset for label counts ...")
            ds2, _ = try_load_hf_dataset(ds_id, config)
            stats = scan_streaming_dataset(
                ds2,
                label_col_candidates=["dx", "label", "diagnosis", "class", "target"],
                label_map=ISIC_MAP,
                progress_every=5000,
            )

            print(f"\n    Total images: {stats['total']:,}")
            print(f"    Has image data: {stats['has_image']}")
            print(f"    Label column: {stats['label_col']}")
            print_distribution(stats["raw_counts"], f"Raw Labels ({ds_id})")
            print_distribution(stats["mapped_counts"], f"Mapped to DrAgnes 7-Class ({ds_id})")

            if stats["unmapped_labels"]:
                total_unmapped = sum(stats["unmapped_labels"].values())
                print(f"\n    Unmapped: {total_unmapped:,} images")
                for lbl, cnt in sorted(stats["unmapped_labels"].items(), key=lambda x: -x[1])[:10]:
                    print(f"      {lbl}: {cnt:,}")

            # Check for ClassLabel names to resolve integer labels
            label_names = resolve_classlabel_names(ds_id, config, stats["label_col"])
            if label_names and all(k.isdigit() for k in stats["raw_counts"].keys()):
                print(f"\n    ClassLabel names: {label_names}")
                remapped = remap_integer_labels(stats["raw_counts"], label_names, ISIC_MAP)
                stats["mapped_counts"] = remapped["mapped"]
                stats["unmapped_labels"] = remapped["unmapped"]
                stats["label_names"] = label_names
                print_distribution(remapped["mapped"], f"Re-Mapped via ClassLabel ({ds_id})")

            found_datasets.append({
                "dataset_id": ds_id,
                "description": description,
                "config": config,
                "stats": stats,
            })

        except Exception as e:
            print(f"    FAILED: {e}")

    # Also query ISIC Archive API for supplementary info
    isic_api_info = query_isic_archive_api()

    # Build manifest
    if found_datasets:
        primary = found_datasets[0]
        manifest = {
            "dataset": "ISIC Dermoscopy",
            "description": "HAM10000 and/or ISIC 2019 Challenge dermoscopy images",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accessible_datasets": [],
            "isic_archive_api": isic_api_info,
            "class_mapping": ISIC_MAP,
        }
        for ds_info in found_datasets:
            manifest["accessible_datasets"].append({
                "source": f"huggingface:{ds_info['dataset_id']}",
                "description": ds_info["description"],
                "config": ds_info["config"],
                "total_images": ds_info["stats"]["total"],
                "has_image_data": ds_info["stats"]["has_image"],
                "metadata_columns": ds_info["stats"]["metadata_cols"],
                "label_column": ds_info["stats"]["label_col"],
                "label_names": ds_info["stats"].get("label_names"),
                "raw_class_distribution": ds_info["stats"]["raw_counts"],
                "dragnes_7class_distribution": ds_info["stats"]["mapped_counts"],
                "unmapped_labels": ds_info["stats"]["unmapped_labels"],
            })
        save_manifest("isic-dermoscopy", manifest)
    else:
        print("\n  No ISIC datasets found on HuggingFace.")
        manifest = {
            "dataset": "ISIC Dermoscopy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "not_found_on_huggingface",
            "isic_archive_api": isic_api_info,
            "note": "Try manual download from https://challenge.isic-archive.com/",
        }
        save_manifest("isic-dermoscopy", manifest)


def resolve_classlabel_names(ds_id: str, config: str | None, label_col: str | None) -> list[str] | None:
    """Try to extract ClassLabel feature names from a streaming dataset."""
    if not label_col:
        return None
    try:
        from datasets import load_dataset
        ds = load_dataset(ds_id, config, split="train", streaming=True)
        if hasattr(ds, "features") and label_col in ds.features:
            feat = ds.features[label_col]
            if hasattr(feat, "names"):
                return feat.names
    except Exception:
        pass
    return None


def remap_integer_labels(
    raw_counts: dict[str, int],
    label_names: list[str],
    label_map: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Re-map integer label indices using ClassLabel names."""
    mapped: Counter = Counter()
    unmapped: Counter = Counter()
    for idx_str, cnt in raw_counts.items():
        idx = int(idx_str)
        if idx < len(label_names):
            name = label_names[idx].strip().lower()
            target = label_map.get(name)
            if target:
                mapped[target] += cnt
            else:
                unmapped[name] += cnt
    return {"mapped": dict(mapped.most_common()), "unmapped": dict(unmapped.most_common())}


def query_isic_archive_api() -> dict[str, Any] | None:
    """Query the ISIC Archive REST API for dataset availability."""
    import requests

    print("\n  Querying ISIC Archive API for supplementary info ...")
    info: dict[str, Any] = {"accessible": False}
    try:
        resp = requests.get(
            "https://api.isic-archive.com/api/v2/images",
            params={"limit": 1, "offset": 0},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            info["accessible"] = True
            if isinstance(data, dict) and "count" in data:
                info["total_archive_images"] = data["count"]
                print(f"    ISIC Archive total images: {data['count']:,}")
        else:
            print(f"    ISIC API returned status {resp.status_code}")
    except Exception as e:
        print(f"    ISIC API query failed: {e}")

    return info


# ============================================================================
# 2. Fitzpatrick17k
# ============================================================================
def download_fitzpatrick17k() -> None:
    print("\n" + "=" * 72)
    print("DATASET 2: Fitzpatrick17k")
    print("=" * 72)

    # Try HuggingFace first
    hf_candidates = [
        "mattgroh/fitzpatrick17k",
        "marmal88/fitzpatrick17k",
        "cmu-delphi/fitzpatrick17k",
    ]

    for ds_id in hf_candidates:
        print(f"\n  Trying HuggingFace: {ds_id} ...")
        try:
            ds, config = try_load_hf_dataset(ds_id)
            sample = next(iter(ds))
            print(f"    SUCCESS -- Columns: {list(sample.keys())}")
            # If we get here, stream it
            ds2, _ = try_load_hf_dataset(ds_id, config)
            stats = scan_streaming_dataset(
                ds2,
                label_col_candidates=["label", "dx", "diagnosis", "three_partition_label", "nine_partition_label", "condition"],
                label_map=FITZ17K_MAP,
                fitz_col_candidates=["fitzpatrick", "fitzpatrick_skin_type", "skin_type", "fitzpatrick_scale", "fst"],
                progress_every=5000,
            )
            print(f"\n    Total: {stats['total']:,}, Has images: {stats['has_image']}")
            print_distribution(stats["raw_counts"], "Raw Labels")
            print_distribution(stats["mapped_counts"], "Mapped to DrAgnes 7-Class")
            if stats["fitzpatrick_dist"]:
                print_distribution(stats["fitzpatrick_dist"], "Fitzpatrick Skin Type")

            manifest = {
                "dataset": "Fitzpatrick17k",
                "source": f"huggingface:{ds_id}",
                "config": config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **stats,
            }
            save_manifest("fitzpatrick17k", manifest)
            return
        except Exception as e:
            print(f"    FAILED: {e}")

    # Fallback: CSV from GitHub
    print("\n  HuggingFace datasets not available. Trying GitHub CSV ...")
    download_fitzpatrick_csv()


def download_fitzpatrick_csv() -> None:
    """Download Fitzpatrick17k CSV annotations from the paper's GitHub repo."""
    import requests
    import pandas as pd

    csv_url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
    print(f"  Downloading: {csv_url}")

    try:
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Download failed: {e}")
        save_manifest("fitzpatrick17k", {
            "dataset": "Fitzpatrick17k",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "download_failed",
            "error": str(e),
        })
        return

    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  CSV loaded: {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")

    # Label distribution
    label_col = "label" if "label" in df.columns else None
    raw_counts: dict[str, int] = {}
    mapped_counts: Counter = Counter()
    unmapped_labels: Counter = Counter()

    if label_col:
        raw_counts = df[label_col].value_counts().to_dict()
        for label, count in raw_counts.items():
            label_lower = str(label).strip().lower()
            mapped = FITZ17K_MAP.get(label_lower)
            if mapped:
                mapped_counts[mapped] += count
            else:
                unmapped_labels[label_lower] += count

    print_distribution(raw_counts, "Raw Label Distribution (all 114 categories)")
    print_distribution(dict(mapped_counts), "Mapped to DrAgnes 7-Class")

    total_unmapped = sum(unmapped_labels.values())
    total_mapped = sum(mapped_counts.values())
    print(f"\n  Mapped to 7-class: {total_mapped:,} images ({pct(total_mapped, len(df))})")
    print(f"  Not mappable (other conditions): {total_unmapped:,} images ({pct(total_unmapped, len(df))})")

    # Fitzpatrick distribution
    fitz_col = None
    fitzpatrick_dist: dict[str, int] = {}
    for col_name in ["fitzpatrick_scale", "fitzpatrick", "fitzpatrick_skin_type"]:
        if col_name in df.columns:
            fitz_col = col_name
            fitzpatrick_dist = df[col_name].dropna().astype(int).value_counts().sort_index().to_dict()
            fitzpatrick_dist = {f"Type {k}": v for k, v in fitzpatrick_dist.items()}
            break

    if fitzpatrick_dist:
        print_distribution(fitzpatrick_dist, "Fitzpatrick Skin Type Distribution")

    # Three-partition and nine-partition label distributions
    three_part_dist = {}
    nine_part_dist = {}
    if "three_partition_label" in df.columns:
        three_part_dist = df["three_partition_label"].value_counts().to_dict()
        print_distribution(three_part_dist, "Three-Partition Label Distribution")
    if "nine_partition_label" in df.columns:
        nine_part_dist = df["nine_partition_label"].value_counts().to_dict()
        print_distribution(nine_part_dist, "Nine-Partition Label Distribution")

    # Check image URL column
    has_urls = "url" in df.columns
    url_sample = None
    if has_urls:
        url_sample = df["url"].iloc[0] if len(df) > 0 else None
        print(f"\n  Image URL column present: {has_urls}")
        print(f"  Sample URL: {url_sample}")

    manifest = {
        "dataset": "Fitzpatrick17k",
        "source": "github-csv:mattgroh/fitzpatrick17k",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_images": len(df),
        "has_image_data": False,
        "has_image_urls": has_urls,
        "columns": list(df.columns),
        "label_column": label_col,
        "fitzpatrick_column": fitz_col,
        "raw_class_distribution": {str(k): int(v) for k, v in raw_counts.items()},
        "dragnes_7class_distribution": dict(mapped_counts.most_common()),
        "mapped_count": total_mapped,
        "unmapped_count": total_unmapped,
        "unmapped_categories": len(unmapped_labels),
        "unmapped_top20": {str(k): int(v) for k, v in list(unmapped_labels.most_common(20))},
        "fitzpatrick_distribution": {str(k): int(v) for k, v in fitzpatrick_dist.items()},
        "three_partition_distribution": {str(k): int(v) for k, v in three_part_dist.items()},
        "nine_partition_distribution": {str(k): int(v) for k, v in nine_part_dist.items()},
        "class_mapping": FITZ17K_MAP,
        "note": "CSV annotations with image URLs. Images must be downloaded separately from source URLs.",
    }
    save_manifest("fitzpatrick17k", manifest)


# ============================================================================
# 3. PAD-UFES-20
# ============================================================================
def download_pad_ufes() -> None:
    print("\n" + "=" * 72)
    print("DATASET 3: PAD-UFES-20")
    print("=" * 72)

    hf_candidates = [
        "mahdavi/PAD-UFES-20",
        "marmal88/PAD-UFES-20",
        "pad-ufes-20",
    ]

    for ds_id in hf_candidates:
        print(f"\n  Trying HuggingFace: {ds_id} ...")
        try:
            ds, config = try_load_hf_dataset(ds_id)
            sample = next(iter(ds))
            print(f"    SUCCESS -- Columns: {list(sample.keys())}")

            ds2, _ = try_load_hf_dataset(ds_id, config)
            stats = scan_streaming_dataset(
                ds2,
                label_col_candidates=["label", "dx", "diagnosis", "diagnostic", "class", "target"],
                label_map=PAD_UFES_MAP,
                progress_every=1000,
            )
            print(f"\n    Total: {stats['total']:,}, Has images: {stats['has_image']}")
            print_distribution(stats["raw_counts"], "Raw Labels")
            print_distribution(stats["mapped_counts"], "Mapped to DrAgnes 7-Class")

            # Resolve integer labels
            label_names = resolve_classlabel_names(ds_id, config, stats["label_col"])
            if label_names and all(k.isdigit() for k in stats["raw_counts"].keys()):
                print(f"\n    ClassLabel names: {label_names}")
                remapped = remap_integer_labels(stats["raw_counts"], label_names, PAD_UFES_MAP)
                stats["mapped_counts"] = remapped["mapped"]
                stats["unmapped_labels"] = remapped["unmapped"]
                stats["label_names"] = label_names
                print_distribution(remapped["mapped"], "Re-Mapped via ClassLabel")

            manifest = {
                "dataset": "PAD-UFES-20",
                "source": f"huggingface:{ds_id}",
                "config": config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **stats,
            }
            save_manifest("pad-ufes-20", manifest)
            return
        except Exception as e:
            print(f"    FAILED: {e}")

    # Fallback
    print("\n  PAD-UFES-20 not found on HuggingFace.")
    print("  Available from: https://data.mendeley.com/datasets/zr7vgbcyr2/1")
    print("  This dataset contains ~2,298 smartphone clinical images of skin lesions.")
    print("  Classes: ACK (actinic keratosis), BCC, MEL, NEV, SCC, SEK (seborrheic keratosis)")

    manifest = {
        "dataset": "PAD-UFES-20",
        "source": "not-found-on-huggingface",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "requires_manual_download",
        "download_url": "https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        "note": "PAD-UFES-20 requires manual download from Mendeley Data. Contains ~2,298 smartphone images.",
        "expected_total": 2298,
        "expected_classes": {
            "ACK": "akiec (actinic keratosis)",
            "BCC": "bcc (basal cell carcinoma)",
            "MEL": "mel (melanoma)",
            "NEV": "nv (nevus)",
            "SCC": "akiec (squamous cell carcinoma)",
            "SEK": "bkl (seborrheic keratosis)",
        },
        "class_mapping": PAD_UFES_MAP,
    }
    save_manifest("pad-ufes-20", manifest)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print("=" * 72)
    print("DrAgnes Dataset Downloader & Verifier")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)
    print(f"\nTarget DrAgnes classes: {DRAGNES_CLASSES}")
    print(f"Manifest output: {MANIFEST_DIR}")

    # Check torch/MPS availability
    try:
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"PyTorch device: {device}")
    except ImportError:
        print("PyTorch not available (not needed for download-only mode)")

    results: dict[str, str] = {}

    # 1. ISIC Dermoscopy
    try:
        download_isic_datasets()
        results["isic_dermoscopy"] = "success"
    except Exception as e:
        print(f"\n  [ERROR] ISIC download failed: {e}")
        traceback.print_exc()
        results["isic_dermoscopy"] = f"error: {e}"

    # 2. Fitzpatrick17k
    try:
        download_fitzpatrick17k()
        results["fitzpatrick17k"] = "success"
    except Exception as e:
        print(f"\n  [ERROR] Fitzpatrick17k failed: {e}")
        traceback.print_exc()
        results["fitzpatrick17k"] = f"error: {e}"

    # 3. PAD-UFES-20
    try:
        download_pad_ufes()
        results["pad_ufes_20"] = "success"
    except Exception as e:
        print(f"\n  [ERROR] PAD-UFES-20 failed: {e}")
        traceback.print_exc()
        results["pad_ufes_20"] = f"error: {e}"

    # Summary
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for ds_name, status in results.items():
        icon = "OK" if status == "success" else "FAIL"
        print(f"  [{icon}] {ds_name}: {status}")

    save_manifest("_download-summary", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "target_classes": DRAGNES_CLASSES,
    })
    print(f"\nDone. All manifests saved to {MANIFEST_DIR}/")


if __name__ == "__main__":
    main()
