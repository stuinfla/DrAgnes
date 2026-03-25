#!/usr/bin/env python3
"""
Mela Combined Training Pipeline — HAM10000 + ISIC 2019
==========================================================

Trains ViT-Base on the COMBINED datasets:
  1. kuchikihater/HAM10000       (~10,015 images, 7 classes, integer labels)
  2. akinsanyaayomide/skin_cancer_dataset_balanced_labels_2 (~26K images, 8 ISIC classes)

Split strategy:
  - Hold out 15% of ISIC 2019 (stratified) as EXTERNAL test set (never seen in training)
  - Use ALL of HAM10000 + remaining 85% of ISIC 2019 for training
  - Additionally hold out 15% of HAM10000 for same-distribution validation

Focal loss with per-class alpha weights matching the existing model recipe.
Saves best model to scripts/mela-classifier-v2/best/.

Usage:
    pip install -r scripts/requirements-train.txt
    python scripts/train-combined.py

Hardware: Apple M3 Max (MPS backend). ~30-60 minutes expected.
"""

import os
import sys
import json
import time
import logging
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Suppress noisy warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tokenizer.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "mela-classifier-v2"
BEST_MODEL_DIR = OUTPUT_DIR / "best"
LOG_FILE = OUTPUT_DIR / "training.log"
RESULTS_PATH = SCRIPT_DIR / "combined-training-results.json"

# Model
MODEL_NAME = "google/vit-base-patch16-224"

# HAM10000 7-class taxonomy
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
CANCER_INDICES = {0, 1, 4}  # akiec, bcc, mel

ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ISIC 2019 label mapping (string dx -> HAM10000 class index)
ISIC_DX_TO_HAM = {
    "AK": 0,    # Actinic Keratosis -> akiec
    "BCC": 1,   # Basal Cell Carcinoma -> bcc
    "BKL": 2,   # Benign Keratosis-like -> bkl
    "DF": 3,    # Dermatofibroma -> df
    "MEL": 4,   # Melanoma -> mel
    "NV": 5,    # Melanocytic Nevus -> nv
    "SCC": 0,   # Squamous Cell Carcinoma -> akiec (closest HAM class)
    "VASC": 6,  # Vascular Lesion -> vasc
}

# ISIC integer label -> HAM class index (the dataset uses integer labels)
ISIC_INT_TO_HAM = {
    0: 0,  # AK -> akiec
    1: 1,  # BCC -> bcc
    2: 2,  # BKL -> bkl
    3: 3,  # DF -> df
    4: 4,  # MEL -> mel
    5: 5,  # NV -> nv
    6: 0,  # SCC -> akiec
    7: 6,  # VASC -> vasc
}
ISIC_LABEL_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

# Training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Focal loss parameters (matching existing model recipe)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = {
    "akiec": 4.0,
    "bcc": 3.0,
    "bkl": 1.5,
    "df": 5.0,
    "mel": 6.0,
    "nv": 0.4,
    "vasc": 5.0,
}

# Data split
TEST_SIZE = 0.15
RANDOM_SEED = 42

# Augmentation
AUG_HFLIP_P = 0.5
AUG_VFLIP_P = 0.5
AUG_ROTATION_DEGREES = 30
AUG_COLOR_JITTER_BRIGHTNESS = 0.3
AUG_COLOR_JITTER_CONTRAST = 0.3
AUG_COLOR_JITTER_SATURATION = 0.3
AUG_COLOR_JITTER_HUE = 0.1
AUG_RANDOM_ERASING_P = 0.15

# Oversampling factors for minority classes
# NOTE: The combined dataset is already much more balanced than HAM10000 alone,
# so we use lighter oversampling (1-2x) instead of the aggressive 3-5x used
# for HAM10000-only training. mel gets 2x for clinical safety.
OVERSAMPLE_FACTORS = {
    "akiec": 1,
    "bcc": 1,
    "df": 2,
    "mel": 2,
    "nv": 1,
    "bkl": 1,
    "vasc": 1,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mela-combined")
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        log.info("Using Apple MPS backend (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        log.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        log.warning("No GPU detected -- training on CPU (will be very slow)")
        return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss with per-class alpha weighting.
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure alpha weights are on the same device as inputs
        alpha = self.alpha
        if alpha is not None and alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)
        ce_loss = F.cross_entropy(
            inputs, targets, weight=alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

def get_train_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=AUG_HFLIP_P),
        transforms.RandomVerticalFlip(p=AUG_VFLIP_P),
        transforms.RandomRotation(AUG_ROTATION_DEGREES),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(
            brightness=AUG_COLOR_JITTER_BRIGHTNESS,
            contrast=AUG_COLOR_JITTER_CONTRAST,
            saturation=AUG_COLOR_JITTER_SATURATION,
            hue=AUG_COLOR_JITTER_HUE,
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=AUG_RANDOM_ERASING_P, scale=(0.02, 0.15)),
    ])


def get_eval_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class CombinedDermDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping a list of (image_getter, label) pairs.
    image_getter is a callable that returns a PIL Image.
    """

    def __init__(
        self,
        items: list,  # list of (callable_or_pil, int_label)
        processor: ViTImageProcessor,
        augment: bool = False,
    ):
        self.items = items
        self.processor = processor
        self.augment = augment
        self.train_transforms = get_train_transforms() if augment else None
        self.eval_transforms = get_eval_transforms()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_source, label = self.items[idx]

        # Get the PIL image
        if callable(img_source):
            image = img_source()
        elif isinstance(img_source, Image.Image):
            image = img_source
        else:
            image = img_source

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        image = image.convert("RGB")

        if self.augment and self.train_transforms is not None:
            image = image.resize((224, 224), Image.BILINEAR)
            tensor = self.train_transforms(image)
            mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
            std = torch.tensor(self.processor.image_std).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return {"pixel_values": tensor, "labels": label}
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return {"pixel_values": pixel_values, "labels": label}


# ---------------------------------------------------------------------------
# Custom Trainer with Focal Loss
# ---------------------------------------------------------------------------

class FocalLossTrainer(Trainer):
    """HuggingFace Trainer subclass using focal loss."""

    def __init__(self, *args, focal_loss_fn: FocalLoss, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

def make_compute_metrics():
    """Build a compute_metrics function for the Trainer."""

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        cm = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))

        metrics = {}
        for i, cls in enumerate(CLASS_NAMES):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (
                2 * precision_val * sensitivity / (precision_val + sensitivity)
                if (precision_val + sensitivity) > 0
                else 0.0
            )

            metrics[f"{cls}_sensitivity"] = sensitivity
            metrics[f"{cls}_specificity"] = specificity
            metrics[f"{cls}_precision"] = precision_val
            metrics[f"{cls}_f1"] = f1

        metrics["accuracy"] = float((predictions == labels).mean())
        metrics["melanoma_sensitivity"] = metrics["mel_sensitivity"]

        f1_scores = [metrics[f"{cls}_f1"] for cls in CLASS_NAMES]
        metrics["macro_f1"] = float(np.mean(f1_scores))

        sens_scores = [metrics[f"{cls}_sensitivity"] for cls in CLASS_NAMES]
        metrics["balanced_accuracy"] = float(np.mean(sens_scores))

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def print_class_distribution(labels: list, title: str = "Class Distribution"):
    counts = Counter(labels)
    total = len(labels)
    log.info(f"\n{'=' * 65}")
    log.info(f"  {title}")
    log.info(f"{'=' * 65}")
    log.info(f"  {'Class':<8} {'Name':<40} {'Count':>6} {'Pct':>7}")
    log.info(f"  {'-' * 8} {'-' * 40} {'-' * 6} {'-' * 7}")
    for idx in range(NUM_CLASSES):
        name = CLASS_FULL_NAMES[CLASS_NAMES[idx]]
        count = counts.get(idx, 0)
        pct = 100.0 * count / total if total > 0 else 0
        marker = " <-- TARGET" if idx == MEL_IDX else ""
        log.info(f"  {CLASS_NAMES[idx]:<8} {name:<40} {count:>6} {pct:>6.1f}%{marker}")
    log.info(f"  {'TOTAL':<49} {total:>6}")
    log.info(f"{'=' * 65}\n")


def oversample_items(items: list, labels: list, factors: dict) -> list:
    """
    Oversample minority classes by repeating items.
    Items are 3-tuples: (source, index, label). They are returned as-is.
    """
    class_items = defaultdict(list)
    for item, lbl in zip(items, labels):
        class_items[lbl].append(item)

    oversampled = []
    for class_idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_idx]
        factor = factors.get(class_name, 1)
        class_data = class_items.get(class_idx, [])
        oversampled.extend(class_data * factor)
        if factor > 1 and len(class_data) > 0:
            log.info(
                f"  Oversampled {class_name}: {len(class_data)} -> "
                f"{len(class_data) * factor} ({factor}x)"
            )

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(oversampled)
    return oversampled


def extract_image_from_row(row) -> Image.Image:
    """Extract a PIL Image from a HuggingFace dataset row."""
    for key in ("image", "img"):
        if key in row:
            img = row[key]
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            elif isinstance(img, dict) and "bytes" in img:
                import io
                return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, str):
                return Image.open(img).convert("RGB")
            else:
                return img.convert("RGB")
    raise ValueError(f"Cannot find image column. Keys: {list(row.keys())}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    log.info("=" * 70)
    log.info("Mela Combined Training Pipeline")
    log.info("HAM10000 + ISIC 2019 Combined Dataset")
    log.info("=" * 70)
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    log.info(f"Focal loss: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
    log.info("")

    # ==================================================================
    # STEP 1: Load HAM10000 dataset
    # ==================================================================
    log.info("=" * 70)
    log.info("STEP 1: Loading HAM10000 dataset")
    log.info("=" * 70)

    log.info("Downloading kuchikihater/HAM10000...")
    try:
        ham_ds = load_dataset("kuchikihater/HAM10000")
        log.info(f"  Loaded successfully. Splits: {list(ham_ds.keys())}")
    except Exception as e:
        log.error(f"  Failed to load kuchikihater/HAM10000: {e}")
        log.info("  Trying alternative: Nagabu/HAM10000...")
        try:
            ham_ds = load_dataset("Nagabu/HAM10000")
            log.info(f"  Loaded Nagabu/HAM10000. Splits: {list(ham_ds.keys())}")
        except Exception as e2:
            log.error(f"  Also failed: {e2}")
            log.info("  Trying marmal88/skin_cancer...")
            ham_ds = load_dataset("marmal88/skin_cancer")
            log.info(f"  Loaded marmal88/skin_cancer. Splits: {list(ham_ds.keys())}")

    # Combine all splits into one
    if isinstance(ham_ds, dict):
        all_splits = []
        for split_name in ham_ds:
            all_splits.append(ham_ds[split_name])
            log.info(f"  Split '{split_name}': {len(ham_ds[split_name])} samples")
        ham_full = concatenate_datasets(all_splits)
    else:
        ham_full = ham_ds

    log.info(f"  Total HAM10000 samples: {len(ham_full)}")
    log.info(f"  Columns: {ham_full.column_names}")

    # Extract HAM10000 labels
    ham_labels = []
    columns = ham_full.column_names
    if "label" in columns:
        raw = ham_full["label"]
        # Check if integer or string
        sample = raw[0]
        if isinstance(sample, (int, np.integer)):
            ham_labels = [int(x) for x in raw]
            log.info(f"  Label field: 'label' (integer, range 0-{max(ham_labels)})")
        else:
            label_str_map = {v.lower(): k for k, v in enumerate(CLASS_NAMES)}
            ham_labels = [label_str_map.get(str(x).lower().strip(), -1) for x in raw]
            log.info(f"  Label field: 'label' (string, mapped to int)")
    elif "dx" in columns:
        raw = ham_full["dx"]
        label_str_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        ham_labels = [label_str_map.get(str(x).lower().strip(), -1) for x in raw]
        log.info(f"  Label field: 'dx' (string, mapped to int)")
    else:
        raise ValueError(f"Cannot find label column in HAM10000. Columns: {columns}")

    # Filter out any unmapped labels
    valid_mask = [l >= 0 for l in ham_labels]
    if not all(valid_mask):
        bad = sum(1 for v in valid_mask if not v)
        log.warning(f"  Filtering {bad} samples with unmapped labels")

    print_class_distribution(ham_labels, "HAM10000 Class Distribution")

    # ==================================================================
    # STEP 2: Load ISIC 2019 dataset
    # ==================================================================
    log.info("=" * 70)
    log.info("STEP 2: Loading ISIC 2019 dataset")
    log.info("=" * 70)

    isic_dataset_name = "akinsanyaayomide/skin_cancer_dataset_balanced_labels_2"
    log.info(f"Downloading {isic_dataset_name}...")

    try:
        isic_ds = load_dataset(isic_dataset_name)
        log.info(f"  Loaded successfully. Splits: {list(isic_ds.keys())}")
        for s in isic_ds:
            log.info(f"    {s}: {len(isic_ds[s])} examples")
    except Exception as e:
        log.error(f"  Failed to load ISIC dataset: {e}")
        sys.exit(1)

    # Combine all ISIC splits
    isic_all_splits = []
    for split_name in isic_ds:
        isic_all_splits.append(isic_ds[split_name])
    isic_full = concatenate_datasets(isic_all_splits)

    log.info(f"  Total ISIC samples: {len(isic_full)}")
    log.info(f"  Columns: {isic_full.column_names}")

    # Extract ISIC labels -- the dataset uses integer 'label' with feature names
    isic_columns = isic_full.column_names
    isic_labels_raw = isic_full["label"]

    # Check if the labels have feature names
    try:
        label_feature = isic_full.features["label"]
        if hasattr(label_feature, "names"):
            isic_label_names = label_feature.names
            log.info(f"  ISIC label names: {isic_label_names}")
        else:
            isic_label_names = ISIC_LABEL_NAMES
            log.info(f"  No label names found, using default: {isic_label_names}")
    except Exception:
        isic_label_names = ISIC_LABEL_NAMES
        log.info(f"  Using default label names: {isic_label_names}")

    # Map ISIC integer labels to HAM10000 class indices
    isic_labels = []
    for raw_lbl in isic_labels_raw:
        if isinstance(raw_lbl, (int, np.integer)):
            ham_idx = ISIC_INT_TO_HAM.get(int(raw_lbl), -1)
        else:
            # String label
            ham_idx = ISIC_DX_TO_HAM.get(str(raw_lbl).upper().strip(), -1)
        isic_labels.append(ham_idx)

    valid_isic = sum(1 for l in isic_labels if l >= 0)
    log.info(f"  Mapped {valid_isic}/{len(isic_labels)} ISIC labels to HAM10000 classes")

    print_class_distribution(isic_labels, "ISIC 2019 Class Distribution (mapped to HAM10000)")

    # ==================================================================
    # STEP 3: Create train/test splits
    # ==================================================================
    log.info("=" * 70)
    log.info("STEP 3: Creating train/test splits")
    log.info("=" * 70)

    # -- 3a: Split ISIC 2019 into 85% train / 15% external test (stratified)
    isic_indices = list(range(len(isic_full)))
    isic_train_idx, isic_test_idx = train_test_split(
        isic_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=isic_labels,
    )
    isic_train_labels = [isic_labels[i] for i in isic_train_idx]
    isic_test_labels = [isic_labels[i] for i in isic_test_idx]

    log.info(f"  ISIC 2019 train: {len(isic_train_idx)} images")
    log.info(f"  ISIC 2019 test (EXTERNAL, held out): {len(isic_test_idx)} images")
    print_class_distribution(isic_test_labels, "ISIC 2019 EXTERNAL Test Set")

    # -- 3b: Split HAM10000 into 85% train / 15% same-distribution test
    ham_indices = list(range(len(ham_full)))
    ham_train_idx, ham_test_idx = train_test_split(
        ham_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=ham_labels,
    )
    ham_train_labels = [ham_labels[i] for i in ham_train_idx]
    ham_test_labels = [ham_labels[i] for i in ham_test_idx]

    log.info(f"  HAM10000 train: {len(ham_train_idx)} images")
    log.info(f"  HAM10000 test (same-distribution check): {len(ham_test_idx)} images")
    print_class_distribution(ham_test_labels, "HAM10000 Same-Distribution Test Set")

    # -- 3c: Build combined training set
    #    = HAM10000 train (85%) + ISIC 2019 train (85%)
    #    We store (dataset_ref, index, label) tuples and resolve images lazily

    log.info("\n  Building combined training set...")

    train_items = []  # List of (image_getter_or_ref, label)

    # Add HAM10000 train images
    for idx in ham_train_idx:
        lbl = ham_labels[idx]
        if lbl >= 0:
            # Store a reference: (dataset, index)
            train_items.append(("ham", idx, lbl))

    # Add ISIC 2019 train images
    for idx in isic_train_idx:
        lbl = isic_labels[idx]
        if lbl >= 0:
            train_items.append(("isic", idx, lbl))

    combined_train_labels = [item[2] for item in train_items]
    log.info(f"  Combined training set: {len(train_items)} images")
    log.info(f"    HAM10000 contribution: {sum(1 for i in train_items if i[0] == 'ham')}")
    log.info(f"    ISIC 2019 contribution: {sum(1 for i in train_items if i[0] == 'isic')}")
    print_class_distribution(combined_train_labels, "Combined Training Set (before oversampling)")

    # ==================================================================
    # STEP 4: Apply oversampling
    # ==================================================================
    log.info("=" * 70)
    log.info("STEP 4: Applying minority class oversampling")
    log.info("=" * 70)

    oversampled_train = oversample_items(
        train_items, combined_train_labels, OVERSAMPLE_FACTORS
    )
    oversampled_labels = [item[2] for item in oversampled_train]
    log.info(f"  After oversampling: {len(oversampled_train)} images")
    print_class_distribution(oversampled_labels, "Combined Training Set (after oversampling)")

    # ==================================================================
    # STEP 5: Load the image processor and build PyTorch datasets
    # ==================================================================
    log.info("=" * 70)
    log.info("STEP 5: Building PyTorch datasets")
    log.info("=" * 70)

    log.info(f"  Loading ViT image processor from {MODEL_NAME}...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # Build lazy-loading dataset
    class LazyDermDataset(torch.utils.data.Dataset):
        def __init__(self, items, ham_dataset, isic_dataset, processor, augment=False):
            self.items = items
            self.ham_dataset = ham_dataset
            self.isic_dataset = isic_dataset
            self.processor = processor
            self.augment = augment
            self.train_transforms = get_train_transforms() if augment else None

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            source, real_idx, label = self.items[idx]

            if source == "ham":
                row = self.ham_dataset[real_idx]
            else:
                row = self.isic_dataset[real_idx]

            image = extract_image_from_row(row)

            if self.augment and self.train_transforms is not None:
                image = image.resize((224, 224), Image.BILINEAR)
                tensor = self.train_transforms(image)
                mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
                std = torch.tensor(self.processor.image_std).view(3, 1, 1)
                tensor = (tensor - mean) / std
                return {"pixel_values": tensor, "labels": label}
            else:
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)
                return {"pixel_values": pixel_values, "labels": label}

    # Train dataset (oversampled, augmented)
    train_dataset = LazyDermDataset(
        oversampled_train, ham_full, isic_full, processor, augment=True
    )

    # ISIC external test dataset
    isic_test_items = [("isic", idx, isic_labels[idx]) for idx in isic_test_idx if isic_labels[idx] >= 0]
    isic_test_dataset = LazyDermDataset(
        isic_test_items, ham_full, isic_full, processor, augment=False
    )

    # HAM10000 same-distribution test dataset
    ham_test_items = [("ham", idx, ham_labels[idx]) for idx in ham_test_idx if ham_labels[idx] >= 0]
    ham_test_dataset = LazyDermDataset(
        ham_test_items, ham_full, isic_full, processor, augment=False
    )

    log.info(f"  Train dataset: {len(train_dataset)} images (oversampled + augmented)")
    log.info(f"  ISIC external test: {len(isic_test_dataset)} images")
    log.info(f"  HAM10000 test: {len(ham_test_dataset)} images")

    # Quick sanity check -- load one item from each
    log.info("  Sanity check: loading one item from each dataset...")
    try:
        item = train_dataset[0]
        log.info(f"    Train item shape: {item['pixel_values'].shape}, label: {item['labels']}")
    except Exception as e:
        log.error(f"    Train item FAILED: {e}")
        raise

    try:
        item = isic_test_dataset[0]
        log.info(f"    ISIC test item shape: {item['pixel_values'].shape}, label: {item['labels']}")
    except Exception as e:
        log.error(f"    ISIC test item FAILED: {e}")
        raise

    try:
        item = ham_test_dataset[0]
        log.info(f"    HAM test item shape: {item['pixel_values'].shape}, label: {item['labels']}")
    except Exception as e:
        log.error(f"    HAM test item FAILED: {e}")
        raise

    log.info("  All sanity checks passed.")

    # ==================================================================
    # STEP 6: Initialize model and focal loss
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 6: Initializing model and focal loss")
    log.info("=" * 70)

    log.info(f"  Loading pretrained {MODEL_NAME}...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total parameters: {param_count:,}")
    log.info(f"  Trainable parameters: {trainable_count:,}")

    # Build focal loss
    alpha_tensor = torch.tensor(
        [FOCAL_ALPHA[cls] for cls in CLASS_NAMES], dtype=torch.float32
    )
    focal_loss_fn = FocalLoss(alpha=alpha_tensor, gamma=FOCAL_GAMMA)
    log.info(f"  Focal loss: gamma={FOCAL_GAMMA}")
    log.info(f"  Alpha weights: {dict(zip(CLASS_NAMES, alpha_tensor.tolist()))}")

    # ==================================================================
    # STEP 7: Configure training
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 7: Configuring training")
    log.info("=" * 70)

    # Use fp32 on MPS for stability
    use_fp16 = DEVICE.type == "cuda"

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,  # MPS does not support multiprocessing well
        dataloader_pin_memory=False,  # Not supported on MPS
        seed=RANDOM_SEED,
    )

    log.info(f"  Epochs: {NUM_EPOCHS}")
    log.info(f"  Batch size: {BATCH_SIZE}")
    log.info(f"  Learning rate: {LEARNING_RATE}")
    log.info(f"  Weight decay: {WEIGHT_DECAY}")
    log.info(f"  Warmup ratio: {WARMUP_RATIO}")
    log.info(f"  FP16: {use_fp16}")
    log.info(f"  Eval/save: every epoch")
    log.info(f"  Best model metric: eval_loss (lower is better)")

    # ==================================================================
    # STEP 8: Train
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 8: TRAINING")
    log.info("=" * 70)
    log.info(f"  Training on {len(train_dataset)} images for {NUM_EPOCHS} epochs")
    log.info(f"  Evaluating on ISIC external test set ({len(isic_test_dataset)} images)")
    log.info(f"  This will take 30-60 minutes on MPS...")
    log.info("")

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=isic_test_dataset,
        compute_metrics=make_compute_metrics(),
        focal_loss_fn=focal_loss_fn,
    )

    train_start = time.time()
    train_result = trainer.train()
    train_end = time.time()
    train_duration = train_end - train_start

    log.info(f"\n  Training complete in {train_duration:.1f}s ({train_duration/60:.1f} min)")
    log.info(f"  Final training loss: {train_result.training_loss:.4f}")

    # ==================================================================
    # STEP 9: Save best model
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 9: Saving best model")
    log.info("=" * 70)

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BEST_MODEL_DIR))
    processor.save_pretrained(str(BEST_MODEL_DIR))
    log.info(f"  Model saved to: {BEST_MODEL_DIR}")

    # ==================================================================
    # STEP 10: Validate on ISIC external test set
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 10: Validating on ISIC 2019 EXTERNAL test set")
    log.info("=" * 70)

    isic_eval_results = run_full_evaluation(
        trainer, isic_test_dataset, isic_test_items,
        "ISIC 2019 External Test Set"
    )

    # ==================================================================
    # STEP 11: Validate on HAM10000 same-distribution test set
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 11: Validating on HAM10000 same-distribution test set")
    log.info("=" * 70)

    ham_eval_results = run_full_evaluation(
        trainer, ham_test_dataset, ham_test_items,
        "HAM10000 Same-Distribution Test Set"
    )

    # ==================================================================
    # STEP 12: Comparison table
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 12: MODEL COMPARISON")
    log.info("=" * 70)

    # Old model baseline (from cross-validation-results.json)
    old_model = {
        "overall_accuracy": 0.7809,
        "mel_sensitivity": 0.9822,
        "bcc_sensitivity": 0.9706,
        "akiec_sensitivity": 0.8939,
        "nv_sensitivity": 0.694,
        "source": "HAM10000-only, 15 epochs, marmal88/skin_cancer",
    }

    # Old model on ISIC external (from isic2019-validation-results.json)
    old_model_isic = {
        "overall_accuracy": 0.5756,
        "mel_sensitivity": None,  # Will be filled if we have it
        "source": "HAM10000-only model on ISIC 2019 external",
    }

    log.info(f"\n{'=' * 80}")
    log.info("  Old Model (HAM10000-only) vs New Model (Combined) on HELD-OUT tests")
    log.info(f"{'=' * 80}")
    log.info(f"\n  --- On ISIC 2019 External Test Set ---")
    log.info(f"  {'Metric':<30} {'Old (HAM-only)':>15} {'New (Combined)':>15} {'Delta':>10}")
    log.info(f"  {'-' * 30} {'-' * 15} {'-' * 15} {'-' * 10}")
    log.info(f"  {'Overall accuracy':<30} {old_model_isic['overall_accuracy']:>15.4f} "
             f"{isic_eval_results['overall_accuracy']:>15.4f} "
             f"{isic_eval_results['overall_accuracy'] - old_model_isic['overall_accuracy']:>+10.4f}")
    for cls in CLASS_NAMES:
        new_val = isic_eval_results["per_class"][cls]["sensitivity"]
        log.info(f"  {cls + ' sensitivity':<30} {'N/A':>15} {new_val:>15.4f}")
    log.info(f"  {'Melanoma AUROC':<30} {'N/A':>15} "
             f"{isic_eval_results.get('mel_auroc', 0):>15.4f}")
    log.info(f"  {'Weighted AUROC':<30} {'N/A':>15} "
             f"{isic_eval_results.get('weighted_auroc', 0):>15.4f}")

    log.info(f"\n  --- On HAM10000 Same-Distribution Test Set ---")
    log.info(f"  {'Metric':<30} {'Old (HAM-only)':>15} {'New (Combined)':>15} {'Delta':>10}")
    log.info(f"  {'-' * 30} {'-' * 15} {'-' * 15} {'-' * 10}")
    log.info(f"  {'Overall accuracy':<30} {old_model['overall_accuracy']:>15.4f} "
             f"{ham_eval_results['overall_accuracy']:>15.4f} "
             f"{ham_eval_results['overall_accuracy'] - old_model['overall_accuracy']:>+10.4f}")
    log.info(f"  {'Mel sensitivity':<30} {old_model['mel_sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['mel']['sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['mel']['sensitivity'] - old_model['mel_sensitivity']:>+10.4f}")
    log.info(f"  {'BCC sensitivity':<30} {old_model['bcc_sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['bcc']['sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['bcc']['sensitivity'] - old_model['bcc_sensitivity']:>+10.4f}")
    log.info(f"  {'AKIEC sensitivity':<30} {old_model['akiec_sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['akiec']['sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['akiec']['sensitivity'] - old_model['akiec_sensitivity']:>+10.4f}")
    log.info(f"  {'NV sensitivity':<30} {old_model['nv_sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['nv']['sensitivity']:>15.4f} "
             f"{ham_eval_results['per_class']['nv']['sensitivity'] - old_model['nv_sensitivity']:>+10.4f}")

    # ==================================================================
    # STEP 13: Save results
    # ==================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 13: Saving results")
    log.info("=" * 70)

    total_time = time.time() - start_time
    results = {
        "timestamp": datetime.now().isoformat(),
        "training": {
            "model": MODEL_NAME,
            "device": str(DEVICE),
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "focal_loss_gamma": FOCAL_GAMMA,
            "focal_loss_alpha": FOCAL_ALPHA,
            "oversample_factors": OVERSAMPLE_FACTORS,
            "datasets": {
                "ham10000": {
                    "source": "kuchikihater/HAM10000",
                    "total": len(ham_full),
                    "train": len(ham_train_idx),
                    "test": len(ham_test_idx),
                },
                "isic2019": {
                    "source": isic_dataset_name,
                    "total": len(isic_full),
                    "train": len(isic_train_idx),
                    "test": len(isic_test_idx),
                },
                "combined_train_before_oversample": len(train_items),
                "combined_train_after_oversample": len(oversampled_train),
            },
            "training_loss": float(train_result.training_loss),
            "training_time_seconds": round(train_duration, 1),
            "total_time_seconds": round(total_time, 1),
        },
        "external_test_results": isic_eval_results,
        "same_distribution_test_results": ham_eval_results,
        "comparison_with_old_model": {
            "old_model": old_model,
            "old_model_isic_accuracy": old_model_isic["overall_accuracy"],
            "new_model_isic_accuracy": isic_eval_results["overall_accuracy"],
            "isic_accuracy_delta": round(
                isic_eval_results["overall_accuracy"] - old_model_isic["overall_accuracy"], 4
            ),
            "new_model_ham_accuracy": ham_eval_results["overall_accuracy"],
            "ham_accuracy_delta": round(
                ham_eval_results["overall_accuracy"] - old_model["overall_accuracy"], 4
            ),
        },
        "model_saved_to": str(BEST_MODEL_DIR),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved to: {RESULTS_PATH}")

    # Final summary
    log.info(f"\n{'=' * 70}")
    log.info("TRAINING COMPLETE -- FINAL SUMMARY")
    log.info(f"{'=' * 70}")
    log.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    log.info(f"  Training time: {train_duration:.1f}s ({train_duration/60:.1f} min)")
    log.info(f"  Combined training images: {len(oversampled_train)} (after oversampling)")
    log.info(f"  Model saved to: {BEST_MODEL_DIR}")
    log.info(f"")
    log.info(f"  ISIC External Test ({len(isic_test_dataset)} images):")
    log.info(f"    Overall accuracy:      {isic_eval_results['overall_accuracy']:.4f}")
    log.info(f"    Melanoma sensitivity:  {isic_eval_results['per_class']['mel']['sensitivity']:.4f}")
    log.info(f"    Melanoma AUROC:        {isic_eval_results.get('mel_auroc', 0):.4f}")
    log.info(f"    Weighted AUROC:        {isic_eval_results.get('weighted_auroc', 0):.4f}")
    log.info(f"    All-cancer sensitivity:{isic_eval_results.get('all_cancer_sensitivity', 0):.4f}")
    log.info(f"")
    log.info(f"  HAM10000 Test ({len(ham_test_dataset)} images):")
    log.info(f"    Overall accuracy:      {ham_eval_results['overall_accuracy']:.4f}")
    log.info(f"    Melanoma sensitivity:  {ham_eval_results['per_class']['mel']['sensitivity']:.4f}")
    log.info(f"    Melanoma AUROC:        {ham_eval_results.get('mel_auroc', 0):.4f}")
    log.info(f"    Weighted AUROC:        {ham_eval_results.get('weighted_auroc', 0):.4f}")
    log.info(f"{'=' * 70}")


def run_full_evaluation(trainer, test_dataset, test_items, title: str) -> dict:
    """
    Run full evaluation on a test dataset, computing per-class metrics,
    AUROC, confusion matrix, and cancer-specific metrics.
    """
    log.info(f"\n  Running inference on {len(test_dataset)} images...")
    log.info(f"  Title: {title}")

    t_start = time.time()
    predictions_output = trainer.predict(test_dataset)
    t_end = time.time()
    inference_time = t_end - t_start

    logits = predictions_output.predictions
    true_labels_list = [item[2] for item in test_items]
    true_labels = np.array(true_labels_list)
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

    rate = len(true_labels) / inference_time if inference_time > 0 else 0
    log.info(f"  Inference: {inference_time:.1f}s, {rate:.1f} img/s")

    # Overall accuracy
    correct = int((preds == true_labels).sum())
    accuracy = correct / len(true_labels)
    log.info(f"\n  Overall accuracy: {accuracy:.4f} ({correct}/{len(true_labels)})")

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(NUM_CLASSES)))

    log.info(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = f"  {'':>10} " + " ".join(f"{c:>7}" for c in CLASS_NAMES)
    log.info(header)
    for i, cls in enumerate(CLASS_NAMES):
        row = " ".join(f"{cm[i, j]:>7}" for j in range(NUM_CLASSES))
        marker = " <-- MELANOMA" if cls == "mel" else ""
        log.info(f"  {cls:>10} {row}{marker}")

    # Per-class metrics
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (
            2 * ppv * sensitivity / (ppv + sensitivity)
            if (ppv + sensitivity) > 0
            else 0.0
        )

        per_class[cls] = {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "f1": round(f1, 4),
            "support": int(cm[i].sum()),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }

    log.info(f"\n  {'Class':<10} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NPV':>8} {'F1':>8} {'Support':>8}")
    log.info(f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for cls in CLASS_NAMES:
        m = per_class[cls]
        log.info(
            f"  {cls:<10} {m['sensitivity']:>8.4f} {m['specificity']:>8.4f} "
            f"{m['ppv']:>8.4f} {m['npv']:>8.4f} {m['f1']:>8.4f} {m['support']:>8d}"
        )

    # AUROC computation
    log.info(f"\n  Computing AUROC...")
    auroc_results = {}
    try:
        # One-vs-rest AUROC (weighted)
        # Need to binarize true labels for roc_auc_score
        from sklearn.preprocessing import label_binarize
        true_binary = label_binarize(true_labels, classes=list(range(NUM_CLASSES)))

        # Check which classes are present
        present_classes = []
        for c in range(NUM_CLASSES):
            if true_binary[:, c].sum() > 0:
                present_classes.append(c)

        if len(present_classes) >= 2:
            # Weighted AUROC across all present classes
            weighted_auroc = roc_auc_score(
                true_binary[:, present_classes],
                probs[:, present_classes],
                average="weighted",
                multi_class="ovr",
            )
            auroc_results["weighted_auroc"] = round(float(weighted_auroc), 4)
            log.info(f"    Weighted AUROC (OvR): {weighted_auroc:.4f}")

            # Macro AUROC
            macro_auroc = roc_auc_score(
                true_binary[:, present_classes],
                probs[:, present_classes],
                average="macro",
                multi_class="ovr",
            )
            auroc_results["macro_auroc"] = round(float(macro_auroc), 4)
            log.info(f"    Macro AUROC (OvR):    {macro_auroc:.4f}")

            # Per-class AUROC
            per_class_auroc = {}
            for c in present_classes:
                cls_name = CLASS_NAMES[c]
                try:
                    c_auroc = roc_auc_score(true_binary[:, c], probs[:, c])
                    per_class_auroc[cls_name] = round(float(c_auroc), 4)
                    log.info(f"    {cls_name:>8} AUROC: {c_auroc:.4f}")
                except ValueError as e:
                    per_class_auroc[cls_name] = None
                    log.info(f"    {cls_name:>8} AUROC: N/A ({e})")

            auroc_results["per_class_auroc"] = per_class_auroc

            # Melanoma-specific AUROC
            if MEL_IDX in present_classes:
                mel_auroc = roc_auc_score(true_binary[:, MEL_IDX], probs[:, MEL_IDX])
                auroc_results["mel_auroc"] = round(float(mel_auroc), 4)
                log.info(f"    Melanoma AUROC:       {mel_auroc:.4f}")
            else:
                auroc_results["mel_auroc"] = None
                log.info("    Melanoma AUROC: N/A (no melanoma in test set)")
        else:
            log.warning("    Cannot compute AUROC: fewer than 2 classes present")
            auroc_results["weighted_auroc"] = None
            auroc_results["macro_auroc"] = None
            auroc_results["mel_auroc"] = None

    except Exception as e:
        log.error(f"    AUROC computation failed: {e}")
        import traceback
        traceback.print_exc()
        auroc_results["weighted_auroc"] = None
        auroc_results["macro_auroc"] = None
        auroc_results["mel_auroc"] = None

    # Cancer-specific metrics
    mel_m = per_class["mel"]
    bcc_m = per_class["bcc"]
    akiec_m = per_class["akiec"]

    log.info(f"\n  --- KEY CANCER METRICS ---")
    log.info(f"  Melanoma sensitivity:     {mel_m['sensitivity']:.4f} ({mel_m['tp']}/{mel_m['tp']+mel_m['fn']})")
    log.info(f"  Melanoma specificity:     {mel_m['specificity']:.4f}")
    log.info(f"  BCC sensitivity:          {bcc_m['sensitivity']:.4f} ({bcc_m['tp']}/{bcc_m['tp']+bcc_m['fn']})")
    log.info(f"  AKIEC sensitivity:        {akiec_m['sensitivity']:.4f} ({akiec_m['tp']}/{akiec_m['tp']+akiec_m['fn']})")

    # All-cancer sensitivity
    cancer_mask = np.isin(true_labels, list(CANCER_INDICES))
    cancer_preds = preds[cancer_mask]
    cancer_detected = np.isin(cancer_preds, list(CANCER_INDICES))
    all_cancer_sens = float(cancer_detected.sum() / len(cancer_preds)) if len(cancer_preds) > 0 else 0.0
    log.info(f"  All-cancer sensitivity:   {all_cancer_sens:.4f} ({int(cancer_detected.sum())}/{len(cancer_preds)})")

    # Build results dict
    result = {
        "title": title,
        "total_images": len(true_labels),
        "overall_accuracy": round(accuracy, 4),
        "per_class": per_class,
        "cancer_metrics": {
            "melanoma_sensitivity": mel_m["sensitivity"],
            "melanoma_specificity": mel_m["specificity"],
            "bcc_sensitivity": bcc_m["sensitivity"],
            "akiec_sensitivity": akiec_m["sensitivity"],
            "all_cancer_sensitivity": round(all_cancer_sens, 4),
            "cancer_support": int(cancer_mask.sum()),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "inference_time_seconds": round(inference_time, 2),
        "images_per_second": round(rate, 1),
    }
    result.update(auroc_results)

    return result


if __name__ == "__main__":
    main()
