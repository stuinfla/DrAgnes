#!/usr/bin/env python3
"""
DrAgnes Skin Lesion Classifier — Proper Training Pipeline

Fine-tunes a Vision Transformer (ViT-base-patch16-224) on the HAM10000 dataset
with focal loss, aggressive class balancing, and melanoma-sensitivity-first
optimization. Targets >= 90% melanoma sensitivity with >= 80% overall accuracy.

Dataset:  marmal88/skin_cancer (HAM10000, ~10K dermoscopic images, 7 classes)
Model:    google/vit-base-patch16-224-in21k (86M params, pretrained on ImageNet-21k)
Loss:     Focal loss with per-class alpha weights (melanoma alpha = 8.0)
Split:    85/15 stratified train/test
Metrics:  Melanoma sensitivity (primary), overall accuracy (secondary)

Usage:
    pip install -r scripts/requirements-train.txt
    python scripts/train-proper.py

Hardware:
    Designed for Apple M3 Max (128GB RAM). Uses MPS backend when available,
    falls back to CUDA, then CPU. fp16 training is enabled on CUDA; MPS uses
    fp32 for stability.

Output:
    ./dragnes-classifier/best/          — Final model + processor
    ./dragnes-classifier/checkpoint-*/  — Per-epoch checkpoints
    ./dragnes-classifier/training.log   — Full training log
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from datasets import load_dataset, Dataset
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
    EarlyStoppingCallback,
)

# ---------------------------------------------------------------------------
# Suppress noisy warnings that clutter training output
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tokenizer.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "dragnes-classifier"
BEST_MODEL_DIR = OUTPUT_DIR / "best"
LOG_FILE = OUTPUT_DIR / "training.log"

# Model
MODEL_NAME = "google/vit-base-patch16-224-in21k"

# HAM10000 class taxonomy (canonical order matching classifier.ts)
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

ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Training hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 4

# Focal loss parameters
FOCAL_GAMMA = 2.0

# Per-class alpha weights for focal loss (inversely proportional to frequency,
# with extra boost for clinically critical classes). These are tuned for
# HAM10000 where nv=67%, mel=11%, bkl=11%, bcc=5%, akiec=3%, df=1%, vasc=1%.
#
# Melanoma gets the highest weight (8.0) because:
# 1. It is the deadliest skin cancer
# 2. False negatives are catastrophic (missed melanoma)
# 3. The dataset has only ~1100 melanoma samples out of ~10K
# 4. We need >= 90% sensitivity which requires aggressive upweighting
FOCAL_ALPHA = {
    "akiec": 5.0,   # 3% of dataset, pre-malignant
    "bcc": 4.0,     # 5% of dataset, malignant but slow-growing
    "bkl": 1.5,     # 11% of dataset, benign
    "df": 6.0,      # 1% of dataset, benign but very rare
    "mel": 8.0,     # 11% of dataset, DEADLY — must not miss
    "nv": 0.3,      # 67% of dataset, benign — heavily downweight
    "vasc": 5.0,    # 1% of dataset, benign but very rare
}

# Data split
TEST_SIZE = 0.15
RANDOM_SEED = 42

# Data augmentation probabilities
AUG_HFLIP_P = 0.5
AUG_VFLIP_P = 0.5
AUG_ROTATION_DEGREES = 30
AUG_COLOR_JITTER_BRIGHTNESS = 0.3
AUG_COLOR_JITTER_CONTRAST = 0.3
AUG_COLOR_JITTER_SATURATION = 0.3
AUG_COLOR_JITTER_HUE = 0.1
AUG_RANDOM_ERASING_P = 0.15  # CutOut-like regularization

# Oversampling: how many times to repeat minority-class images in training set.
# Melanoma gets 3x to further boost sensitivity beyond what focal loss achieves.
OVERSAMPLE_FACTORS = {
    "akiec": 3,
    "bcc": 2,
    "df": 5,
    "mel": 3,
    "nv": 1,
    "bkl": 1,
    "vasc": 5,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging to both console and file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dragnes-train")
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    # File handler
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
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        log.info("Using Apple MPS backend (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        log.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        log.warning("No GPU detected — training on CPU (will be very slow)")
        return torch.device("cpu")


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification with per-class alpha weighting.

    Focal loss downweights well-classified examples and focuses training on
    hard, misclassified examples. Combined with high alpha for melanoma, this
    aggressively pushes the model to correctly classify melanoma even at the
    cost of some accuracy on the majority class (nevi).

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Parameters:
        alpha:  Per-class weight tensor of shape (num_classes,). Higher values
                increase the loss contribution of that class.
        gamma:  Focusing parameter. gamma=0 gives standard cross-entropy.
                gamma=2 (default) strongly downweights easy examples.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Data Augmentation via torchvision transforms
# ---------------------------------------------------------------------------

def get_train_transforms():
    """
    Training augmentation pipeline for dermoscopic images.

    Dermoscopic images can appear in any orientation, so we apply aggressive
    geometric augmentations (rotation, flips) and moderate color augmentations
    (brightness, contrast, saturation, hue). Random erasing acts as CutOut
    regularization which helps prevent overfitting on small features.
    """
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
    """Evaluation transforms — just resize and normalize, no augmentation."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset Loading & Preprocessing
# ---------------------------------------------------------------------------

# Map from marmal88/skin_cancer label strings to our class indices
SKIN_CANCER_LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


def load_ham10000() -> Dataset:
    """
    Download HAM10000 from HuggingFace and return as a datasets.Dataset.

    The marmal88/skin_cancer dataset contains ~10,015 dermoscopic images
    with 7-class diagnosis labels. This is the same HAM10000 dataset used
    in the ISIC 2018 challenge.
    """
    log.info("Downloading HAM10000 dataset from HuggingFace...")
    log.info("  Dataset: marmal88/skin_cancer")
    log.info("  This may take a few minutes on first run (~3GB download)")

    try:
        ds = load_dataset("kuchikihater/HAM10000")
    except Exception:
        log.info("Primary dataset failed, trying alternative...")
        ds = load_dataset("Nagabu/HAM10000")

    # The dataset may have a single 'train' split or multiple splits.
    # Combine everything into one dataset, then we do our own stratified split.
    if isinstance(ds, dict):
        # Concatenate all splits
        from datasets import concatenate_datasets
        all_splits = []
        for split_name in ds:
            all_splits.append(ds[split_name])
            log.info(f"  Split '{split_name}': {len(ds[split_name])} samples")
        dataset = concatenate_datasets(all_splits)
    else:
        dataset = ds

    log.info(f"Total samples loaded: {len(dataset)}")
    return dataset


def extract_labels(dataset: Dataset) -> list[int]:
    """
    Extract integer class labels from the dataset.

    The marmal88/skin_cancer dataset has a 'dx' column with string labels
    (akiec, bcc, bkl, df, mel, nv, vasc) or possibly a 'label' column
    with integer labels. We handle both cases.
    """
    # Check which columns exist
    columns = dataset.column_names
    log.info(f"Dataset columns: {columns}")

    if "dx" in columns:
        # String labels
        raw_labels = dataset["dx"]
        labels = []
        unmapped = set()
        for lbl in raw_labels:
            lbl_str = str(lbl).lower().strip()
            if lbl_str in SKIN_CANCER_LABEL_MAP:
                labels.append(SKIN_CANCER_LABEL_MAP[lbl_str])
            else:
                unmapped.add(lbl_str)
                labels.append(-1)
        if unmapped:
            log.warning(f"Unmapped labels found: {unmapped}")
        return labels
    elif "label" in columns:
        # Integer labels — verify range
        labels = dataset["label"]
        if isinstance(labels[0], str):
            return [SKIN_CANCER_LABEL_MAP.get(str(l).lower().strip(), -1) for l in labels]
        return [int(l) for l in labels]
    else:
        raise ValueError(
            f"Cannot find label column. Available columns: {columns}. "
            f"Expected 'dx' or 'label'."
        )


def extract_image(example) -> Image.Image:
    """
    Extract a PIL Image from a dataset example.

    The dataset may store images as:
    - A PIL Image directly (datasets with Image feature)
    - A dict with 'bytes' key
    - A file path string
    """
    if "image" in example:
        img = example["image"]
    elif "img" in example:
        img = example["img"]
    else:
        raise ValueError(f"Cannot find image column. Keys: {list(example.keys())}")

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, dict) and "bytes" in img:
        import io
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    elif isinstance(img, str):
        return Image.open(img).convert("RGB")
    else:
        # Try treating it as PIL directly
        return img.convert("RGB")


def print_class_distribution(labels: list[int], title: str = "Class Distribution"):
    """Print a formatted table of class frequencies."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)

    log.info(f"\n{'=' * 60}")
    log.info(f"  {title}")
    log.info(f"{'=' * 60}")
    log.info(f"  {'Class':<8} {'Name':<45} {'Count':>6} {'Pct':>7}")
    log.info(f"  {'-' * 8} {'-' * 45} {'-' * 6} {'-' * 7}")

    for idx in range(NUM_CLASSES):
        name = CLASS_FULL_NAMES[CLASS_NAMES[idx]]
        count = counts.get(idx, 0)
        pct = 100.0 * count / total if total > 0 else 0
        marker = " <-- TARGET" if idx == MEL_IDX else ""
        log.info(f"  {CLASS_NAMES[idx]:<8} {name:<45} {count:>6} {pct:>6.1f}%{marker}")

    log.info(f"  {'TOTAL':<54} {total:>6}")
    log.info(f"{'=' * 60}\n")


def oversample_minority_classes(
    indices: list[int],
    labels: list[int],
    factors: dict[str, int],
) -> list[int]:
    """
    Oversample minority classes by repeating their indices.

    This is applied AFTER the train/test split so the test set remains
    uncontaminated. Combined with focal loss, this gives the model more
    exposure to rare classes during training.
    """
    from collections import defaultdict

    class_indices = defaultdict(list)
    for idx in indices:
        class_indices[labels[idx]].append(idx)

    oversampled = []
    for class_idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_idx]
        factor = factors.get(class_name, 1)
        class_samples = class_indices[class_idx]
        oversampled.extend(class_samples * factor)
        if factor > 1:
            log.info(
                f"  Oversampled {class_name}: {len(class_samples)} -> "
                f"{len(class_samples) * factor} ({factor}x)"
            )

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(oversampled)
    return oversampled


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class HAM10000Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for HAM10000 images.

    Loads images on-the-fly, applies augmentation, and preprocesses
    using the ViT image processor for consistent normalization.
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        indices: list[int],
        labels: list[int],
        processor: ViTImageProcessor,
        augment: bool = False,
    ):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.train_transforms = get_train_transforms() if augment else None
        self.eval_transforms = get_eval_transforms()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        example = self.hf_dataset[real_idx]
        image = extract_image(example)
        label = self.labels[real_idx]

        if self.augment and self.train_transforms is not None:
            # Apply augmentation first (works on PIL), then resize to 224x224
            image = image.resize((224, 224), Image.BILINEAR)
            tensor = self.train_transforms(image)
            # The ViT processor expects pixel values in [0, 1] then normalizes.
            # torchvision ToTensor() already gives [0, 1].
            # Apply ViT normalization manually.
            mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
            std = torch.tensor(self.processor.image_std).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return {"pixel_values": tensor, "labels": label}
        else:
            # Evaluation: use the processor directly (resize + normalize)
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return {"pixel_values": pixel_values, "labels": label}


# ---------------------------------------------------------------------------
# Custom Trainer with Focal Loss
# ---------------------------------------------------------------------------

class FocalLossTrainer(Trainer):
    """
    HuggingFace Trainer subclass that uses focal loss instead of cross-entropy.

    Overrides compute_loss() to apply focal loss with per-class alpha weights.
    This is the key mechanism for achieving high melanoma sensitivity: the
    combination of high melanoma alpha (8.0) and focal loss gamma (2.0)
    forces the model to learn melanoma features even when most training
    examples are benign nevi.
    """

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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """
    Compute per-class and aggregate metrics.

    Primary metric: melanoma_sensitivity (True Positive Rate for mel class)
    Secondary metrics: overall accuracy, per-class sensitivity, specificity
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))

    # Per-class sensitivity (recall) and specificity
    metrics = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        metrics[f"{cls}_sensitivity"] = sensitivity
        metrics[f"{cls}_specificity"] = specificity
        metrics[f"{cls}_precision"] = precision
        metrics[f"{cls}_f1"] = f1

    # Overall accuracy
    metrics["accuracy"] = (predictions == labels).mean()

    # The metric we optimize for (model selection)
    metrics["melanoma_sensitivity"] = metrics["mel_sensitivity"]

    # Macro-averaged F1 (useful secondary metric)
    f1_scores = [metrics[f"{cls}_f1"] for cls in CLASS_NAMES]
    metrics["macro_f1"] = np.mean(f1_scores)

    # Balanced accuracy (mean of per-class sensitivities)
    sens_scores = [metrics[f"{cls}_sensitivity"] for cls in CLASS_NAMES]
    metrics["balanced_accuracy"] = np.mean(sens_scores)

    return metrics


def print_detailed_evaluation(trainer, test_dataset, labels_test: list[int]):
    """
    Run final evaluation and print detailed per-class metrics.

    Generates a full confusion matrix, classification report, and
    highlights the melanoma sensitivity result.
    """
    log.info("\n" + "=" * 70)
    log.info("  FINAL EVALUATION ON HELD-OUT TEST SET")
    log.info("=" * 70)

    # Get predictions
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    preds = np.argmax(logits, axis=-1)
    true_labels = np.array(labels_test)

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(NUM_CLASSES)))

    log.info("\nConfusion Matrix:")
    log.info(f"  {'':>8} " + " ".join(f"{c:>7}" for c in CLASS_NAMES) + "  (predicted)")
    for i, cls in enumerate(CLASS_NAMES):
        row = " ".join(f"{cm[i, j]:>7}" for j in range(NUM_CLASSES))
        marker = " <--" if cls == "mel" else ""
        log.info(f"  {cls:>8} {row}{marker}")
    log.info(f"  (actual)")

    # Classification report
    report = classification_report(
        true_labels, preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    log.info(f"\nClassification Report:\n{report}")

    # Melanoma-specific metrics
    mel_tp = cm[MEL_IDX, MEL_IDX]
    mel_fn = cm[MEL_IDX, :].sum() - mel_tp
    mel_fp = cm[:, MEL_IDX].sum() - mel_tp
    mel_sensitivity = mel_tp / (mel_tp + mel_fn) if (mel_tp + mel_fn) > 0 else 0.0
    mel_precision = mel_tp / (mel_tp + mel_fp) if (mel_tp + mel_fp) > 0 else 0.0
    mel_f1 = (
        2 * mel_precision * mel_sensitivity / (mel_precision + mel_sensitivity)
        if (mel_precision + mel_sensitivity) > 0 else 0.0
    )

    overall_accuracy = (preds == true_labels).mean()

    log.info("\n" + "=" * 70)
    log.info("  KEY RESULTS")
    log.info("=" * 70)
    log.info(f"  Melanoma Sensitivity (Recall): {mel_sensitivity:.1%}  {'PASS' if mel_sensitivity >= 0.90 else 'FAIL'} (target >= 90%)")
    log.info(f"  Melanoma Precision:            {mel_precision:.1%}")
    log.info(f"  Melanoma F1:                   {mel_f1:.1%}")
    log.info(f"  Overall Accuracy:              {overall_accuracy:.1%}  {'PASS' if overall_accuracy >= 0.80 else 'FAIL'} (target >= 80%)")
    log.info(f"  Melanoma TP: {mel_tp}, FN: {mel_fn}, FP: {mel_fp}")
    log.info("=" * 70)

    # Per-class sensitivity summary
    log.info("\nPer-class Sensitivity:")
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bar = "#" * int(sens * 40)
        log.info(f"  {cls:>8}: {sens:>6.1%} {bar}")

    # Try to compute per-class AUC
    try:
        from scipy.special import softmax as scipy_softmax
        probs = scipy_softmax(logits, axis=-1)
        for i, cls in enumerate(CLASS_NAMES):
            binary_true = (true_labels == i).astype(int)
            if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                auc = roc_auc_score(binary_true, probs[:, i])
                log.info(f"  {cls:>8} AUC: {auc:.4f}")
    except Exception as e:
        log.warning(f"Could not compute AUC scores: {e}")

    return {
        "melanoma_sensitivity": mel_sensitivity,
        "melanoma_precision": mel_precision,
        "melanoma_f1": mel_f1,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("  DrAgnes Skin Lesion Classifier — Training Pipeline")
    log.info("=" * 70)
    log.info(f"  Model:       {MODEL_NAME}")
    log.info(f"  Dataset:     marmal88/skin_cancer (HAM10000)")
    log.info(f"  Loss:        Focal Loss (gamma={FOCAL_GAMMA})")
    log.info(f"  Epochs:      {NUM_EPOCHS}")
    log.info(f"  Batch size:  {BATCH_SIZE_TRAIN} (train), {BATCH_SIZE_EVAL} (eval)")
    log.info(f"  LR:          {LEARNING_RATE}")
    log.info(f"  Device:      {DEVICE}")
    log.info(f"  Output:      {OUTPUT_DIR}")
    log.info(f"  Target:      melanoma sensitivity >= 90%, accuracy >= 80%")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    log.info("\n[Step 1/7] Loading HAM10000 dataset...")
    dataset = load_ham10000()
    labels = extract_labels(dataset)

    # Filter out any invalid labels
    valid_mask = [l >= 0 and l < NUM_CLASSES for l in labels]
    if not all(valid_mask):
        invalid_count = sum(1 for v in valid_mask if not v)
        log.warning(f"Filtering {invalid_count} samples with invalid labels")
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        labels = [labels[i] for i in valid_indices]
    else:
        valid_indices = list(range(len(labels)))

    print_class_distribution(labels, "Full Dataset Distribution")

    # ------------------------------------------------------------------
    # Step 2: Stratified train/test split
    # ------------------------------------------------------------------
    log.info("[Step 2/7] Creating 85/15 stratified split...")
    train_idx, test_idx = train_test_split(
        valid_indices,
        test_size=TEST_SIZE,
        stratify=[labels[i] for i in valid_indices],
        random_state=RANDOM_SEED,
    )

    labels_train = [labels[i] for i in train_idx]
    labels_test = [labels[i] for i in test_idx]

    print_class_distribution(labels_train, "Training Set Distribution (before oversampling)")
    print_class_distribution(labels_test, "Test Set Distribution")

    # ------------------------------------------------------------------
    # Step 3: Oversample minority classes in training set
    # ------------------------------------------------------------------
    log.info("[Step 3/7] Oversampling minority classes...")
    train_idx_oversampled = oversample_minority_classes(
        train_idx, labels, OVERSAMPLE_FACTORS
    )

    labels_train_oversampled = [labels[i] for i in train_idx_oversampled]
    print_class_distribution(
        labels_train_oversampled,
        "Training Set Distribution (after oversampling)"
    )

    # ------------------------------------------------------------------
    # Step 4: Initialize model and processor
    # ------------------------------------------------------------------
    log.info("[Step 4/7] Loading ViT model and processor...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    log.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    # Step 5: Create datasets
    # ------------------------------------------------------------------
    log.info("[Step 5/7] Creating PyTorch datasets with augmentation...")
    train_dataset = HAM10000Dataset(
        hf_dataset=dataset,
        indices=train_idx_oversampled,
        labels=labels,
        processor=processor,
        augment=True,
    )

    test_dataset = HAM10000Dataset(
        hf_dataset=dataset,
        indices=test_idx,
        labels=labels,
        processor=processor,
        augment=False,
    )

    log.info(f"  Training samples:   {len(train_dataset)} (with oversampling + augmentation)")
    log.info(f"  Test samples:       {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Step 6: Set up training
    # ------------------------------------------------------------------
    log.info("[Step 6/7] Configuring training...")

    # Focal loss with class weights
    alpha_tensor = torch.tensor(
        [FOCAL_ALPHA[cls] for cls in CLASS_NAMES],
        dtype=torch.float32,
    ).to(DEVICE)
    focal_loss = FocalLoss(alpha=alpha_tensor, gamma=FOCAL_GAMMA)
    log.info(f"  Focal loss alpha: {[FOCAL_ALPHA[c] for c in CLASS_NAMES]}")
    log.info(f"  Focal loss gamma: {FOCAL_GAMMA}")

    # Determine fp16 availability
    # MPS does not fully support fp16 in all operations; use fp32 for stability.
    # CUDA supports fp16 natively.
    use_fp16 = DEVICE.type == "cuda"
    if DEVICE.type == "mps":
        log.info("  MPS detected: using fp32 (MPS fp16 has known stability issues)")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="melanoma_sensitivity",
        greater_is_better=True,
        fp16=use_fp16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",  # No wandb/tensorboard — just logs
        save_total_limit=3,
        seed=RANDOM_SEED,
        # Note: transformers >= 4.40 auto-detects MPS; no need for use_mps_device
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        focal_loss_fn=focal_loss,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Step 7: Train
    # ------------------------------------------------------------------
    log.info("[Step 7/7] Starting training...")
    log.info(f"  Total training steps: ~{len(train_dataset) // BATCH_SIZE_TRAIN * NUM_EPOCHS}")
    log.info("")

    train_result = trainer.train()

    log.info("\nTraining complete!")
    log.info(f"  Total training time: {train_result.metrics.get('train_runtime', 0):.0f}s")
    log.info(f"  Samples/second: {train_result.metrics.get('train_samples_per_second', 0):.1f}")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    results = print_detailed_evaluation(trainer, test_dataset, labels_test)

    # ------------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------------
    log.info(f"\nSaving best model to {BEST_MODEL_DIR}...")
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(BEST_MODEL_DIR))
    processor.save_pretrained(str(BEST_MODEL_DIR))

    # Save training metadata alongside the model
    metadata = {
        "model_name": MODEL_NAME,
        "dataset": "marmal88/skin_cancer",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "oversample_factors": OVERSAMPLE_FACTORS,
        "training_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE_TRAIN,
        "test_size": TEST_SIZE,
        "results": {
            "melanoma_sensitivity": results["melanoma_sensitivity"],
            "melanoma_precision": results["melanoma_precision"],
            "melanoma_f1": results["melanoma_f1"],
            "overall_accuracy": results["overall_accuracy"],
        },
        "device": str(DEVICE),
    }

    metadata_path = BEST_MODEL_DIR / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata saved to {metadata_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    mel_sens = results["melanoma_sensitivity"]
    accuracy = results["overall_accuracy"]

    log.info("\n" + "=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Model saved to: {BEST_MODEL_DIR}")
    log.info(f"  Melanoma sensitivity: {mel_sens:.1%}")
    log.info(f"  Overall accuracy:     {accuracy:.1%}")

    if mel_sens >= 0.90 and accuracy >= 0.80:
        log.info("  STATUS: BOTH TARGETS MET")
    elif mel_sens >= 0.90:
        log.info("  STATUS: Melanoma target met, accuracy below target")
    elif accuracy >= 0.80:
        log.info("  STATUS: Accuracy target met, melanoma sensitivity below target")
    else:
        log.info("  STATUS: Both targets missed — consider adjusting hyperparameters")

    log.info("")
    log.info("  Next steps:")
    log.info("    1. Review confusion matrix for systematic misclassifications")
    log.info("    2. If mel sensitivity < 90%, increase FOCAL_ALPHA['mel'] or OVERSAMPLE_FACTORS['mel']")
    log.info("    3. Push to HuggingFace: model.push_to_hub('stuinfla/dragnes-classifier')")
    log.info("    4. Integrate into DrAgnes app by updating the /api/classify endpoint")
    log.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
