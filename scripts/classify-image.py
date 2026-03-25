#!/usr/bin/env python3
"""
Classify a single image using the trained Mela ViT model.

Usage:
    python3 scripts/classify-image.py <image_path>

Outputs JSON array of {label, score} objects sorted by score descending.
The model was trained on HAM10000 (10,015 images) with 98.2% melanoma sensitivity.
"""
import sys
import json
import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "mela-classifier", "best")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: classify-image.py <image_path>"}), file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(MODEL_DIR):
        print(json.dumps({"error": f"Model directory not found: {MODEL_DIR}"}), file=sys.stderr)
        sys.exit(1)

    # Load model and processor
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Build results
    results = []
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        results.append({"label": name, "score": round(float(prob), 6)})

    results.sort(key=lambda x: x["score"], reverse=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
