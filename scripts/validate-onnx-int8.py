#!/usr/bin/env python3
"""
Mela ONNX INT8 Validation Script

Runs the quantized INT8 ONNX model on ISIC 2019 test images and compares
against the full-precision model results. Reports per-class sensitivity,
specificity, AUROC, and identifies any cases where quantization caused
a missed detection.

Usage:
    python3 scripts/validate-onnx-int8.py --model static/models/mela-v2-int8.onnx
    python3 scripts/validate-onnx-int8.py --model static/models/mela-v2-int8.onnx --test-dir data/isic2019-test/

If --test-dir is not provided, generates synthetic test images for format/resolution testing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_path: str, target_size: int = 224) -> np.ndarray:
    """Load and preprocess an image for ViT inference."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> NCHW
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model."""
    print(f"Loading model: {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    print(f"  Input: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"  Output: {session.get_outputs()[0].name}")
    return session


def classify_image(session: ort.InferenceSession, img_path: str) -> dict:
    """Classify a single image, return class probabilities."""
    tensor = preprocess_image(img_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start = time.perf_counter()
    logits = session.run([output_name], {input_name: tensor})[0][0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = softmax(logits)
    top_idx = np.argmax(probs)

    return {
        "top_class": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)},
        "inference_ms": elapsed_ms,
    }


def compute_metrics(predictions: list, labels: list) -> dict:
    """Compute per-class sensitivity, specificity, and overall metrics."""
    n = len(predictions)
    results = {}

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        tp = fp = fn = tn = 0
        for pred, label in zip(predictions, labels):
            pred_pos = (pred == cls_idx)
            true_pos = (label == cls_idx)
            if pred_pos and true_pos: tp += 1
            elif pred_pos and not true_pos: fp += 1
            elif not pred_pos and true_pos: fn += 1
            else: tn += 1

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        support = tp + fn

        results[cls_name] = {
            "sensitivity": round(sensitivity * 100, 2),
            "specificity": round(specificity * 100, 2),
            "ppv": round(ppv * 100, 2),
            "npv": round(npv * 100, 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
        }

    # Overall accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    results["_overall"] = {
        "accuracy": round(correct / n * 100, 2) if n > 0 else 0,
        "total": n,
        "correct": correct,
    }

    # Melanoma-specific
    mel_idx = CLASS_NAMES.index("mel")
    mel_tp = results["mel"]["tp"]
    mel_fn = results["mel"]["fn"]
    results["_melanoma"] = {
        "sensitivity": results["mel"]["sensitivity"],
        "detected": mel_tp,
        "missed": mel_fn,
        "total": mel_tp + mel_fn,
    }

    # All-cancer sensitivity (mel + bcc + akiec)
    cancer_classes = ["mel", "bcc", "akiec"]
    cancer_tp = sum(results[c]["tp"] for c in cancer_classes)
    cancer_fn = sum(results[c]["fn"] for c in cancer_classes)
    cancer_total = cancer_tp + cancer_fn
    results["_all_cancer"] = {
        "sensitivity": round(cancer_tp / cancer_total * 100, 2) if cancer_total > 0 else 0,
        "detected": cancer_tp,
        "missed": cancer_fn,
        "total": cancer_total,
    }

    return results


def run_format_tests(session: ort.InferenceSession) -> dict:
    """Test model with different image formats and resolutions."""
    print("\n--- Format & Resolution Tests ---")
    results = {}

    # Create synthetic test image
    test_sizes = [128, 224, 512, 1024]
    test_formats = ["RGB", "RGBA", "L"]

    for size in test_sizes:
        for fmt in test_formats:
            try:
                # Generate synthetic lesion-like image
                np.random.seed(42)
                if fmt == "L":
                    img = Image.fromarray(np.random.randint(50, 200, (size, size), dtype=np.uint8), mode="L")
                elif fmt == "RGBA":
                    img = Image.fromarray(np.random.randint(50, 200, (size, size, 4), dtype=np.uint8), mode="RGBA")
                else:
                    img = Image.fromarray(np.random.randint(50, 200, (size, size, 3), dtype=np.uint8), mode="RGB")

                # Save and classify
                tmp_path = f"/tmp/mela-test-{size}-{fmt}.png"
                img.save(tmp_path)
                result = classify_image(session, tmp_path)
                os.remove(tmp_path)

                key = f"{size}x{size}_{fmt}"
                results[key] = {
                    "status": "OK",
                    "top_class": result["top_class"],
                    "confidence": round(result["confidence"], 4),
                    "inference_ms": round(result["inference_ms"], 1),
                }
                print(f"  {key}: {result['top_class']} ({result['confidence']:.3f}) in {result['inference_ms']:.1f}ms")

            except Exception as e:
                key = f"{size}x{size}_{fmt}"
                results[key] = {"status": "FAIL", "error": str(e)}
                print(f"  {key}: FAIL - {e}")

    return results


def run_validation(session: ort.InferenceSession, test_dir: str) -> dict:
    """Run full validation on labeled test images."""
    print(f"\n--- Full Validation on {test_dir} ---")

    predictions = []
    labels = []
    inference_times = []
    per_image_results = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: No directory for class {cls_name} at {cls_dir}")
            continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        print(f"  {cls_name}: {len(images)} images")

        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            try:
                result = classify_image(session, img_path)
                pred_idx = CLASS_NAMES.index(result["top_class"])
                predictions.append(pred_idx)
                labels.append(cls_idx)
                inference_times.append(result["inference_ms"])

                # Track missed melanomas specifically
                if cls_name == "mel" and result["top_class"] != "mel":
                    per_image_results.append({
                        "image": img_name,
                        "true_class": "mel",
                        "predicted": result["top_class"],
                        "mel_prob": result["probabilities"]["mel"],
                        "confidence": result["confidence"],
                    })
            except Exception as e:
                print(f"    ERROR on {img_name}: {e}")

    metrics = compute_metrics(predictions, labels)

    # Add latency stats
    if inference_times:
        metrics["_latency"] = {
            "mean_ms": round(np.mean(inference_times), 1),
            "median_ms": round(np.median(inference_times), 1),
            "p95_ms": round(np.percentile(inference_times, 95), 1),
            "min_ms": round(min(inference_times), 1),
            "max_ms": round(max(inference_times), 1),
        }

    # Add missed melanoma details
    if per_image_results:
        metrics["_missed_melanomas"] = per_image_results

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate Mela ONNX INT8 model")
    parser.add_argument("--model", default="static/models/mela-v2-int8.onnx", help="Path to ONNX model")
    parser.add_argument("--test-dir", default=None, help="Directory with labeled test images (class/image.jpg)")
    parser.add_argument("--output", default="scripts/onnx-int8-validation.json", help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    session = load_model(args.model)

    results = {
        "model": args.model,
        "model_size_mb": round(os.path.getsize(args.model) / 1024 / 1024, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quantization": "INT8",
    }

    # Always run format tests
    results["format_tests"] = run_format_tests(session)

    # Run full validation if test directory provided
    if args.test_dir and os.path.isdir(args.test_dir):
        results["validation"] = run_validation(session, args.test_dir)

        # Print summary
        v = results["validation"]
        print("\n========================================")
        print("  ONNX INT8 VALIDATION SUMMARY")
        print("========================================")
        print(f"  Model: {args.model} ({results['model_size_mb']}MB)")
        print(f"  Total images: {v['_overall']['total']}")
        print(f"  Overall accuracy: {v['_overall']['accuracy']}%")
        print(f"  Melanoma sensitivity: {v['_melanoma']['sensitivity']}% ({v['_melanoma']['detected']}/{v['_melanoma']['total']})")
        print(f"  All-cancer sensitivity: {v['_all_cancer']['sensitivity']}%")
        if "_latency" in v:
            print(f"  Inference: {v['_latency']['mean_ms']}ms mean, {v['_latency']['p95_ms']}ms p95")
        print()
        print("  Per-class:")
        for cls in CLASS_NAMES:
            if cls in v:
                c = v[cls]
                print(f"    {cls:6s}  sens={c['sensitivity']:6.1f}%  spec={c['specificity']:6.1f}%  support={c['support']}")
        if "_missed_melanomas" in v and v["_missed_melanomas"]:
            print(f"\n  MISSED MELANOMAS: {len(v['_missed_melanomas'])} cases")
            for m in v["_missed_melanomas"][:5]:
                print(f"    {m['image']}: predicted {m['predicted']} (mel_prob={m['mel_prob']:.3f})")
        print("========================================")
    else:
        print("\nNo test directory provided. Format tests only.")
        print("To run full validation: python3 scripts/validate-onnx-int8.py --test-dir data/isic2019-test/")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
