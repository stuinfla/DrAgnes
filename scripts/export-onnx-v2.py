#!/usr/bin/env python3
"""
ADR-122 Phase 1-2: Export DrAgnes v2 ViT classifier to ONNX and quantize to INT8.

Pipeline:
  1. Load v2 safetensors model via transformers
  2. Export to ONNX FP32 (opset 17)
  3. Quantize to dynamic INT8 via onnxruntime
  4. Validate FP32 and INT8 against PyTorch reference
  5. Report file sizes and save validation results
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "dragnes-classifier-v2" / "best"
ONNX_FP32_DIR = SCRIPT_DIR / "dragnes-onnx-v2"
ONNX_INT8_DIR = SCRIPT_DIR / "dragnes-onnx-v2-int8"
VALIDATION_OUT = SCRIPT_DIR / "onnx-v2-validation.json"

# Find a test image from the fitzpatrick cache (any .jpg will do)
IMAGE_CACHE = SCRIPT_DIR / ".fitzpatrick-image-cache"

LABEL_MAP = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


def find_test_image() -> Path:
    """Return the first .jpg found in the fitzpatrick cache."""
    if IMAGE_CACHE.is_dir():
        for f in sorted(IMAGE_CACHE.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                return f
    # Fallback: any image in static/
    static = SCRIPT_DIR.parent / "static"
    if static.is_dir():
        for f in sorted(static.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                return f
    raise FileNotFoundError("No test image found. Place a .jpg in .fitzpatrick-image-cache/")


def load_pytorch_model():
    """Load ViT model from safetensors and return (model, processor)."""
    print(f"[1/6] Loading PyTorch model from {MODEL_DIR}")
    processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    model = ViTForImageClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"       Model loaded: {param_count:,} parameters ({param_count/1e6:.1f}M)")
    return model, processor


def export_onnx_fp32(model: ViTForImageClassification) -> Path:
    """Export model to ONNX FP32 using torch.onnx.export."""
    print(f"[2/6] Exporting ONNX FP32 to {ONNX_FP32_DIR}")
    ONNX_FP32_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ONNX_FP32_DIR / "model.onnx"

    # Create dummy input on CPU (ONNX export must be on CPU)
    dummy = torch.randn(1, 3, 224, 224)

    # Move model to CPU for export
    model_cpu = model.cpu()

    torch.onnx.export(
        model_cpu,
        (dummy,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,            # Use legacy TorchScript exporter (dynamo produces broken ViT graphs in torch 2.10)
        external_data=False,     # Single-file ONNX (no external data files)
    )

    # Validate the exported ONNX model structure
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"       ONNX FP32 model validated (opset {onnx_model.opset_import[0].version})")

    # Copy config files alongside the ONNX model
    for cfg_name in ("config.json", "preprocessor_config.json"):
        src = MODEL_DIR / cfg_name
        if src.exists():
            shutil.copy2(str(src), str(ONNX_FP32_DIR / cfg_name))

    fp32_size = output_path.stat().st_size
    print(f"       FP32 size: {fp32_size / 1e6:.1f} MB")
    return output_path


def quantize_int8(fp32_path: Path) -> Path:
    """Quantize ONNX FP32 model to dynamic INT8."""
    print(f"[3/6] Quantizing to INT8 in {ONNX_INT8_DIR}")
    ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)

    int8_path = ONNX_INT8_DIR / "model_quantized.onnx"

    # Exclude Conv from quantization -- ConvInteger is not supported by
    # the ORT CPU provider, and the patch-embedding Conv contributes
    # negligible weight compared to the 12 transformer MatMul layers.
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=None,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    )

    # Copy config files
    for cfg_name in ("config.json", "preprocessor_config.json"):
        src = MODEL_DIR / cfg_name
        if src.exists():
            shutil.copy2(str(src), str(ONNX_INT8_DIR / cfg_name))

    int8_size = int8_path.stat().st_size
    fp32_size = fp32_path.stat().st_size
    ratio = int8_size / fp32_size
    print(f"       INT8 size: {int8_size / 1e6:.1f} MB (compression ratio: {ratio:.2f}x)")
    return int8_path


def run_pytorch_inference(model, processor, image: Image.Image, device: str) -> np.ndarray:
    """Run inference through PyTorch model, return softmax probabilities."""
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    model_on_device = model.to(device)

    with torch.no_grad():
        outputs = model_on_device(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
    return probs


def run_onnx_inference(onnx_path: Path, processor, image: Image.Image) -> np.ndarray:
    """Run inference through ONNX model on CPU, return softmax probabilities."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])

    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    ort_inputs = {"pixel_values": pixel_values}
    logits = session.run(["logits"], ort_inputs)[0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = (exp_logits / exp_logits.sum(axis=-1, keepdims=True))[0]
    return probs


def validate_models(
    model,
    processor,
    fp32_path: Path,
    int8_path: Path,
    test_image_path: Path,
) -> dict:
    """Validate ONNX models against PyTorch reference. Returns validation dict."""
    print(f"[4/6] Validating models against PyTorch reference")
    print(f"       Test image: {test_image_path.name}")

    image = Image.open(str(test_image_path)).convert("RGB")

    # Determine PyTorch device
    if torch.backends.mps.is_available():
        pt_device = "mps"
    elif torch.cuda.is_available():
        pt_device = "cuda"
    else:
        pt_device = "cpu"
    print(f"       PyTorch device: {pt_device}")

    # --- PyTorch inference ---
    t0 = time.perf_counter()
    pt_probs = run_pytorch_inference(model, processor, image, pt_device)
    pt_time = time.perf_counter() - t0

    # --- ONNX FP32 inference ---
    t0 = time.perf_counter()
    fp32_probs = run_onnx_inference(fp32_path, processor, image)
    fp32_time = time.perf_counter() - t0

    # --- ONNX INT8 inference ---
    t0 = time.perf_counter()
    int8_probs = run_onnx_inference(int8_path, processor, image)
    int8_time = time.perf_counter() - t0

    # Compute differences
    fp32_max_diff = float(np.max(np.abs(pt_probs - fp32_probs)))
    int8_max_diff = float(np.max(np.abs(pt_probs - int8_probs)))
    fp32_mean_diff = float(np.mean(np.abs(pt_probs - fp32_probs)))
    int8_mean_diff = float(np.mean(np.abs(pt_probs - int8_probs)))

    # Print comparison table
    print()
    print("  Class    | PyTorch  | ONNX FP32 | ONNX INT8 | FP32 diff | INT8 diff")
    print("  " + "-" * 75)
    per_class = []
    for i in range(7):
        label = LABEL_MAP[i]
        d_fp32 = abs(pt_probs[i] - fp32_probs[i])
        d_int8 = abs(pt_probs[i] - int8_probs[i])
        status_fp32 = "OK" if d_fp32 < 0.01 else "WARN"
        status_int8 = "OK" if d_int8 < 0.01 else "WARN"
        print(
            f"  {label:<10}| {pt_probs[i]:.6f} | {fp32_probs[i]:.6f}  | {int8_probs[i]:.6f}  | "
            f"{d_fp32:.6f} {status_fp32} | {d_int8:.6f} {status_int8}"
        )
        per_class.append({
            "label": label,
            "pytorch": float(pt_probs[i]),
            "onnx_fp32": float(fp32_probs[i]),
            "onnx_int8": float(int8_probs[i]),
            "diff_fp32": float(d_fp32),
            "diff_int8": float(d_int8),
        })

    print()
    pt_pred = LABEL_MAP[int(np.argmax(pt_probs))]
    fp32_pred = LABEL_MAP[int(np.argmax(fp32_probs))]
    int8_pred = LABEL_MAP[int(np.argmax(int8_probs))]
    print(f"  PyTorch prediction:  {pt_pred} ({pt_probs[np.argmax(pt_probs)]:.4f})")
    print(f"  ONNX FP32 prediction: {fp32_pred} ({fp32_probs[np.argmax(fp32_probs)]:.4f})")
    print(f"  ONNX INT8 prediction: {int8_pred} ({int8_probs[np.argmax(int8_probs)]:.4f})")
    print()

    # FP32 should be near-exact (0.01); INT8 gets 0.05 tolerance (dynamic quantization norm)
    FP32_THRESHOLD = 0.01
    INT8_THRESHOLD = 0.05

    fp32_pass = fp32_max_diff < FP32_THRESHOLD
    int8_pass = int8_max_diff < INT8_THRESHOLD
    predictions_match = pt_pred == fp32_pred == int8_pred

    print(f"  FP32 max diff: {fp32_max_diff:.6f} {'PASS' if fp32_pass else 'FAIL'} (threshold: {FP32_THRESHOLD})")
    print(f"  INT8 max diff: {int8_max_diff:.6f} {'PASS' if int8_pass else 'FAIL'} (threshold: {INT8_THRESHOLD})")
    print(f"  Predictions match: {predictions_match}")
    print(f"  Inference times: PyTorch={pt_time:.3f}s, FP32={fp32_time:.3f}s, INT8={int8_time:.3f}s")

    return {
        "test_image": test_image_path.name,
        "pytorch_device": pt_device,
        "predictions": {
            "pytorch": pt_pred,
            "onnx_fp32": fp32_pred,
            "onnx_int8": int8_pred,
            "match": predictions_match,
        },
        "max_diff": {
            "fp32_vs_pytorch": fp32_max_diff,
            "int8_vs_pytorch": int8_max_diff,
        },
        "mean_diff": {
            "fp32_vs_pytorch": fp32_mean_diff,
            "int8_vs_pytorch": int8_mean_diff,
        },
        "thresholds": {
            "fp32": FP32_THRESHOLD,
            "int8": INT8_THRESHOLD,
        },
        "within_threshold": {
            "fp32": fp32_pass,
            "int8": int8_pass,
        },
        "inference_time_seconds": {
            "pytorch": round(pt_time, 4),
            "onnx_fp32": round(fp32_time, 4),
            "onnx_int8": round(int8_time, 4),
        },
        "per_class": per_class,
    }


def report_sizes(fp32_path: Path, int8_path: Path) -> dict:
    """Report and return file sizes."""
    print(f"[5/6] File size report")
    safetensors_size = (MODEL_DIR / "model.safetensors").stat().st_size
    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size

    print(f"       Safetensors (source): {safetensors_size / 1e6:.1f} MB")
    print(f"       ONNX FP32:            {fp32_size / 1e6:.1f} MB")
    print(f"       ONNX INT8:            {int8_size / 1e6:.1f} MB")
    print(f"       INT8 / FP32:          {int8_size / fp32_size:.2%}")
    print(f"       INT8 / Safetensors:   {int8_size / safetensors_size:.2%}")

    return {
        "safetensors_bytes": safetensors_size,
        "onnx_fp32_bytes": fp32_size,
        "onnx_int8_bytes": int8_size,
        "safetensors_mb": round(safetensors_size / 1e6, 1),
        "onnx_fp32_mb": round(fp32_size / 1e6, 1),
        "onnx_int8_mb": round(int8_size / 1e6, 1),
        "compression_int8_vs_fp32": round(int8_size / fp32_size, 4),
        "compression_int8_vs_safetensors": round(int8_size / safetensors_size, 4),
    }


def main():
    print("=" * 70)
    print("ADR-122 Phase 1-2: ONNX Export + INT8 Quantization (v2 model)")
    print("=" * 70)
    print()

    # Step 1: Load model
    model, processor = load_pytorch_model()

    # Step 2: Export FP32
    fp32_path = export_onnx_fp32(model)

    # Step 3: Quantize INT8
    int8_path = quantize_int8(fp32_path)

    # Step 4: Validate
    test_image_path = find_test_image()
    validation = validate_models(model, processor, fp32_path, int8_path, test_image_path)

    # Step 5: File sizes
    sizes = report_sizes(fp32_path, int8_path)

    # Step 6: Save results
    print(f"\n[6/6] Saving validation results to {VALIDATION_OUT}")
    results = {
        "adr": "ADR-122",
        "phase": "Phase 1-2: ONNX export + INT8 quantization",
        "model": "dragnes-classifier-v2 (ViT-Base, 85.8M params)",
        "source": str(MODEL_DIR),
        "outputs": {
            "onnx_fp32": str(ONNX_FP32_DIR),
            "onnx_int8": str(ONNX_INT8_DIR),
        },
        "file_sizes": sizes,
        "validation": validation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    with open(str(VALIDATION_OUT), "w") as f:
        json.dump(results, f, indent=2)

    print(f"       Saved to {VALIDATION_OUT}")
    print()

    # Summary
    all_pass = validation["within_threshold"]["fp32"] and validation["within_threshold"]["int8"]
    pred_match = validation["predictions"]["match"]

    if all_pass and pred_match:
        print("SUCCESS: All validations passed. ONNX FP32 and INT8 models are within tolerance.")
    elif pred_match:
        print("WARNING: Predictions match but probability diffs exceed threshold.")
        print("         Review per-class diffs above.")
    else:
        print("FAILURE: Predictions differ between models. Investigate before deploying.")

    print()
    print("Output files:")
    print(f"  FP32:       {fp32_path}")
    print(f"  INT8:       {int8_path}")
    print(f"  Validation: {VALIDATION_OUT}")

    return 0 if (all_pass and pred_match) else 1


if __name__ == "__main__":
    sys.exit(main())
