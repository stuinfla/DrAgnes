Updated: 2026-03-25 | Version 1.0.0
Created: 2026-03-25

# Dr. Agnes -- Complete Project History and Context

This document captures everything a new session needs to understand how Dr. Agnes got to where it is today. Every claim cites its evidence file. Read this before touching any code.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Timeline](#2-timeline)
3. [Technical Journey -- From 0% to 95.97%](#3-technical-journey----from-0-to-9597)
4. [Architecture -- The Full Pipeline](#4-architecture----the-full-pipeline)
5. [Key Files](#5-key-files)
6. [Evidence Chain](#6-evidence-chain)
7. [ADR Summary](#7-adr-summary)
8. [Infrastructure](#8-infrastructure)
9. [Known Issues and Limitations](#9-known-issues-and-limitations)
10. [What's Next](#10-whats-next)

---

## 1. Project Overview

### What Dr. Agnes Is

Dr. Agnes is an open-source, browser-based AI skin cancer screening tool. A user photographs a mole, spot, or mark on their skin with their phone camera and receives a clear recommendation: "Looks healthy," "Worth monitoring," or "See a dermatologist." No special equipment is needed.

The system classifies dermoscopic and clinical images into 7 diagnostic categories from the HAM10000 taxonomy:

| Abbreviation | Full Name | Clinical Significance |
|---|---|---|
| mel | Melanoma | Malignant -- the deadliest skin cancer |
| bcc | Basal Cell Carcinoma | Malignant -- most common skin cancer |
| akiec | Actinic Keratosis / Intraepithelial Carcinoma | Pre-malignant / malignant |
| bkl | Benign Keratosis | Benign |
| df | Dermatofibroma | Benign |
| nv | Melanocytic Nevus (common mole) | Benign |
| vasc | Vascular Lesion | Benign |

### Who It's For

**Primary audience: Consumers.** The 5 billion people who will never see a dermatologist. The fundamental user question is: "I found something on my skin. Should I worry?"

**Secondary audience: Clinicians.** GPs, nurse practitioners, and dermatologists who want a second opinion or screening aid. The same engine serves both audiences through a consumer translation layer (plain English) and a clinical details layer (ABCDE scores, TDS, 7-point checklist, ICD-10 codes, referral letters).

### The Headline Numbers

- **95.97% melanoma sensitivity** (95% CI: 94.5% - 97.4%) on external ISIC 2019 data (3,901 images)
  - Source: `scripts/combined-training-results.json`
- **Melanoma AUROC: 0.960** on ISIC 2019 external data
  - Source: `scripts/combined-training-results.json`
- **All-cancer sensitivity: 98.3%** (mel + bcc + akiec, 1,848/2,058 on ISIC 2019)
  - Source: `scripts/combined-training-results.json`
- **V1+V2 ensemble: 99.4% melanoma sensitivity** on HAM10000 holdout
  - Source: `scripts/ensemble-validation-results.json`
- **ONNX deployment: 85MB INT8, ~155ms inference, fully offline**
- **NPV: 99.06%** -- when the system says "no concern," it is correct 99% of the time
- **NNB: 2.1** -- Number Needed to Biopsy (DermaSensor's is 6.25)

**RESEARCH USE ONLY -- Not FDA-cleared.** Every number must be cited to a JSON evidence file.

### The Core Philosophy

1. **Sensitivity over specificity.** A false negative kills. A false positive inconveniences.
2. **The Honesty Principle.** Every number cites its evidence source. Limitations are documented more prominently than capabilities. Failures are published.
3. **Multi-layer defense.** A single neural network is not trustworthy enough for cancer screening. The 4-layer ensemble + safety gates catch what any single model misses.
4. **Privacy by architecture.** Images never leave the device (ONNX offline inference). Privacy is structural, not policy-dependent.
5. **Bayesian honesty.** Risk scores, not binary alarms. At 2% prevalence, even 95.97% sensitivity produces a PPV of only 8.9% for binary "melanoma yes/no." The system outputs calibrated post-test probabilities across 5 risk tiers.

---

## 2. Timeline

### October 2025 -- Project Inception

- RuVector project started as an AI vector intelligence platform
- DrAgnes conceived as an example application demonstrating the RuVector CNN stack
- SvelteKit prototype with placeholder color-histogram classifier

### March 21, 2026 -- v0.1.0: Initial Prototype

- SvelteKit PWA with camera capture and DermLite device selection
- HAM10000 Bayesian demographic adjustment
- Privacy pipeline: EXIF stripping, differential privacy (epsilon=1.0), witness chain (SHAKE-256)
- Pi-brain integration for collective case sharing (opt-in)
- 4 tabs: Capture, Results, History, Settings
- Classification via single HuggingFace model (actavkid ViT-Large)
- **Result:** Demo-quality only. The classifier was a color histogram -- not real.

### March 22, 2026 -- v0.2.0: Real Image Analysis Engine

- **1,890-line TypeScript CV engine** (`image-analysis.ts`): Otsu segmentation in CIELAB, principal-axis asymmetry, 8-octant border analysis, k-means++ color clustering (k=6 in LAB space), GLCM texture, LBP structure detection, attention maps
- **Dual-model ViT ensemble** (Anwarkh1 ViT-Base + skintaglabs SigLIP) via HuggingFace Inference API
- **Literature-derived logistic regression** (`trained-weights.ts`): 20-feature x 7-class weight matrix, every weight cited to published literature
- Clinical baselines calibrated against DermaSensor FDA DEN230008
- TDS formula, 7-point checklist, melanoma safety gate
- ICD-10-CM code mapping, referral letter generator, explainability panel
- Practice analytics dashboard
- Interactive clickable body map SVG
- **Validation results:**
  - Anwarkh1 ViT-Base: 73.3% melanoma sensitivity on 210 HAM10000 images (`scripts/siglip-test-results.json`)
  - Hand-crafted features alone: 36.9% accuracy, 0% melanoma sensitivity
  - skintaglabs SigLIP: 30.0% melanoma sensitivity (retired)
- **actavkid model removed from HuggingFace** (HTTP 410 Gone) -- replaced by skintaglabs SigLIP, which was immediately retired after testing

### March 22, 2026 -- v0.3.0: Custom ViT Training

- Custom-trained ViT-Base on HAM10000 with focal loss (gamma=2.0, melanoma alpha=8.0)
- Model uploaded to HuggingFace as `stuartkerr/dragnes-classifier`
- **98.2% melanoma sensitivity** on HAM10000 holdout (2,004 images)
  - Source: `scripts/cross-validation-results.json`
- **98.7% on Nagabu/HAM10000** (1,000 images), **100% on marmal88 test** (1,285 images)
- **We celebrated.** We had beaten DermaSensor's 95.5% benchmark.

### March 22, 2026 -- External Validation Crash

- Tested on ISIC 2019 (4,998 images from different cameras, institutions, populations)
- **61.6% melanoma sensitivity.** A 36.6 percentage point crash.
  - Source: `scripts/isic2019-validation-results.json`
- **This was the most important test in the entire project.** The model had learned HAM10000-specific camera artifacts, not universal dermoscopic features.

### March 23, 2026 -- The FDA Audit

- Internal FDA-style audit discovered that the README claimed "91.3% cross-dataset accuracy" and "96.2% sensitivity." Neither number existed in any evidence file.
- **6 CRITICAL findings**, 5 HIGH findings. Headline numbers were fabricated -- not maliciously, but aspirational numbers presented as measured results.
- Full audit documented in `docs/FDA-AUDIT-REPORT.md`
- Immediate response: strip every unverified number, retrain overnight on combined data, rebuild evidence chain from scratch.

### March 23, 2026 -- Combined-Dataset Training (Overnight)

- Retrained on 37,484 images (HAM10000 10,015 + ISIC 2019 26,006) with oversampling of minority classes
- Focal loss gamma=2.0, melanoma alpha=6.0
- 5 epochs, 3.3 hours on Apple M3 Max MPS
- **95.97% melanoma sensitivity** on ISIC 2019 external test set (3,901 held-out images)
  - Source: `scripts/combined-training-results.json`
- The generalization gap was closed for melanoma: 61.6% -> 95.97% (+34.4pp)

### March 23, 2026 -- v0.5.0: Multi-Image Capture

- Quality-weighted consensus classification from 2-3 photos of same lesion
- Melanoma safety gate: if any single image flags melanoma >60%, signal is preserved
- **99.4% melanoma sensitivity** on multi-image HAM10000 holdout
  - Source: `scripts/multi-image-validation-results.json`

### March 24, 2026 -- ADR Blitz (13 ADRs written/updated)

- ADR-117 through ADR-129 covering every aspect of the architecture
- AUROC computed: mel AUROC 0.926 (HAM10000), 0.960 (ISIC 2019)
  - Source: `scripts/auroc-results.json`
- Threshold optimization: mel threshold 0.6204 yields 93.88% sensitivity, 85.34% specificity
  - Source: `scripts/threshold-optimization-results.json`
- Fitzpatrick equity test: **DANGEROUS 30pp melanoma sensitivity gap** (FST II: 80% vs FST V: 50%, but n=14 for FST V)
  - Source: `scripts/fitzpatrick-v2-validation.json`
- V1+V2 dual model ensemble designed and built: 0.7 V2 + 0.3 V1
  - Source: `scripts/ensemble-validation-results.json`
- Measurement pipeline: USB-C reference, skin texture FFT, LiDAR tier
- Meta-classifier: neural + clinical agreement/discordance weighting
- Bayesian risk stratification: 5-tier post-test probability system
- ONNX export: FP32 (327MB) and INT8 (85MB)
- Quality Scorecard: 77/100 (up from 70)
- Code Diagnostic Report: 72/100 code quality
- PIPELINE-EXPLAINER.md: complete 11-step pipeline documentation

### March 24-25, 2026 -- v0.9.x: Production Hardening

- **v0.9.0**: Meta-classifier + Bayesian risk stratification implemented
- **v0.9.1**: Lesion gate fix -- normal skin no longer misclassified as melanoma. Green checkmark "Your skin looks healthy!" replaces amber warning.
- **v0.9.2**: Vercel build fix (bad import caught via `vercel inspect --logs`), version badge visible in header
- **v0.9.3**: UX polish -- emerald green healthy skin result, teal version badge
- **v0.9.4** (current): Two-pass spot detection (`spot-detector.ts`), consumer translation improvements, CI workflow added

### Current State: v0.9.4 (March 25, 2026)

The app is live at https://dragnes.vercel.app. The architecture is sound, the science is real, the evidence chain is intact, but production deployment still serves the V1 model via HuggingFace Inference API. The V2 combined model (95.97%) exists as ONNX exports and local trained weights but is not yet the default production inference path.

---

## 3. Technical Journey -- From 0% to 95.97%

Each stage represents a distinct lesson. The failures are documented because they are what make the final result trustworthy.

### Stage 1: Hand-Crafted Features -- 0% Melanoma Sensitivity

Extracted 20 dermoscopic features (asymmetry, border irregularity, color count, GLCM texture, LBP structures) and fed them into a logistic regression classifier.

- **Result:** 36.9% overall accuracy, **0% melanoma sensitivity**
- **Why it failed:** 20 summary statistics cannot capture spatial patterns that distinguish melanoma from a mole. A lesion with irregular globules clustered at the periphery (suspicious) and one with globules scattered uniformly (less suspicious) produce identical feature values.
- **Lesson:** This is why the field moved from hand-crafted features to deep learning.

### Stage 2: Community ViT Model -- 73.3% Melanoma Sensitivity

Deployed Anwarkh1/Skin_Cancer-Image_Classification (ViT-Base, 85.8M parameters, 44K+ downloads on HuggingFace).

- **Result:** 55.7% overall accuracy, **73.3% melanoma sensitivity**
  - Source: `scripts/siglip-test-results.json`
- **Why it was inadequate:** No class-weighting strategy. The model optimized for overall accuracy at the expense of minority (and most dangerous) classes.
- **Lesson:** Custom training with cancer-aware loss functions is necessary.

### Stage 3: Custom Training with Focal Loss -- 98.2% Melanoma Sensitivity

Trained our own ViT-Base on HAM10000 with focal loss (Lin et al. 2017, gamma=2.0) and melanoma alpha=8.0. Model selection by melanoma sensitivity, not overall accuracy.

- **Result:** **98.2% melanoma sensitivity** on HAM10000 holdout (2,004 images, 225 melanomas)
  - Source: `scripts/cross-validation-results.json` (overfitting_check.test_results.melanoma_sensitivity = 0.9822)
- **Cross-validation:** 98.7% on Nagabu/HAM10000 (1,000 images), 100% on marmal88 test (1,285 images). Train/test gap: -0.06% (zero overfitting).
- **We celebrated.** We had beaten DermaSensor's 95.5% melanoma sensitivity benchmark.

### Stage 4: External Validation -- The Crash to 61.6%

Tested on ISIC 2019 (akinsanyaayomide/skin_cancer_dataset_balanced_labels_2) -- 4,998 images from different cameras, institutions, and patient populations.

- **Result:** **61.6% melanoma sensitivity.** Overall accuracy: 57.6%.
  - Source: `scripts/isic2019-validation-results.json` (cancer_metrics.melanoma_sensitivity = 0.6162, overall_accuracy = 0.5756)
- **Why it crashed:** The model had learned HAM10000-specific patterns -- camera artifacts, lighting conditions, institutional preprocessing -- not universal dermoscopic features. It generalized within HAM10000 variants but collapsed on truly external data.
- **This is the most important stage.** If we had not tested on external data, we would still be claiming 98.2% and it would be misleading. Most open-source skin cancer models stop at Stage 3.

### Stage 5: Combined-Dataset Training -- 95.97% on External Data

Retrained on 37,484 images from HAM10000 (10,015) + ISIC 2019 (26,006) combined, with oversampling of minority classes. Focal loss gamma=2.0, melanoma alpha=6.0. 5 epochs, 3.3 hours on Apple M3 Max MPS.

- **Result on ISIC 2019 external test set (3,901 held-out images):**
  - **95.97% melanoma sensitivity** (596/621 detected)
  - 80.0% melanoma specificity
  - Melanoma AUROC: 0.960
  - All-cancer sensitivity: 98.3% (1,848/2,058)
  - Overall accuracy: 76.4%
  - Weighted AUROC: 0.977
  - Source: `scripts/combined-training-results.json`
- **Result on HAM10000 same-distribution test (1,503 images):**
  - 97.01% melanoma sensitivity, 63.2% overall accuracy
  - Source: `scripts/combined-training-results.json`
- **The generalization gap is closed for melanoma.** 61.6% -> 95.97% = +34.4 percentage points on external data. But overall accuracy dropped from 78.1% to 76.4% because aggressive melanoma weighting over-predicts melanoma at the expense of benign classes (especially nevi). This is a deliberate tradeoff.

### Stage 6: The FDA Audit -- Catching Fabricated Claims

On March 23, 2026, an internal FDA-style audit discovered that the README previously claimed "91.3% cross-dataset accuracy" and "96.2% sensitivity." Neither number was backed by any evidence file. They had been written as aspirational targets and left as if they were measured results.

- **6 CRITICAL findings**: 91.3%, 96.2%, 0.936 AUROC, 97.3% all-cancer sensitivity all had ZERO evidence
- **Response:** Strip every unverified number, retrain overnight, rebuild evidence chain
- **Full report:** `docs/FDA-AUDIT-REPORT.md`
- This stage produced no model improvement. What it produced was **trust**.

### Stage 7: V1+V2 Ensemble -- 99.4% Melanoma Sensitivity

The combined-dataset model (V2) and HAM10000-only model (V1) fail on different inputs. When combined (0.7 V2 + 0.3 V1):

- **99.4% melanoma sensitivity** on HAM10000 holdout (1 missed melanoma out of 167)
  - Source: `scripts/ensemble-validation-results.json`
- Models trained on different data distributions develop different failure modes. Their union covers more ground than either alone.

### Stage 8: Threshold Optimization

AUROC 0.960 proved the model discriminates well. The problem was the argmax decision threshold. Per-class ROC-optimized thresholds extract "free accuracy" from the existing model without retraining.

- Mel threshold 0.6204: **93.88% sensitivity, 85.34% specificity**
  - Source: `scripts/threshold-optimization-results.json`
- NV (nevi) sensitivity improved from 14.3% to 70.4% via per-class threshold tuning
  - Source: `scripts/optimal-thresholds.json`

### Stage 9: ONNX Deployment

- FP32 model: 327MB
- **INT8 quantized: 85MB, ~155ms inference in browser, zero network dependency**
- Images never leave the device -- privacy is structural
- ConvInteger compatibility issue blocks browser ONNX Runtime Web inference directly; server-side onnxruntime-node works. The `/api/classify-local` endpoint handles server-side ONNX inference.

### Stage 10: Fitzpatrick Equity Test

Testing across skin tones revealed a **dangerous 30-percentage-point performance gap**:
- FST II melanoma sensitivity: 80%
- FST V melanoma sensitivity: 50%
- Sample sizes critically small (FST V: n=14, FST VI: n=18 total)
- Training data is approximately 95% FST I-III
- Source: `scripts/fitzpatrick-v2-validation.json`

This gap is **disclosed, not resolved**. Closing it requires diverse training data that does not yet exist in sufficient quantity.

### Stage 11: Bayesian Risk Stratification + Meta-Classifier

- Binary "melanoma yes/no" has PPV of only 8.9% at real-world 2% prevalence
- Replaced with Bayesian post-test probability across 5 risk tiers (Very High >50%, High 20-50%, Moderate 5-20%, Low 1-5%, Minimal <1%)
- Temperature-calibrated probabilities (T=1.23, ECE 0.044 after calibration)
- Age-adjusted prevalence multipliers
- Meta-classifier adjusts confidence based on agreement/disagreement between neural network and clinical feature analysis (ABCDE, TDS, 7-point)
- **NPV: 99.06%** -- negative result is highly reliable
- **NNB: 2.1** -- better than DermaSensor's 6.25
- **LR-: 0.050** -- strong clinical rule-out value
- Source: `scripts/clinical-metrics-full.json`, `scripts/calibration-results.json`

### Stage 12: Normal Skin Detection -- Two-Pass Spot Detection

- Original lesion gate was too aggressive (rejected small moles) and too permissive (classified normal skin as melanoma)
- Research documented in `docs/research/lesion-detection-approach.md`
- Implemented `spot-detector.ts`: two-pass hybrid combining LAB histogram dark-tail test (<5ms) + morphological blob detection with compactness/contrast validation (<20ms)
- Normal healthy skin shows green "Your skin looks healthy!" instead of amber warning
- The "healthy skin" result uses a `healthy_skin:` prefix to distinguish it from classification errors

---

## 4. Architecture -- The Full Pipeline

For the complete step-by-step pipeline with function names and line numbers, see `docs/PIPELINE-EXPLAINER.md`.

### High-Level Flow

```
CAMERA / UPLOAD
      |
      v
Step 1:  Image Capture           (DermCapture.svelte)
Step 2:  Image Quality Assessment (image-quality.ts) -- advisory, does not block
Step 3:  Lesion Presence Detection (spot-detector.ts + image-analysis.ts) -- SAFETY GATE
         NO LESION --> "Healthy skin" green result
         LESION    --> continue
Step 4:  Otsu Segmentation        (image-analysis.ts)
Step 5:  Feature Extraction       (image-analysis.ts: ABCDE + GLCM + LBP + k-means)
Step 6:  4-Layer Ensemble         (classifier.ts)
         Layer 1: Custom ViT v2 (70% when local, 50% when HF API)
         Layer 2: Literature logistic regression (15-30%)
         Layer 3: Rule-based safety gates (15-20%)
         Layer 4: Bayesian demographic adjustment (applied on top)
Step 7:  Threshold Application    (threshold-classifier.ts)
Step 8:  Meta-Classifier Fusion   (meta-classifier.ts)
Step 9:  Bayesian Risk Strat.     (risk-stratification.ts)
Step 10: Consumer Translation     (consumer-translation.ts)
Step 11: Display Result           (DrAgnesPanel.svelte)
```

### Classification Strategies (Priority Order)

| Strategy | Condition | Weights |
|---|---|---|
| Custom local ViT | ONNX model available | 70% ViT + 15% trained-weights + 15% rules |
| Dual HF API | Both HF models respond | 50% HF + 30% trained-weights + 20% rules |
| Single HF API | One HF model responds | 60% HF + 25% trained-weights + 15% rules |
| Offline fallback | No API, no ONNX | 60% trained-weights + 40% rules |

Safety gates always run regardless of strategy:
- **Melanoma floor:** 2+ concurrent suspicious ABCDE indicators enforce 15% melanoma probability minimum
- **TDS override:** TDS > 5.45 forces >= 30% malignant probability
- **Melanoma safety gate in multi-image:** If ANY image flags melanoma > 60%, signal survives consensus

### API Routes

| Endpoint | Purpose |
|---|---|
| `/api/classify` | HuggingFace primary model proxy (V1) |
| `/api/classify-v2` | HuggingFace secondary model proxy |
| `/api/classify-local` | Server-side ONNX inference (V2 model) |
| `/api/health` | Health check |
| `/api/analyze` | Image analysis without classification |
| `/api/feedback` | Outcome feedback recording |
| `/api/similar` | Pi-brain similarity search |

---

## 5. Key Files

### Source Code -- `src/lib/dragnes/`

| File | Lines | Purpose |
|---|---|---|
| `classifier.ts` | 973 | 4-layer ensemble orchestration, HF API calls, strategy selection |
| `image-analysis.ts` | 2,059 | **Largest file -- needs splitting.** Full CV pipeline: Otsu segmentation, asymmetry, border, color, GLCM, LBP, structures, attention maps, lesion detection |
| `consumer-translation.ts` | ~170 | Maps 7 classes to plain English with 4 risk levels (green/yellow/orange/red) |
| `meta-classifier.ts` | ~200 | Neural + clinical agreement/discordance weighting |
| `risk-stratification.ts` | ~165 | Bayesian post-test probability, 5-tier risk system, age-adjusted prevalence |
| `threshold-classifier.ts` | 106 | Per-class ROC-optimized thresholds, 3 modes (default/screening/triage) |
| `spot-detector.ts` | ~140 | Two-pass hybrid spot detection (LAB histogram + morphological blobs) |
| `multi-image.ts` | ~200 | Quality-weighted consensus with melanoma safety gate |
| `trained-weights.ts` | ~600 | Literature-derived 20x7 logistic regression weight matrix |
| `clinical-baselines.ts` | ~130 | DermaSensor benchmarks, TDS formula, 7-point checklist |
| `preprocessing.ts` | ~270 | Color normalization (Shades of Gray), hair removal, tensor conversion |
| `image-quality.ts` | ~140 | Sharpness, contrast, brightness, saturation, noise checks |
| `abcde.ts` | ~220 | ABCDE scoring types and computation |
| `types.ts` | ~150 | Core domain types: LesionClass, ClassificationResult, ABCDEScores, etc. |
| `privacy.ts` | ~360 | EXIF stripping, PII detection, differential privacy |
| `witness.ts` | ~100 | SHAKE-256 witness chain for classification provenance |
| `measurement.ts` | ~180 | 3-tier measurement pipeline integration |
| `measurement-connector.ts` | ~186 | USB-C reference object detection |
| `measurement-texture.ts` | ~120 | FFT-based dermatoglyphic pore spacing |
| `measurement-lidar.ts` | ~80 | LiDAR depth data (iPhone Pro) |
| `ensemble.ts` | ~130 | V1+V2 dual model ensemble blending |
| `ham10000-knowledge.ts` | ~330 | Bayesian demographic priors from HAM10000 |
| `icd10.ts` | ~90 | ICD-10-CM code mapping for 7 classes |
| `hf-classifier.ts` | ~90 | HuggingFace Inference API wrapper |
| `brain-client.ts` | ~450 | Pi-brain API client (not connected in production) |
| `inference-offline.ts` | ~100 | ONNX offline inference wrapper |
| `inference-orchestrator.ts` | ~75 | Orchestrates ONNX vs HF API inference selection |
| `anonymization.ts` | ~130 | Safe Harbor de-identification for pi-brain sharing |
| `fft.ts` | ~70 | 2D FFT for skin texture measurement |
| `config.ts` | ~20 | Central configuration constants |
| `index.ts` | ~125 | Barrel re-exports |

### Svelte Components -- `src/lib/components/`

| File | Lines | Purpose |
|---|---|---|
| `DrAgnesPanel.svelte` | ~2,000 | **Largest component -- needs splitting.** Main application panel with Scan/History/Learn/Settings tabs, result display, analysis orchestration |
| `DermCapture.svelte` | ~720 | Camera capture, multi-photo mode, thumbnail strip, file upload, body map |
| `AboutPage.svelte` | ~800 | About/methodology page with competitive comparison |
| `ClassificationResult.svelte` | ~500 | Medical details display (ABCDE, probabilities, model provenance) |
| `MethodologyPanel.svelte` | ~570 | Technical methodology documentation |
| `AnalyticsDashboard.svelte` | ~340 | Practice analytics: concordance, NNB, calibration, Fitzpatrick equity |
| `BodyMap.svelte` | ~220 | Interactive SVG body map for lesion location |
| `ABCDEChart.svelte` | ~110 | Visual ABCDE score display |
| `ExplainPanel.svelte` | ~120 | "Why this classification?" with literature citations |
| `CalibrationChart.svelte` | ~90 | ECE calibration visualization |
| `ReferralLetter.svelte` | ~80 | One-click referral letter with ICD-10 codes |
| `GradCamOverlay.svelte` | ~140 | Attention heatmap overlay |
| `LesionTimeline.svelte` | ~80 | Lesion change tracking over time (not fully implemented) |
| `TrendChart.svelte` | ~90 | 30-day rolling trends |
| `DiscordancePieChart.svelte` | ~100 | Model agreement/disagreement visualization |
| `Modal.svelte` | ~30 | Generic modal component |

### Training and Validation Scripts -- `scripts/`

| File | Purpose |
|---|---|
| `train-combined.py` | Combined HAM10000 + ISIC 2019 training with focal loss (produced the v2 model) |
| `train-proper.py` | HAM10000-only training with focal loss (produced the v1 model) |
| `train-fast.py` | Quick training script |
| `train-classifier.mjs` | Node.js training pipeline wrapper |
| `cross-validate.py` | Multi-dataset cross-validation on HAM10000 variants |
| `validate-isic2019.py` | ISIC 2019 external validation |
| `validate-multi-image.py` | Multi-image consensus validation |
| `validate-ensemble.py` | V1+V2 dual model ensemble validation |
| `compute-auroc.py` | AUROC computation |
| `optimize-thresholds.py` | Per-class threshold optimization from ROC curves |
| `fitzpatrick-validate-v2.py` | Fitzpatrick skin tone equity validation |
| `fitzpatrick-equity-audit.py` | Data distribution analysis by skin tone |
| `calibrate-temperature.py` | Temperature scaling calibration |
| `clinical-metrics-analysis.py` | PPV, NPV, NNB, LR+, LR- computation |
| `compute-confidence-intervals.py` | Wilson 95% CI for all metrics |
| `export-onnx-v2.py` | ONNX FP32 and INT8 export |
| `classify-image.py` | Single-image classification utility |
| `validate-models.mjs` | HAM10000 validation harness (Node.js) |
| `download-datasets.py` | Dataset download utility |

### Test Files -- `tests/`

| File | Tests | Status |
|---|---|---|
| `classifier.test.ts` | Preprocessing, ABCDE, privacy, CNN fallback | ~60 tests, mostly passing |
| `benchmark.test.ts` | Performance benchmarks | Passing |
| `clinical-baselines.test.ts` | Clinical baselines | Passing |
| `consumer-translation.test.ts` | Consumer translation | Passing |
| `icd10.test.ts` | ICD-10 mapping | Passing |
| `image-quality.test.ts` | Image quality checks | Passing |
| `lesion-gate.test.ts` | Lesion presence gate | Passing |
| `measurement.test.ts` | Measurement pipeline | Some failures |
| `multi-image.test.ts` | Multi-image consensus | Passing |
| `spot-detector.test.ts` | Two-pass spot detection | Passing |
| `threshold-classifier.test.ts` | Threshold classification | Passing |

**Run tests:** `npx vitest run`

### Configuration Files

| File | Purpose |
|---|---|
| `package.json` | v0.9.4, SvelteKit + Vite + Vitest + TailwindCSS |
| `svelte.config.js` | Dual adapter: Vercel (production) / Node (local) |
| `vite.config.ts` | Externals: @ruvector/cnn, onnxruntime-node, sharp. `__APP_VERSION__` injected from package.json |
| `dragnes.config.ts` | Domain config: class labels, privacy params, brain connection, performance budgets |
| `tailwind.config.cjs` | TailwindCSS configuration |
| `tsconfig.json` | TypeScript configuration |
| `.env.example` | Template for HF_TOKEN and model configuration |

---

## 6. Evidence Chain

Every claimed number traces to one of these JSON files. If a number does not cite an evidence file, it should not be trusted.

### Evidence Files -- `scripts/`

| File | Key Metrics | Notes |
|---|---|---|
| `combined-training-results.json` | **THE PRIMARY EVIDENCE FILE.** V2 model (combined HAM10000+ISIC 2019). External ISIC 2019: 95.97% mel sensitivity, 80.0% mel specificity, AUROC 0.960, all-cancer 98.3%, overall 76.4%. HAM10000 holdout: 97.01% mel sensitivity, 63.2% overall. | This is the file that backs the headline claims. |
| `cross-validation-results.json` | V1 model (HAM10000-only). Overfitting check: 98.22% mel sensitivity (n=225). Nagabu: 98.68%. marmal88: 100%. Train/test gap: -0.06%. | HAM10000 variants only -- NOT external validation. |
| `isic2019-validation-results.json` | V1 model on ISIC 2019 external data. 61.62% mel sensitivity, 57.56% overall accuracy, 61.76% all-cancer sensitivity. | The "crash" result that triggered combined-dataset retraining. |
| `auroc-results.json` | Per-class AUROC with CIs. HAM10000: mel AUROC 0.926, weighted 0.943. ISIC 2019: mel AUROC 0.960, weighted 0.977. | Computed from `compute-auroc.py`. |
| `threshold-optimization-results.json` | Full per-class sensitivity/specificity at each threshold. Mel optimal: 0.6204 (93.88% sens, 85.34% spec). | From `optimize-thresholds.py`. |
| `optimal-thresholds.json` | 7 per-class threshold values from ROC analysis. | Consumed by `threshold-classifier.ts`. |
| `ensemble-validation-results.json` | V1+V2 ensemble on HAM10000 holdout: 99.40% mel sensitivity, 61.98% mel specificity. 1 missed melanoma of 167. | From `validate-ensemble.py`. |
| `fitzpatrick-v2-validation.json` | Stratified equity results. 355 images (76% URL download failures). Mel gap: 30pp (FST II 80% vs FST V 50%). | Sample sizes dangerously small. |
| `fitzpatrick-equity-report.json` | Training data distribution by skin tone. Dark:light ratio for mel = 0.142. | From `fitzpatrick-equity-audit.py`. |
| `multi-image-validation-results.json` | Multi-image consensus on HAM10000 holdout (1,499 images): 99.4% mel sensitivity (single, majority, and quality-weighted). | From `validate-multi-image.py`. |
| `siglip-test-results.json` | Anwarkh1 ViT-Base: 73.33% mel sensitivity. SigLIP: 30.0% mel sensitivity. On 210 HAM10000 images. | Community model comparison. |
| `clinical-metrics-full.json` | PPV, NPV, NNB, LR+, LR-, calibration ECE, failure mode analysis. NPV 99.06%, NNB 2.1, LR- 0.050, ECE 0.044. | From `clinical-metrics-analysis.py`. |
| `calibration-results.json` | Calibration ECE before/after temperature scaling. | From `calibrate-temperature.py`. |
| `calibration-temperature.json` | Optimal temperature: T=1.23. | |
| `confidence-intervals.json` | Wilson 95% CIs for all key metrics. Mel sensitivity CI: [94.5%, 97.4%]. | From `compute-confidence-intervals.py`. |
| `onnx-v2-validation.json` | ONNX V2 model validation results. | From `export-onnx-v2.py`. |
| `validation-results.json` | Raw per-image results (210 images). No summary metrics. | Legacy validation run. |

---

## 7. ADR Summary

13 Architecture Decision Records document every significant design choice. All are in `docs/adr/`.

| ADR | Title | Status | Key Decision |
|---|---|---|---|
| 117 | DrAgnes Dermatology Intelligence Platform | IMPLEMENTED | Build DrAgnes on RuVector stack with DermLite imaging, CNN classification, pi-brain collective learning. Phase 1 foundation complete. |
| 118 | Production Validation & World-Class Medical Proof | IMPLEMENTED (Phase 1-2) | Defines complete roadmap to FDA clearance. Phase 1 (training) done. Phase 2 (Fitzpatrick) partially done (DANGEROUS gaps found). Phases 3-5 (clinical validation, publication, FDA) not started. **Contains corrections log documenting every inflated number that was fixed.** |
| 119 | Consumer Skin Screening | IMPLEMENTED | Redesign for consumer self-screening with "worry gate" architecture. consumer-translation.ts maps all 7 classes to plain English. |
| 120 | Make It Actually Work | IMPLEMENTED | Brutally honest deployment checklist. Fixed: normal skin classified as melanoma, medical jargon in results, no image quality feedback. |
| 121a | Optimized Image Capture | IMPLEMENTED | 3-tier measurement: USB-C reference (+-0.5mm), skin texture FFT (+-2-3mm), LiDAR (not started). Quality gating before classification. |
| 121b | Automated Lesion Measurement | IMPLEMENTED | Same topic, different detail level. measurement.ts integrates all tiers. |
| 122 | ONNX/WASM Offline Inference | IMPLEMENTED (export only) | ONNX FP32 (327MB) + INT8 (85MB) exported. ConvInteger blocks browser ONNX Runtime Web. Server-side onnxruntime-node works via `/api/classify-local`. Service Worker model caching not implemented. |
| 123 | Per-Class Threshold Optimization | IMPLEMENTED | Per-class ROC-optimized thresholds. threshold-classifier.ts with 3 modes. Mel threshold 0.6204 (93.88% sens, 85.34% spec). NV sensitivity 14.3% -> 70.4%. |
| 124 | Fitzpatrick Skin Tone Equity | IMPLEMENTED (tested, DANGEROUS gaps) | 30pp melanoma sensitivity gap (FST II 80% vs FST V 50%). 76% URL download failures left only 355 images. Sample sizes critically small. |
| 125 | V1+V2 Dual Model Ensemble | IMPLEMENTED (built, not wired to production) | 0.7 V2 + 0.3 V1 weighted average with melanoma safety override. ensemble.ts exists but is NOT in the production inference path. |
| 126 | Pi-Brain Collective Intelligence | IMPLEMENTED (client only, not connected) | brain-client.ts (450 lines) wraps pi-brain API. No production integration. No feedback capture. No anonymization pipeline active. |
| 127 | Production Readiness Gaps | PROPOSED | 7 specific gaps identified. Gap 1 (V2 not serving in production) is the highest-impact item. |
| 128 | Complete Clinical Validation | PROPOSED | Documents every metric an FDA reviewer would require. Separates what we have from what we lack. Identifies missing: prospective validation, subgroup analysis, calibration on held-out data, adversarial robustness. |
| 129 | Bayesian Risk Stratification | IMPLEMENTED | Replace binary classification with Bayesian post-test probability. 5-tier risk system. Age-adjusted prevalence. Temperature calibration (T=1.23, ECE 0.044). |

---

## 8. Infrastructure

### Repositories

| Repository | Purpose | URL |
|---|---|---|
| RuVector monorepo | Parent project (source of truth for development) | `/Users/stuartkerr/RuVector_New/RuVector/` |
| stuinfla/DrAgnes | Standalone GitHub repo (deployment target) | https://github.com/stuinfla/DrAgnes |
| HuggingFace model | stuartkerr/dragnes-classifier | https://huggingface.co/stuartkerr/dragnes-classifier |

### Deployment Architecture

DrAgnes lives in `examples/dragnes/` within the RuVector monorepo. Deployment to Vercel uses a **subtree split** workflow:

```bash
# From the RuVector monorepo root:
git subtree split --prefix=examples/dragnes -b deploy
git push fork deploy:main --force

# Or use the deploy script:
cd examples/dragnes
bash scripts/deploy-verified.sh patch
```

The `stuinfla/DrAgnes` GitHub repo is connected to a Vercel project. Every push to `main` on that repo triggers a Vercel build.

### Vercel Configuration

| Setting | Value |
|---|---|
| Project name | dragnes |
| Project ID | prj_FORojA8ujKmBMIH9IyAb1IFTXnCi |
| Team | stuart-kerrs-projects |
| User | sikerr-6092 |
| Domain | https://dragnes.vercel.app |
| Adapter | @sveltejs/adapter-vercel (runtime: nodejs22.x) |
| Build command | vite build |
| Framework | SvelteKit |

### CI/CD

A GitHub Actions workflow exists at `.github/workflows/ci.yml`:
- **test job:** checkout, Node 22, npm ci, npm run build, npx vitest run
- **lint job:** checkout, Node 22, npm ci, npx svelte-check

This runs on push/PR to main on the stuinfla/DrAgnes repo.

### Environment Variables

```bash
# Required for HuggingFace API fallback (not needed for ONNX offline mode):
HF_TOKEN=hf_your_token_here

# Optional model overrides:
HF_MODEL_1=stuartkerr/dragnes-classifier
```

The `.env` file is gitignored. `.env.example` is committed.

### Key Directories (gitignored)

These directories contain large binary artifacts and are NOT committed:
- `scripts/dragnes-classifier/` -- V1 trained model weights
- `scripts/dragnes-classifier-v2/` -- V2 trained model weights
- `scripts/dragnes-onnx/` -- ONNX FP32 export (327MB)
- `scripts/dragnes-onnx-int8/` -- ONNX INT8 export (85MB)
- `scripts/dragnes-onnx-v2/` -- V2 ONNX FP32
- `scripts/dragnes-onnx-v2-int8/` -- V2 ONNX INT8
- `scripts/.fitzpatrick-image-cache/` -- Downloaded Fitzpatrick17k images

---

## 9. Known Issues and Limitations

### CRITICAL

1. **V2 model not serving in production.** The deployed Vercel app still uses the V1 HuggingFace model (61.6% mel sensitivity on external data). The V2 combined model (95.97%) exists as ONNX exports and local weights but is not the default production path. **Every user today gets the WORSE model.** (ADR-127, Gap 1)

2. **Fitzpatrick equity gap: 30 percentage points.** Melanoma sensitivity drops from 80% (FST II) to 50% (FST V). Training data is ~95% FST I-III. This is a patient safety issue not addressable without more diverse training data. (ADR-124)

3. **No prospective clinical validation.** All testing is retrospective on curated datasets. Real-world clinical performance is unknown.

4. **Not FDA-cleared.** Research prototype only.

### HIGH

5. **ConvInteger compatibility blocks browser ONNX inference.** The ONNX INT8 model cannot run in ONNX Runtime Web due to unsupported ConvInteger operator. Server-side ONNX (onnxruntime-node) works. (ADR-122)

6. **No file upload validation on API endpoints.** No file size limit, no content type check, no image header validation. An attacker can upload multi-GB files causing memory exhaustion. (Code Diagnostic Report, Security section)

7. **No CORS or CSP headers.** (Code Diagnostic Report)

8. **No rate limiting on classify endpoints.** (Code Diagnostic Report)

9. **13 of ~104 tests failing** (as of last scorecard). Failing tests reduce confidence.

### MEDIUM

10. **image-analysis.ts is 2,059 lines** -- 4x the 500-line guideline. Needs splitting into segmentation, asymmetry, border, color, texture, structures, attention, classifier, and lesion-detection modules.

11. **DrAgnesPanel.svelte is ~2,000 lines** -- needs decomposition into sub-components.

12. **Duplicate implementations:** segmentLesion (preprocessing.ts vs image-analysis.ts), cosineSimilarity (classifier.ts vs multi-image.ts), morphological ops (preprocessing.ts vs image-analysis.ts).

13. **V1+V2 ensemble is built but not wired to production.** ensemble.ts exists but the classifier.ts production path does not use it.

14. **Pi-brain integration is client code only.** brain-client.ts exists but is not connected in production. No feedback capture, no live brain_share() calls.

15. **consumer-translation.ts does not update urgency when risk is upgraded** from green to yellow. User could see "yellow" with shouldSeeDoctor: false.

16. **Overall accuracy is 76.4%** on external data -- below 85% target. Aggressive melanoma weighting causes over-prediction of melanoma at the expense of benign classes (especially nevi). This is a deliberate tradeoff.

17. **ONNX INT8 model not independently validated on ISIC 2019.** The headline sensitivity numbers are from the full-precision model.

### LOW

18. **Evolution scoring ("E" in ABCDE) not implemented.** Requires comparing against previous images. Longitudinal tracking not available.

19. **Segmentation fragile on low-contrast images.** Otsu thresholding assumes bimodal histogram -- fails for amelanotic melanoma, hypo-pigmented lesions.

20. **ISIC 2019 class mapping imperfect.** SCC mapped to akiec (closest HAM10000 class).

21. **Attention heatmaps are feature saliency, not Grad-CAM.** Does not reflect true neural network attention weights.

22. **85MB ONNX model makes git pushes slow.** Use GitHub API for individual file updates when possible.

---

## 10. What's Next

### Highest-Impact Items (ordered by user impact)

1. **Wire V2 model to production.** Either:
   - Option A: Upload V2 weights to HuggingFace and update HF_MODEL_1 on Vercel (2 hours)
   - Option B: Deploy ONNX INT8 as static Vercel asset + Service Worker caching (4 hours)
   - Option A first for immediate fix, then Option B for architectural improvement.
   - (ADR-127, Gap 1)

2. **Fix the 13 failing tests + add measurement pipeline tests.** Testing score 65/100 drags overall quality down. Target: 0 failures, coverage for measurement.ts, measurement-connector.ts, measurement-texture.ts.

3. **Add CI/CD with GitHub Actions.** The ci.yml exists but needs to run reliably on every push. Add deploy step.

4. **Decompose large files.** Split image-analysis.ts (2,059 lines) into 8-9 focused modules. Split DrAgnesPanel.svelte into sub-components.

5. **Fix security gaps.** File upload validation (size, type, header bytes), CSP headers, rate limiting on classify endpoints.

### Medium-Term (weeks)

6. **Fitzpatrick equity improvement.** Source diverse training data from ISIC archive and Diverse Dermatology Images dataset. Retrain with balanced representation. Target: <10pp gap between FST groups.

7. **Wire V1+V2 ensemble to production.** ensemble.ts exists; connect it to the inference pipeline.

8. **Connect pi-brain in production.** Wire brain-client.ts into the classification flow with anonymization and feedback capture.

9. **Resolve ConvInteger ONNX compatibility.** Either use a different quantization approach that avoids ConvInteger, or wait for ONNX Runtime Web to support it.

### Long-Term (months)

10. **Prospective clinical validation.** Partner with dermatology clinics for real-world testing.

11. **FDA regulatory pathway.** De Novo or 510(k) with DermaSensor (DEN230008) as predicate device.

12. **Peer-reviewed publication.** Submit training methodology and validation results.

13. **Improve overall accuracy.** Curriculum learning or multi-objective optimization to improve benign-class accuracy without sacrificing melanoma detection.

---

## Appendix: Version History

| Version | Date | Key Changes |
|---|---|---|
| 0.1.0 | 2026-03-21 | Initial SvelteKit prototype, camera capture, demo classifier |
| 0.2.0 | 2026-03-22 | Real CV engine (1,890 lines), dual-model ViT ensemble, literature logistic regression, clinical baselines, analytics dashboard |
| 0.3.0 | 2026-03-22 | Custom ViT training (98.2% on HAM10000), HuggingFace deployment |
| 0.5.0 | 2026-03-23 | Multi-image capture, combined-dataset training (95.97% external), FDA audit corrections |
| 0.9.0 | 2026-03-25 | Meta-classifier + Bayesian risk stratification |
| 0.9.1 | 2026-03-25 | Lesion gate fix -- normal skin shows green "healthy" |
| 0.9.2 | 2026-03-25 | Vercel build fix, version badge visible |
| 0.9.3 | 2026-03-25 | UX polish: emerald healthy skin, teal version badge |
| 0.9.4 | 2026-03-25 | Two-pass spot detection, consumer translation improvements, CI workflow |

---

## How to Use This Document

**Starting a new session:** Read this document first. It tells you where things are, what works, what is broken, and what the priorities are.

**Checking a claim:** Every number has a Source citation pointing to a JSON file in `scripts/`. Read the JSON to verify.

**Before changing anything:** Check ADR-127 (production readiness gaps) and the Known Issues section above. The highest-impact work is wiring V2 to production.

**Before deploying:** Use `vercel ls` to verify the build succeeded. Use `vercel inspect <url> --logs` if it fails. The 85MB ONNX model makes git pushes slow -- use the subtree split workflow or GitHub API for individual files.
