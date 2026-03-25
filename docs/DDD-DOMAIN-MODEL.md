Updated: 2026-03-25 | Version 1.0.0
Created: 2026-03-25

# Mela Domain-Driven Design Model

## Purpose

This document defines the bounded contexts, aggregates, domain events, ubiquitous language, and context map for Mela. It serves as the blueprint for:
- V2 model wiring (which context owns inference?)
- File decomposition (image-analysis.ts → 9 modules)
- Test organization (test per bounded context)
- Future feature placement

---

## Ubiquitous Language

These terms have precise meanings in this project. Use them consistently in code, docs, and conversation.

| Term | Definition |
|---|---|
| **Lesion** | A distinct skin abnormality detected by the spot detector |
| **Healthy skin** | Image with no detected lesion — routes to green UI, not the classifier |
| **Lesion gate** | Safety check: only images with detected lesions enter the classification pipeline |
| **ABCDE** | Asymmetry, Border, Color, Diameter, Evolution — the 5 clinical features |
| **TDS** | Total Dermoscopy Score: A×1.3 + B×0.1 + C×0.5 + D×1.5. >5.45 = malignant override |
| **Ensemble** | Weighted combination of multiple classification strategies |
| **Strategy** | One of 4 classification paths: custom ViT, dual HF, single HF, offline |
| **Safety gate** | Hard rule that overrides model output (melanoma floor, TDS override) |
| **Melanoma floor** | If 2+ ABCDE criteria suspicious → melanoma probability >= 15% |
| **Post-test probability** | Bayesian: what the real-world chance of disease is after our test result |
| **Consumer result** | Plain-English translation of medical classification |
| **Risk level** | One of: very-high, high, moderate, low, minimal (Bayesian) |
| **Consumer risk** | One of: red, orange, yellow, green (UI display) |

---

## Bounded Contexts

### 1. Image Acquisition

**Responsibility:** Capture or receive an image, validate quality, strip metadata.

**Current code:** `MelaPanel.svelte` (capture logic), `DermCapture.svelte`

**Aggregates:**
- `CapturedImage` — imageData, bodyLocation, capturedAt, magnification

**Domain Events:**
- `ImageCaptured` — raw image available for processing
- `QualityCheckPassed` — image meets minimum requirements

**Key rules:**
- EXIF must be stripped before any processing (privacy)
- Image must be at least 224×224 pixels
- Body location is required for demographic adjustment

---

### 2. Lesion Detection

**Responsibility:** Determine if a skin lesion is present. Gate entry to the classification pipeline.

**Current code:** `spot-detector.ts` (155 LOC), `detectLesionPresence()` in `image-analysis.ts`

**Aggregates:**
- `LesionPresence` — hasLesion, confidence, reason

**Domain Events:**
- `LesionDetected` → proceed to Feature Extraction
- `HealthySkinDetected` → route to green "Healthy" UI (NOT the classifier)

**Key rules:**
- Two-pass detection: (1) LAB histogram dark-tail fast reject, (2) morphology + compactness validation
- Images without lesions MUST NOT reach the classifier (it would assign spurious melanoma risk)
- Healthy skin shows green checkmark, NOT amber warning (v0.1–v0.9.0 bug)

---

### 3. Feature Extraction

**Responsibility:** Compute all computer vision features from the lesion image.

**Current code:** `image-analysis.ts` (2,001 LOC) — needs splitting into 9 modules

**Aggregates:**
- `Segmentation` — mask, bbox, area, perimeter (LAB + Otsu + morphology)
- `AsymmetryScore` — 0–2, principal-axis folding
- `BorderScore` — 0–8, octant irregularity + color gradient
- `ColorAnalysis` — k-means clustering, 6 dermoscopic colors, blue-white structures
- `TextureFeatures` — GLCM contrast, homogeneity, entropy, correlation
- `StructuralPatterns` — irregular network (LBP), globules, streaks, blue-white veil, regression
- `AttentionMap` — 224×224 saliency heatmap
- `LesionMeasurement` — diameter in mm, pixels-per-mm

**Domain Events:**
- `SegmentationComplete` — mask and bbox available
- `FeatureExtractionComplete` — all ABCDE + texture + structure scores computed

**Proposed module decomposition:**

| Module | Lines (est) | Exports |
|---|---|---|
| `segmentation.ts` | 150 | `segmentLesion()`, morphology helpers |
| `asymmetry.ts` | 130 | `measureAsymmetry()` |
| `border.ts` | 130 | `analyzeBorder()` |
| `color-analysis.ts` | 230 | `analyzeColors()`, `kMeansLab()` |
| `texture-glcm.ts` | 120 | `analyzeTexture()` |
| `structures.ts` | 280 | `detectStructures()` + 5 sub-detectors |
| `attention-map.ts` | 230 | `generateAttentionMap()` |
| `feature-classifier.ts` | 360 | `classifyFromFeatures()` (TDS, safety gates, softmax) |
| `measurement.ts` | 80 | `estimateDiameterMm()`, `computeRiskLevel()` |

**Known duplicates to resolve:**
- `morphDilate/morphErode` — identical copies in image-analysis.ts AND preprocessing.ts
- `segmentLesion` — different implementations in image-analysis.ts (LAB) and preprocessing.ts (grayscale)
- `cosineSimilarity` — Float64 in classifier.ts, Float32 in multi-image.ts

---

### 4. Classification (Ensemble)

**Responsibility:** Combine multiple model outputs into a single classification result.

**Current code:** `classifier.ts` (973 LOC), `ensemble.ts` (152 LOC), `inference-orchestrator.ts` (101 LOC)

**Aggregates:**
- `ClassificationResult` — topClass, confidence, probabilities[7], modelId, inferenceTimeMs, strategy used
- `EnsembleResult` — V1+V2 weighted output, agreement flag, disagreement warning

**Domain Events:**
- `ClassificationComplete` — probabilities available for all 7 classes
- `ModelDisagreement` — V1 and V2 predict different top classes

**Strategy selection (priority order):**
1. Custom local ViT (ONNX): 70% ViT + 15% trained-weights + 15% rules
2. Dual HF API: 50% HF + 30% trained-weights + 20% rules
3. Single HF API: 60% HF + 25% trained-weights + 15% rules
4. Offline fallback: 60% trained-weights + 40% rules

**V1+V2 ensemble weights:** V1=0.3, V2=0.7 with melanoma safety override (max, not average)

**Key rules:**
- V2 model should be primary (Gap 1 — currently NOT wired to production)
- Melanoma safety: if either model > 0.15 mel → take max(v1_mel, v2_mel)

---

### 5. Clinical Assessment

**Responsibility:** Apply clinical heuristics, threshold calibration, and Bayesian risk on top of raw classification.

**Current code:** `threshold-classifier.ts` (106 LOC), `meta-classifier.ts` (251 LOC), `risk-stratification.ts` (193 LOC)

**Aggregates:**
- `ThresholdResult` — class after per-class ROC-optimized boundaries (3 modes: default, screening, triage)
- `MetaClassification` — neural+clinical fusion, agreement scoring (concordant/discordant/neutral)
- `RiskAssessment` — post-test probability, risk level, headline, action, urgency, color

**Domain Events:**
- `ThresholdsApplied` — raw probabilities converted to calibrated decisions
- `ClinicalFusionComplete` — neural and ABCDE/TDS signals combined
- `RiskAssessed` — Bayesian post-test probability computed

**Key rules:**
- Melanoma floor: 2+ suspicious ABCDE → mel >= 15%
- TDS override: TDS > 5.45 → malignant >= 30%
- Age-adjusted prevalence: under 30 = 0.3x, 30-50 = 0.7x, 50-70 = 1.5x, over 70 = 2.5x
- Temperature scaling T=1.23 for calibration (ECE 0.078 → 0.044)

**Bayesian risk levels:**

| Level | Post-test prob | Action |
|---|---|---|
| Very high | >50% | See dermatologist within 1 week |
| High | 20–50% | See dermatologist within 2 weeks |
| Moderate | 5–20% | Photograph monthly |
| Low | 1–5% | Routine skin checks |
| Minimal | <1% | Normal schedule |

---

### 6. Consumer Translation

**Responsibility:** Convert medical classification into plain English with clear actions.

**Current code:** `consumer-translation.ts` (243 LOC)

**Aggregates:**
- `ConsumerResult` — headline, riskLevel (green/yellow/orange/red), explanation, action, shouldSeeDoctor, urgency, medicalTerm

**Domain Events:**
- `TranslationComplete` — user-facing result ready for display

**Key rules:**
- Confidence < 0.40 → "Unable to classify" (yellow, no urgency)
- Uses static translations per class + Bayesian override when demographics available
- Red = urgent (melanoma), Orange = soon (BCC), Yellow = routine (akiec), Green = monitor (nv, bkl, df, vasc)

---

### 7. Collective Intelligence

**Responsibility:** Anonymized case sharing, similar case retrieval, literature search via pi-brain.

**Current code:** `brain-client.ts`, `/api/analyze/+server.ts` (124 LOC)

**Aggregates:**
- `AnonymizedCase` — classification + demographics + ABCDE, no images
- `SimilarCaseMatch` — lesion class, confidence, outcome, literature references

**Domain Events:**
- `CaseShared` — anonymized case sent to pi-brain
- `SimilarCasesFound` — HNSW vector search returned matches

**Key rules:**
- No raw images leave the device (privacy)
- Anonymization strips PII, retains clinical features only
- Rate limited: 100 req/min per IP on /api/analyze

---

## Context Map

```
                    ┌─────────────────────┐
                    │  IMAGE ACQUISITION  │
                    │  (Capture + Validate)│
                    └──────────┬──────────┘
                               │ ImageCaptured
                               ▼
                    ┌─────────────────────┐
                    │  LESION DETECTION   │
                    │  (Spot Detector)    │
                    └──┬──────────────┬───┘
               LesionDetected    HealthySkinDetected
                       │                │
                       ▼                ▼
            ┌──────────────────┐   Green "Healthy" UI
            │ FEATURE          │   (no classification)
            │ EXTRACTION       │
            │ (9 CV modules)   │
            └────────┬─────────┘
                     │ FeatureExtractionComplete
                     ▼
            ┌──────────────────┐
            │ CLASSIFICATION   │──── Strategy 1-4 ────┐
            │ (Ensemble)       │                       │
            │ V1+V2 + trained  │◄── ensemble.ts ──────┘
            └────────┬─────────┘
                     │ ClassificationComplete
                     ▼
            ┌──────────────────┐
            │ CLINICAL         │
            │ ASSESSMENT       │
            │ Threshold →      │
            │ Meta-Classifier →│
            │ Bayesian Risk    │
            └────────┬─────────┘
                     │ RiskAssessed
                     ▼
            ┌──────────────────┐      ┌──────────────────┐
            │ CONSUMER         │      │ COLLECTIVE       │
            │ TRANSLATION      │      │ INTELLIGENCE     │
            │ (Plain English)  │      │ (Pi Brain)       │
            └────────┬─────────┘      └────────┬─────────┘
                     │                          │
                     └──────────┬───────────────┘
                                ▼
                         UI Display
                    (MelaPanel.svelte)
```

**Communication patterns:**
- Image Acquisition → Lesion Detection: **direct call** (synchronous)
- Lesion Detection → Feature Extraction: **domain event** (LesionDetected)
- Feature Extraction → Classification: **direct call** (features passed as parameter)
- Classification → Clinical Assessment: **pipeline** (probabilities flow through threshold → meta → risk)
- Clinical Assessment → Consumer Translation: **direct call** (risk assessment → plain English)
- Classification → Collective Intelligence: **async fire-and-forget** (non-blocking)

---

## Implementation Plan

### Phase 1: V2 Model Wiring (COMPLETED 2026-03-25)
1. ~~Wire V2 ONNX as primary inference~~ -- Done, HuggingFace removed
2. ~~3-layer ensemble: 70% ONNX + 15% trained-weights + 15% rules~~ -- Done
3. ~~Service worker caching for offline~~ -- Done

### Phase 2: Clinical Safety (ADR-130, Dr. Chang Feedback)
4. Verify healthy skin gate across diverse inputs (freckles, scars, clear skin)
5. Low-confidence default: confidence < 0.40 -> "See a dermatologist to be safe"
6. Photo capture guidance (lighting, distance, lens, single lesion)
7. Clinical history questions (new? changing? biopsied? family history? symptoms?)
8. Fix referral letter (remove provider signature, add AI disclaimer)
9. Pediatric disclaimer (age < 18 warning)
10. Fitzpatrick equity disclaimer (FST IV-VI warning)
11. Multiple lesion guidance (crop instructions, multi-lesion detection)
12. Amelanotic melanoma warning (low-contrast segmentation failure path)

### Phase 3: Accuracy Validation
13. ONNX INT8 validation on full ISIC 2019 test set (3,901 images)
14. Test across resolutions, formats, device types (iPhone vs Android)
15. Per-category stats: sensitivity, specificity, AUROC per lesion class
16. Phone camera validation with real-world photos

### Phase 4: Model Improvements
17. SCC as 8th classification category (retraining required)
18. Acral/subungual melanoma detection (nail images)
19. Tattoo/scar/piercing detection and warning
20. Evolution tracking (ABCDE "E" -- side-by-side comparison over time)

### Phase 5: File Decomposition
21. Split `image-analysis.ts` into 9 CV modules
22. Split `MelaPanel.svelte` into 9 UI components
23. Create state stores

### Phase 6: Security Hardening
24. Upload validation (file size, MIME, image headers)
25. CSP/CORS headers
26. Fix `any` type on ortSession
27. EXIF stripping on all paths

---

## Aggregate Lifecycle

```
CapturedImage ──[validate]──► LesionPresence
                                    │
                          ┌─────────┴──────────┐
                     [no lesion]          [lesion found]
                          │                     │
                   HealthySkin            SegmentationResult
                   (green UI)                   │
                                          ABCDEFeatures
                                                │
                                        ClassificationResult
                                                │
                                         MetaClassification
                                                │
                                          RiskAssessment
                                                │
                                         ConsumerResult
                                                │
                                          DiagnosisRecord
                                          (persisted)
```

---

## ADR Mapping

| ADR | Bounded Context | Status |
|---|---|---|
| ADR-117 | All (platform definition) | Accepted |
| ADR-119 | Consumer Translation | Accepted |
| ADR-120 | All (make it work) | Accepted |
| ADR-121 | Feature Extraction (measurement) | Implemented |
| ADR-122 | Classification (ONNX offline) | Implemented |
| ADR-123 | Clinical Assessment (thresholds) | Implemented |
| ADR-124 | Feature Extraction (Fitzpatrick equity) | Validated (30pp gap — DANGEROUS) |
| ADR-125 | Classification (V1+V2 ensemble) | Implemented (not wired to production) |
| ADR-126 | Collective Intelligence (Pi Brain) | Implemented |
| ADR-127 | Classification (production gaps) | PROPOSED — Gap 1 is top priority |
| ADR-128 | Clinical Assessment (validation) | Implemented |
| ADR-129 | Clinical Assessment (Bayesian risk) | Implemented |
