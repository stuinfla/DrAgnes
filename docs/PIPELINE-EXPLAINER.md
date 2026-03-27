Updated: 2026-03-26 | Version 1.0.0
Created: 2026-03-25

# Mela Image Processing Pipeline -- Complete Technical Explainer

> **Disclaimer:** This document describes the technical architecture of Mela's AI pattern analysis pipeline. Mela is an educational tool, not a medical device. It does not diagnose, screen for, or detect any disease.

This document traces every step a photograph takes from the moment it is captured
to the moment a consumer-friendly result is displayed. Every function, threshold,
and magic number is referenced by file and line number so a new engineer can
find the code instantly.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step 1: Image Capture](#step-1-image-capture)
3. [Step 2: Image Quality Assessment](#step-2-image-quality-assessment)
4. [Step 3: Lesion Presence Detection (Safety Gate)](#step-3-lesion-presence-detection)
5. [Step 4: Otsu Segmentation](#step-4-otsu-segmentation)
6. [Step 5: Feature Extraction (ABCDE + Structures)](#step-5-feature-extraction)
7. [Step 6: Neural Network Classification (4-Layer Ensemble)](#step-6-neural-network-classification)
8. [Step 7: Threshold Application](#step-7-threshold-application)
9. [Step 8: Meta-Classifier Fusion](#step-8-meta-classifier-fusion)
10. [Step 9: Bayesian Risk Stratification](#step-9-bayesian-risk-stratification)
11. [Step 10: Consumer Translation](#step-10-consumer-translation)
12. [Step 11: Display Result](#step-11-display-result)
13. [Multi-Image Consensus Path](#multi-image-consensus-path)
14. [Lesion Measurement Subsystem](#lesion-measurement-subsystem)
15. [Known Failure Modes](#known-failure-modes)
16. [Where to Improve](#where-to-improve)

---

## Pipeline Overview

```
 CAMERA / UPLOAD
       |
       v
 +---------------------+
 | Step 1: Capture      |  DermCapture.svelte
 | (ImageData + meta)   |
 +---------------------+
       |
       v
 +---------------------+
 | Step 2: Quality      |  image-quality.ts
 | (sharpness, glare..) |
 +---------------------+
       |
       v
 +---------------------+     +-- NO LESION ---> "Healthy skin" result
 | Step 3: Lesion Gate  |-----|
 | (5-layer gate)       |     +-- LESION ------> continue
 +---------------------+
       |
       v
 +---------------------+
 | Step 4: Otsu Seg.    |  image-analysis.ts:segmentLesion
 | (LAB L-channel)      |
 +---------------------+
       |
       v
 +---------------------+
 | Step 5: Features     |  image-analysis.ts
 | A B C D E + structs  |
 +---------------------+
       |
       v
 +---------------------+
 | Step 6: Classifier   |  classifier.ts
 | 4-layer ensemble     |
 |  Custom ViT (70%)    |
 |  HF dual ViT (50%)   |
 |  Trained weights(25%) |
 |  Rule-based (15%)    |
 +---------------------+
       |
       v
 +---------------------+
 | Step 7: Thresholds   |  threshold-classifier.ts
 | Per-class ROC cutoffs|
 +---------------------+
       |
       v
 +---------------------+
 | Step 8: Meta-Classif.|  meta-classifier.ts
 | Neural + clinical    |
 | agreement/discordance|
 +---------------------+
       |
       v
 +---------------------+
 | Step 9: Bayesian     |  risk-stratification.ts
 | Post-test probability|
 | with prevalence prior|
 +---------------------+
       |
       v
 +---------------------+
 | Step 10: Translation |  consumer-translation.ts
 | Plain English result |
 +---------------------+
       |
       v
 +---------------------+
 | Step 11: Display     |  MelaPanel.svelte
 | Color, headline, act.|
 +---------------------+
```

---

## Step 1: Image Capture

**What happens**: The user takes a photo or uploads an image. The raw pixels are
extracted as an `ImageData` object (RGBA, uncompressed). Metadata about body
location, device model, and image type (dermoscopy vs. clinical) is collected.

**Where it lives**: `src/lib/components/DermCapture.svelte`

**Inputs**: Camera stream or file upload

**Outputs**: A `CaptureEvent` object:
```typescript
{
  imageData: ImageData;       // Raw RGBA pixel array
  bodyLocation: BodyLocation; // "head" | "trunk" | "upper_extremity" | ...
  deviceModel: string;        // "phone_only" | "HUD" | "DL5" | ...
  imageType: "dermoscopy" | "clinical";
}
```

**Key functions**:

| Function | Line | Purpose |
|----------|------|---------|
| `captureFrame()` | :268 | Draws video frame to canvas, extracts ImageData |
| `handleFileUpload()` | :305 | Reads uploaded file, draws to canvas, extracts ImageData |
| `detectImageType()` | :90 | Auto-detects dermoscopy vs. clinical from pixel patterns |
| `startCamera()` | :163 | Opens getUserMedia with environment-facing camera |

**Image type auto-detection algorithm** (`detectImageType`, line 90-141):

The system scores dermoscopy likelihood from 0-7+:

| Signal | Score | What it checks |
|--------|-------|---------------|
| Dark corners (>30% corner pixels below brightness 30) | +2 | Dermoscope vignette |
| Bright center (>50% center pixels above brightness 80) | +1 | Contact illumination |
| High average saturation (>0.35) | +1 | Polarized light colors |
| Non-phone device selected (HUD, DL5, etc.) | +3 | User said they have a scope |

Score >= 3 = "dermoscopy", otherwise "clinical".

**Camera constraints** (line 151-161):
- Dermoscope attached: 1920x1080 (higher res for detail)
- Phone only: 1280x720 (faster processing)
- Always rear camera (`facingMode: "environment"`)

**What can go wrong**:
- Camera denied/unavailable on iOS Safari (handled with file upload fallback, line 236-248)
- Zero-dimension video frames on iPhone (guarded at line 218-230)
- getUserMedia completely unsupported (detected at line 178)

**Example values**:
- Typical phone photo: 1280x720 ImageData, ~3.7M pixels, "clinical" type
- DermLite attached: 1920x1080 ImageData, ~8.3M pixels, "dermoscopy" type

---

## Step 2: Image Quality Assessment

**What happens**: Five quality checks run on the raw ImageData. The user gets
immediate feedback (good / acceptable / poor) before classification begins.
This step does NOT block classification -- it is advisory only.

**Where it lives**: `src/lib/mela/image-quality.ts`

**Input**: `ImageData` (raw RGBA pixels)

**Output**: `ImageQualityResult`:
```typescript
{
  grade: "good" | "acceptable" | "poor";
  overallScore: number;      // 0-1 average of all check scores
  checks: QualityCheck[];    // 5 individual checks
  suggestion: string | null; // Worst failure message shown to user
}
```

**The five checks** (all sample every 3rd pixel for speed -- line 29: `const S = 3`):

### 2a. Sharpness (line 38-53)

**Algorithm**: Laplacian variance. For each sampled pixel, compute the discrete
Laplacian (sum of 4 neighbors minus 4x center). The variance of this signal
measures edge sharpness.

```
score = min(variance / 0.04, 1)
```

**Thresholds**:
- `SHARP_POOR = 0.15` -- below this: "Image is blurry"
- `SHARP_OK = 0.30` -- below this: "Slightly soft"

### 2b. Brightness (line 55-66)

**Algorithm**: Mean luminance of the center 50% of the image (center quarter
by area: x0=25%, x1=75%, y0=25%, y1=75%).

```
score = 1 - 2 * |mean - 0.5|   (peaks at 0.5 = ideal)
```

**Thresholds**:
- `BRIGHT_LOW = 50/255 (~0.196)` -- too dark
- `BRIGHT_HIGH = 220/255 (~0.863)` -- too bright

### 2c. Contrast (line 68-81)

**Algorithm**: RMS contrast of the center quarter. Standard deviation of
luminance values, normalized by dividing by 0.25.

**Threshold**: `CONTRAST_POOR = 0.08` -- below this: "Image looks washed out"

### 2d. Framing (line 83-96)

**Algorithm**: Difference between center and periphery mean luminance,
divided by image dimensions. A lesion centered in frame creates a luminance
difference.

**Threshold**: `diff > 0.03` passes (line 94). If the center and edges
have the same brightness, the spot is not centered.

### 2e. Glare (line 98-111)

**Algorithm**: Count pixels in the center quarter where all three channels
exceed 245 (near-white). If more than 5% are glare pixels, fail.

**Thresholds**:
- `GLARE_THRESH = 245` (per channel)
- `GLARE_PCT = 0.05` (5% of center pixels)

### Grade assignment (line 129-132):

```
good       = 0 failures
poor       = any critical failure OR >2 failures
acceptable = everything else
```

A critical failure is sharpness or brightness scoring below `SHARP_POOR` (0.15).

**Example values**:

| Scenario | Sharpness | Brightness | Contrast | Framing | Glare | Grade |
|----------|-----------|------------|----------|---------|-------|-------|
| Good dermoscopy | 0.72 | 0.85 | 0.64 | 0.45 | 0.98 | good |
| Slightly blurry | 0.22 | 0.79 | 0.51 | 0.38 | 0.95 | acceptable |
| Dark, blurry | 0.08 | 0.31 | 0.12 | 0.09 | 1.00 | poor |
| Direct flash | 0.65 | 0.92 | 0.45 | 0.30 | 0.21 | poor |

**Known weakness**: Quality checks use simple statistical measures. A photo
of a blank wall will score "good" on all checks except framing.

---

## Step 3: Lesion Presence Detection (Safety Gate)

**What happens**: Before any classification runs, the image is tested for
whether it actually contains a skin lesion. This is the most critical safety
gate in the entire pipeline. Without it, the classifier would label normal
skin as melanoma (because the model was trained exclusively on lesion images).

**Where it lives**: `src/lib/mela/image-analysis.ts:detectLesionPresence` (line 1990)

**Input**: `ImageData`

**Output**: `LesionPresenceResult`:
```typescript
{
  hasLesion: boolean;
  confidence: number; // 0-1
  reason: string;     // Human explanation of why it passed or failed
}
```

**The five-layer gate** (processed in order, first failure exits):

### Gate 1: Overall Color Uniformity (line 2017-2024)

Computes the combined RGB standard deviation across the entire image
(sampling every 3rd pixel).

```
overallVariance = sqrt(rVar + gVar + bVar)
```

**Threshold**: `overallVariance < 25` -> reject as "uniform skin with no distinct spot"

**Why 25**: Normal skin of a single Fitzpatrick type has RGB variance around
10-20. A mole adds at least 30-40 units of variance from the color contrast.

### Gate 2: Center vs. Periphery Luminance (line 2026-2066)

Measures the luminance difference between the center circle (inner 15% radius)
and the surrounding annulus (15-40% radius).

**Threshold**: `lumDiff < 12` -> reject as "skin appears uniform"

**Why 12**: A visible mole creates at least a 15-30 unit luminance
difference. Shadows typically create 5-10 units.

### Gate 3: Segmentation Area Ratio (line 2068-2081)

Runs full Otsu segmentation (Step 4) and checks the resulting lesion area
as a fraction of total image area.

**Threshold**: `areaRatio < 0.01 OR areaRatio > 0.70` -> reject

- Below 1%: no visible spot (likely just sensor noise triggering segmentation)
- Above 70%: entire image is "lesion" (likely a close-up of skin without boundaries)

### Gate 4: Color Contrast (line 2083-2117)

Computes the Euclidean RGB distance between the mean lesion color and the
mean surrounding skin color (using the segmentation mask).

```
colorDiff = sqrt((lesionR - skinR)^2 + (lesionG - skinG)^2 + (lesionB - skinB)^2)
```

**Threshold**: `colorDiff < 35` -> reject

**Why 35**: Raised from the original value of 15 to reject shadows and veins
that have mild color differences. A real mole has colorDiff of 40-150+.
This is the threshold that caused the "too permissive" vs. "too aggressive"
gate tension (see Known Failure Modes).

### Gate 5: Compactness (line 2119-2129)

Checks that the segmented shape is roughly round/oval, not a linear streak.

```
compactness = expectedPerimeter / actualPerimeter
            = 2 * sqrt(pi * area) / perimeter
```

**Threshold**: `compactness < 0.25` -> reject as "shadow or skin fold"

A perfect circle has compactness = 1.0. Real moles range 0.4-0.9. Shadows
and folds score 0.1-0.2.

### Final Confidence Scoring (line 2131-2144)

If all gates pass, a composite confidence is computed:

| Condition | Points |
|-----------|--------|
| Area ratio 3-50% | +0.30 |
| Area ratio outside that range | +0.10 |
| Compactness > 0.4 | +0.30 |
| Compactness <= 0.4 | +0.15 |
| Color diff > 50 | +0.40 |
| Color diff 35-50 | +0.20 |

**Final gate**: `confidence > 0.5` -> hasLesion = true

**Example values**:

| Scenario | Variance | LumDiff | AreaRatio | ColorDiff | Compactness | Result |
|----------|----------|---------|-----------|-----------|-------------|--------|
| Back of hand, no moles | 14 | -- | -- | -- | -- | GATE 1: reject |
| Arm with small mole | 42 | 28 | 0.04 | 65 | 0.72 | PASS (conf 0.90) |
| Large melanoma | 68 | 45 | 0.18 | 120 | 0.55 | PASS (conf 1.00) |
| Vein on wrist | 31 | 8 | -- | -- | -- | GATE 2: reject |
| Dermoscopy of small nevus | 38 | 14 | 0.02 | 38 | 0.65 | PASS (conf 0.65) |
| Shadow on skin | 28 | 15 | 0.12 | 22 | 0.18 | GATE 4: reject |

**What the UI shows on rejection**: When `classificationError` starts with
`"healthy_skin:"`, `MelaPanel.svelte` renders a reassuring "Healthy skin"
card instead of an error (line 659-662).

**Known weaknesses**:
- Small, pale moles (Fitzpatrick I-II) can fail Gate 4 because the color
  contrast between a light-brown mole and fair skin is below 35
- The 15% inner radius is tuned for centered framing; off-center lesions
  may fail Gate 2
- Very dark skin (Fitzpatrick V-VI) can have high variance from specular
  highlights, potentially passing Gate 1 when no lesion is present

---

## Step 4: Otsu Segmentation

**What happens**: The lesion is separated from surrounding skin using
automatic thresholding in LAB color space, followed by morphological cleanup.

**Where it lives**: `src/lib/mela/image-analysis.ts:segmentLesion` (line 275)

**Input**: `ImageData`

**Output**: `SegmentationResult`:
```typescript
{
  mask: Uint8Array;  // Binary mask (0=skin, 1=lesion), one byte per pixel
  bbox: BBox;        // Bounding box {x, y, w, h}
  area: number;      // Lesion area in pixels
  perimeter: number; // Border length in pixels
}
```

**Algorithm**:

1. **RGB to LAB conversion** (line 65-93): Each pixel is converted from sRGB
   to CIELAB color space. Only the L (lightness) channel is used for
   segmentation because skin lesions are primarily defined by luminance
   differences, and LAB separates luminance from chrominance.

2. **L-channel extraction** (line 280-286): The L value [0,100] is scaled
   to [0,255] for Otsu: `L * 2.55`

3. **Otsu thresholding** (line 113-146): The classic Otsu algorithm finds
   the threshold that maximizes inter-class variance between foreground
   and background. This is a 256-bin histogram search.

4. **Binary mask** (line 292-295): Pixels with L <= threshold become 1
   (lesion = darker than background).

5. **Morphological cleanup**:
   - **Close** (radius 3, line 298): Dilate then erode to fill small gaps
     inside the lesion
   - **Open** (radius 2, line 299): Erode then dilate to remove small
     noise specks outside the lesion

6. **Largest connected component** (line 152-207): BFS flood fill labels
   all connected foreground regions. Only the largest is kept. This
   eliminates secondary artifacts (hair, ruler marks, etc.).

7. **Bounding box + metrics** (line 304-335): Single pass over the mask
   computes area, perimeter (pixels with at least one background neighbor),
   and bounding box.

8. **Fallback** (line 338-397): If the segmented area is less than 1% of
   the image, the algorithm assumes segmentation failed and places a
   centered ellipse (35% radius) as the mask. This happens when:
   - The lesion and skin have similar luminance
   - The image is heavily overexposed
   - The image contains a very pale lesion on pale skin

**Example values**:

| Scenario | Otsu Threshold | Area Ratio | Perimeter |
|----------|---------------|------------|-----------|
| Dark melanoma on fair skin | ~95 | 0.12 | ~2,400 px |
| Light nevus on fair skin | ~140 | 0.06 | ~800 px |
| Nevus on dark skin | ~65 | 0.08 | ~1,100 px |
| Fallback (failed seg) | -- | 0.38 (ellipse) | ~3,500 px |

**Known weakness**: Otsu assumes a bimodal histogram (two distinct peaks for
lesion and skin). Images with hair, rulers, or multiple moles in frame can
create a third peak that confuses the threshold.

---

## Step 5: Feature Extraction

After segmentation, six feature extractors run in sequence. Their outputs
feed both the rule-based classifier and the clinical scoring systems
(TDS, 7-point checklist, ABCDE).

### 5a. Asymmetry -- `measureAsymmetry()` (line 413-470)

**Algorithm**:
1. Compute the lesion centroid (center of mass of mask pixels)
2. Compute second-order central moments (Mxx, Myy, Mxy)
3. Find the principal axis angle: `theta = 0.5 * atan2(2*Mxy, Mxx-Myy)`
4. Fold the mask along the principal axis and its perpendicular
5. For each fold: count the fraction of non-overlapping pixels when one half
   is mirrored onto the other (`measureAxisAsymmetry`, line 476-524)
6. Score: 0 (both axes symmetric), 1 (one axis asymmetric), 2 (both asymmetric)

**Threshold**: 15% non-overlapping area = asymmetric (line 464)

**Example values**:

| Lesion Type | Score | Meaning |
|-------------|-------|---------|
| Round symmetric mole | 0 | Symmetric on both axes |
| Slightly irregular nevus | 1 | Asymmetric on one axis |
| Melanoma | 2 | Asymmetric on both axes |

### 5b. Border -- `analyzeBorder()` (line 540-653)

**Algorithm**:
1. Find all border pixels (lesion pixels with at least one background neighbor)
2. Compute angle from centroid to each border pixel
3. Divide border pixels into 8 angular octants (45 degrees each)
4. For each octant: compute the coefficient of variation (CV) of the
   border radii (distance from centroid to each border pixel)
5. Also compute the average RGB gradient at the border (color transition sharpness)
6. An octant is "irregular" if CV > 0.25 OR average gradient > 80

**Output**: Score 0-8 (number of irregular octants)

**Example values**:

| Lesion Type | Score | Meaning |
|-------------|-------|---------|
| Round nevus | 0-1 | Smooth, regular border |
| Benign keratosis | 2-3 | Some irregularity |
| Melanoma | 5-8 | Highly irregular, abrupt edges |

### 5c. Color -- `analyzeColors()` (line 675-793)

**Algorithm**:
1. Sample up to ~5,000 lesion pixels (line 681)
2. Convert each to LAB color space
3. K-means clustering (k=6, 15 iterations, k-means++ seeding) (line 799-884)
4. Map each cluster centroid to the nearest of 6 standard dermoscopic colors:
   - White, Red, Light-brown, Dark-brown, Blue-gray, Black (line 660-667)
5. Only include colors present in >3% of lesion pixels (line 764)
6. Detect blue-white structures: requires both blue-gray >5% AND white >3%

**Output**: `ColorAnalysisResult`:
```typescript
{
  colorCount: number;           // 1-6
  dominantColors: Array<{name, percentage, rgb}>;
  hasBlueWhiteStructures: boolean;
}
```

**Example values**:

| Lesion Type | colorCount | Dominant Colors | Blue-White |
|-------------|------------|-----------------|------------|
| Common mole | 1-2 | light-brown (85%), dark-brown (15%) | false |
| Melanoma | 3-5 | dark-brown (40%), black (25%), blue-gray (20%), red (15%) | true |
| Age spot | 1-2 | light-brown (90%) | false |

### 5d. Texture (GLCM) -- `analyzeTexture()` (line 902-1005)

**Algorithm**: Gray-Level Co-occurrence Matrix (GLCM) analysis.

1. Quantize grayscale to 32 levels (line 906)
2. Build GLCM for 4 directions: 0, 45, 90, 135 degrees (line 916)
3. Normalize the co-occurrence matrix
4. Compute four Haralick features:
   - **Contrast**: Intensity differences between neighbors. High = rough texture.
   - **Homogeneity**: Closeness to the GLCM diagonal. High = uniform texture.
   - **Entropy**: Randomness/complexity. High = disordered.
   - **Correlation**: Linear dependency of gray levels.

All features are normalized to [0, 1].

**Example values**:

| Lesion Type | Contrast | Homogeneity | Entropy | Correlation |
|-------------|----------|-------------|---------|-------------|
| Uniform mole | 0.05 | 0.85 | 0.30 | 0.70 |
| Melanoma | 0.35 | 0.32 | 0.72 | 0.45 |

### 5e. Structural Patterns -- `detectStructures()` (line 1024-1300+)

**Algorithm**: Multiple detection methods for dermoscopic structures.

**Local Binary Pattern (LBP)** (line 1031-1092):
- For each lesion pixel, compute an 8-bit LBP code from its 8 neighbors
- Build a 256-bin LBP histogram
- Classify patterns as "uniform" (<=2 transitions in the binary code) or
  "non-uniform"
- Irregular network: `nonUniformCount > 0.35` (line 1094)

**Globule/dot detection** (line 1096-1147):
- Search for local minima (dark spots) that are significantly darker than
  their neighborhood (>20 gray levels darker)
- "Irregular" if the neighborhood variance is high (max-min > 60)
- Irregular globules: >3 found AND >40% of all globules are irregular

**Streak/pseudopod detection** (line 1149-1200+):
- Analyze the border region (outer 20% of bounding box)
- Compute Sobel gradients at each border pixel
- Check if gradient direction is radial (pointing outward from center)
- High radial gradient magnitude indicates streaks

**Blue-white veil**: Derived from the color analysis (blue-gray + white presence)

**Regression structures**: White scar-like areas within the lesion

**Output**: `StructureResult`:
```typescript
{
  hasIrregularNetwork: boolean;
  hasIrregularGlobules: boolean;
  hasStreaks: boolean;
  hasBlueWhiteVeil: boolean;
  hasRegressionStructures: boolean;
  structuralScore: number; // 0-1 composite suspicion
}
```

### 5f. Diameter -- `estimateDiameterMm()` (line 1914-1925)

**Algorithm**: Assumes a standard dermoscope field of view.

```
fieldOfViewMm = 25 / (magnification / 10)  // 25mm at 10x
pxPerMm       = imageWidth / fieldOfViewMm
radiusPx      = sqrt(areaPixels / pi)
diameterMm    = 2 * radiusPx / pxPerMm
```

This is the **low-confidence fallback**. The full measurement system
(see [Lesion Measurement Subsystem](#lesion-measurement-subsystem)) tries
LiDAR, connector detection, and texture analysis first.

**Known weakness**: Assumes 10x magnification and 25mm FOV. Clinical photos
(non-dermoscopy) produce wildly inaccurate measurements.

---

## Step 6: Neural Network Classification (4-Layer Ensemble)

**What happens**: The image is classified into one of 7 HAM10000 classes using
a multi-layer ensemble. The layers are tried in priority order; unavailable
layers are skipped and weights redistributed.

**Where it lives**: `src/lib/mela/classifier.ts:classify()` (line 171)

**Input**: `ImageData`

**Output**: `ClassificationResult`:
```typescript
{
  topClass: LesionClass;           // "mel" | "nv" | "bcc" | "bkl" | "akiec" | "df" | "vasc"
  confidence: number;              // 0-1
  probabilities: ClassProbability[]; // All 7 classes, sorted descending
  modelId: string;                 // Which ensemble path was used
  inferenceTimeMs: number;
  usedWasm: boolean;
  usedHF: boolean;
  usedDualModel: boolean;
  usedCustomModel: boolean;
}
```

### The 7 classes (HAM10000 taxonomy)

| Code | Full Name | Base Prevalence | Risk |
|------|-----------|-----------------|------|
| `nv` | Melanocytic nevus (common mole) | 66.95% | Benign |
| `bkl` | Benign keratosis (age spot) | 10.97% | Benign |
| `mel` | Melanoma | 11.11% | Malignant |
| `bcc` | Basal cell carcinoma | 5.13% | Malignant |
| `akiec` | Actinic keratosis | 3.27% | Pre-malignant |
| `vasc` | Vascular lesion | 1.42% | Benign |
| `df` | Dermatofibroma | 1.15% | Benign |

### Layer priority and ensemble weights (line 450-481)

The classifier tries these strategies in order:

**Priority 1: Custom-trained local ViT** (line 502-565)
- Model: ViT-base-patch16-224 trained on HAM10000 + ISIC 2019 (37,484 images)
- Endpoint: `/api/classify-local` (Python subprocess)
- Claimed melanoma sensitivity: 95.97% on external ISIC 2019 data
- **Weights**: 70% custom + 15% trained-weights + 15% rule-based

**Priority 2: Dual HuggingFace ViT models** (line 567-750)
- Model 1: `Anwarkh1/Skin_Cancer-Image_Classification` (ViT-Base, 85.8M params)
- Model 2: `skintaglabs/siglip-skin-lesion-classifier` (SigLIP 400M)
- Both called in parallel via `Promise.allSettled` (line 614)
- Ensembled at 50/50 weighting for all classes (line 734-750)
- **Weights**: 50% dual-HF + 30% trained-weights + 20% rule-based

**Priority 3: Single HuggingFace ViT** (line 226-233)
- Falls back to whichever single model responded
- **Weights**: 60% HF + 25% trained-weights + 15% rule-based

**Priority 4: Offline fallback** (line 234-238)
- No external models available. Local analysis only.
- **Weights**: 40% rule-based + 60% trained-weights

### Rule-based classifier -- `classifyReal()` (line 790-858)

This function runs the full image analysis pipeline (segmentation, asymmetry,
border, color, texture, structures) and then calls two sub-classifiers:

1. **`classifyFromFeatures()`** (image-analysis.ts, line 1535): A hand-crafted
   Bayesian classifier that computes per-class logits from feature values
   combined with HAM10000 log-priors. The melanoma logit includes a gate:
   requires at least 2 of 5 concurrent indicators to score positive
   (line 1616-1624). This prevents single-feature false positives.

2. **`classifyWithTrainedWeights()`** (trained-weights.ts): A logistic
   regression classifier with literature-derived weights for smooth
   discrimination.

These are blended at 40% rule-based + 60% trained-weights (line 843-847).

### Preprocessing (line 179)

Before neural network inference:
- `preprocessImage()`: Normalizes pixel values, resizes to 224x224
- `resizeBilinear()`: Bilinear interpolation for downsizing
- `toNCHWTensor()`: Converts from RGBA to NCHW float tensor

**Example output**:

| Scenario | topClass | confidence | modelId |
|----------|----------|------------|---------|
| Clear mole, online | nv | 0.82 | dual-ensemble-v2-... |
| Suspicious lesion, online | mel | 0.61 | dual-ensemble-v2-... |
| Mole, offline | nv | 0.68 | ensemble-v1-rule40-trained60 |
| Pale age spot, online | bkl | 0.55 | ensemble-v2-hf60-trained25-rule15 |

---

## Step 7: Threshold Application

**What happens**: Instead of simply picking the class with the highest
probability (argmax), per-class optimal thresholds are applied. A class only
"wins" if it exceeds its own threshold. This compensates for class imbalance
in HAM10000 (nv = 67% of training data).

**Where it lives**: `src/lib/mela/threshold-classifier.ts`

**Input**: `ClassProbability[]` + `ThresholdMode`

**Output**: Reranked `{ topClass, confidence, probabilities }`

### Three threshold modes

**Default/Triage** (optimal Youden J from ROC analysis, line 23-31):

| Class | Threshold | Meaning |
|-------|-----------|---------|
| akiec | 0.0586 | Even low probability triggers (rare, dangerous) |
| bcc | 0.1454 | Low threshold (cancer) |
| bkl | 0.2913 | Moderate (common, benign) |
| df | 0.5114 | High threshold (rare, needs strong signal) |
| mel | 0.6204 | High threshold (melanoma requires high confidence) |
| nv | 0.068 | Very low (it's the dominant class) |
| vasc | 0.782 | Highest (very rare, very distinctive) |

**Awareness mode** (line 34-42) -- lowered thresholds for malignant classes:

| Class | Awareness Threshold | Default Threshold | Change |
|-------|-------------------|-------------------|--------|
| mel | 0.25 | 0.6204 | -60% (catch more melanomas) |
| bcc | 0.08 | 0.1454 | -45% |
| akiec | 0.03 | 0.0586 | -49% |

**Algorithm** (line 80-106):
1. Filter to classes whose probability >= their threshold
2. Among passing classes, pick the highest probability
3. If NO class exceeds its threshold: fall back to argmax

**When it runs**: The threshold mode is a user setting. Default is "triage"
(MelaPanel.svelte, line 145). Applied at line 681-691 of MelaPanel.svelte.

**Example**:
```
Raw probabilities: mel=0.35, nv=0.30, bkl=0.25, bcc=0.10
Default thresholds: mel needs 0.6204, nv needs 0.068, bkl needs 0.2913, bcc needs 0.1454

Classes exceeding threshold: nv (0.30 > 0.068), bkl (0.25 < 0.2913? NO)
Only nv passes -> topClass = nv (not mel, even though mel had highest raw prob)

In awareness mode: mel threshold = 0.25 -> mel 0.35 > 0.25 -> mel PASSES
Highest passing class = mel -> topClass = mel
```

---

## Step 8: Meta-Classifier Fusion

**What happens**: The neural network probabilities (from Step 6-7) are fused
with clinical feature scores (from Step 5). When the AI says "melanoma" but
the clinical features say "benign," the melanoma probability is reduced.
This is the core PPV-improvement mechanism.

**Where it lives**: `src/lib/mela/meta-classifier.ts:metaClassify()` (line 181)

**Inputs**:
- `neuralProbs`: ClassProbability[] from the classifier
- `abcdeScores`: ABCDE scoring results
- `tdsScore`: Total Dermoscopy Score
- `sevenPointScore`: 7-point checklist score
- `diameterMm`: Measured lesion diameter

**Output**: `MetaClassification`:
```typescript
{
  adjustedProbabilities: ClassProbability[];  // Re-weighted, sum to 1
  adjustedTopClass: LesionClass;
  adjustedConfidence: number;
  neuralConfidence: number;      // Raw ViT melanoma prob
  clinicalConfidence: number;    // Clinical suspicion score 0-1
  agreement: "concordant" | "discordant" | "neutral";
  adjustmentReason: string;
}
```

### Step 8a: Clinical Suspicion Score (line 45-81)

Computes a 0-1 score from clinical indicators:

| Indicator | Score | Source |
|-----------|-------|--------|
| Asymmetry >= 1 axis | +0.15 | ABCDE A |
| Border >= 4/8 irregular | +0.10 | ABCDE B |
| Color >= 3 distinct colors | +0.10 | ABCDE C |
| Blue-white veil detected | +0.10 | Color analysis |
| Diameter >= 6mm | +0.15 | ABCDE D |
| TDS > 4.75 (Stolz cutoff) | +0.20 | TDS formula |
| 7-point score >= 3 (biopsy threshold) | +0.20 | 7-point checklist |

Maximum possible: 1.00

### Step 8b: Agreement Classification (line 92-152)

Uses thresholds:
- `NEURAL_HIGH = 0.3` (line 37): Neural melanoma prob above this = suspicious
- `CLINICAL_HIGH = 0.35` (line 38): Clinical score above this = suspicious

| Neural | Clinical | Agreement | Mel Multiplier | Meaning |
|--------|----------|-----------|----------------|---------|
| High | High | Concordant | 1.2x (boost) | Both agree: suspicious |
| Low | Low | Concordant | 0.8x (suppress) | Both agree: benign |
| High | Low | Discordant | 0.6x (strong suppress) | AI says mel, features say benign |
| Low | High | Discordant | 1.3x (safety boost) | Features warn, AI missed it |

### Step 8c: Probability Adjustment (line 224-251)

1. Multiply the melanoma probability by the multiplier
2. Leave all other class probabilities unchanged
3. Renormalize so all probabilities sum to 1
4. Sort descending and determine new top class

**Example**:

```
Neural: mel=0.45 (HIGH, >0.3)
Clinical: score=0.20 (LOW, <0.35)
  Asymmetry=0 (+0), Border=2 (+0), Colors=2 (+0), Diameter=4mm (+0)
  TDS=3.2 (+0), 7-point=1 (+0)

Agreement: DISCORDANT (neural high, clinical low)
Multiplier: 0.6x

mel adjusted: 0.45 * 0.6 = 0.27
After renormalize: mel drops from #1 to potentially #2 or #3
Reason: "Clinical features (low ABCDE score) suggest this may be benign despite AI flag"
```

**Where it runs in the pipeline**: MelaPanel.svelte lines 780-794 (single
image) and lines 489-503 (multi-image). The meta-classifier result OVERRIDES
the displayed classification.

---

## Step 9: Bayesian Risk Stratification

**What happens**: The model's probability output is converted into a
real-world post-test probability using Bayes' theorem. This fixes the
fundamental PPV problem: at 2% real-world melanoma prevalence, a model
with 90% sensitivity and 90% specificity gives a PPV of only 8.9%.
By computing an honest post-test probability, users get a calibrated
risk assessment instead of a misleading binary "melanoma detected."

**Where it lives**: `src/lib/mela/risk-stratification.ts:assessRisk()` (line 148)

**Inputs**:
- `topClass`: Top predicted class
- `modelConfidence`: Confidence for that class
- `allProbabilities`: Full probability distribution (all 7 classes)
- `demographics`: Optional age and body location

**Output**: `RiskAssessment`:
```typescript
{
  riskLevel: "very-high" | "high" | "moderate" | "low" | "minimal";
  postTestProbability: number;  // The honest answer: P(malignant | model output)
  preTestProbability: number;   // Base prevalence (age-adjusted)
  likelihoodRatio: number;
  headline: string;
  action: string;
  urgency: string;
  color: string;
}
```

### The math (line 154-178)

**1. Aggregate malignant probability**:
```
malignantProb = P(mel) + P(bcc) + P(akiec)   // capped at 0.9999
```

**2. Age-adjusted prevalence** (line 47-60):

| Age | Multiplier | Adjusted Prevalence (base 10%) |
|-----|------------|-------------------------------|
| <=29 | 0.3x | 3% |
| 30-50 | 0.7x | 7% |
| 51-70 | 1.5x | 15% |
| 71+ | 2.5x | 25% |

Base prevalence: melanoma 2% + BCC 5% + akiec 3% = **10% combined malignant**

**3. Bayesian update**:
```
Likelihood Ratio (LR)     = malignantProb / (1 - malignantProb)
Pre-test odds              = adjustedPrevalence / (1 - adjustedPrevalence)
Post-test odds             = preTestOdds * LR
Post-test probability      = postTestOdds / (1 + postTestOdds)
```

**4. Risk tier mapping** (line 74-115):

| Tier | Post-test Probability | Headline | Action | Color |
|------|----------------------|----------|--------|-------|
| very-high | >= 50% | "Urgent: consider consulting a healthcare provider" | Within 1 week | Red |
| high | >= 20% | "Consider consulting a healthcare provider" | Within 2 weeks | Orange |
| moderate | >= 5% | "Worth monitoring" | Photo monthly | Yellow |
| low | >= 1% | "Low concern" | Routine checks | Green |
| minimal | < 1% | "No concerning features" | Normal schedule | Teal |

**Example calculations**:

| Scenario | mel+bcc+akiec | Age | Adj. Prevalence | LR | Post-test Prob | Tier |
|----------|---------------|-----|-----------------|-----|---------------|------|
| High mel confidence | 0.70 | 55 | 0.15 | 2.33 | 29.2% | high |
| Moderate mel | 0.40 | 35 | 0.07 | 0.67 | 4.8% | low |
| Low mel, young | 0.15 | 25 | 0.03 | 0.18 | 0.5% | minimal |
| Very high, elderly | 0.85 | 75 | 0.25 | 5.67 | 65.4% | very-high |
| Common mole | 0.08 | 40 | 0.07 | 0.09 | 0.6% | minimal |

**Why this matters**: Without Bayesian stratification, a model outputting
"mel=0.40" would be translated as "possible melanoma, see a doctor." With
Bayesian stratification, for a 35-year-old at 7% base prevalence, that same
0.40 translates to a 4.8% post-test probability -- "low concern, routine checks."
This dramatically reduces unnecessary anxiety and referrals.

---

## Step 10: Consumer Translation

**What happens**: The final classification + risk assessment is translated
into plain English that someone with zero medical training can understand.

**Where it lives**: `src/lib/mela/consumer-translation.ts:translateForConsumer()` (line 159)

**Inputs**:
- `topClass`: The winning class after meta-classifier adjustment
- `confidence`: Confidence for that class
- `allProbabilities`: For Bayesian risk computation
- `demographics`: Optional age/location for prevalence adjustment

**Output**: `ConsumerResult`:
```typescript
{
  headline: string;         // "Common mole" or "Consider consulting a healthcare provider"
  riskLevel: "green" | "yellow" | "orange" | "red";
  riskColor: string;        // Hex color for UI
  explanation: string;      // What this condition is, in plain English
  action: string;           // "Monitor for changes" or "Schedule within 2 weeks"
  shouldSeeDoctor: boolean;
  urgency: "none" | "routine" | "soon" | "urgent";
  medicalTerm: string;      // HAM10000 class code
  confidence: number;
  riskAssessment?: RiskAssessment;  // Full Bayesian details
}
```

### Safety guard (line 170-187)

If the top class confidence is below 0.40, the system refuses to classify:
```
headline: "Unable to classify"
riskLevel: "yellow"
action: "Try a closer photo... If you have a mole that concerns you, consider consulting a healthcare provider"
```

This prevents spurious results when the model is essentially guessing.

### Translation table (line 34-119)

| Class | Consumer Name | Risk Level | Urgency | See Doctor? |
|-------|--------------|------------|---------|-------------|
| `nv` | "Common mole" | green | none | No |
| `bkl` | "Age spot / seborrheic keratosis" | green | none | No |
| `df` | "Firm skin bump" | green | none | No |
| `vasc` | "Blood vessel spot" | green | none | No |
| `akiec` | "Sun damage spot" | yellow | routine | Yes |
| `bcc` | "Needs evaluation" | orange | soon | Yes |
| `mel` | "Consider consulting a healthcare provider" | red | urgent | Yes |

### Bayesian override (line 192-223)

When full probability data is available, the Bayesian risk assessment
OVERRIDES the static translation table. The headline and urgency come from
the risk tier, not from the class. This means a "mel" classification with
low post-test probability might show "Low concern" instead of "See a
dermatologist."

The mapping from Bayesian tier to consumer display (line 129-144):

| Bayesian Level | Consumer Risk | Consumer Urgency |
|----------------|--------------|-----------------|
| very-high | red | urgent |
| high | orange | soon |
| moderate | yellow | routine |
| low | green | none |
| minimal | green | none |

---

## Step 11: Display Result

**What happens**: The consumer translation is rendered as a card with color
coding, headline, explanation, action steps, and optional medical details.

**Where it lives**: `src/lib/components/MelaPanel.svelte`

**Key derived values** (line 326-384):

- `sortedProbabilities`: All 7 classes sorted by probability (bar chart)
- `icd10`: ICD-10 code for the top class (for referral letters)
- `consumerResult`: The `translateForConsumer()` output (line 348-352)
- `abcdeItems`: Formatted ABCDE scores for the compact display row

**Result display flow**:

1. **Consumer card**: Headline + risk color + action + explanation
2. **ABCDE score row**: Five compact items (Asym, Border, Color, Diam, Evol)
3. **Probability bar chart**: All 7 classes with bars
4. **Medical details** (collapsible): TDS score, 7-point checklist, ICD-10
5. **Grad-CAM heatmap**: Attention overlay showing which regions drove the decision
6. **Explanation findings**: Per-feature evidence table with citations
7. **Meta-classifier report**: Agreement status and adjustment reason

**Healthy skin display**: When the safety gate rejects the image (Step 3),
`classificationError` is set to `"healthy_skin:..."`. The UI renders a
reassuring green card with the rejection reason instead of an error message.

---

## Multi-Image Consensus Path

When `multiCapture` mode is enabled (default: ON, line 121), the user
captures 2-3 images of the same lesion. Each is classified independently,
then combined via quality-weighted averaging.

**Where it lives**: `src/lib/mela/multi-image.ts:classifyMultiImage()` (line 134)

**Algorithm**:

1. **Classify each image independently** via `classifyWithDemographics()` (line 146)
2. **Score each image's quality** using `scoreImageQuality()` (line 82-122):
   - Sharpness (Laplacian variance, weight 0.4)
   - Contrast (RMS, weight 0.3)
   - Segmentation quality (lesion presence confidence, weight 0.3)
3. **Quality-weighted probability averaging** (line 152-159):
   ```
   For each class:
     weighted_prob[class] = SUM(image_prob[class] * quality_weight[image])
   where quality_weight[i] = quality_score[i] / sum(all_quality_scores)
   ```
4. **Melanoma safety gate** (line 167-179):
   If ANY single image has `mel > 60%`, preserve that probability even if
   the consensus would average it down:
   ```
   if (maxMelProb > 0.6 AND consensus_mel < maxMelProb):
     boost mel to maxMelProb
     reduce other classes proportionally
   ```
5. **Agreement score** (line 54-72): Pairwise cosine similarity of
   probability distributions across all images. Score of 1.0 = identical
   predictions. Low agreement suggests the lesion looks different from
   different angles.

**Orchestration in MelaPanel**: `analyzeMultiImage()` (line 403-529)
- Runs the safety gate on the first image only (line 409-414)
- After consensus classification, selects the BEST quality image
  for ABCDE scoring (line 450-451)
- Applies thresholds, meta-classifier, and records analytics

---

## Lesion Measurement Subsystem

**Where it lives**: `src/lib/mela/measurement.ts:measureLesion()` (line 117)

Four calibration strategies in priority order:

| Priority | Method | Confidence | Accuracy | Source |
|----------|--------|------------|----------|--------|
| 0 | LiDAR depth sensing | high | +/-0.3mm | iPhone Pro / WebXR |
| 1 | USB-C/Lightning connector | high | +/-0.5mm | Physical reference |
| 2 | Skin texture frequency | medium | +/-2-3mm | Texture analysis |
| 3 | Dermoscope FOV estimate | low | rough | Assumes 25mm/10x |

**Connector detection threshold**: confidence > 0.6 (line 44)
**Texture detection threshold**: confidence > 0.4 (line 46)

**Safety annotation** (line 70-97): If confidence is "low" AND the measured
diameter falls between 4mm and 8mm (straddling the 6mm clinical threshold),
a safety warning is appended: "Measurement uncertain near the 6mm clinical
threshold. Consider consulting a healthcare provider if this spot is growing."

**Area to diameter conversion** (line 59-64):
```
radiusPx    = sqrt(lesionAreaPixels / pi)
diameterMm  = 2 * radiusPx / pixelsPerMm
```

---

## Known Failure Modes

### Failure Mode 1: Normal Skin Classified as Melanoma (the original bug)

**What happened**: Before the lesion presence gate existed, photos of normal
skin (e.g., a hand with no moles) were passed directly to the classifier.
Because the classifier was trained exclusively on lesion images, it would
always predict SOME class -- and the class with the highest raw probability
was often melanoma (because mel has relatively high priors in HAM10000 at
11.11% and the model had no "none of the above" option).

**Root cause**: Classification models cannot output "no lesion present" --
they can only choose between the 7 HAM10000 classes.

**Fix**: The 5-layer lesion presence gate (Step 3) now rejects images
that don't contain a visible lesion.

**Residual risk**: The gate can still be fooled by skin features that mimic
lesions (tattoos, scars, insect bites).

### Failure Mode 2: Small Moles Missed by the Gate (current bug)

**What happened**: When Gate 4's color contrast threshold was raised from 15
to 35 (to fix Failure Mode 1), small, pale moles on fair skin started failing
the gate. A mole with colorDiff of 25-34 would be rejected as "no lesion."

**Root cause**: The tension between false positive rate (classifying normal
skin) and false negative rate (missing real but subtle lesions). The colorDiff
threshold of 35 is a single number trying to balance two failure modes across
all Fitzpatrick skin types.

**Current status**: Known tradeoff. The threshold of 35 was chosen to
minimize the more dangerous failure (false melanoma alarms causing unnecessary
anxiety) at the cost of occasionally missing a subtle lesion (which the user
can retake).

**Potential fix**: Adaptive threshold based on the detected Fitzpatrick type
of the surrounding skin. Fair skin (I-II) could use a lower threshold (25),
while darker skin (IV-VI) could use a higher one (45).

### Failure Mode 3: Phone Camera vs. Dermoscopy Quality Gap

**What happens**: The HAM10000 training dataset is entirely dermoscopic
images (taken with specialized dermatoscopes under polarized light). When
users submit clinical photos from phone cameras, the model encounters
images that look fundamentally different from its training data:

- Different lighting (ambient vs. polarized)
- Different magnification (wide-field vs. 10x)
- Different color rendition (sRGB vs. dermoscope-calibrated)
- Hair, shadows, and skin texture visible at macro scale

**Impact**: Classification accuracy degrades significantly on clinical
photos. The dual-HF models partially compensate because at least the
SigLIP model was trained on a broader image distribution, but the core
feature extraction (color analysis, structural patterns) is calibrated
for dermoscopic appearances.

**Mitigation**: The `imageType` detection (DermCapture.svelte, line 90)
flags clinical photos, and the measurement system uses different
calibration strategies for non-dermoscopy images.

### Failure Mode 4: Fitzpatrick Equity Gap (~30pp)

**What happens**: HAM10000 is heavily biased toward Fitzpatrick I-III
(Northern European skin tones). The dataset contains very few images of
lesions on Fitzpatrick V-VI skin. This causes:

1. **Segmentation failure**: Otsu thresholding assumes a bimodal histogram
   with lighter background and darker lesion. On dark skin, the histogram
   may be unimodal, causing the segmentation to either capture the entire
   image or nothing.

2. **Color analysis miscalibration**: The 6 dermoscopic reference colors
   (line 660-667 of image-analysis.ts) are calibrated for lighter skin
   tones. A "light-brown" lesion on Fitzpatrick VI skin may actually be
   very dark in absolute terms.

3. **Model bias**: The neural networks were primarily trained on lighter
   skin, giving lower accuracy on darker skin. Published literature
   suggests a 20-30 percentage point accuracy gap.

**Residual risk**: This is a systemic issue that cannot be fixed without
retraining on a more representative dataset (e.g., Fitzpatrick17k, DDI).

### Failure Mode 5: Low PPV at Real-World Prevalence

**What happens**: The HAM10000 test set has ~11% melanoma prevalence.
At that prevalence, a model with 90% sensitivity and 90% specificity has
a PPV of ~47.6%. But in real-world population analysis, melanoma prevalence is ~2%.
At 2% prevalence, the same model has PPV of only **8.9%** -- meaning
91 out of 100 "melanoma detected" results are false positives.

**Fix**: Bayesian risk stratification (Step 9) computes an honest
post-test probability that accounts for real-world prevalence. Instead of
"melanoma detected," the user sees a calibrated risk tier ("low concern"
for a 4.8% post-test probability).

**Residual risk**: The base prevalence rates (2% melanoma, 5% BCC, 3%
akiec) are population averages. Individual risk varies by genetics, sun
exposure history, and other factors not captured in the model. The age
multipliers (line 47-52 of risk-stratification.ts) are a crude
approximation.

---

## Where to Improve

Ranked by impact and urgency:

### 1. Adaptive Lesion Presence Gate (HIGH PRIORITY)

**Step affected**: Step 3

**Problem**: The fixed colorDiff threshold of 35 is too aggressive for pale
moles on fair skin and too permissive for dark skin artifacts.

**Solution**: Detect the Fitzpatrick type of surrounding skin (using mean
skin RGB), then select an appropriate colorDiff threshold from a lookup
table. Could also incorporate a lightweight neural classifier trained on
"lesion vs. no-lesion" binary task.

### 2. Fitzpatrick-Aware Feature Extraction (HIGH PRIORITY)

**Steps affected**: Steps 4, 5a-5e

**Problem**: All feature extractors assume a light-skinned baseline.

**Solution**:
- Normalize LAB L-channel relative to surrounding skin tone before Otsu
- Adjust DERM_COLORS reference values based on detected skin tone
- Train separate GLCM texture models for different skin type ranges

### 3. ONNX Runtime Offline Model (MEDIUM PRIORITY)

**Step affected**: Step 6

**Problem**: When offline, the system falls back to rule-based + trained-weights
(no neural network), which has significantly lower accuracy.

**Solution**: The ONNX Runtime Web integration exists but uses a generic
MobileNetV3 Small. Training a custom ONNX model from the same ViT weights
used by the custom local model would give near-online accuracy offline.

### 4. Longitudinal Evolution Tracking (MEDIUM PRIORITY)

**Step affected**: Step 5 (ABCDE "E" for Evolution)

**Problem**: The "E" score in ABCDE is always 0 (line 473 and 757 of
MelaPanel.svelte) because there is no baseline comparison. Evolution
(change over time) is the most sensitive indicator of melanoma.

**Solution**: Use the existing LesionTimeline component and pi-brain
integration to compare current measurements against previous scans of
the same lesion. Compute delta-diameter, delta-asymmetry, and
delta-color-count to populate the E score.

### 5. Calibration Verification (MEDIUM PRIORITY)

**Step affected**: Step 9

**Problem**: The Bayesian risk stratification uses a simplified likelihood
ratio (malignantProb / benignProb). This assumes the model's probability
output is well-calibrated (i.e., when the model says 0.40, it's right 40%
of the time). Most neural networks are NOT well-calibrated.

**Solution**: Run Platt scaling or isotonic regression on a held-out
validation set to create a calibration map. Apply the map before the
Bayesian update. Also compute and publish calibration metrics (ECE,
reliability diagrams) per Fitzpatrick type.

### 6. Multi-Angle Feature Fusion (LOW PRIORITY)

**Step affected**: Step 5 + Multi-Image Path

**Problem**: When multiple images are captured, each is classified
independently and probabilities are averaged. But the ABCDE features are
extracted only from the "best quality" image (line 450-451). Features
visible from one angle might be missed from another.

**Solution**: Extract features from all images and take the MAX suspicious
value per feature (worst-case safety approach). For example, if one image
shows asymmetry=2 and another shows asymmetry=1, use asymmetry=2.

### 7. Confidence Calibration Per Model Layer (LOW PRIORITY)

**Step affected**: Step 6

**Problem**: The ensemble weights (70/15/15 for custom model, 50/30/20 for
dual-HF) are fixed constants based on initial testing. They are not
dynamically calibrated per-image based on which model is more confident or
more likely to be correct.

**Solution**: Train a small meta-learner that takes all layer outputs and
image quality features as input and outputs optimal per-image weights. This
is standard stacking in ensemble learning.

---

## Appendix: Complete Threshold Reference

All magic numbers in one place for quick lookup.

### Lesion Presence Gate

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Overall variance threshold | 25 | image-analysis.ts:2018 |
| Center luminance diff threshold | 12 | image-analysis.ts:2059 |
| Min area ratio | 0.01 (1%) | image-analysis.ts:2073 |
| Max area ratio | 0.70 (70%) | image-analysis.ts:2073 |
| Color diff threshold | 35 | image-analysis.ts:2111 |
| Compactness threshold | 0.25 | image-analysis.ts:2123 |
| Overall confidence threshold | 0.50 | image-analysis.ts:2139 |

### Image Quality

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Sharpness poor | 0.15 | image-quality.ts:22 |
| Sharpness OK | 0.30 | image-quality.ts:23 |
| Brightness low | 50/255 | image-quality.ts:24 |
| Brightness high | 220/255 | image-quality.ts:25 |
| Contrast poor | 0.08 | image-quality.ts:26 |
| Glare pixel threshold | 245 | image-quality.ts:27 |
| Glare area threshold | 5% | image-quality.ts:28 |

### Feature Extraction

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Asymmetry threshold | 15% non-overlap | image-analysis.ts:464 |
| Border CV threshold | 0.25 | image-analysis.ts:647 |
| Border gradient threshold | 80 | image-analysis.ts:647 |
| Color min presence | 3% of pixels | image-analysis.ts:764 |
| Blue-gray min for blue-white | 5% | image-analysis.ts:781 |
| White min for blue-white | 3% | image-analysis.ts:783 |
| Irregular network LBP | >0.35 non-uniform | image-analysis.ts:1094 |
| GLCM quantization levels | 32 | image-analysis.ts:906 |

### Classifier Ensemble Weights

| Configuration | Weight Split | File:Line |
|---------------|-------------|-----------|
| Custom ViT available | 70/15/15 | classifier.ts:479-481 |
| Dual HF available | 50/30/20 | classifier.ts:474-476 |
| Single HF available | 60/25/15 | classifier.ts:469-471 |
| Offline only | 40/60 (rule/trained) | classifier.ts:465-466 |
| Dual HF inter-model | 50/50 for all classes | classifier.ts:493-496 |

### Threshold Classifier (ADR-123)

| Class | Default | Awareness | File:Line |
|-------|---------|-----------|-----------|
| akiec | 0.0586 | 0.03 | threshold-classifier.ts:23-42 |
| bcc | 0.1454 | 0.08 | threshold-classifier.ts:23-42 |
| bkl | 0.2913 | 0.20 | threshold-classifier.ts:23-42 |
| df | 0.5114 | 0.30 | threshold-classifier.ts:23-42 |
| mel | 0.6204 | 0.25 | threshold-classifier.ts:23-42 |
| nv | 0.068 | 0.15 | threshold-classifier.ts:23-42 |
| vasc | 0.782 | 0.50 | threshold-classifier.ts:23-42 |

### Meta-Classifier

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Neural high threshold | 0.30 | meta-classifier.ts:37 |
| Clinical high threshold | 0.35 | meta-classifier.ts:38 |
| Concordant high multiplier | 1.2x | meta-classifier.ts:109 |
| Concordant low multiplier | 0.8x | meta-classifier.ts:119 |
| Discordant neural-high | 0.6x | meta-classifier.ts:129 |
| Discordant clinical-high | 1.3x | meta-classifier.ts:140 |

### Bayesian Risk Stratification

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Melanoma prevalence | 2% | risk-stratification.ts:37 |
| BCC prevalence | 5% | risk-stratification.ts:38 |
| Akiec prevalence | 3% | risk-stratification.ts:39 |
| Combined malignant | 10% | risk-stratification.ts:42 |
| Age <=29 multiplier | 0.3x | risk-stratification.ts:48 |
| Age 30-50 multiplier | 0.7x | risk-stratification.ts:49 |
| Age 51-70 multiplier | 1.5x | risk-stratification.ts:50 |
| Age 71+ multiplier | 2.5x | risk-stratification.ts:51 |
| Very-high threshold | >= 50% post-test | risk-stratification.ts:77 |
| High threshold | >= 20% post-test | risk-stratification.ts:83 |
| Moderate threshold | >= 5% post-test | risk-stratification.ts:89 |
| Low threshold | >= 1% post-test | risk-stratification.ts:95 |
| Minimal threshold | < 1% post-test | risk-stratification.ts:101 |

### Consumer Translation

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Min confidence to classify | 0.40 | consumer-translation.ts:170 |
| Melanoma safety gate (multi) | 0.60 | multi-image.ts:13 |

### Measurement

| Parameter | Value | File:Line |
|-----------|-------|-----------|
| Connector confidence min | 0.6 | measurement.ts:44 |
| Texture confidence min | 0.4 | measurement.ts:46 |
| D threshold (clinical) | 6mm | measurement.ts:71 |
| Safety warning range | 4-8mm | measurement.ts:73-74 |
| Default FOV (dermoscope 10x) | 25mm | measurement.ts:188 |
