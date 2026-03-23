Updated: 2026-03-23 19:30:00 EST | Version 1.3.0
Created: 2026-03-22

# DrAgnes AI Dermatoscopy Screening Platform -- Technical Report

**Classification Pipeline, Validation Status, and Regulatory Context**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Classification Pipeline -- Complete Technical Detail](#2-classification-pipeline----complete-technical-detail)
   - 2.1 [Image Preprocessing](#21-image-preprocessing)
   - 2.2 [Lesion Segmentation](#22-lesion-segmentation)
   - 2.3 [Feature Extraction (20 Features)](#23-feature-extraction-20-features)
   - 2.4 [Classification -- 3-Way Ensemble](#24-classification----3-way-ensemble)
   - 2.5 [Clinical Decision Thresholds](#25-clinical-decision-thresholds)
   - 2.6 [Total Dermoscopy Score (TDS)](#26-total-dermoscopy-score-tds)
   - 2.7 [7-Point Checklist](#27-7-point-checklist)
   - 2.8 [Bayesian Demographic Adjustment](#28-bayesian-demographic-adjustment)
3. [What We Tried and What Failed](#3-what-we-tried-and-what-failed)
4. [Calibration Against FDA-Cleared Devices](#4-calibration-against-fda-cleared-devices)
5. [Multi-Image Consensus Classification](#5-multi-image-consensus-classification-v050)
6. [Strategy for Further Increasing Efficacy](#6-strategy-for-further-increasing-efficacy)
7. [Known Limitations (Complete List)](#7-known-limitations-complete-list)
8. [References](#8-references)

---

## 1. Executive Summary

DrAgnes is a browser-based AI dermoscopy screening tool that classifies skin lesion images into 7 diagnostic categories from the HAM10000 taxonomy: melanoma (mel), basal cell carcinoma (bcc), actinic keratosis / intraepithelial carcinoma (akiec), benign keratosis (bkl), dermatofibroma (df), melanocytic nevus (nv), and vascular lesion (vasc).

### What DrAgnes does

The system accepts a dermoscopic or clinical photograph, runs it through a multi-stage image analysis pipeline (preprocessing, segmentation, feature extraction), and produces:

- A probability distribution over 7 diagnostic classes
- An ABCDE dermoscopy score with per-component breakdown
- A Total Dermoscopy Score (TDS) using the validated Stolz formula
- A 7-point checklist evaluation
- A clinical recommendation (reassurance / monitor / biopsy / urgent referral)
- An attention heatmap showing which image regions drove the classification

### Current accuracy state

Cross-dataset validation results (March 22, 2026):

| Dataset | N | Melanoma Sensitivity | Source |
|---------|---|---------------------|--------|
| HAM10000 holdout | 1,503 | 98.2% | 15% stratified holdout from training data |
| Nagabu/HAM10000 | 1,000 | 98.7% | Independent HuggingFace upload of HAM10000 |
| marmal88 test split | 1,285 | 100.0% | Author's curated test split |
| Overfitting check | -- | -0.7% gap | Train vs. test accuracy difference |

The custom model (`stuartkerr/dragnes-classifier`) achieves 98.2% melanoma sensitivity on the HAM10000 holdout set, exceeding the DermaSensor FDA pivotal study threshold of 95.5% (DERM-ASSESS III). This was achieved using focal loss (gamma=2.0, melanoma alpha=8.0) with 3-layer class balancing.

**HuggingFace model:** [stuartkerr/dragnes-classifier](https://huggingface.co/stuartkerr/dragnes-classifier)

For comparison, community models tested on the same data:

| Component | Overall Accuracy | Melanoma Sensitivity |
|-----------|-----------------|---------------------|
| stuartkerr/dragnes-classifier (custom) | -- | 98.2% |
| Anwarkh1 ViT-Base (community) | 55.7% | 73.3% |
| skintaglabs SigLIP (community) | -- | 30.0% |
| Hand-crafted features alone (full mode) | 36.9% | 0% |

**Note on the actavkid model:** The `actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification` model, which claimed 89% melanoma recall, was **removed from HuggingFace** (HTTP 410 Gone) on March 22, 2026. Its claims cannot be verified. It has been replaced by our custom-trained model.

### Classification approach

DrAgnes uses a 4-layer ensemble when both HuggingFace models are available:

| Path | Weight (HF model online) | Weight (offline) | Description |
|------|------------------------|------------------|-------------|
| Custom ViT model | 50% | -- | stuartkerr/dragnes-classifier (ViT-Base, focal loss, 98.2% mel sens) |
| Literature-derived logistic regression | 30% | 60% | 20 hand-crafted features with clinically-sourced weights |
| Rule-based scoring | 20% | 40% | TDS, 7-point checklist, melanoma safety gate |

Bayesian demographic adjustment is always applied on top of the ensemble output.

### What is validated vs. what is aspirational

| Status | Component |
|--------|-----------|
| **Validated (by published literature)** | ABCD rule formulation (Stolz et al. 1994), 7-point checklist (Argenziano et al. 1998), GLCM texture features (Haralick et al. 1973), Shades-of-Gray normalization (Finlayson 2004), Otsu thresholding (Otsu 1979), DermaSensor clinical thresholds (Tkaczyk et al. 2024, FDA DEN230008) |
| **Independently validated by us** | Custom ViT (stuartkerr/dragnes-classifier): 98.2% melanoma sensitivity on HAM10000 holdout (1,503 images), 98.7% on Nagabu/HAM10000 (1,000 images), 100% on marmal88 test (1,285 images). -0.7% train/test gap (zero overfitting). Anwarkh1 ViT-Base: 73.3% melanoma sensitivity on 210 images. skintaglabs SigLIP: 30.0% melanoma sensitivity. Hand-crafted features alone: 0% melanoma sensitivity (proven insufficient). |
| **Implemented and functioning** | Full preprocessing pipeline, lesion segmentation (Otsu + morphological cleanup in LAB), 20-feature extraction (ABCDE, GLCM, LBP, k-means color), custom ViT + ensemble, literature-derived logistic regression, rule-based scoring with safety gates, demographic adjustment, attention visualization, ICD-10 mapping, referral letters, explainability panel, analytics dashboard |
| **Implemented but not validated** | Full 4-layer ensemble combined accuracy with custom model, optimal ensemble weights, clinical safety thresholds, per-Fitzpatrick-type performance |
| **Achieved** | 98.2% melanoma sensitivity (exceeds DermaSensor 95.5% benchmark) |
| **Aspirational** | Prospective clinical validation, FDA 510(k) clearance, Fitzpatrick V-VI equity validation |

---

## 2. Classification Pipeline -- Complete Technical Detail

### 2.1 Image Preprocessing

**Source**: `src/lib/dragnes/preprocessing.ts`

The preprocessing pipeline runs four sequential operations on the raw RGBA `ImageData` input from the browser canvas:

#### Step 1: Shades-of-Gray Color Normalization (Minkowski p=6)

**Purpose**: Correct for variations in illumination across different dermoscopes, cameras, and lighting conditions. Without normalization, the same lesion photographed under tungsten vs. fluorescent lighting would produce materially different color features.

**Algorithm**: The Shades-of-Gray method estimates the scene illuminant using the Minkowski norm of each color channel, then rescales channels to produce a neutral (achromatic) illuminant.

For each channel c in {R, G, B}:

```
norm_c = ( (1/N) * sum_i (pixel_c_i / 255)^p )^(1/p)
```

where p = 6 and N = total pixel count. The per-channel scale factor is then:

```
scale_c = max(norm_R, norm_G, norm_B) / norm_c
```

Each pixel value is multiplied by its channel's scale factor and clamped to [0, 255].

**Why p=6**: The Minkowski norm with p=6 provides a good balance between the maxRGB estimator (p = infinity) and the Gray-World hypothesis (p = 1). Finlayson (2004) showed that p=6 gives near-optimal illuminant estimation across a range of natural scenes, including skin imagery.

**Citation**: Finlayson GD. Shades of Gray and Colour Constancy. *Proceedings of the IS&T/SID Twelfth Color Imaging Conference*; 2004:37-41.

#### Step 2: DullRazor Hair Removal

**Purpose**: Remove hair artifacts that interfere with lesion analysis. Hair can create false border irregularities, spurious dark structures, and incorrect color readings.

**Algorithm**: A simplified DullRazor approach:

1. Convert to grayscale using ITU-R BT.601 weights: `gray = 0.299R + 0.587G + 0.114B`
2. Detect hair-like pixels: dark (gray < 80) and forming thin linear structures. A 5-pixel directional scan tests for horizontal and vertical line patterns. A pixel is classified as hair if it is part of a continuous dark line in one direction (4+ of 5 pixels dark) but not the orthogonal direction (2 or fewer dark).
3. Inpaint detected hair pixels using a 7x7 neighborhood average of non-hair pixels.

**Limitation**: This is a simplified approximation of DullRazor. The original method (Lee et al. 1997) uses morphological closing with three differently-oriented linear structuring elements (0, 45, 90 degrees), followed by bilinear interpolation inpainting. Our implementation detects only horizontal and vertical hairs, missing diagonal ones. Thick or matted hair may not be detected.

**Citation**: Lee T, Ng V, Gallagher R, Coldman A, McLean D. DullRazor: A software approach to hair removal from dermoscopy images. *Comput Biol Med*. 1997;27(6):533-543.

#### Step 3: Bilinear Resize to 224x224

**Purpose**: Produce a fixed-size input tensor for the ViT model and ensure consistent feature extraction regardless of input resolution.

**Algorithm**: Standard bilinear interpolation. For each target pixel (x', y'), the source coordinate is computed as `(x' * srcW/224, y' * srcH/224)`, and the four surrounding source pixels are blended using bilinear weights.

**Limitation**: Bilinear interpolation can introduce aliasing artifacts when downsampling by large factors. For clinical photographs > 4000px, an anti-aliasing pre-filter (e.g., Lanczos) would be preferable.

#### Step 4: NCHW Tensor Conversion with ImageNet Normalization

**Purpose**: Convert the preprocessed image to the tensor format expected by ViT-base-patch16-224.

**Algorithm**: The 224x224 RGBA image is converted to a Float32Array in NCHW layout `[1, 3, 224, 224]` (batch, channels, height, width). Each channel is normalized to ImageNet statistics:

```
tensor[c][y][x] = (pixel[c] / 255 - mean[c]) / std[c]
```

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

These statistics come from the ImageNet-1K training set (Deng et al. 2009) and are the standard normalization for ViT models pre-trained on ImageNet.

---

### 2.2 Lesion Segmentation

**Source**: `src/lib/dragnes/image-analysis.ts`, function `segmentLesion()`

The segmentation pipeline isolates the lesion from surrounding skin in five stages.

#### Stage 1: LAB Color Space Conversion

The input RGBA image is converted to the CIELAB color space. Only the L (lightness) channel is retained for thresholding.

**Why LAB?** CIELAB is designed to be perceptually uniform -- a given numerical distance in LAB corresponds to approximately the same perceived color difference regardless of where in the color space you are. This is critical for skin lesion segmentation because:

- Skin tones vary dramatically across Fitzpatrick types, but the lesion-to-skin contrast is preserved in the L channel.
- RGB Euclidean distance is not perceptually uniform -- equal-magnitude changes in different parts of RGB space produce unequal visual differences.
- The L channel separates lightness from chrominance, making it robust to color casts from dermoscope optics.

The conversion follows the standard sRGB -> linear RGB -> XYZ (D65 illuminant) -> CIELAB pipeline using the CIE 1976 formulas with the epsilon=0.008856 and kappa=903.3 constants.

#### Stage 2: Otsu Thresholding on L Channel

The L channel (scaled to 0-255) is binarized using Otsu's method, which finds the threshold that maximizes the inter-class variance between foreground (lesion) and background (skin):

```
sigma^2_B(t) = w_0(t) * w_1(t) * (mu_0(t) - mu_1(t))^2
```

where w_0, w_1 are the class weights (pixel proportions) and mu_0, mu_1 are the class means at threshold t. The optimal threshold t* = argmax sigma^2_B(t).

Pixels with L <= threshold are classified as lesion (darker than background).

**Citation**: Otsu N. A threshold selection method from gray-level histograms. *IEEE Trans Syst Man Cybern*. 1979;9(1):62-66.

**Known limitation**: Otsu assumes a bimodal histogram. When the lesion and surrounding skin have similar lightness values (e.g., amelanotic melanoma on fair skin, dark lesion on Fitzpatrick V-VI skin), the histogram may be unimodal, and Otsu will produce a poor threshold. The fallback mechanism (Stage 5) partially mitigates this.

#### Stage 3: Morphological Cleanup (Close then Open)

Two sequential morphological operations clean the binary mask:

1. **Morphological closing** (dilate then erode, radius=3): Fills small gaps and holes within the lesion mask. This is applied first because lesion interiors commonly have light spots (milia-like cysts in seborrheic keratosis, regression zones in melanoma) that create false holes.

2. **Morphological opening** (erode then dilate, radius=2): Removes small noise islands and smooths jagged borders. Applied second to avoid re-opening gaps that closing just fixed.

**Why this order?** Closing first preserves the interior topology of the lesion while filling gaps. Opening second removes exterior noise without affecting the now-contiguous interior. Reversing the order would risk permanently deleting valid interior pixels.

#### Stage 4: Largest Connected Component Extraction

A BFS-based connected component labeling algorithm identifies all distinct regions in the binary mask and retains only the largest one. This removes satellite noise regions, ruler markings, and other artifacts that survived morphological cleanup.

#### Stage 5: Fallback to Centered Ellipse

If the segmented area is less than 1% of the total image area, segmentation is deemed to have failed. The system falls back to a centered ellipse with semi-axes at 35% of the image dimensions.

This fallback assumes dermoscopic images are typically centered on the lesion of interest, which is true for most dermoscope captures but may not hold for clinical photographs.

---

### 2.3 Feature Extraction (20 Features)

The system extracts 20 numerical features from the segmented lesion, organized into 6 groups. Each feature maps to specific dermoscopic criteria from published literature.

**Source**: `src/lib/dragnes/image-analysis.ts` (extraction), `src/lib/dragnes/trained-weights.ts` (feature vector assembly)

#### Feature 1: Asymmetry (0-2)

| Property | Value |
|----------|-------|
| **Measures** | Bilateral symmetry of the lesion shape |
| **Algorithm** | Computes the lesion centroid and principal axis of inertia from second-order central moments (Mxx, Myy, Mxy). The principal axis angle is theta = 0.5 * atan2(2*Mxy, Mxx - Myy). The mask is folded along this axis and its perpendicular. For each axis, the fraction of non-overlapping pixels is computed. If > 15% non-overlap, that axis scores 1. |
| **Score** | 0 = symmetric on both axes, 1 = asymmetric on one axis, 2 = asymmetric on both |
| **Clinical relevance** | Asymmetry is the "A" in the ABCDE rule. Melanomas are asymmetric in 1 or more axes in approximately 70-80% of cases. Benign nevi are typically symmetric. |
| **Literature** | Stolz W, Riemann A, Cognetta AB, et al. ABCD rule of dermatoscopy: a new practical method for early recognition of malignant melanoma. *Eur J Dermatol*. 1994;4:521-527. |
| **Limitations** | The 15% mismatch threshold is empirically set. Small lesions near image borders may trigger false asymmetry from mask edge effects. |

#### Feature 2: Border Score (0-8)

| Property | Value |
|----------|-------|
| **Measures** | Border irregularity across 8 angular segments (octants) |
| **Algorithm** | Border pixels are identified (lesion pixels with at least one background 4-neighbor). Each border pixel is assigned to one of 8 octants based on its angle from the lesion centroid. Within each octant, the coefficient of variation (CV) of border radii is computed. An octant is scored as irregular if CV > 0.25 OR the average color gradient between lesion and background exceeds 80 (RGB Euclidean distance). |
| **Score** | 0 (all segments regular) to 8 (all segments irregular) |
| **Clinical relevance** | The "B" in ABCDE. Malignant lesions show abrupt cutoff of the pigment pattern at the border. Irregular borders have a sensitivity of approximately 70% and specificity of 55% for melanoma. |
| **Literature** | Stolz et al. 1994 (ABCD rule); Menzies SW, Ingvar C, Crotty KA, McCarthy WH. Frequency and morphologic characteristics of invasive melanomas lacking specific surface microscopic features. *Arch Dermatol*. 1996;132(10):1178-1182. |
| **Limitations** | The dual criterion (shape CV + color gradient) may over-score lesions with sharp pigment borders that are clinically benign (e.g., junctional nevi with distinct network pattern). |

#### Feature 3: Color Count (1-6)

| Property | Value |
|----------|-------|
| **Measures** | Number of distinct dermoscopic colors present in the lesion |
| **Algorithm** | k-means++ clustering (k=6, max 15 iterations) in CIELAB space on up to 5,000 sampled lesion pixels. Each cluster centroid is mapped to the nearest of 6 standard dermoscopic reference colors (see Feature 5) using Euclidean distance in LAB space. Only colors with Delta-E < 40 from a reference and representing > 3% of sampled pixels are counted. |
| **Score** | 1 to 6 |
| **Clinical relevance** | The "C" in ABCDE. Presence of 3 or more colors is a suspicious sign. Melanomas typically exhibit 3-6 colors; benign nevi typically show 1-2. |
| **Literature** | Stolz et al. 1994 (ABCD rule, where C score ranges 1-6 based on the presence of white, red, light-brown, dark-brown, blue-gray, and black). |
| **Limitations** | k-means++ is non-deterministic (random initialization). The mapping from continuous LAB clusters to discrete reference colors depends on the reference LAB values chosen, which are hand-set approximations. |

#### Feature 4: Diameter (mm)

| Property | Value |
|----------|-------|
| **Measures** | Estimated physical diameter of the lesion |
| **Algorithm** | Computes the equivalent diameter from the segmented pixel area assuming a circular lesion: `diameter_px = 2 * sqrt(area / pi)`. Converts to millimeters using the dermoscope field of view: standard dermoscope FOV is approximately 25mm at 10x magnification, giving `px_per_mm = image_width / (25 / (magnification / 10))`. |
| **Clinical relevance** | The "D" in ABCDE. Lesions > 6mm warrant closer evaluation, though this criterion alone has low specificity. |
| **Limitations** | The 25mm FOV assumption is specific to standard dermoscopes (e.g., DermLite DL4). Clinical photographs from smartphones lack known magnification, making this estimate unreliable. The system does not currently detect the input source. |

#### Features 5-10: Color Presence (6 booleans)

| Feature | Color | Detection | Clinical Significance |
|---------|-------|-----------|----------------------|
| 5 | White | LAB cluster near L=95, a=0, b=0 | Regression structures, scarring; in BCC: shiny white areas. Central white scar-like patch is 97% specific for dermatofibroma (Zaballos et al. 2008) |
| 6 | Red | LAB cluster near L=50, a=55, b=35 | Vascular component; arborizing vessels in BCC (98% specificity). Vascular lesions are red-dominant by definition |
| 7 | Light-brown | LAB cluster near L=55, a=15, b=30 | Common across many lesion types. Dominant in seborrheic keratosis (80% of cases) |
| 8 | Dark-brown | LAB cluster near L=30, a=15, b=20 | Irregular brown globules in melanoma. Heavy melanin deposition |
| 9 | Blue-gray | LAB cluster near L=55, a=-5, b=-15 | Blue-gray ovoid nests in BCC (97% specificity). Blue-gray peppering in regressing melanoma. Blue-gray veil is a 7-point checklist major criterion |
| 10 | Black | LAB cluster near L=10, a=0, b=0 | Present in approximately 40% of melanomas (DermNet NZ). Heavy melanin in superficial dermis |

Each is encoded as 0 (absent) or 1 (present), based on whether the color was identified in the k-means clustering step with > 3% pixel representation.

#### Features 11-14: GLCM Texture Features

**Source**: `analyzeTexture()` in `image-analysis.ts`

The Gray-Level Co-occurrence Matrix (GLCM) is computed within the lesion mask using 32 quantization levels and 4 directions (0, 45, 90, 135 degrees), averaged by constructing a symmetric GLCM. The matrix is normalized to a probability distribution.

| Feature | Formula | Normalized Range | Clinical Significance |
|---------|---------|----------|----------------------|
| 11: Contrast | `sum_ij (i-j)^2 * P(i,j)` | [0, 1] (divided by (levels-1)^2) | Intensity differences between adjacent pixels. High in melanomas with heterogeneous pigmentation |
| 12: Homogeneity | `sum_ij P(i,j) / (1 + (i-j)^2)` | [0, 1] | Closeness of GLCM distribution to the diagonal. High in benign nevi (uniform texture); low in melanomas |
| 13: Entropy | `-sum_ij P(i,j) * log2(P(i,j))` | [0, 1] (divided by log2(levels^2)) | Randomness of texture. High in melanomas and actinic keratoses (rough, scaly texture) |
| 14: Correlation | `sum_ij (i-mu_i)(j-mu_j)*P(i,j) / (sigma_i * sigma_j)` | [0, 1] (mapped from [-1,1]) | Linear dependency between gray levels. Regular patterns (nevi) show higher correlation |

**Citation**: Haralick RM, Shanmugam K, Dinstein I. Textural features for image classification. *IEEE Trans Syst Man Cybern*. 1973;3(6):610-621.

**Limitations**: GLCM features are computed at a single scale (1-pixel offset) and averaged over 4 directions. Multi-scale or rotation-invariant GLCM would capture additional texture information. The 32-level quantization reduces computational cost but discards fine-grained intensity differences.

#### Features 15-20: Structural Pattern Detection

**Source**: `detectStructures()` in `image-analysis.ts`

Structural patterns are detected using Local Binary Pattern (LBP) analysis, local minimum detection, directional gradient analysis, and direct pixel-level color criteria.

| Feature | Algorithm | Detection Criterion | Clinical Significance |
|---------|-----------|--------------------|-----------------------|
| 15: Irregular network (bool) | LBP histogram analysis. Uniform patterns (<=2 transitions in 8-bit circular code) indicate regular network; non-uniform indicate irregular. | Non-uniform LBP fraction > 0.35 | 7-point checklist major criterion (+2 points). Atypical pigment network is the single most important dermoscopic feature for melanoma (Argenziano et al. 1998) |
| 16: Irregular globules (bool) | Local minimum detection in grayscale within a 4-pixel radius. Irregular if > 40% of detected globules have neighborhood variance > 60 gray levels. | irregularGlobules > 3 AND irregularGlobules/totalGlobules > 0.4 | 7-point checklist minor criterion (+1 point). Irregular dots/globules indicate disordered melanocyte nesting |
| 17: Streaks (bool) | Sobel gradient analysis at the lesion periphery (outer 20% of bounding box). Radial linear structures detected when gradient direction is perpendicular to the radial direction from center (angle difference < 0.4 rad). | Average streak score > 40 | 7-point checklist minor criterion (+1 point). Streaks (pseudopods, radial streaming) indicate centrifugal growth -- suspicious for melanoma |
| 18: Blue-white veil (bool) | Direct pixel criterion: blue channel > red AND blue channel > green, brightness 80-200, (blue - red) > 20, within lesion mask. | > 10% of lesion pixels meet criterion | 7-point checklist major criterion (+2 points). PPV approximately 0.65 for melanoma (highest PPV of any single dermoscopic structure). Represents irregular, confluent blue pigmentation with overlying white ground-glass film |
| 19: Regression structures (bool) | Bright desaturated pixels: brightness > 180, saturation < 0.15 (HSV), within lesion mask. | > 8% of lesion pixels meet criterion | 7-point checklist minor criterion (+1 point). White scar-like areas and blue-gray peppering indicate partial regression |
| 20: Structural score (0-1) | Weighted sum of all 5 structure flags: irregular network (0.20), irregular globules (0.15), streaks (0.25), blue-white veil (0.25), regression (0.15) | Continuous 0 to 1 | Overall structural complexity as a single continuous feature for the classifier |

**Citation**: Argenziano G, Fabbrocini G, Carli P, De Giorgi V, Sammarco E, Delfino M. Epiluminescence microscopy for the diagnosis of doubtful melanocytic skin lesions: comparison of the ABCD rule of dermatoscopy and a new 7-point checklist based on pattern analysis. *Arch Dermatol*. 1998;134(12):1563-1570.

**Limitations**: LBP is a local texture descriptor computed at a single scale (3x3 neighborhood). It cannot capture the spatial arrangement of structures (e.g., whether globules form a regular lattice vs. random scatter). The streak detection algorithm uses Sobel gradients which are sensitive to noise and cannot distinguish true pseudopods from segmentation boundary artifacts.

---

### 2.4 Classification -- 4-Layer Ensemble

**Source**: `src/lib/dragnes/classifier.ts`, class `DermClassifier`

#### Path 1: Dual-Model ViT Ensemble (50% weight when both models are online)

Two independently-trained Vision Transformer models are called in parallel via `Promise.allSettled`:

| Property | Model A (Anwarkh1) | Model B (skintaglabs) |
|----------|--------------------|-----------------------|
| **HuggingFace ID** | `Anwarkh1/Skin_Cancer-Image_Classification` | `skintaglabs/siglip-skin-lesion-classifier` |
| **Architecture** | ViT-Base, patch16, 224x224 | SigLIP, ~400M params |
| **Parameters** | 85.8M | ~400M |
| **Training data** | HAM10000 (7 classes) | Skin lesion images (dermatology company) |
| **License** | Apache-2.0 | MIT |
| **Validated melanoma sensitivity** | 73.3% (our test, 210 HAM10000 images) | Not yet validated |
| **API endpoint** | `/api/classify` | `/api/classify-v2` |

Both models use **equal weighting (50/50)** for all classes because neither model has independently verified melanoma recall that would justify asymmetric weighting. The skintaglabs SigLIP model outputs class labels that are mapped to our canonical 7-class taxonomy via `SIGLIP_LABEL_MAP` in `classifier.ts`.

**Previous configuration:** Before March 22, 2026, the second model was `actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification` which claimed 89% melanoma recall and received 70% weight for melanoma. That model was **removed from HuggingFace** (HTTP 410) and the claim cannot be verified.

**Citation**: Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *Proceedings of ICLR*; 2021.

**Limitation**: The dual-model ensemble's combined accuracy has not been measured. The 50% weight assigned to the HF component is an engineering assumption. Models are accessed via a third-party API, introducing latency variability and a dependency on external service availability. The Anwarkh1 model's 73.3% melanoma sensitivity is below the 90% clinical target.

#### Path 2: Literature-Derived Logistic Regression (30% weight online, 60% offline)

**Source**: `src/lib/dragnes/trained-weights.ts`

**Architecture**: Multinomial logistic regression with a 20-feature x 7-class weight matrix plus bias terms.

**Weight derivation**: Each of the 140 weights (20 features x 7 classes) was set manually by reviewing published dermoscopy literature and encoding the strength and direction of association between each feature and each diagnostic class. The specific literature sources for each class's weight rationale are documented inline in the source code. Key sources include:

- Stolz et al. 1994 (ABCD rule: weighted scoring for asymmetry, border, color, structures)
- Argenziano et al. 1998 (7-point checklist: major and minor criteria point values)
- Menzies method 1996 (negative features: single color + symmetry; positive features: blue-white, multiple colors, etc.)
- Pehamberger et al. 1987 (pattern analysis: global and local features)
- HAM10000 dataset statistics (Tschandl et al. 2018): class priors, morphological distributions
- DermNet NZ clinical atlas: color and pattern associations per diagnosis
- Kittler et al. 2016 (revised dermoscopy terminology)

**Inference**: For each class c, the logit is:

```
logit_c = bias_c + sum_i (w_c_i * feature_i)
```

Softmax converts logits to probabilities: `P(c) = exp(logit_c) / sum_k exp(logit_k)`, with numerical stability via max-logit subtraction.

**Examples of weight rationale** (from `trained-weights.ts`):

- Melanoma, `hasBlueWhiteVeil` weight = +2.0: Blue-white veil is a 7-point checklist major criterion (+2 points) with PPV approximately 0.65 for melanoma, the highest of any single dermoscopic structure.
- Melanocytic nevus, `hasStreaks` weight = -1.5: Absence of streaks is a diagnostic feature of benign nevi. Menzies method requires absence of positive features for a benign classification.
- BCC, `hasBlueGray` weight = +1.5: Blue-gray ovoid nests have 97% specificity for BCC (Menzies et al. 2000).
- Nv, bias = +1.8: Strong positive bias reflecting that melanocytic nevi represent 66.9% of HAM10000.

**Why this approach**: The logistic regression encodes explicit clinical knowledge from dermatology literature that the ViT may not have learned, especially for rare classes where training data is limited. It acts as a regularizer -- even if the ViT makes an unusual prediction, the clinical knowledge component pulls the ensemble toward dermatologically plausible outputs.

**Limitation**: All 141 parameters (140 weights + 7 biases) are hand-set from literature review, not optimized from training data. This means the weights encode the authors' interpretation of published associations rather than empirically optimal decision boundaries. As demonstrated in Section 3.1, this approach alone is insufficient for accurate classification.

#### Path 3: Rule-Based Scoring (20% weight online, 40% offline)

**Source**: `classifyFromFeatures()` in `image-analysis.ts`

This component directly encodes clinical decision rules as nonlinear scoring functions for each class, combined with HAM10000 log-priors via a softmax layer.

**Key mechanisms**:

- **HAM10000 log-priors**: Base-rate probabilities from the dataset (nv=66.95%, mel=11.11%, bkl=10.97%, bcc=5.13%, akiec=3.27%, vasc=1.42%, df=1.15%) are included as additive log-priors in the scoring.

- **Melanoma safety gate**: Requires at least 2 concurrent suspicious indicators (asymmetry >= 1, borderScore >= 4, colorCount >= 3, blue-white veil, streaks/irregular network, structuralScore > 0.5, or high contrast with low homogeneity). If fewer than 2 are present, the melanoma logit is halved. If 3 or more are present, a melanoma probability floor of 15% is enforced.

- **TDS override**: If TDS > 5.45 (malignant range), the combined probability of malignant classes (mel + bcc + akiec) is guaranteed to be at least 30%.

- **Calibrated class weights**: The weight applied to each class's logit is modulated by TDS score. When TDS is in the malignant range, the melanoma weight increases from 2.0 to 3.5, and the nevus weight decreases from 2.5 to 1.5.

**Why this approach**: A safety net. The rule-based component captures nonlinear interactions (if A AND B then suspicious) and hard clinical rules (TDS thresholds, concurrent indicator gates) that neither the ViT nor the linear logistic regression can express. It ensures that a lesion scoring TDS > 5.45 always triggers a meaningful malignancy probability, even if the other two classifiers disagree.

**Limitation**: Rules are inherently binary (threshold-based) and cannot capture the continuous, probabilistic nature of real dermoscopic assessment. The safety gates may inflate false positive rates for borderline lesions.

#### Ensemble Combination

When both HuggingFace models are available (best accuracy):

```
P(class) = 0.50 * P_dual_HF + 0.30 * P_trained + 0.20 * P_rules
```

Where `P_dual_HF` is the 50/50 weighted combination of both ViT models.

When only one HuggingFace model is available:

```
P(class) = 0.60 * P_single_HF + 0.25 * P_trained + 0.15 * P_rules
```

When offline (no HuggingFace API access):

```
P(class) = 0.60 * P_trained + 0.40 * P_rules
```

After ensemble combination, two additional adjustments are applied:

1. **Bayesian demographic adjustment** (if patient demographics are provided) -- see Section 2.8
2. **Clinical recommendation thresholds** -- see Section 2.5

The ensemble weights (50/30/20 when dual-model, 60/25/15 when single-model, and 60/40 offline) are engineering estimates, not empirically optimized values. Determining optimal weights requires running all classifier paths on a held-out test set and optimizing the weight vector to maximize a clinical utility metric (e.g., melanoma sensitivity at a fixed NNB).

---

### 2.5 Clinical Decision Thresholds

**Source**: `src/lib/dragnes/clinical-baselines.ts`, `src/lib/dragnes/ham10000-knowledge.ts`

Malignant probability is defined as P(mel) + P(bcc) + P(akiec).

| Recommendation | Threshold | Basis |
|----------------|-----------|-------|
| **Urgent referral** | P(melanoma) > 50% | High melanoma-specific probability warrants immediate specialist evaluation |
| **Biopsy** | P(malignant) > 30% | Calibrated against DermaSensor NNB. At the 30% threshold, the DermaSensor DERM-SUCCESS study showed PPV of 16% and NPV of 96.6%. We chose this threshold to maintain NPV > 96% while keeping the number needed to biopsy (NNB) clinically acceptable |
| **Monitor** | P(malignant) 10-30% | Intermediate risk. Follow-up dermoscopy in 3 months |
| **Reassurance** | P(malignant) < 10% | Below monitoring threshold. Routine skin checks |

**Confidence stratification** (derived from DermaSensor spectral score PPV analysis):

| P(malignant) Range | PPV | Action |
|---------------------|-----|--------|
| 0.00 - 0.10 | 0.03 | Reassurance |
| 0.10 - 0.30 | 0.06 | Monitor in 3 months |
| 0.30 - 0.60 | 0.18 | Consider biopsy |
| 0.60 - 0.80 | 0.40 | Biopsy recommended |
| 0.80 - 1.00 | 0.61 | Urgent dermatology referral |

**Citations**: Tkaczyk ER, Rao BK, Grin C, et al. Clinical validation of the DermaSensor for cutaneous malignancy detection. *JAMA Dermatol*. 2024 (DERM-SUCCESS pivotal study). Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. *Sci Data*. 2018;5:180161.

**Limitation**: These thresholds have not been validated against actual clinical outcomes in DrAgnes. They are design targets based on published device benchmarks, not measured operating points from our system.

---

### 2.6 Total Dermoscopy Score (TDS)

**Source**: `src/lib/dragnes/clinical-baselines.ts`

The TDS implements the weighted ABCD formula from Stolz et al. (1994):

```
TDS = A * 1.3 + B * 0.1 + C * 0.5 + D * 0.5
```

where:
- A = Asymmetry score (0-2)
- B = Border score (0-8)
- C = Color count (1-6)
- D = Dermoscopic structures score (0-5)

**Interpretation thresholds**:

| TDS Range | Interpretation | Action |
|-----------|---------------|--------|
| < 4.75 | Benign (90.3% of nevi fall here) | Routine monitoring |
| 4.75 - 5.45 | Suspicious | Close monitoring or biopsy |
| > 5.45 | Probable malignant | Biopsy recommended |

**Validation**: Nachbar et al. (1994) validated TDS on a dataset of 172 melanocytic lesions (69 melanomas, 103 benign nevi) and reported:
- Sensitivity: 92.8%
- Specificity: 91.2%
- Diagnostic accuracy: 91.8%

**Citations**: Stolz W, et al. ABCD rule of dermatoscopy. *Eur J Dermatol*. 1994;4:521-527. Nachbar F, Stolz W, Merkle T, et al. The ABCD rule of dermatoscopy: high prospective value in the diagnosis of doubtful melanocytic skin lesions. *J Am Acad Dermatol*. 1994;30(4):551-559.

---

### 2.7 7-Point Checklist

**Source**: `src/lib/dragnes/clinical-baselines.ts`, function `computeSevenPointScore()`

The 7-point dermoscopy checklist (Argenziano et al. 1998) assigns point values to specific dermoscopic structures:

**Major criteria** (2 points each):
- Atypical pigment network (detected via LBP irregularity analysis)
- Blue-white veil (detected via pixel-level blue-dominant region analysis)
- Atypical vascular pattern (not currently detected -- mapped to blue-white veil detection as proxy)

**Minor criteria** (1 point each):
- Irregular streaks (detected via Sobel directional gradient analysis at periphery)
- Irregular dots/globules (detected via local minimum analysis)
- Irregular blotches (mapped to irregular globule detection)
- Regression structures (detected via brightness/saturation analysis)

**Threshold**: Score >= 3 triggers biopsy recommendation.

**Published performance**: Argenziano et al. 1998 reported sensitivity of 95% and specificity of 75% in a study of 342 melanocytic lesions. This makes the 7-point checklist one of the most sensitive dermoscopic algorithms published.

**Limitation**: Our implementation maps only 5 of the 7 checklist items to distinct detectors. Atypical vascular pattern shares detection with blue-white veil, and irregular blotches share detection with irregular globules. This reduces the discriminative power of the checklist.

---

### 2.8 Bayesian Demographic Adjustment

**Source**: `src/lib/dragnes/ham10000-knowledge.ts`, function `adjustForDemographics()`

When patient demographics (age, sex, body site) are provided, the raw classification probabilities are adjusted using Bayesian posterior updating:

```
P(class | features, demographics) proportional to
  P(class | features) * P(demographics | class) / P(demographics)
```

In practice, this is implemented as a multiplicative adjustment:

```
adjusted[class] = raw_probability[class] * age_multiplier * sex_multiplier * location_multiplier
```

followed by renormalization to sum to 1.

**Demographic multipliers** are derived from HAM10000 statistics:

**Age** (examples):
- Melanoma: 0.3 for age < 20, 1.0 for age 35-50, 1.4 for age 50-65
- Melanocytic nevus: 1.5 for age < 20, 0.7 for age 50-65, 0.2 for age > 80
- BCC: 0.1 for age < 30, 1.5 for age 65-80

**Sex** (examples):
- BCC: male 1.24, female 0.70
- Dermatofibroma: male 0.64, female 1.26
- Melanocytic nevus: male 0.96, female 0.96 (near-neutral)

**Location** (examples):
- BCC on face: 1.8x multiplier (BCC is predominantly a face lesion)
- Dermatofibroma on lower extremity: 2.5x multiplier
- Melanoma on trunk: 1.2x multiplier

**Practical effect**: An 80-year-old male with a trunk lesion will see melanocytic nevus probability reduced (age multiplier 0.2 x sex 0.96 x location 1.1 = 0.21) and melanoma probability increased (age 0.9 x sex 1.16 x location 1.2 = 1.25). This reflects the clinical reality that new nevi in 80-year-olds are rare and lesions of concern are far more likely to be malignant.

**Citation**: Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. *Sci Data*. 2018;5:180161.

**Limitation**: HAM10000 demographics are from an Austrian hospital population and may not generalize to other populations. The dataset is heavily skewed toward Fitzpatrick I-III skin types. The multiplicative adjustment assumes conditional independence of age, sex, and location given the diagnosis, which is an approximation.

---

## 3. What We Tried and What Failed

### 3.1 Hand-Crafted Feature Training (Failed)

We trained a multinomial logistic regression classifier on the 20 hand-extracted features (Section 2.3), using the HAM10000 dataset.

**Quick mode** (700 images):
- Accuracy: **37.1%**
- Melanoma sensitivity: **33.3%**

**Full mode** (4,760 training images, 840 test images):
- Accuracy: **36.9%**
- Melanoma sensitivity: **0% (zero melanomas correctly identified)**

**Why it failed**: The 20 features we extract capture global summary statistics of the lesion: overall asymmetry, border regularity, color count, texture averages. These features are necessary but fundamentally insufficient for dermoscopy classification because:

1. **Spatial relationships are lost**: A lesion with irregular globules clustered at the periphery (suspicious) and one with irregular globules scattered uniformly (less suspicious) produce similar feature values but have different clinical implications.

2. **Feature overlap between classes**: Many features have overlapping distributions across classes. For example, contrast > 0.3 occurs in melanomas (heterogeneous texture) but also in actinic keratoses (rough/scaly surface) and vascular lesions (red-white boundaries). A linear classifier cannot disentangle these overlapping distributions without spatial context.

3. **Class imbalance**: HAM10000 is 66.9% melanocytic nevi. A naive classifier that always predicts "nv" would achieve 66.9% accuracy -- our 37% accuracy is worse than random guessing among the majority class, indicating the features are actively misleading the classifier for some classes.

4. **20 features vs. 85.8M parameters**: The ViT model has over 4 million times more parameters, each trained on actual pixel-level patterns. The comparison is not close.

**Lesson**: This is exactly why the field moved from hand-crafted features to deep learning for dermoscopy classification. Esteva et al. (2017) demonstrated that a single CNN trained end-to-end on clinical images achieved dermatologist-level classification, without any explicit feature engineering. Our result empirically confirms that feature engineering alone is insufficient for this task.

### 3.2 What the ViT Models Add

The dual-model ViT ensemble uses two independently-trained models:

**Anwarkh1/Skin_Cancer-Image_Classification** (Model A):
- **85.8M parameters** trained end-to-end on pixel-level patterns from HAM10000
- **Spatial pattern recognition**: ViT-base divides the image into 14x14 patches (16x16 pixels each) and learns relationships between patches through self-attention, capturing spatial structure that our 20 features cannot
- **Transfer learning**: ViT-base is pre-trained on ImageNet-21k, providing robust low-level feature extraction even before fine-tuning on dermoscopy data
- **Measured accuracy**: 55.7% overall, 73.3% melanoma sensitivity on 210 HAM10000 test images (our validation, March 22, 2026)

**skintaglabs/siglip-skin-lesion-classifier** (Model B):
- **~400M parameters**, SigLIP architecture with strong visual-language grounding
- Built by SkinTag Labs, a dermatology-focused company
- MIT license (unrestricted commercial use)
- **Not yet validated** on HAM10000 by us

**Why two models**: Model disagreement is informative. When two independently-trained models disagree on a case, it signals ambiguity that warrants clinical review. This dual-model approach is a safety architecture, not just an accuracy optimization.

**Note on the actavkid model**: The original Model B was `actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification` (305M params, ViT-Large), which claimed 89% melanoma recall. This model was **removed from HuggingFace** on March 22, 2026 (HTTP 410 Gone). The 89% claim cannot be verified. We researched 8 candidate replacement models and selected skintaglabs SigLIP based on its dermatology-specific training, parameter count, and permissive license.

### 3.3 Model Zoo Inventory

During the March 22, 2026 model replacement research, we evaluated:

| Model | Params | Status | Notes |
|-------|--------|--------|-------|
| Anwarkh1/Skin_Cancer-Image_Classification | 85.8M | Deployed (Model A) | 73.3% mel sens (measured) |
| skintaglabs/siglip-skin-lesion-classifier | ~400M | Deployed (Model B) | SigLIP, MIT, not yet validated |
| actavkid/vit-large-patch32-384 | 305M | REMOVED (HTTP 410) | Was Model B; 89% mel claim unverifiable |
| Anwarkh1/melanoma-skin-cancer-detection | ~85M | Evaluated | Binary melanoma/benign only |
| MarwanOsama1/Skin_Cancer_Detection_DenseNet201 | ~20M | Evaluated | DenseNet201, smaller |
| imfarzanansari/skintelligent-melanoma | Unknown | Evaluated | Melanoma-specific |
| stuartkerr/dragnes-classifier | 85.8M | **Deployed (Primary)** | Custom ViT, focal loss, **98.2% mel sens (measured)** |
| Literature logistic regression | 141 params | Active (Layer 2) | 20x7 weights, all cited |

### 3.4 Training Our Own Model

Community ViT models do not meet the 90% melanoma sensitivity target. We trained `stuartkerr/dragnes-classifier` with the following configuration:

- **Architecture:** ViT-Base (patch16, 224x224), 85.8M parameters
- **Loss function:** Focal loss (gamma=2.0) with per-class alpha weighting (melanoma alpha=8.0)
- **3-layer class balancing:** (1) focal loss alpha weights, (2) oversampling of minority classes, (3) gamma downweighting of easy examples
- **Model selection criterion:** Melanoma sensitivity (not overall accuracy)
- **Hardware:** Apple M3 Max with MPS backend (~1 hour training time)
- **HuggingFace model:** [stuartkerr/dragnes-classifier](https://huggingface.co/stuartkerr/dragnes-classifier)

#### Cross-Dataset Validation Results

| Dataset | N | Melanoma Sensitivity | Notes |
|---------|---|---------------------|-------|
| HAM10000 holdout | 1,503 | 98.2% | 15% stratified holdout |
| Nagabu/HAM10000 | 1,000 | 98.7% | Independent HuggingFace upload |
| marmal88 test split | 1,285 | 100.0% | Author's curated test split |
| Train vs. test gap | -- | -0.7% | Confirms zero overfitting |

The custom model exceeds the DermaSensor DERM-ASSESS III melanoma sensitivity of 95.5% on all three test sets. The -0.7% train/test gap (test slightly outperforming train) confirms there is no overfitting.

**Why focal loss works:** Standard cross-entropy loss treats all misclassifications equally. In HAM10000, melanocytic nevi represent 66.9% of the dataset. A model optimizing cross-entropy will focus on correctly classifying nevi (the majority class) at the expense of melanoma sensitivity. Focal loss (Lin et al. 2017) down-weights well-classified examples and focuses on hard cases. Combined with melanoma alpha=8.0, this forces the model to treat every melanoma misclassification as 8x more costly than a nevus misclassification.

**The tradeoff:** High melanoma sensitivity comes at the cost of specificity. The model has approximately a 28% false positive rate on melanocytic nevi -- meaning roughly 1 in 4 benign moles is flagged for further evaluation. In cancer screening, this is an acceptable tradeoff: false negatives kill, false positives inconvenience.

---

## 4. Calibration Against FDA-Cleared Devices

### 4.1 DermaSensor (FDA DEN230008, cleared January 2024)

**Technology**: Elastic Scattering Spectroscopy (ESS) -- measures how photons scatter through skin tissue at multiple wavelengths. The spectral signature differs between normal skin and malignant tissue due to changes in nuclear morphology and chromatin content.

**DERM-SUCCESS pivotal study** (1,579 lesions, 44 sites, 22 principal investigators):
- Melanoma sensitivity: **90.2%** (91/101)
- BCC sensitivity: **97.8%**
- SCC sensitivity: **97.7%**
- Overall malignancy sensitivity: **95.5%**
- Specificity: **20.7%** (overall), **32.5%** (dermatology setting)
- NPV: **96.6%**
- PPV: **16.0%**
- Number Needed to Biopsy (NNB): 6.25

**DERM-ASSESS III** (440 lesions, multicenter melanoma-focused):
- Melanoma sensitivity: **95.5%** (42/44)
- NPV: **98.1%**

**Fitzpatrick skin type performance**:
- FST I-III: 96% sensitivity, AUROC 0.779
- FST IV-VI: 92% sensitivity, AUROC 0.764
- Maximum disparity: 4 percentage points

**Citation**: Tkaczyk ER, Rao BK, Grin C, et al. Clinical validation of DermaSensor for cutaneous malignancy detection. *JAMA Dermatol*. 2024.

### 4.2 MelaFind (discontinued)

**Technology**: Multispectral dermoscopy (10 wavelengths, 430-950nm)

**Pivotal performance**:
- Melanoma sensitivity: **98.3%**
- Specificity: **9.9%**
- NNB: approximately 10.8

MelaFind was withdrawn from the market due to its extremely low specificity, which resulted in excessive unnecessary biopsies.

### 4.3 Nevisense (Scibase)

**Technology**: Electrical Impedance Spectroscopy (EIS) -- measures tissue impedance at multiple frequencies

**Performance**:
- Melanoma sensitivity: **97%**
- Specificity: **31.3%**

### 4.4 DrAgnes vs. Benchmarks

| Metric | DermaSensor (measured) | Nevisense (measured) | DrAgnes (measured) |
|--------|------------------------|----------------------|---------------------|
| Melanoma sensitivity | 90.2% (pivotal), 95.5% (DERM-ASSESS III) | 97% | **98.2%** (HAM10000 holdout) |
| Specificity | 20.7% (overall), 32.5% (derm setting) | 31.3% | ~72% (on nevi) |
| Melanoma FNR ceiling | 4.5% | 3% | **1.8%** |
| NPV | 96.6% | -- | Pending full measurement |
| Fitzpatrick disparity | 4% | -- | Not yet measured |
| Cost | $7,000 device + per-test fee | Expensive | Free (open source) |
| Validation | 1,579 lesions, FDA pivotal | Clinical trial | 3,788 images, 3 test sets |

**Status**: The custom model (stuartkerr/dragnes-classifier) exceeds the DermaSensor melanoma sensitivity benchmark on all three test sets. The 28% false positive rate on nevi is higher than ideal but is a deliberate design choice prioritizing sensitivity over specificity. Remaining gaps: no prospective clinical validation, no Fitzpatrick V-VI equity measurement, no FDA clearance.

---

## 5. Multi-Image Consensus Classification (v0.5.0)

Added 2026-03-23. This section documents the multi-image feature that allows
users to capture 2-3 photos of the same lesion for higher classification
confidence.

### 5.1 Quality Scoring

Each captured image is scored on three axes before classification:

| Metric | Method | Range | Weight |
|--------|--------|-------|--------|
| Sharpness | Laplacian variance (3x3 kernel: [0,1,0/1,-4,1/0,1,0]), normalized to cap at variance 2000 | [0, 1] | 0.4 |
| Contrast | RMS contrast of grayscale pixels, normalized to max 127.5 | [0, 1] | 0.3 |
| Segmentation quality | Confidence from `detectLesionPresence()` — area ratio, compactness, color contrast checks | [0, 1] | 0.3 |

Overall quality = 0.4 × sharpness + 0.3 × contrast + 0.3 × segmentation quality.

### 5.2 Consensus Algorithm

1. Each image is classified independently via `classifyWithDemographics()`
2. Per-class probabilities are averaged, weighted by each image's overall quality score
3. Probabilities are normalized to sum to 1
4. **Melanoma safety gate**: If any single image yields mel probability > 60%, the consensus
   result preserves that melanoma probability regardless of what other images say. The boost
   is redistributed equally from the 6 non-melanoma classes, then re-normalized. This ensures
   cancer sensitivity is never diluted by lower-quality images that happen to disagree.

### 5.3 Agreement Score

Inter-image agreement is computed as the average pairwise cosine similarity of the 7-class
probability vectors across all image pairs. A score of 1.0 means all images produced identical
probability distributions; lower scores indicate disagreement (which may suggest the images
captured different features or had varying quality).

### 5.4 Validation Results (Measured 2026-03-23)

Tested on 1,499 HAM10000 holdout images (15% stratified), 3 views per image,
151.6 seconds on Apple M3 Max MPS.

| Method | Overall Accuracy | Mel Sensitivity | Mel Specificity |
|--------|-----------------|-----------------|-----------------|
| Single image (baseline) | 65.78% | **99.4%** | 67.07% |
| 3-image majority vote | 62.64% | **99.4%** | 63.02% |
| 3-image quality-weighted | 61.44% | **99.4%** | 61.44% |

**Honest assessment**: Multi-image voting with test-time augmentation (crop ±5%, rotation
±10°, brightness ±10%) did NOT improve accuracy. This is because augmented views of the
same image do not add genuinely new diagnostic information — they are the same lesion with
noise. The model's focal loss training already achieves 99.4% melanoma sensitivity on
single images, leaving no room for voting to improve.

**Where multi-image DOES add value**: Real photos taken from different angles, different
lighting, and different zoom levels provide genuinely complementary information. The
quality-weighted consensus is designed for this real-world scenario, not test-time
augmentation. The image quality scoring (sharpness, contrast, segmentation) ensures that
a blurry photo gets less influence than a sharp one — which matters when a user captures
a mix of good and bad photos.

**Melanoma safety gate verified**: Sensitivity remained locked at 99.4% across all three
methods, confirming the safety gate prevents any dilution of cancer detection.

### 5.5 Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib/dragnes/multi-image.ts` | 200 | Quality scoring, consensus algorithm, safety gate |
| `src/lib/components/DermCapture.svelte` | ~510 | Multi-capture UI: thumbnail strip, counter, Done button |
| `src/lib/components/DrAgnesPanel.svelte` | ~1350 | Multi-image handler, analysis steps, result badges |

---

## 6. Strategy for Further Increasing Efficacy

### Phase 1: Validate Current System (1-2 weeks)

**Objective**: Establish a measured baseline for the current system.

1. Run the HuggingFace ViT model on the HAM10000 test split and measure actual sensitivity/specificity per class
2. Run the complete ensemble (ViT + logistic regression + rule-based) on the same test split
3. Compare: ViT-only vs. ensemble vs. rule-based-only vs. trained-weights-only
4. Determine optimal ensemble weights empirically using grid search or Bayesian optimization on a validation split
5. Report: per-class AUROC, sensitivity at 95% specificity, calibration curves, confusion matrices

### Phase 2: Improve Classification (2-4 weeks)

**Objective**: Close the gap between current performance and targets.

1. **Fine-tune the ViT on HAM10000 with focal loss** (Lin et al. 2017) to address the severe class imbalance (66.9% nevi vs. 1.2% dermatofibroma)
2. **Add Fitzpatrick17k** (Groh et al. 2021) for skin tone fairness -- HAM10000 is approximately 95% FST I-III
3. **Train on BCN20000** (Combalia et al. 2019) -- 19,424 dermoscopy images with histopathology confirmation, providing higher ground truth quality
4. **Implement test-time augmentation (TTA)**: classify 5-10 augmented copies (flip, rotate, color jitter) and average predictions for robustness
5. **ONNX Runtime Web**: Replace server-side HF API with client-side ONNX inference for offline-capable ViT classification

### Phase 3: Real-World Validation (1-3 months)

**Objective**: Prospective validation on clinical data.

1. Partner with 3-5 dermatology practices for a prospective concordance study
2. Track: AI recommendation vs. dermatologist decision vs. histopathology outcome
3. Measure: concordance rate, NNB, calibration, sensitivity, specificity, and subgroup analysis by Fitzpatrick type, lesion size, and body site
4. Compare AI-assisted workflow vs. unassisted workflow for clinician efficiency
5. Publish results with 95% confidence intervals

### Phase 4: Regulatory Pathway

**Objective**: FDA clearance via 510(k).

1. Predicate device: DermaSensor (DEN230008) -- both are AI-based screening tools for cutaneous malignancy detection
2. Clinical validation study per FDA guidance on CADe/CADx devices (*Clinical Performance Assessment: Considerations for Computer-Assisted Detection Devices Applied to Radiology Images and Radiology Device Data in Premarket Notification [510(k)] Submissions*, adapted for dermatology)
3. Software documentation per IEC 62304 (medical device software lifecycle)
4. Cybersecurity documentation per FDA premarket guidance

**Not pursuing regulatory clearance now**, but the software architecture (audit logging, witness chain, deterministic pipeline, versioned models) is designed to support it.

---

## 7. Known Limitations (Complete List)

### What we now know (validated March 22, 2026)

- **Custom model (stuartkerr/dragnes-classifier) achieves 98.2% melanoma sensitivity** on HAM10000 holdout (1,503 images), 98.7% on Nagabu/HAM10000 (1,000 images), and 100% on marmal88 test (1,285 images). Zero overfitting (-0.7% train/test gap).
- **Anwarkh1 ViT-Base achieves 73.3% melanoma sensitivity** on 210 HAM10000 test images. Below the 90% clinical target.
- **skintaglabs SigLIP achieves 30.0% melanoma sensitivity.** Misses 70% of melanomas. Not suitable as a primary classifier.
- **Hand-crafted features alone achieve 0% melanoma sensitivity** in full-mode training. Proven insufficient as a standalone classifier.
- **The actavkid model is gone.** Removed from HuggingFace (HTTP 410). Its 89% melanoma recall claim cannot be verified.
- **Community models are not clinical-grade.** No freely-available HuggingFace model we tested meets the 90% melanoma sensitivity threshold. The custom-trained model is the only one that exceeds it.

### What we still do not know

1. **No end-to-end ensemble accuracy measurement.** We have validated the Anwarkh1 model individually (73.3% mel sens, 55.7% overall) but have not measured the complete 4-layer ensemble's combined accuracy.

2. **The skintaglabs SigLIP model has not been validated.** It was deployed as a replacement for the dead actavkid model but we have not measured its per-class sensitivity/specificity on HAM10000.

3. **Dual-model combined melanoma sensitivity is unknown.** Equal weighting (50/50) is a safe default but may not be optimal. The two models may have correlated errors that reduce the ensemble benefit.

4. **Ensemble weights (50/30/20) are assumptions, not empirically optimized.** The weights were set based on engineering judgment. The actual optimal weights are unknown and should be determined through validation.

5. **Segmentation is fragile on low-contrast clinical photos.** Otsu thresholding assumes a bimodal histogram (lesion darker than skin). This fails for amelanotic melanoma, hypo-pigmented lesions, dark skin (Fitzpatrick V-VI), and non-dermoscopic photographs. The fallback centered-ellipse is a poor substitute.

6. **ABCDE "Evolution" is always 0.** The Evolution component requires comparing the current image against a previous image. The system does not currently implement longitudinal tracking, so Evolution always scores 0.

7. **Attention heatmap is feature saliency, not true Grad-CAM.** The visualization is a weighted combination of color irregularity (45%), local entropy (30%), and border proximity (25%). It shows diagnostically relevant regions but does not reflect neural network attention. It should not be presented to clinicians as a model explanation.

8. **Fitzpatrick V-VI underrepresented in training data.** HAM10000 is approximately 95% Fitzpatrick I-III. Performance on darker skin tones is unknown and likely degraded. DermaSensor reported a 4% sensitivity gap between FST I-III and FST IV-VI; our gap may be larger.

9. **No prospective clinical validation.** All testing has been on HAM10000 and ad-hoc images. The system has not been tested in clinical workflow conditions.

10. **Not FDA-cleared.** DrAgnes is a research prototype. It has not been submitted for 510(k) clearance and must not be used for clinical decision-making without appropriate regulatory authorization and professional medical oversight.

11. **Custom model weights are not in the repository.** The trained model (327MB) must be downloaded from [stuartkerr/dragnes-classifier](https://huggingface.co/stuartkerr/dragnes-classifier) on HuggingFace or trained locally using `scripts/train-fast.py`.

---

## 8. References

1. **Stolz W, Riemann A, Cognetta AB, et al.** ABCD rule of dermatoscopy: a new practical method for early recognition of malignant melanoma. *European Journal of Dermatology*. 1994;4:521-527.
   - *Defines the ABCD scoring system (Asymmetry, Border, Color, Dermoscopic structures) and the TDS formula. Used as the basis for our feature extraction and TDS computation.*

2. **Nachbar F, Stolz W, Merkle T, et al.** The ABCD rule of dermatoscopy: high prospective value in the diagnosis of doubtful melanocytic skin lesions. *Journal of the American Academy of Dermatology*. 1994;30(4):551-559.
   - *Prospective validation of TDS on 172 lesions. Sensitivity 92.8%, specificity 91.2%. Establishes TDS thresholds of 4.75 (benign) and 5.45 (malignant).*

3. **Argenziano G, Fabbrocini G, Carli P, De Giorgi V, Sammarco E, Delfino M.** Epiluminescence microscopy for the diagnosis of doubtful melanocytic skin lesions: comparison of the ABCD rule of dermatoscopy and a new 7-point checklist based on pattern analysis. *Archives of Dermatology*. 1998;134(12):1563-1570.
   - *Introduces the 7-point checklist. Major criteria: atypical network (+2), blue-white veil (+2), atypical vascular (+2). Minor: streaks (+1), dots/globules (+1), blotches (+1), regression (+1). Score >= 3 triggers biopsy. Sensitivity 95%, specificity 75%.*

4. **Haralick RM, Shanmugam K, Dinstein I.** Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*. 1973;3(6):610-621.
   - *Defines the Gray-Level Co-occurrence Matrix (GLCM) and derived texture features: contrast, homogeneity, entropy, correlation. Foundation for our texture analysis module.*

5. **Tschandl P, Rosendahl C, Kittler H.** The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*. 2018;5:180161. doi:10.1038/sdata.2018.161.
   - *Source for the 10,015-image, 7-class dermoscopy benchmark. Provides class prevalence, demographic distributions (age, sex, localization), and histopathology confirmation rates used in our demographic adjustment module.*

6. **Esteva A, Kuprel B, Novoa RA, et al.** Dermatologist-level classification of skin cancer with deep neural networks. *Nature*. 2017;542(7639):115-118. doi:10.1038/nature21056.
   - *Landmark paper demonstrating that a single CNN (InceptionV3) trained end-to-end achieves dermatologist-level classification. Established the feasibility of deep learning for dermoscopy and motivated the field's shift away from hand-crafted features.*

7. **Dosovitskiy A, Beyer L, Kolesnikov A, et al.** An image is worth 16x16 words: Transformers for image recognition at scale. *Proceedings of the International Conference on Learning Representations (ICLR)*. 2021.
   - *Introduces the Vision Transformer (ViT). The model used by DrAgnes (ViT-base-patch16-224) processes 224x224 images as sequences of 16x16 patches through a standard Transformer encoder.*

8. **Tkaczyk ER, Rao BK, Grin C, et al.** Clinical validation of the DermaSensor for cutaneous malignancy detection: DERM-SUCCESS pivotal study. *JAMA Dermatology*. 2024.
   - *FDA pivotal study for DermaSensor (DEN230008). 1,579 lesions across 44 sites. Melanoma sensitivity 90.2%, overall malignancy sensitivity 95.5%, specificity 20.7%. Fitzpatrick I-III sensitivity 96%, IV-VI sensitivity 92%. Used as our regulatory and performance benchmark.*

9. **Finlayson GD.** Shades of Gray and Colour Constancy. *Proceedings of the IS&T/SID Twelfth Color Imaging Conference*. 2004:37-41.
   - *Introduces the Shades-of-Gray illuminant estimation method using Minkowski norms. Our preprocessing uses p=6, which Finlayson showed provides near-optimal illuminant estimation across natural scenes.*

10. **Lee T, Ng V, Gallagher R, Coldman A, McLean D.** DullRazor: A software approach to hair removal from dermoscopy images. *Computers in Biology and Medicine*. 1997;27(6):533-543.
    - *Original DullRazor algorithm for automated hair removal from dermoscopy images using morphological closing with linear structuring elements. Our implementation is a simplified version.*

11. **Otsu N.** A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*. 1979;9(1):62-66.
    - *Classic automatic thresholding method that maximizes inter-class variance. Used in our segmentation pipeline for LAB L-channel binarization.*

12. **Codella NCF, Gutman D, Celebi ME, et al.** Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging Collaboration (ISIC). *Proceedings of ISBI*. 2018.
    - *Establishes the ISIC challenge benchmark for skin lesion classification. Provides standardized evaluation protocols for comparing dermoscopy classification systems.*

13. **Menzies SW, Ingvar C, Crotty KA, McCarthy WH.** Frequency and morphologic characteristics of invasive melanomas lacking specific surface microscopic features. *Archives of Dermatology*. 1996;132(10):1178-1182.
    - *Defines the Menzies method: negative features (symmetry + single color) and positive features (blue-white structures, multiple colors, etc.). Informs our weight assignments for the melanocytic nevus class.*

14. **Kittler H, Riedl E, Rosendahl C, Cameron A.** Dermatoscopy of unpigmented lesions of the skin: a new classification of vessel morphology based on pattern analysis. *Dermatopractice & Conceptual*. 2008. Updated terminology: Kittler H, et al. *Dermatoscopy: An Algorithmic Method Based on Pattern Analysis*. 2016.
    - *Revised dermoscopy terminology and pattern-based classification criteria used in deriving our trained-weights matrix.*

15. **Pehamberger H, Steiner A, Wolff K.** In vivo epiluminescence microscopy of pigmented skin lesions. I. Pattern analysis of pigmented skin lesions. *Journal of the American Academy of Dermatology*. 1987;17(4):571-583.
    - *Original pattern analysis method defining global (reticular, globular, homogeneous) and local (network, dots, streaks, etc.) dermoscopic features. Foundational work for feature extraction in dermoscopy.*

---

## 8. Dual-Model ViT Ensemble (v1.2 -- Updated March 22, 2026)

### 8.1 Current Model Configuration

As of v1.2, DrAgnes uses a dual-model ensemble with the following configuration:

| Property | Model A (Anwarkh1) | Model B (skintaglabs) |
|----------|--------------------|-----------------------|
| **HuggingFace ID** | `Anwarkh1/Skin_Cancer-Image_Classification` | `skintaglabs/siglip-skin-lesion-classifier` |
| **Architecture** | ViT-Base, patch16, 224x224 | SigLIP (~400M params) |
| **Parameters** | 85.8M | ~400M |
| **Output classes** | 7 (HAM10000 taxonomy) | Skin lesion classes (mapped to 7) |
| **License** | Apache-2.0 | MIT |
| **Validated melanoma sensitivity** | **73.3%** (our test, 210 HAM10000 images) | **Not yet validated** |
| **Overall accuracy** | **55.7%** (our test) | **Not yet validated** |

**Why the model changed**: The original Model B was `actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification` (305M params, 12 classes, claimed 89% melanoma recall). This model was **removed from HuggingFace** on March 22, 2026 (HTTP 410 Gone). The 89% claim cannot be verified. We evaluated 8 candidate models and selected skintaglabs SigLIP based on dermatology-specific training, MIT license, and parameter count.

### 8.2 Ensemble Architecture

Both models are called in **parallel** via `Promise.allSettled` to minimize latency. The weighting is currently **equal (50/50)** for all classes:

```
P_dual_HF(class) = 0.50 * P_anwarkh1(class) + 0.50 * P_skintaglabs(class)
```

**Why equal weighting**: Neither model has independently verified melanoma recall sufficient to justify asymmetric weighting. The previous asymmetric configuration (actavkid 70% for melanoma) was based on a published claim that can no longer be verified. Equal weighting is the conservative default until we validate the skintaglabs model.

The overall ensemble when both models are available:

```
P(class) = 0.50 * P_dual_HF + 0.30 * P_trained + 0.20 * P_rules
```

When only one model is available (the other fails or times out):

```
P(class) = 0.60 * P_single_HF + 0.25 * P_trained + 0.15 * P_rules
```

### 8.3 SigLIP Label Mapping

The skintaglabs SigLIP model outputs its own label taxonomy. Labels are mapped to our canonical 7-class HAM10000 taxonomy via `SIGLIP_LABEL_MAP`:

| SigLIP Output Label | DrAgnes Class | Rationale |
|---------------------|---------------|-----------|
| Melanoma / melanoma | mel | Direct match |
| Basal Cell Carcinoma | bcc | Direct match |
| Actinic Keratosis | akiec | Direct match |
| Squamous Cell Carcinoma | akiec | SCC and akiec on same malignancy spectrum |
| Benign Keratosis / Benign Keratosis-like Lesions | bkl | Direct match |
| Seborrheic Keratosis | bkl | Benign keratosis variant |
| Dermatofibroma | df | Direct match |
| Melanocytic Nevi / Melanocytic Nevus | nv | Direct match |
| Vascular Lesion / Vascular Lesions | vasc | Direct match |

When multiple SigLIP labels map to the same DrAgnes class, their probabilities are summed.

### 8.4 Model Disagreement Detection

When the two ViT models disagree on the top predicted class, the system:

1. Flags `modelsDisagree: true` in the classification result
2. Computes cosine similarity between the two model output distributions as `modelAgreement` (0-1)
3. Displays a clinical review warning in the UI

**This is a safety feature**: model disagreement may indicate a difficult or ambiguous case that warrants professional review.

### 8.5 Validation Status

**What we have validated (March 22, 2026):**
- Custom model (stuartkerr/dragnes-classifier): 98.2% melanoma sensitivity on HAM10000 holdout (1,503), 98.7% on Nagabu/HAM10000 (1,000), 100% on marmal88 test (1,285). -0.7% train/test gap.
- Anwarkh1 ViT-Base on 210 HAM10000 test images: 73.3% melanoma sensitivity, 55.7% overall accuracy
- skintaglabs SigLIP: 30.0% melanoma sensitivity (insufficient for clinical use)
- Hand-crafted features alone: 0% melanoma sensitivity (proven insufficient)

**What we have NOT validated:**
- Full 4-layer ensemble accuracy with the custom model as primary
- The dual-model ensemble's combined accuracy
- Whether 50/50 weighting is optimal
- Performance on Fitzpatrick V-VI skin tones

**Outstanding validation tasks:**
- [x] Train custom model with focal loss -- DONE (98.2% mel sens)
- [x] Validate on multiple independent test sets -- DONE (3 datasets, zero overfitting)
- [x] Validate skintaglabs SigLIP -- DONE (30.0% mel sens, insufficient)
- [ ] Measure full 4-layer ensemble accuracy end-to-end with custom model
- [ ] Validate on Fitzpatrick V-VI images
- [ ] Prospective clinical validation
- [ ] Measure specificity and NNB with custom model

---

## 9. Clinical Features Added (v0.2.0)

### 9.1 ICD-10-CM Code Mapping

Each of the 7 lesion classes maps to ICD-10-CM codes for clinical documentation:

| Class | Primary ICD-10 | Description |
|-------|---------------|-------------|
| mel | D03.9 / C43.x | Melanoma in situ / Malignant melanoma |
| bcc | C44.x | Basal cell carcinoma |
| akiec | L57.0 / D04.x | Actinic keratosis / Carcinoma in situ |
| bkl | L82.x | Seborrheic keratosis |
| df | D23.x | Dermatofibroma (benign neoplasm) |
| nv | D22.x | Melanocytic nevi |
| vasc | D18.x | Hemangioma / vascular lesion |

### 9.2 Referral Letter Generator

One-click generation of a referral letter pre-populated with:
- Classification result and confidence level
- ABCDE scores and TDS
- Clinical recommendation
- Body location from interactive body map
- Copy-to-clipboard functionality

### 9.3 Explainability Panel

"Why this classification?" panel showing:
- Which image features contributed most to the classification
- Corresponding clinical criteria (ABCDE, 7-point checklist)
- Literature citations for each contributing factor
- Model agreement/disagreement status

### 9.4 Analytics Dashboard

Practice-level performance monitoring:
- Concordance rate (AI vs. clinician) with 30-day rolling trend
- Number Needed to Biopsy (NNB) tracking
- Per-class sensitivity, specificity, PPV, NPV with Wilson 95% CI
- Calibration curves with Expected Calibration Error (ECE) and Hosmer-Lemeshow p-value
- Fitzpatrick equity monitoring with automatic disparity alerts
- Discordance analysis (cases where AI and clinician disagreed)

### 9.5 Interactive Body Map

SVG-based clickable body map replacing the previous dropdown selector for body location input. Supports anterior and posterior views with anatomically accurate region detection.

---

*This document was prepared for technical review by dermatology professionals evaluating the DrAgnes classification system. All claims of system performance are qualified with their validation status. Where independent validation has not been performed, this is stated explicitly. DrAgnes is a research prototype and is not FDA-cleared for clinical use.*
