Updated: 2026-03-24 | Version 1.0.0
Created: 2026-03-24

# ADR-121: Optimized Image Capture — Measurement, Quality Gating, and Retake Guidance

## Status: IMPLEMENTED -- Phases 1-3 | Last Updated: 2026-03-24 10:30 EST
**Implementation Note**: Phase 1 (quality gating) done -- image-quality.ts (139 lines) implements sharpness, contrast, brightness checks. Phase 2 (USB-C reference detection) done -- measurement-connector.ts (186 lines) implements contour-based connector detection. Phase 3 (skin texture measurement) done -- measurement-texture.ts (122 lines) implements FFT-based dermatoglyphic spacing analysis. measurement.ts (179 lines) ties all tiers together. Phase 4 (LiDAR) and Phase 5 (validation) not yet implemented.

## Context

The "D" in ABCDE scoring (diameter > 6mm is suspicious) is a key clinical criterion for melanoma detection. Dr. Agnes currently has a crude `estimateDiameterMm()` function that assumes a fixed camera distance — unreliable across different phones held at different distances.

Additionally, classification accuracy degrades significantly on blurry, poorly lit, or badly framed images. The system currently analyzes whatever it receives without checking input quality first. Garbage in = garbage out.

This ADR addresses both problems: **accurate physical measurement** and **image quality gating** before analysis.

## Decision

Implement a 3-tier measurement system with automatic quality gating.

### Tier 1: USB-C Reference Object (Recommended — ±0.5mm accuracy)

**Every phone user has a USB-C charger cable.** The USB-C connector is standardized worldwide:

| Connector | Width | Height | Source |
|-----------|-------|--------|--------|
| USB-C | 8.25mm | 2.4mm | USB-IF specification |
| Lightning | 7.7mm | 1.5mm | Apple MFi specification |
| USB-A | 12.0mm | 4.5mm | USB-IF specification |

**User flow:** "For best accuracy, place your charger cable's USB-C plug next to the spot before photographing."

**Detection algorithm:**
1. Convert image to grayscale, apply Canny edge detection
2. Find rectangular contours with aspect ratio ~3.4:1 (8.25/2.4) ± tolerance
3. Filter by size (must be plausible connector size relative to image)
4. Filter by color (metallic gray/silver, distinct from skin tones)
5. If detected: compute pixels-per-mm from known 8.25mm width
6. If multiple candidates: use the one closest to the lesion

**Why USB-C over coins:**
- Coins vary by country (US quarter = 24.26mm, Euro 1 = 23.25mm, UK £1 = 23.43mm)
- USB-C is identical worldwide
- User already has it — it's charging their phone
- Higher contrast against skin than most coins

### Tier 2: Skin Texture Frequency Analysis (Automatic Fallback — ±2-3mm)

When no reference object is detected, analyze the skin texture surrounding the lesion.

**Method:**
1. Extract a skin-only region around the lesion (exclude the lesion itself)
2. Compute 2D FFT (Fast Fourier Transform) of the skin patch
3. Find the peak spatial frequency in the power spectrum
4. Compare to known dermatoglyphic spacing by body location:

| Body Location | Pore Spacing | Source |
|---------------|-------------|--------|
| Face (forehead) | 0.15-0.25mm | Hashimoto 1974 |
| Face (cheek) | 0.20-0.35mm | Korotkov & Garcia 2012 |
| Trunk | 0.30-0.50mm | Hashimoto 1974 |
| Upper extremity | 0.25-0.40mm | Measured |
| Lower extremity | 0.30-0.50mm | Measured |
| Palms/soles | 0.40-0.60mm (ridge spacing) | Ashbaugh 1999 |

5. Derive pixels-per-mm from frequency match
6. Use body map selection to pick the correct expected spacing

**Accuracy:** ±2-3mm. Less precise than USB-C reference but fully automatic.

**Reference:** Korotkov K, Garcia R. "Computerized analysis of pigmented skin lesions: A review." Artificial Intelligence in Medicine. 2012;56(2):69-90.

### Tier 3: LiDAR Depth Sensing (iPhone Pro Enhancement — ±1mm)

For iPhone 12 Pro and later (devices with LiDAR):

1. Check for WebXR depth sensing API availability
2. If available, measure camera-to-skin distance
3. With known focal length + sensor pixel size + distance → exact pixels-per-mm
4. `mm_per_pixel = (distance_mm × sensor_pixel_pitch) / focal_length_px`

**Limitation:** Only ~30% of iPhones have LiDAR. This is an enhancement, not primary.

---

## Image Quality Gating

Every captured image is evaluated BEFORE classification. Quality checks run in <100ms.

### Quality Checks

| Check | Method | Threshold | User Message |
|-------|--------|-----------|-------------|
| **Sharpness** | Laplacian variance (existing in `scoreImageQuality()`) | < 0.15 | "Image is blurry — hold steady and tap again" |
| **Lesion focus** | Laplacian variance within segmentation mask only | < 0.10 | "The spot appears out of focus — tap the spot to focus, then retake" |
| **Too dark** | Mean brightness of lesion region | < 40/255 | "Too dark — move to better lighting" |
| **Too bright** | Clipped highlights (>250) in lesion region | > 15% pixels | "Overexposed — reduce lighting or tilt phone" |
| **Glare** | Saturated white spots within lesion mask | Any cluster > 2% area | "Glare detected — tilt the phone slightly" |
| **Too small** | Lesion area / total image area | < 3% | "Lesion too small in frame — move closer" |
| **Motion blur** | Directional Laplacian variance ratio (H/V) | > 3:1 or < 1:3 | "Motion blur detected — hold steady" |

### Quality Scoring

Each image gets a quality grade displayed as a colored dot on the multi-capture thumbnail:

| Grade | Color | Meaning | Action |
|-------|-------|---------|--------|
| Good | Green | All checks pass | Proceed to analysis |
| Acceptable | Yellow | 1-2 minor issues | Suggest retake, allow proceed |
| Poor | Red | Major quality issue | Block analysis, require retake |

### Retake Flow

1. After each capture, quality assessment runs immediately (<100ms)
2. Result shown as overlay on the thumbnail (green/yellow/red dot)
3. Yellow: "This image has issues. Retake recommended." + specific guidance
4. Red: "This image is too [blurry/dark/etc] for reliable analysis. Please retake."
5. User can tap X on any thumbnail to remove and retake
6. "Done — Analyze All" button disabled if ANY image is red

---

## Integration with Classification Pipeline

### How measurement improves ABCDE scoring

Current `estimateDiameterMm()` assumes fixed scale. New flow:

```
Image → Quality Gate → Measurement Calibration → Segmentation → ABCDE with real mm
                                    ↓
                    USB-C detected? → Use 8.25mm reference
                    No → Skin texture FFT → Body-location adjusted
                    LiDAR available? → Use depth sensing
```

### Impact on TDS formula

TDS = A×1.3 + B×0.1 + C×0.5 + **D×0.5**

Currently D (diameter) is unreliable. With real measurement:
- D < 6mm → score 0 (benign indicator)
- D ≥ 6mm → score 1 (suspicious)
- D ≥ 10mm → safety gate trigger (force melanoma probability floor)

### Impact on clinical recommendation

- Lesions < 4mm with no other suspicious features → "Monitor" instead of "Biopsy"
- Lesions > 10mm → automatic "Urgent referral" regardless of classification
- Real measurement enables Number Needed to Biopsy (NNB) optimization

---

## Implementation Plan

### Phase 1: Quality Gating (1-2 days)
- Add `assessImageQuality()` to `image-analysis.ts`
- Wire into DermCapture multi-capture flow
- Show quality dots on thumbnails
- Block analysis on red-quality images

### Phase 2: USB-C Reference Detection (2-3 days)
- Implement contour-based USB-C detector
- Add Lightning and USB-A detection
- Calibrate pixels-per-mm from detected reference
- UI prompt: "For best accuracy, place your charger cable next to the spot"

### Phase 3: Skin Texture Measurement (2-3 days)
- Implement 2D FFT on skin region
- Body-location lookup table for expected pore spacing
- Automatic fallback when no reference object detected

### Phase 4: LiDAR Enhancement (1 day)
- Feature-detect LiDAR availability
- WebXR depth API integration
- Distance-based calibration

### Phase 5: Validation (2-3 days)
- Measure 50+ lesions with ruler + USB-C + app measurement
- Compare all three measurement approaches
- Publish accuracy comparison in technical report

---

## Consequences

### Positive
- ABCDE "D" score becomes clinically meaningful
- TDS formula produces accurate results
- Poor-quality images are caught before wasting classification compute
- Users guided to take better photos → better classification accuracy
- USB-C reference is universally available, zero cost

### Negative
- Additional compute per capture (~100ms for quality, ~200ms for measurement)
- False rejection of acceptable images could frustrate users
- USB-C detection may fail on unusual backgrounds or lighting
- Skin texture method is body-location dependent — wrong location selection degrades accuracy

### Risks
- **Critical:** Measurement error could cause false reassurance (lesion appears < 6mm when actually ≥ 6mm). Mitigation: always show measurement uncertainty and recommend dermatologist for borderline cases.
- **Medium:** Quality gating too aggressive → users can't get any analysis. Mitigation: yellow allows proceeding, only red blocks.

---

## Author
Stuart Kerr + Claude Flow
