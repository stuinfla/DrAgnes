Updated: 2026-03-24 12:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-121: Optimized Image Capture -- Measurement, Quality Gating, and Retake Guidance

## Status: IMPLEMENTED -- Phases 0-3 | Last Updated: 2026-03-24 10:30 EST
**Implementation Note**: Phase 0 (quality gating) done -- image-quality.ts. Phase 1 (USB-C reference) done -- measurement-connector.ts. Phase 2 (skin texture FFT) done -- measurement-texture.ts. Phase 3 (measurement pipeline integration) done -- measurement.ts (179 lines) integrates all measurement tiers with body-location lookup. Phase 4 (LiDAR) not started. Phase 5 (validation with ruler comparison) not started.

## Context

The "D" in the ABCDE melanoma screening rule stands for **Diameter** -- lesions larger than 6mm are clinically suspicious. This is a hard numerical threshold used by dermatologists worldwide and is a key input to the Total Dermoscopy Score (TDS) formula that drives Dr. Agnes risk classification.

**The current implementation is unreliable.** There are two diameter estimation functions, and both depend on assumptions that do not hold for consumer phone cameras:

1. **`computeDiameter()` in `abcde.ts` (line 229)** -- Assumes a fixed calibration constant of 40 pixels per mm at 10x magnification. This only works with a specific dermoscope model at a specific distance.

2. **`estimateDiameterMm()` in `image-analysis.ts` (line 1914)** -- Assumes the image represents the full field of view of a 25mm dermoscope. When the user is taking a photo with their phone camera (no dermoscope), this assumption is completely wrong.

**Why this matters for patient safety:** A 9mm melanoma photographed close-up could be measured as 4mm by the current code, which would suppress the diameter risk flag and lower the TDS score. The user sees "low risk" when they should see "see a doctor." This is the single most dangerous failure mode in the ABCDE pipeline -- false reassurance from a measurement error.

**Constraint:** Users do not have rulers, reference stickers, dermoscopes, or special hardware. Solutions must work with the phone the user already has, or at most with objects they already own (like their phone charger cable).

### Current Code Under Replacement

```typescript
// abcde.ts, line 229 -- fixed calibration constant
function computeDiameter(seg: SegmentationMask, magnification: number): number {
    const diagonalPx = Math.sqrt(bb.w ** 2 + bb.h ** 2);
    const pxPerMm = 4 * magnification; // arbitrary, wrong for phones
    return Math.round((diagonalPx / pxPerMm) * 10) / 10;
}

// image-analysis.ts, line 1914 -- assumes full dermoscope field of view
export function estimateDiameterMm(
    areaPixels: number, imageWidth: number, magnification: number = 10
): number {
    const fieldOfViewMm = 25 / (magnification / 10); // only valid for dermoscopes
    const pxPerMm = imageWidth / fieldOfViewMm;
    const radiusPx = Math.sqrt(areaPixels / Math.PI);
    return Math.round((2 * radiusPx) / pxPerMm * 10) / 10;
}
```

Both functions produce a single `diameterMm` number that feeds directly into:
- `ABCDEScores.diameterMm` (types.ts, line 107)
- The TDS structure score: `diameterMm > 6 ? 1 : 0` (abcde.ts, line 49)
- `computeRiskLevel()` diameter check (image-analysis.ts)
- Consumer translation layer risk messaging

## Decision

Implement a **layered measurement strategy** with four approaches in priority order. The system attempts each approach in order and uses the most accurate one available. When multiple approaches succeed, their estimates are cross-validated to produce a confidence interval rather than a single point estimate.

### Approach 0: USB-C / Connector Reference (Recommended -- User-Guided)

**Rationale:** The most accurate phone-based measurement uses a reference object of known size. Rather than requiring a specialized sticker or ruler, we use something every smartphone user already has: **their charger cable.**

**How it works:**
- UI prompts: "For best accuracy, place your charger cable's USB-C plug next to the spot."
- The user photographs the lesion with the USB-C connector tip visible in frame.
- The app detects the connector using template matching or contour detection (distinctive rectangular shape with rounded corners, high contrast metal against skin).
- USB-C plug dimensions are standardized worldwide: **8.25mm wide x 2.4mm tall.**
- Known physical width gives exact pixels-per-mm calibration.

**Connector detection and dimensions:**

| Connector Type | Width (mm) | Height (mm) | Detection Method |
|----------------|-----------|-------------|------------------|
| USB-C          | 8.25      | 2.4         | Rounded rectangle, metallic color |
| Lightning      | 7.7       | 1.5         | Rounded rectangle, thinner profile |
| USB-A          | 12.0      | 4.5         | Sharp rectangle, larger |

**Detection pipeline:**
1. Convert ROI to grayscale, apply Canny edge detection.
2. Find contours, filter for rectangles with aspect ratio 3.0-4.0 (USB-C) or 4.5-5.5 (Lightning).
3. Verify metallic color signature in the bounding region (high luminance, low saturation).
4. Measure the detected rectangle width in pixels. Divide by known physical width to get pixels-per-mm.
5. If multiple connectors detected, use the one closest to the lesion.

**Expected accuracy:** +/- 0.5mm (limited primarily by the user's ability to place the connector flat and parallel to the camera sensor plane).

**Failure modes:**
- Connector not detected (angled, partially occluded, out of focus) -- fall through to Approach 1.
- Connector at an angle to camera -- perspective distortion introduces ~5-10% error at 15-degree tilt. Mitigate by checking the detected aspect ratio against the known aspect ratio and warning if they diverge significantly.

**This is the recommended approach** because it gives the best accuracy without any special hardware, and the UX is simple: everyone has a charger cable within arm's reach of their phone.

### Approach 1: Skin Texture Frequency Analysis (Primary Automatic Fallback)

**Rationale:** When no reference object is present, human skin itself contains measurement information. Dermatoglyphic patterns (pores, fine wrinkles, sulci) have characteristic spatial frequencies that vary by body location but are well-studied in the dermatology literature.

**How it works:**
- Extract a region of healthy skin surrounding the lesion (using the segmentation mask boundary plus a margin).
- Compute the 2D Fast Fourier Transform (FFT) of this skin patch.
- Identify the dominant frequency peak corresponding to pore/ridge spacing.
- Convert the peak frequency in cycles-per-pixel to pixels-per-mm using known dermatoglyphic spacing for the selected body location.

**Body location adjustment (pore spacing from dermatology literature):**

| Body Location (`BodyLocation` enum) | Expected Pore Spacing (mm) | Confidence |
|--------------------------------------|---------------------------|------------|
| `head` (face)                        | 0.2 - 0.4                | High       |
| `neck`                               | 0.3 - 0.5                | Medium     |
| `trunk` (chest, back, abdomen)       | 0.4 - 0.7                | Medium     |
| `upper_extremity` (arms, hands)      | 0.3 - 0.5                | High       |
| `lower_extremity` (legs, feet)       | 0.3 - 0.5                | Medium     |
| `palms_soles`                        | 0.5 - 0.8 (ridge spacing)| High       |
| `genital`                            | 0.3 - 0.5                | Low        |
| `unknown`                            | 0.3 - 0.5 (median)       | Low        |

Reference: Korotkov & Garcia 2012 (dermatoglyphic frequency analysis); Hashimoto 1974 (pore spacing by body site).

**Algorithm:**
1. Extract skin-only ROI: pixels within 50-150px of the lesion boundary where `mask === 0`.
2. Convert to grayscale. Apply Hanning window to reduce spectral leakage.
3. Compute 2D FFT using a pure-TypeScript radix-2 implementation (no dependencies).
4. Compute power spectrum. Mask out DC component and very low frequencies (< 2 cycles across patch).
5. Find the radial frequency peak in the expected range for the body location.
6. `pixelsPerMm = peakFrequencyCyclesPerPixel / expectedPoreSpacingMm`.
7. Apply `pixelsPerMm` to the segmentation bounding box to get diameter.

**Expected accuracy:** +/- 2-3mm. Adequate for the 6mm clinical threshold when the lesion is clearly above or below it. Ambiguous for lesions in the 4-8mm range.

**Failure modes:**
- Skin too smooth (young skin, heavily moisturized) -- no clear frequency peak. Detectable: peak power below noise floor threshold. Fall through to Approach 2 or 3.
- Hair obscures texture -- partially mitigated by the existing `removeHair()` preprocessing step.
- Lesion fills the entire image (no surrounding skin) -- no ROI for analysis. Fall through.
- Wrong body location selected -- the `BodyLocation` from the body map picker feeds directly into the spacing table. If the user selects "trunk" but photographs their face, the estimate will be off by up to 2x.

### Approach 2: LiDAR Depth Sensing (Enhancement -- iPhone Pro / iPad Pro Only)

**Rationale:** Devices with LiDAR sensors can measure the exact camera-to-skin distance. Combined with the camera's known focal length and sensor dimensions (available via EXIF or device model lookup), this gives an exact pixels-per-mm calculation.

**How it works:**
- Feature detection: Check for LiDAR via `navigator.xr?.isSessionSupported('immersive-ar')` or the `XRDepthInformation` API.
- If available, request a short ARKit/WebXR session to capture depth data alongside the photo.
- Extract the depth value at the center of the lesion.
- `pixelsPerMm = (focalLengthPx) / (distanceMm)` where `focalLengthPx = (focalLengthMm / sensorWidthMm) * imageWidthPx`.

**Device coverage:**

| Device | LiDAR | Depth API |
|--------|-------|-----------|
| iPhone 12 Pro / Pro Max+ | Yes | ARKit, limited WebXR |
| iPad Pro (2020+) | Yes | ARKit, limited WebXR |
| iPhone 12, 13, 14, 15, 16 (non-Pro) | No | -- |
| Android (select models) | ToF sensor | ARCore Depth API |

**Expected accuracy:** +/- 1mm. The LiDAR depth resolution is ~1mm at 30cm distance, and camera intrinsics are known precisely.

**Failure modes:**
- Device does not have LiDAR -- feature detection returns false; skip entirely.
- WebXR depth API not available in Safari/WebView -- fall through to ARKit native bridge if the app is wrapped in a native container, otherwise skip.
- Very close distance (< 10cm) -- LiDAR minimum range. Warn user to hold phone further away.
- Shiny/wet skin reflects LiDAR differently -- depth reading may be noisy. Use median of depth values across the lesion region.

### Approach 3: Multi-Photo Parallax (Uses Existing Multi-Capture)

**Rationale:** Dr. Agnes already supports multi-image capture (`classifyMultiImage` in `multi-image.ts`). Two photos from slightly different angles contain parallax information that enables depth estimation, and modern phones have gyroscope/accelerometer data that can quantify the camera motion between shots.

**How it works:**
1. During multi-capture mode, the user takes 2-3 photos while slightly shifting their angle (the UI already encourages this for consensus classification).
2. Detect matching feature points between image pairs using ORB-style descriptors (oriented FAST keypoints + binary descriptors, implementable in pure TypeScript).
3. Compute the fundamental matrix from point correspondences.
4. Read accelerometer/gyroscope data from the `DeviceMotionEvent` API to estimate baseline distance (camera translation between shots).
5. Triangulate depth from disparity + baseline.
6. With depth known, apply the same focal-length calculation as Approach 2.

**Expected accuracy:** +/- 1-2mm, depending on the angle difference between shots. At least 5-10 degrees of rotation is needed for meaningful parallax.

**Failure modes:**
- User holds phone perfectly still between shots -- no parallax, no depth. Detectable: accelerometer reports near-zero motion. UI can prompt: "Try tilting your phone slightly between photos."
- Insufficient feature points on smooth skin -- falls back to Approach 1.
- `DeviceMotionEvent` permission denied (iOS 13+ requires explicit permission) -- without baseline distance, only relative depth is available, not absolute. Can still improve relative sizing.

## Measurement Fusion and Confidence

When multiple approaches produce estimates, the system does not simply pick one. Instead:

```
MeasurementResult {
    diameterMm: number;           // Best estimate
    confidenceInterval: [number, number]; // 95% CI in mm
    method: "usbc-ref" | "connector-ref" | "skin-texture" | "lidar" | "parallax" | "legacy";
    methodsAttempted: string[];
    methodsFailed: string[];
    confidence: "high" | "medium" | "low";  // Drives UI and safety messaging
    rawEstimates: Array<{ method: string; diameterMm: number; uncertainty: number }>;
}
```

**Fusion rules:**
1. If USB-C/connector reference succeeds, use it as primary (highest accuracy).
2. If LiDAR succeeds, use it as primary (or to cross-validate connector reference).
3. If both connector + LiDAR agree within 1mm, confidence = "high".
4. If skin texture is the only method, confidence = "medium" when the FFT peak is clear, "low" when it is marginal.
5. If parallax provides an estimate, use it to cross-validate other methods.
6. If no method succeeds, fall back to the existing legacy calculation and mark confidence = "low" with a UI warning.

**Critical safety rule:** When confidence is "low" and the estimated diameter is near the 6mm threshold (4-8mm range), the UI must display: "Diameter measurement is uncertain. If this lesion is growing or concerns you, see a dermatologist regardless of size."

## Consequences

### What This Enables

1. **Reliable ABCDE "D" scoring** -- The diameter component of TDS becomes trustworthy for consumer phone photos, not just dermoscope images.
2. **Longitudinal size tracking** -- With calibrated measurements, the app can detect growth between visits (evolution scoring). A lesion that grew from 4mm to 7mm over 3 months is clinically significant even if both measurements individually seem "low risk."
3. **Confidence-aware UI** -- Instead of showing a single number ("5.2mm"), the app can show "approximately 5mm (4-6mm)" with color coding based on confidence level.
4. **Research data quality** -- Calibrated measurements in the feedback/analytics pipeline are far more useful for research than the current arbitrary pixel-ratio numbers.

### What Could Go Wrong

1. **False precision leading to false reassurance.** A measurement of "5.8mm +/- 2mm" is really "4-8mm" which straddles the 6mm threshold. If the UI shows just "5.8mm" with a green checkmark, the user may be falsely reassured. **Mitigation:** Always show the confidence interval, and when the 6mm threshold falls within the interval, flag it as "uncertain -- monitor closely."

2. **Skin texture analysis fails on certain populations.** Very dark skin (Fitzpatrick V-VI) has different dermatoglyphic contrast. Very young skin has finer pores. Elderly skin has wider spacing. **Mitigation:** Body location table should be cross-referenced with age bracket when available. Validate specifically on Fitzpatrick V-VI test images.

3. **Users hold connector at an angle.** If the USB-C plug is not parallel to the camera sensor plane, perspective foreshortening makes it appear narrower, leading to an over-estimated pixels-per-mm and an under-estimated lesion diameter. A 15-degree tilt introduces ~3.4% error. A 30-degree tilt introduces ~13.4% error. **Mitigation:** Check detected aspect ratio vs. known aspect ratio. If the ratio deviates by more than 10%, warn the user to hold the cable flat against the skin.

4. **Parallax requires user cooperation.** The user must actually move between shots. If they take two nearly-identical photos, no depth information is recovered. **Mitigation:** Measure inter-frame motion from accelerometer. If below threshold, prompt the user.

5. **Implementation complexity.** A 2D FFT in pure TypeScript is non-trivial. Feature matching (ORB) is even more complex. Both must run at interactive speed on phone hardware. **Mitigation:** Phase the implementation. Start with Approach 0 (connector reference) which requires only contour detection, and Approach 1 (skin texture) which requires FFT. Defer Approach 3 (parallax) to a later release.

### What We Deliberately Do NOT Do

- **We do not require special hardware** (dermoscope adapters, reference stickers, coin placement). The app works with just a phone, or a phone + charger cable.
- **We do not use ML-based size estimation** (training a model to guess size from photos). This would require a large dataset of photos with ground-truth measurements, which we do not have, and would be a black box with unknown failure modes.
- **We do not claim clinical-grade precision.** The measurement feeds into a screening tool, not a diagnostic one. The goal is to correctly flag lesions that are "probably above 6mm" vs. "probably below 6mm," not to replace a calibrated dermatoscope.

## Integration with Existing Pipeline

### Where Measurement Plugs In

```
DermCapture (photo + body location)
    |
    v
[NEW] MeasurementEngine.calibrate(imageData, bodyLocation, deviceInfo)
    |
    +--> Approach 0: detectConnectorReference(imageData)
    |        |-> USB-C detected? -> pixelsPerMm = connectorWidthPx / 8.25
    |
    +--> Approach 1: analyzeSkintexture(imageData, segmentation, bodyLocation)
    |        |-> FFT peak found? -> pixelsPerMm = peakFreq / poreSpacing
    |
    +--> Approach 2: getLiDARDepth(imageData)  [if available]
    |        |-> depth known? -> pixelsPerMm = focalLengthPx / depthMm
    |
    +--> Approach 3: parallaxFromMultiCapture(images, motionData)  [if multi-cap]
    |        |-> disparity found? -> pixelsPerMm from triangulation
    |
    v
MeasurementResult { diameterMm, confidenceInterval, method, confidence }
    |
    v
computeABCDE(imageData, measurement)   <-- signature change
    |
    +--> ABCDEScores.diameterMm = measurement.diameterMm
    +--> ABCDEScores.diameterConfidence = measurement.confidence     <-- new field
    +--> ABCDEScores.diameterMethod = measurement.method             <-- new field
    +--> TDS structure score uses confidence-adjusted threshold
    |
    v
computeRiskLevel() / consumer translation
    |
    +--> If diameterConfidence === "low" && 4 < diameterMm < 8:
    |        "Diameter is uncertain. See a doctor if concerned."
    +--> If diameterConfidence === "high" && diameterMm > 6:
             "This lesion appears larger than 6mm, which is a flag."
```

### Type Changes Required

```typescript
// New types in types.ts
type MeasurementMethod =
    | "usbc-ref" | "lightning-ref" | "usba-ref"  // Approach 0
    | "skin-texture"                              // Approach 1
    | "lidar"                                     // Approach 2
    | "parallax"                                  // Approach 3
    | "legacy";                                   // Current fallback

type MeasurementConfidence = "high" | "medium" | "low";

interface MeasurementResult {
    diameterMm: number;
    confidenceInterval: [number, number];
    method: MeasurementMethod;
    confidence: MeasurementConfidence;
    methodsAttempted: MeasurementMethod[];
    methodsFailed: MeasurementMethod[];
    rawEstimates: Array<{
        method: MeasurementMethod;
        diameterMm: number;
        uncertaintyMm: number;
    }>;
    pixelsPerMm: number;
}

// Extended ABCDEScores
interface ABCDEScores {
    // ... existing fields ...
    diameterMm: number;
    diameterConfidence: MeasurementConfidence;  // NEW
    diameterMethod: MeasurementMethod;          // NEW
    diameterCI: [number, number];               // NEW: 95% confidence interval
}
```

### Files Modified

| File | Change |
|------|--------|
| `types.ts` | Add `MeasurementMethod`, `MeasurementConfidence`, `MeasurementResult` types. Extend `ABCDEScores`. |
| `image-analysis.ts` | Replace `estimateDiameterMm()` with new `MeasurementEngine` class or module. |
| `abcde.ts` | Change `computeDiameter()` to accept `MeasurementResult` instead of computing internally. Update `computeABCDE()` signature. |
| `clinical-baselines.ts` | Add confidence-adjusted TDS logic (when measurement confidence is "low," D component uses a cautious default). |
| `index.ts` | Export new measurement types and functions. |
| `consumer-translation.ts` | Add diameter-uncertainty messaging. |
| `DermCapture.svelte` | Add UI prompt for USB-C reference. Show measurement confidence. |
| `ClassificationResult.svelte` | Display confidence interval instead of bare number. |
| `DrAgnesPanel.svelte` | Pass measurement data through pipeline. |

### New Files

| File | Purpose |
|------|---------|
| `measurement.ts` | `MeasurementEngine` class: orchestrates all four approaches, fusion logic. |
| `measurement-connector.ts` | Approach 0: USB-C / connector detection and calibration. |
| `measurement-texture.ts` | Approach 1: FFT-based skin texture frequency analysis. |
| `measurement-lidar.ts` | Approach 2: LiDAR / WebXR depth sensing. |
| `measurement-parallax.ts` | Approach 3: Multi-photo parallax depth estimation. |
| `fft.ts` | Pure TypeScript 2D FFT implementation (radix-2 Cooley-Tukey). |

## Bounded Context Diagram (DDD)

The measurement subsystem is a new bounded context that sits between Capture and Analysis:

```
+===========================================================================+
|  DrAgnes Bounded Contexts                                                 |
+===========================================================================+

+-----------------------+
|   CAPTURE CONTEXT     |
|   (DermCapture.svelte)|
|                       |
|  Responsibilities:    |
|  - Camera/file input  |
|  - Body location pick |
|  - Multi-capture      |
|  - Image type detect  |
|  - Device model ID    |
|  - Retake flow        |
|                       |
|  Emits:               |
|  - ImageData          |
|  - BodyLocation       |
|  - DeviceModel        |
|  - MotionData (gyro)  |
+-----------------------+
         |
         v
+---------------------------+
|   QUALITY GATE CONTEXT    |
|   (image-quality.ts)      |
|                           |
|  Responsibilities:        |
|  - Sharpness check        |
|  - Lesion-region focus    |
|  - Lighting assessment    |
|  - Framing check          |
|  - Motion blur detect     |
|  - Quality score per      |
|    thumbnail (G/Y/R)      |
|  - Retake guidance        |
|                           |
|  Emits:                   |
|  - ImageQualityAssessment |
|  - actionableMessages[]   |
|  - pass/fail per check    |
|                           |
|  Gate rule:               |
|  - CRITICAL fail: prompt  |
|    retake (user can       |
|    bypass with "Analyze   |
|    anyway")               |
|  - WARNING: show yellow,  |
|    allow proceed          |
|  - PASS: green, proceed   |
+---------------------------+
         |
         | (quality-gated image)
         v
+---------------------------+       +---------------------------+
|   MEASUREMENT CONTEXT     |       |  PREPROCESSING CONTEXT    |
|   (measurement.ts)        |       |  (preprocessing.ts)       |
|                           |       |                           |
|  Responsibilities:        |       |  Responsibilities:        |
|  - Scale calibration      |<----->|  - Color normalization    |
|  - Connector detection    |       |  - Hair removal           |
|    (USB-C/Lightning/      |       |  - Segmentation           |
|     USB-A)                |       |  - Resize                 |
|  - Skin texture FFT       |       |                           |
|  - LiDAR depth sensing    |       |  (Measurement needs       |
|  - Parallax estimation    |       |   segmentation mask for   |
|  - Measurement fusion     |       |   skin ROI extraction)    |
|  - Confidence scoring     |       +---------------------------+
|                           |                   |
|  Emits:                   |                   |
|  - MeasurementResult      |                   |
|  - pixelsPerMm            |                   |
+---------------------------+                   |
         |                                      |
         v                                      v
+---------------------------+        +----------------------+
|  ANALYSIS CONTEXT         |        |  CLASSIFICATION      |
|  (abcde.ts,               |        |  CONTEXT             |
|   image-analysis.ts)      |        |  (classifier.ts,     |
|                           |        |   trained-weights.ts, |
|  Responsibilities:        |        |   hf-classifier.ts)  |
|  - Asymmetry scoring      |        |                      |
|  - Border scoring         |        |  - CNN inference     |
|  - Color scoring          |        |  - ViT ensemble      |
|  - Diameter scoring  <----+--------+  - Feature classify  |
|    (from Measurement)     |        |  - Multi-image       |
|  - Structure detection    |        |    consensus         |
|  - Evolution tracking     |        +----------------------+
|  - TDS computation        |                 |
+---------------------------+                 |
                   |                          |
                   v                          v
+-----------------------------------------------+
|  CLINICAL DECISION CONTEXT                    |
|  (clinical-baselines.ts,                      |
|   consumer-translation.ts)                    |
|                                               |
|  Responsibilities:                            |
|  - TDS risk level                             |
|  - 7-point checklist                          |
|  - Confidence stratification                  |
|  - Consumer-friendly translation              |
|  - Safety gate messaging                      |
|  - Diameter-uncertainty warnings              |
|  - Image-quality disclaimers                  |
+-----------------------------------------------+
                        |
                        v
+---------------------------+
|  PRESENTATION CONTEXT     |
|  (Svelte components)      |
|                           |
|  - ClassificationResult   |
|  - DrAgnesPanel           |
|  - BodyMap                |
|  - Quality indicators     |
|    (green/yellow/red)     |
|  - Confidence display     |
|  - Measurement method     |
|    indicator              |
|  - Retake prompts         |
+---------------------------+
```

### Anti-Corruption Layer

The Measurement Context communicates with the Analysis Context via a single interface:

```typescript
// Anti-corruption layer: Analysis Context only sees this
interface CalibratedDiameter {
    diameterMm: number;
    confidence: MeasurementConfidence;
    method: MeasurementMethod;
    confidenceInterval: [number, number];
}
```

The Analysis Context does not know or care how the measurement was obtained. It receives a `CalibratedDiameter` and uses it. This means:
- New measurement approaches can be added without changing the analysis code.
- The legacy `computeDiameter()` function can be wrapped to emit a `CalibratedDiameter` with `method: "legacy"` and `confidence: "low"` during migration.

### Data Flow (Sequence)

```
User       DermCapture   QualityGate    MeasurementEngine  Preprocessing  ABCDE
  |             |              |                |                |           |
  |--photo----->|              |                |                |           |
  |             |--imageData-->|                |                |           |
  |             |              |                |                |           |
  |             |              |--assess------->|                |           |
  |             |              |  sharpness,    |                |           |
  |             |              |  lighting,     |                |           |
  |             |              |  framing,      |                |           |
  |             |              |  motion blur   |                |           |
  |             |              |                |                |           |
  |             |<--quality----|                |                |           |
  |             |  assessment  |                |                |           |
  |             |  (G/Y/R)    |                |                |           |
  |             |              |                |                |           |
  |  [if RED]:  |              |                |                |           |
  |<--"Retake?"-|              |                |                |           |
  |--"Analyze"->|  (user can bypass)           |                |           |
  |             |              |                |                |           |
  |             |--imageData + bodyLocation + deviceModel------->|           |
  |             |              |                |                |           |
  |             |              |                |--get mask----->|           |
  |             |              |                |<--segmention---|           |
  |             |              |                |                |           |
  |             |              |                |--connector?--->|           |
  |             |              |                |--skin FFT?---->|           |
  |             |              |                |--LiDAR?------->|           |
  |             |              |                |--parallax?---->|           |
  |             |              |                |                |           |
  |             |              |                |--fuse--------->|           |
  |             |              |                |                |           |
  |             |              |                |--MeasurementResult-------->|
  |             |              |                |                |           |
  |             |              |                |                | ABCDE with|
  |             |              |                |                | calibrated|
  |             |              |                |                | diameter  |
  |             |              |                |                |           |
  |<-----------result with quality score + confidence interval + method-----|
```

## Image Quality Assessment Before Analysis

Accurate measurement and classification both depend on high-quality input images. The principle is simple: **garbage in, garbage out.** Rather than silently analyzing a blurry, dark, or poorly-framed photo and producing unreliable results, the app should evaluate each captured image BEFORE running the classification pipeline and guide the user to retake when quality is insufficient.

### Quality Checks

#### 1. Sharpness Check (Laplacian Variance)

The existing `scoreImageQuality()` function in `multi-image.ts` already computes a Laplacian-based sharpness score. This should be extracted into a standalone utility and applied to every image, not just multi-capture sessions.

**Method:** Convolve with the Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]`, compute variance of the output. Higher variance = sharper image.

**Thresholds:**
- Variance > 500: sharp (green)
- Variance 200-500: acceptable (yellow) -- "Image is slightly soft but usable"
- Variance < 200: blurry (red) -- "Image is blurry -- hold steady and tap again"

#### 2. Lesion-Region Focus Detection

Overall image sharpness is not sufficient. A photo can be sharp at the edges but out of focus on the lesion in the center. After segmentation produces a mask, compute the Laplacian variance only within the lesion ROI (where `mask === 1`).

**Method:** Apply the same Laplacian variance, but only accumulate pixels inside the bounding box where the mask is active.

**Why this matters:** Autofocus on phones sometimes locks onto the background or the user's finger rather than the skin. Checking lesion-specific focus catches this failure mode.

#### 3. Lighting Assessment

| Condition | Detection Method | Threshold | User Message |
|-----------|-----------------|-----------|--------------|
| **Too dark** | Mean brightness of image | < 60 (on 0-255 scale) | "Too dark -- move to better lighting" |
| **Overexposed** | Percentage of pixels with R,G,B all > 250 | > 10% of image | "Too bright -- move away from direct light" |
| **Uneven lighting** | Divide image into 4x4 grid, compute brightness variance across grid cells | Variance > 2000 | "Uneven lighting -- try to light the area evenly" |
| **Glare / reflection** | Saturated white regions (R,G,B > 245) within the lesion mask | > 5% of lesion pixels | "Glare detected -- tilt the phone slightly" |

Glare within the lesion is the most damaging quality issue because it destroys the color and texture information that the classifier depends on. A small specular highlight is tolerable; a large glare region makes the image useless for analysis.

#### 4. Framing Check

The lesion should be reasonably centered and large enough in the frame for the classifier to work with.

| Condition | Detection Method | Threshold | User Message |
|-----------|-----------------|-----------|--------------|
| **Lesion too small** | `segmentation.areaPixels / (imageWidth * imageHeight)` | < 5% of image area | "Lesion too small in frame -- move closer" |
| **Lesion too large** | Same ratio | > 80% of image area | "Too close -- move back slightly so surrounding skin is visible" (needed for skin texture analysis) |
| **Off-center** | Centroid of lesion mask vs. image center | Centroid > 30% of image dimension from center | "Try to center the spot in the frame" |

**Note:** The framing check interacts with measurement. If the lesion fills the entire frame (> 80%), there is no surrounding skin for the texture frequency analysis (Approach 1), and no room for a USB-C reference (Approach 0). The framing guidance actively improves measurement accuracy.

#### 5. Motion Blur Detection

Motion blur is directional -- it reduces sharpness along one axis more than the other. This is detectable by comparing Laplacian variance in horizontal and vertical directions separately.

**Method:** Compute Laplacian variance using a horizontal-only kernel `[[0,0,0],[1,-2,1],[0,0,0]]` and a vertical-only kernel `[[0,1,0],[0,-2,0],[0,1,0]]`. If the ratio of variances exceeds 3:1, directional blur is present.

**User message:** "Image is blurry from movement -- hold the phone steady and tap again"

### Quality Gate Flow

```
User captures image
        |
        v
Compute quality scores (all 5 checks, ~50ms total)
        |
        v
Any CRITICAL issues? (blurry, too dark, severe glare)
   |            |
   YES          NO
   |            |
   v            v
Show retake    Any WARNINGS? (slight softness, uneven light, off-center)
prompt with    |            |
specific       YES          NO
message        |            |
               v            v
          Show yellow      Show green
          indicator,       indicator,
          allow proceed    proceed to
          OR retake        analysis
```

**Critical principle:** Quality assessment must NEVER block the user from proceeding. If someone is in an urgent situation and needs a quick screening, they should be able to tap "Analyze anyway" even with a blurry photo. The quality gate is advisory, not mandatory. But the result should carry a quality disclaimer: "Analysis based on a low-quality image -- results may be less accurate."

### Multi-Capture Quality Display

In multi-capture mode, each thumbnail in the capture strip should display a quality indicator:

```
[Photo 1]  [Photo 2]  [Photo 3]
  (green)    (yellow)    (red)
  "Sharp"    "OK"       "Blurry"
   [X]        [X]        [X] [Retake]
```

- **Green dot:** All quality checks pass. Good to analyze.
- **Yellow dot:** Minor issues detected. Usable but not ideal. User can proceed or retake.
- **Red dot:** Critical quality issue. Strong retake recommendation.

Each photo must have:
- An **X button** to delete it from the multi-capture strip (this partially exists already).
- A **Retake button** that removes the current photo and re-opens the camera for that slot.
- A **quality score tooltip** showing which checks passed/failed.

### Quality Score Type

```typescript
interface ImageQualityAssessment {
    overall: "good" | "acceptable" | "poor";
    sharpness: { score: number; pass: boolean; message?: string };
    lesionFocus: { score: number; pass: boolean; message?: string };
    brightness: { mean: number; pass: boolean; message?: string };
    evenness: { variance: number; pass: boolean; message?: string };
    glare: { percentage: number; pass: boolean; message?: string };
    framing: { areaRatio: number; centered: boolean; pass: boolean; message?: string };
    motionBlur: { detected: boolean; direction?: "horizontal" | "vertical"; message?: string };
    actionableMessages: string[];  // Ordered list of things to fix
}
```

### Files Affected by Quality Gating

| File | Change |
|------|--------|
| `image-quality.ts` (NEW) | Standalone quality assessment module with all 5 checks. |
| `multi-image.ts` | Refactor `scoreImageQuality()` to use the new module. |
| `DermCapture.svelte` | Add quality gate after capture, show indicator dots on thumbnails, add retake flow. |
| `types.ts` | Add `ImageQualityAssessment` type. |
| `index.ts` | Export quality assessment functions. |

## Implementation Plan

### Phase 0: Image Quality Gating (Week 1)

**Goal:** Ensure every image is assessed for quality before analysis begins. This is prerequisite to measurement -- accurate measurement requires a sharp, well-lit, properly-framed image.

1. Create `image-quality.ts` with standalone quality assessment functions:
   - `assessSharpness(imageData): { score, pass, message }`
   - `assessLesionFocus(imageData, mask): { score, pass, message }`
   - `assessLighting(imageData): { brightness, evenness, glare, messages }`
   - `assessFraming(segmentation, imageWidth, imageHeight): { areaRatio, centered, messages }`
   - `detectMotionBlur(imageData): { detected, direction, message }`
   - `assessImageQuality(imageData, segmentation): ImageQualityAssessment`
2. Refactor `scoreImageQuality()` in `multi-image.ts` to delegate to the new module.
3. Update `DermCapture.svelte`:
   - After each capture, run `assessImageQuality()` (~50ms).
   - Show quality indicator (green/yellow/red dot) on each thumbnail.
   - Show actionable messages ("Image is blurry -- hold steady and tap again").
   - Add "Retake" button on thumbnails with poor quality.
   - Add "Analyze anyway" bypass for users who want to proceed despite quality warnings.
4. Add `ImageQualityAssessment` type to `types.ts`.

### Phase 1: Measurement Foundation + Connector Reference (Week 2-3)

**Goal:** Replace the broken measurement functions with the measurement engine skeleton and implement Approach 0 (connector reference).

1. Define new types (`MeasurementResult`, `MeasurementMethod`, `MeasurementConfidence`, `CalibratedDiameter`) in `types.ts`.
2. Create `measurement.ts` with the `MeasurementEngine` class and approach orchestration logic.
3. Implement `measurement-connector.ts`:
   - Grayscale conversion + Canny edge detection (reuse existing edge utilities if present, otherwise implement).
   - Contour detection and rectangle filtering.
   - Aspect ratio validation for USB-C (3.44:1), Lightning (5.13:1), USB-A (2.67:1).
   - Metallic color signature check.
   - Perspective distortion warning (aspect ratio deviation > 10%).
4. Wire `MeasurementEngine` into `computeABCDE()`. Keep legacy fallback active.
5. Update `ABCDEScores` type with `diameterConfidence`, `diameterMethod`, `diameterCI` fields.
6. Add UI prompt in `DermCapture.svelte`: "For best accuracy, place your charger cable's USB-C plug next to the spot."
7. Update `ClassificationResult.svelte` to show confidence interval and method indicator.

### Phase 2: Skin Texture Analysis (Week 4-5)

**Goal:** Implement the fully-automatic fallback that works without any reference object.

1. Implement `fft.ts`: Pure TypeScript radix-2 Cooley-Tukey 2D FFT.
   - Power-of-2 padding.
   - Hanning window.
   - Forward and inverse transforms.
   - Benchmark: must complete 256x256 FFT in < 200ms on iPhone 12.
2. Implement `measurement-texture.ts`:
   - Skin ROI extraction (pixels near lesion boundary, outside mask).
   - Power spectrum computation.
   - Radial frequency peak detection.
   - Body-location pore spacing lookup table.
   - Confidence scoring based on peak prominence (peak-to-noise ratio).
3. Add the texture approach to the `MeasurementEngine` cascade.
4. Cross-validation: when both connector and texture approaches succeed, compare results. Log discrepancies for analysis.

### Phase 3: LiDAR Integration (Week 6-7)

**Goal:** Add high-accuracy depth sensing for devices that support it.

1. Implement `measurement-lidar.ts`:
   - Feature detection for LiDAR / WebXR Depth API availability.
   - ARKit session management (request, capture depth frame, tear down).
   - Depth-to-scale conversion using camera intrinsics.
   - Median depth filtering across lesion region.
2. Device model lookup table for camera intrinsics (focal length, sensor size) for common iPhone/iPad models.
3. Graceful degradation: if WebXR is unavailable, skip silently.
4. Add to `MeasurementEngine` cascade.

### Phase 4: Parallax Estimation (Week 8-9)

**Goal:** Extract depth from multi-capture sessions.

1. Implement ORB-style feature detection in pure TypeScript:
   - FAST keypoint detector.
   - Binary descriptors (BRIEF-style).
   - Brute-force Hamming distance matching.
2. Implement `measurement-parallax.ts`:
   - Feature matching between image pairs.
   - Fundamental matrix estimation (8-point algorithm + RANSAC).
   - `DeviceMotionEvent` integration for baseline distance.
   - Depth triangulation.
3. UI enhancement in multi-capture mode: show accelerometer-derived motion indicator. Prompt user if motion is insufficient.
4. Add to `MeasurementEngine` cascade.

### Phase 5: Validation and Safety (Week 10-11)

**Goal:** Validate accuracy and ensure measurement errors do not cause harm.

1. **Ground truth dataset:** Collect 50+ lesion images with ruler-measured diameters across:
   - Multiple body locations.
   - Multiple Fitzpatrick skin types (I-VI).
   - Multiple device models.
   - With and without USB-C reference.
2. **Accuracy benchmarking:**
   - Mean absolute error (MAE) for each approach.
   - Percentage of measurements where the 6mm threshold classification is correct.
   - Stratified by body location, skin type, device.
3. **Safety gate testing:**
   - Verify that low-confidence measurements near 6mm always trigger the uncertainty warning.
   - Verify that high-confidence measurements above 6mm correctly flag the lesion.
   - Verify that the system never drops to "low risk" purely because of a measurement error.
4. **Performance testing:**
   - FFT + full measurement pipeline must complete in < 500ms on iPhone 12.
   - LiDAR session must not block the UI thread.
   - Memory footprint of FFT intermediate arrays must stay under 50MB.

## Validation Strategy

### Accuracy Validation Protocol

1. **Ruler-measured ground truth:** Photograph 50+ real lesions (or printed lesion phantoms of known sizes) with a millimeter ruler visible, then crop out the ruler and run the measurement pipeline. Compare pipeline output to ruler measurement.

2. **Cross-method validation:** For images where multiple approaches succeed, check that they agree within their stated confidence intervals. Systematic disagreement indicates a bug or calibration error.

3. **Threshold classification accuracy:** The clinically relevant question is not "how close is the mm estimate?" but "does the system correctly classify lesions as above or below 6mm?" Report the true positive rate and false negative rate for the 6mm threshold specifically.

4. **Fitzpatrick stratification:** Report accuracy separately for Fitzpatrick I-III and IV-VI. If the skin texture approach has significantly worse accuracy on darker skin, this must be documented and the confidence scoring adjusted accordingly.

### Regression Tests

```typescript
// Synthetic tests for each measurement approach
describe("MeasurementEngine", () => {
    it("detects USB-C connector and calibrates correctly", () => {
        // Synthetic image with a known-size rectangle
        // Verify pixelsPerMm within 5% of ground truth
    });

    it("finds skin texture frequency peak on forearm image", () => {
        // Real or synthetic image with known pore spacing
        // Verify FFT peak frequency within expected range
    });

    it("falls through gracefully when no method succeeds", () => {
        // Blank image, no skin texture, no connector
        // Should return legacy estimate with confidence: "low"
    });

    it("flags uncertainty when diameter straddles 6mm threshold", () => {
        // Measurement of 5.5mm with CI [4.0, 7.0]
        // Should set confidence: "low" and trigger warning
    });

    it("never returns diameterMm < 0", () => {
        // Edge case: empty segmentation, zero-area mask
    });
});
```

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Measurement error causes false reassurance (lesion appears < 6mm when actually > 6mm) | **Critical** | Medium | Always show confidence interval. When CI includes 6mm, warn explicitly. Safety gate: "When in doubt, see a doctor." |
| FFT implementation has a bug producing wrong frequencies | High | Medium | Validate against known test signals (pure sinusoids of known frequency). Cross-reference with a reference FFT library during development. |
| USB-C detection false positives (detects a non-connector rectangle) | Medium | Low | Require metallic color signature + aspect ratio match. Allow user to confirm/reject detected reference. |
| Performance regression on older phones | Medium | Medium | Benchmark on iPhone 11 (minimum target). Use Web Workers for FFT if main thread budget is exceeded. |
| Users trust the measurement too much | Medium | High | Never show bare numbers without confidence. Consumer translation says "approximately" not "exactly." Disclaimer: "This is an estimate, not a clinical measurement." |
| Body location selection is wrong, skewing texture analysis | Medium | Medium | Cross-validate: if FFT peak frequency is far outside the expected range for the selected body location, flag it and widen the confidence interval. |

## References

- Stolz W, Riemann A, Cognetta AB, et al. "ABCD rule of dermatoscopy: a new practical method for early recognition of malignant melanoma." Eur J Dermatol. 1994;4:521-527.
- Korotkov K, Garcia R. "Computerized analysis of pigmented skin lesions: a review." Artif Intell Med. 2012;56(2):69-90.
- Hashimoto K. "Fine structure of the Meissner corpuscle of human palmar skin." J Invest Dermatol. 1973;60(1):20-28.
- USB Implementers Forum. "USB Type-C Cable and Connector Specification, Revision 2.2." 2022. (Plug dimensions: 8.25mm x 2.4mm)
- Apple Inc. "ARKit - Depth." developer.apple.com/augmented-reality/arkit/ (LiDAR depth API documentation).
- W3C. "WebXR Depth Sensing Module." w3.org/TR/webxr-depth-sensing-1/ (WebXR depth API specification).
