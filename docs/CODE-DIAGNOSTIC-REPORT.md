Updated: 2026-03-24 | Version 1.0.0
Created: 2026-03-24

# Dr. Agnes Code Diagnostic Report

**Overall Code Quality Score: 72 / 100**

Scoring breakdown:
- Code Quality: 78/100
- Architecture: 82/100
- Security: 58/100
- Performance: 65/100
- UX Flow Integrity: 80/100
- Data Integrity: 70/100

What was NOT tested: runtime behavior, actual HF API responses, WASM module loading, camera/file-upload on real devices, production build bundle sizes, actual ONNX inference, Python subprocess execution, pi.ruv.io brain connectivity.

---

## 1. Code Quality

### CRITICAL: `image-analysis.ts` is 2,059 lines -- must be split

**File:** `src/lib/dragnes/image-analysis.ts:1-2059`
**Severity:** HIGH
**Details:** This file contains 8 distinct subsystems (segmentation, asymmetry, border analysis, color analysis, texture/GLCM, structure detection, attention heatmap, and combined classification) plus utility functions. At 2,059 lines it is 4x the 500-line limit specified in the project's CLAUDE.md.
**Recommended fix:** Split into separate modules:
- `segmentation.ts` (lines 1-397) -- LAB conversion, Otsu, morphology, connected components
- `asymmetry.ts` (lines 399-524) -- principal axis folding
- `border-analysis.ts` (lines 526-653) -- 8-octant border scoring
- `color-analysis.ts` (lines 655-884) -- k-means LAB clustering
- `texture-analysis.ts` (lines 886-1005) -- GLCM features
- `structure-detection.ts` (lines 1007-1287) -- LBP, streaks, blue-white veil
- `attention-map.ts` (lines 1289-1516) -- heatmap generation
- `feature-classifier.ts` (lines 1518-1962) -- combined classification
- `lesion-detection.ts` (lines 1964-2059) -- safety gate
Keep `image-analysis.ts` as a barrel re-export for backward compatibility.

### HIGH: `DrAgnesPanel.svelte` is 1,420 lines -- must be split

**File:** `src/lib/components/DrAgnesPanel.svelte:1-1420`
**Severity:** HIGH
**Details:** This component contains the entire application flow: capture handling, multi-image analysis, single-image analysis, ABCDE scoring, 7-point checklist computation, consumer translation display, medical details view, history tab, settings tab, and all 4 navigation tabs. It is nearly 3x the 500-line limit.
**Recommended fix:** Extract sub-components:
- `ScanResultHero.svelte` -- the consumer-friendly result display (lines ~688-749)
- `MedicalDetails.svelte` -- the collapsible medical details panel
- `HistoryView.svelte` -- the history tab
- `SettingsView.svelte` -- the settings tab
- `AnalysisProgress.svelte` -- the loading/analysis step indicator
Keep `DrAgnesPanel.svelte` as the orchestrator, delegating rendering to child components.

### MEDIUM: `any` type in `classify-local/+server.ts`

**File:** `src/routes/api/classify-local/+server.ts:28`
**Severity:** MEDIUM
**Details:** `let ortSession: any = null;` uses an untyped variable for the ONNX Runtime inference session. This bypasses TypeScript's type checking for all session operations.
**Recommended fix:**
```typescript
import type { InferenceSession } from "onnxruntime-node";
let ortSession: InferenceSession | null = null;
```
Note: `onnxruntime-node` is dynamically imported, so the type import should be a separate `import type` statement that does not trigger the dynamic import.

### MEDIUM: Duplicate `segmentLesion` implementations

**File:** `src/lib/dragnes/preprocessing.ts:168` and `src/lib/dragnes/image-analysis.ts:275`
**Severity:** MEDIUM
**Details:** There are two independent `segmentLesion` functions:
1. `preprocessing.ts:168` -- simpler version using grayscale Otsu + morphological closing. Returns `SegmentationMask` type.
2. `image-analysis.ts:275` -- more sophisticated version using LAB color space L-channel Otsu + morphological closing + opening + largest connected component extraction. Returns `SegmentationResult` type.

The `abcde.ts` imports from `preprocessing.ts` (the simpler one), while `classifier.ts` and `detectLesionPresence` use the one from `image-analysis.ts` (the better one). This means ABCDE scores computed via the standalone `computeABCDE()` function use a lower-quality segmentation than the classifier.
**Recommended fix:** Deprecate the `preprocessing.ts` segmentation. Have `abcde.ts` import from `image-analysis.ts` instead. If backward compatibility is needed, make `preprocessing.segmentLesion` a thin wrapper that delegates to the `image-analysis` version with a type adapter.

### MEDIUM: Duplicate `cosineSimilarity` implementations

**File:** `src/lib/dragnes/classifier.ts:893` and `src/lib/dragnes/multi-image.ts:39`
**Severity:** MEDIUM
**Details:** Two `cosineSimilarity` functions exist:
1. `classifier.ts:893` -- operates on `number[]`
2. `multi-image.ts:39` -- operates on `Float32Array`

Both compute the same formula. The only difference is the input type.
**Recommended fix:** Create a shared utility (e.g., `src/lib/dragnes/math-utils.ts`) with a single generic implementation. Use `ArrayLike<number>` as the parameter type to cover both `number[]` and `Float32Array`.

### MEDIUM: Duplicate `morphDilate` / `morphErode` / `morphClose` implementations

**File:** `src/lib/dragnes/preprocessing.ts:264-306` and `src/lib/dragnes/image-analysis.ts:214-260`
**Severity:** MEDIUM
**Details:** Identical morphological operation implementations exist in both files.
**Recommended fix:** Extract to a shared `morphology.ts` utility module.

### LOW: Unused exports from `index.ts`

**File:** `src/lib/dragnes/index.ts`
**Severity:** LOW
**Details:** The barrel file exports many symbols that are not imported anywhere in the component or route code:
- `PrivacyPipeline` (from `privacy.ts`) -- not used in any component or route
- `preprocessImage`, `colorNormalize`, `removeHair`, `toNCHWTensor` -- only used internally by `classifier.ts`, not imported via `index.ts`
- `DRAGNES_CONFIG` -- only imported directly in `health/+server.ts`

These are not harmful but bloat the public API surface. The `PrivacyPipeline`, `datasets.ts`, `federated.ts`, `deployment-runbook.ts`, `benchmark.ts`, `witness.ts`, and `offline-queue.ts` modules appear to be scaffolding that is not wired into the application.
**Recommended fix:** Mark unused modules as `@internal` or move them to a `_future/` directory. Document which exports are part of the stable public API.

### LOW: `recordClassification` called with different signatures

**File:** `src/lib/components/DrAgnesPanel.svelte:578` vs `src/lib/components/DrAgnesPanel.svelte:370`
**Severity:** LOW
**Details:** The single-image path calls `recordClassification({ predictedClass, confidence, allProbabilities, modelId, demographics, bodyLocation })` while the multi-image path calls `recordClassification({ eventId, topClass, confidence, probabilities, bodyLocation, modelId })`. The `ClassificationEvent` interface uses `predictedClass` as the field name, but the multi-image path passes `topClass` instead. The `allProbabilities` field expects `Record<string, number>` but the multi-image path passes the `ClassProbability[]` array directly. Additionally, the multi-image path pre-generates `eventId` with `crypto.randomUUID()` and passes it, but `recordClassification` also generates its own id internally -- the passed `eventId` will be silently ignored since it is not in the `Omit<ClassificationEvent, "id" | "timestamp">` type.
**Recommended fix:** Standardize the call sites. The multi-image path should use `predictedClass` (not `topClass`) and convert probabilities to `Record<string, number>` format. Remove the pre-generated `eventId` since it is overwritten anyway. This may be causing the `lastEventId` in the multi-image path to be out of sync with the actual stored event id.

---

## 2. Architecture

### The 4-layer ensemble is properly implemented

**File:** `src/lib/dragnes/classifier.ts:153-240`
**Severity:** N/A (positive finding)
**Details:** The classification strategy is well-designed with clear priority ordering:
1. Custom-trained local ViT (70% custom + 15% trained-weights + 15% rule-based)
2. Dual HF models (50% dual-HF + 30% trained-weights + 20% rule-based)
3. Single HF model (60% HF + 25% trained-weights + 15% rule-based)
4. Offline fallback (60% trained-weights + 40% rule-based)

Each tier falls back gracefully. The dual-model ensemble uses `Promise.allSettled` for parallel execution with individual failure handling. Ensemble weights are documented with rationale.

### Multi-image consensus integrates cleanly

**File:** `src/lib/dragnes/multi-image.ts`
**Severity:** N/A (positive finding)
**Details:** The quality-weighted consensus with melanoma safety gate is well-implemented. Quality scoring uses Laplacian variance (sharpness), RMS contrast, and segmentation quality. The safety gate ensures that if ANY image flags melanoma > 60%, that signal is preserved in the consensus -- this is the correct approach for a safety-critical system (prioritize sensitivity over specificity for melanoma).

### MEDIUM: Inconsistency between API route response formats

**File:** `src/routes/api/classify/+server.ts:65`, `src/routes/api/classify-v2/+server.ts:59`, `src/routes/api/classify-local/+server.ts:180`
**Severity:** MEDIUM
**Details:** The three classify endpoints return slightly different response shapes:
- `/api/classify`: `{ results, model }`
- `/api/classify-v2`: `{ results, model }`
- `/api/classify-local`: `{ results, model, backend, melanomaSensitivity, trainedOn }`

The HF proxy endpoints return the raw HF API response (array of `{ label, score }`) in `results`, while the local endpoint returns the same format but after local softmax normalization. The client-side `callHFModel()` method handles both, but there is no shared response type definition.
**Recommended fix:** Define a shared `ClassifyApiResponse` type in `types.ts`. Have all three endpoints conform to it. The extra fields in classify-local can be optional.

### Consumer translation is properly wired as primary display

**File:** `src/lib/components/DrAgnesPanel.svelte:252-256`, `src/lib/dragnes/consumer-translation.ts`
**Severity:** N/A (positive finding)
**Details:** The `consumerResult` derived state is computed from `translateForConsumer(r.topClass, r.confidence, r.probabilities)` and is rendered as the primary "hero" result display. Medical details are hidden behind a "Show Medical Details" toggle. The consumer translation includes an upgrade mechanism: if combined cancer probability > 30%, a "green" result gets bumped to "yellow". This is correct safety behavior.

### LOW: `consumer-translation.ts` does not update urgency when risk is upgraded

**File:** `src/lib/dragnes/consumer-translation.ts:151-156`
**Severity:** LOW
**Details:** When the cancer probability check upgrades `effectiveRisk` from "green" to "yellow", only `effectiveRisk` and `effectiveAction` are updated. The `shouldSeeDoctor` and `urgency` fields still come from the original translation (which would be `false` and `"none"` for a green class). A user seeing risk level "yellow" with `shouldSeeDoctor: false` would receive contradictory information.
**Recommended fix:** When upgrading to yellow, also set `shouldSeeDoctor = true` and `urgency = "routine"`.

---

## 3. Security

### CRITICAL: No file upload validation (size, type, or content)

**File:** `src/routes/api/classify/+server.ts:36-39`, `src/routes/api/classify-v2/+server.ts:30-35`, `src/routes/api/classify-local/+server.ts:149-155`
**Severity:** CRITICAL
**Details:** All three classification endpoints accept file uploads with zero validation:
- No file size limit -- an attacker can upload multi-GB files, causing memory exhaustion
- No content type check -- the server will forward any file type (PDF, ZIP, executable) to HuggingFace or attempt to process it with sharp/ONNX
- No image header validation -- there is no check that the uploaded content is actually an image
- For the local endpoint, the image buffer is passed directly to `sharp()` and `onnxruntime`, both of which could have vulnerabilities with malformed inputs

**Recommended fix:**
```typescript
// Add to all classify endpoints:
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

const imageFile = formData.get("image");
if (!imageFile || !(imageFile instanceof Blob)) {
    throw error(400, "No image provided");
}
if (imageFile.size > MAX_FILE_SIZE) {
    throw error(413, "Image too large. Maximum 10 MB.");
}
if (!ALLOWED_TYPES.includes(imageFile.type)) {
    throw error(415, "Invalid file type. Only JPEG, PNG, and WebP accepted.");
}
```

For the local endpoint, additionally validate the image header bytes (magic numbers) before passing to sharp.

### HIGH: No CORS or CSP headers configured

**File:** Project-wide
**Severity:** HIGH
**Details:** There are no CORS restrictions, Content-Security-Policy headers, or security middleware configured anywhere in the SvelteKit application. The `src/routes/+layout.svelte`, `svelte.config.js`, and hooks files do not set any security headers.

For a medical application handling dermoscopic images:
- Without CSP, the app is vulnerable to XSS injection that could exfiltrate classification data
- Without CORS restrictions, any website can make requests to the API endpoints
- The `/api/analyze` endpoint does have rate limiting, but `/api/classify`, `/api/classify-v2`, and `/api/classify-local` do not

**Recommended fix:** Add a `hooks.server.ts` file with security headers:
```typescript
// src/hooks.server.ts
import type { Handle } from '@sveltejs/kit';

export const handle: Handle = async ({ event, resolve }) => {
    const response = await resolve(event);
    response.headers.set('X-Content-Type-Options', 'nosniff');
    response.headers.set('X-Frame-Options', 'DENY');
    response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
    response.headers.set('Content-Security-Policy',
        "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self' 'unsafe-inline';"
    );
    return response;
};
```

### MEDIUM: HF_TOKEN handling is correct but fragile

**File:** `src/routes/api/classify/+server.ts:30-32`
**Severity:** MEDIUM (positive with caveat)
**Details:** The API key is correctly read from `$env/dynamic/private` (server-side only) and never exposed to the browser. The `hf-classifier.ts` client-side module accepts an optional `apiKey` parameter but it is never passed from the browser -- all calls go through the server proxy. This is the correct pattern.

However, the `hf-classifier.ts` module exports `HF_API_URL` and `HF_MODEL` constants that could be used by browser code to bypass the proxy. While they are not currently used that way, they represent a footgun.
**Recommended fix:** Move `HF_API_URL` and `HF_MODEL` exports to a server-only module, or add JSDoc `@server` annotations to discourage browser use.

### MEDIUM: No rate limiting on classify endpoints

**File:** `src/routes/api/classify/+server.ts`, `src/routes/api/classify-v2/+server.ts`, `src/routes/api/classify-local/+server.ts`
**Severity:** MEDIUM
**Details:** The `/api/analyze` endpoint has an in-memory rate limiter (100 req/min per IP), but none of the three classify endpoints have any rate limiting. Since each classify call makes an external API call to HuggingFace (which has its own rate limits), an attacker could exhaust the HF API quota or cause HF to throttle/block the account.
**Recommended fix:** Apply the same rate limiting pattern (or a shared middleware) to all API endpoints.

### LOW: `classify-local` uses `execSync` for Python fallback

**File:** `src/routes/api/classify-local/+server.ts:99`
**Severity:** LOW
**Details:** `execSync` with a file path argument could be a command injection vector if the temp file path contained shell metacharacters. The current code uses `Date.now()` for the filename (safe), but the pattern is fragile. Additionally, `execSync` blocks the Node.js event loop for up to 30 seconds.
**Recommended fix:** Use `execFile` (async, no shell interpretation) instead of `execSync`. This eliminates both the injection risk and the event loop blocking.

---

## 4. Performance

### HIGH: `image-analysis.ts` performs blocking synchronous computation

**File:** `src/lib/dragnes/image-analysis.ts` (multiple functions)
**Severity:** HIGH
**Details:** All image analysis functions (`segmentLesion`, `measureAsymmetry`, `analyzeBorder`, `analyzeColors`, `analyzeTexture`, `detectStructures`, `generateAttentionMap`) are synchronous and perform O(n) to O(n^2) operations on every pixel. For a 1920x1080 image (2M pixels):

- `segmentLesion`: LAB conversion (2M iterations) + Otsu (2M) + morphological close (2M * 7^2 kernel) + morphological open (2M * 5^2 kernel) + connected components BFS (2M) = approximately 5 heavy passes
- `analyzeBorder`: full pixel scan for centroid + full scan for border detection + per-border-pixel color gradient = 2-3 passes
- `analyzeColors`: k-means with 15 iterations over 5000 sampled pixels = fast due to sampling
- `analyzeTexture`: GLCM construction with 4 directions = 4 passes over all pixels
- `detectStructures`: LBP computation + globule detection + streak detection + blue-white detection + regression detection = 5+ passes
- `generateAttentionMap`: full pass for mean LAB + full pass for attention + Gaussian smooth (2 passes) + resize = 4+ passes

Total: approximately 20+ full-image passes, all synchronous on the main thread. This will cause UI freezing for 2-5 seconds on typical mobile devices.
**Recommended fix:**
1. Downsample images before analysis (e.g., to 512x512) -- most dermoscopic features are preserved at this resolution
2. Use `requestIdleCallback` or `setTimeout(0)` between analysis stages to yield to the UI thread
3. Consider moving heavy computation to a Web Worker
4. The `classifyReal()` method in `classifier.ts` already runs all these sequentially -- it should be made async with yield points

### MEDIUM: Camera stream not cleaned up on component remount

**File:** `src/lib/components/DermCapture.svelte:302-306`
**Severity:** MEDIUM
**Details:** The `onDestroy` callback correctly stops camera tracks: `stream.getTracks().forEach((t) => t.stop())`. However, if the component is hidden/shown via the tab navigation in `DrAgnesPanel.svelte` using `{#if activeView === "scan"}`, Svelte will destroy and recreate the component each time the user switches away from and back to the Scan tab. Each recreation could start a new camera stream without the old one being fully released, especially on iOS where `getUserMedia` has known issues with rapid acquire/release cycles.
**Recommended fix:** Use `{#if}` blocks that keep the component alive but hidden (via CSS `display: none`) instead of destroying it, OR manage the camera stream in the parent component. Alternatively, add a guard in `startCamera()` to ensure the previous stream is fully stopped before requesting a new one.

### MEDIUM: Multi-image classification runs images sequentially

**File:** `src/lib/dragnes/multi-image.ts:145-149`
**Severity:** MEDIUM
**Details:** In `classifyMultiImage`, images are classified in a sequential `for...of` loop:
```typescript
for (const img of images) {
    const result = await classifier.classifyWithDemographics(img, demographics);
    ...
}
```
Each classification involves server calls (HF API) that could run in parallel. With 3 images, this triples the wall-clock time.
**Recommended fix:** Use `Promise.all` to classify images concurrently:
```typescript
const results = await Promise.all(
    images.map(img => classifier.classifyWithDemographics(img, demographics))
);
```
Note: this assumes the HF API proxy can handle concurrent requests, which it can since each request creates a new fetch.

### LOW: `DermCapture.svelte` creates temporary canvases without cleanup

**File:** `src/lib/dragnes/classifier.ts:519-530`, `src/lib/dragnes/classifier.ts:597-608`
**Severity:** LOW
**Details:** Both `classifyCustomLocal()` and `classifyHFDual()` create `document.createElement("canvas")` instances to convert ImageData to JPEG blobs. These canvases are never appended to the DOM but they do consume memory until garbage collected. On repeated classifications, this could lead to memory pressure on low-end devices.
**Recommended fix:** Reuse a single off-screen canvas instance across calls, stored as a class property on `DermClassifier`.

---

## 5. UX Flow Integrity

### The full flow works: capture -> classify -> result -> new scan

**File:** `src/lib/components/DrAgnesPanel.svelte`
**Severity:** N/A (positive finding)
**Details:** The flow is:
1. `DermCapture` fires `oncapture` or `onmulticapture` event
2. `handleCapture()` or `handleMultiCapture()` stores the imageData and calls `analyzeImage()` or `analyzeMultiImage()`
3. Analysis runs the classifier, computes ABCDE scores, generates Grad-CAM, and builds explanation findings
4. Results display as consumer-friendly hero, with medical details collapsible
5. `handleNewScan()` resets all state including calling `dermCaptureRef?.resetCapture()`
This flow is complete and correctly wired.

### Error states are properly handled at each step

**File:** `src/lib/components/DrAgnesPanel.svelte:491-495`, `src/lib/components/DrAgnesPanel.svelte:593-596`
**Severity:** N/A (positive finding)
**Details:**
- Lesion presence safety gate catches non-lesion images before classification
- Classification errors are caught and displayed as `classificationError`
- Low confidence warnings are shown when confidence < 0.4 (single) or < 0.5 (multi)
- Camera errors show retry button + upload fallback
- HF API failures gracefully fall back to local analysis
- ONNX failures fall back to Python, then to error message

### MEDIUM: Multi-capture "Done" button fires at 2 images but maxImages is 3

**File:** `src/lib/components/DermCapture.svelte:470-479`
**Severity:** MEDIUM
**Details:** The "Done" button appears when `capturedImages.length >= 2`, but the default `maxImages` is 3. This means users can submit 2 images even though the UI shows "Photo 2 of 3" and has an empty slot remaining. While this is intentional flexibility, the UX is confusing -- the user sees an empty slot and a "Done" button simultaneously. The auto-fire at `maxImages` (line 213) means reaching 3 images will immediately trigger analysis without requiring the user to tap Done.
**Recommended fix:** Either:
- Change the "Done" button text to "Analyze {n} Photos (or add more)" to clarify the choice
- Remove auto-fire at maxImages and always require explicit Done tap
- Add a brief instruction like "Tap Done when ready, or add up to 3 photos"

### LOW: Navigation tabs (History, Learn, Settings) are stub implementations

**File:** `src/lib/components/DrAgnesPanel.svelte` (beyond line 750)
**Severity:** LOW
**Details:** Based on the component structure, the History tab shows `records` (which starts empty and is never populated from `localStorage` analytics data), the Learn tab shows the `MethodologyPanel` and `AboutPage`, and the Settings tab shows basic toggles. The History tab does not load persisted `ClassificationEvent` data from the analytics store, so it always appears empty even after scans have been performed.
**Recommended fix:** On mount or tab switch, call `getEvents()` from the analytics store to populate the records array.

---

## 6. Data Integrity

### HIGH: AUROC results from `auroc-results.json` are not used in the UI

**File:** `scripts/auroc-results.json`
**Severity:** HIGH
**Details:** The AUROC computation results exist as a JSON file in `scripts/` but are not imported or referenced anywhere in the source code. The `AboutPage.svelte` mentions AUROC in text ("AUROC not yet computed") and the `AnalyticsDashboard.svelte` displays the DermaSensor benchmark AUROC (0.758) from `clinical-baselines.ts`, but neither references the actual computed AUROC results for the DrAgnes models.

Similarly, `combined-training-results.json` in `scripts/` is not referenced by any source file.
**Recommended fix:** Import the AUROC results into `clinical-baselines.ts` or a new `model-benchmarks.ts` file, and display them alongside the DermaSensor comparison in the About and Analytics pages. Update the "AUROC not yet computed" text in `AboutPage.svelte`.

### MEDIUM: `trained-weights-empirical.json` is present but not imported

**File:** `src/lib/dragnes/trained-weights-empirical.json`
**Severity:** MEDIUM
**Details:** This JSON file exists alongside `trained-weights.ts` but is not imported by any TypeScript module. The `trained-weights.ts` file contains hardcoded literature-derived weights. The empirical JSON likely represents data-derived weights from actual training, which could provide better accuracy. It is unclear whether this file is a newer version intended to replace the hardcoded weights, or supplementary data.
**Recommended fix:** Determine the relationship between the two files. If the empirical weights are validated, import them and use them. If they are experimental, document their status.

### MEDIUM: `AboutPage.svelte` reports "98.2% melanoma sensitivity" with important caveat buried in fine print

**File:** `src/lib/components/AboutPage.svelte:183`
**Severity:** MEDIUM
**Details:** The page reports "98.2% melanoma sensitivity" prominently, but footnote text reveals this is "on HAM10000 same-distribution holdout" and that "on genuinely external data (ISIC 2019, 4,998 images), melanoma sensitivity drops to 61.6%". The 61.6% figure is the clinically meaningful number (it measures generalization), but it appears only in small print. This could mislead clinicians about the model's real-world performance.
**Recommended fix:** Present both numbers with equal prominence. Consider showing the external validation figure first, with the holdout figure as context. For a medical device, the external validation result is the one that matters.

### LOW: Evidence JSON files in `docs/` are documentation-only

**File:** `docs/HAM10000_stats.json`
**Severity:** LOW
**Details:** The HAM10000 statistics JSON in `docs/` contains dataset distribution data (class counts, age distributions, sex ratios, localization data) but is only used for documentation/analysis purposes, not by the runtime application. The runtime instead uses hardcoded constants in `image-analysis.ts:1555-1563` (LOG_PRIORS) and `ham10000-knowledge.ts` for demographic adjustment.
**Recommended fix:** Consider importing the JSON data directly rather than hardcoding the same numbers, to ensure they stay in sync.

---

## Summary of Issues by Severity

| Severity | Count | Key Issues |
|----------|-------|------------|
| CRITICAL | 2 | No file upload validation; `image-analysis.ts` at 2,059 lines |
| HIGH     | 5 | No CORS/CSP headers; `DrAgnesPanel.svelte` at 1,420 lines; synchronous blocking computation; no rate limiting on classify endpoints; AUROC data unused |
| MEDIUM   | 12 | `any` type; duplicate segmentation; duplicate cosine similarity; duplicate morphology; API response inconsistency; consumer-translation urgency bug; HF_TOKEN footgun; no rate limiting; camera stream lifecycle; sequential multi-image; AboutPage misleading metrics; empirical weights unused |
| LOW      | 6 | Unused exports; recordClassification signature mismatch; execSync usage; temp canvas cleanup; multi-capture UX confusion; history tab unpopulated |

---

## Priority Action Items

1. **Immediate (security):** Add file upload validation (size, type, magic bytes) to all classify endpoints. Add security headers via `hooks.server.ts`. Add rate limiting to classify endpoints.

2. **This sprint (quality):** Split `image-analysis.ts` into 8+ modules. Split `DrAgnesPanel.svelte` into sub-components. Fix the duplicate `segmentLesion` -- have `abcde.ts` use the better `image-analysis.ts` version. Fix the `recordClassification` call signature mismatch in multi-image path.

3. **Next sprint (performance):** Downsample images before analysis. Move heavy computation to a Web Worker. Parallelize multi-image classification. Reuse off-screen canvas.

4. **Backlog (data):** Wire AUROC results into the UI. Update the "98.2%" claim to show external validation prominently. Evaluate `trained-weights-empirical.json` for use. Populate history tab from analytics store.
