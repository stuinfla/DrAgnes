Updated: 2026-03-26 | Version 1.0.0
Created: 2026-03-26

# Mela Production Readiness Audit

**Audit date:** 2026-03-26
**Auditors:** 3 parallel agents (pipeline, UI, tests/security) reading every line of code
**Files audited:** 49 pipeline files, all Svelte components, all API routes, all stores, all 117 tests
**Score: 52/100 -- NOT PRODUCTION READY**

---

## Checklist

### Phase 1: Safety Blockers

- [ ] **1.1** Fix config version string "v1.0.0-demo" (config.ts:23)
- [ ] **1.2** Fix segmentation inconsistency -- abcde.ts uses worse algorithm than classifier.ts
- [ ] **1.3** Remove or disable non-functional "Biopsy" button (MelaPanel.svelte:845)
- [ ] **1.4** Remove or disable non-functional "Confirm" and "Correct" buttons (ClassificationResult.svelte:363)
- [ ] **1.5** Remove "Similar Cases" placeholder (ClassificationResult.svelte:350)
- [ ] **1.6** Fix History tab -- either wire records or remove the tab (MelaPanel.svelte:146)
- [ ] **1.7** Wire clinical history questions to risk stratification OR remove the form (MelaPanel.svelte:94-98)
- [ ] **1.8** Wire privacy toggles (Strip EXIF, Local Only) to real behavior OR remove them (MelaPanel.svelte:151-152)
- [ ] **1.9** Fix Brain Sync toggle -- wire to actual network logic OR remove (MelaPanel.svelte:150)
- [ ] **1.10** Address 0% melanoma sensitivity in fallback trained-weights (trained-weights-empirical.json)

### Phase 2: Security

- [ ] **2.1** Add hooks.server.ts with CSP, X-Frame-Options, HSTS, X-Content-Type-Options headers
- [ ] **2.2** Add rate limiting to /api/classify endpoint
- [ ] **2.3** Add rate limiting to /api/classify-v2 endpoint
- [ ] **2.4** Add rate limiting to /api/classify-local endpoint
- [ ] **2.5** Add rate limiting to /api/feedback endpoint
- [ ] **2.6** Stop forwarding HF API error messages verbatim to client (classify/+server.ts:74, classify-v2/+server.ts:65)
- [ ] **2.7** Add magic byte validation to /api/classify (match classify-local's approach)
- [ ] **2.8** Add magic byte validation to /api/classify-v2

### Phase 3: Type Safety and Tests

- [ ] **3.1** Add src/icons.d.ts declaration for ~icons/* virtual modules (8 type errors)
- [ ] **3.2** Add type declaration stub for onnxruntime-node (2 type errors)
- [ ] **3.3** Fix Uint8Array/BufferSource type error in privacy.ts:277
- [ ] **3.4** Fix instanceof File/Blob type errors in classify/+server.ts and classify-v2/+server.ts
- [ ] **3.5** Fix eventId property mismatch in MelaPanel.svelte:517
- [ ] **3.6** Exclude ruvector submodule from svelte-check scope
- [ ] **3.7** Add tests for risk-stratification.ts
- [ ] **3.8** Add tests for meta-classifier.ts
- [ ] **3.9** Add smoke tests for API endpoints

### Phase 4: High-Priority Fixes

- [ ] **4.1** Type ONNX session properly instead of `any` (inference-offline.ts:28)
- [ ] **4.2** Remove or deprecate dead HF classifier code (hf-classifier.ts)
- [ ] **4.3** Remove or clearly mark 526-line federated.ts as not-in-use
- [ ] **4.4** Fix multi-image analytics field name mismatch (topClass vs predictedClass)
- [ ] **4.5** Fix multi-image eventId pre-generation conflict with recordClassification
- [ ] **4.6** Add error handling to referral letter clipboard copy (ReferralLetter.svelte:59)
- [ ] **4.7** Fix Settings showing V2 model label when V1 may be running (MelaPanel.svelte:149)
- [ ] **4.8** Add Fitzpatrick skin type collection to demographics UI
- [ ] **4.9** Fix anonymization hardcoded "mela-v1" -- read from config (anonymization.ts:166)
- [ ] **4.10** Remove ADR-122 jargon from Settings UI (MelaPanel.svelte:1861)

### Phase 5: Moderate Fixes

- [ ] **5.1** Fix LiDAR error swallowing -- log instead of silently defaulting to 300mm
- [ ] **5.2** Remove or rewrite deployment-runbook.ts (references nonexistent infrastructure)
- [ ] **5.3** Fix privacy hash FNV-1a fallback (privacy.ts:287)
- [ ] **5.4** Fix k-anonymity stub (privacy.ts:329)
- [ ] **5.5** Wire BodyMap component rendering in DermCapture (currently imported but never shown)
- [ ] **5.6** Fix full-screen image overlay keyboard trap (WCAG) (MelaPanel.svelte:889)
- [ ] **5.7** Add localStorage size cap to analytics store (analytics.ts:107)
- [ ] **5.8** Remove setInterval from serverless rate limiter (analyze/+server.ts:39)
- [ ] **5.9** Remove console.log from handleAction (MelaPanel.svelte:845)
- [ ] **5.10** Remove or redirect console.error to structured logging

### Phase 6: Model Wiring (When Ready)

- [ ] **6.1** Wire V2 ONNX model to production inference path
- [ ] **6.2** Wire ensemble.ts into inference path (currently dead code)
- [ ] **6.3** Load empirical trained weights instead of literature-derived weights
- [ ] **6.4** Validate ONNX INT8 quantized model against original PyTorch accuracy

---

## Detailed Findings

### BLOCKER 1: Config says "v1.0.0-demo"

**File:** `src/lib/mela/config.ts:23`
```typescript
modelVersion: "v1.0.0-demo",
```

The central configuration declares the model version as `"v1.0.0-demo"`. This string propagates to any component that reads `MELA_CONFIG.modelVersion`. A production build must not ship with "demo" in its version identifier.

**Fix:** Change to the actual model version (e.g., `"v2.0.0"` or read from package.json).

---

### BLOCKER 2: Fallback classifier is BLIND to melanoma

**File:** `src/lib/mela/trained-weights-empirical.json:39-45`
```json
"mel": {
  "sensitivity": 0,
  "specificity": 1,
  "precision": 0,
  "f1": 0,
  "nTest": 54
}
```

The empirically trained weights achieve zero melanoma sensitivity. When the ONNX model is unavailable, the `classifyReal()` path uses a 40/60 blend of rule-based and trained-weights classifiers. The 60% weight given to a melanoma-blind component is dangerous. The safety gates (melanoma floor, TDS override) partially mitigate this, but the fundamental issue remains.

**Fix:** Either (a) load the empirical weights and retrain the melanoma class, or (b) increase the rule-based weight in the fallback path, or (c) disable the fallback entirely and show "offline -- cannot classify" instead.

---

### BLOCKER 3: Buttons that do nothing

**File:** `src/lib/components/MelaPanel.svelte:845`
```typescript
function handleAction(action: string, payload?: unknown) {
    if (action === "refer") {
        showReferralLetter = true;
        return;
    }
    console.log("Mela action:", action, payload);
}
```

"Biopsy", "Confirm", and "Correct" buttons all fall through to `console.log`. The "Similar Cases" section is a permanent placeholder with dashed border. The History tab is permanently empty because `records: DiagnosisRecord[]` is initialized as `[]` and never written to.

**Fix:** Remove all non-functional buttons and the History tab. They can be re-added when the features are implemented. Shipping dead buttons in a medical app erodes trust.

---

### BLOCKER 4: Clinical history collected but discarded

**File:** `src/lib/components/MelaPanel.svelte:94-98`

Five clinical history variables (`clinicalIsNew`, `clinicalHasChanged`, `clinicalPreviouslyBiopsied`, `clinicalFamilyHistory`, `clinicalSymptoms`) are collected via a full UI form but never passed to `analyzeImage()`, `analyzeMultiImage()`, or any classification function.

**Fix:** Either wire these into `risk-stratification.ts` as Bayesian priors (the architecture supports it), or remove the form and add it back when it works.

---

### BLOCKER 5: No security headers

No CSP, CORS, X-Frame-Options, HSTS, or X-Content-Type-Options headers anywhere. No rate limiting on 4 of 6 API endpoints. The one rate limiter uses an in-memory Map that resets on every Vercel cold start.

**Fix:** Add `src/hooks.server.ts` with security headers. Add rate limiting middleware.

---

### BLOCKER 6: Type checking fails (16 errors)

`npm run check` exits with code 1. Missing type declarations for `~icons/*` virtual modules (8 errors), `onnxruntime-node` (2 errors), `Uint8Array/BufferSource` mismatch (1 error), `instanceof File/Blob` on union type (2 errors), missing `eventId` property (1 error), ruvector submodule scan (1 error), missing `@types/sharp` (1 error).

**Fix:** Add declaration files, fix type mismatches, scope svelte-check to exclude submodules.

---

### BLOCKER 7: Two different segmentation algorithms

`abcde.ts` imports `segmentLesion` from `preprocessing.ts` (simple grayscale Otsu). `classifier.ts` imports from `image-analysis.ts` via `cv/segmentation.ts` (LAB L-channel Otsu + morphological open + largest connected component). ABCDE scores and main classification use different segmentation masks for the same image.

**Fix:** Make `abcde.ts` use the same segmentation as `classifier.ts`.

---

### HIGH: Silent error swallowing (10+ locations)

| File | Line | What it swallows |
|---|---|---|
| classifier.ts | 96 | WASM module import failure |
| inference-offline.ts | 56 | ONNX model load failure |
| inference-orchestrator.ts | 69 | ONNX inference failure |
| measurement-lidar.ts | 64 | Depth sensing failure (defaults to 300mm) |
| measurement-lidar.ts | 77 | WebXR session failure |
| brain-client.ts | 246 | Network error (returns success: true) |
| brain-client.ts | 293, 305, 349, 383, 429, 441, 502, 571 | Multiple API failures |
| privacy.ts | 283 | SHA-256 failure falls back to FNV-1a |
| offline-queue.ts | 179 | Fetch failure during sync |

In production, you will have zero visibility into whether the ONNX model loaded, whether brain contributions are failing, or whether privacy hashing fell back to a weak algorithm.

---

### HIGH: Dead code that should be live

| Module | What it does | Why it matters |
|---|---|---|
| ensemble.ts | V1+V2 ensemble (99.4% mel sensitivity) | Never called by classifier.ts |
| trained-weights-empirical.json | Actual HAM10000-fitted weights | Never loaded by trained-weights.ts |
| hf-classifier.ts | HuggingFace API classification | No longer called (intentionally removed, but not cleaned up) |
| federated.ts (526 lines) | LoRA, EWC++, Byzantine detection | Not imported by anything |
| ClassificationResult.svelte | Full result display component | Not imported by MelaPanel |

---

### HIGH: Settings shows V2 but users may get V1

`MelaPanel.svelte:149`:
```typescript
let modelVersion: string = $state("V2 ONNX (95.97% sensitivity)");
```

Per CLAUDE.md Priority Work: "Every user today gets the V1 model (61.6% external mel sens) instead of V2 (95.97%)." The Settings page claims V2 is running when it may not be.

---

### TESTS: What's covered vs. what's not

**Covered (117 tests, all passing):** classifier, benchmark, clinical-baselines, consumer-translation, icd10, image-quality, lesion-gate, measurement, multi-image, spot-detector, threshold-classifier.

**NOT covered (zero tests):**
- `risk-stratification.ts` -- computes the final medical risk level shown to users
- `meta-classifier.ts` -- neural + clinical agreement scoring
- `ensemble.ts` -- V1+V2 ensemble logic
- `trained-weights.ts` -- the weight matrix used in offline classification
- `inference-orchestrator.ts` -- full pipeline orchestration
- All 7 API endpoints
- All UI components

---

### POSITIVE FINDINGS

- Zero TODO/FIXME/HACK/STUB comments in any file
- Zero hardcoded API keys or secrets
- All 117 tests pass
- .env properly excluded from git
- Core CV pipeline is real (not placeholders)
- Safety gates correctly implemented
- Build succeeds and deploys cleanly
- Clinical references cited in trained weights

---

## Source

This audit was generated by 3 parallel production-validator and quality-engineer agents reading every line of the Mela codebase on 2026-03-26. Full agent transcripts available in `/private/tmp/claude-501/` task output files.
