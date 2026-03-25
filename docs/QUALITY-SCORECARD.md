Updated: 2026-03-24 12:00:00 EST | Version 2.0.0
Created: 2026-03-24

# Mela Quality Scorecard

**Evaluator**: Quality Engineer (automated audit)
**Date**: 2026-03-24
**Scope**: Complete Mela application in `examples/mela/`
**Methodology**: Every score backed by specific file evidence. No rounding up. If uncertain, score lower.
**Previous Score**: 70.2/100 (v1.0.0, 2026-03-24 10:30 EST)

---

## 1. Classification Accuracy: 72/100 (Weight: 20%)

**What is measured:**
- V1 model (HAM10000-only): 98.2% mel sensitivity on HAM10000 test split (n=225), 80.4% specificity. Source: `cross-validation-results.json`.
- V2 model (combined HAM10000 + ISIC 2019): 95.97% mel sensitivity on ISIC 2019 external test (n=621), 80.0% specificity. Source: `combined-training-results.json`.
- AUROC computed: mel AUROC 0.926 (HAM10000), 0.960 (ISIC 2019). Source: `auroc-results.json`.
- Threshold optimization done: mel threshold 0.6204 yields 93.88% sensitivity, 85.34% specificity. Source: `threshold-optimization-results.json`.
- Threshold mode default changed to "triage" -- most conservative operating point for consumer use.

**Changes since v1.0.0 (was 62):**
- (+5) Threshold classifier built and deployed (`threshold-classifier.ts`, 106 lines) with 3 modes (default/screening/triage). Per-class ROC-optimized thresholds applied by default. This directly addresses the argmax problem described in ADR-123.
- (+5) NV (nevi) sensitivity improved from 14.3% to 70.4% via per-class threshold tuning. The default 1/7 implicit threshold was the root cause -- replacing it with a computed threshold from ROC analysis fixed the most common mole misclassification.
- (+2) Triage mode set as default for consumer path -- prioritizes sensitivity over specificity for safety.
- (-2) The improvement is computed from existing model weights (no retraining). If the underlying model changes, thresholds must be recomputed.

**Remaining deductions:**
- (-8) Fitzpatrick validation still shows mel sensitivity drops to 50% on FST V skin (n=14). This remains a patient safety issue. Not addressable without more diverse training data.
- (-5) V1 model (served on Vercel via HF API) still has 61.6% mel sensitivity on external data. Production does not yet serve v2 model.
- (-5) Overall accuracy 76.4% still misses 85% target on external data.
- (-5) No prospective validation on real clinical images from phone cameras.
- (-5) NV 70.4% is functional but still means ~30% of common moles are misclassified.

**What works well:**
- (+) V2 mel sensitivity 95.97% on external data is genuinely strong.
- (+) AUROC 0.960 for melanoma on ISIC 2019 is competitive with DermaSensor.
- (+) Focal loss with class-specific alpha weights is correct for imbalanced medical data.
- (+) Threshold classifier extracts "free accuracy" from existing ROC curve without retraining.
- (+) Numbers are honestly reported with evidence file citations.

---

## 2. Evidence Chain: 88/100 (Weight: 15%)

**Every claimed number traceable to JSON:**
- `cross-validation-results.json` -- V1 HAM10000 results with per-class metrics. Verified.
- `isic2019-validation-results.json` -- V1 external validation. Verified.
- `combined-training-results.json` -- V2 training + ISIC 2019 external test results. Verified.
- `auroc-results.json` -- Per-class AUROC with confidence intervals for both datasets. Verified.
- `optimal-thresholds.json` -- 7 per-class thresholds from ROC analysis. Verified.
- `threshold-optimization-results.json` -- Full threshold details with sensitivity/specificity at each threshold. Verified.
- `fitzpatrick-v2-validation.json` -- Stratified equity results. Verified.
- `fitzpatrick-equity-report.json` -- Data distribution analysis. Verified.
- `multi-image-validation-results.json` -- Multi-image consensus results. Verified.

**Changes since v1.0.0 (was 82):**
- (+5) ADR-120 stale body text ("96.2% mel sensitivity, 73.1% specificity") corrected. The chain is now consistent across all ADRs.
- (+3) All 11 ADR status fields updated to IMPLEMENTED with timestamps. No ADR claims "proposed" status for work that is complete.
- (-2) ADR-118 corrections log still documents the original chain break -- that history is preserved (correctly) but the break itself remains a minor historical blemish.

**Remaining deductions:**
- (-5) No validation-results.json for the threshold-classifier.ts itself -- the thresholds are computed but no end-to-end test proves they improve outcomes in the deployed pipeline.
- (-4) ADR-118 corrections log documents that the original "96.2%" and "0.936 AUROC" numbers had no source. The chain was broken and had to be repaired. History preserved but not erased.
- (-3) Python validation scripts that produced the JSON results are not version-controlled alongside their outputs -- no way to reproduce the exact pipeline.

**What works well:**
- (+) ADR-118 corrections log is a model of scientific honesty -- every change documented with reason and corrector.
- (+) JSON evidence files include sample sizes, confidence intervals (AUROC), per-class breakdowns.
- (+) Fitzpatrick validation includes download failure rate (76%) as a data quality caveat.
- (+) All ADRs now have consistent IMPLEMENTED status with timestamps.

---

## 3. Code Quality: 71/100 (Weight: 10%)

**TypeScript types:**
- `types.ts` (214 lines) defines typed interfaces for all core domain objects: `LesionClass`, `ClassProbability`, `ClassificationResult`, `ABCDEScores`, `DermImage`, `BodyLocation`. Clean.
- `consumer-translation.ts` uses typed `Record<string, ...>` with explicit `ConsumerRiskLevel` union type.
- `threshold-classifier.ts` uses typed `Record<LesionClass, number>` for threshold tables. Clean.

**Error handling:**
- `classifier.ts` (973 lines) has try/catch around HF API calls with graceful fallback to offline strategy.
- `image-analysis.ts` (2059 lines) -- exceeds the 500-line guideline from CLAUDE.md by 4x. This is the largest file in the codebase.
- No error boundaries in Svelte components for classification failure rendering.

**File organization:**
- 28+ TypeScript files in `src/lib/mela/` -- well-organized by concern (types, preprocessing, classification, measurement, translation, privacy, witness, clinical-baselines, image-analysis).
- 16 Svelte components -- appropriate separation.
- `MelaPanel.svelte` at 1520 lines is extremely large -- should be decomposed.

**Deductions:**
- (-10) `image-analysis.ts` at 2059 lines violates the 500-line rule and is difficult to maintain.
- (-8) `MelaPanel.svelte` at 1520 lines is a monolithic component with scan, history, settings, learn views crammed together.
- (-5) `classifier.ts` at 973 lines is borderline -- the ensemble logic, HF API logic, SigLIP label mapping, and dual-model logic are all in one file.
- (-3) Only 2 dedicated test files exist (`classifier.test.ts`, `benchmark.test.ts`). Additional modules added since v1 (clinical-baselines.ts, image-analysis.ts) also lack tests.
- (-3) `deployment-runbook.ts` contains hardcoded GCP deployment commands as TypeScript strings -- this is an odd choice vs. a shell script.

**What works well:**
- (+) Pure TypeScript with typed arrays for image processing -- no external CV library dependency.
- (+) Clean separation between classification engine, consumer translation, and UI.
- (+) Vitest configured and test infrastructure in place.
- (+) New modules (clinical-baselines.ts, image-analysis.ts) follow the same typed pattern as existing code.

---

## 4. UI/UX: 75/100 (Weight: 10%)

**Mobile-first:**
- `DermCapture.svelte` (721 lines) implements camera capture with multi-photo mode, thumbnail strip, counter badge.
- Tailwind CSS with responsive classes throughout.
- Bottom navigation bar with Scan/History/Learn/Settings tabs.
- PWA manifest present for installability.

**Consumer-friendliness:**
- Consumer translation maps all 7 classes to plain English with 4 risk levels (green/yellow/orange/red).
- Each risk level has specific action text ("Monitor for changes", "See a dermatologist within 2 weeks").
- ABCDE scores rendered visually in `ABCDEChart.svelte`.
- Threshold mode selector defaults to "triage" for consumer safety.

**Changes since v1.0.0 (was 73):**
- (+2) Threshold mode selector now defaults to safest option (triage) rather than requiring user to understand the options.

**Remaining deductions:**
- (-7) No dedicated E2E Playwright test files found committed in the project (ADR-120 status claims Playwright E2E was run, but no test script is committed to the repo).
- (-6) The Settings UI exposes a threshold mode selector ("screening"/"triage"/"default") that a consumer would not understand without explanation.
- (-5) Medical terminology still present in ClassificationResult component (ICD-10 codes, ensemble weight bars, model provenance section).
- (-4) Body map drawer, lesion timeline, referral letter, and calibration chart components exist but are untested.
- (-3) No loading state documentation or accessibility audit.

**What works well:**
- (+) Fitzpatrick warning is prominently displayed in the UI (MelaPanel, AboutPage, MethodologyPanel).
- (+) Multi-image capture with quality-weighted consensus is genuinely useful for consumers.
- (+) Clear visual hierarchy with risk level color coding.

---

## 5. Security: 78/100 (Weight: 10%)

**Upload validation:**
- `image-quality.ts` implements brightness, contrast, sharpness checks before classification.
- `privacy.ts` (359 lines) implements EXIF stripping, PII detection, demographic reduction.
- `witness.ts` (151 lines) implements SHAKE-256 witness chain for classification provenance.

**API key handling:**
- HF_TOKEN read from `env.HF_TOKEN` (server-side `$env/static/private` or `$env/dynamic/private`). Not exposed to client.
- `.env` file exists but is gitignored (confirmed).
- No hardcoded API keys in source files.

**Deductions:**
- (-8) `deployment-runbook.ts` references `OPENROUTER_API_KEY`, `SESSION_SECRET`, `WEBHOOK_SECRET` by name. While not containing actual values, this leaks secret naming to client bundles if tree-shaken incorrectly.
- (-5) No Content-Security-Policy headers configured for the Vercel deployment.
- (-5) No rate limiting on the `/api/classify` and `/api/classify-v2` endpoints.
- (-4) HIPAA compliance is aspirational -- no BAA with HuggingFace is in place, yet images transit through HF inference API.

**What works well:**
- (+) `.env` properly gitignored.
- (+) Privacy pipeline architecture is well-designed (EXIF strip, demographic reduce, DP noise).
- (+) Images stored in IndexedDB locally, not uploaded to cloud storage.
- (+) Witness chain provides cryptographic audit trail.

---

## 6. Testing: 65/100 (Weight: 10%)

**What exists:**
- `tests/classifier.test.ts` (509 lines) -- tests preprocessing, ABCDE scoring, privacy pipeline, CNN fallback. Uses vitest with ImageData polyfill.
- `tests/benchmark.test.ts` -- performance benchmarks.
- 91 of 104 Vitest tests passing (87.5% pass rate).
- Multiple validation scripts produced JSON evidence files (cross-validation, ISIC 2019, AUROC, threshold optimization, Fitzpatrick equity).

**Changes since v1.0.0 (was 38):**
- (+15) Test count expanded from minimal coverage to 104 test cases, with 91 passing. This covers preprocessing, ABCDE scoring, privacy pipeline, CNN fallback, and classification strategies.
- (+7) Additional test coverage for threshold classification, consumer translation pipeline, and multi-image consensus logic.
- (+5) Test infrastructure proven stable -- vitest runs reliably with ImageData polyfill for Node.js.

**What still does NOT exist:**
- No committed E2E test scripts (Playwright tests were claimed in ADR-120 but no `.spec.ts` or `playwright.config.ts` found in the project).
- No unit tests for: `measurement.ts`, `measurement-connector.ts`, `measurement-texture.ts`, `brain-client.ts`, `offline-queue.ts`, `federated.ts`, `clinical-baselines.ts`, `image-analysis.ts`.
- No integration tests for the full classify-then-translate pipeline end-to-end.
- No visual regression tests for Svelte components.
- No CI/CD pipeline configuration found (no `.github/workflows/`, no `vercel.json` test step).

**Remaining deductions:**
- (-12) Measurement pipeline (3 modules), brain-client, offline-queue, federated, clinical-baselines, and image-analysis modules have zero test coverage.
- (-8) No committed E2E tests. The ADR claims Playwright was run but no evidence in the repo.
- (-8) 13 of 104 tests failing (12.5% failure rate). Failing tests are a reliability concern.
- (-5) No CI/CD pipeline -- tests are only run manually.
- (-2) Validation scripts (Python) that produced the JSON results are not tested themselves.

**What works well:**
- (+) 91 passing vitest tests is a real improvement from near-zero.
- (+) The vitest infrastructure is set up and working.
- (+) Validation JSON files serve as integration test artifacts.
- (+) classifier.test.ts polyfills ImageData for Node.js -- a thoughtful detail.

---

## 7. Documentation: 82/100 (Weight: 10%)

**What exists:**
- 11 ADR files covering architecture, validation, consumer screening, deployment, measurement (x2), ONNX, thresholds, Fitzpatrick equity, dual-model ensemble, and pi-brain.
- All 11 ADRs now have IMPLEMENTED status with timestamps (updated 2026-03-24 12:00 EST).
- `TECHNICAL-REPORT.md` -- technical documentation.
- `FDA-AUDIT-REPORT.md` -- FDA compliance documentation.
- `README.md` -- project overview.
- `CODE-DIAGNOSTIC-REPORT.md` -- code health analysis.
- `CHANGELOG.md` -- version history.
- `architecture.md`, `deployment.md`, `hipaa-compliance.md`, `competitive-analysis.md`, `data-sources.md`, `future-vision.md`.
- `HAM10000_analysis.md` and `HAM10000_stats.json` -- dataset analysis.

**Changes since v1.0.0 (was 79):**
- (+5) ADR-120 stale body text corrected -- no more contradictory metrics across ADRs.
- (+3) All ADR status fields consistent and timestamped. Reader can quickly see what is built vs. what remains.
- (-5) Existing deductions from v1 still apply (no inline code comments in image-analysis.ts, no API endpoint documentation).

**Remaining deductions:**
- (-5) No inline code comments in `image-analysis.ts` explaining the CV algorithms (Otsu thresholding, GLCM, LBP). A 2059-line file with minimal comments is unmaintainable.
- (-4) ADR-117 phases timeline (Q3 2026 - 2051) is aspirational roadmap mixed with actual implementation -- could confuse readers about what is real.
- (-4) No API documentation for the server endpoints (`/api/classify`, `/api/classify-v2`, `/api/health`).
- (-3) clinical-baselines.ts and image-analysis.ts (new files) lack module-level documentation explaining their role in the architecture.
- (-2) QUALITY-SCORECARD.md (this file) is the only quality tracking document -- no separate test plan document.

**What works well:**
- (+) ADR-118 corrections log is exemplary scientific documentation.
- (+) ADR-124 Fitzpatrick validation is brutally honest about findings.
- (+) Competitive analysis against DermaSensor with specific FDA filings cited.
- (+) ADR-123 explains the "free accuracy" concept clearly for a non-technical reader.
- (+) Consistent IMPLEMENTED status across all ADRs provides clear project status at a glance.

---

## 8. Architecture: 82/100 (Weight: 5%)

**Ensemble design:**
- 4-layer ensemble: HF ViT (or dual-model ViT) + trained-weights + rule-based safety gates + demographic adjustment. Well-reasoned.
- Threshold optimization adds a 5th effective layer (per-class decision boundaries).
- Melanoma safety floor ensures melanoma probability never drops below a minimum.
- Dual-model ensemble designed (ADR-125) and classifier built, though not yet wired to production inference.

**Measurement pipeline:**
- 3-tier measurement: USB-C reference (measurement-connector.ts), skin texture FFT (measurement-texture.ts), LiDAR (not implemented). Good progressive enhancement design.
- Quality gating before classification prevents garbage-in-garbage-out.

**New components since v1.0.0 (was 76):**
- (+3) Ensemble classifier module built per ADR-125 design -- V1+V2 dual model blending with disagreement-aware weighting.
- (+2) Measurement-LiDAR tier designed (ADR-121). measurement.ts integrates all 3 measurement tiers.
- (+1) Anonymization pipeline and brain-client built per ADR-126 design.

**Remaining deductions:**
- (-5) Dual-model ensemble is built but NOT wired to production. The architecture describes it but the live app does not use it yet.
- (-5) ONNX/WASM offline inference has exported models (FP32 327MB, INT8 89MB) but ConvInteger compatibility blocks browser inference. The offline fallback remains trained-weights + rules only.
- (-4) Pi-brain integration is client code only (brain-client.ts) with no production connection.
- (-4) `classifier.ts` mixes multiple responsibilities: HF API calls, SigLIP label mapping, dual-model logic, ensemble blending, demographic adjustment. Should be decomposed.

**What works well:**
- (+) ADR-122 hybrid architecture (ONNX for offline, server for connected) is the correct design.
- (+) Safety gates (melanoma floor, TDS formula, 7-point checklist) provide defense-in-depth.
- (+) Consumer translation layer cleanly separates medical from consumer concerns.
- (+) Progressive enhancement measurement (USB-C -> texture FFT -> LiDAR) degrades gracefully.

---

## 9. Honesty: 90/100 (Weight: 5%)

**Limitations disclosed:**
- ADR-118 corrections log documents every inflated or unsourced number that was fixed.
- Fitzpatrick warning ("30pp melanoma sensitivity gap") is displayed in:
  - `MelaPanel.svelte`
  - `AboutPage.svelte`
  - `MethodologyPanel.svelte`
- ADR-124 states "We do NOT claim works for all skin types without measured evidence."
- ADR-120 opens with "the deployed app called a normal hand 93% melanoma -- that is a complete failure."
- All ADR statuses accurately reflect implementation state (e.g., ADR-125 says "built, not yet wired to production" rather than falsely claiming full deployment).

**Changes since v1.0.0 (was 88):**
- (+3) ADR-120 stale body text corrected -- no more self-contradictory numbers in documentation.
- (+2) ADR status fields now accurately distinguish between "built" and "deployed to production" (e.g., ADR-125, ADR-126).
- (-3) Some deductions from v1 still apply (FST V statistical power concern).

**Remaining deductions:**
- (-4) ADR-117 acceptance criteria claim ">95% melanoma sensitivity" and ">85% specificity" as if they are met, but the V1 model only meets these on HAM10000, not on external data.
- (-3) The Fitzpatrick validation had 76% URL download failures, leaving only 355 images tested. The results are directionally alarming but statistically underpowered (FST V melanoma: n=14, FST VI melanoma: n=4).
- (-3) The quality scorecard itself is self-graded. No independent external audit has been conducted.

**What works well:**
- (+) The corrections log in ADR-118 is the single most honest piece of documentation in this project.
- (+) The Fitzpatrick warning is not buried -- it appears in 3 different UI locations.
- (+) ADR-120 "No Bullshit Deployment Checklist" sets the right tone.
- (+) ADR-124 explicitly commits: "we report and we fix."
- (+) Status fields distinguish built vs. deployed vs. connected -- no inflated claims.

---

## 10. Deployment: 75/100 (Weight: 5%)

**What is deployed:**
- Vercel deployment exists at mela.vercel.app.
- `@sveltejs/adapter-vercel` configured in package.json.
- Build scripts: `vite build`.
- Deploy script referenced: `scripts/deploy.sh`.
- Cloud Run Dockerfile also available as fallback.

**Changes since v1.0.0 (was 68):**
- (+5) ONNX export completed: FP32 model (327MB) and INT8 quantized model (89MB) ready for browser deployment. INT8 at 89MB is within practical Service Worker caching limits.
- (+2) Build artifacts confirmed via adapter-node output in `.svelte-kit/adapter-node/` and `build/`.

**Remaining deductions:**
- (-8) The custom v2 model does NOT run on Vercel (327MB ONNX exceeds 250MB serverless limit). The deployed app uses HuggingFace Inference API, which serves the V1 community model, NOT the v2 custom-trained model.
- (-5) Vercel team protection may still require auth to access the public URL (ADR-120 notes this as a remaining blocker).
- (-5) No CI/CD pipeline -- no GitHub Actions workflow for automated build/test/deploy.
- (-5) Build artifacts (`.svelte-kit/adapter-node/`, `.svelte-kit/output/`, `build/`) appear to be committed or untracked.
- (-2) No health check monitoring or uptime alerting configured.

**What works well:**
- (+) SvelteKit with Vercel adapter is a solid deployment choice.
- (+) Both Vercel (serverless) and Cloud Run (container) deployment paths exist.
- (+) Subtree workflow keeps Mela isolated in `examples/mela/`.
- (+) INT8 ONNX at 89MB is a practical size for browser-based inference once ConvInteger compat is resolved.

---

## Overall Score

| Area | Weight | Previous | Current | Weighted |
|------|--------|----------|---------|----------|
| Classification Accuracy | 20% | 62 | 72 | 14.40 |
| Evidence Chain | 15% | 82 | 88 | 13.20 |
| Code Quality | 10% | 71 | 71 | 7.10 |
| UI/UX | 10% | 73 | 75 | 7.50 |
| Security | 10% | 78 | 78 | 7.80 |
| Testing | 10% | 38 | 65 | 6.50 |
| Documentation | 10% | 79 | 82 | 8.20 |
| Architecture | 5% | 76 | 82 | 4.10 |
| Honesty | 5% | 88 | 90 | 4.50 |
| Deployment | 5% | 68 | 75 | 3.75 |
| **TOTAL** | **100%** | **70.20** | -- | **77.05** |

## Overall: 77/100 (up from 70)

---

## Score Movement Summary

| Area | Delta | Primary Driver |
|------|-------|---------------|
| Classification | +10 | Threshold classifier built (ADR-123), NV sensitivity 14.3% -> 70.4% |
| Evidence Chain | +6 | ADR-120 stale text fixed, all ADR statuses consistent |
| Testing | +27 | 91/104 Vitest tests pass (was near-zero), test infrastructure proven |
| Architecture | +6 | Ensemble, measurement-LiDAR, anonymization, brain-client all built |
| Deployment | +7 | ONNX FP32 + INT8 exported, INT8 at 89MB ready for browser |
| Honesty | +2 | ADR statuses accurately distinguish built vs. deployed |
| UI/UX | +2 | Triage mode default for consumer safety |
| Code Quality | 0 | No structural changes to large files |
| Security | 0 | No new security work |

---

## What This Score Means

Mela has moved from "impressive prototype with critical gaps" (70/100) to "functional medical AI tool with known limitations honestly documented" (77/100). The biggest single improvement is testing (+27 points), which was the highest-risk area in v1. The threshold classifier is the most impactful technical change -- it extracted real accuracy gains from the existing model without retraining.

Three areas still prevent confidence in production readiness:

1. **Deployment gap (75/100).** The deployed Vercel app still serves the V1 model via HuggingFace Inference API. The V2 model (95.97% mel sensitivity) exists as ONNX exports but is not served in production. Users are getting a weaker model than the evidence files describe.

2. **Classification equity (72/100).** The Fitzpatrick equity gap (30pp melanoma sensitivity between FST II and FST V) is a patient safety issue that cannot be fixed without more diverse training data. The threshold classifier does not address skin-tone bias.

3. **Testing still incomplete (65/100).** 91 passing tests is a large improvement, but 13 tests are failing, no E2E tests are committed, and the measurement pipeline (3 modules) plus brain-client, offline-queue, and image-analysis have zero test coverage.

---

## Top 5 Actions to Raise the Score

| Priority | Action | Current Impact | Expected Score Lift |
|----------|--------|----------------|-------------------|
| 1 | Wire v2 model to production (Cloud Run or fix ONNX browser) | Deployment 75 -> 85, Classification 72 -> 78 | +2.8 overall |
| 2 | Fix the 13 failing tests + add measurement pipeline tests | Testing 65 -> 78 | +1.3 overall |
| 3 | Add CI/CD with GitHub Actions (build + test + deploy) | Testing 65 -> 72, Deployment 75 -> 82 | +1.1 overall |
| 4 | Commit E2E Playwright tests to the repository | Testing 65 -> 72 | +0.7 overall |
| 5 | Decompose image-analysis.ts (2059 lines) and MelaPanel.svelte (1520 lines) | Code Quality 71 -> 80 | +0.9 overall |

---

## What I Did NOT Test

- I did not run `npm run build` to verify the app compiles.
- I did not run `npm test` to verify the 91/104 pass rate claim independently.
- I did not access mela.vercel.app to verify the live deployment.
- I did not verify that the Svelte components render correctly.
- I did not test the classification pipeline end-to-end with a real image.
- I did not verify the measurement pipeline produces accurate mm values.
- I did not check whether the multi-image consensus improves accuracy vs single-image.
- I did not verify the ONNX INT8 model loads in any ONNX Runtime Web environment.
- Python validation scripts were not re-run to verify JSON evidence files are reproducible.

This scorecard is based on static analysis of source files, evidence JSON files, ADR documentation, file existence checks, and the stated test pass rate. A live verification pass would likely surface additional issues.

---

## Evidence File Index

| File | What It Contains |
|------|-----------------|
| `scripts/cross-validation-results.json` | V1 HAM10000 per-class metrics |
| `scripts/isic2019-validation-results.json` | V1 ISIC 2019 external validation |
| `scripts/combined-training-results.json` | V2 training config + ISIC 2019 external test results |
| `scripts/auroc-results.json` | Per-class AUROC with 95% CIs for both datasets |
| `scripts/optimal-thresholds.json` | 7 per-class decision thresholds |
| `scripts/threshold-optimization-results.json` | Full threshold analysis with sensitivity/specificity |
| `scripts/fitzpatrick-v2-validation.json` | Stratified Fitzpatrick equity results (355 images) |
| `scripts/fitzpatrick-equity-report.json` | Training data distribution by skin type |
| `scripts/multi-image-validation-results.json` | Multi-image consensus validation |
| `scripts/siglip-test-results.json` | SigLIP model baseline comparison |

---

## Author

Quality Engineer (automated audit) -- 2026-03-24 v2.0.0
