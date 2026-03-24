Updated: 2026-03-24 10:30:00 EST | Version 1.0.0
Created: 2026-03-24

# Dr. Agnes Quality Scorecard

**Evaluator**: Quality Engineer (automated audit)
**Date**: 2026-03-24
**Scope**: Complete Dr. Agnes application in `examples/dragnes/`
**Methodology**: Every score backed by specific file evidence. No rounding up. If uncertain, score lower.

---

## 1. Classification Accuracy: 62/100 (Weight: 20%)

**What is measured:**
- V1 model (HAM10000-only): 98.2% mel sensitivity on HAM10000 test split (n=225), 80.4% specificity. Source: `cross-validation-results.json`.
- V2 model (combined HAM10000 + ISIC 2019): 95.97% mel sensitivity on ISIC 2019 external test (n=621), 80.0% specificity. Source: `combined-training-results.json`.
- AUROC computed: mel AUROC 0.926 (HAM10000), 0.960 (ISIC 2019). Source: `auroc-results.json`.
- Threshold optimization done: mel threshold 0.6204 yields 93.88% sensitivity, 85.34% specificity. Source: `threshold-optimization-results.json`.

**What is claimed vs reality:**
- ADR-118 claims were corrected on 2026-03-23 (corrections log in the ADR). Previously claimed numbers like "96.2%" had no source. This correction is excellent honesty.
- The v2 combined model overall accuracy on ISIC 2019 external data is 76.4% -- not the 85%+ target.
- The v1 model drops to 61.6% mel sensitivity on ISIC 2019 external data. This was honestly reported.
- Nevi (nv) sensitivity on v2 is catastrophically low: 14.3% (89/621). The model mislabels most common moles.

**Deductions:**
- (-10) Overall accuracy 76.4% misses 85% target on external data.
- (-10) Nevi sensitivity 14.3% means most common moles will be misclassified, generating false alarms or wrong benign labels.
- (-8) Fitzpatrick validation shows mel sensitivity drops to 50% on FST V skin (n=14). This is a patient safety issue.
- (-5) V1 model (still served on Vercel via HF API) has 61.6% mel sensitivity on external data.
- (-5) No prospective validation on real clinical images from phone cameras.

**What works well:**
- (+) V2 mel sensitivity 95.97% on external data is genuinely strong.
- (+) AUROC 0.960 for melanoma on ISIC 2019 is competitive.
- (+) Focal loss with class-specific alpha weights is a correct architectural choice for imbalanced medical data.
- (+) Numbers are now honestly reported with evidence file citations.

---

## 2. Evidence Chain: 82/100 (Weight: 15%)

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

**Deductions:**
- (-8) ADR-118 corrections log documents that the original "96.2%" and "0.936 AUROC" numbers had no source. The chain was broken and had to be repaired.
- (-5) ADR-120 still contains stale numbers in the body text ("96.2% mel sensitivity, 73.1% specificity") that were not corrected when the status was updated.
- (-5) No validation-results.json for the threshold-classifier.ts itself -- the thresholds are computed but no end-to-end test proves they improve outcomes in the deployed pipeline.

**What works well:**
- (+) ADR-118 corrections log is a model of scientific honesty -- every change documented with reason and corrector.
- (+) JSON evidence files include sample sizes, confidence intervals (AUROC), per-class breakdowns.
- (+) Fitzpatrick validation includes download failure rate (76%) as a data quality caveat.

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
- 28 TypeScript files in `src/lib/dragnes/` -- well-organized by concern (types, preprocessing, classification, measurement, translation, privacy, witness).
- 16 Svelte components -- appropriate separation.
- `DrAgnesPanel.svelte` at 1520 lines is extremely large -- should be decomposed.

**Deductions:**
- (-10) `image-analysis.ts` at 2059 lines violates the 500-line rule and is difficult to maintain.
- (-8) `DrAgnesPanel.svelte` at 1520 lines is a monolithic component with scan, history, settings, learn views crammed together.
- (-5) `classifier.ts` at 973 lines is borderline -- the ensemble logic, HF API logic, SigLIP label mapping, and dual-model logic are all in one file.
- (-3) Only 2 test files exist (`classifier.test.ts` at 509 lines, `benchmark.test.ts`). No tests for `consumer-translation.ts`, `threshold-classifier.ts`, `measurement.ts`, `image-quality.ts`, `multi-image.ts`, or `abcde.ts`.
- (-3) `deployment-runbook.ts` contains hardcoded GCP deployment commands as TypeScript strings -- this is an odd choice vs. a shell script.

**What works well:**
- (+) Pure TypeScript with typed arrays for image processing -- no external CV library dependency.
- (+) Clean separation between classification engine, consumer translation, and UI.
- (+) Vitest configured and test infrastructure in place.

---

## 4. UI/UX: 73/100 (Weight: 10%)

**Mobile-first:**
- `DermCapture.svelte` (721 lines) implements camera capture with multi-photo mode, thumbnail strip, counter badge.
- Tailwind CSS with responsive classes throughout.
- Bottom navigation bar with Scan/History/Learn/Settings tabs.
- PWA manifest present for installability.

**Consumer-friendliness:**
- Consumer translation maps all 7 classes to plain English with 4 risk levels (green/yellow/orange/red).
- Each risk level has specific action text ("Monitor for changes", "See a dermatologist within 2 weeks").
- ABCDE scores rendered visually in `ABCDEChart.svelte`.

**Deductions:**
- (-8) No dedicated E2E Playwright test files found in the project (the ADR-120 status claims Playwright E2E was run, but no test script is committed).
- (-7) The Settings UI exposes a threshold mode selector ("screening"/"triage"/"default") that a consumer would not understand.
- (-5) Medical terminology still present in ClassificationResult component (ICD-10 codes, ensemble weight bars, model provenance section).
- (-4) Body map drawer, lesion timeline, referral letter, and calibration chart components exist but are untested.
- (-3) No loading state documentation or accessibility audit.

**What works well:**
- (+) Fitzpatrick warning is prominently displayed in the UI (DrAgnesPanel line 842-845, AboutPage line 84, MethodologyPanel line 442).
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

## 6. Testing: 38/100 (Weight: 10%)

**What exists:**
- `tests/classifier.test.ts` (509 lines) -- tests preprocessing, ABCDE scoring, privacy pipeline, CNN fallback. Uses vitest with ImageData polyfill.
- `tests/benchmark.test.ts` -- performance benchmarks.
- Multiple validation scripts produced JSON evidence files (cross-validation, ISIC 2019, AUROC, threshold optimization, Fitzpatrick equity).

**What does NOT exist:**
- No E2E test scripts committed (Playwright tests were claimed in ADR-120 but no `.spec.ts` or `playwright.config.ts` found in the project).
- No unit tests for: `consumer-translation.ts`, `threshold-classifier.ts`, `measurement.ts`, `measurement-connector.ts`, `measurement-texture.ts`, `image-quality.ts`, `multi-image.ts`, `brain-client.ts`, `offline-queue.ts`, `federated.ts`.
- No integration tests for the full classify-then-translate pipeline.
- No visual regression tests for Svelte components.
- No test for the consumer result -- does the UI actually show "Common mole" when classification returns "nv"?

**Deductions:**
- (-25) 26 out of 28 TypeScript modules have zero unit test coverage. Only `classifier.ts`, `abcde.ts`, `preprocessing.ts`, and `privacy.ts` have any test coverage.
- (-15) No committed E2E tests. The ADR claims Playwright was run but no evidence in the repo.
- (-12) Validation scripts (Python) that produced the JSON results are not tested themselves -- no way to verify the metrics pipeline is correct.
- (-10) No CI/CD pipeline configuration found (no `.github/workflows/`, no `vercel.json` test step).

**What works well:**
- (+) The vitest infrastructure is set up and working.
- (+) Validation JSON files serve as integration test artifacts.
- (+) classifier.test.ts polyfills ImageData for Node.js -- a thoughtful detail.

---

## 7. Documentation: 79/100 (Weight: 10%)

**What exists:**
- 11 ADR files covering architecture, validation, consumer screening, deployment, measurement, ONNX, thresholds, Fitzpatrick equity, dual-model, and pi-brain.
- `TECHNICAL-REPORT.md` -- technical documentation.
- `FDA-AUDIT-REPORT.md` -- FDA compliance documentation.
- `README.md` -- project overview.
- `CODE-DIAGNOSTIC-REPORT.md` -- code health analysis.
- `CHANGELOG.md` -- version history.
- `architecture.md`, `deployment.md`, `hipaa-compliance.md`, `competitive-analysis.md`, `data-sources.md`, `future-vision.md`.
- `HAM10000_analysis.md` and `HAM10000_stats.json` -- dataset analysis.

**Deductions:**
- (-8) ADR-120 body text still contains stale metrics ("96.2% mel sensitivity, 73.1% specificity") that contradict the corrected ADR-118.
- (-5) No inline code comments in `image-analysis.ts` explaining the CV algorithms (Otsu thresholding, GLCM, LBP). A 2059-line file with minimal comments is unmaintainable.
- (-4) ADR-117 has "Proposed" status (now corrected) but the implementation phases timeline (Q3 2026 - 2051) is aspirational roadmap mixed with actual implementation -- could confuse readers about what is real.
- (-4) No API documentation for the server endpoints (`/api/classify`, `/api/classify-v2`, `/api/health`).

**What works well:**
- (+) ADR-118 corrections log is exemplary scientific documentation.
- (+) ADR-124 Fitzpatrick validation is brutally honest about findings.
- (+) Competitive analysis against DermaSensor with specific FDA filings cited.
- (+) ADR-123 explains the "free accuracy" concept clearly for a non-technical reader.

---

## 8. Architecture: 76/100 (Weight: 5%)

**Ensemble design:**
- 4-layer ensemble: HF ViT (or dual-model ViT) + trained-weights + rule-based safety gates + demographic adjustment. Well-reasoned.
- Threshold optimization adds a 5th effective layer (per-class decision boundaries).
- Melanoma safety floor ensures melanoma probability never drops below a minimum.

**Measurement pipeline:**
- 3-tier measurement: USB-C reference (measurement-connector.ts), skin texture FFT (measurement-texture.ts), LiDAR (not implemented). Good progressive enhancement design.
- Quality gating before classification prevents garbage-in-garbage-out.

**Deductions:**
- (-8) The dual-model ensemble (ADR-125) is designed but NOT implemented. The architecture documents describe it but the code does not use it.
- (-6) ONNX/WASM offline inference (ADR-122) is designed but NOT implemented. The offline fallback is trained-weights + rules only -- no ViT model runs offline.
- (-5) `classifier.ts` mixes multiple responsibilities: HF API calls, SigLIP label mapping, dual-model logic, ensemble blending, demographic adjustment. Should be decomposed.
- (-5) Pi-brain integration is client code only (brain-client.ts) with no production connection.

**What works well:**
- (+) ADR-122 hybrid architecture (ONNX for offline, server for connected) is the correct design.
- (+) Safety gates (melanoma floor, TDS formula, 7-point checklist) provide defense-in-depth.
- (+) Consumer translation layer cleanly separates medical from consumer concerns.

---

## 9. Honesty: 88/100 (Weight: 5%)

**Limitations disclosed:**
- ADR-118 corrections log documents every inflated or unsourced number that was fixed.
- Fitzpatrick warning ("30pp melanoma sensitivity gap") is displayed in:
  - `DrAgnesPanel.svelte` (line 842-845)
  - `AboutPage.svelte` (line 84)
  - `MethodologyPanel.svelte` (line 442)
- ADR-124 states "We do NOT claim works for all skin types without measured evidence."
- ADR-120 opens with "the deployed app called a normal hand 93% melanoma -- that is a complete failure."

**Deductions:**
- (-5) ADR-120 body text still contains the stale "96.2% mel sensitivity, 73.1% specificity" numbers from before the corrections, which contradicts ADR-118.
- (-4) ADR-117 acceptance criteria claim ">95% melanoma sensitivity" and ">85% specificity" as if they are met, but the V1 model only meets these on HAM10000, not on external data.
- (-3) The Fitzpatrick validation had 76% URL download failures, leaving only 355 images tested. The results are directionally alarming but statistically underpowered (FST V melanoma: n=14, FST VI melanoma: n=4).

**What works well:**
- (+) The corrections log in ADR-118 is the single most honest piece of documentation in this project.
- (+) The Fitzpatrick warning is not buried -- it appears in 3 different UI locations.
- (+) ADR-120 "No Bullshit Deployment Checklist" sets the right tone.
- (+) ADR-124 explicitly commits: "we report and we fix."

---

## 10. Deployment: 68/100 (Weight: 5%)

**What is deployed:**
- Vercel deployment exists at dragnes.vercel.app.
- `@sveltejs/adapter-vercel` configured in package.json.
- Build scripts: `vite build`.
- Deploy script referenced: `scripts/deploy.sh`.
- Cloud Run Dockerfile also available as fallback.

**Deductions:**
- (-10) Vercel team protection may still require auth to access the public URL (ADR-120 notes this as a remaining blocker).
- (-8) The custom v2 model does NOT run on Vercel (327MB ONNX exceeds 250MB serverless limit). The deployed app uses HuggingFace Inference API, which is the V1 community model, NOT the v2 custom-trained model.
- (-7) No CI/CD pipeline -- no GitHub Actions workflow for automated build/test/deploy.
- (-5) Build artifacts (`.svelte-kit/adapter-node/`, `.svelte-kit/output/`, `build/`) appear to be committed or untracked.
- (-2) No health check monitoring or uptime alerting configured.

**What works well:**
- (+) SvelteKit with Vercel adapter is a solid deployment choice.
- (+) Both Vercel (serverless) and Cloud Run (container) deployment paths exist.
- (+) Subtree workflow keeps DrAgnes isolated in `examples/dragnes/`.

---

## Overall Score

| Area | Weight | Score | Weighted |
|------|--------|-------|----------|
| Classification Accuracy | 20% | 62 | 12.40 |
| Evidence Chain | 15% | 82 | 12.30 |
| Code Quality | 10% | 71 | 7.10 |
| UI/UX | 10% | 73 | 7.30 |
| Security | 10% | 78 | 7.80 |
| Testing | 10% | 38 | 3.80 |
| Documentation | 10% | 79 | 7.90 |
| Architecture | 5% | 76 | 3.80 |
| Honesty | 5% | 88 | 4.40 |
| Deployment | 5% | 68 | 3.40 |
| **TOTAL** | **100%** | -- | **70.20** |

## Overall: 70/100

---

## What This Score Means

Dr. Agnes is a genuinely impressive prototype with strong foundational architecture and unusually honest documentation. The 4-layer ensemble, consumer translation layer, and measurement pipeline demonstrate serious engineering thought. The ADR-118 corrections log and Fitzpatrick UI warnings show rare integrity for an AI medical tool.

However, three areas prevent confidence in production readiness:

1. **Testing is critically weak (38/100).** 26 of 28 TypeScript modules have zero test coverage. No committed E2E tests. No CI/CD. For a medical application, this is the single biggest risk.

2. **Classification accuracy is uneven (62/100).** The v2 model is strong on melanoma (95.97% sensitivity on ISIC 2019) but the nevi sensitivity of 14.3% means most common moles will be misclassified. The Fitzpatrick equity gap (30pp melanoma sensitivity) is a patient safety issue.

3. **Deployment gap (68/100).** The deployed Vercel app serves a different (weaker) model than the one validated in the evidence files. The v2 custom model cannot run on Vercel's serverless infrastructure.

---

## Top 5 Actions to Raise the Score

| Priority | Action | Current Impact | Expected Score Lift |
|----------|--------|----------------|-------------------|
| 1 | Add unit tests for consumer-translation, threshold-classifier, measurement, image-quality, multi-image | Testing 38 -> 55 | +1.7 overall |
| 2 | Deploy v2 model via Cloud Run or fix ONNX browser inference | Deployment 68 -> 80, Accuracy 62 -> 70 | +2.2 overall |
| 3 | Fix nevi sensitivity (14.3%) via threshold tuning or retraining | Accuracy 62 -> 72 | +2.0 overall |
| 4 | Add CI/CD with GitHub Actions (build + test + deploy) | Testing 38 -> 48, Deployment 68 -> 78 | +1.5 overall |
| 5 | Commit E2E Playwright tests to the repository | Testing 38 -> 50 | +1.2 overall |

---

## What I Did NOT Test

- I did not run `npm run build` to verify the app compiles.
- I did not run `npm test` to verify existing tests pass.
- I did not access dragnes.vercel.app to verify the live deployment.
- I did not verify that the Svelte components render correctly.
- I did not test the classification pipeline end-to-end with a real image.
- I did not verify the measurement pipeline produces accurate mm values.
- I did not check whether the multi-image consensus improves accuracy vs single-image.
- Python validation scripts were not re-run to verify JSON evidence files are reproducible.

This scorecard is based entirely on static analysis of source files, evidence JSON files, ADR documentation, and file existence checks. A live verification pass would likely surface additional issues.

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

Quality Engineer (automated audit) -- 2026-03-24
