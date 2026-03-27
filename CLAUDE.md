Updated: 2026-03-25 | Version 1.0.0
Created: 2026-03-25

# Mela -- Claude Code Project Instructions

**Read `docs/PROJECT-HISTORY.md` first.** It contains the full project context, timeline, architecture, evidence chain, and known issues.

---

## Critical Rules

### 1. Every Number Must Cite a JSON Evidence File

The FDA audit (2026-03-23) found that previous claims of "91.3% cross-dataset" and "96.2% sensitivity" were fabricated -- aspirational numbers presented as measured results. See `docs/FDA-AUDIT-REPORT.md`.

**If you write a number in documentation, code comments, or UI text, it MUST cite its source file.** Example:

```
95.97% melanoma sensitivity (Source: scripts/combined-training-results.json)
```

If no evidence file contains the number, do not claim it. "Not yet measured" is a valid answer.

The primary evidence files are in `scripts/`:

| File | What it contains |
|---|---|
| `combined-training-results.json` | V2 model -- headline claims (95.97% mel sens, AUROC 0.960) |
| `cross-validation-results.json` | V1 model -- HAM10000-only results (98.22% mel sens) |
| `isic2019-validation-results.json` | V1 model on external data (61.62% mel sens) |
| `auroc-results.json` | Per-class AUROC with CIs |
| `threshold-optimization-results.json` | Per-class threshold details |
| `ensemble-validation-results.json` | V1+V2 ensemble (99.40% mel sens) |
| `fitzpatrick-v2-validation.json` | Fitzpatrick equity (30pp gap, DANGEROUS) |
| `clinical-metrics-full.json` | PPV, NPV, NNB, LR+, LR-, ECE |
| `confidence-intervals.json` | Wilson 95% CIs |

### 2. Version Numbering -- Three Places

When bumping the version, update ALL THREE:

1. `package.json` -- `"version": "x.y.z"`
2. `src/routes/+page.svelte` -- the version badge in the header (e.g., `v0.9.4`)
3. `README.md` -- the version line near the top (e.g., `**Version 0.9.4**`)

The `vite.config.ts` injects `__APP_VERSION__` from package.json, but the +page.svelte header has a hardcoded version string that must also be updated.

### 3. Normal Skin Must Show Green "Healthy" -- Not Amber Warning

This was a critical bug from v0.1 through v0.9.0. When a user photographs normal skin with no lesion, the system MUST show a green checkmark with "Your skin looks healthy!" -- NOT an amber warning triangle.

The mechanism: classification errors with the `healthy_skin:` prefix are routed to the green healthy-skin UI in `MelaPanel.svelte`. If you modify the lesion gate, spot detector, or error handling, verify this path still works.

### 4. Deployment Workflow

```bash
# From examples/mela/ directory:
# 1. Update version in package.json AND +page.svelte AND README.md
# 2. Commit to the RuVector monorepo
# 3. Subtree split and push to stuinfla/Mela:
git subtree split --prefix=examples/mela -b deploy
git push fork deploy:main --force
# 4. ALWAYS verify the build:
vercel ls
# 5. If it fails, check the logs:
vercel inspect <deployment-url> --logs
```

Or use the deploy script: `bash scripts/deploy-verified.sh patch`

**ALWAYS check `vercel ls` after pushing.** The build was broken for hours because of a bad import and nobody checked.

The 85MB ONNX model makes git pushes slow. For small changes, consider using the GitHub API for individual file updates instead of pushing the entire tree.

### 5. Testing

```bash
# Run all tests:
npx vitest run

# Run a specific test file:
npx vitest run tests/classifier.test.ts

# Run tests in watch mode:
npx vitest
```

Tests use vitest with an ImageData polyfill for Node.js (browser APIs are not available in the test environment). If you add a test that needs ImageData, follow the pattern in `tests/classifier.test.ts`.

### 6. Build

```bash
# Local development:
npm run dev

# Production build (also verifies compilation):
npm run build

# Type checking:
npm run check
```

The build externalizes `@ruvector/cnn`, `onnxruntime-node`, and `sharp` (see `vite.config.ts`). These are server-side only dependencies.

---

## Architecture Quick Reference

### Pipeline (11 steps)

```
Camera/Upload -> Quality Check -> Lesion Gate -> Segmentation -> Feature Extraction
-> 4-Layer Ensemble -> Threshold -> Meta-Classifier -> Bayesian Risk -> Consumer Translation -> Display
```

Full details: `docs/PIPELINE-EXPLAINER.md`

### Classification Strategies (priority order)

1. **Custom local ViT** (ONNX available): 70% ViT + 15% trained-weights + 15% rules
2. **Dual HF API** (both models respond): 50% HF + 30% trained-weights + 20% rules
3. **Single HF API** (one model responds): 60% HF + 25% trained-weights + 15% rules
4. **Offline fallback** (no API, no ONNX): 60% trained-weights + 40% rules

### Safety Gates (always run)

- Melanoma floor: 2+ suspicious ABCDE criteria -> melanoma >= 15%
- TDS override: TDS > 5.45 -> malignant >= 30%
- Multi-image melanoma gate: any image > 60% melanoma -> signal preserved in consensus

### Key Modules

| Module | Responsibility |
|---|---|
| `classifier.ts` | Ensemble orchestration, strategy selection, HF API calls |
| `image-analysis.ts` | CV pipeline: segmentation, features, lesion detection |
| `consumer-translation.ts` | Medical -> plain English translation |
| `meta-classifier.ts` | Neural + clinical agreement scoring |
| `risk-stratification.ts` | Bayesian post-test probability |
| `threshold-classifier.ts` | Per-class ROC-optimized thresholds |
| `spot-detector.ts` | Two-pass lesion presence detection |
| `MelaPanel.svelte` | Main UI orchestrator |

---

## Known Code Quality Issues

These are documented in `docs/CODE-DIAGNOSTIC-REPORT.md` and `docs/QUALITY-SCORECARD.md`:

1. **`image-analysis.ts` is 2,059 lines.** Needs splitting into ~9 focused modules. See the diagnostic report for the recommended split.

2. **`MelaPanel.svelte` is ~2,000 lines.** Extract ScanResultHero, MedicalDetails, HistoryView, SettingsView, AnalysisProgress.

3. **Duplicate implementations.** segmentLesion exists in both preprocessing.ts and image-analysis.ts. cosineSimilarity exists in both classifier.ts and multi-image.ts. Morphological ops duplicated.

4. ~~**`any` type in classify-local/+server.ts.**~~ FIXED. ONNX session properly typed in inference-offline.ts.

5. ~~**No file upload validation on API endpoints.**~~ FIXED. Magic byte validation, MIME checks, size limits.

6. ~~**No CSP or CORS headers.**~~ FIXED. hooks.server.ts adds CSP, HSTS, X-Frame-Options, nosniff, Referrer-Policy, Permissions-Policy. Rate limiting on all classify endpoints.

---

## What NOT To Do

- **NEVER use `git add -A`** -- it grabbed 6,216 cached training images once
- **NEVER edit files via GitHub API with sed** -- use exact file content
- **NEVER claim something works without checking `vercel ls` and `vercel inspect --logs`**
- **NEVER assume Vercel deployed** -- VERIFY with browser or curl
- **NEVER write a performance number without citing its JSON evidence file**
- **NEVER lower the melanoma safety gate thresholds** without documenting the clinical rationale
- **NEVER commit `.env` files** or HF_TOKEN values
- **NEVER commit the model weight directories** (`scripts/mela-classifier/`, `scripts/mela-classifier-v2/`, `scripts/mela-onnx*/`)

---

## Connections

| System | URL / Identifier |
|---|---|
| Live app | https://mela-app.vercel.app |
| GitHub repo | https://github.com/stuinfla/Mela |
| Vercel project | https://vercel.com/stuart-kerrs-projects/mela |
| Vercel team | stuart-kerrs-projects |
| Vercel user | sikerr-6092 |
| HuggingFace model | https://huggingface.co/stuartkerr/mela-classifier |
| Pi-brain | https://pi.ruv.io (namespace: mela) |
| Monorepo location | examples/mela/ within RuVector |

---

## Priority Work (as of v1.0.0)

1. ~~**Wire V2 model to production**~~ -- DONE. V2 ONNX (85MB INT8) is Priority 1 in classifier.ts. 70% ONNX + 15% trained-weights + 15% rules.
2. ~~**Fix failing tests**~~ -- DONE. 170/170 tests passing (14 files).
3. ~~**Add upload validation to API endpoints**~~ -- DONE. Magic byte validation, rate limiting, size limits on all endpoints.
4. **Decompose large files** -- image-analysis.ts (~2000 lines) and MelaPanel.svelte (~2000 lines) still need splitting.
5. **Wire V1+V2 ensemble to production** -- ensemble.ts exists but is not in the inference path. Would boost to 99.4% mel sensitivity.
6. **Validate ONNX INT8 model** -- Run full ISIC 2019 validation suite against the deployed INT8 model specifically.
7. **Fix remaining 5 type errors** -- ort null check, customModelAvailable declaration ordering.

---

## ADRs

13 ADRs in `docs/adr/`. Read ADR-127 (production readiness gaps) and ADR-128 (clinical validation requirements) for the current state assessment. Read ADR-129 (Bayesian risk stratification) for the latest architectural addition.

All ADR statuses are current as of 2026-03-25.

---

## Documentation Index

| Document | Purpose |
|---|---|
| `docs/PROJECT-HISTORY.md` | Complete project context -- START HERE |
| `docs/PIPELINE-EXPLAINER.md` | Step-by-step pipeline with function names and line numbers |
| `docs/FDA-AUDIT-REPORT.md` | The audit that caught fabricated claims |
| `docs/TECHNICAL-REPORT.md` | Full classification pipeline and validation |
| `docs/QUALITY-SCORECARD.md` | Quality assessment: 77/100 |
| `docs/CODE-DIAGNOSTIC-REPORT.md` | Code quality analysis: 72/100 |
| `docs/CHANGELOG.md` | Version history |
| `docs/DUAL-MODEL-ARCHITECTURE.md` | V1+V2 ensemble design |
| `docs/architecture.md` | Component architecture |
| `docs/deployment.md` | Cloud deployment plan (aspirational) |
| `docs/competitive-analysis.md` | DermaSensor, Nevisense, MelaFind comparison |
| `docs/data-sources.md` | HAM10000, ISIC 2019, Fitzpatrick17k details |
| `docs/hipaa-compliance.md` | HIPAA compliance considerations |
| `docs/future-vision.md` | Long-term roadmap |
| `docs/research/lesion-detection-approach.md` | Two-pass spot detection research |
| `README.md` | Public-facing project documentation |
