Updated: 2026-03-24 16:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-127: Production Readiness — Closing Every Gap

## Status: PROPOSED | Last Updated: 2026-03-24 16:00 EST

## Context

Mela v0.8.0 scores 74/100 in an honest effectiveness evaluation. The architecture is sound and the science is real, but 7 specific gaps prevent production readiness. This ADR addresses every one.

The gaps are ordered by impact — fixing #1 alone moves the needle more than fixing #3-7 combined.

---

## Gap 1: V2 Model Not Serving in Production (CRITICAL)

**Problem:** The deployed app at mela.vercel.app uses the v1 HuggingFace model (`stuartkerr/mela-classifier` — trained on HAM10000 only, 61.6% mel sensitivity on external data). The v2 combined model (95.97% on external data) exists only locally at `scripts/mela-classifier-v2/best/`.

**Impact:** Every user today gets the WORSE model. This is the single most impactful gap.

**Fix (2 options, choose one):**

### Option A: Upload v2 to HuggingFace (2 hours)
1. Push v2 safetensors + config to `stuartkerr/mela-classifier` (overwrite v1)
2. Or create `stuartkerr/mela-classifier-v2` and update the HF_MODEL_1 env var on Vercel
3. Test the deployed endpoint returns v2 probabilities

### Option B: Deploy ONNX INT8 to Vercel (4 hours)
1. Copy the 89MB INT8 ONNX to `static/models/mela-v2-int8.onnx`
2. Wire `inference-orchestrator.ts` as the primary classification path
3. Register `sw-model-cache.js` Service Worker in app.html
4. Vercel serves the ONNX file as a static asset (no serverless size limit)
5. Test offline inference works in Chrome and Safari

**Recommendation:** Do Option A first (immediate fix), then Option B (architectural improvement).

**Validation:** After deploying, run 10 test images through the live URL and verify mel sensitivity matches v2 expectations.

---

## Gap 2: No Phone-Camera Validation (HIGH)

**Problem:** All training and testing used dermoscopy images. Consumers will use phone cameras — fundamentally different images (no polarized light, different color balance, hair/shadow artifacts, variable distance).

**Impact:** Classification accuracy on phone photos is completely unknown.

**Fix (1 day):**

### Phase 1: Collect phone photos (2 hours)
1. Photograph 50 real skin lesions (moles, freckles, any spots) with an iPhone
2. Include a USB-C cable in frame for measurement validation
3. Vary: lighting, distance, angle, skin tone (ask 3-5 people)
4. Label each: what a dermatologist would call it (or "unknown")

### Phase 2: Test the model (2 hours)
1. Run v2 model on the 50 phone photos
2. Report: detection rate, false positive rate, quality gate rejection rate
3. Compare phone-photo accuracy to dermoscopy accuracy
4. If accuracy drops >15%: implement phone-specific preprocessing (color normalization, shadow removal)

### Phase 3: Test-time augmentation for phone photos (2 hours)
1. For each phone photo, generate 5 augmented views (crop, rotate, brightness, contrast)
2. Run multi-image consensus on the augmented views
3. Compare: single phone photo vs 5-view consensus

**Validation:** Publish `scripts/phone-camera-validation-results.json` with all metrics.

---

## Gap 3: ISIC Test Set Isn't Truly External (MEDIUM)

**Problem:** The "external" ISIC 2019 test set is a 15% holdout from `akinsanyaayomide/skin_cancer_dataset_balanced_labels_2` — the same dataset used for training. While images weren't seen during training, they come from the same data pipeline.

**Impact:** The 95.97% claim is weaker than "truly external" because train/test come from the same distribution pipeline.

**Fix (1 day):**

1. Download PH2 dataset (200 dermoscopy images, University of Porto) — completely independent
2. Download ISIC 2020 Challenge dataset (if available on HuggingFace)
3. Run v2 model on these genuinely external datasets
4. Report mel sensitivity, specificity, AUROC
5. If >5pp drop from 95.97%: investigate and document

**Validation:** `scripts/truly-external-validation-results.json` with dataset provenance.

---

## Gap 4: No Real User Testing (MEDIUM)

**Problem:** All UX evaluation was automated (Playwright + screenshots). No real human has used the app end-to-end.

**Fix (1 week):**

1. Recruit 10 people (mix of ages, tech comfort levels)
2. Task: "Open Mela on your phone, photograph a mole, see the result"
3. Observe: where they get stuck, what confuses them, what reassures them
4. Record: time to first classification, number of retakes, quality gate rejections
5. Ask: "Would you trust this result? Why or why not?"
6. Fix the top 3 issues found

**Validation:** User study report with quantitative metrics + qualitative findings.

---

## Gap 5: No CI/CD Pipeline (MEDIUM)

**Problem:** Every deployment is a manual `git subtree split + push --force`. No tests run automatically. No build verification.

**Fix (2 hours):**

1. Create `.github/workflows/ci.yml`:
   - Trigger: push to main, pull requests
   - Jobs: npm install, npm run build, npx vitest run
   - Fail PR if tests fail or build breaks
2. Create `.github/workflows/deploy.yml`:
   - Trigger: push to main (after CI passes)
   - Auto-deploy to Vercel via Vercel CLI or GitHub integration
3. Add status badges to README

**Validation:** Push a commit, verify CI runs, verify auto-deploy works.

---

## Gap 6: No Confidence Intervals on Sensitivity (LOW)

**Problem:** The headline "95.97%" has no confidence interval. Bootstrap CI exists for AUROC but not for sensitivity/specificity.

**Fix (1 hour):**

1. Add bootstrap CI (1000 iterations) to the combined-training validation
2. Report: "95.97% melanoma sensitivity (95% CI: XX.X% - XX.X%)"
3. Add CI to the README headline and trust banner
4. Repeat for specificity and all-cancer sensitivity

**Validation:** Update `combined-training-results.json` with CI fields.

---

## Gap 7: Camera Unreliable on iPhone (LOW-MEDIUM)

**Problem:** The `capture="environment"` fix was built but iPhone camera behavior is unpredictable across Safari versions. getUserMedia may fail silently.

**Fix (2 hours):**

1. Test on real iPhone (not simulator):
   - Safari 17+: does capture="environment" open camera directly?
   - Chrome iOS: same test
   - Denied permission: does error state show Upload Photo fallback?
2. If capture="environment" doesn't work reliably:
   - Make Upload Photo the PRIMARY action (not secondary)
   - Remove getUserMedia entirely — just use native file input with capture
   - This is simpler and more reliable on iOS

**Validation:** Test on 3 different iPhones (different models/OS versions).

---

## Implementation Priority

| Gap | Fix Effort | Score Impact | Priority |
|-----|-----------|-------------|----------|
| **#1 V2 model in prod** | 2 hours | +10 | **DO IMMEDIATELY** |
| **#5 CI/CD** | 2 hours | +3 | **DO NEXT** |
| **#6 Confidence intervals** | 1 hour | +2 | **DO NEXT** |
| **#7 Camera testing** | 2 hours | +3 | **DO NEXT** |
| **#2 Phone-camera validation** | 1 day | +5 | **DO THIS WEEK** |
| **#3 Truly external validation** | 1 day | +3 | **DO THIS WEEK** |
| **#4 User testing** | 1 week | +5 | **DO THIS MONTH** |

**Completing #1 + #5 + #6 + #7 (7 hours of work) would move the score from 74 to ~85.**

---

## Success Criteria

| Criterion | Target | How to Verify |
|-----------|--------|--------------|
| V2 model serving in production | mela.vercel.app returns v2 probabilities | Test with known image |
| Phone-camera accuracy | >80% mel sensitivity on 50 phone photos | phone-camera-validation-results.json |
| Truly external validation | >85% mel sensitivity on PH2/ISIC2020 | truly-external-validation-results.json |
| CI/CD operational | Tests run on every push | GitHub Actions badge green |
| Confidence intervals | 95% CI on all headline numbers | Updated JSON + README |
| Camera works on iPhone | Successful capture on 3 iPhone models | Manual test log |
| User study complete | 10 users, <3 min to first classification | User study report |

---

## Author
Stuart Kerr + Claude Flow
