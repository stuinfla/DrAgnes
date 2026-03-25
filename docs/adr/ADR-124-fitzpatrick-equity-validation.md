Updated: 2026-03-24 10:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-124: Fitzpatrick Skin Tone Equity -- Proving It Works for Everyone

## Status: IMPLEMENTED -- Tested, DANGEROUS Gaps Found (30pp) | Last Updated: 2026-03-24 12:00 EST
**Implementation Note**: fitzpatrick-v2-validation.json contains full results. 355 images evaluated (76% URL download failures from stale Fitzpatrick17k links). Results: melanoma sensitivity gap = 30% (FST II: 80% vs FST V: 50%) -- severity DANGEROUS. Accuracy gap = 30.55% (FST II: 46.3% vs FST V: 15.8%) -- severity DANGEROUS. Cancer sensitivity gap = 39.58% -- severity DANGEROUS. Overall accuracy only 25.9%. Sample sizes critically small (FST VI: n=18, FST I: n=27). Remediation plan needed before any claims of equity.

---

## Context

Our Fitzpatrick equity analysis (`scripts/fitzpatrick-equity-report.json`) reveals a severe data imbalance. For melanoma, the dark-skin (FST V-VI) to light-skin (FST I-III) sample ratio is **0.142** -- that is, 47 dark-skin melanoma samples vs 331 light-skin melanoma samples. The training data (HAM10000) is roughly 95% Fitzpatrick I-III.

We claim our model works. We have **zero measured evidence** of performance stratified by skin type.

DermaSensor, the only FDA-cleared skin cancer AI device, reported a 4% sensitivity gap between FST I-III and FST IV-VI in their pivotal trial (96% vs 92%). Our gap could be substantially worse because:

1. Our training data is even more skewed than theirs.
2. We have never measured per-skin-type performance.
3. The model has no explicit skin-tone awareness -- it could be relying on lightness cues that break on dark skin.

**This is a patient safety issue.** Melanoma on dark skin is already diagnosed at later stages and carries worse outcomes. If our AI performs worse on dark skin, we are not just failing to help -- we are actively amplifying an existing health disparity. Black patients have a 5-year melanoma survival rate of 71% vs 93% for white patients (ACS, 2024), partly because of later detection. An AI tool that underperforms on dark skin makes this worse.

### Data We Have (from fitzpatrick-equity-report.json)

| Class | Dark (FST V+VI) | Light (FST I+II+III) | Ratio | Concern |
|-------|-----------------|---------------------|-------|---------|
| Melanoma | 47 | 331 | 0.142 | HIGH |
| BCC | 33 | 347 | 0.095 | CRITICAL |
| Nevi | 30 | 235 | 0.128 | HIGH |
| Dermatofibroma | 1 | 46 | 0.022 | CRITICAL -- nearly zero dark-skin samples |
| Benign Keratosis | 14 | 90 | 0.156 | HIGH |
| Actinic Keratosis | 91 | 462 | 0.197 | MODERATE |
| Vascular | 71 | 318 | 0.223 | MODERATE |

Critical gaps where fewer than 10 samples exist for a skin-type/class pair: BCC on FST VI (7 samples), BKL on FST VI (4), DF on FST IV (7), DF on FST V (1), DF on FST VI (0), NV on FST VI (8).

---

## Decision

Run a stratified Fitzpatrick validation on the v2 model. Measure performance by skin type. Report it honestly. If we find a gap, we fix it or we warn about it. We do NOT hide the results.

---

## Method

### Step 1: Acquire Fitzpatrick17k Test Images

Download the 3,413 Fitzpatrick17k images that map to our 7 classes. The manifest is at `scripts/dataset-manifests/fitzpatrick17k.json`. These images have Fitzpatrick labels (1-6) per image and diagnostic labels that map to our class taxonomy.

```bash
python scripts/download-fitzpatrick17k.py \
  --manifest scripts/dataset-manifests/fitzpatrick17k.json \
  --output data/fitzpatrick17k/ \
  --classes mel,bcc,akiec,bkl,nv,df,vasc
```

### Step 2: Run V2 Model Inference

Run the v2 model on every downloaded image. Record raw softmax probabilities per class.

```bash
python scripts/fitzpatrick-stratified-eval.py \
  --model models/mela-v2.onnx \
  --images data/fitzpatrick17k/ \
  --manifest scripts/dataset-manifests/fitzpatrick17k.json \
  --output scripts/fitzpatrick-stratified-results.json
```

### Step 3: Compute Per-Skin-Type Metrics

For each Fitzpatrick type (I through VI) and for each diagnostic class, compute:

| Metric | Why It Matters |
|--------|---------------|
| Sensitivity (recall) | Does the model catch the disease on this skin type? |
| Specificity | Does it avoid false alarms on this skin type? |
| AUROC | Overall discriminative ability per skin type |
| PPV / NPV | Clinical usefulness per skin type |
| Confusion matrix | Where exactly does it fail? |

Group results into light (FST I-III) vs dark (FST IV-VI) for gap analysis.

### Step 4: Apply Gap Thresholds

| Gap (any metric, any skin-type pair) | Severity | Action |
|--------------------------------------|----------|--------|
| < 5% | ACCEPTABLE | Document and monitor |
| 5-10% | WARNING | Flag in UI, add to training roadmap |
| > 10% | CRITICAL | Model is NOT safe for deployment on dark skin without mitigation |
| > 15% | BLOCKING | Halt deployment claims for affected populations |

### Step 5: Report and Remediate

Publish `scripts/fitzpatrick-stratified-results.json` with full per-type breakdowns. If gaps exceed thresholds:

1. **Augmentation**: Apply skin-tone augmentation (color jitter calibrated to FST IV-VI appearance) to synthetically increase dark-skin representation.
2. **Targeted training**: Fine-tune on Fitzpatrick17k with oversampling of underrepresented skin-type/class pairs.
3. **Confidence gating**: Lower confidence for skin types where the model has known weak performance. Show the user "limited evidence for your skin type" instead of a false-confident answer.
4. **Prominent warning**: If we cannot close the gap, add a visible warning in the UI: "This tool has been primarily validated on lighter skin tones. Results for darker skin tones may be less reliable."

---

## DDD: Equity Audit Bounded Context

```
Aggregates:
  EquityAuditRun { runId, modelVersion, datasetManifest, timestamp, status }
  StratifiedResult { fitzpatrickType(1-6), diagnosticClass, sampleCount,
                     sensitivity, specificity, auroc, ppv, npv, confusionMatrix }
  GapAnalysis { lightPerformance(FST I-III), darkPerformance(FST IV-VI),
                sensitivityGap, specificityGap, aurocGap,
                severity: acceptable | warning | critical | blocking }

Domain Events:
  EquityAuditCompleted -> GapAnalysis
  CriticalGapDetected  -> RemediationPlan
  BlockingGapDetected  -> DeploymentHalt

Integration: ClassifierService, DatasetService, ReportingService, UIWarningService
```

---

## Implementation Plan

| Phase | Task | Effort | Depends On |
|-------|------|--------|------------|
| 1 | Download Fitzpatrick17k mapped images (3,413) | 1 hour | Manifest exists |
| 2 | Write stratified evaluation script | 2 hours | Phase 1 |
| 3 | Run v2 model inference on all images | 1 hour | Phase 2, v2 model |
| 4 | Compute per-skin-type metrics | 1 hour | Phase 3 |
| 5 | Gap analysis and severity classification | 30 min | Phase 4 |
| 6 | Write remediation plan (if gaps found) | 2 hours | Phase 5 |
| 7 | Implement remediations | 4-8 hours | Phase 6 |
| 8 | Re-run audit to verify improvement | 2 hours | Phase 7 |

Total estimated effort: 1-2 days (phases 1-5), plus 1-2 days for remediation if needed.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Audit reveals >10% gap on dark skin | HIGH | Targeted training, augmentation, UI warning |
| Fitzpatrick17k is clinical photos (not dermoscopy) | MEDIUM | Report domain mismatch as limitation |
| Too few FST VI samples (77 total) for significance | HIGH | Report CIs; flag n < 30 as "insufficient data" |
| Model is fundamentally biased | LOW-MEDIUM | Skin-tone normalization before classification |
| Results are bad and we bury them | ZERO | This ADR is the commitment: we report and we fix |

We do NOT claim "works for all skin types" without measured evidence. We do NOT set thresholds that conveniently make our model pass. We do NOT exclude FST VI because n is small.

---

## Success Criteria

| Criterion | Threshold | How Measured |
|-----------|----------|-------------|
| Melanoma sensitivity gap (FST I-III vs IV-VI) | < 5% | Stratified evaluation |
| AUROC gap (FST I-III vs IV-VI) | < 0.02 | Stratified evaluation |
| Per-FST-type confusion matrix published | Yes/No | fitzpatrick-stratified-results.json |
| FST V-VI representation in test set | >= 8% of total (287+ of 3,413) | Dataset manifest -- currently 287 (8.4%), meets minimum |
| Remediation plan written if gap > 5% | Yes/No | Updated ADR |
| UI warning added if gap > 10% and unfixed | Yes/No | Code review |

---

## References

1. Groh M, et al. (2021). "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset." CVPR Workshop.
2. Daneshjou R, et al. (2022). "Disparities in Dermatology AI Performance Across Skin Tones." Science Advances.
3. Tkaczyk ER, et al. (2024). "DermaSensor DERM-SUCCESS." FDA DEN230008.
4. American Cancer Society (2024). Melanoma survival statistics by race.
5. Fitzpatrick equity report: `scripts/fitzpatrick-equity-report.json`
6. Fitzpatrick17k manifest: `scripts/dataset-manifests/fitzpatrick17k.json`

## Author

Stuart Kerr + Claude Flow (RuVector/RuFlo)
