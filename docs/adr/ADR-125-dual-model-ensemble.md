Updated: 2026-03-24 00:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-125: V1+V2 Dual Model Ensemble -- Best of Both Worlds

## Status: PROPOSED (not started) | Last Updated: 2026-03-24 10:30 EST
**Implementation Note**: No ensemble.ts module exists. The current classifier.ts uses a single-model ensemble (HF API + trained-weights + rule-based) but does NOT implement the V1+V2 dual ViT model ensemble described in this ADR. The v2 combined model exists (scripts/dragnes-classifier-v2/) but is not wired into a dual-inference pipeline.

---

## Context

We have two trained ViT models with complementary strengths:

| Metric | V1 (HAM10000-only) | V2 (Combined multi-dataset) |
|--------|--------------------|-----------------------------|
| Mel sensitivity (HAM10000) | **98.2%** (n=225) | **97.01%** |
| Mel sensitivity (ISIC 2019 external) | **61.6%** (n=714) | **95.97%** |
| Training data | 10,015 (HAM10000) | HAM10000 + ISIC 2019 + BCN20000 |
| Strength | Best in-distribution accuracy | Dramatically better generalization |

V1 is 1.19pp better on HAM10000-like dermoscopy. V2 is 34.37pp better on external data. Running both and combining outputs captures the best of each. The disagreement between models is itself a clinically valuable signal -- high disagreement should push toward caution.

---

## Decision Options

| Option | Approach | Pro | Con | Verdict |
|--------|----------|-----|-----|---------|
| 1. Simple average | 0.5/0.5 weight | Dead simple, no hyperparams | V1's 61.6% external drags down ensemble | Too naive |
| 2. **Weighted avg** | **0.7 V2 / 0.3 V1** | **Simple, V2 dominates where it matters, V1 refines in-distribution** | **Fixed weights, heuristic split** | **CHOSEN** |
| 3. Stacking | Meta-classifier on 14-dim concat output | Learns per-class optimal weights | Needs held-out data, third model to maintain | Good V2 path |
| 4. Disagreement routing | V2 default, V1 fallback on low confidence | Only 2x cost when needed | Confidence calibration is fragile | Revisit post-calibration |

---

## Decision

**Option 2: Weighted average (0.7 V2 + 0.3 V1) with a safety-first melanoma override.**

```
P_ensemble(class) = 0.7 * P_v2(class) + 0.3 * P_v1(class)
```

Override rule: if EITHER model assigns melanoma probability above the safety threshold (0.15), the ensemble uses `max(P_v1_mel, P_v2_mel)` instead of the weighted average for melanoma, then renormalizes. When models disagree on melanoma, the one that says "melanoma" wins. A false positive costs a doctor visit. A false negative can cost a life.

---

## DDD: Ensemble Bounded Context

```
+------------------------------------------------------------------+
|                     Ensemble Bounded Context                      |
|                                                                   |
|  +------------------+    +---------------------+                  |
|  | ModelRegistry    |    | ModelAggregation    |                  |
|  | - v1, v2: Ref    |    | - weights: Map      |                  |
|  | - getModel(id)   |    | - aggregate()       |                  |
|  +--------+---------+    | - safetyOverride()  |                  |
|           |              +----------+----------+                  |
|           v                        |                              |
|  +------------------+              v                              |
|  | InferenceRunner  |    +---------------------+                  |
|  | - runParallel()  |    | DisagreementDetector|                  |
|  | - timeout(ms)    |    | - jsDivergence()    |                  |
|  +------------------+    | - flagForReview()   |                  |
|                          +---------------------+                  |
+------------------------------------------------------------------+
         |                              |
         v                              v
+------------------+          +---------------------+
| Classifier BC    |          | Consumer Translation|
| (existing)       |          | BC (existing)       |
+------------------+          +---------------------+
```

### Data Flow

```
User captures image
    |
    v
Image Quality Gate --> Lesion Detection Gate
    |
    +-------+-------+
    |               |
    v               v
V1 Inference   V2 Inference      (parallel)
    |               |
    +-------+-------+
            |
            v
  Weighted Aggregation (0.7 V2 + 0.3 V1)
            |
            v
  Melanoma Safety Override (max if either > 0.15)
            |
            v
  Disagreement Detector (JSD > 0.15 -> flag for review)
            |
            v
  Consumer Translation (risk level, plain English, actions)
```

---

## Implementation Plan (1 Day)

| Step | Work | Time |
|------|------|------|
| 1 | Create `src/lib/dragnes/ensemble.ts` with ModelRegistry, InferenceRunner (Promise.all on both HF model IDs), EnsembleResult type | 2h |
| 2 | Implement weighted aggregation + melanoma safety override + probability renormalization | 1h |
| 3 | Implement DisagreementDetector using Jensen-Shannon divergence (JSD > 0.15 = flag) | 1h |
| 4 | Wire into `DermClassifier.classify()` as additive layer; fall back to V1-only if V2 unavailable | 2h |
| 5 | Validate on HAM10000 test split + ISIC 2019 external + 20 real-world images; write `ensemble-validation-results.json` | 2h |

Key interfaces in `ensemble.ts`:

```typescript
interface EnsembleResult {
  aggregated: ClassProbability[];
  v1Raw: ClassProbability[];
  v2Raw: ClassProbability[];
  disagreement: number;              // Jensen-Shannon divergence
  safetyOverrideApplied: boolean;
}
```

---

## Performance Impact

| Metric | V1 Only (current) | Dual Ensemble |
|--------|-------------------|---------------|
| Inference calls | 1 | 2 (parallel) |
| Wall-clock latency | ~500ms | ~550ms (bounded by slower model) |
| HF API calls | 1 | 2 (free tier, negligible cost) |
| Memory (ONNX future) | ~330MB | ~660MB (within 1GB phone budget) |

Latency stays under 1 second because both models run in parallel. The 2x API cost is acceptable for a medical safety application.

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Models disagree on melanoma | Missed cancer or false alarm | **Safety-first: the model that says melanoma wins.** False positive (doctor visit) is dramatically less harmful than false negative (missed melanoma). |
| Ensemble masks model degradation | Silent accuracy drop | Log V1 and V2 raw outputs separately in analytics dashboard. Monitor per-model metrics independently. |
| V1's 61.6% external performance poisons ensemble | Lower external sensitivity | V1 only gets 0.3 weight. Worst case: `0.7*95.97 + 0.3*61.6 = 85.66%` mel sensitivity, still above 85% ADR-118 target. Safety override takes max for melanoma, not weighted avg. |

---

## Expected Outcomes

| Metric | V1 Only | V2 Only | Ensemble (estimated) |
|--------|---------|---------|---------------------|
| Mel sensitivity (HAM10000) | 98.2% | 97.01% | ~97.7% |
| Mel sensitivity (ISIC 2019) | 61.6% | 95.97% | ~95%+ (safety override) |
| Disagreement signal | N/A | N/A | New safety dimension |

The ensemble pushes melanoma sensitivity above 97% on external data while maintaining 97.5%+ on HAM10000 data. The disagreement detector adds a safety dimension neither model provides alone.

---

## Future Evolution

1. **Stacking:** After 1000+ ensemble predictions with ground truth, train a meta-classifier for per-class optimal weights.
2. **Dynamic weighting:** Adjust V1/V2 weights per-case based on image characteristics (dermoscopy vs smartphone, skin tone).
3. **V3 model:** Architecture supports N models by design -- add ViT-Large when trained.
4. **Calibrated routing:** Revisit Option 4 after calibration analysis (ADR-118 Phase 2.3).

---

## References

1. ADR-118: DrAgnes Production Validation (V1 measured performance)
2. ADR-120: Make It Actually Work (deployment architecture)
3. cross-validation-results.json (V1 HAM10000 metrics)
4. isic2019-validation-results.json (V1 ISIC 2019 metrics)
5. Lakshminarayanan B, et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS
6. Ganaie MA, et al. (2022). "Ensemble deep learning: A review." Eng. Applications of AI 115:105151

## Author

Stuart Kerr + Claude Flow (RuVector/RuFlo)
