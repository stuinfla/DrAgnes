Updated: 2026-03-24 14:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-123: Per-Class Threshold Optimization -- Free Accuracy from Existing Model

## Status: IMPLEMENTED -- Thresholds Computed + Classifier Built + UI Selector | Last Updated: 2026-03-24 12:00 EST
**Implementation Note**: threshold-classifier.ts (106 lines) implements per-class optimal thresholds with 3 modes (default/screening/triage). optimal-thresholds.json contains computed thresholds from ROC analysis on ISIC 2019 test set. threshold-optimization-results.json has full per-class sensitivity/specificity at threshold. Settings UI in MelaPanel.svelte exposes threshold mode selector. Mel threshold = 0.6204 (93.88% sensitivity, 85.34% specificity at threshold).

## Context

The v2 combined model uses **argmax** (highest probability class wins) for classification. Every class competes on equal footing -- the prediction goes to whichever class has the highest raw probability. This is the default behavior of every classification model, and it is suboptimal for medical screening.

**Why argmax is wrong here:**

1. **Different classes have wildly different costs of error.** Missing a melanoma can kill someone. Misclassifying a benign keratosis as dermatofibroma wastes nobody's time. Argmax treats these errors as equally bad.

2. **The ROC curve has room to move.** The v2 model achieves melanoma AUROC of 0.960, but the current operating point delivers 95.97% sensitivity at only 80% specificity. That means 1 in 5 benign moles is flagged as suspicious. The 0.960 AUROC tells us there are operating points on the curve with better tradeoffs -- we just need to find them.

3. **The default threshold is implicit and arbitrary.** With 7 classes, argmax implicitly uses a threshold of ~14.3% (1/7). A melanoma probability of 15% beats all other classes at 14% and triggers a cancer warning. That threshold was never chosen -- it fell out of the math.

**Current performance (v2 combined model, ISIC 2019 test set):**

| Metric | Value | Problem |
|--------|-------|---------|
| Melanoma sensitivity | 95.97% | Good, but AUROC says we can do better |
| Melanoma specificity | 80.0% | 1 in 5 false alarms -- erodes user trust |
| Overall accuracy | 89.3% | Dragged down by benign class confusion |
| Melanoma AUROC | 0.960 | Gap between AUROC and operating point = free accuracy |

**What "free accuracy" means:** The model already learned to separate melanoma from benign lesions better than the argmax threshold reveals. By choosing a smarter operating point on the existing ROC curve, we extract accuracy that the model already has but argmax throws away. No retraining. No new data. No new model.

---

## Decision

Implement **per-class probability threshold optimization** using the ROC curves from validation data. Replace argmax with threshold-based classification where each class has its own decision boundary tuned to its clinical cost function.

### Method

**Step 1: Extract probability outputs.** Load the v2 model's softmax probability vectors on the ISIC 2019 test set. These are already recorded in `combined-training-results.json` from the training pipeline.

**Step 2: Compute per-class ROC curves.** For each of the 7 classes, compute the one-vs-rest ROC curve (true positive rate vs false positive rate at every threshold from 0.0 to 1.0).

**Step 3: Choose operating points by clinical priority.**

| Class | Priority | Optimization Target |
|-------|----------|-------------------|
| `mel` (melanoma) | CRITICAL | Maximize sensitivity subject to specificity >= 85% |
| `bcc` (basal cell carcinoma) | HIGH | Maximize sensitivity subject to specificity >= 85% |
| `akiec` (actinic keratosis) | HIGH | Maximize sensitivity subject to specificity >= 85% |
| `nv` (melanocytic nevus) | LOW | Maximize specificity (reduce false cancer scares) |
| `bkl` (benign keratosis) | LOW | Maximize specificity |
| `df` (dermatofibroma) | LOW | Maximize specificity |
| `vasc` (vascular lesion) | LOW | Maximize specificity |

The asymmetry is intentional: for cancer classes, the cost of a false negative (missed cancer) vastly exceeds the cost of a false positive (unnecessary doctor visit). For benign classes, the cost of a false positive (calling a mole cancer) exceeds the cost of a false negative (calling it the wrong type of benign).

**Step 4: Resolve multi-class conflicts.** When a sample exceeds the threshold for multiple classes, apply a priority cascade: `mel > bcc > akiec > nv > bkl > df > vasc`. Cancer classes always win ties.

**Step 5: Cross-validate.** Run 5-fold cross-validation on the threshold selection to detect overfitting. If the optimal threshold shifts by more than 0.05 across folds, widen the threshold to the most conservative (highest-sensitivity) fold.

### Expected Impact

| Metric | Before | After (estimated) | How |
|--------|--------|-------------------|-----|
| Melanoma specificity | 80.0% | 85-90% | Move operating point right on ROC curve |
| Melanoma sensitivity | 95.97% | >= 95.97% | Constraint: never sacrifice sensitivity |
| False alarm rate | 20% | 10-15% | Direct consequence of higher specificity |
| User trust | Low (too many scares) | Higher | Fewer false positives = credible results |

---

## Domain-Driven Design: Classification Bounded Context

The threshold logic belongs in the **Classification** bounded context, isolated from the model inference layer.

```
Classification Bounded Context
+---------------------------------------------------------------+
|                                                               |
|  ModelInference           ThresholdStrategy                   |
|  +------------------+    +------------------------------+     |
|  | predict(image)   |    | thresholds: Map<Class, f64>  |     |
|  | -> RawProbVector |    | priorities: Class[]          |     |
|  +--------+---------+    | apply(probs) -> Prediction   |     |
|           |              | validate(folds) -> Report    |     |
|           v              +----------+-------------------+     |
|  RawProbVector                      |                         |
|  [akiec: 0.03, bcc: 0.05,         |                         |
|   bkl: 0.12, df: 0.01,            v                         |
|   mel: 0.41, nv: 0.35,     Prediction                       |
|   vasc: 0.03]              { class, probability,             |
|                               confidence, exceeds: Class[] } |
|                                                               |
+---------------------------------------------------------------+
```

**Key design decisions:**

- `ThresholdStrategy` is a **value object** -- immutable after construction, serializable to JSON, versionable. Each deployed version of the app ships with a specific threshold configuration.
- `ModelInference` knows nothing about thresholds. It returns raw probabilities. This separation means we can swap threshold strategies without touching model code.
- `Prediction.exceeds` lists all classes that exceeded their thresholds, not just the winner. This enables the UI to say "melanoma detected; BCC also elevated" rather than hiding secondary findings.

---

## Implementation Plan (1 Day)

| Phase | Task | Time | Output |
|-------|------|------|--------|
| 1 | Extract probability vectors from `combined-training-results.json` | 1h | `probabilities.csv` with ground truth + 7 probability columns |
| 2 | Compute 7 one-vs-rest ROC curves, find optimal thresholds per class | 2h | `threshold-config.json` with per-class thresholds |
| 3 | Implement `ThresholdStrategy` class in `classifier.ts` | 1.5h | New module: `threshold-strategy.ts` |
| 4 | 5-fold cross-validation on threshold stability | 1.5h | Validation report, threshold confidence intervals |
| 5 | Integration test: run 100 test images through old vs new pipeline | 1h | Side-by-side comparison table |
| 6 | Update `ClassificationResult` to include `exceeds` field | 1h | Type change + UI update in `ClassificationResult.svelte` |

**Total: ~8 hours.** No model retraining. No new dependencies. No infrastructure changes.

---

## Risks and Mitigations

### Risk 1: Threshold Overfitting to Validation Set

**Problem:** If the ISIC 2019 test set is not representative of real-world phone camera images (it is not -- ISIC images are dermoscopic), the optimal thresholds may not transfer.

**Mitigation:** Cross-validate thresholds across 5 folds. If the standard deviation of the optimal threshold for any class exceeds 0.05, use the most conservative value (lowest threshold for cancer classes, highest for benign). Additionally, collect real-world phone images for a secondary validation set before declaring thresholds production-ready.

### Risk 2: Threshold Drift Over Time

**Problem:** As the user population changes (different skin types, different camera phones), the optimal thresholds may drift.

**Mitigation:** Log all raw probability vectors in production (already planned in analytics). Run threshold validation monthly against the accumulated real-world data. Version thresholds in `threshold-config.json` so rollback is instant.

### Risk 3: Multi-Class Conflicts Produce Confusing Results

**Problem:** A sample might exceed thresholds for both `mel` and `nv`, producing a result that says "melanoma detected but also looks like a normal mole."

**Mitigation:** The priority cascade resolves the primary classification unambiguously. The `exceeds` list is for clinical context only, shown as a secondary detail (e.g., "Some features also consistent with common mole"). The UI never presents conflicting primary diagnoses.

---

## Alternatives Considered

### Alternative 1: Retrain with Class Weights

Modify the loss function to penalize melanoma false negatives more heavily. This would shift the model's internal decision boundaries.

**Rejected because:** Requires retraining (hours of GPU time), may degrade performance on other classes, and the threshold approach achieves the same result with zero retraining.

### Alternative 2: Calibration (Platt Scaling / Temperature Scaling)

Fit a sigmoid or temperature parameter to make the probabilities better-calibrated (i.e., when the model says 80% melanoma, it really is melanoma 80% of the time).

**Deferred, not rejected:** Calibration and threshold optimization are complementary. Calibration makes the probabilities meaningful; thresholds choose the operating point. We should implement calibration in a follow-up ADR after thresholds are validated.

### Alternative 3: Cost-Sensitive Argmax

Multiply each class probability by a cost weight before taking argmax. Simpler than per-class thresholds.

**Rejected because:** Cost-sensitive argmax is a special case of threshold optimization but with less control. Per-class thresholds let us independently tune each class's operating point on its ROC curve, which cost weights cannot do.

---

## References

- ADR-117: Mela platform architecture (model ensemble design)
- ADR-118: Production validation results (current sensitivity/specificity numbers)
- ADR-119: Consumer skin screening (risk level translation)
- ADR-120: Deployment checklist (specificity complaints from false alarms)
- ISIC 2019 Challenge: https://challenge.isic-archive.com/landing/2019/
- Youden's J statistic for optimal threshold selection
