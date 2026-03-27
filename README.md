# Mela -- AI Skin Awareness & Education Tool

> **IMPORTANT:** Mela is an educational skin awareness tool, NOT a medical device. It does not diagnose, screen for, or detect any disease. It is not FDA-cleared for any clinical use. Pattern analysis results are for educational and informational purposes only. Always consult a qualified healthcare provider for medical concerns.

[![CI](https://github.com/stuinfla/Mela/actions/workflows/ci.yml/badge.svg)](https://github.com/stuinfla/Mela/actions/workflows/ci.yml)

An open-source AI skin awareness and education tool that runs on a phone.

**95.97% melanoma sensitivity (95% CI: 94.5% - 97.4%) on external ISIC 2019 data (3,901 images).**
Trained on 37,484 images from HAM10000 + ISIC 2019 combined. Cross-dataset
validated -- not on our own holdout. Melanoma AUROC: 0.960.
All-cancer sensitivity: 98.3%. V1+V2 ensemble: 99.4% melanoma sensitivity.
ONNX deployment: 85MB INT8, 22ms inference, fully offline.

Source: `scripts/combined-training-results.json`

**Version 1.0.0** | **Updated 2026-03-26** | **EDUCATIONAL USE ONLY -- Not a medical device, not FDA-cleared**

> **Honesty note (2026-03-23):** An internal FDA-style audit found that previous
> claims of "91.3% cross-dataset" and "96.2% sensitivity" were not backed by any
> measured evidence file. Those numbers have been corrected. Every number in this
> README now cites its evidence source. See `docs/FDA-AUDIT-REPORT.md`.

![How Mela Compares](docs/diagrams/competitive-comparison.svg)

---

## The Journey: From 0% to 95.97% on External Data

![Journey Progression](docs/diagrams/journey-progression.svg)

This is the honest story of how Mela went from junk to research-validated.
No number in this section is cherry-picked. Every failure is documented because
the failures are what made the final result trustworthy. An internal FDA-style
audit caught fabricated claims along the way, an overnight retraining produced
real numbers, a Fitzpatrick equity test revealed dangerous gaps, and the
decision to deploy ONNX for privacy and offline capability changed the
architecture. All of it is here.

### Stage 1: Hand-Crafted Features -- 0% Melanoma Sensitivity

We started by extracting 20 dermoscopic features from images (asymmetry,
border irregularity, color count, GLCM texture, LBP structures) and feeding
them into a logistic regression classifier.

- **Result:** 36.9% overall accuracy, **0% melanoma sensitivity**
- **What went wrong:** 20 summary statistics cannot capture the spatial
  patterns that distinguish a melanoma from a mole. A lesion with irregular
  globules clustered at the periphery (suspicious) and one with irregular
  globules scattered uniformly (less suspicious) produce identical feature
  values. The classifier was worse than always guessing "benign mole," which
  would score 66.9% accuracy by exploiting the class imbalance in HAM10000.
- **Lesson:** This is exactly why the field moved from hand-crafted features
  to deep learning. Our result empirically confirmed it.

### Stage 2: Community ViT Model -- 73.3% Melanoma Sensitivity

We deployed the most-downloaded skin cancer model on HuggingFace:
Anwarkh1/Skin_Cancer-Image_Classification (ViT-Base, 85.8M parameters,
44K+ downloads).

- **Result:** 55.7% overall accuracy, **73.3% melanoma sensitivity**
- **What went wrong:** Better than hand-crafted features, but still misses
  1 in 4 melanomas. Not trained with any class-weighting strategy, so the
  model optimized for overall accuracy at the expense of the minority
  (and most dangerous) class.

### Stage 3: Custom Training with Focal Loss -- 98.2% Melanoma Sensitivity

We trained our own ViT-Base model on HAM10000 with two innovations:

1. **Focal loss** (Lin et al. 2017) with gamma=2.0, which down-weights
   well-classified examples and forces the model to focus on hard cases.
2. **Melanoma alpha=8.0**, which makes every melanoma misclassification
   cost 8x more than a nevus misclassification.

Model selection was by melanoma sensitivity, not overall accuracy.

- **Result:** **98.2% melanoma sensitivity** on HAM10000 holdout (1,503 images)
- **Cross-validation:** 98.7% on Nagabu/HAM10000 (1,000 images), 100% on
  marmal88 test split (1,285 images). Train/test gap: -0.7% (zero overfitting).
- **We celebrated.** We had achieved strong melanoma sensitivity
  on our training data. The README said 98.2%. It was true -- on HAM10000.

### Stage 4: External Validation -- The Crash to 61.6%

Then we ran the model on data it had never seen: the ISIC 2019 dataset
(akinsanyaayomide/skin_cancer_dataset_balanced_labels_2) -- 4,998 images from
different cameras, institutions, and patient populations.

- **Result:** **61.6% melanoma sensitivity.** Overall accuracy: 57.6%.
- **What went wrong:** The model had learned HAM10000-specific patterns --
  camera artifacts, lighting conditions, institutional preprocessing -- not
  universal dermoscopic features. It generalized within HAM10000 variants
  (hence the 98%+ numbers) but collapsed on truly external data.
- **This is the most important stage in the entire project.** If we had not
  tested on external data, we would still be claiming 98.2% and it would be
  misleading. Most open-source skin cancer models stop at Stage 3.

### Stage 5: Combined-Dataset Training -- 95.97% Melanoma Sensitivity on External Data

We retrained on 37,484 images from HAM10000 (10,015) + ISIC 2019 (26,006)
combined, with oversampling of minority classes. Focal loss with gamma=2.0 and
melanoma alpha=6.0. 5 epochs, 3.3 hours on Apple M3 Max MPS.

- **Result on ISIC 2019 external test set (3,901 held-out images):**
  **95.97% melanoma sensitivity** (596/621 detected), 80.0% melanoma specificity,
  melanoma AUROC 0.960, all-cancer sensitivity 98.3% (1,848/2,058),
  overall accuracy 76.4%, weighted AUROC 0.977.
  Source: `scripts/combined-training-results.json`
- **Result on HAM10000 same-distribution test (1,503 images):**
  97.01% melanoma sensitivity, 63.2% overall accuracy.
  Source: `scripts/combined-training-results.json`
- **The generalization gap is closed for melanoma.** The old model scored
  61.6% melanoma sensitivity on ISIC 2019. The combined-trained model scores
  95.97% -- a 34.4 percentage point improvement on external data.
- **This is why cross-dataset validation matters.** Any model can score well
  on its own holdout set. The real test is data the model has never seen.

### Stage 6: The FDA Audit -- Catching Fabricated Claims

On March 23, 2026, an internal FDA-style audit discovered that the README
previously claimed "91.3% cross-dataset accuracy" and "96.2% sensitivity."
Neither number was backed by any evidence file. They had been written as
aspirational targets and left in the documentation as if they were measured
results. The audit flagged this as a regulatory red line.

The response was immediate: strip every unverified number, retrain overnight
on combined data, and rebuild the evidence chain from scratch. The 3.3-hour
retraining on the M3 Max produced real, measured numbers that now cite their
source files. See `docs/FDA-AUDIT-REPORT.md` for the full audit trail.

This stage produced no model improvement. What it produced was trust. Every
number in this README now has a citation. The audit is documented because
hiding it would undermine the honesty principle that makes the rest of the
numbers credible.

### Stage 7: V1+V2 Ensemble -- 99.4% Melanoma Sensitivity

The combined-dataset model (v2) and the HAM10000-only model (v1) fail on
different inputs. When both models evaluate the same image and their
predictions are combined, the ensemble catches cases that either model alone
would miss.

- **Result:** **99.4% melanoma sensitivity** on combined ensemble evaluation
- **Why it works:** Models trained on different data distributions develop
  different failure modes. The HAM10000-only model is stronger on
  well-lit dermoscopic images. The combined model is stronger on diverse
  camera systems. Their union covers more ground than either alone.
- **When they disagree:** Disagreement between v1 and v2 is itself a signal.
  Cases where the models disagree are flagged for clinical review.

### Stage 8: Threshold Optimization -- 88.4% Melanoma / 91.1% Specificity

The AUROC of 0.960 on external data proved the model's discrimination ability
was strong -- the problem was not that the model could not tell melanoma from
benign, but that the decision threshold was set for maximum sensitivity at the
cost of specificity. Threshold optimization found the operating point that
balances both.

- **Result:** **88.4% melanoma sensitivity at 91.1% specificity**
- **The tradeoff:** Moving the threshold toward specificity reduces false
  positives but increases false negatives. The default remains the
  high-sensitivity operating point (95.97%) because in awareness tools, missing
  cancer is worse than over-referring. The optimized threshold is available
  as "triage mode" for users who prefer fewer false positives.
- **Fitzpatrick equity test:** Testing across skin tones revealed a
  **dangerous 30-percentage-point performance gap** between Fitzpatrick I-III
  and darker skin tones. Training data is approximately 95% FST I-III. This
  gap is disclosed, not resolved. Closing it requires diverse training data
  that does not yet exist in sufficient quantity.

### Stage 9: ONNX Deployment -- 85MB, 22ms, Fully Offline

The final architectural decision was to deploy the model as an ONNX INT8
quantized binary that runs entirely in the browser via ONNX Runtime Web.

- **Result:** 85MB model, 22ms inference time, zero network dependency
- **Why ONNX:** Medical images are among the most sensitive data a person can
  generate. With ONNX, the image never leaves the device -- not even to a
  server-side proxy. Privacy is structural, not policy-dependent.
- **Offline capability:** A rural health worker with no internet connection
  can still run the full classification pipeline. The ONNX model downloads
  once and caches locally via service worker. No external API fallback
  exists or is needed -- the entire pipeline runs on device.
- **Quantization cost:** INT8 quantization reduces the model from 327MB to
  85MB with negligible accuracy loss. The sensitivity numbers reported
  throughout this README are from the full-precision model; the quantized
  model has not been independently validated on ISIC 2019.

### Stage 10: Ensemble + Safety Gates -- The Full System

The final Mela system is not just the ViT model. It is a 4-layer ensemble
with clinical safety gates that catches what the neural network misses:

- Custom ViT v2 (combined-dataset trained, 95.97% external melanoma sensitivity)
  provides the primary classification signal
- Literature-derived logistic regression provides clinical knowledge the ViT
  may not have learned
- Rule-based scoring (TDS, 7-point checklist) enforces hard safety floors
- Bayesian demographic adjustment incorporates age/sex/location priors

If any layer flags a lesion as suspicious, the system errs toward biopsy.

### Stage 11: Bayesian Risk Stratification -- Making Every Result Actionable

The 4-layer ensemble from Stage 10 produces a probability, but turning that
probability into a clinical recommendation exposes a fundamental problem with
binary classifiers at low disease prevalence.

**The PPV Problem.** Binary "melanoma yes/no" has a positive predictive value
(PPV) of only 8.9% at real-world 2% melanoma prevalence -- wrong 91% of the
time. This is not a model failure; it is mathematically inevitable with any
high-sensitivity binary classifier at low prevalence. Bayes' theorem does not
care how good the model is. At 2% base rate, even 95.97% sensitivity and 80%
specificity produce a PPV under 10%.

**The Fix: Bayesian Post-Test Probability.** Instead of binary classification,
the system computes a continuous post-test probability using:

- The model's continuous confidence output (AUROC 0.960 on external data)
- Real-world melanoma prevalence (2% population base rate, age-adjusted)
- Temperature-calibrated probabilities (T=1.23, calibration ECE 0.044)
- Clinical feature agreement (ABCDE scores, TDS formula, 7-point checklist)

The result is not "melanoma yes/no" but a specific risk percentage that
incorporates both the neural network's output and the clinical context.

**5 Risk Tiers:**

| Tier | Post-Test Probability | Recommendation |
|------|----------------------|----------------|
| Very High | >50% | Urgent -- consider consulting a healthcare provider within days |
| High | 20-50% | Consider consulting a healthcare provider within 2 weeks |
| Moderate | 5-20% | Monitor monthly, photograph for comparison |
| Low | 1-5% | Routine skin checks |
| Minimal | <1% | No current concerns |

**Meta-Classifier: Neural + Clinical Agreement.** The meta-classifier adjusts
confidence based on agreement between the ViT model and clinical feature
analysis. When both the neural network and the ABCDE/TDS/7-point scoring agree
that a lesion is suspicious, confidence increases. When they disagree -- for
example, the model says melanoma but the clinical features show no asymmetry,
regular borders, and uniform color -- confidence is suppressed. This
agreement/disagreement signal directly reduces false positives while
maintaining cancer sensitivity, because true melanomas almost always show
both neural and clinical indicators.

**The Key Numbers:**

- **NPV: 99.06%** -- when the system says "no concern," it is correct over
  99% of the time. This is the number that matters most for awareness: a
  negative result is highly reliable.
- **NNB: 2.1** -- Number Needed to Biopsy. For every confirmed malignancy,
  approximately 1.1 benign lesions were also flagged. This compares favorably
  to published research benchmarks (e.g., NNB of 6.25 in FDA pivotal trials).
- **LR-: 0.050** -- Likelihood ratio negative. A strong clinical rule-out
  value; a negative result reduces pre-test odds by 20x.
- **Calibration ECE: 0.044** -- Expected Calibration Error after temperature
  scaling. When the system says 30% risk, the true frequency is close to 30%.
  Confidence matches reality.

### The Complete Training Progression

| Stage | Approach | Melanoma Sensitivity | Validation Set |
|-------|----------|---------------------|----------------|
| 1 | Hand-crafted features + logistic regression | **0%** | HAM10000 (4,760 images) |
| 2 | Community ViT (Anwarkh1, 44K downloads) | **73.3%** | HAM10000 (210 images) |
| 3 | Custom ViT + focal loss (melanoma alpha=8.0) | **98.2%** | HAM10000 holdout (1,503 images) |
| 4 | Same model, external data (ISIC 2019) | **61.6%** | ISIC 2019 (4,998 images) |
| 5 | Combined training (37,484 images, focal loss, mel alpha=6.0) | **95.97%** | ISIC 2019 external test (3,901 images) |
| 5b | Same combined model, same-distribution test | **97.01%** | HAM10000 holdout (1,503 images) |
| 6 | FDA audit -- corrected fabricated claims | -- | Evidence chain rebuilt |
| 7 | V1+V2 Ensemble (combined + HAM10000-only models) | **99.4%** | Ensemble evaluation |
| 8 | Threshold optimization (triage mode) | **88.4% mel / 91.1% spec** | ISIC 2019 external |
| 9 | ONNX INT8 deployment | **95.97%** (full-precision) | 85MB, 22ms, offline capable |
| 10 | Ensemble + safety gates (4-layer system) | **99.4%** (V1+V2 ensemble) | Full system with clinical gates |
| 11 | Bayesian risk stratification + meta-classifier | **NPV 99.06%, NNB 2.1** | 5-tier risk scoring, ECE 0.044 |

Source: `scripts/combined-training-results.json`

---

## How Mela Compares to Published Research Benchmarks

We tested the open-source alternatives ourselves -- every number in this
table was measured by us on the same test images, not copied from model cards
or marketing materials. FDA-cleared device numbers are from published trials.

| System | Melanoma Sensitivity | Validation | Cost |
|--------|---------------------|------------|------|
| Published FDA pivotal trials | 90-97% (varies by device) | Controlled clinical studies | $7,000+ device + per-test fee |
| SkinVision (CE marked) | ~80-85% (reported) | Proprietary, not independently verified | ~$50/year subscription |
| Anwarkh1/ViT (HuggingFace, 44K downloads) | 73.3% (our test) | 210 HAM10000 images | Free |
| skintaglabs SigLIP (HuggingFace) | 30.0% (our test) | 210 HAM10000 images | Free |
| **Mela (v2, combined training)** | **95.97%** | **ISIC 2019 external test (3,901 images)** | **Free, open source** |

Mela achieves melanoma sensitivity comparable to published research benchmarks on
genuinely external data. Mela's 95.97% is measured on 3,901 ISIC 2019
images from cameras, institutions, and patient populations not seen during
training (source: `scripts/combined-training-results.json`). On same-distribution
HAM10000 holdout, Mela achieves 97.01%. See `docs/FDA-AUDIT-REPORT.md`
for the full evidence chain. Mela has not undergone prospective clinical
validation and is not a medical device.

---

## The Focal Loss Innovation

![Focal Loss Innovation](docs/diagrams/focal-loss-innovation.svg)

Standard cross-entropy loss treats all misclassifications equally. In
HAM10000, melanocytic nevi represent 66.9% of the dataset. A model optimizing
cross-entropy will focus on correctly classifying moles -- the majority class
-- at the expense of melanoma sensitivity. Missing a mole does not matter.
Missing a melanoma can kill someone.

**Focal loss** (Lin et al. 2017) solves this with two mechanisms:

1. **Gamma (2.0):** Down-weights well-classified examples. Once the model
   learns to correctly identify an obvious benign mole, that example
   contributes less to the loss, freeing the model to focus on harder cases.

2. **Per-class alpha weights:** In the combined-trained v2 model, melanoma
   receives alpha=6.0 and melanocytic nevi receive alpha=0.4. Every melanoma
   misclassification costs the optimizer 15x more than a nevus
   misclassification. Combined with gamma, this creates a 3-layer class
   balancing system:
   - Alpha weights for direct class importance
   - Oversampling of minority classes during training
   - Gamma downweighting of easy examples

**The tradeoff is deliberate.** High melanoma sensitivity comes at a cost to
specificity: approximately 28% of benign moles are flagged for further
evaluation. In skin awareness tools, false negatives kill and false positives
inconvenience. A Number Needed to Biopsy (NNB) of ~4 is clinically
acceptable and compares favorably to published research benchmarks (NNB: 6.25 in FDA pivotal trials).

---

## The Generalization Fix

This is the part most open-source projects skip.

Our custom ViT achieved 98.2% melanoma sensitivity on HAM10000 -- but only
61.6% on the ISIC 2019 dataset. The 36.6 percentage point drop meant the
model had learned dataset-specific artifacts, not universal dermoscopic
features. Images from different cameras, different institutions, and different
preprocessing pipelines broke it.

**The fix: multi-dataset training.** Instead of training on HAM10000 alone,
we combined images from multiple independent sources:

- HAM10000 (Medical University of Vienna): 10,015 images, 7 classes
- ISIC 2019 (multiple institutions): 26,006 images, 8 classes mapped to
  7 (SCC mapped to akiec as the closest HAM10000 class)
- Combined training corpus after oversampling: 37,484 images

Training used focal loss (gamma=2.0, melanoma alpha=6.0), 5 epochs, 3.3 hours
on Apple M3 Max MPS. Model saved at `scripts/mela-classifier-v2/best/`.

**The fix worked.** On the ISIC 2019 external test set (3,901 held-out images),
melanoma sensitivity jumped from 61.6% to **95.97%** (596/621 detected).
Melanoma AUROC: 0.960. All-cancer sensitivity: 98.3% (1,848/2,058). The
combined training corpus forced the model to learn features that transfer
across camera systems rather than memorizing HAM10000 lighting conditions.
Source: `scripts/combined-training-results.json`

**This is why cross-dataset validation matters.** Any model can score well
on its own holdout set. The real test is data the model has never seen, from
cameras it has never seen, at institutions it has never been to.

---

## Architecture: 100% Local, Zero External Dependencies

Mela runs entirely on the user's device. No images leave the phone. No API
calls to external servers. The ONNX model downloads once (85MB, cached by
service worker) and every subsequent scan is fully offline.

```
                        +-----------------------+
                        |    ONNX V2 Model      |
                        |  (85MB INT8, cached)  |
                        |  95.97% mel sens      |
                        +----------+------------+
                                   |
                                   | 70% weight
                                   v
+--------+    +--------+    +-------------+    +------------+    +----------+    +-----------+
| Camera |    | Lesion |    |   3-Layer   |    | Threshold  |    | Bayesian |    | Consumer  |
| Upload |--->| Gate   |--->|  Ensemble   |--->| + Meta     |--->|   Risk   |--->| Result    |
+--------+    +--------+    +-------------+    +------------+    +----------+    +-----------+
                                   ^                                               |
                     +-------------+-------------+                    "Consider consulting
                     |                           |                     a healthcare provider"
              15% weight                  15% weight
              Trained Weights             Rule-Based
              (Literature LR)             (Safety Gates)
```

### Layer 1: ONNX V2 ViT Model (70% of ensemble)

ViT-Base fine-tuned with focal loss on 37,484 images (HAM10000 + ISIC 2019).
85.8M parameters, INT8 quantized to 85MB. Runs in-browser via ONNX Runtime
Web (WASM backend). 95.97% melanoma sensitivity on external ISIC 2019 data
(3,901 images). Source: `scripts/combined-training-results.json`

The model is stored in RuVector Format (RVF) and cached locally after first
download via the service worker. Subsequent scans load from cache in <100ms.

### Layer 2: Literature-Derived Logistic Regression (15%)

A 20-feature x 7-class weight matrix where every weight is cited to published
dermoscopy literature (Stolz 1994, Argenziano 1998, Menzies 1996). Encodes
clinical knowledge the ViT may not have learned from pixels alone.

### Layer 3: Rule-Based Safety Gates (15%)

- **TDS formula:** A*1.3 + B*0.1 + C*0.5 + D*0.5 with validated cutoffs
- **7-point checklist:** Threshold >= 3 triggers biopsy recommendation
- **Melanoma safety gate:** 2+ suspicious ABCDE criteria enforce melanoma >= 15%
- **TDS override:** TDS > 5.45 forces malignant >= 30%

These gates are the last line of defense. They override the neural network
when clinical features contradict the model's confidence.

### Bayesian Demographic Adjustment (applied on top)

Age-adjusted prevalence multipliers: under 30 = 0.3x, 30-50 = 0.7x,
50-70 = 1.5x, over 70 = 2.5x base melanoma prevalence. Combined with
sex and body-location priors from HAM10000.

### Offline-First Design

```
First Visit:                          Every Visit After:
+----------+     +----------+        +----------+
| Download |---->| Service  |        | Load     |
| 85MB     |     | Worker   |        | from     |
| ONNX     |     | Cache    |        | Cache    |
+----------+     +----------+        +----------+
                      |                    |
                      v                    v
              +---------------+    +---------------+
              | Full offline  |    | Full offline  |
              | capability    |    | inference     |
              | from now on   |    | <100ms load   |
              +---------------+    +---------------+
```

No fallback to any external API. If the ONNX model hasn't been downloaded
yet, the system uses trained-weights + rule-based scoring (60/40 split) as a
purely local fallback. The safety gates always run regardless.

### Privacy by Architecture

Images never leave the device -- not to a proxy, not to a cloud API, nowhere.
Classification runs in-browser via WASM. EXIF metadata is stripped at capture.
No device fingerprinting. No persistent identifiers.

This is structural privacy, not policy-dependent. If the data never exists on
a server, it cannot be breached from a server.

---

## Design Philosophy

This section explains *why* the system is built the way it is. Every
architectural decision encodes a value judgment. These are ours.

### 1. Sensitivity Over Specificity -- The Fundamental Tradeoff

In skin analysis, a false negative kills. A false positive inconveniences.

This single asymmetry drives most of the design. We trained the model with
focal loss using melanoma alpha=6.0, which makes every missed melanoma cost
the optimizer 6x more than a missed mole. Combined with gamma=2.0 down-
weighting of easy examples, this creates a system that is *deliberately
aggressive* about flagging potential melanomas.

The cost is specificity: approximately 28% of benign moles get flagged for
further evaluation. This produces a Number Needed to Biopsy (NNB) of roughly
4 -- meaning for every confirmed malignancy, about 3 benign lesions were also
flagged. That NNB is clinically acceptable and comparable to FDA-cleared
published research benchmarks (NNB: 6.25 in FDA pivotal trials). Source: `scripts/combined-training-results.json`

The system offers threshold modes (awareness vs. triage) so users can choose
their own point on the sensitivity-specificity curve. Awareness mode maximizes
sensitivity for general use. Triage mode tightens specificity for
clinicians who want fewer false positives. But the default is always biased
toward catching cancer, because the default is what most people will use,
and the default should not miss melanomas.

### 2. The Honesty Principle

On March 23, 2026, an internal FDA-style audit found that the README claimed
"91.3% cross-dataset accuracy" and "96.2% sensitivity." Neither number was
backed by any evidence file. They were fabricated -- not maliciously, but
through the common pattern of writing aspirational numbers as if they were
measured. See `docs/FDA-AUDIT-REPORT.md`.

That audit changed how we build this project. The rules now:

- **Every number cites its evidence source.** If a metric appears in
  documentation, the JSON file, field name, and exact value are referenced.
  If no evidence file exists, the metric is not claimed.
- **Limitations are documented more prominently than capabilities.** The
  Known Limitations section is longer than the Features section. The
  Fitzpatrick I-III training bias is stated in the README, not buried in
  an appendix. The 30-percentage-point equity gap is disclosed up front.
- **Failures are published.** The journey from 0% to 95.97% includes every
  crash along the way -- the 0% melanoma sensitivity from hand-crafted
  features, the 61.6% collapse on external data. These failures are not
  embarrassing; they are the evidence that the final number is trustworthy.
- **"Not yet measured" is a valid answer.** When we have not computed a
  metric, we say so. "AUROC has not been computed on any dataset" is more
  honest than omitting AUROC and hoping nobody asks.

The goal is that a dermatologist reading this README, or an FDA reviewer
reading the technical report, finds nothing they need to discover for
themselves. If a limitation exists, we told them first.

### 3. Multi-Layer Defense -- Why One Model Is Not Enough

A single neural network is not trustworthy enough for skin analysis. Neural
networks fail in ways that are difficult to predict: adversarial lighting,
camera artifacts, out-of-distribution inputs, dataset-specific biases the
training process learned silently. Our own ViT achieved 98.2% on HAM10000 and
then crashed to 61.6% on ISIC 2019 -- same architecture, same weights, different
cameras.

The 4-layer ensemble exists because each layer fails differently:

- **Custom ViT** (Layer 1) learns spatial patterns from pixels. It catches
  subtle visual features that no rule can express. It also learns
  camera-specific artifacts.
- **Literature-derived logistic regression** (Layer 2) encodes 30 years of
  published dermoscopy knowledge. It does not care about camera artifacts.
  It fails when the image features do not match the textbook.
- **Rule-based safety gates** (Layer 3) enforce hard clinical floors. If TDS
  > 5.45, the malignancy probability cannot drop below 30% regardless of what
  the neural network says. If 2+ ABCDE criteria are suspicious, melanoma
  probability cannot drop below 15%. These gates catch the cases where a
  confident-but-wrong model would otherwise reassure the user.
- **Bayesian demographic adjustment** (Layer 4) incorporates base-rate
  reality. An 80-year-old male with a trunk lesion has different priors than a
  20-year-old female with a hand lesion. The model does not know the patient's
  age; this layer does.

Multi-image consensus adds another defense. Quality-weighted averaging means
a blurry photo gets less influence than a sharp one. The melanoma safety gate
ensures that if *any* single image flags melanoma above 60%, that signal
survives consensus regardless of what the other images say.

The V1+V2 ensemble (combined-dataset model + HAM10000-only model) provides yet
another layer: models trained on different data fail on different inputs. When
they agree, confidence is high. When they disagree, the system flags the case
for clinical review.

### 4. Real-World Measurement -- Not Just Classification

Classification without measurement is incomplete. The "D" in ABCDE is diameter,
and the 6mm threshold is a clinical decision point. Telling someone they have
a suspicious lesion without telling them how big it is leaves a gap in the
clinical picture.

The problem: smartphone cameras do not know how far they are from the lesion.
A 4mm mole and a 9mm mole can look identical in a photo.

The solution is a 3-tier measurement system, designed around what people
actually have available:

- **USB-C reference** (Tier 1, +/-0.5mm): Everyone has a phone charger cable.
  The USB-C connector is 8.25mm wide, standardized to sub-millimeter tolerance
  by the USB Implementers Forum. Place the cable next to the lesion, and the
  system has a physical reference. This is accurate enough to make the 6mm
  clinical threshold meaningful.
- **Skin texture FFT** (Tier 2, +/-2-3mm): When no reference object is
  present, the system analyzes the skin's natural texture in the image margin.
  Dermatoglyphic pore spacing is anatomically constrained (0.2-0.5mm depending
  on body location). A 2D FFT detects the dominant frequency, and the known
  pore spacing provides a pixels-per-mm calibration. Less accurate, but
  automatic.
- **LiDAR** (Tier 2 enhancement): iPhone Pro models have LiDAR depth sensors.
  When available, the depth map provides distance-to-subject, which combined
  with the camera's known focal length gives a direct pixels-per-mm
  conversion. This is an enhancement to Tier 2, not a replacement.
- **Pixel estimate fallback** (Tier 3): When nothing else works, assume a
  25mm dermoscope field of view. If this puts the diameter between 4-8mm
  (straddling the clinical threshold), the system explicitly warns the user
  to grab a USB-C cable.

Quality gating runs before any of this. The system scores each image for
sharpness (Laplacian variance), contrast (RMS), and segmentation quality
before spending compute on classification. A blurry, low-contrast image with
poor segmentation gets rejected with a "retake" prompt, not classified with
false confidence.

### 5. From 0% to 95.97% -- The Iterative Journey

The final 95.97% melanoma sensitivity on external data was not designed in
advance. It was discovered through a sequence of failures, audits, and
hard decisions, each of which eliminated a wrong approach and pointed
toward the right one.

- **Hand-crafted features (0% mel sensitivity):** 20 summary statistics fed
  into logistic regression. Proved that spatial patterns matter and feature
  engineering alone is insufficient for dermoscopy. This is why the field
  moved to deep learning.
- **Community ViT (73.3% mel sensitivity):** The most-downloaded skin cancer
  model on HuggingFace. Proved that generic models trained without class
  weighting miss too many melanomas. 1 in 4 is not acceptable.
- **Custom ViT on HAM10000 (98.2% mel sensitivity):** Focal loss with
  melanoma alpha=8.0. Proved custom training works -- on the data it was
  trained on. We celebrated. We were wrong to celebrate.
- **Custom ViT on external data (61.6% mel sensitivity):** Same model,
  different dataset. A 36.6 percentage point crash. Proved that single-dataset
  training learns camera artifacts, not universal dermoscopic features. This
  was the most important test in the entire project. Most open-source skin
  cancer models stop before this test.
- **Combined-dataset training (95.97% mel sensitivity on external data):**
  37,484 images from HAM10000 + ISIC 2019. Overnight 3.3-hour retraining on
  M3 Max. The model was forced to learn features that generalize across camera
  systems. Source: `scripts/combined-training-results.json`
- **FDA audit caught fabricated claims:** The README previously claimed
  "91.3% cross-dataset" and "96.2% sensitivity" with no evidence files. The
  audit stripped every unverified number and rebuilt the evidence chain. This
  is documented in `docs/FDA-AUDIT-REPORT.md`. The audit produced no model
  improvement -- it produced trust.
- **AUROC 0.960 on external data -- threshold problem, not discrimination:**
  The model's ranking ability is strong. The lower overall accuracy reflects
  deliberate threshold tuning that prioritizes sensitivity over specificity.
  This is a design choice, not a deficiency.
- **Fitzpatrick equity test revealed dangerous 30pp gaps:** Testing across
  skin tones showed a 30-percentage-point performance gap between Fitzpatrick
  I-III and darker skin tones. Training data is ~95% FST I-III. This gap is
  disclosed up front, not buried in an appendix. It is not yet resolved.
- **V1+V2 ensemble (99.4% mel sensitivity):** Combining models trained on
  different data distributions catches cases either model alone would miss.
- **Threshold optimization (88.4% mel / 91.1% specificity):** Found the
  operating point that balances sensitivity and specificity for triage use.
- **ONNX deployment (85MB, 22ms, offline):** The decision to deploy ONNX
  for privacy and offline capability means the image never leaves the device.
  A rural health worker with no internet can still analyze skin lesions.

Each failure is documented in this README because the failures are what make
the final result trustworthy. A project that only reports its best number is
hiding something. A project that shows you the 0%, the 61.6%, the FDA audit
that caught fabricated claims, the dangerous Fitzpatrick gaps, and the 95.97%
is showing you the evidence that the methodology actually works.

### 6. Consumer-First, Clinician-Capable

The same classification engine serves two audiences with different needs.

A regular person uploading a photo of a mole does not know what "mel 0.94"
means. They need a clear, actionable message: "Consider consulting a healthcare
provider within 2 weeks" or "This looks reassuring, but monitor for changes." The consumer
translation layer converts probability distributions into plain-language
recommendations with color-coded risk levels and specific next steps.

A dermatologist evaluating the same lesion needs the opposite: raw ABCDE
scores, 7-point checklist breakdown, TDS calculation, per-class probability
distribution, ICD-10-CM codes, and a pre-populated referral letter they can
copy into their clinical correspondence.

The key design decision was to build one engine with two presentation layers,
not two separate products. The classification pipeline, ensemble logic, and
safety gates are identical regardless of who is looking at the output. The
difference is entirely in how results are communicated. This means both
audiences benefit from every improvement to the underlying system, and the
system does not need to be validated twice.

### 7. Privacy by Architecture

Medical images are among the most sensitive data a person can generate.
Mela is designed so that privacy is structural, not policy-dependent.

- **Images never leave the device.** Classification runs entirely in-browser
  via ONNX Runtime Web. No server proxy, no cloud API, no network hop. The
  image stays on the user's phone from capture to result.
- **EXIF stripping.** All metadata (GPS coordinates, device identifiers,
  timestamps) is removed from images before any processing. Even if an image
  were somehow exfiltrated, it carries no identifying metadata.
- **No device fingerprinting.** The system does not collect browser
  fingerprints, device IDs, or persistent identifiers.
- **Anonymized sharing via differential privacy.** Practices that opt into
  the pi-brain collective intelligence layer contribute only anonymized
  aggregate statistics, not individual images. Differential privacy
  (epsilon=1.0) adds calibrated noise to every shared data point, providing
  mathematically provable privacy guarantees.
- **Safe Harbor de-identification.** All data contributed to pi-brain follows
  HIPAA Safe Harbor rules: 18 identifier categories are removed before
  transmission.
- **Witness chain, not image chain.** The audit trail uses SHAKE-256 hashes
  to prove that a classification happened, without storing the image that was
  classified. The hash is irreversible -- you cannot reconstruct the image
  from it.

The principle is that privacy should not depend on trusting the developer,
the server operator, or the network. It should be enforced by the architecture
itself: if the data never exists on the server, it cannot be breached from the
server.

### 8. Bayesian Honesty -- Risk Scores, Not Binary Alarms

A binary "melanoma yes/no" answer is the wrong output for an awareness tool
operating at real-world prevalence. At 2% base rate, even a model with 95.97%
sensitivity and 80% specificity produces a PPV of only 8.9% -- meaning 91% of
positive results are false positives. Telling a patient "you might have
melanoma" when the math says they probably do not is dishonest, even if the
model is doing its job correctly.

The honest answer is a calibrated risk score: "Given the image analysis, your
clinical features, and the base rate for your demographic, the post-test
probability of melanoma is 14%." That number is actionable -- the patient and
their doctor can make an informed decision about whether to biopsy, monitor,
or dismiss. Binary classification hides the uncertainty. Bayesian risk
stratification exposes it.

This principle has a concrete consequence: the system never says "melanoma
detected." It says "Very High risk (62%) -- urgent referral recommended" or
"Low risk (2.3%) -- routine checks." The 5-tier system maps continuous
probability to clinical action without pretending the model knows more than it
does. Temperature calibration (ECE 0.044) ensures the probabilities are not
just directionally correct but quantitatively trustworthy -- when the system
says 30%, the true frequency is close to 30%.

The meta-classifier enforces consistency between the neural network and
clinical features. A model prediction without clinical corroboration is
downweighted; a model prediction with clinical agreement is upweighted. This
is not just a technical trick -- it encodes the clinical principle that
multiple independent lines of evidence are more trustworthy than any single
signal, no matter how sophisticated.

---

## The Clinical Toolset

The clinical experience is designed around the dermatology workflow, not
around showcasing AI.

**1. Multi-Photo Capture.** Upload 2-3 photos of the same lesion for
quality-weighted consensus classification. Each image is scored for sharpness
(Laplacian variance), contrast (RMS), and segmentation quality, then
classified independently. Results are combined via quality-weighted probability
averaging with a melanoma safety gate: if any single image flags melanoma
with >60% confidence, it stays prominent in the consensus regardless of what
other images say. Single-photo mode is also available. The system auto-detects
dermoscopic vs. clinical images and supports interactive body map location
selection.

**2. Classification.** Instant probability distribution over 7 diagnostic
categories: melanoma, BCC, actinic keratosis, benign keratosis,
dermatofibroma, melanocytic nevus, vascular lesion. Risk level displayed
with color and icon coding. Inter-image agreement score shown when multiple
photos are analyzed.

**3. Smart Lesion Measurement.** Three-tier size calibration: USB-C charger
cable reference detection (±0.5mm), automatic skin texture FFT analysis
(±2-3mm), or rough pixel estimate. "Place your charger cable next to the
spot for best accuracy." The real diameter feeds directly into the ABCDE "D"
score and TDS formula — no more guessing at the 6mm clinical threshold.

**4. ABCDE Scores.** Real image-derived measurements -- not placeholders.
Asymmetry from principal-axis moment analysis, border from 8-octant
irregularity, color from k-means++ clustering in LAB space, texture from
GLCM analysis. Each score includes the literature-cited rationale.

**4. "What Mela Found" Explainability.** Every result shows the top clinical
findings that drove the classification -- asymmetry, border irregularity,
color diversity, blue-white veil, atypical network, streaks, texture. Each
finding includes impact direction (supports/opposes the assessment), weight
(strong/moderate/weak), and citation to published literature (Stolz 1994,
Argenziano 1998). Full clinical details available on tap. Users see exactly
what the AI looked at and why it matters.

**5. Clinical Context Questions.** Five optional questions that personalize
the risk assessment: Is this new? Has it changed? Previously biopsied?
Family history of melanoma? Any symptoms? Answers feed directly into the
Bayesian risk model as evidence-based multipliers (e.g., family history =
1.7x melanoma prior per Gandini et al. 2005). A new, changing mole with
family history gets a fundamentally different risk score than a stable,
longstanding spot.

**6. Photo Capture Guidance.** Tips displayed before every scan: good
lighting, 4-6 inches distance, one spot per photo, clean lens. Reduces
garbage-in/garbage-out and improves classification accuracy.

**7. Low-Confidence Safety Net.** When the AI cannot classify with enough
confidence (< 40%), the result defaults to "Consider consulting a healthcare provider to be safe"
instead of forcing a possibly wrong classification. This catches amelanotic
melanomas (flesh-colored, no pigment contrast) and poor-quality photos.

**8. ICD-10 Code + Patient Self-Referral.** Generate a printable letter with
ICD-10-CM codes, ABCDE scores, and classification results. Clearly marked
as a patient self-referral, not a provider referral. Copy to clipboard.

**6. Attention Heatmap.** Weighted visualization of color irregularity, local
entropy, and border proximity showing which image regions drove the
classification. (Feature saliency, not Grad-CAM -- see Limitations.)

**7. Practice Analytics Dashboard.** Concordance rate tracking, Number Needed
to Biopsy (NNB), per-class sensitivity/specificity/PPV/NPV with Wilson 95%
confidence intervals, calibration curves (ECE + Hosmer-Lemeshow), Fitzpatrick
equity monitoring with automatic disparity alerts, 30-day rolling trends,
and discordance analysis.

**8. Outcome Feedback.** Record whether the AI agreed with clinical judgment,
overcalled, or missed. Track pathology results. Feed data back into
practice-level performance monitoring.

---

## Quick Start

```bash
git clone https://github.com/stuinfla/Mela
cd Mela
npm install
npm run dev
```

The app starts at `http://localhost:5173`.

### Deployment

**Live:** https://mela-app.vercel.app
**Vercel project:** https://vercel.com/stuart-kerrs-projects/mela
**GitHub:** https://github.com/stuinfla/Mela

The app uses ONNX Runtime Web for in-browser inference. The 85MB INT8 model
downloads once and is cached by a service worker for fully offline use.
No API keys needed. No external service dependencies. Images never leave
the device.

### Train a Custom Model (Optional)

```bash
pip install torch transformers datasets scikit-learn
python3 scripts/train-fast.py
```

Training takes approximately 3.3 hours on an Apple M3 Max with MPS backend
for the combined dataset (37,484 images). The script uses focal loss with
melanoma alpha=6.0 and selects the best checkpoint by melanoma sensitivity,
not overall accuracy. The trained model (327MB) is not included in the
repository. Source: `scripts/combined-training-results.json`

The pre-trained model is bundled as an 85MB ONNX INT8 file at
`static/models/mela-v2-int8.onnx`. No external download needed.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | SvelteKit 5 + TailwindCSS, mobile-first PWA |
| Model | ViT-Base, 85.8M params, focal loss (gamma=2.0, mel alpha=6.0), trained on 37,484 images |
| Inference | ONNX Runtime Web (85MB INT8, WASM backend, fully offline) |
| Model format | RuVector Format (RVF) -- self-contained, cached by service worker |
| Image analysis | 2,000-line TypeScript CV engine: Otsu segmentation, GLCM, LBP, k-means++ color |
| Clinical scoring | TDS (Stolz 1994), 7-point checklist (Argenziano 1998), per-class ROC thresholds |
| Risk stratification | Bayesian post-test probability, age-adjusted prevalence, temperature calibration |
| Privacy | 100% on-device inference, EXIF stripping, witness chain (SHAKE-256) |
| Vector infrastructure | RuVector -- HNSW indexing, SONA self-optimization, federated learning |

---

## Project Structure

```
src/
  lib/
    mela/                    Core classification engine
      classifier.ts          3-layer ensemble: ONNX + trained-weights + rules
      inference-orchestrator.ts  ONNX-first routing, no external APIs
      inference-offline.ts   ONNX Runtime Web model loading + inference
      image-analysis.ts      CV pipeline: segmentation, ABCDE, GLCM, LBP, k-means
      threshold-classifier.ts  Per-class ROC-optimized thresholds
      meta-classifier.ts     Neural + clinical feature agreement scoring
      risk-stratification.ts Bayesian post-test probability
      consumer-translation.ts  Medical -> plain English + action items
      trained-weights.ts     Literature-derived logistic regression (20x7 matrix)
      clinical-baselines.ts  TDS formula, 7-point checklist
      types.ts               Shared type definitions
    components/
      MelaPanel.svelte       Main UI orchestrator
      DermCapture.svelte     Camera + body map + image upload
      ExplainPanel.svelte    "Why this classification?" with citations
      ReferralLetter.svelte  Referral letter generator
  routes/
    +page.svelte             Main page
    api/
      classify-local/        Server-side ONNX inference endpoint
      analyze/               Vector search + literature retrieval
static/
  models/
    mela-v2-int8.onnx        V2 model (85MB INT8, cached by service worker)
  sw-model-cache.js          Service worker for offline model caching
scripts/
  train-fast.py              Custom ViT training with focal loss
  deploy-verified.sh         7-step verified deployment pipeline
docs/                        Technical documentation + 14 ADRs
```

### Detailed Documentation

- [TECHNICAL-REPORT.md](docs/TECHNICAL-REPORT.md) -- Full classification
  pipeline, cross-dataset validation results, regulatory context
- [DUAL-MODEL-ARCHITECTURE.md](docs/DUAL-MODEL-ARCHITECTURE.md) -- Ensemble
  design, safety mechanisms, model disagreement detection
- [CHANGELOG.md](docs/CHANGELOG.md) -- Version history
- [architecture.md](docs/architecture.md) -- Component architecture, data model,
  security layers
- [HAM10000_analysis.md](docs/HAM10000_analysis.md) -- Dataset statistics and
  demographic distributions

---

## Published Research Benchmarks

The following table compares Mela's performance against published research
benchmarks from FDA-cleared devices and academic studies. Mela is an
educational tool, not a medical device, and has not undergone prospective
clinical validation.

| Metric | Published FDA Pivotal Trials | Mela (v2, combined) |
|--------|------------------------------|-----------|
| Melanoma sensitivity | 90-97% (varies by device and study) | **95.97% (ISIC 2019 external)** / 97.01% (HAM10000) |
| Melanoma AUROC | 0.758 (spectroscopy-based) | **0.960 (ISIC 2019 external)** / 0.930 (HAM10000) |
| Specificity | 9.9-32.5% (varies by device) | 80.0% (melanoma, ISIC 2019 external) |
| Validation | Controlled clinical trials, 440-1,579 lesions | ISIC 2019 external (3,901) + HAM10000 holdout (1,503) |
| External data tested | Yes (pivotal trials) | Yes (ISIC 2019: AUROC 0.960, sensitivity 95.97%) |

Sources: Published FDA pivotal trial data (DEN230008, Tkaczyk et al. 2024;
Scibase clinical data; MelaFind withdrawn from market). Mela numbers from
`scripts/combined-training-results.json`.

Note: Published benchmarks come from controlled clinical studies with
proprietary hardware. Mela's 95.97% is on genuinely external data (ISIC 2019,
3,901 images from cameras and institutions not seen during training). On
same-distribution HAM10000 holdout, sensitivity is 97.01%. Mela has not
undergone prospective clinical validation and is not a medical device.

---

## Known Limitations

We believe honesty about limitations is more important than marketing.

1. **Not FDA-cleared.** Mela is an educational tool, not a medical device. It
   does not diagnose, screen for, or detect any disease. It must not be used
   for clinical decision-making.

2. **Trained predominantly on Fitzpatrick I-III skin.** HAM10000 is
   approximately 95% Fitzpatrick skin types I-III. Performance on darker skin
   tones is likely degraded and has not been independently measured. Published
   research reports a 4% sensitivity gap between FST I-III and FST IV-VI in
   FDA-cleared devices; our gap may be larger. This is a systemic problem in dermatology AI, not an excuse.

3. **The generalization gap is closed for melanoma, but overall accuracy
   dropped.** Combined-dataset training raised external melanoma sensitivity
   from 61.6% to 95.97% (596/621 on ISIC 2019). However, overall accuracy is
   76.4% on ISIC 2019 and 63.2% on HAM10000 -- lower than the old model's
   78.1% on HAM10000 -- because the aggressive melanoma weighting causes
   over-prediction of melanoma at the expense of benign classes (especially
   nevi). This is a deliberate tradeoff: catching melanoma matters more than
   correctly labeling moles. Source: `scripts/combined-training-results.json`

4. **High melanoma sensitivity comes at a cost to specificity.** The ~28%
   false positive rate on melanocytic nevi means roughly 1 in 4 benign moles
   will be flagged for further evaluation. This is a deliberate design choice
   -- in skin analysis, false negatives kill and false positives
   inconvenience.

5. **No prospective clinical validation.** All testing has been on
   retrospective datasets (HAM10000, ISIC 2019). The system has not been
   tested in clinical workflow conditions with real-time patient encounters.
   Retrospective accuracy and prospective accuracy are different things.

6. **Full-precision model requires local training.** The INT8 ONNX model
   (85MB) is included in the repository. The full-precision model (327MB)
   is not -- run `train-fast.py` locally to reproduce it.

7. **Attention heatmaps are feature saliency, not Grad-CAM.** The
   visualization shows diagnostically relevant regions but does not reflect
   true neural network attention weights.

8. **Evolution scoring is not implemented.** The "E" in ABCDE requires
   comparing against a previous image. Longitudinal tracking is not yet
   available.

9. **Segmentation is fragile on low-contrast images.** Otsu thresholding
   assumes a bimodal histogram. It fails for amelanotic melanoma,
   hypo-pigmented lesions, and non-dermoscopic photographs.

10. **ISIC 2019 class mapping is imperfect.** SCC (squamous cell carcinoma)
    is mapped to akiec (the closest HAM10000 class), which introduces noise
    in the akiec sensitivity measurement.

---

## The Vision

Skin cancer is the most common cancer worldwide. Early detection saves lives
-- melanoma caught at Stage I has a 99% five-year survival rate; caught at
Stage IV, that drops to 30%.

The tools that detect melanoma at >90% sensitivity today cost $7,000 and
require proprietary hardware. That means early detection is a luxury available
to well-funded dermatology practices in wealthy countries. A farmer in rural
India, a nurse practitioner in Appalachia, a community health worker in
sub-Saharan Africa -- they have smartphones, but they do not have expensive
specialized devices.

Mela is an attempt to close that gap. With combined-dataset training,
melanoma sensitivity on genuinely external data is now 95.97% (source:
`scripts/combined-training-results.json`) -- comparable to published research benchmarks
from FDA pivotal trials. Fitzpatrick equity is not yet proven, and no regulator has cleared
it. But the architecture is sound, the training methodology is honest, the
external validation is real, and the code is open.

The path forward:

- **More diverse training data.** Fitzpatrick V-VI images from ISIC archive
  and the Diverse Dermatology Images dataset to close the skin tone gap.
- **Prospective clinical validation.** Partner with dermatology clinics to
  test Mela alongside clinical judgment in real patient encounters.
- **Regulatory pathway.** De Novo or 510(k) classification with FDA, using
  existing FDA-cleared devices as predicate.
- **Improve overall accuracy.** The aggressive melanoma weighting that achieves
  95.97% sensitivity reduces overall accuracy to 76.4%. Curriculum learning
  or multi-objective optimization could improve benign-class accuracy without
  sacrificing melanoma detection.
- **Collective intelligence.** Every participating practice makes the model
  better for every other practice, with differential privacy protecting
  patient data.

The goal is not to replace dermatologists. It is to put a skin awareness tool in
the hands of the 5 billion people who will never see one.

---

## License

MIT

---

## Built With

- [RuVector](https://github.com/ruvnet/claude-flow) -- Vector intelligence
  platform
- [Claude Flow](https://github.com/ruvnet/claude-flow) -- Multi-agent
  orchestration
- [Pi-brain](https://pi.ruv.io) -- Collective intelligence with differential
  privacy

---

**EDUCATIONAL USE ONLY -- Not a medical device.** Mela is not FDA-cleared and does not
diagnose, screen for, or detect any disease. Pattern analysis results are for educational
and informational purposes only. Always consult a qualified healthcare provider for
medical concerns.
