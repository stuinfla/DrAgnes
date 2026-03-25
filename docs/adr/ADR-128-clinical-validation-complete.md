Updated: 2026-03-24 18:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-128: Complete Clinical Validation -- What an FDA Reviewer Would Require

## Status: PROPOSED | Last Updated: 2026-03-24 18:00 EST

## Context

Mela is a consumer-facing AI skin lesion classifier targeting Class II medical device clearance under FDA CDRH review. The proposed regulatory pathway is 510(k) with DermaSensor (DEN230008, cleared January 2024) as the predicate device. This ADR documents every metric and validation step needed for FDA clearance, written from the perspective of an FDA CDRH reviewer. It separates what we have measured from what we have not, what we can compute from code from what requires a clinical study, and what gaps make the current regulatory pathway inadequate.

---

## Section 1: Metrics We Have (with Evidence)

Sources: `scripts/combined-training-results.json`, `scripts/threshold-optimization-results.json`, `scripts/ensemble-validation-results.json`. Model: google/vit-base-patch16-224 fine-tuned on HAM10000 + ISIC 2019 (37,484 images after oversampling). Test sets: ISIC 2019 holdout (n=3,901) and HAM10000 holdout (n=1,503).

### 1.1 Sensitivity, Specificity, AUROC (ISIC 2019 External Test)

| Class | Sensitivity | Specificity | AUROC | Support |
|-------|------------|-------------|-------|---------|
| mel (melanoma) | 95.97% | 80.00% | 0.9595 | 621 |
| bcc (basal cell) | 87.03% | 96.20% | 0.9792 | 478 |
| akiec (actinic keratosis) | 87.17% | 98.57% | 0.9916 | 959 |
| bkl (benign keratosis) | 55.08% | 97.41% | 0.9452 | 394 |
| df (dermatofibroma) | 99.51% | 99.97% | 1.0000 | 406 |
| nv (melanocytic nevus) | 14.33% | 99.97% | 0.9577 | 621 |
| vasc (vascular lesion) | 99.76% | 99.97% | 1.0000 | 422 |

All-cancer sensitivity (mel+bcc+akiec): **98.3%** (n=2,058). Weighted AUROC: 0.9767. Macro AUROC: 0.9762.

### 1.2 Threshold Optimization (ADR-123)

At optimized mel threshold (0.6204): sensitivity = 93.88%, specificity = 85.34%. Overall accuracy improves from 76.37% to 83.18%.

### 1.3 Ensemble Validation (ADR-125)

V1+V2 weighted ensemble (0.7 V2 + 0.3 V1) with melanoma safety override on HAM10000 holdout (n=1,503): ensemble mel sensitivity = **99.40%**, mel specificity = 61.98%. One missed melanoma out of 167.

### 1.4 Fitzpatrick Equity (ADR-124)

Tested on Fitzpatrick17k subset (n=355). Results: 30pp melanoma sensitivity gap (FST II: 80% vs FST V: 50%). Sample sizes critically small (FST VI: n=18). Severity: DANGEROUS.

### 1.5 What This Establishes

An FDA reviewer would accept 1.1-1.3 as preliminary analytical validation on dermoscopic images. They demonstrate discriminative ability. They do NOT establish clinical validity, clinical utility, or real-world performance.

---

## Section 2: Metrics We Are Missing (CRITICAL)

### 2.1 PPV/NPV at Real-World Prevalence

**Why:** PPV answers "when Mela says melanoma, is it right?" -- the metric consumers act on. Current test-set PPV (47.6%) is misleading because test prevalence (15.9%) vastly exceeds real-world screening prevalence (~2%). At 2% prevalence:
- PPV = (0.9597 * 0.02) / (0.9597 * 0.02 + 0.20 * 0.98) = **8.9%**
- NPV = (0.80 * 0.98) / (0.80 * 0.98 + 0.0403 * 0.02) = **99.9%**

When Mela says "melanoma" at real-world prevalence, it is correct ~9% of the time. When it says "not melanoma," it is right 99.9%. This is the fundamental screening tradeoff.

**How:** Re-derive at prevalences 1%, 2%, 5%, 10% using Bayes' theorem. An FDA reviewer will demand all four.

### 2.2 Number Needed to Biopsy (NNB)

**Why:** NNB = 1/PPV. "How many biopsies to find one cancer?" DermaSensor's NNB = 6.25.

| Prevalence | Mela NNB | DermaSensor NNB |
|------------|--------------|-----------------|
| 2% | 11.2 | 6.25 |
| 5% | 4.7 | -- |
| 10% | 2.4 | -- |

At 2% prevalence, Mela is worse than DermaSensor (11.2 vs 6.25) because our specificity (80%) is lower.

### 2.3 Calibration (Expected Calibration Error)

**Why:** Does "80% confidence" mean 80% accuracy? Neural networks are notoriously overconfident. Uncalibrated confidence scores systematically mislead users -- a design flaw under 21 CFR 820.30(g).

**How:** Bin predictions by confidence, compute actual accuracy per bin, ECE = weighted average of |accuracy - confidence|. If ECE > 0.05, apply temperature scaling. Plot reliability diagram. Not yet measured; ADR-123 deferred this.

### 2.4 Net Benefit / Decision Curve Analysis

**Why:** Is Mela better than "refer all lesions" or "refer none"? If net benefit falls below "treat all" at clinical thresholds, the device is actively harmful.

**How:** Net benefit = (TP/n) - (FP/n) * (p_t / (1 - p_t)). Plot across threshold probabilities 1-50%. The model adds value only where its curve exceeds both baselines.

### 2.5 Failure Mode Analysis

**Why:** The 25 missed melanomas (4.03% FN rate) are the most important images in the dataset. Are they systematically amelanotic, nodular, small, or regressing? Systematic blind spots require disclosure per ISO 14971 FMEA.

**How:** Extract 25 FN images, categorize by subtype (superficial spreading, nodular, lentigo maligna, amelanotic, acral), compute sensitivity per subtype. If any subtype < 80% sensitivity, add UI warning.

### 2.6 Subgroup Analysis

**Why:** Aggregate sensitivity hides disparities. Required strata: Fitzpatrick type (ADR-124 found 30pp gap -- blocking), age (<30/30-50/50-70/>70), sex, body location (head-neck/trunk/extremity/acral), image quality (dermoscopy/clinical/phone). HAM10000 includes age/sex/location metadata enabling most of this on the HAM10000 holdout.

### 2.7 Operating Point Justification

**Why:** FDA will ask "why 85% specificity constraint, not 90% or 80%?" Requires: literature review on screening device specificity norms, sensitivity analysis of NNB/PPV/net-benefit at multiple specificity targets, comparison with DermaSensor's operating point justification.

---

## Section 3: Validation Steps We Cannot Do With Code

### 3.1 Prospective Clinical Study
FDA requires images collected in the intended use environment (phone camera, consumer setting), with histopathology ground truth for biopsied lesions, expert panel consensus for non-biopsied, IRB approval, and pre-registration on clinicaltrials.gov.

### 3.2 Inter-Reader Reliability (MRMC Study)
Compare Mela against GPs (70-80% mel sensitivity), dermatologists (85-95%), and DermaSensor (96% sensitivity, 84% specificity) on the same lesion set.

### 3.3 Real-World Phone Camera Testing
All training used dermoscopy. Consumers will submit phone photos with variable lighting, no polarization, surface reflections, hair/shadows, variable distance/angle, and diverse camera sensors. ADR-127 Gap #2 describes a validation protocol. Until done, ALL performance claims carry the caveat "on dermoscopic images only."

### 3.4 Longitudinal Tracking
Track same lesion over time (the "E" in ABCDE: Evolving). Requires user accounts, image registration, change detection, and 12+ month study. Not required for initial clearance but strengthens the clinical case.

### 3.5 Intended Use Statement (Draft)
> "Mela is a software-only medical device intended to assist adult consumers (18+) in evaluating skin lesions for potential malignancy using smartphone camera images. It provides a risk assessment and recommends whether to seek professional dermatological evaluation. It is NOT intended to diagnose, treat, or replace professional medical evaluation."

Must be validated by regulatory counsel. Every word constrains testing requirements.

### 3.6 Risk Classification (ISO 14971)

| Hazard | Severity | Risk | Mitigation |
|--------|----------|------|------------|
| Missed melanoma (FN) | Catastrophic | HIGH | High-sensitivity threshold, "when in doubt, see a doctor" |
| False alarm causing unnecessary biopsy | Moderate | MEDIUM | Specificity optimization, "screening not diagnosis" language |
| Overconfidence in benign result delays care | Serious | HIGH | Never display "definitely benign," always recommend follow-up |
| Performance disparity on dark skin | Serious | HIGH | Fitzpatrick equity remediation (ADR-124), visible warning |
| Poor image quality, unreliable result | Moderate | MEDIUM | Quality gate (implemented), re-capture prompts |

---

## Section 4: Recommended Clinical Study Design

**Population:** 500+ lesions from 3+ clinical sites. Adults 18+. Minimum 20% Fitzpatrick IV-VI, 15% age >70, 10% acral. Enrich melanoma to ~15% prevalence for power; report enriched and prevalence-adjusted results.

**Gold standard:** Histopathology for biopsied lesions. Consensus of 3 dermatologists for non-biopsied, with 12-month follow-up. Discordant cases resolved by 5-member panel.

**Primary endpoint:** Melanoma sensitivity at pre-specified 85% specificity. H0: sensitivity <= 85%, H1: sensitivity > 85%. Alpha = 0.025 (one-sided).

**Secondary endpoints:** PPV at study prevalence, NPV at 2% real-world prevalence (>99.5%), NNB at 2% (<10), mel sensitivity Fitzpatrick IV-VI (>90%), BCC+akiec sensitivity (>85%), ECE (<0.05), net benefit at 5% threshold (> "treat all").

**Sample size:** Exact binomial test, alpha=0.025, power=0.90, true sensitivity ~95%. Required: ~80 melanoma cases. At 15% enrichment: ~534 lesions. Inflated 10% for dropouts: **590 lesions minimum**. Fitzpatrick IV-VI subgroup (~18 melanomas) underpowered -- report as exploratory.

---

## Section 5: 510(k) vs De Novo -- Why the Pathway Matters

DermaSensor was cleared via **De Novo** (not 510(k)) because no predicate existed. It uses elastic scattering spectroscopy (ESS), not imaging.

| Dimension | DermaSensor | Mela |
|-----------|-------------|-----------|
| Input | Spectroscopy signal (ESS) | RGB image (phone camera) |
| Hardware | Proprietary device ($5,900) | Any smartphone |
| Setting | Physician office | Consumer home |
| Operator | Licensed healthcare provider | Untrained consumer |
| Output | Binary (investigate/monitor) | 7-class risk + confidence |

**510(k) is unlikely to succeed.** The technology (camera AI vs spectroscopy), setting (home vs clinic), and operator (consumer vs physician) are all fundamentally different. FDA will almost certainly require **De Novo classification** with special controls: clinical validation study, mandatory "see a doctor" messaging, post-market surveillance, Fitzpatrick equity evidence, IEC 62304 software validation, and cybersecurity documentation.

**Alternative: Wellness exemption.** Position as "skin health awareness" with no disease claims, no "melanoma" in consumer language, and prominent disclaimer. Avoids regulation but limits the product to near-uselessness.

---

## Gap Closure Roadmap

| Gap | Effort | Computable | Blocks FDA |
|-----|--------|-----------|------------|
| PPV/NPV at real-world prevalence | 2 hours | YES | YES |
| NNB calculation | 1 hour | YES | YES |
| Calibration (ECE + temperature scaling) | 4 hours | YES | YES |
| Decision curve analysis | 4 hours | YES | YES |
| Failure mode analysis (25 FN melanomas) | 1 day | PARTIAL | YES |
| Subgroup analysis (age, sex, location) | 1 day | YES | YES |
| Operating point justification | 2 days | PARTIAL | YES |
| Fitzpatrick equity remediation | 1-2 weeks | YES | YES |
| Prospective clinical study | 6-12 months | NO | YES |
| Inter-reader reliability (MRMC) | 3-6 months | NO | YES |
| Phone camera validation | 1-2 weeks | PARTIAL | YES |
| Intended use + regulatory counsel | 1 week | NO | YES |
| ISO 14971 risk management file | 1-2 months | NO | YES |
| IEC 62304 software lifecycle | 1-2 months | NO | YES |

**Realistic timeline to FDA submission: 18-24 months.** The code-computable gaps (PPV, NNB, calibration, DCA, subgroup analysis) can be closed in 2 weeks and should be done immediately -- they inform every downstream decision.

---

## References

1. FDA. DermaSensor De Novo (DEN230008), January 2024.
2. FDA. Software as a Medical Device (SaMD): Clinical Evaluation. 2017.
3. FDA. General Wellness: Policy for Low Risk Devices. 2016 (updated 2019).
4. IEC 62304:2006+AMD1:2015. Medical device software -- Software life cycle processes.
5. ISO 14971:2019. Medical devices -- Application of risk management.
6. Vickers AJ, Elkin EB. Decision Curve Analysis. Med Decis Making. 2006;26(6):565-574.
7. Niculescu-Mizil A, Caruana R. Predicting Good Probabilities. ICML 2005.
8. ADR-123: Threshold Optimization. ADR-124: Fitzpatrick Equity. ADR-125: Ensemble. ADR-127: Production Gaps.
9. `scripts/combined-training-results.json`, `scripts/threshold-optimization-results.json`, `scripts/ensemble-validation-results.json`.

## Author

Stuart Kerr + Claude Flow (RuVector/RuFlo)
