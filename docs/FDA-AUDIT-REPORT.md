Updated: 2026-03-23 | Version 1.0.0
Created: 2026-03-23

# FDA CDRH Audit Report: Dr. Agnes AI Dermatoscopy Screening Platform

## Audit Scope

This audit cross-references every numerical performance claim found in Dr. Agnes documentation and user-facing UI components against the actual measured data stored in the project's JSON evidence files. The audit was conducted from the perspective of an FDA CDRH reviewer evaluating a pre-submission package for a Class II medical device.

### Files Audited (Claims)

| ID | File | Description |
|----|------|-------------|
| C1 | `README.md` | Primary public-facing claims |
| C2 | `docs/TECHNICAL-REPORT.md` | Detailed technical claims |
| C3 | `docs/adr/ADR-118-dragnes-production-validation.md` | Roadmap and "current" metrics |
| C4 | `src/lib/components/AboutPage.svelte` | UI-displayed numbers (user-facing) |
| C5 | `src/lib/components/DrAgnesPanel.svelte` | Trust banner (user-facing) |

### Files Audited (Evidence)

| ID | File | Description |
|----|------|-------------|
| E1 | `scripts/cross-validation-results.json` | Cross-validation on HAM10000 variants |
| E2 | `scripts/isic2019-validation-results.json` | External validation on ISIC 2019 |
| E3 | `scripts/multi-image-validation-results.json` | Multi-image consensus validation |
| E4 | `scripts/validation-results.json` | Raw per-image results (210 images, no summary metrics) |
| E5 | `scripts/siglip-test-results.json` | SigLIP and community model comparison |

---

## Section 1: Claim-by-Claim Verification Table

### 1.1 The "91.3%" Headline Claim

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 1 | Melanoma sensitivity, cross-dataset | C1 (README) | 7, 96, 124 | **91.3%** | NONE | **NO EVIDENCE FILE EXISTS** | **RED - NO EVIDENCE** |
| 2 | "cross-dataset validated on 29,540 diverse images" | C1 (README) | 7, 97, 124, 204 | 29,540 images | NONE | **No evidence file contains this number** | **RED - NO EVIDENCE** |
| 3 | 91.3% melanoma detection accuracy | C4 (AboutPage) | 109, 114 | 91.3% | NONE | No evidence file | **RED - NO EVIDENCE** |
| 4 | 29,540 images count | C4 (AboutPage) | 111, 116 | 29,540 | NONE | No evidence file | **RED - NO EVIDENCE** |
| 5 | 91.3% in comparison table | C1 (README) | 142 | 91.3% | NONE | No evidence file | **RED - NO EVIDENCE** |

**Audit Finding**: The single most prominent claim in the entire project -- "91.3% melanoma sensitivity, cross-dataset validated on 29,540 diverse images" -- has ZERO supporting evidence in any JSON results file. No script output, no validation run, no measured data produces this number. The string "91.3" and "0.913" do not appear in any evidence file in the `scripts/` directory. The number 29,540 does not appear in any evidence file either. This claim appears to be fabricated or aspirational and presented as measured.

### 1.2 The "96.2%" Claim (ADR-118 / AboutPage / DrAgnesPanel)

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 6 | Melanoma sensitivity | C3 (ADR-118) | 14, 263 | **96.2%** | NONE | **NO EVIDENCE FILE EXISTS** | **RED - NO EVIDENCE** |
| 7 | Melanoma sensitivity display | C4 (AboutPage) | 110, 115, 169 | 96.2% | NONE | No evidence file | **RED - NO EVIDENCE** |
| 8 | Mel. Sensitivity in comparison table | C4 (AboutPage) | 209 | 96.2% | NONE | No evidence file | **RED - NO EVIDENCE** |
| 9 | Mel. Sensitivity in trust banner | C5 (DrAgnesPanel) | 1182 | 96.2% | NONE | No evidence file | **RED - NO EVIDENCE** |

**Audit Finding**: The number 96.2% does not appear in any evidence file. The closest measured melanoma sensitivities are 98.22% (HAM10000 variants, E1) and 61.62% (ISIC 2019, E2). No measured value of 96.2% exists anywhere. This number appears to be invented.

### 1.3 The "0.936 AUROC" Claim

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 10 | Melanoma AUROC | C3 (ADR-118) | 16, 265 | **0.936** | NONE | **NO AUROC COMPUTED IN ANY EVIDENCE FILE** | **RED - NO EVIDENCE** |
| 11 | AUROC in AboutPage | C4 (AboutPage) | 173 | 0.936 | NONE | No AUROC anywhere | **RED - NO EVIDENCE** |
| 12 | AUROC in comparison table | C4 (AboutPage) | 219 | 0.936 | NONE | No AUROC anywhere | **RED - NO EVIDENCE** |

**Audit Finding**: No evidence file in the project contains the word "AUROC", "auroc", or "auc". No ROC curve analysis was performed. No evidence file computes any area-under-the-curve metric. The 0.936 AUROC claim is entirely unsupported. This is a computed metric that requires specific analysis -- it cannot be derived from the available confusion matrices without the original probability scores.

### 1.4 The "97.3% All-Cancer Sensitivity" Claim

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 13 | All-cancer sensitivity | C3 (ADR-118) | 17, 266 | **97.3%** | NONE | **NO EVIDENCE FILE EXISTS** | **RED - NO EVIDENCE** |

**Audit Finding**: No evidence file contains 97.3% or 0.973 as a measured metric. The ISIC 2019 validation (E2) measures `all_cancer_sensitivity: 0.6176` (61.76%), which is the only "all-cancer" metric in any evidence file. The 97.3% claim contradicts the actual measured data by 35.5 percentage points.

### 1.5 The "73.1% Melanoma Specificity" Claim

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 14 | Melanoma specificity | C3 (ADR-118) | 15, 264 | **73.1%** | NONE | **NO EVIDENCE FILE EXISTS** | **RED - NO EVIDENCE** |
| 15 | Mel. Specificity in AboutPage | C4 (AboutPage) | 214 | 73.1% | NONE | No evidence file | **RED - NO EVIDENCE** |

**Audit Finding**: No evidence file produces 73.1%. The measured melanoma specificities are: 72.08% (Nagabu, E1), 79.32% (marmal88 test, E1), 80.38% (overfitting check, E1), 88.19% (ISIC 2019, E2), 67.07% (multi-image single, E3). None of these is 73.1%.

### 1.6 The "4,431 Test Images" Claim

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 16 | Test image count | C3 (ADR-118) | 14 | **4,431** | NONE | **NO EVIDENCE FILE EXISTS** | **RED - NO EVIDENCE** |

**Audit Finding**: No evidence file contains 4,431 as an image count. The actual test set sizes in evidence files are: 1,000 (Nagabu, E1), 1,285 (marmal88, E1), 2,004 (overfitting check, E1), 4,998 (ISIC 2019, E2), 1,499 (multi-image, E3), 210 (validation, E4/E5). None sum to 4,431.

### 1.7 The "67.1% to 81.4% Overall Accuracy" Claim

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 17 | Overall accuracy range | C3 (ADR-118) | 267 | **67.1% to 81.4%** | NONE | **NO EVIDENCE FILE EXISTS** | **RED - NO EVIDENCE** |

### 1.8 Claims WITH Supporting Evidence (Verified)

| # | Claim | Source | Line(s) | Claimed Value | Evidence File | Measured Value | MATCH? |
|---|-------|--------|---------|---------------|---------------|----------------|--------|
| 18 | Nagabu melanoma sensitivity | C1, C2 | README:67, TR:53 | 98.7% | E1 | 0.9868 (98.68%) | **CLOSE - Rounded up** |
| 19 | marmal88 melanoma sensitivity | C1, C2 | README:67, TR:54 | 100.0% | E1 | 1.0 (100.0%) | **MATCH** |
| 20 | Nagabu image count | C1, C2 | README:67 | 1,000 | E1 | 1,000 | **MATCH** |
| 21 | marmal88 image count | C1, C2 | README:68 | 1,285 | E1 | 1,285 | **MATCH** |
| 22 | Train/test gap | C1, C2 | README:68 | -0.7% | E1 | -0.0006 (-0.06%) | **MISMATCH** |
| 23 | Overfitting check melanoma sens (test) | C2 | TR:645 | 98.2% | E1 | 0.9822 (98.22%) | **MATCH** |
| 24 | ISIC 2019 melanoma sensitivity | C1 | README:78, 123 | 61.6% | E2 | 0.6162 (61.62%) | **MATCH** |
| 25 | ISIC 2019 overall accuracy | C1 | README:78 | 57.6% | E2 | 0.5756 (57.56%) | **MATCH** |
| 26 | ISIC 2019 image count | C1 | README:123 | 4,998 | E2 | 4,998 | **MATCH** |
| 27 | Anwarkh1 melanoma sensitivity | C1, C2 | README:49 | 73.3% | E5 | 0.7333 (73.33%) | **MATCH** |
| 28 | SigLIP melanoma sensitivity | C1, C2 | README:141 | 30.0% | E5 | 0.3 (30.0%) | **MATCH** |
| 29 | Anwarkh1 overall accuracy | C2 | TR:603 | 55.7% | E5 | **NOT IN JSON** | **RED - NO EVIDENCE** |
| 30 | Multi-image single mel sensitivity | C2 | TR:762 | 99.4% | E3 | 0.994 (99.4%) | **MATCH** |
| 31 | Multi-image majority vote mel sens | C2 | TR:763 | 99.4% | E3 | 0.994 (99.4%) | **MATCH** |
| 32 | Multi-image quality-weighted mel sens | C2 | TR:764 | 99.4% | E3 | 0.994 (99.4%) | **MATCH** |
| 33 | Multi-image single accuracy | C2 | TR:762 | 65.78% | E3 | 0.6578 (65.78%) | **MATCH** |
| 34 | Multi-image holdout count | C2 | TR:757 | 1,499 | E3 | 1,499 | **MATCH** |
| 35 | Hand-crafted features accuracy | C1, C2 | README:33 | 36.9% | NONE | **NOT IN ANY JSON** | **RED - NO EVIDENCE** |
| 36 | HAM10000 holdout count | C1, C2 | README:66, 122 | **1,503** | E1, E3 | E1: 2,004 (test split); E3: 1,499 | **MISMATCH** |
| 37 | Community ViT test count | C1, C2 | README:49, 121 | 210 | E4, E5 | 210 | **MATCH** |
| 38 | Anwarkh1 downloads | C1 | README:47 | 44K+ | N/A | External claim, not verifiable from evidence | UNVERIFIABLE |

### 1.9 The "98.2% on HAM10000 holdout (1,503 images)" Claim

| # | Claim | Source | Claimed Value | Evidence | MATCH? |
|---|-------|--------|---------------|----------|--------|
| 39 | 98.2% mel sens on HAM10000 holdout | C1, C2 | 98.2% on 1,503 images | E1: 98.22% on 2,004 images | **PARTIAL - Sensitivity matches, image count does not** |

**Audit Finding**: The 98.22% melanoma sensitivity is confirmed in the overfitting_check section of cross-validation-results.json, but on a test split of 2,004 images (with 225 melanomas), not 1,503. The number 1,503 does not appear in any evidence file. The multi-image validation uses 1,499. The README claims "1,503" consistently. This appears to be an error in the reported test set size.

---

## Section 2: RED FLAGS -- Claims With NO Supporting Evidence

These claims are presented as measured results but have no corresponding data in any evidence file.

| Priority | Claim | Severity | Rationale |
|----------|-------|----------|-----------|
| **CRITICAL** | 91.3% melanoma sensitivity (headline claim, README, AboutPage) | **Patient Safety** | The single most prominent claim in the project is entirely unsupported by measured data. No evidence file produces this number. This is the claim most likely to influence clinical decisions. |
| **CRITICAL** | 96.2% melanoma sensitivity (ADR-118, AboutPage, DrAgnesPanel trust banner) | **Patient Safety** | Displayed to users in the UI trust banner before they upload images. No evidence file produces this number. Actively misleading. |
| **CRITICAL** | 0.936 AUROC (ADR-118, AboutPage) | **Scientific Fraud** | No ROC analysis was performed. No AUROC was computed. This metric was fabricated. |
| **CRITICAL** | 97.3% all-cancer sensitivity (ADR-118) | **Patient Safety** | The only measured all-cancer sensitivity is 61.76% (ISIC 2019). Claiming 97.3% overstates actual performance by 35.5 percentage points. |
| **HIGH** | 73.1% melanoma specificity (ADR-118, AboutPage) | **Misleading** | No evidence file produces this number. Closest values range from 67% to 88% depending on dataset. |
| **HIGH** | 4,431 test images (ADR-118) | **Misleading** | No test set of this size exists in any evidence file. |
| **HIGH** | 67.1% to 81.4% overall accuracy (ADR-118) | **Misleading** | Neither number appears in any evidence file. |
| **HIGH** | 29,540 cross-dataset image count (README, AboutPage) | **Misleading** | No evidence file validates a combined test set of this size. The individual test sets sum to approximately 1,000 + 1,285 + 2,004 + 4,998 + 1,499 + 210 = 10,996 -- and these are NOT independent (multiple HAM10000 splits overlap). |
| **MEDIUM** | 55.7% Anwarkh1 overall accuracy | **Unverified** | Claimed in TECHNICAL-REPORT but no JSON evidence. |
| **MEDIUM** | 36.9% hand-crafted features accuracy | **Unverified** | Claimed in README and TECHNICAL-REPORT but no JSON evidence. |

---

## Section 3: MISMATCHES -- Numbers That Do Not Match Evidence

| # | Claim | Claimed | Measured | Delta | Severity |
|---|-------|---------|----------|-------|----------|
| M1 | HAM10000 holdout size | 1,503 images | 2,004 images (E1) or 1,499 (E3) | Off by 500 or 4 | **HIGH** -- Misrepresents test set size |
| M2 | Train/test gap | -0.7% | -0.06% (mel sensitivity gap in E1) or -0.69% (overall accuracy gap in E1) | Ambiguous | **MEDIUM** -- The -0.7% appears to refer to the overall accuracy gap (-0.0069 = -0.69%), not the melanoma sensitivity gap (-0.0006 = -0.06%). The README text says "Train/test gap: -0.7%" in the context of melanoma sensitivity claims, which is misleading. |
| M3 | Nagabu sensitivity rounded | 98.7% | 98.68% | 0.02pp | **LOW** -- Standard rounding, but always rounded up |
| M4 | DermaSensor specificity comparison | "Our specificity advantage means 2.2x fewer unnecessary biopsies" (AboutPage line 236) | Not computed from any evidence | N/A | **MEDIUM** -- Marketing claim without supporting calculation |

---

## Section 4: INCONSISTENCIES -- Same Metric Claimed Differently

| Metric | README | TECHNICAL-REPORT | ADR-118 | AboutPage | DrAgnesPanel | Evidence |
|--------|--------|-----------------|---------|-----------|-------------|----------|
| **Headline melanoma sensitivity** | 91.3% | 98.2% (HAM10000) | 96.2% | 91.3% and 96.2% | 96.2% | 98.22% (HAM10000), 61.62% (ISIC 2019), 99.4% (multi-image HAM10000) |
| **Test set size** | 29,540 | 3,788 (Section 4.4) | 4,431 | 29,540 | 29,540 | See individual files; no set of 29,540 or 4,431 or 3,788 exists |
| **Ensemble layer weights** | ViT 50% (L1) | ViT 50% (dual HF) | ViT 70% | ViT 50% (L1) | 50% weight | Contradictory across documents |
| **DermaSensor melanoma sensitivity** | 95.5% | 90.2% (pivotal) / 95.5% (ASSESS III) | 90.2-95.5% | 90.2-95.5% | 90.2-95.5% | External claim, documented correctly in most places |

**Critical Inconsistency**: Three different headline melanoma sensitivity numbers are used across the project:
- **91.3%** in README and AboutPage
- **96.2%** in ADR-118, AboutPage secondary stats, and DrAgnesPanel trust banner
- **98.2%** in TECHNICAL-REPORT

None of these three numbers (91.3%, 96.2%) has a supporting evidence file. The only one with evidence is 98.2% on HAM10000, but this is a same-dataset metric that has been shown to drop to 61.6% on external data.

**Critical Inconsistency in ADR-118**: ADR-118 simultaneously claims:
- "Current" melanoma sensitivity: 96.2% (Section: Context table)
- "Current" independent external validation: 91.3% (Section: Success Criteria)

These cannot both be the "current" melanoma sensitivity. The document does not explain which test set produces which number.

---

## Section 5: MISSING CONTEXT -- Numbers Without Required Statistical Context

| Missing Element | Affected Claims | FDA Requirement | Status |
|-----------------|----------------|-----------------|--------|
| **Confidence intervals** | ALL sensitivity/specificity claims | 95% CI required for all primary endpoints per FDA guidance on AI/ML devices | **ABSENT** -- No confidence interval appears in any evidence file or documentation |
| **Sample size justification** | All test sets | Statistical power analysis required | **ABSENT** -- No power analysis or sample size justification |
| **Pre-specified analysis plan** | All validation results | Required to prevent post-hoc selection bias | **ABSENT** -- No evidence of pre-registration or pre-specified endpoints |
| **Multiple testing correction** | Multiple test sets reported | Required when reporting across multiple datasets | **ABSENT** -- No Bonferroni or other correction applied |
| **Subgroup analysis** | Fitzpatrick skin type | Required for dermatology AI devices | **ABSENT** -- No Fitzpatrick-stratified results exist |
| **Prevalence-adjusted PPV/NPV** | All recommendation thresholds | Required for clinical interpretation | **ABSENT** -- PPV/NPV not computed at actual melanoma prevalence |
| **Calibration analysis** | All probability outputs | Required for probability-based recommendations | **ABSENT** -- No ECE, no reliability diagrams, no Hosmer-Lemeshow |
| **Inter-rater ground truth** | HAM10000 label quality | Required to characterize reference standard | **ABSENT** -- No discussion of label disagreement rates |

**Specific CI gaps for key claims (if they had evidence)**:

For the 98.22% melanoma sensitivity on 225 melanomas (E1):
- Wilson 95% CI: approximately [95.5%, 99.4%]
- This CI is never reported

For the 61.62% melanoma sensitivity on 714 melanomas (E2):
- Wilson 95% CI: approximately [58.0%, 65.1%]
- This CI is never reported

For the 99.4% melanoma sensitivity on 166 melanomas (E3):
- Wilson 95% CI: approximately [96.7%, 99.9%]
- This CI is never reported

---

## Section 6: REGULATORY CONCERNS

### 6.1 Data Leakage / Test Set Contamination (CRITICAL)

The cross-validation-results.json (E1) contains three test strategies, all on HAM10000 variants:
- **Nagabu/HAM10000**: Described as an "external binary melanoma dataset" but is an independent HuggingFace upload of the SAME HAM10000 data. This is NOT external validation. It is the same images from the same institution, just hosted on a different platform.
- **marmal88/skin_cancer**: Another HuggingFace upload of HAM10000. The 100% melanoma sensitivity on this split is likely because many of these exact images were in the training set.
- **Overfitting check**: Uses an 85/15 split of marmal88, but the model was already trained on HAM10000 data.

**Audit Finding**: All three "cross-validation" test sets are HAM10000 variants from the Medical University of Vienna. Testing on three copies of the same underlying dataset and calling it "cross-dataset validation" is misleading. The only genuinely external validation is ISIC 2019 (E2), which shows 61.62% melanoma sensitivity -- a 36.6 percentage point drop from HAM10000 performance.

### 6.2 The "Multi-Dataset Training" Claim Has No Evidence (CRITICAL)

The README describes a "Stage 5: Multi-Dataset Training" that supposedly combined HAM10000 and ISIC 2019 data to achieve 91.3% melanoma sensitivity. However:
- No evidence file contains results from a model trained on combined data
- The cross-validation-results.json (E1) tests the same `dragnes-classifier/best` model on HAM10000 variants only
- The ISIC 2019 validation (E2) tests the same model on ISIC 2019 and shows 61.6% melanoma sensitivity
- There is no JSON file showing results from a retrained multi-dataset model

The "multi-dataset training" described in the README does not appear to have been performed. The 91.3% result does not exist.

### 6.3 Selection Bias in Reported Results

The project reports the highest melanoma sensitivity achieved on each test set:
- 98.2% on HAM10000 (same-distribution)
- 98.7% on Nagabu (same data, different upload)
- 100% on marmal88 (same data, different split)
- 99.4% on multi-image (same data with augmentation)

But the ISIC 2019 result (61.6%) -- the only genuine external validation -- is presented as a problem that was "fixed" by multi-dataset training. The documentation claims the fix produced 91.3%, but no evidence supports this. The current model's actual external validation performance is 61.6%.

### 6.4 Predicate Device Comparison Is Misleading (HIGH)

The ADR-118 and AboutPage present a comparison table claiming Dr. Agnes EXCEEDS DermaSensor performance:
- Dr. Agnes: 96.2% melanoma sensitivity (NO EVIDENCE)
- DermaSensor: 90.2-95.5% (from FDA pivotal trial)

The actual measured external performance of Dr. Agnes (61.6% on ISIC 2019) is dramatically WORSE than DermaSensor's worst reported number (90.2%). Presenting an unsupported 96.2% claim alongside FDA-validated DermaSensor numbers is misleading and could constitute a violation of FDA advertising regulations for devices claiming substantial equivalence.

### 6.5 User-Facing Trust Banner Displays Unverified Claims (CRITICAL)

The `DrAgnesPanel.svelte` trust banner (line 1182) displays:
> "Validated on 29,540 images. 96.2% melanoma detection."

This is shown to every user before they upload an image. Neither number is supported by evidence. A patient or clinician seeing "96.2% melanoma detection" may make clinical decisions based on this false claim.

### 6.6 Insufficient Diversity Testing

- HAM10000 is approximately 95% Fitzpatrick I-III skin types
- No Fitzpatrick-stratified performance data exists
- No testing on dark skin tones has been performed
- The project acknowledges this but still claims headline performance numbers without qualification

### 6.7 No Prospective Validation

All testing is retrospective on curated datasets. No real-world clinical testing has been performed. This is acknowledged in the documentation but the headline claims do not reflect this limitation.

### 6.8 Ensemble Has Not Been Validated As a System

The TECHNICAL-REPORT explicitly states (Section 7, item 1): "No end-to-end ensemble accuracy measurement." The 4-layer ensemble that constitutes the actual deployed system has never been measured as a complete system. All reported metrics are for the ViT model alone. The weights (50/30/20) are "engineering estimates, not empirically optimized values."

---

## Section 7: Summary of Findings

### By Severity

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 6 | Headline claims (91.3%, 96.2%, 0.936 AUROC, 97.3%) have no evidence; trust banner displays false claims; data leakage in "cross-validation" |
| **HIGH** | 5 | Test set size mismatches (29,540 / 4,431 / 1,503); specificity claims without evidence; misleading DermaSensor comparison |
| **MEDIUM** | 4 | Train/test gap ambiguity; unverified accuracy claims; marketing claims without calculation |
| **LOW** | 1 | Rounding of Nagabu sensitivity (98.68% to 98.7%) |

### What IS Verified and Trustworthy

The following claims ARE supported by evidence:

1. **98.22% melanoma sensitivity on HAM10000 overfitting check** (2,004 images, E1) -- but this is same-distribution, not external
2. **98.68% melanoma sensitivity on Nagabu/HAM10000** (1,000 images, E1) -- but Nagabu IS HAM10000
3. **100% melanoma sensitivity on marmal88 test split** (1,285 images / 144 melanomas, E1) -- but marmal88 IS HAM10000
4. **61.62% melanoma sensitivity on ISIC 2019** (4,998 images, E2) -- the only genuine external validation
5. **99.4% melanoma sensitivity on multi-image HAM10000** (1,499 images, E3) -- same-distribution with augmentation
6. **73.33% melanoma sensitivity for Anwarkh1 model** (210 images, E5)
7. **30.0% melanoma sensitivity for SigLIP model** (210 images, E5)
8. **57.56% overall accuracy on ISIC 2019** (4,998 images, E2)

### The Actual State of Dr. Agnes

Based solely on evidence files:

| Metric | Same-Distribution (HAM10000) | External (ISIC 2019) |
|--------|------------------------------|----------------------|
| Melanoma sensitivity | 98.2-100% | **61.6%** |
| Overall accuracy | 65.8-78.1% | **57.6%** |
| All-cancer sensitivity | Not measured | **61.8%** |
| AUROC | Not measured | Not measured |

The actual external validation performance (61.6% melanoma sensitivity) means approximately **2 in 5 melanomas are missed on data from different institutions**. This is dramatically worse than any number presented in the README, AboutPage, ADR-118, or trust banner.

---

## Section 8: Recommendations

1. **IMMEDIATELY** remove the 91.3%, 96.2%, 0.936 AUROC, and 97.3% claims from all documentation and UI components. Replace with actually measured values with confidence intervals.

2. **IMMEDIATELY** update the DrAgnesPanel trust banner to reflect actual external validation performance (61.6% on ISIC 2019) or remove numerical claims entirely.

3. **IMMEDIATELY** retitle the "cross-validation" results to accurately describe them as "same-distribution validation on HAM10000 variants" rather than "cross-dataset validation."

4. If a multi-dataset trained model exists, produce the evidence file. If it does not exist, remove all Stage 5 claims from the README.

5. Compute and report 95% Wilson confidence intervals for all sensitivity/specificity claims.

6. Perform genuine external validation on datasets with no overlap with training data (Derm7pt, PH2, ISIC 2024).

7. Perform Fitzpatrick-stratified analysis before making any claims about clinical utility.

8. Compute AUROC from probability scores before claiming any AUROC value.

9. Validate the full ensemble system end-to-end before making performance claims about the deployed system.

---

## Auditor's Conclusion

This audit found that the most prominent performance claims in the Dr. Agnes project are not supported by the project's own evidence files. The headline claim of "91.3% melanoma sensitivity, cross-dataset validated on 29,540 diverse images" has no corresponding measured result. The "96.2%" claim displayed in the user-facing trust banner has no corresponding measured result. The "0.936 AUROC" was never computed. The "97.3% all-cancer sensitivity" contradicts the actual measured value of 61.8% by 35.5 percentage points.

The only genuine external validation shows 61.6% melanoma sensitivity -- meaning the system misses approximately 2 in 5 melanomas on data from different institutions than the training data. All other reported sensitivities (98-100%) are on variants of the same HAM10000 dataset used for training.

For a cancer screening device, presenting unsupported performance claims in user-facing interfaces is a patient safety concern. A clinician or patient relying on the stated "96.2% melanoma detection" rate would have a materially different risk assessment than one informed of the actual 61.6% external validation rate.

**This device, as documented, would not pass FDA pre-submission review.** The discrepancy between claims and evidence would trigger a Refuse to Accept decision and a request for a complete resubmission with verified performance data.

---

*Audit conducted: 2026-03-23*
*Auditor role: FDA CDRH reviewer (simulated)*
*Files examined: 9 claim files, 5 evidence files*
*Method: Automated text search cross-referencing every numerical claim against JSON evidence*
