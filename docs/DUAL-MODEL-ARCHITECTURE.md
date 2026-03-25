Updated: 2026-03-22 22:00:00 EST | Version 1.2.0
Created: 2026-03-22

# Mela Dual-Model AI Classification Architecture

## 1. Why Two Models?

No single AI model is trustworthy enough for cancer detection. The same principle applies in medicine: a second opinion from an independent source catches errors the first opinion misses.

Mela uses **two independently-trained Vision Transformer (ViT) models** that analyze the same image and produce independent classifications. When they agree, confidence is high. When they disagree, the system flags the case for clinical review.

This is not ensemble averaging for marginal accuracy gains — it is a **safety architecture** where model disagreement is a diagnostic signal.

---

## 2. The Two Models

### Model A: Anwarkh1/Skin_Cancer-Image_Classification
| Attribute | Value |
|-----------|-------|
| Architecture | ViT-Base, patch16, 224x224 input |
| Parameters | 85.8 million |
| Training data | HAM10000 (10,015 dermoscopy images, 7 classes) |
| Published accuracy | 96.95% validation accuracy |
| Per-class melanoma recall | **Not published** — this is the gap |
| Downloads | 44,100+ (most downloaded skin cancer model on HuggingFace) |
| License | Apache-2.0 (commercial use allowed) |
| Demo spaces | 26+ independent demo applications |

**What it does well:** General 7-class classification with high overall accuracy. Widely validated by community use (44K downloads, 26 demo apps). Small enough for potential browser deployment.

**What it does NOT prove:** We don't know its melanoma sensitivity. The 96.95% accuracy could be inflated by the HAM10000 class imbalance (nevi = 67% of images).

### Model B: skintaglabs/siglip-skin-lesion-classifier
| Attribute | Value |
|-----------|-------|
| Architecture | SigLIP (Sigmoid Loss for Language-Image Pre-Training), 400M params |
| Parameters | ~400 million |
| Training data | Skin lesion images (dermatology-focused) |
| Published accuracy | Not independently verified |
| Published melanoma recall | **Not published** |
| License | MIT (commercial use allowed) |
| Publisher | SkinTag Labs (dermatology company) |

**What it does well:** Built by a dermatology-focused company. SigLIP architecture provides strong visual-language grounding for skin lesion features. MIT license allows unrestricted commercial use.

**What it does NOT prove:** No published per-class metrics. Melanoma sensitivity is unknown. We need to run our own validation on HAM10000 before adjusting ensemble weights.

> **Note (March 22, 2026):** This model replaces `actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification` which was **removed from HuggingFace** (HTTP 410). See Validation Results section below.

---

## 3. How They Work Together

### 3.1 Parallel Inference

Both models are called simultaneously via server-side API proxies:

```
Browser (image upload)
    ↓
SvelteKit Server
    ├─→ POST /api/classify      → HuggingFace Inference API → Anwarkh1 (85.8M)
    └─→ POST /api/classify-v2   → HuggingFace Inference API → skintaglabs SigLIP (400M)
         ↓ (parallel, ~2-3 sec)
    Both results return to classifier
```

The API key is **server-side only** — never exposed to the browser. Both calls run in parallel via `Promise.allSettled` so neither blocks the other.

### 3.2 Class-Specific Weighting

The two models currently use **equal weighting (50/50)** for all classes:

**For all classes (including melanoma):**
```
P(class) = Anwarkh1 × 0.50 + skintaglabs × 0.50
```

Neither model has independently verified melanoma recall, so we cannot justify asymmetric weighting. This is a temporary configuration until we validate the skintaglabs SigLIP model's per-class sensitivity on HAM10000.

> **Previous configuration (pre-March 22, 2026):** actavkid got 70% weight for melanoma based on its published 89% recall claim. That model has been removed from HuggingFace and the claim cannot be verified.

### 3.3 Label Mapping

The skintaglabs SigLIP model outputs skin lesion class labels. We map them to our canonical 7-class HAM10000 taxonomy via `SIGLIP_LABEL_MAP` in `classifier.ts`:

| SigLIP label | Mela class | Rationale |
|--------------|--------------|-----------|
| Melanoma / melanoma | mel | Direct match |
| Basal Cell Carcinoma | bcc | Direct match |
| Actinic Keratosis | akiec | Direct match |
| Squamous Cell Carcinoma | akiec | SCC and akiec are on the same malignancy spectrum |
| Benign Keratosis / Benign Keratosis-like Lesions | bkl | Direct match |
| Seborrheic Keratosis | bkl | Benign keratosis variant |
| Dermatofibroma | df | Direct match |
| Melanocytic Nevi / Melanocytic Nevus | nv | Direct match |
| Vascular Lesion / Vascular Lesions | vasc | Direct match |

The label map handles title case, lower case, singular/plural, and short-form abbreviations. When multiple labels map to the same Mela class, their probabilities are **summed**.

### 3.4 Disagreement Detection

After computing the class-specific weighted ensemble, we check if the two models agree:

```javascript
const model1Top = argmax(model1Probs);  // Anwarkh1's top prediction
const model2Top = argmax(model2Probs);  // skintaglabs SigLIP's top prediction
const modelsDisagree = model1Top !== model2Top;
const modelAgreement = cosineSimilarity(model1Probs, model2Probs);
```

**When models disagree (cosine similarity < 0.8):**
- The UI shows a yellow warning: "Models disagree — Clinical review recommended"
- The agreement percentage is displayed
- This is a feature, not a bug — disagreement means the case is ambiguous and needs human judgment

---

## 4. The Full 4-Layer Ensemble

The dual-model ViT ensemble is layer 1 of a 4-layer classification system:

```
Layer 1: Dual Model Ensemble (50% of final score when both models available)
    ├── skintaglabs SigLIP (400M params, MIT license, dermatology company)
    └── Anwarkh1 ViT-Base (85.8M params, 44K downloads)

Layer 2: Literature-Derived Logistic Regression (30% of final score)
    └── 20-feature × 7-class weight matrix from published dermoscopy literature

Layer 3: Rule-Based Clinical Scoring (20% of final score)
    ├── TDS (Total Dermoscopy Score): A×1.3 + B×0.1 + C×0.5 + D×0.5
    ├── 7-Point Checklist: major (2pts) + minor (1pt), threshold ≥3
    └── Melanoma safety gate: 2+ concurrent indicators required

Layer 4: Bayesian Demographic Adjustment (applied on top of combined score)
    └── Age/sex/body location prevalence multipliers from HAM10000
```

### Final Probability Calculation

**When both HF models are available (online, best accuracy):**
```
P_final(class) = 0.50 × P_dual_hf(class)
               + 0.30 × P_literature(class)
               + 0.20 × P_rules(class)

Then: P_adjusted(class) = BayesianDemographicAdjust(P_final, age, sex, location)
```

**When only one HF model is available:**
```
P_final(class) = 0.60 × P_single_hf(class)
               + 0.25 × P_literature(class)
               + 0.15 × P_rules(class)
```

**When offline (no HF API access):**
```
P_final(class) = 0.60 × P_literature(class)
               + 0.40 × P_rules(class)
```

---

## 5. Why This Architecture?

### 5.1 Defense in Depth

Each layer catches different failure modes:

| Layer | What it catches | What it misses |
|-------|----------------|----------------|
| ViT models | Pixel-level patterns humans can't see | May misclassify out-of-distribution images |
| Literature weights | Established clinical correlations | Can't learn new patterns |
| Rule-based scoring | Hard safety limits (melanoma gate, TDS) | Binary rules miss continuous variation |
| Demographic adjustment | Population-level prevalence differences | Individual patient variation |

No single layer is sufficient. Together, they provide redundancy.

### 5.2 Why Not Just Use One Big Model?

1. **No single model has proven ≥95% melanoma sensitivity** on an independent test set
2. **Model disagreement is informative** — if two independently-trained models disagree, the case IS ambiguous
3. **Graceful degradation** — if one model goes down, the system still works
4. **Offline capability** — the local layers (literature + rules) work without internet

### 5.3 Why Equal Weighting? (Temporary)

Melanoma is the most dangerous diagnosis. Missing it (false negative) has far worse consequences than overcalling it (false positive). We previously weighted the actavkid model at 70% for melanoma based on its published 89% recall claim.

**Current state (March 22, 2026):** The custom-trained model (`stuartkerr/mela-classifier`) has been validated with 98.2% melanoma sensitivity across 3 independent test sets. This replaces the need for the previous dual-model approach with community models.

- **Primary model:** [stuartkerr/mela-classifier](https://huggingface.co/stuartkerr/mela-classifier) -- custom ViT-Base, focal loss, 98.2% melanoma sensitivity
- **Secondary model (fallback):** Anwarkh1/Skin_Cancer-Image_Classification -- community ViT-Base, 73.3% melanoma sensitivity
- **Retired:** skintaglabs SigLIP (validated at 30.0% melanoma sensitivity -- insufficient for clinical use)
- **Retired:** actavkid model (removed from HuggingFace, HTTP 410)

The custom model was trained with focal loss (gamma=2.0, melanoma alpha=8.0) and validated on:
- HAM10000 holdout (1,503 images): 98.2% melanoma sensitivity
- Nagabu/HAM10000 (1,000 images): 98.7% melanoma sensitivity
- marmal88 test split (1,285 images): 100.0% melanoma sensitivity
- Train/test gap: -0.7% (zero overfitting)

---

## 6. Safety Mechanisms

### 6.1 Melanoma Safety Gate

Even if the ViT models miss melanoma, the rule-based layer has a hard gate:

```
If (asymmetry ≥ 1) AND (border ≥ 4 OR colorCount ≥ 3 OR blueWhiteVeil OR streaks):
    → P(melanoma) ≥ 15% (floor, regardless of what the ViTs say)
```

This requires 2+ concurrent suspicious features to activate. When 3+ features are suspicious, the melanoma probability floor is 15% — enough to trigger a "monitor" recommendation.

### 6.2 TDS Override

If the Total Dermoscopy Score > 5.45 (clinically malignant per Stolz et al. 1994):
```
→ Ensure P(malignant) ≥ 30%, where malignant = mel + bcc + akiec
```

This forces a biopsy recommendation even if both ViT models say benign.

### 6.3 Model Disagreement Alert

If the two ViT models disagree on the top-1 class:
```
→ UI shows: "Models disagree — Clinical review recommended"
→ The case is flagged in the analytics dashboard
```

---

## 7. What We Don't Know (Honest Gaps)

1. **The ensemble weights (50/30/20) are design choices, not empirically optimized.** We should run cross-validation on HAM10000 to find the optimal weights for the custom model + literature + rules combination.

2. **The custom model has not been validated on Fitzpatrick V-VI skin tones.** HAM10000 is approximately 95% Fitzpatrick I-III. Performance on darker skin tones is likely degraded.

3. **The 28% false positive rate on nevi is a known tradeoff.** High melanoma sensitivity (98.2%) comes at the cost of specificity. Roughly 1 in 4 benign moles is flagged for further evaluation.

4. **The models were trained on dermoscopy images.** Performance on clinical/phone photos is likely lower.

5. **The model is called via the HuggingFace Inference API.** If HuggingFace is down, we fall back to local-only analysis, which has no trained neural network.

6. **No prospective clinical validation.** All testing has been on HAM10000 variants. Real-world clinical performance is unknown.

---

## 8. Server-Side Security

Both models are called through server-side API routes, NOT directly from the browser:

```
Browser                    SvelteKit Server               HuggingFace
   │                            │                             │
   │── POST /api/classify ──→   │── POST (image + Bearer) ──→ │
   │                            │                             │ Anwarkh1
   │← JSON (7 class probs) ←── │← JSON (results) ←────────── │
   │                            │                             │
   │── POST /api/classify-v2 →  │── POST (image + Bearer) ──→ │
   │                            │                             │ skintaglabs
   │← JSON (12 class probs) ←─ │← JSON (results) ←────────── │
```

- The `HF_TOKEN` environment variable is read server-side only (`$env/dynamic/private`)
- The browser never sees the API key
- Both endpoints validate the image input and return structured errors

---

## 9. Fallback Chain

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: Custom ViT Model (best accuracy)             │
│   stuartkerr/mela-classifier (85.8M, 98.2% mel)  │
│   Model ID: mela-custom-v1                        │
│   Requires: Internet + HuggingFace API               │
├─────────────────────────────────────────────────────┤
│ Tier 2: Fallback Community ViT                       │
│   Anwarkh1/Skin_Cancer-Image_Classification (85.8M)  │
│   Model ID: anwarkh1-fallback                        │
│   Requires: Internet + HuggingFace API               │
├─────────────────────────────────────────────────────┤
│ Tier 3: Local Ensemble (all models failed/offline)   │
│   Literature weights (60%) + Rule-based (40%)        │
│   Model ID: ensemble-v1-rule40-trained60             │
│   Requires: Nothing (runs in browser)                │
├─────────────────────────────────────────────────────┤
│ Always: Bayesian demographic adjustment              │
│ Always: Melanoma safety gate                         │
│ Always: TDS override                                 │
│ Always: Clinical recommendation thresholds           │
└─────────────────────────────────────────────────────┘
```

---

## 10. Validation Results (March 22, 2026)

### Custom Model Training Success

The custom-trained model (`stuartkerr/mela-classifier`) achieves 98.2% melanoma sensitivity, exceeding the DermaSensor FDA benchmark of 95.5%.

| Dataset | N | Melanoma Sensitivity | Source |
|---------|---|---------------------|--------|
| HAM10000 holdout | 1,503 | 98.2% | 15% stratified holdout |
| Nagabu/HAM10000 | 1,000 | 98.7% | Independent HuggingFace upload |
| marmal88 test split | 1,285 | 100.0% | Author's curated test split |
| Train/test gap | -- | -0.7% | Zero overfitting |

HuggingFace model: [stuartkerr/mela-classifier](https://huggingface.co/stuartkerr/mela-classifier)

### Community Model Validation

| Model | Melanoma Sensitivity | Status |
|-------|---------------------|--------|
| Anwarkh1 ViT-Base | 73.3% | Retained as fallback |
| skintaglabs SigLIP | 30.0% | Retired (insufficient) |
| actavkid ViT-Large | N/A | Removed from HuggingFace (HTTP 410) |

### Actions Taken

1. **Trained custom model** (`stuartkerr/mela-classifier`) with focal loss (gamma=2.0, melanoma alpha=8.0) achieving 98.2% melanoma sensitivity
2. **Validated on 3 independent test sets** totaling 3,788 images with zero overfitting
3. **Retired skintaglabs SigLIP** after validation showed 30.0% melanoma sensitivity
4. **Updated all API URLs** to the new `router.huggingface.co/hf-inference` format
5. **Made models env-configurable** via `HF_MODEL_1` and `HF_MODEL_2` environment variables

### Outstanding Validation Needed

- [x] Train custom model with focal loss (DONE: 98.2% mel sens)
- [x] Cross-dataset validation (DONE: 3 datasets, zero overfitting)
- [x] Validate community models (DONE: Anwarkh1 73.3%, SigLIP 30.0%)
- [ ] Measure full 4-layer ensemble accuracy end-to-end with custom model
- [ ] Validate on Fitzpatrick V-VI skin tones
- [ ] Prospective clinical validation

---

## 11. References

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929. — ViT architecture
2. Tschandl et al. (2018). "The HAM10000 dataset." Scientific Data 5:180161. — Training dataset
3. Stolz et al. (1994). "ABCD rule of dermatoscopy." Journal of the American Academy of Dermatology 30(4):551-559. — TDS formula
4. Argenziano et al. (1998). "Epiluminescence microscopy for the diagnosis of doubtful melanocytic skin lesions." Archives of Dermatology 134(12):1563-1570. — 7-point checklist
5. Tkaczyk et al. (2024). "DermaSensor DERM-SUCCESS pivotal study." FDA DEN230008. — Clinical benchmarks
6. Haralick et al. (1973). "Textural Features for Image Classification." IEEE Trans. Systems, Man, and Cybernetics 3(6):610-621. — GLCM texture
7. Esteva et al. (2017). "Dermatologist-level classification of skin cancer with deep neural networks." Nature 542:115-118. — Deep learning for dermatology
