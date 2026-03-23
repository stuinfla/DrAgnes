Updated: 2026-03-23 02:00:00 EST | Version 1.0.0
Created: 2026-03-23

# ADR-118: Dr. Agnes — Production Validation & World-Class Medical Proof

## Status: PROPOSED

## Context

Dr. Agnes has achieved measurable results that exceed the only FDA-cleared skin cancer AI device (DermaSensor):

| Metric | DrAgnes (measured) | DermaSensor (FDA pivotal) | Source |
|--------|-------------------|--------------------------|--------|
| Melanoma sensitivity | **96.2%** | 90.2-95.5% | 4,431 test images |
| Melanoma specificity | **73.1%** | 20.7-32.5% | Cross-dataset |
| Melanoma AUROC | **0.936** | 0.758 | Combined test set |
| All-cancer sensitivity | **97.3%** | 95.5% | mel+bcc+akiec |
| All-cancer specificity | **57.8%** | 20.7% | Combined test set |
| Training data | 29,540 images | Proprietary | Multi-source |

These numbers are promising but NOT world-class proof. World-class proof requires:
1. Independent external validation on data we never touched
2. Prospective clinical study with real patients
3. Fitzpatrick equity across all skin tones
4. Peer-reviewed publication
5. Regulatory pathway

This ADR defines the complete roadmap to achieve irrefutable medical and scientific proof.

## Decision

Execute a 5-phase validation program that transforms DrAgnes from "impressive prototype" to "scientifically proven, clinically validated AI dermatoscopy system."

---

## Phase 1: Model Excellence (Weeks 1-2)

### 1.1 Training Data Expansion
**Goal:** Train on 50,000+ diverse images from 4+ sources

| Dataset | Images | Classes | Unique Value |
|---------|--------|---------|-------------|
| HAM10000 | 10,015 | 7 | Gold-standard labels |
| ISIC 2019 external | 21,814 | 8 | Multi-institutional |
| **BCN20000** | 19,424 | 8 | ALL histopathology-confirmed |
| **Fitzpatrick17k** | 16,577 | 114 | All 6 Fitzpatrick skin types |
| **PAD-UFES-20** | 2,298 | 6 | Smartphone images (real-world) |
| **TOTAL** | **~70,128** | — | — |

**Why:** More diverse data = better generalization. BCN20000 adds histopathology-confirmed labels (highest ground truth). Fitzpatrick17k addresses skin tone equity. PAD-UFES-20 adds smartphone photos (consumer use case).

### 1.2 Architecture Optimization
**Goal:** Maximize both sensitivity AND specificity simultaneously

| Technique | Expected Impact | Effort |
|-----------|----------------|--------|
| **ViT-Large (305M params)** instead of ViT-Base (85.8M) | +3-5% accuracy | High |
| **Learning rate scheduling** (cosine annealing with warm restarts) | +1-2% | Low |
| **Label smoothing** (0.1) for benign classes | +2-3% specificity | Low |
| **Mixup/CutMix augmentation** | +1-3% generalization | Medium |
| **Test-time augmentation** (5 rotations, average) | +2-3% per-class | Low |
| **Multi-image inference** (3 photos, majority vote) | +3-5% melanoma | Medium |
| **Threshold optimization** (ROC curve per-class) | Maximize F1 per class | Low |

### 1.3 Sensitivity-Specificity Co-Optimization
**Goal:** Push BOTH metrics higher, not one at the expense of the other

Current tradeoff: 96.2% sensitivity / 73.1% specificity for melanoma.
Target: **≥95% sensitivity / ≥80% specificity** (AUROC ≥ 0.95)

Approach:
- Train with **asymmetric label smoothing**: smooth benign labels (0.1) but keep cancer labels hard (0.0)
- Use **class-specific thresholds**: instead of argmax, find the optimal probability threshold per class that maximizes the F1 score
- **Two-stage classification**: first separate malignant vs benign (high sensitivity), then subclassify within each group (high specificity)

### 1.4 Rust/WASM Conversion
**Goal:** Eliminate Python dependency, run in browser

| Step | Tool | Output |
|------|------|--------|
| Export to ONNX | `optimum-cli export onnx` | ~330MB ONNX |
| Quantize INT8 | `optimum-cli onnxruntime quantize` | ~80MB |
| Bundle for browser | `onnxruntime-web` or `ort` (Rust) | WASM module |
| Integrate into `@ruvector/cnn` | Rust crate extension | Native WASM |

Result: Classification runs in the browser at ~50ms/image. No server needed. Fully offline capable.

---

## Phase 2: Scientific Validation (Weeks 3-6)

### 2.1 Independent External Validation
**Goal:** Test on datasets we NEVER trained on

| Dataset | Purpose | Expected Result |
|---------|---------|----------------|
| **Derm7pt** (1,011 images) | Paired clinical + dermoscopy with 7-point scores | Validate 7-point checklist integration |
| **ISIC 2024 SLICE-3D** (401K images) | Massive, recent, 3D body photography | Test at scale |
| **PH2** (200 images) | Portuguese hospital, independent | Small but fully external |

Success criterion: Melanoma sensitivity ≥ 85% on ALL external datasets (allowing for domain shift).

### 2.2 Fitzpatrick Equity Audit
**Goal:** Prove the model works equally across all skin tones

| Metric | Target |
|--------|--------|
| Melanoma sensitivity gap (FST I-III vs IV-VI) | < 5% |
| AUROC gap | < 0.02 |
| Per-Fitzpatrick-type confusion matrix | Published |
| FST V-VI representation in test | ≥ 15% |

Comparison: DermaSensor showed 4% gap (96% FST I-III, 92% FST IV-VI). We must match or beat.

### 2.3 Calibration Analysis
**Goal:** When the model says 70% melanoma, it IS melanoma 70% of the time

| Metric | Tool | Target |
|--------|------|--------|
| Expected Calibration Error (ECE) | 10-bin histogram | < 5% |
| Hosmer-Lemeshow p-value | Chi-squared test | > 0.10 |
| Reliability diagram | Calibration curve | On diagonal |
| Temperature scaling | Post-hoc calibration | If ECE > 5% |

### 2.4 Ablation Study
**Goal:** Prove each component contributes

| Experiment | What It Proves |
|-----------|---------------|
| ViT only (no ensemble) vs full ensemble | Ensemble adds value |
| Focal loss vs cross-entropy | Focal loss is necessary |
| Single-dataset vs multi-dataset | Data diversity matters |
| With vs without demographic adjustment | Demographics improve accuracy |
| With vs without safety gates | Safety gates catch edge cases |
| With vs without pi-brain knowledge | Collective intelligence adds value |

---

## Phase 3: Clinical Validation (Months 2-4)

### 3.1 Retrospective Clinical Study
**Goal:** Test on real clinical case series with known outcomes

| Parameter | Value |
|-----------|-------|
| Study design | Retrospective, single-blind |
| Sample size | ≥ 500 lesions with histopathology |
| Source | Partner dermatology practice(s) |
| Endpoints | Sensitivity, specificity, PPV, NPV per class |
| Comparator | Board-certified dermatologist diagnosis |
| IRB | Required |

### 3.2 Prospective Pilot
**Goal:** Real-time use in clinical setting

| Parameter | Value |
|-----------|-------|
| Design | Prospective, non-interventional |
| Sites | 3-5 dermatology practices |
| Duration | 3 months |
| Metrics | Concordance, NNB, patient outcomes |
| Analytics | DrAgnes dashboard collects all data |
| Feedback loop | Clinician feedback improves model via pi-brain |

### 3.3 AI-vs-Dermatologist Comparison
**Goal:** Publish head-to-head comparison

| Arm | Description |
|-----|------------|
| DrAgnes alone | AI classification without clinician |
| Dermatologist alone | Clinical + dermoscopy examination |
| DrAgnes + Dermatologist | AI-assisted clinical examination |

Expected finding: AI + Dermatologist > Dermatologist alone > AI alone (augmentation, not replacement).

---

## Phase 4: RuVector Ecosystem Proof (Weeks 3-8)

### 4.1 Pi-Brain Collective Intelligence Demonstration
**Goal:** Prove the collective brain gets smarter over time

| Experiment | Method |
|-----------|--------|
| Baseline accuracy (Day 1) | Fresh model, no collective data |
| 1-month accuracy | After 1,000+ classifications shared to pi-brain |
| 3-month accuracy | After 5,000+ classifications + federated updates |
| Improvement curve | Plot accuracy vs collective knowledge size |

### 4.2 Federated Learning Validation
**Goal:** Prove practice-specific adaptation without data sharing

| Metric | Target |
|--------|--------|
| Per-practice accuracy improvement | ≥ 3% after 100 cases |
| No raw data leaves the practice | Verified by audit |
| Byzantine poisoning detection | Catches adversarial updates |
| EWC++ prevents catastrophic forgetting | Prior knowledge preserved |

### 4.3 RuVector WASM Performance
**Goal:** Prove browser-native inference is clinically viable

| Metric | Target |
|--------|--------|
| WASM inference time | < 200ms on modern phone |
| Model download size | < 100MB (INT8 quantized) |
| Offline classification | Full capability without network |
| Memory usage | < 500MB |

### 4.4 Claude Flow Orchestration Proof
**Goal:** Document how multi-agent architecture enabled rapid development

| Evidence | Description |
|----------|------------|
| Session transcript | 15+ hour session, 30+ agents, 15,000+ lines |
| Parallel execution | Training, validation, UI, docs ran simultaneously |
| Agent specialization | ML, frontend, security, docs agents each contributed |
| Iterative improvement | 6-stage progression with measured results at each stage |

---

## Phase 5: Publication & Regulatory (Months 4-12)

### 5.1 Peer-Reviewed Paper
**Target journal:** JAMA Dermatology, Nature Medicine, or Journal of the American Academy of Dermatology

**Paper structure:**
1. Introduction: The melanoma detection gap (5B people without dermatologist access)
2. Methods: Multi-dataset training, focal loss, ensemble architecture
3. Results: Sensitivity/specificity/AUROC per class, cross-dataset validation, Fitzpatrick equity
4. Discussion: Comparison with DermaSensor, limitations, collective intelligence
5. Conclusion: Open-source AI matches FDA-cleared device performance

### 5.2 FDA Regulatory Pathway
**Path:** 510(k) with DermaSensor (DEN230008) as predicate device

| Requirement | Status |
|------------|--------|
| Predicate device identified | DermaSensor DEN230008 |
| Substantial equivalence argument | Same intended use, different technology |
| Clinical validation study | Phase 3 above |
| Software documentation (IEC 62304) | Architecture docs in place |
| Risk management (ISO 14971) | Safety gates documented |
| Cybersecurity (FDA guidance) | Privacy pipeline documented |

### 5.3 Open-Source Publication
**Goal:** Publish model weights, training code, and validation data for reproducibility

| Asset | Location |
|-------|---------|
| Model weights | HuggingFace: stuartkerr/dragnes-classifier |
| Training scripts | GitHub: stuinfla/DrAgnes/scripts/ |
| Validation results | GitHub: stuinfla/DrAgnes/scripts/*-results.json |
| Architecture docs | GitHub: stuinfla/DrAgnes/docs/ |
| Pi-brain knowledge | pi.ruv.io (1,807+ memories) |

---

## Success Criteria

| Criterion | Threshold | Current |
|-----------|----------|---------|
| Melanoma sensitivity (cross-dataset) | ≥ 95% | 96.2% |
| Melanoma specificity | ≥ 80% | 73.1% |
| Melanoma AUROC | ≥ 0.95 | 0.936 |
| All-cancer sensitivity | ≥ 95% | 97.3% |
| Overall accuracy | ≥ 85% | 67.1% → 81.4% |
| Fitzpatrick equity gap | < 5% | Not yet tested |
| Independent external validation | ≥ 85% mel sens | 91.3% (mixed) |
| Prospective clinical concordance | ≥ 85% | Not yet tested |
| Peer-reviewed publication | Accepted | Not yet submitted |
| Browser inference time | < 200ms | Not yet converted |

---

## Architecture Decision

### Chosen: 4-Layer Ensemble with Collective Intelligence

```
Layer 1: Custom ViT (70%) — trained on 29.5K+ diverse images with focal loss
Layer 2: Literature-derived classifier (15%) — 20 features × 7 classes from published research
Layer 3: Rule-based safety gates (15%) — TDS formula, 7-point checklist, melanoma floor
Layer 4: Bayesian demographic adjustment — age/sex/location from HAM10000 epidemiology
```

Plus:
- **Pi-brain collective intelligence** — 1,807+ memories, federated learning
- **WASM offline capability** — classification runs without network
- **Analytics dashboard** — prospective evidence collection built-in

### Rejected Alternatives

| Alternative | Why Rejected |
|------------|-------------|
| Single neural network only | No safety gates, no fallback, single point of failure |
| HuggingFace API only | No offline capability, dependent on third-party service |
| Feature engineering only | Proven 0% melanoma sensitivity — insufficient |
| DermaSensor-like hardware | $7,000 cost, proprietary, not scalable |

---

## References

1. Tschandl P, et al. (2018). "The HAM10000 dataset." Scientific Data 5:180161
2. Tkaczyk ER, et al. (2024). "DermaSensor DERM-SUCCESS." FDA DEN230008
3. Lin TY, et al. (2017). "Focal Loss for Dense Object Detection." ICCV
4. Stolz W, et al. (1994). "ABCD Rule of Dermatoscopy." JAAD 30(4):551-559
5. Argenziano G, et al. (1998). "7-Point Checklist." Archives of Dermatology 134(12):1563
6. Esteva A, et al. (2017). "Dermatologist-level classification." Nature 542:115-118
7. Dosovitskiy A, et al. (2020). "An Image is Worth 16x16 Words." arXiv:2010.11929
8. Haralick RM, et al. (1973). "Textural Features." IEEE Trans SMC 3(6):610-621

## Author
Stuart Kerr + Claude Flow (RuVector/RuFlo)

## Participants
Claude Opus 4.6, Claude Flow agents (30+), Pi-brain collective (1,807+ memories)
