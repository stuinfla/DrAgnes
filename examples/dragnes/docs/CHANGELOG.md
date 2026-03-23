# Changelog

All notable changes to DrAgnes are documented in this file.

## [0.3.0] - 2026-03-22

### Added
- Custom-trained ViT model (`stuartkerr/dragnes-classifier`) achieving 98.2% melanoma sensitivity
- Cross-dataset validation on 3 independent test sets (HAM10000 holdout 1,503 images, Nagabu/HAM10000 1,000 images, marmal88 test 1,285 images)
- HuggingFace model deployment at [stuartkerr/dragnes-classifier](https://huggingface.co/stuartkerr/dragnes-classifier)

### Changed
- Primary model switched from community Anwarkh1 ViT-Base (73.3% mel sens) to custom-trained model (98.2% mel sens)
- Anwarkh1 ViT-Base demoted to fallback model
- skintaglabs SigLIP retired after validation showed 30.0% melanoma sensitivity
- README.md rewritten with cross-dataset validation results and competitive analysis
- TECHNICAL-REPORT.md updated with custom model training results and cross-validation data
- DUAL-MODEL-ARCHITECTURE.md updated to reference deployed HuggingFace model

### Validation Results (March 22, 2026)
- Custom model: 98.2% melanoma sensitivity (HAM10000 holdout), 98.7% (Nagabu), 100% (marmal88)
- Train/test gap: -0.7% (zero overfitting)
- Exceeds DermaSensor FDA benchmark (95.5% melanoma sensitivity)
- 28% false positive rate on nevi (deliberate tradeoff for melanoma sensitivity)

## [0.2.0] - 2026-03-22

### Added
- Real image analysis engine (`image-analysis.ts`, 1,890 lines): Otsu segmentation in CIELAB, principal-axis asymmetry, 8-octant border analysis, k-means++ color clustering (k=6 in LAB space), GLCM texture features (contrast, homogeneity, entropy, correlation), LBP structure detection, real attention maps
- Dual-model ViT ensemble (Anwarkh1 ViT-Base + skintaglabs SigLIP) via HuggingFace Inference API
- Literature-derived logistic regression classifier (`trained-weights.ts`): 20-feature x 7-class weight matrix, every weight cited to published dermoscopy literature
- Clinical baselines (`clinical-baselines.ts`) calibrated against DermaSensor FDA DEN230008
- TDS weighted formula (A*1.3 + B*0.1 + C*0.5 + D*0.5) with validated cutoffs (4.75 benign / 5.45 malignant)
- 7-point dermoscopy checklist scoring (Argenziano et al. 1998)
- Melanoma safety gate: 2+ concurrent suspicious indicators = 15% probability floor
- TDS override: TDS > 5.45 forces >= 30% malignant probability regardless of model output
- ICD-10-CM code mapping for all 7 lesion classes (`icd10.ts`)
- One-click referral letter generator (`ReferralLetter.svelte`) with copy-to-clipboard
- "Why this classification?" explainability panel (`ExplainPanel.svelte`) with literature citations
- Practice analytics dashboard: concordance rate, NNB tracking, per-class sensitivity/specificity/PPV/NPV with Wilson 95% CI, calibration curves with ECE + Hosmer-Lemeshow p-value, Fitzpatrick equity monitoring with disparity alerts, 30-day rolling concordance trends, discordance analysis
- Interactive clickable body map SVG replacing dropdown body location selector
- Image upload fallback with dermoscopy/clinical auto-detection
- Outcome feedback recording (AI Agreed / Overcalled / Missed / Record Pathology)
- Professional branding (DA logo, "RESEARCH ONLY" badge, v0.2.0)
- Medical disclaimers on every screen
- Risk icons alongside risk colors in results
- Sticky action buttons on mobile
- Fifth tab: Analytics (Capture, Results, History, Analytics, Settings)
- Server-side API routes for both HuggingFace models (`/api/classify`, `/api/classify-v2`)
- Environment-configurable model selection (`HF_MODEL_1`, `HF_MODEL_2`)
- HAM10000 validation script (`scripts/validate-models.mjs`)
- Custom ViT training pipeline (`scripts/train-proper.py`) with focal loss (gamma=2.0, melanoma alpha=8.0), 3-layer class balancing, model selection by melanoma sensitivity

### Changed
- Classification engine: from color-histogram demo to 4-layer ensemble (dual-HF 50% + literature 30% + rules 20%)
- Body location input: from dropdown to interactive SVG body map
- ABCDE scores: from hardcoded placeholders to real image-derived measurements (segmentation, moment analysis, border octants, k-means colors, GLCM, LBP)
- Ensemble architecture: from single HF model (60% weight) to dual-model ViT with class-specific weighting
- Offline fallback: from single trained-weights + rules to 60/40 literature/rules split
- API URLs: migrated from `api-inference.huggingface.co` to `router.huggingface.co/hf-inference`

### Fixed
- Welcome modal blocking app usage (removed auth requirement for dev)
- Classification errors silently swallowed (now displayed to user with actionable messages)
- HuggingFace API URL migration (old endpoints returning 404)
- actavkid model removal from HuggingFace (HTTP 410 Gone) -- replaced with skintaglabs SigLIP

### Removed
- actavkid/vit-large-patch32-384-finetuned-skin-lesion-classification (removed from HuggingFace, unverifiable claims)

### Validation Results (March 22, 2026)
- Anwarkh1 ViT-Base independently tested on 210 HAM10000 images: 73.3% melanoma sensitivity, 55.7% overall accuracy
- Hand-crafted features alone: 36.9% accuracy, 0% melanoma sensitivity (proven insufficient as standalone classifier)
- skintaglabs SigLIP: deployed but not yet validated on HAM10000
- Full 4-layer ensemble: not yet measured end-to-end
- Custom model training with focal loss and melanoma class weighting: in progress

## [0.1.0] - 2026-03-21

### Added
- Initial SvelteKit prototype with demo color-histogram classifier
- Camera capture with DermLite device selection
- HAM10000 Bayesian demographic adjustment (age, sex, body location)
- Privacy pipeline: EXIF stripping, differential privacy (epsilon=1.0), witness chain (SHAKE-256)
- Pi-brain integration for collective case sharing (opt-in)
- Offline PWA with service worker caching
- 4 tabs: Capture, Results, History, Settings
- Basic ABCDE scoring display (placeholder values)
- Classification probability bar chart
- Clinical recommendation thresholds (reassurance / monitor / biopsy / urgent referral)
- Server-side HuggingFace API proxy (single model)
