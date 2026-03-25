---
name: DrAgnes v0.9.4 — Standalone Project
description: AI dermatoscopy — 95.97% mel sensitivity, two-pass spot detection, Bayesian risk, ONNX offline, Vercel live
type: project
---

## Dr. Agnes v0.9.4 — AI Dermatoscopy Screening

**Live:** https://dragnes.vercel.app
**GitHub:** https://github.com/stuinfla/DrAgnes
**Vercel:** https://vercel.com/stuart-kerrs-projects/dragnes

### Read These First
- `CLAUDE.md` at repo root — project instructions
- `docs/PROJECT-HISTORY.md` — full development history (718 lines)
- `docs/PIPELINE-EXPLAINER.md` — technical pipeline walkthrough (1,430 lines)

### Key Facts
- Version 0.9.4 with two-pass spot detection (LAB histogram + blob detection)
- 95.97% melanoma sensitivity on external ISIC 2019 data (3,901 images)
- 95% CI: 94.5% - 97.4% (source: confidence-intervals.json)
- Mel AUROC: 0.960 (source: combined-training-results.json)
- NPV: 99.06% — "not melanoma" is very reliable
- NNB: 2.1 — better than DermaSensor (6.25)
- 13 ADRs (117-129), all with code implementation
- ONNX INT8 model (85MB) at static/models/dragnes-v2-int8.onnx
- Bayesian risk stratification (5 tiers) replaces binary classification
- Meta-classifier fuses neural + clinical features to reduce false positives
- Fitzpatrick equity: DANGEROUS 30pp gap (needs diverse training data)
- Temperature calibration: T=1.23, ECE 0.078→0.044

### Deployment (MANDATORY PROCESS)
1. Bump version in THREE places: package.json, src/routes/+page.svelte, README.md
2. Commit and push to main
3. Verify: vercel ls --scope stuart-kerrs-projects (must show ● Ready)
4. Verify: curl the live site for version string
5. If build fails: vercel inspect <url> --logs

### Evidence Files (scripts/)
- combined-training-results.json — THE source of truth (95.97%)
- clinical-metrics-full.json — PPV, NPV, NNB, calibration, failure modes
- auroc-results.json — per-class AUROC with bootstrap CI
- confidence-intervals.json — 95% CI on all headline metrics
- threshold-optimization-results.json — 3 threshold modes
- ensemble-validation-results.json — V1+V2 (99.4% mel)
- fitzpatrick-v2-validation.json — equity gap (30pp)
- onnx-v2-validation.json — INT8 accuracy verified

### What NOT To Do
- NEVER use git add -A (grabbed 6,216 cached images once)
- NEVER claim a number without citing a JSON evidence file
- NEVER deploy without verifying vercel ls shows ● Ready
- NEVER skip testing: npx vitest run
- NEVER present normal skin as an error — show green "healthy"

### Setup After Clone
```bash
npm install
vercel link --project dragnes --scope stuart-kerrs-projects --yes
npx ruflo@latest init --force
echo "HF_TOKEN=your_token_here" > .env
npm run dev
```
