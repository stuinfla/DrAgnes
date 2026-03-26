Updated: 2026-03-26 | Version 1.0.0
Created: 2026-03-26

# FDA Regulatory Analysis — Mela

## Decision: Option B — Skin Awareness & Education Tool

Based on deep research into FDA guidance documents, the 21st Century Cures Act, real-world precedents (Apple Watch, SkinVision, DermaSensor, 23andMe), and FTC enforcement actions, Mela has been reframed from a "screening tool" to a "skin awareness and education tool" to stay outside FDA medical device jurisdiction.

## Key Findings

1. Any software that analyzes skin images and outputs disease-specific information IS a medical device under FDA rules, regardless of disclaimers, licensing, or cost.
2. The CDS exemption (21st Century Cures Act Section 3060) does NOT apply — fails Criterion 1 (image analysis).
3. The General Wellness exception does NOT apply — disease-specific outputs disqualify.
4. No consumer-facing skin cancer screening app has ever received FDA authorization.
5. DermaSensor was cleared for physician use only, not consumer use.
6. FTC fined MelApp and Mole Detective in 2015 for unsupported diagnostic claims.

## Changes Made

- Mandatory disclaimer gate before app access
- All diagnostic language replaced with pattern similarity language
- "See a dermatologist" recommendations replaced with general health advice
- Comparison table vs FDA-cleared devices removed
- Referral letter generation removed from UI
- Risk tiers replaced with neutral pattern analysis language
- "Screening" removed from all marketing and meta tags
- App framed as educational/awareness tool, not diagnostic

## Legal Basis

The app now functions as a photo documentation + skin education + pattern awareness tool. It does NOT:
- Diagnose any disease
- Recommend specific clinical actions based on analysis
- Generate referral letters
- Compare itself to FDA-cleared medical devices
- Use "screening," "detection," or "diagnosis" in consumer-facing text

## Sources

Full source list in the FDA research agent output. Key references:
- FDA SaMD guidance: fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- 21st Century Cures Act Section 3060 / FD&C Act Section 520(o)(1)(E)
- FDA CDS Guidance (January 6, 2026 update)
- FDA General Wellness Guidance (January 6, 2026 update)
- DermaSensor De Novo: DEN230008
- FTC enforcement: MelApp, Mole Detective (2015)
- FDA Skin Lesion Analyzer Panel Meeting (July 28, 2022)
