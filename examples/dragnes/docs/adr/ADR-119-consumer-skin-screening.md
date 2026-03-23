Updated: 2026-03-23 03:00:00 EST | Version 1.0.0
Created: 2026-03-23

# ADR-119: Consumer Skin Screening — "Is This Something I Should Worry About?"

## Status: PROPOSED

## Context

Dr. Agnes was designed for dermatologists analyzing dermoscopy images. But the real opportunity — and the real impact — is **5 billion people who will never see a dermatologist** being able to point their phone at their skin and get a trustworthy answer.

**The fundamental user question is:** "I found something on my skin. Should I worry?"

The current system fails this use case spectacularly. When a user shows a normal hand, the model says "93% melanoma." This happens because:

1. The model was trained only on lesion images — it has no concept of "normal skin"
2. It forces every image into one of 7 lesion classes
3. There is no pre-screening gate that says "this doesn't look like a lesion"
4. The interface assumes the user knows what a "lesion" is and can photograph it properly

**A consumer tool must solve a fundamentally different problem than a clinical tool.**

## Decision

Redesign Dr. Agnes for consumer self-screening with a "worry gate" architecture that handles the full spectrum of inputs a regular person would provide.

---

## The Consumer Journey

```
User: "I found a spot on my arm. Is this something?"
                    ↓
Step 1: CAPTURE — "Take a photo of the area that concerns you"
                    ↓
Step 2: GATE — Is this even a skin lesion?
        ├── Normal skin → "This looks like normal skin. No lesion detected."
        ├── Unclear → "Can you zoom in on the specific spot?"
        └── Lesion found → Continue to Step 3
                    ↓
Step 3: ANALYZE — What kind of lesion is this?
        ├── Classify into 7 HAM10000 categories
        ├── Compute ABCDE scores
        └── Generate risk assessment
                    ↓
Step 4: TRANSLATE — What does this mean for ME?
        ├── "Low concern" → "This looks benign. Monitor for changes."
        ├── "Watch it" → "This has some features worth tracking. Check again in 3 months."
        ├── "See a doctor" → "This has features that a dermatologist should evaluate."
        └── "Urgent" → "Please see a dermatologist within 2 weeks."
                    ↓
Step 5: GUIDE — What should I do next?
        ├── Set a reminder to re-check
        ├── Generate a summary for your doctor
        ├── Find a dermatologist near you
        └── Track this lesion over time
```

## What Must Change

### 1. Lesion Detection Gate (ADR-119.1)
**Before classification, determine if the image contains a skin lesion.**

| Input | Expected Output |
|-------|----------------|
| Normal hand/arm/face (no lesion) | "No skin lesion detected. If you see a specific spot, try zooming in." |
| A photograph of a table/food/pet | "This doesn't appear to be a skin image." |
| Blurry/dark image | "Image is too blurry/dark. Please retake in good lighting." |
| Clear photo of a mole/spot | Proceed to classification |
| Dermoscopy image | Proceed to classification (best quality) |

Implementation: `detectLesionPresence()` function that checks:
- Segmentation quality (did Otsu find a real boundary?)
- Color contrast (is there a distinct area vs surrounding skin?)
- Area ratio (lesion should be 2-60% of image)
- Shape compactness (roughly circular/oval)
- Image quality (brightness, blur detection)

### 2. Plain English Results (ADR-119.2)
**Replace medical jargon with human-readable language.**

| Medical Term | Consumer Translation |
|-------------|---------------------|
| "Melanocytic Nevus (nv)" | "Common mole — looks typical" |
| "Melanoma (mel)" | "This needs professional evaluation — it has concerning features" |
| "Basal Cell Carcinoma (bcc)" | "This should be checked by a dermatologist" |
| "Actinic Keratosis (akiec)" | "Sun damage spot — should be monitored" |
| "Benign Keratosis (bkl)" | "Age spot or seborrheic keratosis — typically harmless" |
| "Dermatofibroma (df)" | "Firm skin bump — usually benign" |
| "Vascular Lesion (vasc)" | "Blood vessel-related spot" |

### 3. Risk Level Communication (ADR-119.3)
**Four clear levels with specific actions:**

| Level | Color | Message | Action |
|-------|-------|---------|--------|
| **Green** | Green | "Low concern — this looks typical" | "Monitor for changes. Re-check in 6 months." |
| **Yellow** | Amber | "Worth watching — some features to track" | "Take a photo monthly. See a doctor if it changes." |
| **Orange** | Orange | "See a doctor — this has concerning features" | "Schedule a dermatology appointment within 1 month." |
| **Red** | Red | "Urgent — please see a dermatologist soon" | "Contact a dermatologist within 2 weeks." |

### 4. Image Quality Guidance (ADR-119.4)
**Help the user take a good photo:**

- "Move closer — the spot should fill most of the frame"
- "Better lighting needed — move toward a window"
- "Hold steady — the image is blurry"
- "Remove hair/shadow from the area if possible"

### 5. Multi-Image Comparison (ADR-119.5)
**The "E" in ABCDE is Evolution — the most important feature.**

- "Take 3 photos from different angles"
- "Come back in 1 month and photograph the same spot"
- Side-by-side comparison showing changes
- Automated evolution scoring: "This lesion has grown 15% since last month"

### 6. Know Your Skin Education (ADR-119.6)
**Help users understand what they're looking at:**

- Interactive ABCDE guide: "Here's what asymmetry looks like"
- "Ugly duckling" concept: "Does this mole look different from your others?"
- Sun exposure risk factors
- When to self-screen and how often

---

## Architecture Changes

### New Component: LesionGate
```
Image → LesionGate → { hasLesion, quality, confidence }
                          ↓ (only if hasLesion)
                    → Classifier → Results → ConsumerTranslation
```

### New Component: ConsumerTranslation
```
ClassificationResult → ConsumerTranslation → {
    riskLevel: "green" | "yellow" | "orange" | "red",
    headline: "Low concern — this looks typical",
    explanation: "This appears to be a common mole...",
    action: "Monitor for changes. Re-check in 6 months.",
    shouldSeeDoctor: boolean,
    urgency: "routine" | "soon" | "urgent",
}
```

### New Component: ImageQualityChecker
```
Image → ImageQualityChecker → {
    quality: "good" | "acceptable" | "poor",
    issues: ["too_dark", "blurry", "too_far"],
    guidance: "Move closer to the spot",
}
```

### New Component: LesionTracker
```
LesionHistory → LesionTracker → {
    hasChanged: boolean,
    growthPercent: number,
    colorChange: boolean,
    recommendation: "stable" | "changed" | "concerning_change",
}
```

---

## RuVector Ecosystem Integration

### Pi-Brain for Consumer Intelligence
- Store anonymized classification patterns from consumer use
- "Users who had lesions like this — 95% were benign"
- Case-based reassurance: "Similar spots in our database were safe"
- Trending: "Most common spot type for your age/skin type"

### RuVector CNN for Offline
- ONNX model runs in browser (83MB INT8)
- Full classification without internet
- Critical for rural/underserved areas with poor connectivity
- Sub-200ms inference on modern phones

### Claude Flow for Iteration
- Consumer feedback loop: "Was this helpful?" → improves responses
- A/B testing different communication styles
- Multi-language support via translation agents

---

## Success Criteria

| Criterion | Threshold |
|-----------|----------|
| Normal skin correctly identified as "no lesion" | ≥ 95% |
| Melanoma flagged as "see a doctor" or "urgent" | ≥ 95% |
| User understands the recommendation (usability study) | ≥ 90% |
| False "urgent" on benign lesions | < 20% |
| User completes recommended action (sees doctor when told to) | ≥ 50% |
| Time from photo to recommendation | < 10 seconds |
| Works offline | Full classification |
| Supports Fitzpatrick I-VI equally | < 5% gap |

---

## What This Changes

| From (Current) | To (Consumer) |
|----------------|---------------|
| "Melanocytic Nevus, 73.1% confidence" | "Common mole — low concern" |
| Classifies any image as a lesion | "No lesion detected" for normal skin |
| Medical jargon (ABCDE, TDS, ICD-10) | "Here's what to do next" |
| Designed for dermatologists | Designed for your grandmother |
| Requires dermoscopy knowledge | "Just take a photo" |
| Single image analysis | Track changes over time |
| Clinical recommendation | "See a doctor" / "You're fine" |

---

## Implementation Priority

1. **Lesion detection gate** (fixes the hand-as-melanoma bug) — IMMEDIATE
2. **Consumer-friendly results translation** — Week 1
3. **Image quality guidance** — Week 1
4. **Multi-image comparison** — Week 2
5. **Lesion tracking over time** — Week 3
6. **Education content** — Week 4
7. **Pi-brain consumer intelligence** — Ongoing

## References

- ADR-117: Original DrAgnes architecture
- ADR-118: Production validation roadmap
- DermaSensor consumer use data (DERM-SUCCESS)
- AAD public screening guidelines 2024-2026
- SkinVision consumer app research

## Author
Stuart Kerr + Claude Flow (RuVector/RuFlo)
