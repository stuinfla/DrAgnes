Updated: 2026-03-26 | Version 1.0.0
Created: 2026-03-26

# Clinical Feedback -- Dr. Agnes Ju Chang

Initial clinical evaluation of Mela, received 2026-03-25.

Dr. Chang is the dermatologist who inspired the project (originally named "Dr. Agnes" after her). This is her first detailed clinical review of the tool.

---

## What She Liked

1. **Privacy design** -- AI runs on phone, photos don't get uploaded to a server.
2. **Four layers of safety checks** -- AI model, medical literature, clinical checklists, demographics.
3. **Risk output, not binary** -- gives actual risk percentages, not just yes/no.

---

## Clinical Concerns (Prioritized by Risk)

### TIER 1: Patient Safety

#### 1. No "Normal Skin" Category
> "If someone takes a picture of a freckle, birthmark or their leg, Dr. Agnes forces it into one of 7 categories. There is no 'normal, you are good' option."

**Status:** Partially addressed. The lesion gate routes "no lesion detected" to a green "Your skin looks healthy!" UI (`MelaPanel.svelte:1464`). However, benign lesions that PASS the lesion gate (freckles, birthmarks) still get force-classified into one of 7 categories. A freckle classified as "melanocytic nevus -- 5% risk" could cause unnecessary panic and erode trust.

**Recommendation:** Expand the classification output to include a "normal/benign -- no concern" tier. When the model's top-class confidence is low AND clinical features (ABCDE) are unremarkable, display a green "This looks like a normal spot" result instead of forcing a diagnostic category.

#### 2. SCC Needs Its Own Category
> "SCC is the second most common type of skin cancer. It can kill people because it can spread. Right now it is linked to AK (actinic keratosis) which is a pre-cancerous lesion."

**Status:** Not addressed. The HAM10000 taxonomy lumps SCC under `akiec`. The consumer translation in `consumer-translation.ts:93` mentions SCC as a possible progression but treats the category as pre-cancerous.

**Recommendation:** Short-term: update consumer translation to make the SCC risk explicit and recommend dermatologist evaluation for any akiec result. Long-term: retrain with ISIC 2019 data which separates SCC as its own class (8 classes instead of 7).

#### 3. Amelanotic Melanoma Blind Spot
> "This is a flesh colored melanoma that has NO pigment and no color contrast -- just looks like a pink bump. They are very difficult to see even by dermatologists."

**Status:** No special handling. Low-pigment lesions will get low model confidence and likely misclassify.

**Recommendation:** Add a low-confidence safety net in `consumer-translation.ts`: when confidence is below threshold AND the image shows flesh-toned/pink features, display: "Low confidence result. Some dangerous skin cancers (amelanotic melanoma) lack visible pigment. When in doubt, see a dermatologist."

#### 4. Dermoscopy vs. Phone Camera Training Gap
> "All the training came from dermoscopic images from a dermatoscope, not real camera photos. A dermoscopic photo of a melanoma is VERY different from an iPhone photo of the same cancer."

**Status:** Not addressed. HAM10000 and ISIC 2019 are 100% dermoscopic. The app is marketed for phone cameras. This is a fundamental validity gap.

**Recommendation:** Short-term: add prominent disclaimer that the model was trained on dermoscopic images. Long-term: fine-tune the V2 model on clinical (non-dermoscopic) images. Fitzpatrick17k has clinical photos. This is a multi-month research effort but is the single biggest gap between "research tool" and "deployable product."

#### 5. Fitzpatrick Equity Gap
> "If you have a drop of 30%, it is not a great tool to be deployed in 50% of skin types."

**Status:** Partially addressed. The 30pp gap is disclosed in MethodologyPanel, AboutPage, and result views. But she's asking for more: a per-scan disclaimer for Fitzpatrick IV-VI users.

**Recommendation:** If user's Fitzpatrick type is IV-VI, show inline disclaimer on every result: "This tool has not been validated for darker skin tones. Results should be interpreted with extra caution."

### TIER 2: Clinical Completeness

#### 6. HuggingFace / HIPAA Concern
> "You took Dr. Agnes and uploaded photos to HuggingFace which means photos get sent to HuggingFace. Concerned about HIPAA violation."

**Status:** V2 model (ONNX, 85MB) runs fully on-device. HF API is a fallback path, not primary. But the HF fallback path does exist in code.

**Recommendation:** ONNX-only for production. Remove or gate the HF API path behind an explicit developer/research toggle that is OFF by default. Add clear disclosure about which inference mode is active.

#### 7. ONNX Model Not Validated (Compression Fidelity)
> "ONNX model isn't validated because it is compressed and not original size."

**Status:** The INT8 quantized ONNX model has not been separately validated against the original PyTorch model.

**Recommendation:** Run the full ISIC 2019 validation suite against the deployed INT8 ONNX model specifically. Document any accuracy delta from quantization.

#### 8. No Photo Guidance for Users
> "Most people will mess up -- you need guidance for lighting, 6 inches away, clean lens, etc."

**Status:** Post-capture quality checks exist in `image-quality.ts` (blur, darkness, contrast). `AboutPage.svelte:205` has brief tips. But there is no pre-capture guidance screen.

**Recommendation:** Add a "How to take a good photo" overlay before first capture: good lighting (natural daylight best), 6 inches away, clean lens, center one spot, use macro mode if available.

#### 9. No Clinical History Questions
> "If the lesion has been biopsied before, we need to know that because recurrent nevi can look like cancer. We also need to ask: Is this new? Do you have symptoms? Do you have a family history?"

**Status:** The app asks age, sex, body location, Fitzpatrick type. No clinical history.

**Recommendation:** Add pre-scan questionnaire (5-7 questions): Is this new or changing? Prior biopsy at this site? Family history of melanoma? Symptoms (itching, bleeding)? Feed answers into Bayesian risk stratification layer (`risk-stratification.ts`).

#### 10. Multiple Lesions in One Photo
> "If they take a picture of their arm and they have freckles, moles, scars, etc, which one is being analyzed?"

**Status:** Spot detector finds dominant lesion but no user guidance about isolating a single lesion.

**Recommendation:** Pre-capture instruction: "Center one spot in the frame." Post-detection: show crop overlay / bounding box indicating which lesion was analyzed. If multiple detected, prompt to crop or retake.

### TIER 3: Edge Cases and Disclaimers

#### 11. Pediatric Patients
> "If a parent uses this for a congenital mole on their infant, this would produce inappropriate results."

**Status:** No age-based warnings.

**Recommendation:** If age < 18, show disclaimer: "This tool was trained on adult skin lesion data. Congenital moles and pediatric skin conditions may produce unreliable results."

#### 12. Rare Non-Melanoma Skin Cancers
> "What about Merkel cell, DFSP, cutaneous lymphoma, Kaposi's sarcoma -- rare but deadly."

**Recommendation:** Add to limitations: "This tool screens for the 7 most common lesion categories. Rare skin cancers (Merkel cell carcinoma, DFSP, Kaposi's sarcoma, cutaneous lymphoma) are not in the training data and will not be detected."

#### 13. Acral / Subungual Melanomas
> "Subungual melanomas (under the nails) look like dark lines under the fingernail or toenail. 100% different visual."

**Recommendation:** Photo guidance: "This tool cannot analyze lesions under fingernails or toenails. Dark lines under nails should be evaluated by a dermatologist." Consider nail-image detector that auto-triggers this warning.

#### 14. Tattoos, Scars, Piercings
> "Lesions near tattoos and biopsy scars will confuse the tool."

**Recommendation:** Photo guidance: "Lesions near tattoos, scars, or piercings may produce unreliable results due to color and texture interference."

### TIER 4: UX Enhancements

#### 15. ABCDE "E" for Evolution -- Side-by-Side Comparison
> "It's not just ABCD -- it is ABCDE. E stands for 'Evolving'. Tools should have standardized comparison tools over time like side by side photos."

**Status:** ABCDE scoring includes E. `LesionTimeline.svelte` tracks history and computes evolution deltas. But there is no visual side-by-side comparison view.

**Recommendation:** Add comparison view in history: two images side-by-side with ABCDE delta overlays. Data infrastructure exists, needs UI.

#### 16. PDF Summary for Doctors
> "Probably want to have a PDF one page summary they can print out to show doctors."

**Status:** `ReferralLetter.svelte` generates a text referral letter. No PDF export.

**Recommendation:** Add "Download Summary" button generating a one-page PDF: photo, classification, risk level, ABCDE scores, recommendation. Printable for doctor visits.

#### 17. Referral Letter Signature Issue
> "The referral letter has a referring provider's signature on it even though this is AI generated."

**Status:** The referral letter has a "referring provider" signature block.

**Recommendation:** Reword to "AI Screening Report." Remove language implying a referring physician. Add: "Generated by Mela AI screening tool -- not a medical referral."

---

## Implementation Phases

| Phase | Items | Effort |
|---|---|---|
| Phase 1: Safety text | #3 amelanotic, #5 Fitzpatrick per-scan, #11 pediatric, #12 rare cancers, #13 acral, #14 tattoos, #17 referral | 1-2 days |
| Phase 2: UX gaps | #10 multi-lesion, #8 photo guidance, #9 clinical questionnaire, #16 PDF export | 3-5 days |
| Phase 3: Classification | #1 normal skin tier, #2 SCC consumer warning | 2-3 days |
| Phase 4: Comparison UX | #15 side-by-side evolution | 2-3 days |
| Phase 5: Validation | #7 ONNX validation, #4 dermoscopy disclaimer, #6 HF path gating | 1-2 days |
| Phase 6: Model work | #2 SCC retraining, #4 clinical photo training | Weeks-months |

---

## Source

Full text of Dr. Chang's feedback is preserved in this document. All quotes are verbatim.
