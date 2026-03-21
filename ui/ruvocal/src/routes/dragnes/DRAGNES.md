# DrAgnes -- Dermatology Intelligence

DrAgnes is an AI-powered dermoscopy analysis tool that runs a lightweight CNN
directly in your browser (via WebAssembly) and contributes anonymized learning
signals to a collective knowledge graph hosted on the pi.ruv.io brain network.

---

## Getting Started

1. **Open DrAgnes** -- navigate to `/dragnes` in your browser.
2. **Allow camera access** when prompted, or tap the upload button to select an
   existing dermoscopy image.
3. **Capture or upload** the lesion photo. For best results use a DermLite or
   equivalent dermatoscope attachment.
4. DrAgnes will classify the lesion in under 200 ms and display the results.

### Install as PWA

On supported browsers (Chrome, Edge, Safari 17+) you can install DrAgnes to
your home screen for a native-like experience with offline support:

- Tap the browser menu and select **"Install DrAgnes"** or **"Add to Home
  Screen"**.
- Once installed the app runs in standalone mode and caches the CNN model for
  offline use.

---

## Using a DermLite with DrAgnes

DrAgnes is optimized for polarized dermoscopy images captured with DermLite
devices:

1. Attach the DermLite to your phone camera.
2. Place the lens directly on the skin lesion.
3. Enable polarized mode on the DermLite for subsurface detail.
4. Capture the image through DrAgnes -- the app will auto-crop and normalize
   the image before classification.

Tip: Ensure even contact pressure and consistent lighting for reproducible
results.

---

## Understanding Classification Results

DrAgnes classifies lesions into seven categories from the HAM10000 taxonomy:

| Code    | Label                 | Clinical Significance |
|---------|-----------------------|-----------------------|
| akiec   | Actinic Keratosis     | Pre-cancerous         |
| bcc     | Basal Cell Carcinoma  | Malignant             |
| bkl     | Benign Keratosis      | Benign                |
| df      | Dermatofibroma        | Benign                |
| mel     | Melanoma              | Malignant             |
| nv      | Melanocytic Nevus     | Benign                |
| vasc    | Vascular Lesion       | Benign                |

Each classification includes:

- **Top prediction** with confidence score (0--100%).
- **Full probability distribution** across all seven classes.
- **Embedding vector** (128-dim) used for similarity search against the brain
  knowledge graph.

---

## ABCDE Scoring Explained

DrAgnes supplements CNN classification with the ABCDE dermoscopy checklist:

- **A -- Asymmetry**: Is the lesion asymmetric in shape or color?
- **B -- Border**: Are the borders irregular, ragged, or blurred?
- **C -- Color**: Does the lesion contain multiple colors or unusual shades?
- **D -- Diameter**: Is the lesion larger than 6 mm?
- **E -- Evolution**: Has the lesion changed over time?

Each criterion is scored 0 (absent) to 2 (strongly present). A total score of
3 or above warrants clinical review.

---

## Privacy and Compliance

DrAgnes is designed with privacy at its core:

- **On-device inference** -- the CNN runs entirely in the browser via WASM.
  Images never leave the device.
- **Differential privacy** -- gradient updates contributed to the brain use
  epsilon = 1.0 differential privacy noise.
- **k-Anonymity** -- contributions are batched and only submitted when at
  least k = 5 local samples exist, preventing individual identification.
- **Witness hashing** -- all brain contributions are hashed with SHA-256 to
  create an auditable, tamper-evident record.
- **No PII** -- DrAgnes does not collect names, emails, or any personally
  identifiable information.

DrAgnes is a clinical decision support tool and does NOT store or transmit
patient images.

---

## Offline Mode

DrAgnes works fully offline after the first visit:

- The WASM CNN model (~5 MB) is cached by the service worker.
- Classifications run locally with no network required.
- Brain contributions are queued and synced automatically when connectivity
  is restored (via Background Sync API).
- Model updates are fetched in the background when available and trigger a
  push notification.

---

## Troubleshooting

### Camera not working
- Ensure you have granted camera permissions in your browser settings.
- On iOS, DrAgnes requires Safari 17+ for full WASM support.
- Try reloading the page or clearing the site data.

### Classification seems inaccurate
- Verify the image is in focus and well-lit.
- Use polarized dermoscopy mode for better subsurface detail.
- Ensure the lesion fills most of the frame.
- DrAgnes performs best on the HAM10000 taxonomy; unusual lesions may not be
  well-represented.

### Offline mode not working
- Ensure you have visited `/dragnes` at least once while online.
- Check that your browser supports service workers (all modern browsers do).
- Clear the service worker cache and reload if assets seem stale.

### Slow performance
- Close other browser tabs to free memory for WASM execution.
- DrAgnes targets < 200 ms inference on modern devices. Older hardware may be
  slower.

---

## Clinical Disclaimer

**DrAgnes is a research and clinical decision support tool. It is NOT a
medical device and is NOT intended to replace professional dermatological
evaluation.**

- All classifications are probabilistic estimates and should be interpreted by
  a qualified healthcare professional.
- DrAgnes has not been cleared or approved by the FDA, EMA, or any other
  regulatory body.
- Always refer patients with suspicious lesions for biopsy and
  histopathological confirmation.
- The developers of DrAgnes accept no liability for clinical decisions made
  based on its output.

Use DrAgnes to augment -- never replace -- your clinical judgment.
