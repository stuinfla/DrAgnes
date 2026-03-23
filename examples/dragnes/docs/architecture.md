# DrAgnes System Architecture

Updated: 2026-03-22 18:30:00 EST | Version 1.1.0
Created: 2026-03-21

**Status**: Research Prototype (v0.2.0)

## Overview

DrAgnes is a layered architecture that connects dermoscopic imaging hardware through a mobile-first web application to a 4-layer classification ensemble and collective intelligence brain. The design prioritizes offline capability, privacy preservation, clinical safety, and honest accuracy reporting.

## High-Level Architecture

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                      DrAgnes Platform (v0.2.0)               │
                    │                                                              │
  ┌──────────┐     │  ┌──────────────┐    ┌─────────────────────────────────────┐ │
  │ DermLite │────>│  │  SvelteKit   │───>│  4-Layer Classification Ensemble    │ │
  │ HUD/DL5  │     │  │  PWA UI      │    │                                     │ │
  └──────────┘     │  │  (5 tabs)    │    │  L1: Dual ViT (Anwarkh1 + SigLIP)  │ │
                    │  │              │    │  L2: Literature Logistic Regression  │ │
  ┌──────────┐     │  │  Capture     │    │  L3: Rule-Based (TDS, 7pt, gates)   │ │
  │ Phone    │────>│  │  Results     │    │  L4: Bayesian Demographics           │ │
  │ Camera   │     │  │  History     │    └─────────────────────────────────────┘ │
  └──────────┘     │  │  Analytics   │                    │                       │
                    │  │  Settings    │                    ▼                       │
  ┌──────────┐     │  └──────────────┘    ┌─────────────────────────────────────┐ │
  │ Image    │────>│         │            │  Image Analysis Engine (1,890 LOC)   │ │
  │ Upload   │     │         │            │  Otsu seg | ABCDE | GLCM | LBP     │ │
  └──────────┘     │         │            │  k-means color | attention maps      │ │
                    │         │            └─────────────────────────────────────┘ │
                    │         │                                                    │
                    │         ▼                                                    │
                    │  ┌──────────────────────────────────────────────────────┐    │
                    │  │  Clinical Tools                                      │    │
                    │  │  ICD-10 | Referral Letters | Explainability          │    │
                    │  │  Analytics Dashboard | Outcome Feedback              │    │
                    │  └──────────────────────────────────────────────────────┘    │
                    │                                                              │
                    │  ┌──────────────────────────────────────────────────────┐    │
                    │  │  Privacy & Compliance Layer                          │    │
                    │  │  EXIF Strip | Diff. Privacy (e=1.0) | Witness Chain  │    │
                    │  └──────────────────────────────────────────────────────┘    │
                    │                                                              │
                    │  ┌──────────────────────────────────────────────────────┐    │
                    │  │  External Services (server-side only)                │    │
                    │  │  HuggingFace Inference API | pi.ruv.io Brain         │    │
                    │  └──────────────────────────────────────────────────────┘    │
                    └──────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. DermLite Device Integration Layer

DermLite devices attach to smartphones and provide standardized dermoscopic imaging.

**Supported Devices**:
- **DermLite HUD** (Heads-Up Display): Hands-free dermoscopy with built-in camera. Connects via Bluetooth for metadata. Captures 1920x1080 polarized and non-polarized images.
- **DermLite DL5**: Flagship handheld dermatoscope. 10x magnification, hybrid polarized/non-polarized mode. USB-C or Lightning adapter for phone attachment.
- **DermLite DL4**: Compact pocket dermatoscope. Smartphone adapter available. LED illumination with polarization.
- **DermLite DL200 Hybrid**: Contact and non-contact dermoscopy. Magnetic phone adapter.

**Image Capture Flow**:
```
DermLite Adapter
    │
    ├── Phone Camera (MediaStream API)
    │       │
    │       ▼
    │   getUserMedia({ video: { facingMode: 'environment',
    │                           width: 1920, height: 1080 } })
    │       │
    │       ▼
    │   Canvas capture (ImageData → Uint8Array)
    │       │
    │       ▼
    │   Preprocessing Pipeline
    │       ├── Color normalization (Shades of Gray)
    │       ├── Hair removal (DullRazor algorithm via WASM)
    │       ├── Lesion segmentation (Otsu + GrabCut via WASM)
    │       ├── Resize to 224x224 (bilinear interpolation)
    │       └── ImageNet normalization (mean=[0.485,0.456,0.406],
    │                                   std=[0.229,0.224,0.225])
    │
    ▼
Preprocessed Tensor [1, 3, 224, 224] float32
```

**DermLite-Specific Processing**:
- Auto-detect polarization mode from EXIF metadata
- Calibrate white balance using DermLite's known LED spectrum (4500K)
- Extract measurement scale from DermLite's ruler overlay
- Compensate for contact plate reflection artifacts in contact dermoscopy mode

### 2. 4-Layer Classification Ensemble

The classification engine uses a 4-layer ensemble that degrades gracefully when offline.

**Architecture**:
```
Input Image (dermoscopy or clinical photo)
    │
    ▼
Preprocessing Pipeline
    │   ├── Shades-of-Gray color normalization (Minkowski p=6)
    │   ├── DullRazor hair removal
    │   ├── Bilinear resize to 224x224
    │   └── NCHW tensor + ImageNet normalization
    │
    ├──────────────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      ▼
Image Analysis Engine (1,890 LOC)                   HuggingFace Inference API
    │                                                      │
    ├── Segmentation (Otsu in LAB + morphological)        ├── /api/classify
    ├── Asymmetry (principal-axis inertia)                 │     Anwarkh1 ViT-Base (85.8M)
    ├── Border (8-octant CV + color gradient)              │
    ├── Color (k-means++ in LAB, 6 reference colors)      ├── /api/classify-v2
    ├── Texture (GLCM: contrast, homogeneity, entropy)    │     skintaglabs SigLIP (400M)
    ├── Structures (LBP, globules, streaks, BWV, reg)     │
    └── Attention map generation                           └── Promise.allSettled (parallel)
    │                                                      │
    ▼                                                      ▼
Layer 2: Literature LR    Layer 3: Rules          Layer 1: Dual ViT Ensemble
(20x7 weight matrix)      (TDS, 7pt, gates)       (50/50 equal weighting)
    │  30% weight              │  20% weight            │  50% weight
    │                          │                        │
    └──────────────────────────┴────────────────────────┘
                               │
                               ▼
                    Layer 4: Bayesian Demographic Adjustment
                    (age/sex/location multipliers from HAM10000)
                               │
                               ▼
                    Final 7-class probability distribution
                               │
                               ▼
                    Clinical recommendation thresholds
                    (reassurance / monitor / biopsy / urgent referral)
```

**Fallback chain**:
- Both HF models available: 50% dual-HF + 30% literature + 20% rules
- One HF model available: 60% single-HF + 25% literature + 15% rules
- Offline: 60% literature + 40% rules
- Always applied: demographic adjustment, melanoma safety gate, TDS override

**Classification Taxonomy** (7 classes, aligned with HAM10000):
| Class | Label | Risk Level |
|-------|-------|-----------|
| akiec | Actinic keratosis / Bowen's | Medium-High |
| bcc | Basal cell carcinoma | High |
| bkl | Benign keratosis (solar lentigo, seborrheic keratosis) | Low |
| df | Dermatofibroma | Low |
| mel | Melanoma | Critical |
| nv | Melanocytic nevus (mole) | Low |
| vasc | Vascular lesion (angioma, angiokeratoma, pyogenic granuloma) | Low |

**Measured Performance** (March 22, 2026):
| Metric | Measured | Target | Notes |
|--------|----------|--------|-------|
| Anwarkh1 melanoma sensitivity | 73.3% | >95% | Tested on 210 HAM10000 images |
| Anwarkh1 overall accuracy | 55.7% | >85% | Below published ViT benchmarks |
| Hand-crafted features alone | 0% mel sens | -- | Proven insufficient |
| Full ensemble | Not measured | >95% mel sens | Validation needed |
| Fitzpatrick V-VI performance | Not measured | <5% disparity | HAM10000 is ~95% FST I-III |

**Safety Mechanisms**:
| Mechanism | Trigger | Effect |
|-----------|---------|--------|
| Melanoma safety gate | 2+ concurrent suspicious features | P(mel) >= 15% floor |
| TDS override | TDS > 5.45 (malignant range) | P(malignant) >= 30% |
| Model disagreement alert | ViT models disagree on top-1 | Clinical review warning |
| Probability floor | 3+ suspicious features | 15% melanoma minimum |

### 3. Brain Integration Layer

The pi.ruv.io brain serves as the collective intelligence backbone.

**Data Flow**:
```
Diagnosis Complete
    │
    ▼
PII Stripping Pipeline
    │   ├── Remove patient identifiers
    │   ├── Remove GPS/location from EXIF
    │   ├── Remove device serial numbers
    │   ├── Generalize age to decade bracket
    │   ├── Generalize skin type to Fitzpatrick scale
    │   └── Hash remaining quasi-identifiers (k-anonymity, k>=5)
    │
    ▼
Differential Privacy Layer (epsilon=1.0)
    │   ├── Laplace noise on continuous features
    │   ├── Randomized response on categorical features
    │   └── Privacy budget tracking per practice per epoch
    │
    ▼
RVF Cognitive Container
    │   ├── Segment 0: 576-dim embedding (no raw image)
    │   ├── Segment 1: Classification probabilities
    │   ├── Segment 2: ABCDE scores
    │   ├── Segment 3: De-identified metadata
    │   │       ├── Fitzpatrick type (I-VI)
    │   │       ├── Body location (categorical)
    │   │       ├── Age decade
    │   │       ├── Lesion diameter (mm, bucketed)
    │   │       └── Dermoscopic features present
    │   └── Segment 4: Witness chain (SHAKE-256)
    │
    ▼
Brain Memory Insert
    │   ├── HNSW index update (128-dim projected via RlmEmbedder)
    │   ├── Knowledge graph edge creation
    │   ├── Sparsifier incremental update (ADR-116)
    │   └── GNN topology enrichment
    │
    ▼
Cross-Practice Learning
    │   ├── PageRank-weighted similarity across all practices
    │   ├── SONA meta-learning for population-level patterns
    │   ├── PubMed enrichment for newly observed lesion subtypes
    │   └── Federated model update (no raw data exchange)
```

**Brain Endpoints Used**:
| Endpoint | Purpose |
|----------|---------|
| `brain_share` | Submit de-identified diagnosis embedding to collective |
| `brain_search` | Find similar historical cases by embedding similarity |
| `brain_page_create` | Create structured dermatology knowledge pages |
| `brain_page_evidence` | Attach PubMed evidence to diagnostic findings |
| `brain_drift` | Monitor embedding space drift as new lesion types emerge |
| `brain_partition` | Cluster lesion subtypes via MinCut partitioning |
| `brain_sync` | Sync local model updates with collective |

### 4. SvelteKit UI (5-Tab Interface)

DrAgnes is a standalone SvelteKit application with 5 tabs. The main panel is `DrAgnesPanel.svelte`.

**UI Components**:
```
DrAgnes Panel (DrAgnesPanel.svelte)
    │
    ├── Tab 1: Capture (DermCapture.svelte)
    │       ├── Camera viewfinder with DermLite device selection
    │       ├── Interactive clickable body map SVG (replaces dropdown)
    │       ├── Image upload fallback
    │       ├── Dermoscopy / clinical photo auto-detection
    │       ├── Capture button (high-res still)
    │       └── Medical disclaimer
    │
    ├── Tab 2: Results (ClassificationResult.svelte)
    │       ├── Classification probabilities (7-class bar chart)
    │       ├── Risk level with color-coded icons
    │       ├── ABCDE score breakdown (real image-derived values)
    │       ├── TDS score with interpretation
    │       ├── 7-point checklist evaluation
    │       ├── Attention heatmap overlay
    │       ├── Clinical recommendation (reassurance / monitor / biopsy / urgent)
    │       ├── ICD-10-CM code for predicted class
    │       ├── Model agreement/disagreement indicator
    │       ├── Referral letter generator (ReferralLetter.svelte)
    │       ├── "Why this classification?" panel (ExplainPanel.svelte)
    │       ├── Outcome feedback buttons (Agreed / Overcalled / Missed / Pathology)
    │       └── Sticky action buttons on mobile
    │
    ├── Tab 3: History
    │       ├── Prior classifications with timestamps
    │       ├── Outcome feedback status
    │       └── Re-review capability
    │
    ├── Tab 4: Analytics
    │       ├── Practice concordance rate (AI vs. clinician)
    │       ├── 30-day rolling concordance trend
    │       ├── Number Needed to Biopsy (NNB) tracking
    │       ├── Per-class sensitivity / specificity / PPV / NPV
    │       ├── Wilson 95% confidence intervals
    │       ├── Calibration curves (ECE + Hosmer-Lemeshow p-value)
    │       ├── Fitzpatrick equity monitoring with disparity alerts
    │       └── Discordance analysis
    │
    └── Tab 5: Settings
            ├── Model configuration (HF_MODEL_1, HF_MODEL_2)
            ├── Privacy preferences
            ├── Practice profile
            └── Brain sync settings
```

**Branding**: DA logo with "RESEARCH ONLY" badge, v0.2.0 version display, medical disclaimers on every screen.

### 5. Offline Architecture

**Service Worker Strategy**:
```
Service Worker (Workbox)
    │
    ├── Cache-First Strategy
    │       ├── CNN model weights (.onnx → WASM, ~5MB)
    │       ├── Application shell (HTML, CSS, JS)
    │       ├── WASM module (ruvector-cnn-wasm)
    │       └── Reference image embeddings (top-1000 from brain)
    │
    ├── Network-First Strategy
    │       ├── Brain search queries
    │       ├── PubMed enrichment
    │       └── Cross-practice sync
    │
    └── Background Sync
            ├── Queue diagnosis submissions for brain
            ├── Sync model updates when online
            └── Pull new reference embeddings nightly
```

**Offline Capabilities**:
- Full CNN inference (WASM, no server needed)
- ABCDE scoring (local computation)
- Grad-CAM visualization (local computation)
- HNSW search against cached reference embeddings
- Queue-and-sync for brain submissions

### 6. Training Pipeline

**Custom ViT Training** (`scripts/train-proper.py`):

```
HAM10000 Dataset (HuggingFace datasets library)
    │
    ▼
3-Layer Class Balancing
    ├── Layer 1: Focal loss alpha weights (melanoma alpha=8.0)
    ├── Layer 2: Oversampling of minority classes
    └── Layer 3: Gamma downweighting of easy examples (gamma=2.0)
    │
    ▼
ViT-Base Fine-Tuning
    ├── Pre-trained on ImageNet-21k
    ├── Fine-tuned on HAM10000 (7 classes)
    ├── Loss: Focal loss (gamma=2.0)
    ├── Hardware: M3 Max with MPS backend
    └── Model selection: by melanoma sensitivity (not overall accuracy)
    │
    ▼
stuinfla/dragnes-classifier (HuggingFace Hub)
```

**Validation Harness** (`scripts/validate-models.mjs`):

- Loads HAM10000 test split via HuggingFace datasets
- Runs each model on the test set via HuggingFace Inference API
- Reports per-class sensitivity, specificity, confusion matrix
- Used to validate Anwarkh1 (73.3% mel sens, 55.7% overall)

### 7. Multi-Practice Knowledge Sharing

**Privacy-Preserving Federation**:
```
Practice A                    pi.ruv.io Brain                  Practice B
    │                              │                               │
    ├── De-identified ────────────▶│◀──────────── De-identified ───┤
    │   embedding                  │              embedding        │
    │                              │                               │
    │                    ┌─────────┴─────────┐                     │
    │                    │  Collective Model  │                     │
    │                    │  ── ── ── ── ── ─  │                     │
    │                    │  No raw images     │                     │
    │                    │  No patient IDs    │                     │
    │                    │  No practice IDs   │                     │
    │                    │  Only: embeddings  │                     │
    │                    │  + de-id metadata  │                     │
    │                    │  + witness chains  │                     │
    │                    └─────────┬─────────┘                     │
    │                              │                               │
    │◀── Updated model ───────────┤──────────── Updated model ───▶│
    │    weights (LoRA)            │             weights (LoRA)    │
    │                              │                               │
```

**Key Privacy Guarantees**:
1. No raw images ever leave the device
2. Only 576-dim embeddings are shared (non-invertible)
3. Differential privacy (epsilon=1.0) applied to all shared data
4. Practice identifiers are stripped before brain ingestion
5. k-anonymity (k>=5) enforced on metadata attributes

### 8. Data Model

**Core Entities**:

```typescript
interface DermImage {
  id: string;                          // UUID v7 (time-ordered)
  captureTimestamp: number;            // Unix ms
  deviceModel: DermLiteModel;          // 'HUD' | 'DL5' | 'DL4' | 'DL200'
  polarizationMode: 'polarized' | 'non_polarized' | 'hybrid';
  contactMode: 'contact' | 'non_contact';
  resolution: [number, number];        // pixels
  bodyLocation: BodyLocation;          // anatomical enum
  preprocessed: boolean;
  localStorageRef: string;             // IndexedDB key (never uploaded)
}

interface LesionClassification {
  imageId: string;
  modelVersion: string;                // semver of CNN weights
  brainEpoch: number;                  // brain state at classification time
  probabilities: Record<LesionClass, number>;  // 7-class
  topClass: LesionClass;
  confidence: number;
  abcdeScores: ABCDEScores;
  sevenPointScore: number;
  menziesScore: MenziesResult;
  gradCamOverlay: Uint8Array;          // local only, never uploaded
  witnessHash: string;                 // SHAKE-256
}

interface DiagnosisRecord {
  classificationId: string;
  clinicianReview: 'confirmed' | 'corrected' | 'pending';
  correctedClass?: LesionClass;        // ground truth if corrected
  clinicalAction: 'monitor' | 'biopsy' | 'excision' | 'refer' | 'dismiss';
  histopathologyResult?: HistopathClass; // gold standard if biopsy performed
  followUpScheduled?: number;          // Unix ms
}

interface PatientEmbedding {
  // This is what gets shared with the brain -- NO PHI
  embedding: Float32Array;             // 576-dim CNN embedding
  projectedEmbedding: Float32Array;    // 128-dim for HNSW
  classLabel: LesionClass;             // 7-class
  fitzpatrickType: FitzpatrickScale;   // I-VI
  bodyLocationCategory: string;        // generalized (e.g., 'trunk', 'extremity')
  ageDecade: number;                   // 20, 30, 40, ... (bucketed)
  diameterBucket: string;              // '<3mm', '3-6mm', '6-10mm', '>10mm'
  dermoscopicFeatures: string[];       // ['globules', 'streaks', 'blue_white_veil']
  dpNoise: Float32Array;               // Laplace noise applied (epsilon=1.0)
  witnessChain: Uint8Array;            // SHAKE-256 provenance
}

interface ABCDEScores {
  asymmetry: number;         // 0-2 (0=symmetric, 2=asymmetric both axes)
  border: number;            // 0-8 (irregular border segments out of 8)
  color: number;             // 1-6 (number of colors present)
  diameter: number;          // mm (calibrated from DermLite)
  evolution: number | null;  // change score vs prior image, null if first capture
  totalScore: number;        // weighted sum
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
}
```

### 9. API Design

**Implemented SvelteKit API Routes** (server-side, HF_TOKEN never exposed to browser):

```
POST   /api/classify              Proxy to HuggingFace Model A (Anwarkh1 ViT-Base)
POST   /api/classify-v2           Proxy to HuggingFace Model B (skintaglabs SigLIP)
GET    /api/health                Health check with model info
POST   /api/analyze               Classify an image embedding (legacy)
POST   /api/feedback              Submit clinician feedback
GET    /api/similar/[id]          Find similar cases
```

**Planned RESTful + WebSocket Endpoints** (not yet implemented):

```
POST   /api/v1/analyze              Analyze a dermoscopic image (returns classification)
POST   /api/v1/analyze/batch        Batch analyze multiple images
GET    /api/v1/similar/:embeddingId Search brain for similar cases
POST   /api/v1/feedback             Submit clinician feedback/correction
GET    /api/v1/patient/:id/timeline Get lesion evolution timeline
WS     /api/v1/stream               Real-time analysis with progressive results

POST   /api/v1/brain/contribute     Share de-identified embedding with collective
GET    /api/v1/brain/search         Search collective for similar cases
GET    /api/v1/brain/literature     PubMed-enriched context for a lesion type
GET    /api/v1/brain/stats          Brain health and contribution metrics

GET    /api/v1/model/status         Current model version and performance metrics
POST   /api/v1/model/sync           Trigger model sync with brain
GET    /api/v1/model/weights        Download latest LoRA weights

GET    /api/v1/audit/trail/:id      Witness chain verification for a classification
GET    /api/v1/audit/provenance     Full provenance graph for a diagnosis
```

### 10. Security Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Security Layers                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ L1: Transport Security                             │  │
│  │     TLS 1.3 (all connections)                      │  │
│  │     Certificate pinning (mobile)                   │  │
│  │     HSTS with preloading                           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ L2: Authentication & Authorization                 │  │
│  │     OAuth 2.0 + PKCE (Google Identity)             │  │
│  │     RBAC: Admin, Clinician, Technician, Viewer     │  │
│  │     Practice-level tenancy isolation                │  │
│  │     Session timeout: 15 min inactive               │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ L3: Data Protection                                │  │
│  │     AES-256-GCM at rest (Google CMEK)              │  │
│  │     Field-level encryption for sensitive metadata  │  │
│  │     Raw images never leave device (IndexedDB)      │  │
│  │     Embeddings are non-invertible by design        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ L4: Privacy Engineering                            │  │
│  │     PII stripping (brain redaction pipeline)       │  │
│  │     Differential privacy (epsilon=1.0, Laplace)    │  │
│  │     k-anonymity (k>=5) on quasi-identifiers        │  │
│  │     Witness chain audit trail (SHAKE-256)          │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ L5: Application Security                           │  │
│  │     CSP headers (strict)                           │  │
│  │     CORS whitelist (practice domains only)         │  │
│  │     Input validation at all boundaries             │  │
│  │     Rate limiting (100 analyses/hour/practice)     │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 11. 25-Year Architecture Evolution

**Phase 1 (2026-2028): Foundation**
- Mobile-first PWA with DermLite integration
- 7-class CNN classification (HAM10000 base)
- Brain integration for collective learning
- HIPAA-compliant deployment on Google Cloud

**Phase 2 (2028-2032): Expansion**
- 50+ lesion subtypes (expanded taxonomy)
- Multi-modal input (clinical photo + dermoscopic + metadata)
- EHR integration (Epic FHIR, Cerner, athenahealth)
- Teledermatology workflow (store-and-forward)
- Whole-body photography with lesion change detection

**Phase 3 (2032-2040): Advanced Imaging**
- Confocal microscopy integration (RCM)
- Optical coherence tomography (OCT) fusion
- Multispectral imaging analysis
- 3D lesion reconstruction and volumetric analysis
- Genomic risk score integration (GWAS SNP panels)

**Phase 4 (2040-2051): Autonomous Intelligence**
- AR-guided biopsy and surgery overlay
- Continuous monitoring via smart wearables and ambient sensors
- Brain-computer interface for clinical gestalt augmentation
- Self-evolving models that discover new lesion subtypes
- Global elimination of late-stage melanoma detection
