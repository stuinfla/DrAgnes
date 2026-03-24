Updated: 2026-03-24 14:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-126: Pi-Brain Collective Intelligence -- Every Practice Makes Every Other Practice Better

## Status: PROPOSED

## Context

Dr. Agnes currently learns nothing from its own classifications. Every deployment is an island. A dermatologist in Miami who biopsies a lesion that Dr. Agnes flagged as suspicious never feeds the outcome back into the system. A GP in rural India using Dr. Agnes gains nothing from the Miami dermatologist's clinical experience. Every practice starts from scratch, limited to the frozen training set.

This is a solved problem architecturally. Pi-brain (pi.ruv.io) is RuVector's collective intelligence layer and it already exists:

- **1,807+ memories** across multiple knowledge domains
- **Differential privacy** (epsilon=1.0) with calibrated Laplacian noise
- **SHAKE-256 witness chains** for cryptographic provenance on every memory
- **350K+ graph edges** enabling similarity traversal and knowledge clustering
- **HNSW vector search** for sub-millisecond nearest-neighbor retrieval on feature embeddings

The infrastructure is designed but NOT connected in production. Dr. Agnes writes classifications to local state and discards them. Pi-brain sits idle with respect to dermatology. Connecting them means that when a clinician records a biopsy outcome ("that lesion Dr. Agnes scored 5.2 TDS turned out to be a Spitz nevus"), that outcome -- anonymized, noise-added, stripped of all patient identifiers -- flows into the collective graph. Every other Dr. Agnes instance in the world benefits the next time it encounters a similar lesion.

**The core insight: dermatology is a pattern-matching discipline.** The more patterns the system has seen with confirmed outcomes, the better it gets. No single practice sees enough volume to build this corpus alone. Collective intelligence is the only way to build it without centralized data collection that violates patient privacy.

## Decision

Connect Dr. Agnes to pi-brain for anonymized, differentially private case sharing. Every classification with a confirmed clinical outcome becomes a collective learning event.

### Data Flow Architecture

```
 Practice A (Miami)                          Practice B (Rural India)
 ========================                    ========================
 1. Classify lesion                          5. Classify new lesion
 2. Clinician records outcome                6. Query pi-brain:
    (biopsy: Spitz nevus)                       "What happened to lesions
 3. Anonymize:                                   with similar features?"
    - Strip EXIF metadata                    7. Receive: "12 similar cases,
    - Drop raw image                            8 benign, 3 dysplastic,
    - Add DP noise to probabilities             1 melanoma in situ"
    - Reduce demographics to                 8. Clinician sees enriched
      age-decade + sex only                     context alongside TDS score
 4. Push to pi-brain via
    brain_share() API
         |
         v
 ┌──────────────────────────────────────────────────────┐
 │                    pi.ruv.io Brain                    │
 │                                                      │
 │  ┌────────────────┐  ┌───────────────┐  ┌─────────┐ │
 │  │ HNSW Index     │  │ Knowledge     │  │ Witness │ │
 │  │ (576-dim CNN   │  │ Graph         │  │ Chain   │ │
 │  │  embeddings)   │  │ (350K+ edges) │  │ (SHAKE) │ │
 │  └────────────────┘  └───────────────┘  └─────────┘ │
 │                                                      │
 │  Differential Privacy: epsilon=1.0 Laplacian noise   │
 │  PII: Stripped at source, verified at ingestion      │
 └──────────────────────────────────────────────────────┘
```

### Domain-Driven Design: Collective Intelligence Bounded Context

```
┌─────────────────────────────────────────────────────────────────┐
│              Collective Intelligence Bounded Context             │
│                                                                 │
│  ┌─────────────────────┐                                        │
│  │   FeedbackLoop      │  Aggregates:                           │
│  │   (Subdomain)       │  - CaseOutcome (biopsy result + TDS)  │
│  │                     │  - OutcomeFeedback (clinician input)   │
│  │                     │  - FeedbackWindow (time-bounded batch) │
│  └──────────┬──────────┘                                        │
│             │ emits CaseOutcomeRecorded event                   │
│             v                                                   │
│  ┌─────────────────────┐                                        │
│  │ AnonymizationPipeline│  Services:                            │
│  │   (Subdomain)        │  - ExifStripper (removes all EXIF)   │
│  │                      │  - DPNoiseInjector (eps=1.0)         │
│  │                      │  - DemographicReducer (age-decade)   │
│  │                      │  - EmbeddingExtractor (576-dim, non- │
│  │                      │    invertible CNN features only)      │
│  │                      │  - SafeHarborValidator (18 HIPAA     │
│  │                      │    identifiers checked pre-share)     │
│  └──────────┬───────────┘                                       │
│             │ emits AnonymizedCaseReady event                   │
│             v                                                   │
│  ┌──────────────────────┐                                       │
│  │  KnowledgeRetrieval  │  Services:                            │
│  │   (Subdomain)        │  - BrainShareClient (push to pi.io)  │
│  │                      │  - SimilarCaseQuery (HNSW k-NN on    │
│  │                      │    576-dim embeddings)                │
│  │                      │  - OutcomeAggregator (summarize       │
│  │                      │    outcomes for similar cases)        │
│  │                      │  - ProvenanceVerifier (validate       │
│  │                      │    witness chain on retrieved cases)  │
│  └──────────────────────┘                                       │
│                                                                 │
│  Anti-Corruption Layer:                                         │
│  - Maps between DrAgnes domain (TDS, ABCDE, HAM10000 taxonomy) │
│    and pi-brain domain (memories, graph edges, embeddings)      │
│  - Ensures pi-brain schema changes do not break DrAgnes         │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Shared (and What Does Not)

| Data Element | Shared? | Format | Notes |
|---|---|---|---|
| Raw dermoscopic image | NEVER | -- | Stays on device only |
| CNN feature embedding | Yes | 576-dim float32 vector | Non-invertible; 261:1 dimensionality reduction from image |
| Classification probabilities | Yes | 7-class + DP noise | Laplacian noise (eps=1.0) added before sharing |
| TDS score | Yes | Float, noised | +/- 0.3 noise added |
| ABCDE subscores | Yes | 5x float, noised | Individual subscores with calibrated noise |
| Biopsy/clinical outcome | Yes | HAM10000 taxonomy label | Confirmed diagnosis only, no free-text pathology |
| Patient age | Yes | Decade bucket only | "40s", "50s" -- never exact age |
| Patient sex | Yes | M/F/Other | No additional demographics |
| Patient name/ID/MRN | NEVER | -- | Stripped at source |
| Practice/clinician ID | NEVER | -- | Replaced with random pseudonym per session |
| GPS/location | NEVER | -- | EXIF stripped; no geolocation shared |
| Device identifiers | NEVER | -- | No IMEI, MAC, or device fingerprint |

### Implementation Plan

**Phase 1: Feedback Capture (Week 1-2)**
- Add outcome recording UI in DrAgnesPanel: "What was the clinical outcome?"
- Options: biopsy-confirmed diagnosis (HAM10000 labels), clinical follow-up (stable/changed), lost to follow-up
- Store outcome locally in IndexedDB alongside the original classification
- No data leaves the device in this phase

**Phase 2: Anonymization Pipeline (Week 3-4)**
- Implement SafeHarborValidator: check all 18 HIPAA Safe Harbor identifiers are absent
- Implement DPNoiseInjector: Laplacian mechanism on probabilities and TDS
- Implement DemographicReducer: age to decade, strip everything else
- Implement EmbeddingExtractor: extract 576-dim feature vector from CNN penultimate layer
- Unit tests with synthetic cases to verify no PII leakage

**Phase 3: Brain Integration (Week 5-6)**
- Connect to pi-brain via `brain_share()` with category `dermatology`
- Push anonymized case bundles (embedding + noised probabilities + outcome + demographics)
- Implement `brain_search()` query for similar cases at classification time
- Display "Similar cases from the collective" panel in UI

**Phase 4: Knowledge Retrieval (Week 7-8)**
- HNSW k-NN search on 576-dim embeddings to find the 10 most similar cases with outcomes
- Aggregate outcomes: "Of 10 similar cases, 7 were benign, 2 dysplastic nevi, 1 melanoma"
- Display outcome distribution as supplementary context (NOT as a diagnosis)
- Witness chain verification on retrieved cases for provenance

### API Contract

```typescript
// Push: anonymized case to pi-brain
interface AnonymizedCase {
  embedding: Float32Array;     // 576-dim CNN features
  probabilities: number[];     // 7-class, DP-noised
  tdsScore: number;            // noised +/- 0.3
  abcdeScores: ABCDENoised;   // individual subscores, noised
  outcome: HAM10000Label;      // biopsy-confirmed or clinical
  outcomeConfidence: 'biopsy' | 'clinical' | 'follow-up';
  ageBucket: string;           // "40s", "50s", etc.
  sex: 'M' | 'F' | 'Other';
  witnessHash: string;         // SHAKE-256 of classification state
}

// Pull: similar cases from pi-brain
interface SimilarCaseResult {
  cases: Array<{
    similarity: number;        // cosine distance from HNSW
    outcome: HAM10000Label;
    outcomeConfidence: string;
    tdsScore: number;          // noised
    witnessHash: string;
  }>;
  outcomeDistribution: Record<HAM10000Label, number>;
  totalCasesInCorpus: number;
}
```

## Risks and Mitigations

### Risk 1: Privacy Breach -- Re-identification from Feature Vectors

**Severity**: Critical
**Likelihood**: Low (CNN embeddings are non-invertible by design)
**Mitigation**:
- 576-dim embeddings represent a 261:1 dimensionality reduction; reconstruction is computationally infeasible
- Differential privacy noise (eps=1.0) further degrades any reconstruction attempt
- No raw images ever leave the device
- Safe Harbor validation runs before every share; if any of the 18 identifiers are detected, the share is blocked
- Annual third-party privacy audit once the system is in production

### Risk 2: Regulatory Challenge -- FDA/HIPAA/GDPR Implications

**Severity**: High
**Likelihood**: Medium
**Mitigation**:
- HIPAA Safe Harbor method satisfies the de-identification standard (45 CFR 164.514(b))
- Differential privacy provides a mathematically provable privacy guarantee (epsilon=1.0 is conservative)
- No raw images or identifiable data cross organizational boundaries
- Witness chain provides full audit trail for regulatory inspection
- Legal review required before Phase 3 goes to production
- GDPR: data minimization principle satisfied (only what is needed for similarity search)

### Risk 3: Poisoned Data -- Malicious or Incorrect Outcomes

**Severity**: Medium
**Likelihood**: Low
**Mitigation**:
- Outcome confidence levels distinguish biopsy-confirmed from clinical impression
- Statistical outlier detection: flag cases where the outcome contradicts the embedding cluster consensus
- Reputation scoring per pseudonymized practice (high-quality feedback weighted higher)
- Manual review queue for cases flagged as statistical outliers
- Witness chain makes every contribution traceable and revocable

### Risk 4: Cold Start -- Not Enough Cases for Useful Similarity

**Severity**: Medium
**Likelihood**: High (initially)
**Mitigation**:
- Seed the brain with published dermoscopy datasets (HAM10000, ISIC Archive) as the initial corpus
- Clearly label seeded cases vs. real-world feedback in the UI
- Set minimum corpus threshold: require at least 50 similar cases before displaying outcome distribution
- Below threshold, display "Not enough collective data yet" rather than misleading statistics

## Consequences

**Positive:**
- Every confirmed clinical outcome improves every Dr. Agnes deployment worldwide
- Rural practices with low case volume benefit from urban specialist feedback
- Creates a dermoscopy knowledge graph that grows with use, unlike static training sets
- Witness chains enable regulatory-grade audit trails
- No single point of data collection; privacy preserved by design

**Negative:**
- Adds network dependency for the collective intelligence features (classification still works offline)
- Differential privacy noise reduces the precision of shared data (acceptable tradeoff)
- Cold start period where collective data is sparse and less useful
- Ongoing operational cost for pi-brain infrastructure
- Requires legal review in each jurisdiction before enabling data sharing

## References

- ADR-117: DrAgnes Dermatology Intelligence Platform (foundational architecture)
- ADR-118: DrAgnes Production Validation (clinical accuracy targets)
- ADR-119: Consumer Skin Screening (non-clinical use case)
- HIPAA Safe Harbor Method: 45 CFR 164.514(b)(2)
- Dwork, C. "Differential Privacy" (ICALP 2006) -- epsilon-differential privacy framework
- Pi-brain architecture: pi.ruv.io/v1/status (1,807+ memories, SHAKE-256 witness chains)
- HAM10000 dataset: Tschandl et al., "The HAM10000 dataset" (2018)
