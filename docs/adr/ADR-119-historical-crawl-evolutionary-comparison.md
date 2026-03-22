# ADR-119: Historical Common Crawl Evolutionary Comparison

**Status**: Accepted
**Date**: 2026-03-22
**Author**: Claude (ruvnet)
**Related**: ADR-094 (Shared Web Memory), ADR-115 (Common Crawl Compression), ADR-118 (Cost-Effective Crawl)

## Context

The pi.ruv.io brain ingests current Common Crawl data (ADR-115 Phase 1), but medical knowledge evolves over time. Understanding HOW dermatology content changed across years enables:
- Detecting when new treatment protocols emerged
- Tracking consensus formation on diagnostic criteria
- Identifying knowledge fragmentation (narrative fractures)
- Measuring the pace of AI adoption in dermatology

Common Crawl maintains monthly crawl archives from 2008 to present, each with its own CDX index. By querying the same medical domains across multiple crawl snapshots, we can build temporal knowledge evolution graphs.

## Decision

Implement a historical crawl importer that queries the same domains across quarterly Common Crawl snapshots (2020-2026), computes embedding drift between temporal versions, and stores WebPageDelta chains in the brain for evolutionary analysis.

### Architecture

```
Quarterly Crawl Snapshots (24 crawls, 2020-2026)
       │
       ▼  CDX Query: same domains across each crawl
  ┌──────────────────────────────────────┐
  │  For each crawl snapshot:            │
  │  1. Query CDX for target domains     │
  │  2. Range-GET page content           │
  │  3. Extract text, embed (128-dim)    │
  │  4. Compare to previous snapshot     │
  │  5. Compute WebPageDelta             │
  │  6. Store with crawl_timestamp       │
  └──────────┬───────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │  Evolutionary Analysis:              │
  │  • Embedding drift per URL over time │
  │  • Concept birth detection           │
  │  • Consensus formation tracking      │
  │  • Narrative fracture via MinCut     │
  │  • Lyapunov stability per domain     │
  └──────────────────────────────────────┘
```

### Target Domains (Medical/Dermatology)

| Domain | Content |
|--------|---------|
| aad.org | American Academy of Dermatology — guidelines, patient info |
| dermnetnz.org | DermNet NZ — comprehensive dermatology reference |
| skincancer.org | Skin Cancer Foundation — screening, prevention |
| cancer.org | American Cancer Society — cancer statistics, guidelines |
| ncbi.nlm.nih.gov | PubMed/NCBI — research abstracts |
| who.int | WHO — global health guidance |
| melanoma.org | Melanoma Research Foundation |

### Crawl Schedule (Quarterly Sampling)

24 crawl indices from 2020 Q1 to 2026 Q1:
CC-MAIN-2020-16, CC-MAIN-2020-34, CC-MAIN-2020-50,
CC-MAIN-2021-10, CC-MAIN-2021-25, CC-MAIN-2021-43,
CC-MAIN-2022-05, CC-MAIN-2022-21, CC-MAIN-2022-40,
CC-MAIN-2023-06, CC-MAIN-2023-23, CC-MAIN-2023-40,
CC-MAIN-2024-10, CC-MAIN-2024-26, CC-MAIN-2024-42,
CC-MAIN-2025-05, CC-MAIN-2025-22, CC-MAIN-2025-40,
CC-MAIN-2026-06, CC-MAIN-2026-08

### Cost

| Item | Cost |
|------|------|
| CDX queries (24 crawls x 7 domains) | $0 |
| Page extraction (~200 pages/crawl) | $0 (free CC egress) |
| Cloud Run compute | ~$5-10 one-time |
| Firestore storage | ~$2-5 |
| **Total** | **~$7-15 one-time** |

### Outputs

1. `GET /v1/web/evolution?url=X` — temporal delta history for a URL
2. `GET /v1/web/drift?topic=X&months=N` — drift score and trend
3. `GET /v1/web/concepts/births?since=2020` — newly emerged concepts
4. Brain memories tagged with `crawl_index` for temporal queries

## Acceptance Criteria

1. Import >=100 pages across >=4 quarterly crawl snapshots
2. Compute WebPageDelta with embedding_drift for each URL across time
3. Store temporal chain in brain with crawl_timestamp metadata
4. Verify search returns time-ordered results for evolved content
5. Total cost <= $15

## Consequences

### Positive
- Brain gains historical context — not just current knowledge
- Drift detection shows which medical topics are evolving fastest
- DrAgnes can reference "how guidelines changed over time"
- Foundation for concept birth detection and narrative tracking

### Negative
- Historical CC CDX can be slow (older indices, less maintained)
- Some URLs may not appear in every crawl snapshot
- Content extraction quality varies across crawl periods
