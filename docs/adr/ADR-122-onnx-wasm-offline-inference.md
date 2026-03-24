Updated: 2026-03-24 | Version 1.0.0
Created: 2026-03-24

# ADR-122: ONNX/WASM Offline Inference -- Zero Network Dependency

## Status: IMPLEMENTED -- ONNX FP32 + INT8 Exported, Browser Integration Pending | Last Updated: 2026-03-24 12:00 EST
**Implementation Note**: No ONNX Runtime Web integration exists yet. The ONNX FP32 model (327MB) and INT8 model (83MB) have been exported (scripts/dragnes-onnx/ and scripts/dragnes-onnx-int8/) but ConvInteger compatibility issue blocks browser inference. No Service Worker model caching implemented. Classification still relies on HuggingFace Inference API or local trained-weights fallback.

## Context

Dr. Agnes currently relies on the HuggingFace Inference API for its primary classification
pathway (`hf-classifier.ts` calls `https://router.huggingface.co/hf-inference/models/...`).
The `classifier.ts` ensemble blends HF API results with local trained-weights and rule-based
analysis, but the HF component carries the highest weight (50-70%) in all online strategies.

This creates four concrete problems:

1. **No offline capability.** Rural clinics, field deployments, and areas with intermittent
   connectivity cannot use the classification pipeline at all. The offline fallback
   (60% trained-weights + 40% rule-based) discards the most accurate model layer entirely.

2. **High latency.** HF API calls take 2-15 seconds per image depending on model cold-start
   and queue depth. For multi-image capture (3-5 images per lesion), total wait exceeds 30
   seconds. Clinicians abandon the tool.

3. **API quota and rate limits.** Free-tier HF inference is throttled. Production use at any
   scale requires a paid Inference Endpoint ($0.06/hr minimum), and rate-limiting during
   peak hours degrades availability.

4. **Privacy.** Medical images leave the device and transit through HuggingFace infrastructure.
   Even with HTTPS, this violates the data minimization principle for medical data and
   complicates HIPAA/GDPR compliance. The `privacy.ts` module strips metadata, but the
   pixel data itself is still transmitted.

The v2 combined model (ViT-Base, 85.8M parameters) is a fine-tuned `google/vit-base-patch16-224`
trained on HAM10000 + ISIC 2019 with 98.2% melanoma sensitivity. It is saved as PyTorch
safetensors. RuVector already ships a `@ruvector/cnn` WASM module (`ruvector_cnn_wasm.d.ts`)
with SIMD128-accelerated INT8 kernels (`int8_wasm.rs`, `simd/wasm.rs`), MobileNet-V3 backbone,
and `WasmCnnEmbedder` -- but this is an embedder, not a classifier.

## Decision

**Option 4: Hybrid -- ONNX Runtime Web for offline, server for connected speed.**

### Options Evaluated

| # | Option | Bundle Size | Latency | Offline | Dev Effort | Privacy |
|---|--------|-------------|---------|---------|------------|---------|
| 1 | ONNX Runtime Web (browser WASM) | ~83MB INT8 / ~21MB INT4 | 100-400ms | Full | Medium | Full |
| 2 | @ruvector/cnn Rust WASM custom | ~8-15MB | 50-200ms | Full | Very High | Full |
| 3 | ONNX on Vercel serverless | 0 (server) | 500-2000ms | None | Low | Partial |
| 4 | **Hybrid (1 + server)** | ~83MB cached | **<200ms local** | **Full** | Medium | **Full offline** |

**Why not Option 1 alone?** Pure client-side works but the 83MB initial download is hostile
on the exact low-bandwidth networks where offline matters most. The hybrid caches
progressively: first classification uses the server if available, while the WASM model
downloads in the background via Service Worker. Subsequent uses are fully offline.

**Why not Option 2?** The `@ruvector/cnn` WASM module provides SIMD-accelerated CNN
primitives (conv2d, batch norm, pooling, INT8 dot products) but is an embedder architecture,
not a ViT classifier. Building a ViT-Base forward pass from scratch in Rust would require
implementing multi-head self-attention, patch embedding, layer norm, and the classification
head -- roughly 3-4 weeks of work to match what ONNX Runtime Web provides out of the box.
However, `@ruvector/cnn`'s `SimdOps` and `LayerOps` remain useful for pre/post-processing.

**Why not Option 3?** Eliminates HF dependency but still requires network. Does not solve
the core problem.

## Architecture

### Inference Bounded Context (DDD)

```
+------------------------------------------------------------------+
|                    Inference Bounded Context                      |
|                                                                   |
|  +---------------------------+  +-----------------------------+  |
|  |   InferenceStrategyPort   |  |    ModelRegistryPort        |  |
|  |   (interface)             |  |    (interface)              |  |
|  |                           |  |                             |  |
|  |  classify(image): Result  |  |  getModel(id): ModelAsset   |  |
|  |  isAvailable(): boolean   |  |  cacheStatus(): CacheState  |  |
|  |  latencyEstimate(): ms    |  |  downloadProgress(): 0-1    |  |
|  +------+--------+-----------+  +-----------------------------+  |
|         |        |                                                |
|  +------v--+  +--v-----------+  +-----------------------------+  |
|  | Online   |  | Offline     |  |   InferenceOrchestrator     |  |
|  | Strategy |  | Strategy    |  |                             |  |
|  |          |  |             |  |  Selects strategy based on: |  |
|  | - Server |  | - ONNX WASM |  |   1. navigator.onLine       |  |
|  |   proxy  |  | - Cached    |  |   2. Model cache state      |  |
|  | - HF API |  |   model     |  |   3. Last server latency    |  |
|  | - Custom |  | - INT8      |  |   4. User preference        |  |
|  |   ViT    |  |   quantized |  +-----------------------------+  |
|  +----------+  +-------------+                                    |
|                                                                   |
|  +-----------------------------+  +----------------------------+ |
|  |   PreprocessingService      |  |  PostprocessingService     | |
|  |   (shared by both)          |  |  (shared by both)          | |
|  |                             |  |                            | |
|  |  - Resize to 224x224       |  |  - Label mapping           | |
|  |  - ImageNet normalization   |  |  - Probability calibration | |
|  |  - NCHW tensor conversion   |  |  - Ensemble blending       | |
|  |  - @ruvector/cnn SimdOps    |  |  - ABCDE score integration | |
|  +-----------------------------+  +----------------------------+ |
+------------------------------------------------------------------+

            |                               |
            v                               v
+---------------------+       +------------------------+
| Classification      |       | Ensemble Bounded       |
| Result Aggregate    |       | Context (existing)     |
|                     |       |                        |
| - inferenceSource:  |       | - trained-weights.ts   |
|   "server"|"wasm"   |       | - image-analysis.ts    |
| - latencyMs         |       | - abcde.ts             |
| - modelVersion      |       | - clinical-baselines   |
| - offlineCapable    |       +------------------------+
+---------------------+
```

### Data Flow

```
User captures image
       |
       v
  Quality Gate (ADR-121)
       |
       v
  InferenceOrchestrator.classify(imageData)
       |
       +--- navigator.onLine && !userPrefersOffline?
       |         |
       |    YES  |  NO (or server unreachable)
       |         |
       v         v
  OnlineStrategy    OfflineStrategy
  - POST /api/     - Load ONNX from Cache API
    classify-local  - ort.InferenceSession.create()
  - 500-2000ms     - session.run({pixel_values: tensor})
       |           - 100-400ms (INT8 on WASM SIMD)
       |                |
       +-------+--------+
               |
               v
    PostprocessingService
    - Map logits to 7-class probabilities
    - Blend with trained-weights (existing)
    - Blend with rule-based ABCDE (existing)
    - Apply demographic adjustment
               |
               v
    ClassificationResult
    { source: "server"|"wasm", latencyMs, ... }
```

### Background Model Caching (Service Worker)

```
First visit (online):
  1. App loads, registers ServiceWorker
  2. SW starts background fetch of ONNX model (~83MB INT8)
  3. Classification uses OnlineStrategy (server)
  4. SW stores model in Cache API with version key
  5. UI shows progress: "Downloading offline model... 47%"

Subsequent visits:
  1. SW intercepts model request, serves from cache
  2. OfflineStrategy available immediately
  3. SW checks for model updates (ETag/version) periodically

Offline visit:
  1. SW serves cached app shell + model
  2. OfflineStrategy runs entirely in browser
  3. Results queued for brain-sync when back online (existing offline-queue.ts)
```

## Bundle Size Analysis

| Component | FP32 | INT8 (chosen) | INT4 (future) |
|-----------|------|---------------|---------------|
| ViT-Base weights (85.8M params) | 327MB | 82MB | 21MB |
| ONNX Runtime Web WASM | 8.2MB | 8.2MB | 8.2MB |
| ONNX Runtime Web JS | 0.4MB | 0.4MB | 0.4MB |
| **Total download** | **336MB** | **91MB** | **30MB** |
| Gzip compressed estimate | ~280MB | ~72MB | ~24MB |

INT8 is the target for v1. The 82MB model downloads once and is cached indefinitely by the
Service Worker. The ONNX Runtime WASM (8.6MB) loads with the app. INT4 is a future
optimization (~30MB total) pending accuracy validation on medical images.

## Performance Targets

| Metric | Target | Basis |
|--------|--------|-------|
| WASM inference latency (modern phone, INT8) | <200ms | ViT-Base INT8 benchmarks on Pixel 7 |
| WASM inference latency (low-end phone, INT8) | <800ms | Estimated 4x slower than flagship |
| WASM inference latency (desktop, INT8) | <100ms | M3 Max benchmark baseline |
| Model load from Cache API | <500ms | 82MB from local storage |
| First-run model download (4G) | ~3 min | 72MB compressed at 3 Mbps |
| First-run model download (3G) | ~15 min | Background, non-blocking |
| Memory usage during inference | <400MB | ONNX Runtime Web with INT8 |

## Privacy Impact

| Concern | Current (HF API) | After (Hybrid) |
|---------|-------------------|-----------------|
| Image leaves device | Always (online) | Never (when WASM cached) |
| Third-party data processing | HuggingFace infra | None (offline) / own server (online) |
| HIPAA compliance | Requires BAA with HF | Compliant by default (offline) |
| GDPR data minimization | Violated | Satisfied (no transfer) |
| Network metadata exposure | API call observable | No network call (offline) |
| Model IP protection | N/A (public HF model) | ONNX model in Cache API (inspectable) |

The offline-first architecture makes the default path privacy-compliant. The online path
routes through our own infrastructure, not third-party APIs.

## Implementation Plan

### Phase 1: ONNX Model Export and Quantization (2-3 days)

1. Export PyTorch ViT-Base to ONNX format via `torch.onnx.export()`
2. Validate ONNX model output matches PyTorch within tolerance (max abs diff < 1e-5)
3. Quantize to INT8 using `onnxruntime.quantization.quantize_dynamic()`
4. Validate INT8 model accuracy on HAM10000 test set (target: <0.5% accuracy drop)
5. Store ONNX artifacts in `scripts/dragnes-classifier-v2/onnx/`

### Phase 2: ONNX Runtime Web Integration (3-4 days)

1. Add `onnxruntime-web` dependency to `examples/dragnes/package.json`
2. Create `src/lib/dragnes/onnx-inference.ts` implementing `InferenceStrategyPort`
3. Implement tensor preprocessing using existing `preprocessing.ts` + `@ruvector/cnn SimdOps`
4. Wire `OfflineStrategy` into `classifier.ts` ensemble as a new inference source
5. Add `inferenceSource` field to `ClassificationResult` type

### Phase 3: Service Worker Model Caching (2-3 days)

1. Create Service Worker with Cache API model management
2. Implement background fetch with progress reporting
3. Add model version checking (ETag-based cache invalidation)
4. Wire download progress into UI (settings panel or first-run dialog)
5. Handle cache eviction gracefully (re-download prompt)

### Phase 4: InferenceOrchestrator and Strategy Selection (1-2 days)

1. Implement `InferenceOrchestrator` with strategy selection logic
2. Auto-detect: online + model cached = prefer WASM (faster, private); online + no cache = server
3. Add user preference toggle: "Always use on-device analysis"
4. Add `navigator.connection` awareness (slow connection = prefer cached WASM)
5. Emit telemetry: which strategy used, latency, for ensemble weight tuning

### Phase 5: Validation and Accuracy Gating (2-3 days)

1. Run full HAM10000 test split through ONNX INT8 WASM pipeline
2. Compare per-class sensitivity/specificity against server pipeline
3. Gate: melanoma sensitivity must remain >= 97% (current: 98.2%)
4. Gate: overall accuracy must remain >= 90%
5. If INT8 degrades melanoma sensitivity, fall back to FP16 (164MB, still viable)

**Total estimated effort: 10-15 days.**

## Consequences

### Positive

- Full offline classification with the same model that powers the online pipeline
- Sub-200ms inference on modern devices (10-75x faster than HF API)
- Medical images never leave the device when using WASM path
- Eliminates HF API costs, quota limits, and rate throttling
- Progressive enhancement: online-first on first visit, offline-capable after caching
- Reuses `@ruvector/cnn` SIMD ops for preprocessing

### Negative

- 82MB cached model consumes device storage (acceptable for medical tool)
- ONNX Runtime Web WASM adds 8.6MB to initial app bundle
- Two inference paths to maintain (server and WASM) increases test surface
- INT8 quantization may degrade accuracy on edge cases (mitigated by Phase 5 gates)
- Service Worker caching adds complexity to deployment and cache invalidation
- Low-end phones (pre-2020, <3GB RAM) may not have enough memory for inference

### Risks

- **Critical:** INT8 quantization reduces melanoma sensitivity below 97%.
  Mitigation: accuracy gates in Phase 5; fall back to FP16 or FP32 if needed.
- **High:** 82MB download fails on poor connections.
  Mitigation: resumable downloads via Service Worker; chunked fetch with retry.
- **Medium:** WASM SIMD not supported on older browsers.
  Mitigation: feature-detect SIMD; fall back to non-SIMD WASM (2-3x slower).
- **Low:** Cache API storage limit hit. Mitigation: check `navigator.storage.estimate()`.

## References

- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- `@ruvector/cnn` WASM: `npm/packages/ruvector-cnn/ruvector_cnn_wasm.d.ts`
- Current classifier: `examples/dragnes/src/lib/dragnes/classifier.ts`
- ADR-117: DrAgnes platform architecture; ADR-121: Image capture and quality gating

## Author
Stuart Kerr + Claude Flow
