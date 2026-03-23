<script lang="ts">
	let pipelineOpen: boolean = $state(true);
	let featuresOpen: boolean = $state(true);
	let dataOpen: boolean = $state(false);
	let benchmarksOpen: boolean = $state(false);
	let safetyOpen: boolean = $state(false);
	let limitationsOpen: boolean = $state(true);
</script>

<div class="space-y-4 p-4 max-h-[calc(100vh-200px)] overflow-y-auto">
	<div class="mb-2">
		<h2 class="text-sm font-bold text-gray-200">How DrAgnes Works</h2>
		<p class="text-[10px] text-gray-500 mt-1">
			Full transparency into the classification pipeline, datasets, and clinical scoring logic.
		</p>
	</div>

	<!-- Section 1: Classification Pipeline -->
	<section class="rounded-xl border border-gray-800 bg-gray-900/50">
		<button
			onclick={() => (pipelineOpen = !pipelineOpen)}
			class="flex w-full items-center justify-between p-4 text-left"
		>
			<h3 class="text-sm font-semibold text-teal-400 flex items-center gap-2">
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
				</svg>
				Classification Pipeline
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {pipelineOpen ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if pipelineOpen}
			<div class="px-4 pb-4">
				<p class="text-xs text-gray-400 mb-3">
					DrAgnes uses a 4-layer ensemble classification system. Each layer catches different failure modes:
				</p>
				<div class="space-y-2">
					<div class="flex items-start gap-2 rounded-lg bg-gray-800/50 p-3">
						<span class="text-xs font-mono text-teal-400 w-6 flex-shrink-0">L1</span>
						<div>
							<p class="text-xs font-medium text-gray-200">Dual ViT Neural Network (50%)</p>
							<p class="text-[10px] text-gray-500 mt-0.5">
								Two independently-trained Vision Transformers analyze pixel-level patterns.
								Anwarkh1 (85.8M params, HAM10000-finetuned) + SigLIP (400M params, SkinTag Labs).
								When both models agree, confidence is high. When they disagree, the case is flagged
								for clinical review.
							</p>
						</div>
					</div>
					<div class="flex items-start gap-2 rounded-lg bg-gray-800/50 p-3">
						<span class="text-xs font-mono text-amber-400 w-6 flex-shrink-0">L2</span>
						<div>
							<p class="text-xs font-medium text-gray-200">Literature-Derived Classifier (30%)</p>
							<p class="text-[10px] text-gray-500 mt-0.5">
								20-feature logistic regression with weights derived from published dermoscopy literature:
								Stolz 1994 (ABCD rule), Argenziano 1998 (7-point checklist), DermNet NZ clinical guides.
								Operates entirely locally -- no network required.
							</p>
						</div>
					</div>
					<div class="flex items-start gap-2 rounded-lg bg-gray-800/50 p-3">
						<span class="text-xs font-mono text-orange-400 w-6 flex-shrink-0">L3</span>
						<div>
							<p class="text-xs font-medium text-gray-200">Rule-Based Clinical Scoring (20%)</p>
							<p class="text-[10px] text-gray-500 mt-0.5">
								TDS formula (A x 1.3 + B x 0.1 + C x 0.5 + D x 0.5), 7-point checklist
								(score of 3 or more triggers biopsy recommendation), melanoma safety gate
								(2+ concurrent suspicious indicators forces minimum probability floor).
							</p>
						</div>
					</div>
					<div class="flex items-start gap-2 rounded-lg bg-gray-800/50 p-3">
						<span class="text-xs font-mono text-purple-400 w-6 flex-shrink-0">L4</span>
						<div>
							<p class="text-xs font-medium text-gray-200">Bayesian Demographic Adjustment</p>
							<p class="text-[10px] text-gray-500 mt-0.5">
								Adjusts probabilities based on patient age, sex, and lesion body site using
								HAM10000 epidemiological data (Tschandl et al. 2018). For example, melanoma
								probability increases with age and trunk location in males.
							</p>
						</div>
					</div>
				</div>

				<div class="mt-3 rounded-lg bg-gray-800/30 p-2">
					<p class="text-[10px] text-gray-500">
						<strong class="text-gray-400">Ensemble fallback modes:</strong>
						Dual-ViT online = L1(50%) + L2(30%) + L3(20%).
						Single-ViT online = L1(60%) + L2(25%) + L3(15%).
						Offline = L2(60%) + L3(40%). L4 applies post-ensemble when demographics are provided.
					</p>
				</div>
			</div>
		{/if}
	</section>

	<!-- Section 2: Image Analysis Features -->
	<section class="rounded-xl border border-gray-800 bg-gray-900/50">
		<button
			onclick={() => (featuresOpen = !featuresOpen)}
			class="flex w-full items-center justify-between p-4 text-left"
		>
			<h3 class="text-sm font-semibold text-teal-400 flex items-center gap-2">
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
				</svg>
				Image Analysis -- What We Measure
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {featuresOpen ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if featuresOpen}
			<div class="px-4 pb-4">
				<div class="grid grid-cols-2 gap-2 text-[10px]">
					<div class="rounded-lg bg-gray-800/50 p-2">
						<p class="font-medium text-gray-300">Lesion Segmentation</p>
						<p class="text-gray-500">LAB color space + Otsu thresholding (Otsu 1979). Isolates lesion from surrounding skin for all downstream analysis.</p>
					</div>
					<div class="rounded-lg bg-gray-800/50 p-2">
						<p class="font-medium text-gray-300">Asymmetry (0-2)</p>
						<p class="text-gray-500">Principal axis folding -- ABCD Rule (Stolz 1994). TDS weight: 1.3x. Melanomas are asymmetric in >90% of cases.</p>
					</div>
					<div class="rounded-lg bg-gray-800/50 p-2">
						<p class="font-medium text-gray-300">Border (0-8)</p>
						<p class="text-gray-500">8-octant border irregularity analysis. TDS weight: 0.1x per segment. Sharp cutoffs = higher suspicion.</p>
					</div>
					<div class="rounded-lg bg-gray-800/50 p-2">
						<p class="font-medium text-gray-300">Color (1-6)</p>
						<p class="text-gray-500">K-means++ in LAB space, 6 dermoscopic colors (white, red, light-brown, dark-brown, blue-gray, black). TDS weight: 0.5x.</p>
					</div>
					<div class="rounded-lg bg-gray-800/50 p-2">
						<p class="font-medium text-gray-300">Texture (GLCM)</p>
						<p class="text-gray-500">Contrast, homogeneity, entropy via Gray-Level Co-occurrence Matrix (Haralick 1973). Low homogeneity = more disordered texture.</p>
					</div>
					<div class="rounded-lg bg-gray-800/50 p-2">
						<p class="font-medium text-gray-300">Structures (LBP)</p>
						<p class="text-gray-500">Atypical pigment network, globules, streaks, blue-white veil. Detected via Local Binary Patterns + gradient analysis.</p>
					</div>
				</div>
			</div>
		{/if}
	</section>

	<!-- Section 3: Training Data Sources -->
	<section class="rounded-xl border border-gray-800 bg-gray-900/50">
		<button
			onclick={() => (dataOpen = !dataOpen)}
			class="flex w-full items-center justify-between p-4 text-left"
		>
			<h3 class="text-sm font-semibold text-teal-400 flex items-center gap-2">
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"></path>
				</svg>
				Training Data Sources
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {dataOpen ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if dataOpen}
			<div class="px-4 pb-4 space-y-2 text-[10px]">
				<div class="flex justify-between items-center rounded-lg bg-gray-800/50 p-2">
					<div>
						<p class="font-medium text-gray-300">HAM10000</p>
						<p class="text-gray-500">Tschandl et al. 2018 -- Medical University of Vienna. 7-class dermoscopy benchmark.</p>
					</div>
					<span class="text-gray-400 flex-shrink-0 ml-2">10,015 images</span>
				</div>
				<div class="flex justify-between items-center rounded-lg bg-gray-800/50 p-2">
					<div>
						<p class="font-medium text-gray-300">ISIC Archive</p>
						<p class="text-gray-500">International Skin Imaging Collaboration -- global dermoscopy archive.</p>
					</div>
					<span class="text-gray-400 flex-shrink-0 ml-2">70,000+ images</span>
				</div>
				<div class="flex justify-between items-center rounded-lg bg-gray-800/50 p-2">
					<div>
						<p class="font-medium text-gray-300">Clinical Literature</p>
						<p class="text-gray-500">AAD 2024-2026 guidelines, DermaSensor FDA DEN230008 pivotal study data, Dermoscopedia.</p>
					</div>
					<span class="text-gray-400 flex-shrink-0 ml-2">15+ publications</span>
				</div>
				<div class="flex justify-between items-center rounded-lg bg-gray-800/50 p-2">
					<div>
						<p class="font-medium text-gray-300">Anwarkh1 ViT-Base</p>
						<p class="text-gray-500">85.8M parameter Vision Transformer, fine-tuned on HAM10000 for 7-class classification.</p>
					</div>
					<span class="text-gray-400 flex-shrink-0 ml-2">HuggingFace</span>
				</div>
				<div class="flex justify-between items-center rounded-lg bg-gray-800/50 p-2">
					<div>
						<p class="font-medium text-gray-300">SigLIP (skintaglabs)</p>
						<p class="text-gray-500">400M parameter SigLIP model from SkinTag Labs, trained on dermoscopy image-text pairs.</p>
					</div>
					<span class="text-gray-400 flex-shrink-0 ml-2">HuggingFace</span>
				</div>
			</div>
		{/if}
	</section>

	<!-- Section 4: Clinical Benchmarks -->
	<section class="rounded-xl border border-gray-800 bg-gray-900/50">
		<button
			onclick={() => (benchmarksOpen = !benchmarksOpen)}
			class="flex w-full items-center justify-between p-4 text-left"
		>
			<h3 class="text-sm font-semibold text-teal-400 flex items-center gap-2">
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
				</svg>
				Calibrated Against
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {benchmarksOpen ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if benchmarksOpen}
			<div class="px-4 pb-4 space-y-2 text-[10px]">
				<div class="rounded-lg bg-gray-800/50 p-2">
					<p class="font-medium text-gray-300">DermaSensor (FDA DEN230008, cleared Jan 2024)</p>
					<p class="text-gray-500">Melanoma sensitivity 90.2%, overall sensitivity 95.5%, NPV 96.6%.</p>
					<p class="text-gray-500">DERM-SUCCESS pivotal study, 1,579 lesions. DERM-ASSESS III, 440 lesions.</p>
				</div>
				<div class="rounded-lg bg-gray-800/50 p-2">
					<p class="font-medium text-gray-300">ABCD Rule (Stolz et al. 1994)</p>
					<p class="text-gray-500">TDS formula validated: 92.8% of melanomas score >5.45. Benign cutoff at 4.75 captures 90.3% of nevi.</p>
				</div>
				<div class="rounded-lg bg-gray-800/50 p-2">
					<p class="font-medium text-gray-300">7-Point Checklist (Argenziano et al. 1998)</p>
					<p class="text-gray-500">Score of 3 or more: 95% sensitivity, 75% specificity (retrospective). Major criteria = 2 pts, minor = 1 pt.</p>
				</div>
				<div class="rounded-lg bg-gray-800/50 p-2">
					<p class="font-medium text-gray-300">DrAgnes Targets</p>
					<p class="text-gray-500">
						Melanoma sensitivity target: 95%. Overall sensitivity: 95%.
						Specificity target: 50%. Melanoma FNR ceiling: 5%. NPV minimum: 96%.
						Fitzpatrick disparity cap: 5%.
					</p>
				</div>
			</div>
		{/if}
	</section>

	<!-- Section 5: Safety Mechanisms -->
	<section class="rounded-xl border border-gray-800 bg-gray-900/50">
		<button
			onclick={() => (safetyOpen = !safetyOpen)}
			class="flex w-full items-center justify-between p-4 text-left"
		>
			<h3 class="text-sm font-semibold text-amber-400 flex items-center gap-2">
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
				</svg>
				Safety Mechanisms
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {safetyOpen ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if safetyOpen}
			<div class="px-4 pb-4">
				<ul class="space-y-2 text-[10px] text-gray-400">
					<li class="flex items-start gap-2">
						<span class="text-amber-400 mt-0.5 flex-shrink-0">1.</span>
						<span>
							<strong class="text-gray-300">Melanoma Safety Gate:</strong>
							If 2+ suspicious features (asymmetry, border irregularity, multiple colors, blue-white veil, streaks) are present,
							melanoma probability floor is set to 15% -- triggering at minimum a "monitor" recommendation regardless of model output.
						</span>
					</li>
					<li class="flex items-start gap-2">
						<span class="text-amber-400 mt-0.5 flex-shrink-0">2.</span>
						<span>
							<strong class="text-gray-300">TDS Override:</strong>
							If Total Dermoscopy Score exceeds 5.45, P(malignant) is forced to 30% or above --
							triggering biopsy recommendation regardless of model output. This threshold captures 92.8% of histologically confirmed melanomas.
						</span>
					</li>
					<li class="flex items-start gap-2">
						<span class="text-amber-400 mt-0.5 flex-shrink-0">3.</span>
						<span>
							<strong class="text-gray-300">Model Disagreement Alert:</strong>
							When the two ViT models disagree on the top-1 class, the case is flagged for clinical review
							with the inter-model agreement score displayed. This catches cases where one model may have a blind spot.
						</span>
					</li>
					<li class="flex items-start gap-2">
						<span class="text-amber-400 mt-0.5 flex-shrink-0">4.</span>
						<span>
							<strong class="text-gray-300">Offline Fallback:</strong>
							If neural network access is unavailable, local rule-based analysis (L2 + L3) still runs with clinical scoring.
							Ensemble shifts to 60% trained-weights + 40% rule-based for continued operation without network dependency.
						</span>
					</li>
					<li class="flex items-start gap-2">
						<span class="text-amber-400 mt-0.5 flex-shrink-0">5.</span>
						<span>
							<strong class="text-gray-300">Decision Thresholds:</strong>
							P(malignant) &gt; 50% = urgent referral.
							P(malignant) 30-50% = biopsy advised.
							P(malignant) 10-30% = monitor.
							P(malignant) &lt; 10% = reassurance.
							Thresholds calibrated against DERM-SUCCESS NPV of 96.6%.
						</span>
					</li>
				</ul>
			</div>
		{/if}
	</section>

	<!-- Section 6: Known Limitations -->
	<section class="rounded-xl border border-red-900/30 bg-red-950/20">
		<button
			onclick={() => (limitationsOpen = !limitationsOpen)}
			class="flex w-full items-center justify-between p-4 text-left"
		>
			<h3 class="text-sm font-semibold text-red-400 flex items-center gap-2">
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"></path>
				</svg>
				Known Limitations
			</h3>
			<svg
				class="h-4 w-4 text-gray-500 transition-transform {limitationsOpen ? 'rotate-180' : ''}"
				fill="none" stroke="currentColor" viewBox="0 0 24 24"
			>
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</button>
		{#if limitationsOpen}
			<div class="px-4 pb-4">
				<ul class="space-y-1.5 text-[10px] text-red-300/70">
					<li class="flex items-start gap-1.5">
						<span class="mt-0.5 flex-shrink-0">--</span>
						<span>This tool is for research screening only -- not a clinical diagnosis.</span>
					</li>
					<li class="flex items-start gap-1.5">
						<span class="mt-0.5 flex-shrink-0">--</span>
						<span>Community ViT models achieved 73.3% melanoma sensitivity in internal testing (below DermaSensor's 95.5%).</span>
					</li>
					<li class="flex items-start gap-1.5">
						<span class="mt-0.5 flex-shrink-0">--</span>
						<span>Training data is predominantly Fitzpatrick I-III (lighter skin tones). Performance on Fitzpatrick IV-VI has not been validated.</span>
					</li>
					<li class="flex items-start gap-1.5">
						<span class="mt-0.5 flex-shrink-0">--</span>
						<span>ABCDE "Evolution" scoring requires longitudinal comparison (not available from a single image).</span>
					</li>
					<li class="flex items-start gap-1.5">
						<span class="mt-0.5 flex-shrink-0">--</span>
						<span>Not FDA-cleared for any clinical use. Not CE-marked. Not TGA-approved.</span>
					</li>
					<li class="flex items-start gap-1.5">
						<span class="mt-0.5 flex-shrink-0">--</span>
						<span>Diameter estimation relies on a fixed assumption of 30cm camera distance and may be inaccurate.</span>
					</li>
				</ul>
			</div>
		{/if}
	</section>
</div>
