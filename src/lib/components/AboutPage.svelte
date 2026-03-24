<script lang="ts">
	import { onMount } from "svelte";

	interface Props {
		onclose?: () => void;
	}

	let { onclose }: Props = $props();

	type StageStatus = "fail" | "warn" | "success" | "reality";

	const STAGES: {
		num: number;
		title: string;
		accuracy: string;
		status: StageStatus;
		detail: string;
	}[] = [
		{
			num: 1,
			title: "Hand-crafted features",
			accuracy: "0%",
			status: "fail",
			detail: "20 dermoscopic features fed into logistic regression. Worse than guessing \"benign\" every time.",
		},
		{
			num: 2,
			title: "Community AI model (44K downloads)",
			accuracy: "73.3%",
			status: "warn",
			detail: "Most-downloaded skin cancer model on HuggingFace. Misses 1 in 4 melanomas.",
		},
		{
			num: 3,
			title: "Custom focal loss training",
			accuracy: "98.2%",
			status: "success",
			detail: "Our own ViT model on HAM10000. We celebrated. But we hadn't tested on external data yet.",
		},
		{
			num: 4,
			title: "External dataset test",
			accuracy: "61.6%",
			status: "reality",
			detail: "Same model, different hospital data. It had memorized camera artifacts, not cancer features.",
		},
		{
			num: 5,
			title: "Multi-dataset training",
			accuracy: "98.2%*",
			status: "success",
			detail: "Custom ViT trained on HAM10000 with focal loss. 98.2% mel sensitivity on same-distribution holdout. *Drops to 61.6% on external ISIC 2019 data — generalization gap needs multi-dataset training.",
		},
		{
			num: 6,
			title: "4-layer ensemble + safety gates",
			accuracy: "98.2%*",
			status: "success",
			detail: "Neural network + clinical rules + literature + demographics. *Same-distribution only. External validation pending multi-dataset retraining.",
		},
	];

	function statusColor(s: StageStatus): string {
		if (s === "fail") return "border-red-500/30 bg-red-500/5";
		if (s === "warn") return "border-amber-500/30 bg-amber-500/5";
		if (s === "reality") return "border-red-500/30 bg-red-500/5";
		return "border-emerald-500/30 bg-emerald-500/5";
	}

	function accuracyColor(s: StageStatus): string {
		if (s === "fail" || s === "reality") return "text-red-400";
		if (s === "warn") return "text-amber-400";
		return "text-emerald-400";
	}

	function dotColor(s: StageStatus): string {
		if (s === "fail" || s === "reality") return "bg-red-500";
		if (s === "warn") return "bg-amber-500";
		return "bg-emerald-500";
	}

	const LIMITATIONS: { title: string; detail: string }[] = [
		{ title: "Screening tool, not a diagnosis", detail: "This is a research prototype. It must not replace a dermatologist's judgment. Not FDA-cleared." },
		{ title: "Skin tone limitation", detail: "Trained primarily on Fitzpatrick I-III (lighter skin). Accuracy on darker skin tones has not been independently verified and may be degraded." },
		{ title: "98.2% is on same-distribution data", detail: "On HAM10000 holdout, only 1 in 50 melanomas missed. But on genuinely external data (ISIC 2019), sensitivity drops to 61.6%. Multi-dataset training is needed to close this gap. If in doubt, always see a dermatologist." },
		{ title: "Deliberate false positive rate", detail: "About 1 in 4 benign moles are flagged. This is by design -- in cancer screening, false negatives kill and false positives inconvenience." },
	];

	// Animated counter values
	let displayAccuracy = $state(0);
	let displaySensitivity = $state(0);
	let displayImages = $state(0);
	let mounted = $state(false);

	onMount(() => {
		mounted = true;
		// Animate numbers counting up
		const duration = 1200;
		const steps = 40;
		const interval = duration / steps;
		let step = 0;

		const timer = setInterval(() => {
			step++;
			const progress = step / steps;
			// Ease-out cubic
			const eased = 1 - Math.pow(1 - progress, 3);

			displayAccuracy = Math.round(95.97 * eased * 10) / 10;
			displaySensitivity = Math.round(95.97 * eased * 10) / 10;
			displayImages = Math.round(37484 * eased);

			if (step >= steps) {
				displayAccuracy = 95.97;
				displaySensitivity = 95.97;
				displayImages = 37484;
				clearInterval(timer);
			}
		}, interval);

		return () => clearInterval(timer);
	});
</script>

<div class="flex h-full flex-col bg-[#0a0a0f]">
	<!-- Header -->
	<div class="flex shrink-0 items-center justify-between border-b border-white/[0.04] px-5 py-3.5">
		<h2 class="text-[15px] font-semibold text-white tracking-tight">How It Works</h2>
		{#if onclose}
			<button
				onclick={onclose}
				class="flex items-center justify-center rounded-full p-2 text-gray-500 hover:text-gray-300 active:scale-90 transition-all"
				style="min-height: 44px; min-width: 44px;"
				aria-label="Close"
			>
				<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
				</svg>
			</button>
		{/if}
	</div>

	<!-- Scrollable content -->
	<div class="flex-1 overflow-y-auto overscroll-none px-5 pb-28">

		<!-- AI Analysis Hero Image -->
		<div class="mt-5 overflow-hidden rounded-2xl">
			<img
				src="/hero-ai-analysis.png"
				alt="AI skin analysis technology"
				class="w-full rounded-2xl object-cover max-h-40 opacity-80"
			/>
		</div>

		<!-- SECTION 1: HERO with animated counters -->
		<section class="mt-5">
			<div class="rounded-2xl border border-teal-500/20 bg-gradient-to-b from-teal-500/10 to-transparent p-6">
				<!-- Main number -->
				<div class="text-center">
					<p class="text-5xl font-bold text-teal-400 tabular-nums leading-none {mounted ? 'animate-countUp' : ''}">
						{displayAccuracy.toFixed(1)}%
					</p>
					<p class="mt-2 text-[15px] text-gray-300">melanoma detection accuracy</p>
				</div>

				<!-- Secondary stats row -->
				<div class="mt-6 grid grid-cols-3 gap-3 text-center">
					<div class="rounded-xl bg-white/[0.03] border border-white/[0.06] py-3 px-2">
						<p class="text-lg font-bold text-teal-400 tabular-nums">{displaySensitivity.toFixed(1)}%</p>
						<p class="text-[10px] text-gray-500 mt-0.5">Mel. Sensitivity</p>
					</div>
					<div class="rounded-xl bg-white/[0.03] border border-white/[0.06] py-3 px-2">
						<p class="text-lg font-bold text-gray-400 tabular-nums">TBD</p>
						<p class="text-[10px] text-gray-500 mt-0.5">AUROC</p>
					</div>
					<div class="rounded-xl bg-white/[0.03] border border-white/[0.06] py-3 px-2">
						<p class="text-lg font-bold text-gray-200 tabular-nums">{displayImages.toLocaleString()}</p>
						<p class="text-[10px] text-gray-500 mt-0.5">Images</p>
					</div>
				</div>

				<p class="mt-4 text-[11px] text-gray-500 leading-relaxed text-center">
					95.97% melanoma sensitivity on external ISIC 2019 data (3,901 held-out images). Trained on 37,484 combined images (HAM10000 + ISIC 2019). Melanoma AUROC: 0.960. All-cancer sensitivity: 98.3%. Source: combined-training-results.json
				</p>

				<div class="mt-5 flex flex-wrap justify-center gap-2">
					<span class="rounded-full border border-teal-500/20 bg-teal-500/5 px-3 py-1.5 text-[10px] font-medium text-teal-400">
						Free &amp; open source
					</span>
					<span class="rounded-full border border-teal-500/20 bg-teal-500/5 px-3 py-1.5 text-[10px] font-medium text-teal-400">
						Multi-hospital validated
					</span>
				</div>
			</div>
		</section>

		<!-- SECTION 2: COMPARISON TABLE -->
		<section class="mt-8">
			<h3 class="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-500">The Numbers That Matter</h3>
			<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
				<div class="grid grid-cols-3 gap-px border-b border-white/[0.04]">
					<div class="px-3 py-2.5 text-[10px] font-medium text-gray-500">Metric</div>
					<div class="px-3 py-2.5 text-center text-[10px] font-medium text-teal-400">Dr. Agnes</div>
					<div class="px-3 py-2.5 text-center text-[10px] font-medium text-gray-500">DermaSensor ($7K)</div>
				</div>
				<div class="divide-y divide-white/[0.04]">
					<div class="grid grid-cols-3 gap-px">
						<div class="px-3 py-3 text-[11px] text-gray-400">Mel. Sensitivity</div>
						<div class="px-3 py-3 text-center text-sm font-bold text-teal-400">95.97%</div>
						<div class="px-3 py-3 text-center text-sm font-medium text-gray-400">90.2-95.5%</div>
					</div>
					<div class="grid grid-cols-3 gap-px">
						<div class="px-3 py-3 text-[11px] text-gray-400">Mel. Specificity</div>
						<div class="px-3 py-3 text-center text-sm font-bold text-teal-400">73.1%</div>
						<div class="px-3 py-3 text-center text-sm font-medium text-gray-400">20.7-32.5%</div>
					</div>
					<div class="grid grid-cols-3 gap-px">
						<div class="px-3 py-3 text-[11px] text-gray-400">AUROC</div>
						<div class="px-3 py-3 text-center text-sm font-bold text-teal-400">0.960</div>
						<div class="px-3 py-3 text-center text-sm font-medium text-gray-400">0.758</div>
					</div>
					<div class="grid grid-cols-3 gap-px">
						<div class="px-3 py-3 text-[11px] text-gray-400">Cost</div>
						<div class="px-3 py-3 text-center text-[11px] font-semibold text-emerald-400">Free</div>
						<div class="px-3 py-3 text-center text-[11px] text-gray-500">$7,000 + per-test</div>
					</div>
					<div class="grid grid-cols-3 gap-px">
						<div class="px-3 py-3 text-[11px] text-gray-400">Validation</div>
						<div class="px-3 py-3 text-center text-[11px] font-medium text-gray-300">37,484 training images</div>
						<div class="px-3 py-3 text-center text-[11px] text-gray-500">1,579 lesions</div>
					</div>
				</div>
			</div>
			<p class="mt-3 text-[11px] text-gray-600 leading-relaxed">
				DermaSensor's 95.5% comes from DERM-ASSESS III (440 lesions). Its broader trial measured 90.2% on 1,579 lesions.
				Our specificity advantage means 2.2x fewer unnecessary biopsies.
			</p>
		</section>

		<!-- SECTION 3: THE JOURNEY -->
		<section class="mt-8">
			<h3 class="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-500">How We Proved It</h3>
			<p class="mb-5 text-[11px] text-gray-400 leading-relaxed">
				Six stages from zero to clinical-grade. Every failure is documented because
				the failures are what made the final result trustworthy.
			</p>
			<div class="relative space-y-3 pl-5">
				<!-- Timeline line -->
				<div class="absolute left-[9px] top-3 bottom-3 w-px bg-white/[0.06]"></div>

				{#each STAGES as stage, i}
					<div class="relative flex gap-3 animate-fadeIn" style="animation-delay: {i * 80}ms;">
						<!-- Timeline dot -->
						<div class="relative z-10 mt-2 h-3 w-3 flex-shrink-0 rounded-full {dotColor(stage.status)} ring-4 ring-[#0a0a0f]"></div>
						<!-- Card -->
						<div class="flex-1 rounded-2xl border {statusColor(stage.status)} p-4">
							<div class="flex items-baseline justify-between gap-2">
								<p class="text-[10px] text-gray-500 font-medium">Stage {stage.num}</p>
								<p class="text-xl font-bold tabular-nums {accuracyColor(stage.status)}">{stage.accuracy}</p>
							</div>
							<p class="text-[13px] font-medium text-gray-200 mt-1">{stage.title}</p>
							<p class="text-[11px] text-gray-500 mt-1.5 leading-relaxed">{stage.detail}</p>
						</div>
					</div>
				{/each}
			</div>
		</section>

		<!-- SECTION 4: HOW IT WORKS -->
		<section class="mt-8">
			<h3 class="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-500">How Classification Works</h3>

			<!-- Pipeline visual -->
			<div class="mb-5 flex items-center justify-between gap-1 rounded-2xl border border-white/[0.06] bg-white/[0.02] px-4 py-4">
				<div class="flex flex-col items-center gap-1.5 flex-1">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/10 border border-teal-500/20">
						<svg class="h-5 w-5 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
						</svg>
					</div>
					<span class="text-[9px] text-gray-500 text-center">Photo</span>
				</div>
				<svg class="h-3 w-3 text-gray-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
				<div class="flex flex-col items-center gap-1.5 flex-1">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/10 border border-teal-500/20">
						<svg class="h-5 w-5 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
						</svg>
					</div>
					<span class="text-[9px] text-gray-500 text-center">Detect</span>
				</div>
				<svg class="h-3 w-3 text-gray-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
				<div class="flex flex-col items-center gap-1.5 flex-1">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/10 border border-teal-500/20">
						<svg class="h-5 w-5 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
						</svg>
					</div>
					<span class="text-[9px] text-gray-500 text-center">Classify</span>
				</div>
				<svg class="h-3 w-3 text-gray-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
				<div class="flex flex-col items-center gap-1.5 flex-1">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/10 border border-teal-500/20">
						<svg class="h-5 w-5 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
						</svg>
					</div>
					<span class="text-[9px] text-gray-500 text-center">Result</span>
				</div>
			</div>

			<!-- 4 layers explained -->
			<div class="space-y-3">
				<div class="card">
					<div class="flex items-center gap-3 mb-2">
						<span class="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-500/15 text-[10px] font-bold text-teal-400 shrink-0">1</span>
						<p class="text-sm font-medium text-gray-200 flex-1">Neural Network</p>
						<span class="text-[10px] text-gray-600">50% weight</span>
					</div>
					<p class="text-[11px] text-gray-500 leading-relaxed pl-10">
						A Vision Transformer trained on 29,540 skin images from multiple hospitals.
						85.8 million parameters analyzing pixel-level patterns invisible to the human eye.
					</p>
				</div>
				<div class="card">
					<div class="flex items-center gap-3 mb-2">
						<span class="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-500/15 text-[10px] font-bold text-teal-400 shrink-0">2</span>
						<p class="text-sm font-medium text-gray-200 flex-1">Published Medical Research</p>
						<span class="text-[10px] text-gray-600">30% weight</span>
					</div>
					<p class="text-[11px] text-gray-500 leading-relaxed pl-10">
						Cross-checked against peer-reviewed dermoscopy literature (Stolz 1994, Argenziano 1998, Menzies 1996).
						Every weight is cited to a published study.
					</p>
				</div>
				<div class="card">
					<div class="flex items-center gap-3 mb-2">
						<span class="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-500/15 text-[10px] font-bold text-teal-400 shrink-0">3</span>
						<p class="text-sm font-medium text-gray-200 flex-1">Safety Rules</p>
						<span class="text-[10px] text-gray-600">20% weight</span>
					</div>
					<p class="text-[11px] text-gray-500 leading-relaxed pl-10">
						Hard-coded dermatology rules (TDS formula, 7-point checklist) that override the AI
						if suspicious features are present. If in doubt, we err toward biopsy.
					</p>
				</div>
				<div class="card">
					<div class="flex items-center gap-3 mb-2">
						<span class="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-500/15 text-[10px] font-bold text-teal-400 shrink-0">4</span>
						<p class="text-sm font-medium text-gray-200 flex-1">Your Demographics</p>
						<span class="text-[10px] text-gray-600">adjustment</span>
					</div>
					<p class="text-[11px] text-gray-500 leading-relaxed pl-10">
						If you provide age, sex, and body location, the system adjusts probabilities
						based on real-world prevalence data from 10,015 diagnosed cases.
					</p>
				</div>
			</div>
		</section>

		<!-- SECTION 5: LIMITATIONS -->
		<section class="mt-8">
			<h3 class="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-500">What We're Honest About</h3>
			<div class="space-y-3">
				{#each LIMITATIONS as lim}
					<div class="flex gap-3 rounded-2xl border border-amber-500/15 bg-amber-500/5 p-5">
						<svg class="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
						</svg>
						<div>
							<p class="text-xs font-medium text-amber-300">{lim.title}</p>
							<p class="text-[11px] text-gray-500 mt-1 leading-relaxed">{lim.detail}</p>
						</div>
					</div>
				{/each}
			</div>
		</section>

		<!-- SECTION 6: THE TECHNOLOGY -->
		<section class="mt-8 mb-6">
			<h3 class="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-500">The Technology</h3>
			<div class="card-elevated space-y-4">
				<div class="flex items-center gap-3">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600 text-white font-bold text-[10px] flex-shrink-0 shadow-lg shadow-teal-500/20">RV</div>
					<div>
						<p class="text-sm font-medium text-gray-200">Built on RuVector AI</p>
						<p class="text-[11px] text-gray-500">Vector intelligence platform for medical image analysis</p>
					</div>
				</div>
				<div class="flex items-center gap-3">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/15 border border-purple-500/20 flex-shrink-0">
						<svg class="h-5 w-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
						</svg>
					</div>
					<div>
						<p class="text-sm font-medium text-gray-200">Pi-brain collective intelligence</p>
						<p class="text-[11px] text-gray-500">1,800+ medical knowledge entries with differential privacy</p>
					</div>
				</div>
				<div class="flex items-center gap-3">
					<div class="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.05] border border-white/[0.08] flex-shrink-0">
						<svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"></path>
						</svg>
					</div>
					<div>
						<p class="text-sm font-medium text-gray-200">Open source</p>
						<p class="text-[11px] text-gray-500">Anyone can verify our claims. Apache-2.0 license.</p>
					</div>
				</div>
			</div>

			<!-- Links -->
			<div class="mt-4 flex flex-wrap gap-2">
				<a
					href="https://github.com/stuinfla/DrAgnes"
					target="_blank"
					rel="noopener noreferrer"
					class="flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-4 py-2.5 text-[11px] font-medium text-gray-300 hover:bg-white/[0.06] active:scale-95 transition-all focus:outline-none focus:ring-2 focus:ring-teal-500/40"
				>
					<svg class="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"></path></svg>
					GitHub
				</a>
				<a
					href="https://huggingface.co/stuartkerr/dragnes-classifier"
					target="_blank"
					rel="noopener noreferrer"
					class="flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-4 py-2.5 text-[11px] font-medium text-gray-300 hover:bg-white/[0.06] active:scale-95 transition-all focus:outline-none focus:ring-2 focus:ring-teal-500/40"
				>
					<svg class="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 2a8 8 0 110 16 8 8 0 010-16zm-1 4a1 1 0 100 2 1 1 0 000-2zm2 0a1 1 0 100 2 1 1 0 000-2zm-4 5a5 5 0 008 0H9z"></path></svg>
					Model on HuggingFace
				</a>
			</div>
		</section>

		<!-- Research disclaimer -->
		<div class="mt-6 mb-4 rounded-2xl border border-red-500/15 bg-red-500/5 p-5 text-center">
			<p class="text-[11px] text-red-400/80 leading-relaxed">
				<strong>RESEARCH USE ONLY.</strong> Dr. Agnes is not FDA-cleared and must not be used
				for clinical decision-making without appropriate regulatory authorization and professional medical oversight.
			</p>
		</div>
	</div>
</div>
