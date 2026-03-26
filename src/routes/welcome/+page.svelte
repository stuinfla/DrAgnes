<script lang="ts">
	import { onMount } from "svelte";

	// Animated counter values
	let sensitivity = $state(0);
	let images = $state(0);
	let auroc = $state(0);
	let mounted = $state(false);
	let textSent = $state(false);

	onMount(() => {
		mounted = true;
		const duration = 1500;
		const steps = 50;
		const interval = duration / steps;
		let step = 0;
		const timer = setInterval(() => {
			step++;
			const p = step / steps;
			const e = 1 - Math.pow(1 - p, 3);
			sensitivity = Math.round(9597 * e) / 100;
			images = Math.round(37484 * e);
			auroc = Math.round(960 * e) / 1000;
			if (step >= steps) {
				sensitivity = 95.97;
				images = 37484;
				auroc = 0.960;
				clearInterval(timer);
			}
		}, interval);
	});

	// Scroll reveal
	let sections: HTMLElement[] = [];
	let visible = $state(new Set<number>());

	onMount(() => {
		const observer = new IntersectionObserver(
			(entries) => {
				entries.forEach((entry) => {
					const idx = sections.indexOf(entry.target as HTMLElement);
					if (entry.isIntersecting && idx >= 0) {
						visible = new Set([...visible, idx]);
					}
				});
			},
			{ threshold: 0.15 }
		);
		sections.forEach((el) => el && observer.observe(el));
		return () => observer.disconnect();
	});

	const PIPELINE_STEPS = [
		{ icon: "camera", label: "Capture", desc: "Phone camera photo of the spot that concerns you" },
		{ icon: "check-circle", label: "Quality Check", desc: "Blur, lighting, and contrast validation" },
		{ icon: "search", label: "Lesion Gate", desc: "Is there actually a lesion? Normal skin gets an all-clear" },
		{ icon: "scissors", label: "Segmentation", desc: "Isolate the lesion from surrounding skin" },
		{ icon: "cpu", label: "Feature Extraction", desc: "ABCDE scoring, color analysis, texture mapping" },
		{ icon: "layers", label: "4-Layer Ensemble", desc: "AI model + medical literature + clinical rules + demographics" },
		{ icon: "sliders", label: "Threshold Optimization", desc: "Per-class ROC-optimized decision boundaries" },
		{ icon: "git-merge", label: "Meta-Classifier", desc: "Neural and clinical agreement scoring" },
		{ icon: "bar-chart", label: "Bayesian Risk", desc: "Calibrated post-test probability with prevalence adjustment" },
		{ icon: "message-circle", label: "Plain English", desc: "Medical jargon translated to clear recommendations" },
		{ icon: "shield", label: "Display", desc: "Color-coded result with evidence and next steps" },
	];

	const EVIDENCE = [
		{ metric: "Melanoma Sensitivity", value: "95.97%", source: "combined-training-results.json", detail: "95% CI: 94.5% - 97.4%" },
		{ metric: "Melanoma AUROC", value: "0.960", source: "combined-training-results.json", detail: "Area under receiver operating characteristic" },
		{ metric: "All-Cancer Sensitivity", value: "98.3%", source: "combined-training-results.json", detail: "mel + bcc + akiec combined" },
		{ metric: "NPV", value: "99.06%", source: "clinical-metrics-full.json", detail: "When we say no concern, we're right 99% of the time" },
		{ metric: "NNB", value: "2.1", source: "clinical-metrics-full.json", detail: "Number Needed to Biopsy" },
		{ metric: "Training Images", value: "37,484", source: "combined-training-results.json", detail: "HAM10000 + ISIC 2019 combined dataset" },
	];

	const LIMITATIONS = [
		{ title: "Educational tool, not a medical device. Not FDA-cleared for any clinical use.", desc: "Mela is a skin awareness tool. It does not diagnose, screen for, or detect any disease. It does not replace a dermatologist's evaluation." },
		{ title: "Trained on dermoscopic images", desc: "Performance on phone camera photos has not been clinically validated yet." },
		{ title: "30pp Fitzpatrick equity gap", desc: "Melanoma sensitivity drops significantly for darker skin tones (IV-VI). We're working on it." },
		{ title: "The AI model flags more patterns for awareness rather than fewer", desc: "~20% of benign moles get flagged. The model errs on the side of caution to encourage professional consultation." },
	];
</script>

<svelte:head>
	<title>Mela - AI Skin Awareness Tool | Because catching it early changes everything</title>
	<meta name="description" content="Free, private AI skin awareness on your phone. Research validation: 95.97% melanoma sensitivity, trained on 37,484 images. No uploads, no accounts. Your photos never leave your device." />
	<meta name="keywords" content="skin awareness, skin pattern analysis, AI dermatology, skin education, skin cancer app, ABCDE melanoma, skin lesion analysis, free skin check" />

	<!-- Open Graph -->
	<meta property="og:type" content="website" />
	<meta property="og:url" content="https://mela-app.vercel.app/welcome" />
	<meta property="og:title" content="Mela - AI Skin Awareness Tool" />
	<meta property="og:description" content="Free, private AI skin awareness on your phone. 95.97% melanoma sensitivity in research validation. Your photos never leave your device." />
	<meta property="og:image" content="https://mela-app.vercel.app/og-image.png" />
	<meta property="og:site_name" content="Mela" />

	<!-- Twitter Card -->
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="Mela - AI Skin Awareness Tool" />
	<meta name="twitter:description" content="Free, private AI skin awareness. 95.97% melanoma sensitivity in research validation. Photos never leave your phone." />
	<meta name="twitter:image" content="https://mela-app.vercel.app/og-image.png" />

	<!-- Canonical -->
	<link rel="canonical" href="https://mela-app.vercel.app/welcome" />

	<!-- Structured Data -->
	{@html `<script type="application/ld+json">${JSON.stringify({
		"@context": "https://schema.org",
		"@type": "WebPage",
		"name": "Mela - AI Skin Awareness Tool",
		"description": "Free, private AI-powered skin awareness and education tool. Research validation: 95.97% melanoma sensitivity on external data.",
		"url": "https://mela-app.vercel.app/welcome",
		"audience": {
			"@type": "PeopleAudience",
			"suggestedMinAge": 18
		},
		"disclaimer": "Educational use only. Not a medical device. Not FDA-cleared. Does not diagnose, screen for, or detect any disease.",
		"publisher": {
			"@type": "Organization",
			"name": "Mela"
		}
	})}</script>`}
</svelte:head>

<!-- HERO SECTION -->
<div class="min-h-screen bg-[#0a0a0f] text-white selection:bg-teal-500/30">

	<!-- Nav -->
	<nav class="fixed top-0 left-0 right-0 z-50 border-b border-white/[0.04] bg-[#0a0a0f]/90 backdrop-blur-xl">
		<div class="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
			<div class="flex items-center gap-3">
				<img src="/icons/icon-32x32.png" alt="Mela" class="h-8 w-8 rounded-full ring-1 ring-white/10" />
				<span class="text-lg font-semibold tracking-tight">Mela</span>
			</div>
			<div class="flex items-center gap-6">
				<a href="#how-it-works" class="hidden text-sm text-gray-400 transition hover:text-white sm:block">How It Works</a>
				<a href="#evidence" class="hidden text-sm text-gray-400 transition hover:text-white sm:block">Evidence</a>
				<a href="#limitations" class="hidden text-sm text-gray-400 transition hover:text-white sm:block">Honest About</a>
				<a
					href="/"
					class="rounded-full bg-teal-500 px-5 py-2 text-sm font-semibold text-black transition hover:bg-teal-400 active:scale-95"
				>
					Open App
				</a>
			</div>
		</div>
	</nav>

	<!-- Hero -->
	<section class="relative flex min-h-screen flex-col items-center justify-center px-6 pt-20">
		<!-- Gradient orbs -->
		<div class="pointer-events-none absolute inset-0 overflow-hidden">
			<div class="absolute -top-40 left-1/4 h-[600px] w-[600px] rounded-full bg-teal-500/[0.07] blur-[120px]"></div>
			<div class="absolute -bottom-20 right-1/4 h-[400px] w-[400px] rounded-full bg-purple-500/[0.05] blur-[100px]"></div>
		</div>

		<div class="relative z-10 max-w-3xl text-center">
			<div class="mb-6 inline-flex items-center gap-2 rounded-full border border-teal-500/20 bg-teal-500/[0.08] px-4 py-1.5">
				<div class="h-2 w-2 rounded-full bg-teal-400 animate-pulse"></div>
				<span class="text-xs font-medium text-teal-300">Research Preview</span>
			</div>

			<h1 class="mb-6 text-4xl font-bold leading-[1.1] tracking-tight sm:text-6xl lg:text-7xl">
				Because catching it early
				<span class="bg-gradient-to-r from-teal-400 to-emerald-400 bg-clip-text text-transparent">
					changes everything
				</span>
			</h1>

			<p class="mx-auto mb-10 max-w-xl text-lg leading-relaxed text-gray-400 sm:text-xl">
				AI-powered skin awareness on your phone. Photograph a spot. Learn about your skin.
				Your photos never leave your device.
			</p>

			<!-- Stats row -->
			<div class="mb-12 flex flex-wrap items-center justify-center gap-8 sm:gap-12" class:opacity-0={!mounted} class:opacity-100={mounted} style="transition: opacity 0.8s ease">
				<div class="text-center">
					<div class="text-3xl font-bold tabular-nums text-teal-400 sm:text-4xl">{sensitivity.toFixed(2)}%</div>
					<div class="mt-1 text-xs text-gray-500 uppercase tracking-wider">Melanoma Sensitivity</div>
				</div>
				<div class="hidden h-8 w-px bg-white/10 sm:block"></div>
				<div class="text-center">
					<div class="text-3xl font-bold tabular-nums text-white sm:text-4xl">{images.toLocaleString()}</div>
					<div class="mt-1 text-xs text-gray-500 uppercase tracking-wider">Training Images</div>
				</div>
				<div class="hidden h-8 w-px bg-white/10 sm:block"></div>
				<div class="text-center">
					<div class="text-3xl font-bold tabular-nums text-emerald-400 sm:text-4xl">{auroc.toFixed(3)}</div>
					<div class="mt-1 text-xs text-gray-500 uppercase tracking-wider">Melanoma AUROC</div>
				</div>
			</div>
			<p class="mt-3 text-xs text-gray-600 text-center">Research validation metrics (Source: scripts/combined-training-results.json). Not clinical diagnostic performance claims.</p>

			<!-- CTAs -->
			<div class="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
				<a
					href="/"
					class="group flex items-center gap-3 rounded-2xl bg-teal-500 px-8 py-4 text-lg font-semibold text-black shadow-lg shadow-teal-500/25 transition hover:bg-teal-400 hover:shadow-teal-400/30 active:scale-[0.98]"
				>
					Scan a Spot Now
					<svg class="h-5 w-5 transition group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
				</a>
				<a
					href="#how-it-works"
					class="flex items-center gap-2 rounded-2xl border border-white/10 px-8 py-4 text-lg font-medium text-gray-300 transition hover:border-white/20 hover:text-white"
				>
					See How It Works
				</a>
			</div>

			<!-- Privacy badge -->
			<div class="mt-10 flex items-center justify-center gap-2 text-sm text-gray-500">
				<svg class="h-4 w-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
				100% on-device. No uploads. No accounts. No data collection.
			</div>
		</div>

		<!-- Scroll indicator -->
		<div class="absolute bottom-8 flex flex-col items-center gap-2 text-gray-600">
			<span class="text-[10px] uppercase tracking-widest">Scroll</span>
			<svg class="h-5 w-5 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
		</div>
	</section>

	<!-- WHY WE BUILT THIS -->
	<section bind:this={sections[0]} class="relative px-6 py-24 transition-all duration-700 {visible.has(0) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">The Problem</div>
			<h2 class="mb-8 text-3xl font-bold tracking-tight sm:text-4xl">
				5 billion people will never see a dermatologist
			</h2>
			<div class="grid gap-6 sm:grid-cols-3">
				<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6">
					<div class="mb-3 text-3xl font-bold text-red-400">1 in 5</div>
					<p class="text-sm text-gray-400">Americans will develop skin cancer by age 70. Early detection is the single biggest factor in survival.</p>
				</div>
				<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6">
					<div class="mb-3 text-3xl font-bold text-amber-400">99%</div>
					<p class="text-sm text-gray-400">5-year survival rate for melanoma caught early. Drops to 35% when found late. The difference is time.</p>
				</div>
				<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6">
					<div class="mb-3 text-3xl font-bold text-teal-400">Free</div>
					<p class="text-sm text-gray-400">Mela is open-source, runs entirely on your phone, and costs nothing. Because skin awareness shouldn't be a luxury.</p>
				</div>
			</div>
		</div>
	</section>

	<!-- HOW TO USE IT -->
	<section bind:this={sections[1]} class="relative px-6 py-24 transition-all duration-700 {visible.has(1) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">Get Started</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				Three taps. No app store.
			</h2>
			<p class="mb-12 max-w-2xl text-gray-400">
				Mela is a Progressive Web App. Visit the link on your phone and add it to your home screen. Works offline.
			</p>

			<div class="grid gap-8 sm:grid-cols-3">
				<div class="relative">
					<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-500/10 text-teal-400">
						<span class="text-xl font-bold">1</span>
					</div>
					<h3 class="mb-2 text-lg font-semibold">Open on your phone</h3>
					<p class="text-sm text-gray-400">
						Visit <a href="/" class="text-teal-400 underline decoration-teal-400/30 underline-offset-2 hover:decoration-teal-400">mela-app.vercel.app</a> on Safari or Chrome.
						Tap "Add to Home Screen" for the full app experience.
					</p>
				</div>
				<div>
					<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-500/10 text-teal-400">
						<span class="text-xl font-bold">2</span>
					</div>
					<h3 class="mb-2 text-lg font-semibold">Photograph a spot</h3>
					<p class="text-sm text-gray-400">
						Good lighting. Six inches away. Center one spot in the frame.
						The app checks image quality automatically.
					</p>
				</div>
				<div>
					<div class="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-500/10 text-teal-400">
						<span class="text-xl font-bold">3</span>
					</div>
					<h3 class="mb-2 text-lg font-semibold">See what AI pattern analysis finds</h3>
					<p class="text-sm text-gray-400">
						The analysis shows pattern similarity to reference categories. This is educational, not diagnostic.
						Always consult a dermatologist for medical concerns.
					</p>
				</div>
			</div>

			<div class="mt-12 rounded-2xl border border-teal-500/20 bg-teal-500/[0.04] p-6">
				<div class="flex items-start gap-3">
					<svg class="mt-0.5 h-5 w-5 shrink-0 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
					<div>
						<div class="font-semibold text-teal-300">Privacy by architecture, not by promise</div>
						<p class="mt-1 text-sm text-gray-400">
							The AI model runs entirely on your device. Your photos are never uploaded to any server.
							There are no accounts, no tracking, no data collection. We can't see your images because they never leave your phone.
						</p>
					</div>
				</div>
			</div>
		</div>
	</section>

	<!-- HOW IT WORKS (Pipeline) -->
	<section id="how-it-works" bind:this={sections[2]} class="relative px-6 py-24 transition-all duration-700 {visible.has(2) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">Under the Hood</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				11 steps between your photo and a recommendation
			</h2>
			<p class="mb-12 max-w-2xl text-gray-400">
				A single neural network isn't trustworthy enough for skin analysis. Mela uses a 4-layer ensemble
				with safety gates that catch what any single model misses.
			</p>

			<div class="space-y-3">
				{#each PIPELINE_STEPS as step, i}
					<div class="flex items-start gap-4 rounded-xl border border-white/[0.04] bg-white/[0.015] p-4 transition hover:border-white/[0.08] hover:bg-white/[0.025]">
						<div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-500/10 text-xs font-bold text-teal-400">
							{i + 1}
						</div>
						<div>
							<div class="font-semibold text-sm">{step.label}</div>
							<div class="text-sm text-gray-500">{step.desc}</div>
						</div>
					</div>
				{/each}
			</div>
		</div>
	</section>

	<!-- 4-LAYER DEFENSE -->
	<section bind:this={sections[3]} class="relative px-6 py-24 transition-all duration-700 {visible.has(3) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">Multi-Layer Defense</div>
			<h2 class="mb-12 text-3xl font-bold tracking-tight sm:text-4xl">
				Four independent systems. One recommendation.
			</h2>

			<div class="grid gap-6 sm:grid-cols-2">
				<div class="rounded-2xl border border-white/[0.06] bg-gradient-to-br from-blue-500/[0.04] to-transparent p-6">
					<div class="mb-3 text-sm font-bold text-blue-400 uppercase tracking-wider">Layer 1: AI Model</div>
					<p class="text-sm text-gray-400">Custom Vision Transformer trained on 37,484 dermoscopic images from HAM10000 and ISIC 2019. 95.97% melanoma sensitivity on external validation.</p>
					<div class="mt-3 text-xs text-gray-600">70% ensemble weight</div>
				</div>
				<div class="rounded-2xl border border-white/[0.06] bg-gradient-to-br from-purple-500/[0.04] to-transparent p-6">
					<div class="mb-3 text-sm font-bold text-purple-400 uppercase tracking-wider">Layer 2: Medical Literature</div>
					<p class="text-sm text-gray-400">20-feature logistic regression with every weight cited to published dermatology literature. Cross-checks the AI's pixel-level assessment with clinical knowledge.</p>
					<div class="mt-3 text-xs text-gray-600">15% ensemble weight</div>
				</div>
				<div class="rounded-2xl border border-white/[0.06] bg-gradient-to-br from-amber-500/[0.04] to-transparent p-6">
					<div class="mb-3 text-sm font-bold text-amber-400 uppercase tracking-wider">Layer 3: Clinical Checklists</div>
					<p class="text-sm text-gray-400">ABCDE criteria, Total Dermoscopy Score, 7-point checklist. The same tools dermatologists use in-office, computed from computer vision features.</p>
					<div class="mt-3 text-xs text-gray-600">15% ensemble weight</div>
				</div>
				<div class="rounded-2xl border border-white/[0.06] bg-gradient-to-br from-emerald-500/[0.04] to-transparent p-6">
					<div class="mb-3 text-sm font-bold text-emerald-400 uppercase tracking-wider">Layer 4: Demographics</div>
					<p class="text-sm text-gray-400">Age, sex, body location, skin type. Bayesian adjustment of risk using epidemiological prevalence data. A mole on a 60-year-old's back means something different than on a 20-year-old's arm.</p>
					<div class="mt-3 text-xs text-gray-600">Bayesian risk adjustment</div>
				</div>
			</div>

			<div class="mt-8 rounded-2xl border border-red-500/20 bg-red-500/[0.04] p-6">
				<div class="flex items-start gap-3">
					<svg class="mt-0.5 h-5 w-5 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
					<div>
						<div class="font-semibold text-red-300">Safety Gates (always active)</div>
						<p class="mt-1 text-sm text-gray-400">
							Hard-wired overrides that cannot be disabled: if 2+ ABCDE criteria are suspicious, melanoma probability floors at 15%.
							If Total Dermoscopy Score exceeds 5.45, malignant probability floors at 30%.
							These gates run after the ensemble and override lower estimates.
						</p>
					</div>
				</div>
			</div>
		</div>
	</section>

	<!-- EVIDENCE -->
	<section id="evidence" bind:this={sections[4]} class="relative px-6 py-24 transition-all duration-700 {visible.has(4) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">The Numbers</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				Every number cites its source file
			</h2>
			<p class="mb-12 max-w-2xl text-gray-400">
				Previous versions had fabricated claims. An internal audit caught them.
				Now every performance number must trace back to a JSON evidence file. No exceptions.
			</p>

			<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
				{#each EVIDENCE as item}
					<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5">
						<div class="text-2xl font-bold text-white">{item.value}</div>
						<div class="mt-1 text-sm font-semibold text-gray-300">{item.metric}</div>
						<div class="mt-2 text-xs text-gray-500">{item.detail}</div>
						<div class="mt-3 flex items-center gap-1.5">
							<svg class="h-3 w-3 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
							<span class="text-[10px] font-mono text-teal-500/70">{item.source}</span>
						</div>
					</div>
				{/each}
			</div>
			<p class="mt-6 text-xs text-gray-500 leading-relaxed">
				These metrics reflect research validation on reference datasets. They do not represent clinical diagnostic performance.
			</p>
		</div>
	</section>

	<!-- WHAT WE'RE HONEST ABOUT -->
	<section id="limitations" bind:this={sections[5]} class="relative px-6 py-24 transition-all duration-700 {visible.has(5) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-amber-400">What We're Honest About</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				Limitations matter more than capabilities
			</h2>
			<p class="mb-12 max-w-2xl text-gray-400">
				We publish our failures more prominently than our successes. You deserve to know what this tool cannot do.
			</p>

			<div class="space-y-4">
				{#each LIMITATIONS as item}
					<div class="rounded-2xl border border-amber-500/10 bg-amber-500/[0.02] p-6">
						<div class="font-semibold text-amber-300">{item.title}</div>
						<p class="mt-2 text-sm text-gray-400">{item.desc}</p>
					</div>
				{/each}
			</div>
		</div>
	</section>

	<!-- OPEN SOURCE & CONTRIBUTE -->
	<section bind:this={sections[6]} class="relative px-6 py-24 transition-all duration-700 {visible.has(6) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">Open Source</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				Every line of code is public
			</h2>
			<p class="mx-auto mb-12 max-w-2xl text-gray-400">
				Mela's source code, training data references, evidence files, and architecture decisions are all public on GitHub.
				We believe health-related AI should be auditable.
			</p>

			<div class="grid gap-6 sm:grid-cols-2">
				<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6">
					<div class="mb-3 flex items-center gap-3">
						<svg class="h-6 w-6 text-teal-400" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
						<span class="font-semibold">View the Code</span>
					</div>
					<p class="mb-4 text-sm text-gray-400">
						Browse the full source, read the architecture decisions (13 ADRs), review the evidence files,
						and see exactly how every classification decision is made.
					</p>
					<a
						href="https://github.com/stuinfla/Mela"
						target="_blank"
						rel="noopener noreferrer"
						class="inline-flex items-center gap-2 text-sm font-medium text-teal-400 hover:text-teal-300 transition"
					>
						github.com/stuinfla/Mela
						<svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
					</a>
				</div>
				<div class="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6">
					<div class="mb-3 flex items-center gap-3">
						<svg class="h-6 w-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" /></svg>
						<span class="font-semibold">Contribute</span>
					</div>
					<p class="mb-4 text-sm text-gray-400">
						We need help from dermatologists, ML engineers, designers, and anyone who cares about
						making skin awareness accessible. Check the README for contribution guidelines.
					</p>
					<a
						href="https://github.com/stuinfla/Mela#contributing"
						target="_blank"
						rel="noopener noreferrer"
						class="inline-flex items-center gap-2 text-sm font-medium text-purple-400 hover:text-purple-300 transition"
					>
						How to contribute
						<svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
					</a>
				</div>
			</div>
		</div>
	</section>

	<!-- SEND TO YOUR PHONE -->
	<section bind:this={sections[7]} class="relative px-6 py-24 transition-all duration-700 {visible.has(7) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-xl text-center">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">Get It On Your Phone</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				Text yourself the link
			</h2>
			<p class="mb-8 text-gray-400">
				Enter your phone number and we'll text you the Mela app link. Open it on your phone, add to home screen, done.
			</p>

			<form
				onsubmit={(e) => {
					e.preventDefault();
					const form = e.target as HTMLFormElement;
					const input = form.querySelector('input') as HTMLInputElement;
					const phone = input.value.replace(/\D/g, '');
					if (phone.length >= 10) {
						const smsBody = encodeURIComponent('Check out Mela - AI skin awareness tool on your phone: https://mela-app.vercel.app');
						window.open(`sms:${phone}?body=${smsBody}`, '_self');
						textSent = true;
					}
				}}
				class="flex flex-col gap-3 sm:flex-row"
			>
				<input
					type="tel"
					placeholder="Your phone number"
					class="flex-1 rounded-xl border border-white/10 bg-white/[0.03] px-5 py-3.5 text-white placeholder-gray-600 outline-none transition focus:border-teal-500/50 focus:ring-1 focus:ring-teal-500/30"
				/>
				<button
					type="submit"
					class="rounded-xl bg-teal-500 px-6 py-3.5 font-semibold text-black transition hover:bg-teal-400 active:scale-[0.98]"
				>
					{textSent ? 'Sent!' : 'Send Link'}
				</button>
			</form>

			<p class="mt-4 text-xs text-gray-600">
				This opens your phone's text messaging app with a pre-filled message. We don't store your number.
			</p>

			<div class="mt-6 flex items-center justify-center gap-2 text-sm text-gray-500">
				<span>Or just open</span>
				<a href="/" class="font-mono text-teal-400 underline decoration-teal-400/30 underline-offset-2">mela-app.vercel.app</a>
				<span>on your phone</span>
			</div>
		</div>
	</section>

	<!-- FEEDBACK & REVIEWS -->
	<section bind:this={sections[8]} class="relative px-6 py-24 transition-all duration-700 {visible.has(8) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}">
		<div class="mx-auto max-w-4xl">
			<div class="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">Help Us Improve</div>
			<h2 class="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
				Your feedback makes Mela better
			</h2>
			<p class="mb-12 max-w-2xl text-gray-400">
				This is an open-source project built for you. Every piece of feedback helps us improve accuracy,
				add features, and catch issues we might miss.
			</p>

			<div class="grid gap-6 sm:grid-cols-3">
				<a
					href="https://github.com/stuinfla/Mela/issues/new?template=feedback.md&title=Feedback:+"
					target="_blank"
					rel="noopener noreferrer"
					class="group rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6 transition hover:border-teal-500/20 hover:bg-teal-500/[0.03]"
				>
					<svg class="mb-3 h-8 w-8 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" /></svg>
					<div class="font-semibold text-white group-hover:text-teal-300 transition">Share Feedback</div>
					<p class="mt-2 text-xs text-gray-500">Tell us what works, what doesn't, and what you'd like to see. Opens a GitHub issue.</p>
				</a>
				<a
					href="https://github.com/stuinfla/Mela/issues/new?template=bug_report.md&title=Bug:+"
					target="_blank"
					rel="noopener noreferrer"
					class="group rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6 transition hover:border-amber-500/20 hover:bg-amber-500/[0.03]"
				>
					<svg class="mb-3 h-8 w-8 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
					<div class="font-semibold text-white group-hover:text-amber-300 transition">Report a Bug</div>
					<p class="mt-2 text-xs text-gray-500">Found something wrong? Help us fix it. Every bug report makes the tool safer.</p>
				</a>
				<a
					href="https://github.com/stuinfla/Mela/discussions"
					target="_blank"
					rel="noopener noreferrer"
					class="group rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6 transition hover:border-purple-500/20 hover:bg-purple-500/[0.03]"
				>
					<svg class="mb-3 h-8 w-8 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" /></svg>
					<div class="font-semibold text-white group-hover:text-purple-300 transition">Join the Discussion</div>
					<p class="mt-2 text-xs text-gray-500">Ask questions, share experiences, connect with other users and contributors.</p>
				</a>
			</div>
		</div>
	</section>

	<!-- FINAL CTA -->
	<section class="relative px-6 py-32">
		<div class="mx-auto max-w-3xl text-center">
			<h2 class="mb-6 text-3xl font-bold tracking-tight sm:text-5xl">
				Check a spot in
				<span class="bg-gradient-to-r from-teal-400 to-emerald-400 bg-clip-text text-transparent">30 seconds</span>
			</h2>
			<p class="mx-auto mb-10 max-w-lg text-lg text-gray-400">
				Open Mela on your phone. No download, no signup, no cost.
			</p>
			<a
				href="/"
				class="inline-flex items-center gap-3 rounded-2xl bg-teal-500 px-10 py-5 text-xl font-semibold text-black shadow-lg shadow-teal-500/25 transition hover:bg-teal-400 active:scale-[0.98]"
			>
				Open Mela
				<svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
			</a>
		</div>
	</section>

	<!-- FOOTER -->
	<footer class="border-t border-white/[0.04] px-6 py-12">
		<div class="mx-auto max-w-4xl">
			<div class="flex flex-col items-center gap-6 sm:flex-row sm:justify-between">
				<div class="flex items-center gap-2">
					<img src="/icons/icon-32x32.png" alt="Mela" class="h-6 w-6 rounded-full" />
					<span class="text-sm font-semibold">Mela</span>
					<span class="text-xs text-gray-600">v0.9.4</span>
				</div>
				<div class="flex items-center gap-6 text-xs text-gray-600">
					<a href="https://github.com/stuinfla/Mela" target="_blank" rel="noopener noreferrer" class="hover:text-gray-400">GitHub</a>
					<a href="/how-it-works.html" class="hover:text-gray-400">How It Works</a>
					<span>Educational Use Only</span>
				</div>
			</div>
			<div class="mt-8 rounded-xl border border-white/[0.04] bg-white/[0.01] p-4 text-center text-xs text-gray-600 leading-relaxed">
				Mela is an educational skin awareness tool, not a medical device. It is not FDA-cleared for any clinical use.
				Mela does not diagnose, screen for, or detect any disease. Always consult a qualified dermatologist for medical evaluation.
				Performance numbers cite JSON evidence files in the source repository.
			</div>
		</div>
	</footer>

</div>
