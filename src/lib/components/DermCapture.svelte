<script lang="ts">
	import { onMount, onDestroy, tick } from "svelte";
	import CarbonCamera from "~icons/carbon/camera";
	import type { BodyLocation } from "$lib/dragnes/types";
	import { assessImageQuality } from "$lib/dragnes/image-quality";
	import type { ImageQualityResult } from "$lib/dragnes/image-quality";
	import BodyMap from "./BodyMap.svelte";

	type ImageType = "dermoscopy" | "clinical" | "auto";

	interface CaptureEvent {
		imageData: ImageData;
		bodyLocation: BodyLocation;
		deviceModel: string;
		imageType: ImageType;
	}

	interface MultiCaptureItem {
		imageData: ImageData;
		preview: string;
		imageType: ImageType;
		quality: ImageQualityResult;
	}

	interface Props {
		oncapture: (event: CaptureEvent) => void;
		onmulticapture?: (events: Array<CaptureEvent>) => void;
		multiCapture?: boolean;
		maxImages?: number;
	}

	let { oncapture, onmulticapture, multiCapture = false, maxImages = 3 }: Props = $props();

	let videoEl: HTMLVideoElement | undefined = $state(undefined);
	let canvasEl: HTMLCanvasElement | undefined = $state(undefined);
	let stream: MediaStream | null = $state(null);
	let cameraReady: boolean = $state(false);
	let cameraError: string = $state("");
	let cameraLoading: boolean = $state(false);
	let showUploadFallback: boolean = $state(false);
	/** Whether the getUserMedia live-preview camera view is active */
	let cameraActive: boolean = $state(false);

	let bodyLocation: BodyLocation = $state("unknown");
	let deviceModel: string = $state("phone_only");
	let imageType: ImageType = $state("auto");
	let detectedImageType: string = $state("");
	let capturedPreview: string | null = $state(null);

	// Multi-capture state
	let capturedImages: MultiCaptureItem[] = $state([]);
	let multiCaptureDone: boolean = $derived(
		multiCapture && capturedImages.length >= maxImages
	);

	// Image quality state (single-capture)
	let lastQuality: ImageQualityResult | null = $state(null);

	// Derived: worst suggestion across all multi-capture images
	let qualitySuggestion: string | null = $derived.by(() => {
		if (!multiCapture) return lastQuality?.suggestion ?? null;
		const worst = capturedImages
			.filter((img) => img.quality.grade !== "good")
			.sort((a, b) => a.quality.overallScore - b.quality.overallScore);
		return worst.length > 0 ? worst[0].quality.suggestion : null;
	});

	// Body map drawer state
	let showBodyMap: boolean = $state(false);

	/**
	 * Detect mobile/tablet devices. On iOS Safari, getUserMedia is unreliable --
	 * a native file input with capture="environment" opens the camera directly
	 * and is far more dependable.
	 */
	let isMobileDevice: boolean = $state(false);

	onMount(() => {
		isMobileDevice = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)
			|| (navigator.maxTouchPoints > 0 && /Macintosh/i.test(navigator.userAgent));
	});

	const IMAGE_TYPES: { value: ImageType; label: string; description: string }[] = [
		{ value: "auto", label: "Auto-detect", description: "Automatically determine image type" },
		{ value: "dermoscopy", label: "Dermoscopy", description: "Taken with DermLite or dermatoscope" },
		{ value: "clinical", label: "Clinical / Phone", description: "Standard camera or iPhone photo" },
	];

	/** Auto-detect dermoscopy vs clinical photo from image characteristics */
	function detectImageType(imageData: ImageData): "dermoscopy" | "clinical" {
		const { data, width, height } = imageData;
		const totalPixels = width * height;

		let cornerDarkPixels = 0;
		let centerBrightPixels = 0;
		const cornerSize = Math.floor(Math.min(width, height) * 0.15);
		const centerX = width / 2;
		const centerY = height / 2;
		const centerRadius = Math.min(width, height) * 0.25;

		for (let y = 0; y < height; y += 3) {
			for (let x = 0; x < width; x += 3) {
				const i = (y * width + x) * 4;
				const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;

				const inCorner = (x < cornerSize || x > width - cornerSize) &&
					(y < cornerSize || y > height - cornerSize);
				if (inCorner && brightness < 30) cornerDarkPixels++;

				const dx = x - centerX, dy = y - centerY;
				if (Math.sqrt(dx * dx + dy * dy) < centerRadius && brightness > 80) {
					centerBrightPixels++;
				}
			}
		}

		const sampledPixels = totalPixels / 9;
		const cornerRatio = cornerDarkPixels / (sampledPixels * 0.04);
		const centerRatio = centerBrightPixels / (sampledPixels * 0.2);

		let totalSaturation = 0;
		let satSamples = 0;
		for (let i = 0; i < data.length; i += 12) {
			const r = data[i], g = data[i + 1], b = data[i + 2];
			const max = Math.max(r, g, b);
			const min = Math.min(r, g, b);
			if (max > 20) {
				totalSaturation += (max - min) / max;
				satSamples++;
			}
		}
		const avgSaturation = satSamples > 0 ? totalSaturation / satSamples : 0;

		let dermScore = 0;
		if (cornerRatio > 0.3) dermScore += 2;
		if (centerRatio > 0.5) dermScore += 1;
		if (avgSaturation > 0.35) dermScore += 1;
		if (deviceModel !== "phone_only") dermScore += 3;

		return dermScore >= 3 ? "dermoscopy" : "clinical";
	}

	const DEVICE_MODELS = [
		{ value: "phone_only", label: "Phone Only" },
		{ value: "HUD", label: "DermLite HUD" },
		{ value: "DL5", label: "DermLite DL5" },
		{ value: "DL4", label: "DermLite DL4" },
		{ value: "DL200", label: "DermLite DL200" },
	];

	function getConstraints(): MediaStreamConstraints {
		const isDermLite = deviceModel !== "phone_only";
		return {
			video: {
				facingMode: "environment",
				width: { ideal: isDermLite ? 1920 : 1280 },
				height: { ideal: isDermLite ? 1080 : 720 },
			},
			audio: false,
		};
	}

	export async function startCamera() {
		cameraError = "";
		cameraReady = false;
		cameraLoading = true;
		capturedPreview = null;
		cameraActive = true;

		// Show upload fallback after 3 seconds if camera hasn't loaded
		const fallbackTimer = setTimeout(() => {
			if (!cameraReady && !cameraError) {
				showUploadFallback = true;
			}
		}, 3000);

		try {
			if (!navigator.mediaDevices?.getUserMedia) {
				cameraError = "Camera not supported in this browser. Tap Upload Photo to use your camera roll instead.";
				showUploadFallback = true;
				return;
			}

			if (stream) {
				stream.getTracks().forEach((t) => t.stop());
			}

			stream = await navigator.mediaDevices.getUserMedia(getConstraints());

			// Wait for Svelte to flush DOM updates so the video element exists
			await tick();

			if (videoEl) {
				videoEl.srcObject = stream;

				// Wait for actual video dimensions before marking ready.
				// On iPhone Safari, videoWidth/videoHeight are 0 until
				// the loadedmetadata event fires.
				await new Promise<void>((resolve, reject) => {
					const onMeta = () => {
						videoEl!.removeEventListener('loadedmetadata', onMeta);
						resolve();
					};
					videoEl!.addEventListener('loadedmetadata', onMeta);
					// If metadata already loaded (desktop Chrome fires synchronously),
					// resolve immediately
					if (videoEl!.readyState >= 1) {
						videoEl!.removeEventListener('loadedmetadata', onMeta);
						resolve();
					}
					// Timeout after 5 seconds
					setTimeout(() => reject(new Error('Video metadata timeout')), 5000);
				});

				await videoEl.play();

				// Final safety check: wait for non-zero dimensions
				if (videoEl.videoWidth === 0) {
					await new Promise<void>((resolve) => {
						const check = () => {
							if (videoEl && videoEl.videoWidth > 0) {
								resolve();
							} else {
								requestAnimationFrame(check);
							}
						};
						requestAnimationFrame(check);
						setTimeout(resolve, 2000); // give up after 2s
					});
				}

				cameraReady = true;
				showUploadFallback = false;
			}
		} catch (err) {
			if (err instanceof DOMException && err.name === 'NotAllowedError') {
				cameraError = "Camera access denied. Tap Upload Photo to use your camera roll instead.";
			} else if (err instanceof DOMException && err.name === 'NotFoundError') {
				cameraError = "No camera found on this device. Tap Upload Photo to select a photo.";
			} else if (err instanceof DOMException && err.name === 'NotReadableError') {
				cameraError = "Camera is in use by another app. Close other apps using the camera and try again.";
			} else if (err instanceof Error && err.message === 'Video metadata timeout') {
				cameraError = "Camera took too long to respond. Tap Upload Photo to use your camera roll instead.";
			} else {
				cameraError = "Camera unavailable. Tap Upload Photo to select a photo instead.";
			}
			showUploadFallback = true;
			cameraActive = false;
		} finally {
			clearTimeout(fallbackTimer);
			cameraLoading = false;
		}
	}

	export function resetCapture() {
		capturedPreview = null;
		detectedImageType = "";
		capturedImages = [];
		lastQuality = null;
		cameraActive = false;
		cameraReady = false;
		if (stream) {
			stream.getTracks().forEach((t) => t.stop());
			stream = null;
		}
	}

	function captureFrame() {
		if (!videoEl || !canvasEl || !cameraReady) return;

		const ctx = canvasEl.getContext("2d");
		if (!ctx) return;

		// Guard against zero-dimension video (can happen on iPhone if
		// metadata hasn't fully loaded despite the loadedmetadata event)
		const vw = videoEl.videoWidth;
		const vh = videoEl.videoHeight;
		if (vw === 0 || vh === 0) return;

		canvasEl.width = vw;
		canvasEl.height = vh;
		ctx.drawImage(videoEl, 0, 0);

		const imageData = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);
		const preview = canvasEl.toDataURL("image/jpeg", 0.9);
		const resolvedType = imageType === "auto" ? detectImageType(imageData) : imageType;
		detectedImageType = resolvedType === "dermoscopy" ? "Dermoscopy detected" : "Clinical photo detected";
		const quality = assessImageQuality(imageData);
		lastQuality = quality;

		if (multiCapture) {
			if (capturedImages.length < maxImages) {
				capturedImages = [...capturedImages, { imageData, preview, imageType: resolvedType, quality }];
			}
			// Auto-fire when max reached
			if (capturedImages.length >= maxImages) {
				fireMultiCapture();
			}
		} else {
			capturedPreview = preview;
			oncapture({ imageData, bodyLocation, deviceModel, imageType: resolvedType });
		}
	}

	function handleFileUpload(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		const reader = new FileReader();
		reader.onload = () => {
			const img = new Image();
			img.onload = () => {
				const canvas = document.createElement('canvas');
				canvas.width = img.width;
				canvas.height = img.height;
				const ctx = canvas.getContext('2d')!;
				ctx.drawImage(img, 0, 0);
				const imageData = ctx.getImageData(0, 0, img.width, img.height);
				const preview = canvas.toDataURL("image/jpeg", 0.9);
				const resolvedType = imageType === "auto" ? detectImageType(imageData) : imageType;
				detectedImageType = resolvedType === "dermoscopy" ? "Dermoscopy detected" : "Clinical photo detected";
				const quality = assessImageQuality(imageData);
				lastQuality = quality;

				if (multiCapture) {
					if (capturedImages.length < maxImages) {
						capturedImages = [...capturedImages, { imageData, preview, imageType: resolvedType, quality }];
					}
					if (capturedImages.length >= maxImages) {
						fireMultiCapture();
					}
				} else {
					capturedPreview = preview;
					oncapture({ imageData, bodyLocation, deviceModel, imageType: resolvedType });
				}
			};
			img.src = reader.result as string;
		};
		reader.readAsDataURL(file);
		// Reset input so the same file can be re-selected
		input.value = '';
	}

	function retake() {
		capturedPreview = null;
		detectedImageType = "";
		lastQuality = null;
	}

	function fireMultiCapture() {
		if (!onmulticapture || capturedImages.length < 2) return;
		const events: CaptureEvent[] = capturedImages.map((item) => ({
			imageData: item.imageData,
			bodyLocation,
			deviceModel,
			imageType: item.imageType,
		}));
		onmulticapture(events);
	}

	function removeMultiCaptureImage(index: number) {
		capturedImages = capturedImages.filter((_, i) => i !== index);
	}

	function resetMultiCapture() {
		capturedImages = [];
		detectedImageType = "";
		lastQuality = null;
	}

	function bodyLocationLabel(): string {
		if (bodyLocation === "unknown") return "Location";
		const labels: Record<string, string> = {
			head: "Head",
			neck: "Neck",
			trunk: "Trunk",
			upper_extremity: "Arms",
			lower_extremity: "Legs",
			palms_soles: "Hands/Feet",
			genital: "Genital",
		};
		return labels[bodyLocation] || bodyLocation;
	}

	// Don't auto-start camera — let user choose Take Photo or Upload
	// onMount(() => { startCamera(); });

	onDestroy(() => {
		if (stream) {
			stream.getTracks().forEach((t) => t.stop());
			stream = null;
		}
		cameraActive = false;
	});
</script>

<div class="flex flex-col gap-3 w-full">
	{#if !cameraActive && !cameraError && !capturedPreview && (!multiCapture || !multiCaptureDone)}
		<div class="flex flex-col sm:flex-row gap-3 justify-center py-6">
			{#if isMobileDevice}
				<!-- Mobile: native camera capture via file input -- most reliable on iOS Safari -->
				<label class="rounded-full bg-teal-600 px-8 py-4 text-base font-semibold text-white hover:bg-teal-500 active:scale-95 transition-all flex items-center gap-2.5 justify-center cursor-pointer {multiCaptureDone ? 'opacity-40 pointer-events-none' : ''}">
					<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
					Take Photo
					<input type="file" accept="image/*" capture="environment" class="hidden" onchange={handleFileUpload} />
				</label>
			{:else}
				<!-- Desktop: getUserMedia live preview -->
				<button onclick={startCamera} disabled={multiCaptureDone} class="rounded-full bg-teal-600 px-8 py-4 text-base font-semibold text-white hover:bg-teal-500 active:scale-95 transition-all flex items-center gap-2.5 justify-center disabled:opacity-40 disabled:pointer-events-none">
					<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
					Take Photo
				</button>
			{/if}
			<label class="rounded-full border border-white/[0.10] bg-white/[0.04] px-8 py-4 text-base font-semibold text-gray-200 cursor-pointer hover:bg-white/[0.08] active:scale-95 transition-all flex items-center gap-2.5 justify-center {multiCaptureDone ? 'opacity-40 pointer-events-none' : ''}">
				<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
				Upload Photo
				<input type="file" accept="image/*" class="hidden" onchange={handleFileUpload} />
			</label>
		</div>
		<p class="text-[10px] text-gray-500 text-center mt-2">
			Tip: Place a USB-C charger cable next to the spot for accurate size measurement
		</p>
	{/if}
	<!-- Camera / preview area -->
	{#if cameraActive || cameraError || capturedPreview}
	<div
		class="relative aspect-[3/4] sm:aspect-[4/3] w-full max-h-[55vh] sm:max-h-none overflow-hidden rounded-2xl bg-gray-900 border border-white/[0.06]"
	>
		{#if cameraError}
			<div
				class="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center"
			>
				<svg class="h-12 w-12 text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
				</svg>
				<p class="text-[15px] text-gray-400 leading-relaxed max-w-xs">{cameraError}</p>
				<div class="flex flex-col sm:flex-row gap-3">
					<button
						onclick={startCamera}
						class="rounded-full bg-teal-600 px-6 py-3 text-sm font-medium text-white hover:bg-teal-500 active:scale-95 transition-all focus:outline-none focus:ring-2 focus:ring-teal-500/40 touch-target"
					>
						Retry Camera
					</button>
					<label class="rounded-full border border-white/[0.08] bg-white/[0.03] px-6 py-3 text-sm font-medium text-gray-300 cursor-pointer hover:bg-white/[0.06] active:scale-95 transition-all touch-target flex items-center justify-center gap-2">
						<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
						Upload Photo
						<input type="file" accept="image/*" class="hidden" onchange={handleFileUpload} />
					</label>
				</div>
			</div>
		{:else if capturedPreview}
			<img
				src={capturedPreview}
				alt="Captured lesion"
				class="h-full w-full object-contain"
			/>
			<button
				onclick={retake}
				class="absolute bottom-3 right-3 rounded-full bg-gray-800/80 px-4 py-2.5 text-sm font-medium text-white backdrop-blur-sm hover:bg-gray-700/80 active:scale-95 transition-all touch-target"
			>
				Retake
			</button>
		{:else}
			<!-- Live camera feed -- bind:this so startCamera() can assign srcObject -->
			<!-- svelte-ignore element_invalid_self_closing_tag -->
			<video
				bind:this={videoEl}
				autoplay
				playsinline
				muted
				class="h-full w-full object-cover"
			/>
			{#if !cameraReady}
				<div class="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6">
					{#if showUploadFallback}
						<p class="text-sm text-gray-400 text-center">Camera is loading slowly...</p>
						<label class="rounded-full bg-teal-600 px-6 py-3 text-sm font-medium text-white cursor-pointer hover:bg-teal-500 active:scale-95 transition-all">
							Upload a Photo Instead
							<input type="file" accept="image/*" class="hidden" onchange={handleFileUpload} />
						</label>
					{:else}
						<div class="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
						<p class="text-xs text-gray-500">Starting camera...</p>
					{/if}
				</div>
			{/if}
		{/if}

		<!-- Body location badge (tap to open drawer) -->
		<button
			onclick={() => (showBodyMap = !showBodyMap)}
			class="absolute top-2 left-2 sm:top-3 sm:left-3 flex items-center gap-1.5 rounded-full bg-gray-900/80 px-3 py-2 text-[11px] font-medium backdrop-blur-sm transition-all touch-target
				{bodyLocation !== 'unknown' ? 'text-teal-400 border border-teal-500/40 shadow-sm shadow-teal-500/20' : 'text-gray-300 border-2 border-gray-500/50 animate-pulse'}"
			aria-label="Select body location"
		>
			<svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
			</svg>
			{bodyLocationLabel()}
			{#if bodyLocation === 'unknown'}
				<svg class="h-2.5 w-2.5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
				</svg>
			{/if}
		</button>

		<!-- Multi-capture counter badge -->
		{#if multiCapture}
			<div
				class="absolute top-2 right-2 sm:top-3 sm:right-3 flex items-center gap-1.5 rounded-full bg-gray-900/80 px-3 py-2 text-[11px] font-semibold text-teal-400 backdrop-blur-sm border border-teal-500/30"
				aria-live="polite"
			>
				<svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
				</svg>
				Photo {capturedImages.length} of {maxImages}
			</div>
		{/if}
	</div>
	{/if}

	<!-- Multi-capture thumbnail strip -->
	{#if multiCapture && capturedImages.length > 0}
		<div class="flex flex-col gap-2.5">
			<div class="flex items-center gap-3 overflow-x-auto pb-1 px-1" role="list" aria-label="Captured images">
				{#each capturedImages as item, i (i)}
					<div
						class="relative flex-shrink-0 h-14 w-14 rounded-xl overflow-hidden border-2 transition-all
							{item.quality.grade === 'poor' ? 'border-red-500/70' : i === capturedImages.length - 1 ? 'border-teal-500 shadow-md shadow-teal-500/20' : 'border-white/[0.08]'}"
						role="listitem"
					>
						<img
							src={item.preview}
							alt="Captured photo {i + 1}"
							class="h-full w-full object-cover"
						/>
						<!-- Quality indicator dot -->
						<div class="absolute top-0.5 left-0.5 flex items-center justify-center h-4 w-4 rounded-full
							{item.quality.grade === 'good' ? 'bg-green-500' : item.quality.grade === 'acceptable' ? 'bg-yellow-500' : 'bg-red-500'}">
							{#if item.quality.grade === 'acceptable'}
								<svg class="h-2.5 w-2.5 text-yellow-900" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M12 9v2m0 4h.01"></path></svg>
							{:else if item.quality.grade === 'poor'}
								<svg class="h-2.5 w-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M6 18L18 6M6 6l12 12"></path></svg>
							{/if}
						</div>
						<button
							onclick={() => removeMultiCaptureImage(i)}
							class="absolute -top-0.5 -right-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-gray-900/90 text-gray-300 hover:text-white hover:bg-red-600/80 transition-all border border-white/[0.10]"
							aria-label="Remove photo {i + 1}"
						>
							<svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12"></path>
							</svg>
						</button>
						<div class="absolute bottom-0 left-0 right-0 bg-gray-900/70 text-[9px] text-center text-gray-300 py-0.5">
							{item.imageType === 'dermoscopy' ? 'Derm' : 'Clin'}
						</div>
					</div>
				{/each}

				<!-- Empty slots with + icon -->
				{#each Array(maxImages - capturedImages.length) as _, i (i)}
					<div
						class="flex-shrink-0 h-14 w-14 rounded-xl border-2 border-dashed border-white/[0.08] flex flex-col items-center justify-center gap-0.5"
						role="listitem"
						aria-label="Empty slot {capturedImages.length + i + 1}"
					>
						<svg class="h-4 w-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
						</svg>
					</div>
				{/each}
			</div>

			<!-- Quality suggestion banner -->
			{#if qualitySuggestion}
				{@const hasPoor = capturedImages.some((img) => img.quality.grade === 'poor')}
				<div class="flex items-start gap-2 rounded-xl px-3 py-2.5 text-[12px] leading-relaxed
					{hasPoor ? 'bg-red-500/10 border border-red-500/30 text-red-400' : 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-400'}">
					{#if hasPoor}
						<svg class="h-4 w-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
						<span><strong>Retake recommended.</strong> {qualitySuggestion}</span>
					{:else}
						<svg class="h-4 w-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
						<span>{qualitySuggestion}</span>
					{/if}
				</div>
			{/if}

			<!-- Instruction text -->
			{#if capturedImages.length === 1}
				<p class="text-[11px] text-teal-400/70 text-center">Add 1-2 more photos of the same spot for better accuracy</p>
			{/if}

			<!-- Done button (visible when >= 2 images captured) -->
			{#if capturedImages.length >= 2}
				<button
					onclick={fireMultiCapture}
					class="w-full rounded-full bg-teal-600 px-6 py-3.5 text-sm font-semibold text-white hover:bg-teal-500 active:scale-[0.98] transition-all focus:outline-none focus:ring-2 focus:ring-teal-500/40 flex items-center justify-center gap-2"
				>
					<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
					</svg>
					Done &mdash; Analyze All ({capturedImages.length})
				</button>
			{/if}

			<!-- Reset multi-capture -->
			{#if capturedImages.length > 0}
				<button
					onclick={resetMultiCapture}
					class="mx-auto text-[11px] text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
				>
					Clear all and start over
				</button>
			{/if}
		</div>
	{/if}

	<!-- Capture button row (only when camera is live) -->
	{#if stream && !capturedPreview && !cameraError}
		<div class="flex flex-col items-center gap-3">
			<button
				onclick={captureFrame}
				disabled={!cameraReady || multiCaptureDone}
				class="flex h-16 w-16 items-center justify-center rounded-full bg-white shadow-lg shadow-white/10 transition-all active:scale-90 disabled:opacity-40 focus:outline-none focus:ring-4 focus:ring-teal-500/30"
				aria-label="Capture image"
			>
				<CarbonCamera class="h-6 w-6 sm:h-7 sm:w-7 text-gray-900" />
			</button>
			<div class="flex items-center justify-center gap-2.5 flex-wrap">
				<label class="flex items-center gap-1.5 rounded-full border border-white/[0.06] bg-white/[0.02] px-3.5 py-2.5 text-xs text-gray-400 hover:bg-white/[0.04] cursor-pointer transition-all touch-target {multiCaptureDone ? 'opacity-40 pointer-events-none' : ''}">
					<svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
					Upload
					<input type="file" accept="image/*" class="hidden" onchange={handleFileUpload} />
				</label>
				<select
					bind:value={deviceModel}
					onchange={() => startCamera()}
					class="rounded-full border border-white/[0.06] bg-white/[0.02] px-3 py-2.5 text-xs text-gray-400 touch-target"
					aria-label="Select device"
				>
					{#each DEVICE_MODELS as dev}
						<option value={dev.value}>{dev.label}</option>
					{/each}
				</select>
			</div>
		</div>
	{/if}

	<canvas bind:this={canvasEl} class="hidden"></canvas>

	<!-- Image type detection result (single-capture only; multi-capture shows type on thumbnails) -->
	{#if detectedImageType && !multiCapture}
		<div class="mx-auto flex items-center gap-1.5 rounded-full border border-teal-500/30 bg-teal-500/10 px-3 py-1 text-[11px] text-teal-400">
			<svg class="h-2.5 w-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
			{detectedImageType}
		</div>
	{/if}

	<!-- Single-capture quality feedback -->
	{#if lastQuality && !multiCapture && lastQuality.grade !== 'good'}
		<div class="flex items-start gap-2 rounded-xl px-3 py-2.5 text-[12px] leading-relaxed
			{lastQuality.grade === 'poor' ? 'bg-red-500/10 border border-red-500/30 text-red-400' : 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-400'}">
			{#if lastQuality.grade === 'poor'}
				<svg class="h-4 w-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
				<span><strong>Retake recommended.</strong> {lastQuality.suggestion}</span>
			{:else}
				<svg class="h-4 w-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
				<span>{lastQuality.suggestion}</span>
			{/if}
		</div>
	{/if}
</div>

<!-- Body map slide-out drawer -->
{#if showBodyMap}
	<!-- Backdrop (prevents body scroll) -->
	<div
		class="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
		onclick={() => (showBodyMap = false)}
		ontouchmove={(e) => e.preventDefault()}
		role="presentation"
	></div>
	<!-- Drawer: full-height on mobile, narrower on desktop -->
	<div class="fixed left-0 top-0 bottom-0 z-50 w-[75vw] max-w-[280px] sm:w-56 bg-[#0a0a0f] border-r border-white/[0.06] p-5 overflow-y-auto overscroll-none animate-fadeIn shadow-2xl">
		<div class="flex items-center justify-between mb-4">
			<h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400">Body Location</h3>
			<button
				onclick={() => (showBodyMap = false)}
				class="rounded-full p-2 sm:p-1 text-gray-500 hover:text-gray-300 transition-colors touch-target"
				aria-label="Close body map"
			>
				<svg class="h-5 w-5 sm:h-4 sm:w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
				</svg>
			</button>
		</div>
		<BodyMap
			selected={bodyLocation}
			onselect={(loc) => { bodyLocation = loc; showBodyMap = false; }}
		/>

		<!-- Image type selector -->
		<div class="mt-4">
			<label class="flex flex-col gap-1">
				<span class="text-[10px] font-medium text-gray-500">Image Type</span>
				<select
					bind:value={imageType}
					class="h-10 sm:h-8 rounded-lg border border-gray-700 bg-gray-800 px-2 text-xs text-gray-200 touch-target"
				>
					{#each IMAGE_TYPES as t}
						<option value={t.value}>{t.label}</option>
					{/each}
				</select>
			</label>
		</div>
	</div>
{/if}
