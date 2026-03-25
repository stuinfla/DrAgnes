<script lang="ts">
	import type { BodyLocation } from "$lib/mela/types";

	interface Props {
		selected: BodyLocation;
		onselect: (location: BodyLocation) => void;
	}

	let { selected, onselect }: Props = $props();
	let hoveredRegion: BodyLocation | null = $state(null);

	interface BodyRegion {
		id: BodyLocation;
		label: string;
		elements: Array<{
			type: "ellipse" | "rect" | "path" | "circle";
			attrs: Record<string, string | number>;
		}>;
	}

	const REGIONS: BodyRegion[] = [
		{
			id: "head",
			label: "Head",
			elements: [
				{ type: "ellipse", attrs: { cx: 100, cy: 35, rx: 25, ry: 30 } },
			],
		},
		{
			id: "neck",
			label: "Neck",
			elements: [
				{ type: "rect", attrs: { x: 88, y: 63, width: 24, height: 20, rx: 4 } },
			],
		},
		{
			id: "trunk",
			label: "Trunk",
			elements: [
				{ type: "rect", attrs: { x: 65, y: 83, width: 70, height: 110, rx: 8 } },
			],
		},
		{
			id: "upper_extremity",
			label: "Arms",
			elements: [
				// Left arm
				{
					type: "path",
					attrs: {
						d: "M65 90 C55 90, 40 110, 32 140 L26 190 C24 200, 22 215, 25 225 L30 225 C32 218, 34 205, 36 195 L44 150 C48 135, 55 115, 60 100 Z",
					},
				},
				// Right arm
				{
					type: "path",
					attrs: {
						d: "M135 90 C145 90, 160 110, 168 140 L174 190 C176 200, 178 215, 175 225 L170 225 C168 218, 166 205, 164 195 L156 150 C152 135, 145 115, 140 100 Z",
					},
				},
			],
		},
		{
			id: "lower_extremity",
			label: "Legs",
			elements: [
				// Left leg
				{
					type: "path",
					attrs: {
						d: "M72 208 C70 230, 68 260, 66 290 L62 340 C60 355, 58 365, 56 375 L76 375 C74 365, 73 355, 72 340 L76 290 C78 260, 82 230, 88 208 Z",
					},
				},
				// Right leg
				{
					type: "path",
					attrs: {
						d: "M112 208 C118 230, 122 260, 124 290 L128 340 C129 355, 130 365, 132 375 L144 375 C142 365, 140 355, 138 340 L134 290 C132 260, 130 230, 128 208 Z",
					},
				},
			],
		},
		{
			id: "palms_soles",
			label: "Hands / Feet",
			elements: [
				// Left hand
				{ type: "circle", attrs: { cx: 24, cy: 237, r: 12 } },
				// Right hand
				{ type: "circle", attrs: { cx: 176, cy: 237, r: 12 } },
				// Left foot
				{ type: "ellipse", attrs: { cx: 63, cy: 387, rx: 14, ry: 10 } },
				// Right foot
				{ type: "ellipse", attrs: { cx: 137, cy: 387, rx: 14, ry: 10 } },
			],
		},
		{
			id: "genital",
			label: "Genital",
			elements: [
				{ type: "rect", attrs: { x: 85, y: 192, width: 30, height: 16, rx: 6 } },
			],
		},
	];

	function isActive(regionId: BodyLocation): boolean {
		return selected === regionId;
	}

	function isHovered(regionId: BodyLocation): boolean {
		return hoveredRegion === regionId;
	}

	function getFill(regionId: BodyLocation): string {
		if (isActive(regionId)) return "rgba(20, 184, 166, 0.30)";
		if (isHovered(regionId)) return "rgba(20, 184, 166, 0.15)";
		return "rgba(107, 114, 128, 0.08)";
	}

	function getStroke(regionId: BodyLocation): string {
		if (isActive(regionId)) return "rgba(20, 184, 166, 0.8)";
		if (isHovered(regionId)) return "rgba(20, 184, 166, 0.4)";
		return "rgba(156, 163, 175, 0.3)";
	}

	function getStrokeWidth(regionId: BodyLocation): number {
		if (isActive(regionId)) return 2;
		if (isHovered(regionId)) return 1.5;
		return 1;
	}

	function getFilter(regionId: BodyLocation): string {
		if (isActive(regionId)) return "url(#glow)";
		return "none";
	}

	function handleClick(regionId: BodyLocation) {
		onselect(regionId);
	}

	function handleMouseEnter(regionId: BodyLocation) {
		hoveredRegion = regionId;
	}

	function handleMouseLeave() {
		hoveredRegion = null;
	}

	/** Compute tooltip position from region elements (rough centroid) */
	function getTooltipPos(region: BodyRegion): { x: number; y: number } {
		const el = region.elements[0];
		switch (el.type) {
			case "ellipse":
				return { x: el.attrs.cx as number, y: (el.attrs.cy as number) - (el.attrs.ry as number) - 8 };
			case "circle":
				return { x: el.attrs.cx as number, y: (el.attrs.cy as number) - (el.attrs.r as number) - 8 };
			case "rect":
				return {
					x: (el.attrs.x as number) + (el.attrs.width as number) / 2,
					y: (el.attrs.y as number) - 8,
				};
			case "path":
				// For arms, show tooltip to the side
				if (region.id === "upper_extremity") return { x: 100, y: 130 };
				if (region.id === "lower_extremity") return { x: 100, y: 280 };
				return { x: 100, y: 200 };
			default:
				return { x: 100, y: 200 };
		}
	}

	function selectedLabel(): string {
		if (selected === "unknown") return "Tap body region";
		return REGIONS.find((r) => r.id === selected)?.label ?? "Unknown";
	}
</script>

<div class="flex flex-col items-center">
	<svg
		viewBox="0 0 200 400"
		class="w-full max-w-[200px] sm:max-w-[180px] h-auto"
		xmlns="http://www.w3.org/2000/svg"
		role="img"
		aria-label="Body map for selecting lesion location"
	>
		<defs>
			<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
				<feGaussianBlur stdDeviation="4" result="blur" />
				<feMerge>
					<feMergeNode in="blur" />
					<feMergeNode in="SourceGraphic" />
				</feMerge>
			</filter>
		</defs>

		<!-- Body outline silhouette (non-interactive background) -->
		<path
			d="M100 5
			   C120 5, 125 18, 125 35
			   C125 52, 120 63, 112 65
			   L112 83
			   C140 85, 158 100, 170 140
			   L178 200
			   C180 212, 178 225, 176 237
			   C172 255, 165 240, 162 225
			   L155 170
			   C148 148, 140 120, 135 100
			   L135 193
			   C138 220, 142 260, 142 290
			   L146 345
			   C148 360, 150 375, 148 387
			   C145 398, 130 400, 128 387
			   L124 340
			   C122 310, 118 260, 112 210
			   L100 208
			   L88 210
			   C82 260, 78 310, 76 340
			   L72 387
			   C70 400, 55 398, 52 387
			   C50 375, 52 360, 54 345
			   L58 290
			   C58 260, 62 220, 65 193
			   L65 100
			   C60 120, 52 148, 45 170
			   L38 225
			   C35 240, 28 255, 24 237
			   C22 225, 20 212, 22 200
			   L30 140
			   C42 100, 60 85, 88 83
			   L88 65
			   C80 63, 75 52, 75 35
			   C75 18, 80 5, 100 5 Z"
			fill="none"
			stroke="rgba(107, 114, 128, 0.35)"
			stroke-width="1"
			stroke-linejoin="round"
		/>

		<!-- Interactive regions -->
		{#each REGIONS as region}
			<g
				role="button"
				tabindex="0"
				aria-label="Select {region.label} body region"
				class="cursor-pointer outline-none"
				onclick={() => handleClick(region.id)}
				onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleClick(region.id); } }}
				onmouseenter={() => handleMouseEnter(region.id)}
				onmouseleave={handleMouseLeave}
				onfocus={() => handleMouseEnter(region.id)}
				onblur={handleMouseLeave}
				ontouchstart={() => handleMouseEnter(region.id)}
				ontouchend={() => { handleMouseLeave(); handleClick(region.id); }}
			>
				{#each region.elements as el}
					{#if el.type === "ellipse"}
						<ellipse
							cx={el.attrs.cx}
							cy={el.attrs.cy}
							rx={el.attrs.rx}
							ry={el.attrs.ry}
							fill={getFill(region.id)}
							stroke={getStroke(region.id)}
							stroke-width={getStrokeWidth(region.id)}
							filter={getFilter(region.id)}
							style="transition: fill 0.2s ease, stroke 0.2s ease, stroke-width 0.15s ease;"
						/>
					{:else if el.type === "circle"}
						<circle
							cx={el.attrs.cx}
							cy={el.attrs.cy}
							r={el.attrs.r}
							fill={getFill(region.id)}
							stroke={getStroke(region.id)}
							stroke-width={getStrokeWidth(region.id)}
							filter={getFilter(region.id)}
							style="transition: fill 0.2s ease, stroke 0.2s ease, stroke-width 0.15s ease;"
						/>
					{:else if el.type === "rect"}
						<rect
							x={el.attrs.x}
							y={el.attrs.y}
							width={el.attrs.width}
							height={el.attrs.height}
							rx={el.attrs.rx}
							fill={getFill(region.id)}
							stroke={getStroke(region.id)}
							stroke-width={getStrokeWidth(region.id)}
							filter={getFilter(region.id)}
							style="transition: fill 0.2s ease, stroke 0.2s ease, stroke-width 0.15s ease;"
						/>
					{:else if el.type === "path"}
						<path
							d={String(el.attrs.d)}
							fill={getFill(region.id)}
							stroke={getStroke(region.id)}
							stroke-width={getStrokeWidth(region.id)}
							filter={getFilter(region.id)}
							style="transition: fill 0.2s ease, stroke 0.2s ease, stroke-width 0.15s ease;"
						/>
					{/if}
				{/each}

				<!-- Hover tooltip -->
				{#if hoveredRegion === region.id}
					{@const pos = getTooltipPos(region)}
					<text
						x={pos.x}
						y={pos.y}
						text-anchor="middle"
						font-size="11"
						fill="rgba(94, 234, 212, 0.9)"
						font-family="system-ui, sans-serif"
						font-weight="500"
						style="pointer-events: none;"
					>
						{region.label}
					</text>
				{/if}
			</g>
		{/each}
	</svg>

	<p class="text-center text-sm sm:text-xs mt-3 sm:mt-2 font-medium {selected === 'unknown' ? 'text-gray-500' : 'text-teal-400'}">
		{selectedLabel()}
	</p>
</div>
