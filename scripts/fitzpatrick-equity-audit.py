#!/usr/bin/env python3
"""Fitzpatrick skin tone equity analysis for Mela 7-class classifier."""

import csv, io, json, sys, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/master/fitzpatrick17k.csv"
MANIFEST = Path(__file__).parent / "dataset-manifests" / "fitzpatrick17k.json"
OUT_FILE = Path(__file__).parent / "fitzpatrick-equity-report.json"

CLASS_NAMES = {"mel": "Melanoma", "nv": "Melanocytic Nevi", "bcc": "Basal Cell Carcinoma",
               "akiec": "Actinic Keratosis", "bkl": "Benign Keratosis", "df": "Dermatofibroma",
               "vasc": "Vascular Lesion"}
SKIN_TYPES = [1, 2, 3, 4, 5, 6]
SKIN_LABELS = {-1: "Unknown", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# Load class mapping from manifest
with open(MANIFEST) as f:
    mapping = json.load(f)["class_mapping"]

# Download and parse CSV
print("Downloading Fitzpatrick17k CSV...")
resp = urllib.request.urlopen(CSV_URL)
text = resp.read().decode("utf-8")
reader = csv.DictReader(io.StringIO(text))
rows = list(reader)
print(f"  Total rows: {len(rows)}")

# Filter to our 7 classes and build cross-tabulation
cross = defaultdict(Counter)  # cross[mela_class][skin_type] = count
total_by_skin = Counter()
mapped_rows = []
for row in rows:
    label = row["label"].strip().lower()
    cls = mapping.get(label)
    if cls is None:
        continue
    try:
        ft = int(float(row.get("fitzpatrick_scale") or row.get("fitzpatrick") or -1))
    except (ValueError, TypeError):
        ft = -1
    cross[cls][ft] += 1
    total_by_skin[ft] += 1
    mapped_rows.append(row)

print(f"  Mapped to 7 classes: {len(mapped_rows)}")

# Print summary table
hdr = f"{'Class':<8} {'Name':<22}" + "".join(f"{'T'+SKIN_LABELS[s]:>6}" for s in [-1]+SKIN_TYPES) + f"{'Total':>8}"
print(f"\n{hdr}\n{'='*len(hdr)}")
class_totals = {}
for cls in sorted(cross.keys()):
    row_counts = [cross[cls][s] for s in [-1]+SKIN_TYPES]
    total = sum(row_counts)
    class_totals[cls] = total
    name = CLASS_NAMES.get(cls, cls)[:22]
    line = f"{cls:<8} {name:<22}" + "".join(f"{c:>6}" for c in row_counts) + f"{total:>8}"
    print(line)
totals_line = f"{'TOTAL':<8} {'':<22}" + "".join(f"{total_by_skin[s]:>6}" for s in [-1]+SKIN_TYPES) + f"{sum(total_by_skin.values()):>8}"
print(f"{'='*len(hdr)}\n{totals_line}")

# Equity gap analysis: for each class, compute ratio of darkest (V+VI) to lightest (I+II)
print("\n--- Equity Gap Analysis (Dark:Light ratio per class) ---")
print(f"  Dark = Type V + VI, Light = Type I + II")
print(f"  Ratio < 0.5 = significant underrepresentation of dark skin\n")
gaps = []
for cls in sorted(cross.keys()):
    dark = cross[cls][5] + cross[cls][6]
    light = cross[cls][1] + cross[cls][2]
    ratio = dark / light if light > 0 else 0.0
    gaps.append({"class": cls, "name": CLASS_NAMES.get(cls, cls), "dark": dark,
                 "light": light, "ratio": round(ratio, 3)})
    flag = " ** EQUITY GAP" if ratio < 0.5 else ""
    print(f"  {cls:<8} dark={dark:>4}  light={light:>4}  ratio={ratio:.3f}{flag}")

# Per skin-type worst gaps (classes with < 10 samples)
print("\n--- Critical Gaps (< 10 samples) ---")
critical = []
for cls in sorted(cross.keys()):
    for s in SKIN_TYPES:
        count = cross[cls][s]
        if count < 10:
            entry = {"class": cls, "skin_type": s, "skin_label": SKIN_LABELS[s], "count": count}
            critical.append(entry)
            print(f"  {cls:<8} Type {SKIN_LABELS[s]:<3} = {count} samples")
if not critical:
    print("  None found")

# Build report
report = {
    "dataset": "Fitzpatrick17k (filtered to Mela 7 classes)",
    "total_mapped": len(mapped_rows),
    "cross_tabulation": {cls: {str(s): cross[cls][s] for s in [-1]+SKIN_TYPES} for cls in sorted(cross)},
    "totals_by_skin_type": {SKIN_LABELS[s]: total_by_skin[s] for s in [-1]+SKIN_TYPES},
    "totals_by_class": {cls: class_totals[cls] for cls in sorted(class_totals)},
    "equity_gaps": sorted(gaps, key=lambda g: g["ratio"]),
    "critical_gaps": critical,
    "recommendation": "Prioritize augmentation and targeted data collection for dark-skin underrepresented classes."
}
with open(OUT_FILE, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to {OUT_FILE.name}")
