"""
Script 3: Create Train / Dev / Test Splits

What this script does:
  1. Loads processed_manifest.csv from Script 2
  2. Creates speaker-independent splits (80/10/10 by child_id, stratified by region)
  3. Creates ASR subset splits (only connected speech: Sentence + Paragraph + Story)
  4. Verifies zero speaker overlap between splits
  5. Prints comprehensive final statistics (by split, language, reading level)

Input:  ASER-Dataset/processed_manifest.csv
Output: ASER-Dataset/splits/train.csv, dev.csv, test.csv,
        asr_train.csv, asr_dev.csv, asr_test.csv

Why split by child, not by clip?
  A child's voice is unique. If the same child appears in both train and test,
  the model can "memorize" that voice and inflate test scores — speaker leakage.
  Splitting by child_id guarantees the test set contains only unheard voices.
"""

import csv
import random
import statistics
from pathlib import Path
from collections import defaultdict

random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
INPUT_CSV    = ASER_ROOT / "processed_manifest.csv"
SPLITS_DIR   = ASER_ROOT / "splits"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

ASR_LEVELS = {"Paragraph", "Story", "Sentence (English)"}

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load processed manifest
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading processed manifest...")
print("=" * 70)

with open(INPUT_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_clips = list(reader)
    fieldnames = reader.fieldnames

total_clips = len(all_clips)
total_hrs = sum(float(r["duration_sec"]) for r in all_clips) / 3600
total_children = len(set(r["child_id"] for r in all_clips))

print(f"Loaded: {total_clips:,} clips | {total_hrs:.2f} hours | {total_children:,} children")
print(f"Fields: {fieldnames}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Group children by region for stratified splitting
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 2: Grouping children by region...")
print("=" * 70)

child_region = {}
for r in all_clips:
    child_region[r["child_id"]] = r["region"]

region_children = defaultdict(list)
for cid, reg in child_region.items():
    region_children[reg].append(cid)

print("Children per region:")
for reg in sorted(region_children):
    print(f"  {reg}: {len(region_children[reg])} children")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Create speaker-independent splits (80/10/10)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Creating speaker-independent splits...")
print("=" * 70)

train_ids, dev_ids, test_ids = set(), set(), set()

for reg in sorted(region_children):
    children = region_children[reg][:]
    random.shuffle(children)
    n = len(children)
    n_test = max(1, int(n * 0.10))
    n_dev  = max(1, int(n * 0.10))

    test_ids.update(children[:n_test])
    dev_ids.update(children[n_test:n_test + n_dev])
    train_ids.update(children[n_test + n_dev:])

    print(f"  {reg}: train={n - n_test - n_dev}, dev={n_dev}, test={n_test}")

print(f"\nTotal children: train={len(train_ids)}, dev={len(dev_ids)}, test={len(test_ids)}")
print(f"Sum: {len(train_ids) + len(dev_ids) + len(test_ids)} (should be {total_children})\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Verify zero speaker overlap
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Verifying zero speaker overlap...")
print("=" * 70)

overlap_train_dev  = train_ids & dev_ids
overlap_train_test = train_ids & test_ids
overlap_dev_test   = dev_ids & test_ids

print("Speaker Leakage Check:")
print(f"  Train ∩ Dev  = {len(overlap_train_dev)} children  (must be 0)")
print(f"  Train ∩ Test = {len(overlap_train_test)} children  (must be 0)")
print(f"  Dev   ∩ Test = {len(overlap_dev_test)} children  (must be 0)")

assert len(overlap_train_dev) == 0, "LEAKAGE: train and dev share children!"
assert len(overlap_train_test) == 0, "LEAKAGE: train and test share children!"
assert len(overlap_dev_test) == 0, "LEAKAGE: dev and test share children!"

print("\n  ALL CLEAR — zero speaker overlap between any splits.\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Assign clips to splits and save
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Assigning clips to splits and saving...")
print("=" * 70)

train_clips = [r for r in all_clips if r["child_id"] in train_ids]
dev_clips   = [r for r in all_clips if r["child_id"] in dev_ids]
test_clips  = [r for r in all_clips if r["child_id"] in test_ids]

asr_train = [r for r in train_clips if r["reading_level"] in ASR_LEVELS]
asr_dev   = [r for r in dev_clips   if r["reading_level"] in ASR_LEVELS]
asr_test  = [r for r in test_clips  if r["reading_level"] in ASR_LEVELS]

splits = {
    "train": train_clips,
    "dev": dev_clips,
    "test": test_clips,
    "asr_train": asr_train,
    "asr_dev": asr_dev,
    "asr_test": asr_test,
}

for name, clips in splits.items():
    out_path = SPLITS_DIR / f"{name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clips)
    hrs = sum(float(r["duration_sec"]) for r in clips) / 3600
    children = len(set(r["child_id"] for r in clips))
    print(f"  {name + '.csv':<18} {len(clips):>6,} clips | {children:>5} children | {hrs:>6.2f} hrs  -> {out_path}")

print(f"\nAll splits saved to: {SPLITS_DIR}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Full split statistics
# ══════════════════════════════════════════════════════════════════════════════
def fmt_hrs(clips):
    return sum(float(r["duration_sec"]) for r in clips) / 3600

print("=" * 70)
print("  FINAL DATASET SPLITS — OVERALL SUMMARY")
print("=" * 70)
print(f"  {'Split':<12} {'Clips':>8} {'Children':>10} {'Hours':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")

for name, clips in [("Train", train_clips), ("Dev", dev_clips), ("Test", test_clips)]:
    children = len(set(r["child_id"] for r in clips))
    print(f"  {name:<12} {len(clips):>8,} {children:>10} {fmt_hrs(clips):>8.2f}")

print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
print(f"  {'TOTAL':<12} {total_clips:>8,} {total_children:>10} {total_hrs:>8.2f}")

asr_total = asr_train + asr_dev + asr_test

print(f"\n  ASR subset (Sentence + Paragraph + Story only):")
print(f"  {'Split':<12} {'Clips':>8} {'Children':>10} {'Hours':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
for name, clips in [("ASR Train", asr_train), ("ASR Dev", asr_dev), ("ASR Test", asr_test)]:
    children = len(set(r["child_id"] for r in clips))
    print(f"  {name:<12} {len(clips):>8,} {children:>10} {fmt_hrs(clips):>8.2f}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
print(f"  {'ASR TOTAL':<12} {len(asr_total):>8,} {len(set(r['child_id'] for r in asr_total)):>10} {fmt_hrs(asr_total):>8.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Language-wise breakdown per split
# ══════════════════════════════════════════════════════════════════════════════
languages = sorted(set(r["language"] for r in all_clips))

print(f"\n{'=' * 70}")
print("  LANGUAGE-WISE BREAKDOWN — FULL SPLITS")
print("=" * 70)
print(f"  {'Language':<15}", end="")
for split_name in ["Train", "Dev", "Test", "Total"]:
    print(f" {'Clips':>7} {'Hours':>6}  |", end="")
print()
print(f"  {'-'*15}", end="")
for _ in range(4):
    print(f" {'-'*7} {'-'*6}  |", end="")
print()

split_data = [("Train", train_clips), ("Dev", dev_clips), ("Test", test_clips), ("Total", all_clips)]

for lang in languages:
    print(f"  {lang:<15}", end="")
    for sname, sclips in split_data:
        lang_clips = [r for r in sclips if r["language"] == lang]
        hrs = fmt_hrs(lang_clips)
        print(f" {len(lang_clips):>7,} {hrs:>6.2f}  |", end="")
    print()

print(f"\n{'=' * 70}")
print("  LANGUAGE-WISE BREAKDOWN — ASR SPLITS")
print("=" * 70)
print(f"  {'Language':<15}", end="")
for split_name in ["Train", "Dev", "Test", "Total"]:
    print(f" {'Clips':>7} {'Hours':>6}  |", end="")
print()
print(f"  {'-'*15}", end="")
for _ in range(4):
    print(f" {'-'*7} {'-'*6}  |", end="")
print()

asr_split_data = [("Train", asr_train), ("Dev", asr_dev), ("Test", asr_test), ("Total", asr_total)]

for lang in languages:
    print(f"  {lang:<15}", end="")
    for sname, sclips in asr_split_data:
        lang_clips = [r for r in sclips if r["language"] == lang]
        hrs = fmt_hrs(lang_clips)
        print(f" {len(lang_clips):>7,} {hrs:>6.2f}  |", end="")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Reading level breakdown per split
# ══════════════════════════════════════════════════════════════════════════════
reading_levels = sorted(set(r["reading_level"] for r in all_clips))

print(f"\n{'=' * 80}")
print("  READING LEVEL BREAKDOWN — FULL SPLITS")
print("=" * 80)
print(f"  {'Level':<25} {'ASR?':<5}", end="")
for split_name in ["Train", "Dev", "Test", "Total"]:
    print(f" {'Clips':>7} {'Hours':>6}  |", end="")
print()
print(f"  {'-'*25} {'-'*5}", end="")
for _ in range(4):
    print(f" {'-'*7} {'-'*6}  |", end="")
print()

for lvl in reading_levels:
    asr_mark = "YES" if lvl in ASR_LEVELS else ""
    print(f"  {lvl:<25} {asr_mark:<5}", end="")
    for sname, sclips in split_data:
        lvl_clips = [r for r in sclips if r["reading_level"] == lvl]
        hrs = fmt_hrs(lvl_clips)
        print(f" {len(lvl_clips):>7,} {hrs:>6.2f}  |", end="")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Region balance verification
# ══════════════════════════════════════════════════════════════════════════════
regions = sorted(set(r["region"] for r in all_clips))

print(f"\n{'=' * 70}")
print("  REGION BALANCE ACROSS SPLITS")
print("=" * 70)
print(f"  {'Region':<15}", end="")
for split_name in ["Train", "Dev", "Test"]:
    print(f" {'Children':>10} {'%':>6}  |", end="")
print()
print(f"  {'-'*15}", end="")
for _ in range(3):
    print(f" {'-'*10} {'-'*6}  |", end="")
print()

for reg in regions:
    print(f"  {reg:<15}", end="")
    for sname, sids in [("Train", train_ids), ("Dev", dev_ids), ("Test", test_ids)]:
        reg_in_split = [cid for cid in sids if child_region[cid] == reg]
        total_in_reg = len(region_children[reg])
        pct = len(reg_in_split) / total_in_reg * 100
        print(f" {len(reg_in_split):>10} {pct:>5.1f}%  |", end="")
    print()

print(f"\n  Target: ~80% / ~10% / ~10% per region. Small rounding differences are normal.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 10: Duration distribution summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  DURATION DISTRIBUTION PER SPLIT")
print("=" * 70)
print(f"  {'Split':<15} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}  (seconds)")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for name, clips in [("Train", train_clips), ("Dev", dev_clips), ("Test", test_clips),
                     ("ASR Train", asr_train), ("ASR Dev", asr_dev), ("ASR Test", asr_test)]:
    durs = [float(r["duration_sec"]) for r in clips]
    if durs:
        print(f"  {name:<15} {statistics.mean(durs):>8.2f} {statistics.median(durs):>8.2f} {min(durs):>8.2f} {max(durs):>8.2f}")

print(f"\n  All clips are between 0.5s and 30s: ", end="")
all_durs = [float(r["duration_sec"]) for r in all_clips]
if min(all_durs) >= 0.5 and max(all_durs) <= 30.0:
    print("YES")
else:
    print(f"NO (min={min(all_durs):.2f}s, max={max(all_durs):.2f}s)")

print("\n" + "=" * 70)
print("  DONE! All splits created. Ready for model training.")
print("=" * 70)
