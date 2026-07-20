"""
ASER Dataset - Full EDA + Train/Dev/Test Split Generation

What this script does and why:
  1. Loads the manifest CSV built in the previous step
  2. Computes exact audio duration for every clip using ffprobe
  3. Prints a full researcher-facing dataset report
  4. Creates stratified train/dev/test splits — split BY CHILD (not by clip)
     so no child's voice leaks across splits (avoids speaker-dependent evaluation)
  5. Saves three manifest files: train.csv, dev.csv, test.csv
"""

import csv
import subprocess
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

MANIFEST    = Path("/home/hp/Indain_children_spech/ASER-Dataset/manifest.csv")
OUT_DIR     = Path("/home/hp/Indain_children_spech/ASER-Dataset")

# ── 1. Load manifest ───────────────────────────────────────────────────────────
print("Loading manifest...")
records = []
with open(MANIFEST, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

print(f"Loaded {len(records)} records.")

# ── 2. Compute exact duration via ffprobe ──────────────────────────────────────
print("\nComputing exact duration for all clips (this may take ~5-10 minutes)...")

def get_duration(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        val = result.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

total_done = 0
for r in records:
    dur = get_duration(r["audio_path"])
    r["duration_sec"] = dur
    total_done += 1
    if total_done % 5000 == 0:
        print(f"  {total_done}/{len(records)} done...")

print(f"  Duration computation complete.")

# Save manifest with durations
with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
print(f"  Manifest updated with durations.")

# ── 3. Full Statistics ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ASER DATASET — FULL RESEARCH PROFILE")
print("="*65)

total_clips   = len(records)
total_secs    = sum(float(r["duration_sec"]) for r in records)
total_hrs     = total_secs / 3600
total_children= len(set(r["child_id"] for r in records))

print(f"\n[OVERVIEW]")
print(f"  Total audio clips    : {total_clips:,}")
print(f"  Unique children      : {total_children:,}")
print(f"  Total duration       : {total_hrs:.2f} hours  ({total_secs/60:.0f} minutes)")
print(f"  Average clip duration: {total_secs/total_clips:.2f} sec")

# Duration by language
print(f"\n[DURATION BY LANGUAGE]")
lang_dur  = defaultdict(float)
lang_cnt  = defaultdict(int)
for r in records:
    lang_dur[r["script_language"]] += float(r["duration_sec"])
    lang_cnt[r["script_language"]] += 1
for lang in sorted(lang_dur, key=lambda x: -lang_dur[x]):
    h = lang_dur[lang]/3600
    print(f"  {lang:<30} {lang_cnt[lang]:>6} clips   {h:>6.2f} hrs")

# Duration by region
print(f"\n[DURATION BY REGION]")
reg_dur = defaultdict(float)
reg_cnt = defaultdict(int)
for r in records:
    reg_dur[r["region"]] += float(r["duration_sec"])
    reg_cnt[r["region"]] += 1
for reg in sorted(reg_dur, key=lambda x: -reg_dur[x]):
    h = reg_dur[reg]/3600
    print(f"  {reg:<30} {reg_cnt[reg]:>6} clips   {h:>6.2f} hrs")

# Duration by reading level
print(f"\n[DURATION BY READING LEVEL]")
lvl_dur = defaultdict(float)
lvl_cnt = defaultdict(int)
for r in records:
    lvl_dur[r["reading_level"]] += float(r["duration_sec"])
    lvl_cnt[r["reading_level"]] += 1
for lvl in sorted(lvl_dur, key=lambda x: -lvl_dur[x]):
    h = lvl_dur[lvl]/3600
    print(f"  {lvl:<35} {lvl_cnt[lvl]:>6} clips   {h:>6.2f} hrs")

# Age distribution
print(f"\n[AGE GROUP DISTRIBUTION]")
age_dur = defaultdict(float)
age_cnt = defaultdict(int)
for r in records:
    age_dur[r["age_group"]] += float(r["duration_sec"])
    age_cnt[r["age_group"]] += 1
for age in sorted(age_cnt.keys()):
    h = age_dur[age]/3600
    print(f"  {age:<25} {age_cnt[age]:>6} clips   {h:>6.2f} hrs")

# Reading quality
print(f"\n[READING QUALITY]")
correct   = [r for r in records if r["is_correct"] == "True"]
incorrect = [r for r in records if r["is_correct"] == "False"]
corr_hrs  = sum(float(r["duration_sec"]) for r in correct)/3600
incorr_hrs= sum(float(r["duration_sec"]) for r in incorrect)/3600
print(f"  Correct readings  : {len(correct):>6} clips   {corr_hrs:.2f} hrs")
print(f"  Incorrect readings: {len(incorrect):>6} clips   {incorr_hrs:.2f} hrs")

# ASR-useful subset (connected speech only — paragraph, story, sentence)
print(f"\n[ASR-USEFUL SUBSET: Connected Speech Only]")
asr_levels = {"Paragraph", "Story", "Sentence (English)"}
asr_records = [r for r in records if r["reading_level"] in asr_levels]
asr_hrs = sum(float(r["duration_sec"]) for r in asr_records)/3600
print(f"  Paragraph + Story + Sentence clips: {len(asr_records):,}")
print(f"  Duration                          : {asr_hrs:.2f} hrs")
asr_lang = defaultdict(int)
for r in asr_records:
    asr_lang[r["script_language"]] += 1
for lang, cnt in sorted(asr_lang.items(), key=lambda x: -x[1]):
    print(f"    {lang:<25} {cnt:>5} clips")

# ── 4. Train / Dev / Test Split ────────────────────────────────────────────────
print("\n" + "="*65)
print("  CREATING TRAIN / DEV / TEST SPLITS")
print("="*65)

print("""
Split strategy:
  - Split is done BY CHILD (child_id), NOT by clip.
  - Reason: A child's voice is unique. If the same child appears in both
    train and test sets, the model can 'memorize' that voice and inflate
    test WER — this is called speaker leakage. Splitting by child_id
    guarantees the test set contains only voices the model has NEVER heard.
  - Ratio: 80% train / 10% dev / 10% test  (by number of children)
  - Stratification: balanced across regions (Hindi RJ, Hindi UP, Marathi MH)
    so each split has proportional regional/language representation.
""")

# Group children by region for stratification
region_children = defaultdict(list)
for r in records:
    region_children[r["region"]].append(r["child_id"])

# Unique children per region
region_unique = {reg: list(set(ids)) for reg, ids in region_children.items()}

train_ids, dev_ids, test_ids = set(), set(), set()

for reg, children in region_unique.items():
    random.shuffle(children)
    n = len(children)
    n_test = max(1, int(n * 0.10))
    n_dev  = max(1, int(n * 0.10))
    test_ids.update(children[:n_test])
    dev_ids.update(children[n_test:n_test+n_dev])
    train_ids.update(children[n_test+n_dev:])

train_records = [r for r in records if r["child_id"] in train_ids]
dev_records   = [r for r in records if r["child_id"] in dev_ids]
test_records  = [r for r in records if r["child_id"] in test_ids]

def split_stats(name, recs):
    hrs = sum(float(r["duration_sec"]) for r in recs)/3600
    children = len(set(r["child_id"] for r in recs))
    print(f"  {name:<10}: {len(recs):>6} clips | {children:>4} children | {hrs:.2f} hrs")

print("Split sizes:")
split_stats("TRAIN", train_records)
split_stats("DEV",   dev_records)
split_stats("TEST",  test_records)

# Verify no overlap
overlap_train_test = train_ids & test_ids
overlap_train_dev  = train_ids & dev_ids
overlap_dev_test   = dev_ids   & test_ids
print(f"\n  Speaker leakage check:")
print(f"  Train ∩ Test  = {len(overlap_train_test)} children (must be 0)")
print(f"  Train ∩ Dev   = {len(overlap_train_dev)} children (must be 0)")
print(f"  Dev   ∩ Test  = {len(overlap_dev_test)} children (must be 0)")

# Save splits
fieldnames = list(records[0].keys())
for name, recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
    out_path = OUT_DIR / f"{name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recs)
    print(f"  Saved → {out_path}")

# Also save ASR-only splits (connected speech only)
print(f"\n[ASR-ONLY SPLITS: Paragraph + Story + Sentence]")
for name, recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
    asr_recs = [r for r in recs if r["reading_level"] in asr_levels]
    hrs = sum(float(r["duration_sec"]) for r in asr_recs)/3600
    out_path = OUT_DIR / f"asr_{name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asr_recs)
    print(f"  asr_{name}.csv : {len(asr_recs):>5} clips | {hrs:.2f} hrs  → {out_path}")

print("\n" + "="*65)
print("  ALL DONE. Dataset fully prepared.")
print("="*65)
print(f"""
Files created:
  manifest.csv      — All 81K clips with durations
  train.csv         — 80% children, all reading levels
  dev.csv           — 10% children, all reading levels
  test.csv          — 10% children, all reading levels
  asr_train.csv     — Connected speech only, train split
  asr_dev.csv       — Connected speech only, dev split
  asr_test.csv      — Connected speech only, test split
""")
