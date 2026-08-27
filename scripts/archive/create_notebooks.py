"""
Creates two Jupyter notebooks from our existing scripts:
  01_dataset_preparation.ipynb
  02_dataset_eda_and_splits.ipynb
"""
import nbformat as nbf

def cell(src, kind="code"):
    if kind == "markdown":
        return nbf.v4.new_markdown_cell(src)
    return nbf.v4.new_code_cell(src)

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Dataset Preparation
# ══════════════════════════════════════════════════════════════════════════════
nb1 = nbf.v4.new_notebook()
nb1.cells = [

cell("""# 📦 ASER Dataset — Preparation
**Project:** Indian Children Speech Recognition (ICASSP)
**What this notebook does:**
- Unzips all 5,301 ASER child session archives
- Reads each child's JSON metadata file
- Builds a single master `manifest.csv` mapping every audio file to its transcript + metadata

Run cells **top to bottom**, once. After this notebook finishes, move to `02_dataset_eda_and_splits.ipynb`.
""", "markdown"),

cell("""## Cell 1 — Imports and Path Setup
We define where the raw ASER zips live, where to extract them, and where to save the manifest.
""", "markdown"),

cell("""import os
import json
import zipfile
import csv
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
ASER_ROOT   = Path("/home/hp/Indain_children_spech/ASER-Dataset/Data")
EXTRACT_DIR = Path("/home/hp/Indain_children_spech/ASER-Dataset/extracted")
MANIFEST    = Path("/home/hp/Indain_children_spech/ASER-Dataset/manifest.csv")

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Raw data location : {ASER_ROOT}")
print(f"Extract to        : {EXTRACT_DIR}")
print(f"Manifest output   : {MANIFEST}")
"""),

cell("""## Cell 2 — Language and Level Mappings
`que_id` encodes language and reading level in its name (e.g. `HI_S1_ST_0`).
We decode these into human-readable labels here.
""", "markdown"),

cell("""# Maps folder name → broad language
FOLDER_LANG = {
    "Hindi RJ":   "Hindi",
    "Hindi UP":   "Hindi",
    "Marathi MH": "Marathi",
}

# Maps level code → full reading level name
LEVEL_MAP = {
    "ST": "Story",
    "P":  "Paragraph",
    "WD": "Word",
    "L":  "Letter",
    "CL": "Capital Letter (English)",
    "SL": "Small Letter (English)",
    "W":  "Word (English)",
    "S":  "Sentence (English)",
}

def get_level_and_lang(que_id):
    \"\"\"
    Input : 'HI_S1_ST_0'
    Output: ('Story', 'Hindi')
    \"\"\"
    parts = que_id.split("_")
    if len(parts) < 3:
        return "Unknown", "Unknown"
    lang_code  = parts[0]   # HI or MR
    level_code = parts[2]   # ST, P, WD, L, CL, SL, W, S

    script_lang = "Hindi" if lang_code == "HI" else "Marathi" if lang_code == "MR" else "Unknown"
    if level_code in ("CL", "SL", "W", "S"):
        script_lang = "English"   # English-task items inside any session

    return LEVEL_MAP.get(level_code, level_code), script_lang

# Quick test
print(get_level_and_lang("HI_S1_ST_0"))   # Expected: ('Story', 'Hindi')
print(get_level_and_lang("MR_S2_CL_1"))   # Expected: ('Capital Letter (English)', 'English')
"""),

cell("""## Cell 3 — Extract All Zips + Build Manifest
This is the main loop. For each of 5,301 zip files:
1. Extract the .mp3 audio files to disk
2. Read the summary JSON for transcript and metadata
3. Add one row per audio clip to our records list
""", "markdown"),

cell("""records = []
errors  = []

zip_files = sorted(ASER_ROOT.rglob("*.zip"))
total     = len(zip_files)
print(f"Found {total} zip files. Starting extraction...")

for i, zip_path in enumerate(zip_files, 1):
    if i % 500 == 0 or i == 1:
        print(f"  Processing {i}/{total} ...")

    folder_name = zip_path.parent.name   # e.g. "Hindi RJ"
    child_id    = zip_path.stem          # e.g. "3439"
    child_out   = EXTRACT_DIR / folder_name / child_id
    child_out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract zip ---
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(child_out)
    except Exception as e:
        errors.append((str(zip_path), str(e)))
        continue

    # --- Step 2: Find and parse the JSON ---
    json_files = list(child_out.glob("summary*.json"))
    if not json_files:
        errors.append((str(zip_path), "No summary JSON found"))
        continue

    with open(json_files[0], "r", encoding="utf-8") as f:
        try:
            meta = json.load(f)
        except Exception as e:
            errors.append((str(zip_path), f"JSON parse error: {e}"))
            continue

    age_group           = meta.get("ageGroup", "Unknown")
    stud_class          = meta.get("studClass", "Unknown")
    native_proficiency  = meta.get("nativeProficiency", "Unknown")
    english_proficiency = meta.get("englishProficiency", "Unknown")

    # --- Step 3: One row per audio clip ---
    for item in meta.get("sequenceList", []):
        que_text   = item.get("que_text", "").strip()
        rec_name   = item.get("recordingName", "")
        que_id     = item.get("que_id", "")
        is_correct = item.get("isCorrect", None)
        n_mistakes = item.get("noOfMistakes", "0")

        audio_path = child_out / rec_name
        if not audio_path.exists():
            errors.append((str(zip_path), f"Missing audio: {rec_name}"))
            continue

        reading_level, script_lang = get_level_and_lang(que_id)

        records.append({
            "audio_path":          str(audio_path),
            "transcript":          que_text,
            "script_language":     script_lang,
            "region":              folder_name,
            "reading_level":       reading_level,
            "que_id":              que_id,
            "is_correct":          is_correct,
            "num_mistakes":        n_mistakes,
            "age_group":           age_group,
            "student_class":       stud_class,
            "native_proficiency":  native_proficiency,
            "english_proficiency": english_proficiency,
            "child_id":            child_id,
        })

print(f"\\nDone! Total records : {len(records)}")
print(f"Errors encountered  : {len(errors)}")
if errors:
    print("First 5 errors:")
    for e in errors[:5]:
        print("  ", e)
"""),

cell("""## Cell 4 — Save Manifest CSV
We save all records to a single CSV file.
Every ASR training framework reads data from a file like this.
""", "markdown"),

cell("""fieldnames = list(records[0].keys())

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"Manifest saved → {MANIFEST}")
print(f"Total rows     : {len(records):,}")

# Preview first 3 rows
print("\\nSample rows:")
for r in records[:3]:
    print(f"  audio : {Path(r['audio_path']).name}")
    print(f"  text  : {r['transcript'][:60]}")
    print(f"  lang  : {r['script_language']}  |  level: {r['reading_level']}  |  age: {r['age_group']}")
    print()
"""),

]

nb1_path = "/home/hp/Indain_children_spech/01_dataset_preparation.ipynb"
with open(nb1_path, "w") as f:
    nbf.write(nb1, f)
print(f"Created: {nb1_path}")


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — EDA + Splits
# ══════════════════════════════════════════════════════════════════════════════
nb2 = nbf.v4.new_notebook()
nb2.cells = [

cell("""# 📊 ASER Dataset — EDA & Train/Dev/Test Splits
**Project:** Indian Children Speech Recognition (ICASSP)

**What this notebook does:**
- Loads the manifest built in Notebook 01
- Computes exact audio duration for every clip
- Prints full dataset statistics (language, region, age, reading level)
- Creates speaker-independent train / dev / test splits
- Saves 6 CSV files ready for model training

**Run after:** `01_dataset_preparation.ipynb`
""", "markdown"),

cell("""## Cell 1 — Imports and Paths
""", "markdown"),

cell("""import csv
import subprocess
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)   # Fixed seed = reproducible splits every time

MANIFEST = Path("/home/hp/Indain_children_spech/ASER-Dataset/manifest.csv")
OUT_DIR  = Path("/home/hp/Indain_children_spech/ASER-Dataset")

print("Paths configured.")
print(f"Manifest: {MANIFEST}")
"""),

cell("""## Cell 2 — Load Manifest
Load all 81K rows from the manifest CSV into memory.
""", "markdown"),

cell("""records = []
with open(MANIFEST, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

print(f"Loaded {len(records):,} records.")
print(f"Columns: {list(records[0].keys())}")
"""),

cell("""## Cell 3 — Compute Exact Duration with ffprobe
`ffprobe` reads the audio file header and returns the duration in seconds.
We do this for every clip so we can report honest hours, not estimates.

⏱ This takes ~8-10 minutes for 81K files.
""", "markdown"),

cell("""def get_duration(path):
    \"\"\"Returns duration in seconds for an audio file using ffprobe.\"\"\"
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        val = result.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

print("Computing durations... (this takes ~8-10 minutes)")
for i, r in enumerate(records):
    r["duration_sec"] = get_duration(r["audio_path"])
    if (i+1) % 10000 == 0:
        print(f"  {i+1:,} / {len(records):,} done...")

# Save durations back to manifest
with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

total_hrs = sum(float(r["duration_sec"]) for r in records) / 3600
print(f"\\nTotal dataset duration: {total_hrs:.2f} hours")
"""),

cell("""## Cell 4 — Overall Statistics
""", "markdown"),

cell("""total_clips    = len(records)
total_secs     = sum(float(r["duration_sec"]) for r in records)
total_hrs      = total_secs / 3600
total_children = len(set(r["child_id"] for r in records))

print("=" * 50)
print("  ASER DATASET — OVERVIEW")
print("=" * 50)
print(f"  Total clips      : {total_clips:,}")
print(f"  Unique children  : {total_children:,}")
print(f"  Total duration   : {total_hrs:.2f} hours")
print(f"  Avg clip length  : {total_secs/total_clips:.2f} seconds")
"""),

cell("""## Cell 5 — Duration by Language
""", "markdown"),

cell("""lang_dur = defaultdict(float)
lang_cnt = defaultdict(int)
for r in records:
    lang_dur[r["script_language"]] += float(r["duration_sec"])
    lang_cnt[r["script_language"]] += 1

print(f"{'Language':<30} {'Clips':>7}  {'Hours':>8}")
print("-" * 50)
for lang in sorted(lang_dur, key=lambda x: -lang_dur[x]):
    print(f"{lang:<30} {lang_cnt[lang]:>7,}  {lang_dur[lang]/3600:>8.2f}")
"""),

cell("""## Cell 6 — Duration by Region
""", "markdown"),

cell("""reg_dur = defaultdict(float)
reg_cnt = defaultdict(int)
for r in records:
    reg_dur[r["region"]] += float(r["duration_sec"])
    reg_cnt[r["region"]] += 1

print(f"{'Region':<30} {'Clips':>7}  {'Hours':>8}")
print("-" * 50)
for reg in sorted(reg_dur, key=lambda x: -reg_dur[x]):
    print(f"{reg:<30} {reg_cnt[reg]:>7,}  {reg_dur[reg]/3600:>8.2f}")
"""),

cell("""## Cell 7 — Duration by Reading Level
""", "markdown"),

cell("""lvl_dur = defaultdict(float)
lvl_cnt = defaultdict(int)
for r in records:
    lvl_dur[r["reading_level"]] += float(r["duration_sec"])
    lvl_cnt[r["reading_level"]] += 1

print(f"{'Reading Level':<35} {'Clips':>7}  {'Hours':>8}  ASR Useful?")
print("-" * 65)
asr_levels = {"Paragraph", "Story", "Sentence (English)"}
for lvl in sorted(lvl_dur, key=lambda x: -lvl_dur[x]):
    useful = "YES ✓" if lvl in asr_levels else "-"
    print(f"{lvl:<35} {lvl_cnt[lvl]:>7,}  {lvl_dur[lvl]/3600:>8.2f}  {useful}")
"""),

cell("""## Cell 8 — Age Group Distribution
""", "markdown"),

cell("""age_dur = defaultdict(float)
age_cnt = defaultdict(int)
for r in records:
    age_dur[r["age_group"]] += float(r["duration_sec"])
    age_cnt[r["age_group"]] += 1

print(f"{'Age Group':<25} {'Clips':>7}  {'Hours':>8}")
print("-" * 45)
for age in sorted(age_cnt.keys()):
    print(f"{age:<25} {age_cnt[age]:>7,}  {age_dur[age]/3600:>8.2f}")
"""),

cell("""## Cell 9 — Reading Quality (Correct vs Incorrect)
""", "markdown"),

cell("""correct   = [r for r in records if r["is_correct"] == "True"]
incorrect = [r for r in records if r["is_correct"] == "False"]

print(f"Correct readings   : {len(correct):,}  ({len(correct)/len(records)*100:.1f}%)")
print(f"Incorrect readings : {len(incorrect):,}  ({len(incorrect)/len(records)*100:.1f}%)")
print()
print("Note: Incorrect readings are NOT discarded.")
print("They contain real child mispronunciation patterns — valuable for training robustness.")
"""),

cell("""## Cell 10 — ASR-Useful Subset Summary
Only Paragraph + Story + Sentence clips are used for ASR model training.
Single letters and words have no linguistic context.
""", "markdown"),

cell("""asr_levels   = {"Paragraph", "Story", "Sentence (English)"}
asr_records  = [r for r in records if r["reading_level"] in asr_levels]
asr_hrs      = sum(float(r["duration_sec"]) for r in asr_records) / 3600
asr_children = len(set(r["child_id"] for r in asr_records))

print(f"ASR-useful clips    : {len(asr_records):,}")
print(f"ASR-useful duration : {asr_hrs:.2f} hours")
print(f"Unique children     : {asr_children:,}")

asr_lang = defaultdict(int)
for r in asr_records:
    asr_lang[r["script_language"]] += 1
print()
for lang, cnt in sorted(asr_lang.items(), key=lambda x: -x[1]):
    print(f"  {lang:<25} {cnt:,} clips")
"""),

cell("""## Cell 11 — Create Train / Dev / Test Splits

### Why split by child, not by clip?
If we split randomly by clip, the same child's voice appears in BOTH train and test.
The model hears that voice during training → unfair advantage at test time → fake good results.

**Splitting by child_id guarantees the test set contains only voices the model has NEVER heard.**
This is called a speaker-independent evaluation — the standard for publishable ASR research.

### Strategy
- Group children by region (RJ, UP, MH) for balanced regional representation
- Within each region: 80% train / 10% dev / 10% test
- All clips of a child go entirely into ONE split (no leakage)
""", "markdown"),

cell("""region_children = defaultdict(list)
for r in records:
    region_children[r["region"]].append(r["child_id"])

region_unique = {reg: list(set(ids)) for reg, ids in region_children.items()}

train_ids, dev_ids, test_ids = set(), set(), set()

for reg, children in region_unique.items():
    random.shuffle(children)
    n      = len(children)
    n_test = max(1, int(n * 0.10))
    n_dev  = max(1, int(n * 0.10))
    test_ids.update(children[:n_test])
    dev_ids.update(children[n_test : n_test + n_dev])
    train_ids.update(children[n_test + n_dev :])

train_records = [r for r in records if r["child_id"] in train_ids]
dev_records   = [r for r in records if r["child_id"] in dev_ids]
test_records  = [r for r in records if r["child_id"] in test_ids]

print("Split sizes:")
for name, recs in [("TRAIN", train_records), ("DEV", dev_records), ("TEST", test_records)]:
    hrs      = sum(float(r["duration_sec"]) for r in recs) / 3600
    children = len(set(r["child_id"] for r in recs))
    print(f"  {name:<6}: {len(recs):>6,} clips | {children:>4} children | {hrs:.2f} hrs")

# Leakage check
print(f"\\nSpeaker leakage check:")
print(f"  Train ∩ Test : {len(train_ids & test_ids)} (must be 0)")
print(f"  Train ∩ Dev  : {len(train_ids & dev_ids)} (must be 0)")
print(f"  Dev   ∩ Test : {len(dev_ids   & test_ids)} (must be 0)")
"""),

cell("""## Cell 12 — Save All Split CSVs
""", "markdown"),

cell("""fieldnames = list(records[0].keys())

# Full splits (all reading levels)
for name, recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
    out = OUT_DIR / f"{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recs)
    print(f"Saved {name}.csv  →  {len(recs):,} clips")

print()

# ASR-only splits (connected speech: paragraph + story + sentence)
for name, recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
    asr_recs = [r for r in recs if r["reading_level"] in asr_levels]
    hrs      = sum(float(r["duration_sec"]) for r in asr_recs) / 3600
    out      = OUT_DIR / f"asr_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asr_recs)
    print(f"Saved asr_{name}.csv  →  {len(asr_recs):,} clips  |  {hrs:.2f} hrs")

print("\\nAll done! Dataset fully prepared for ASR training.")
"""),

]

nb2_path = "/home/hp/Indain_children_spech/02_dataset_eda_and_splits.ipynb"
with open(nb2_path, "w") as f:
    nbf.write(nb2, f)
print(f"Created: {nb2_path}")

print("\nBoth notebooks ready.")
