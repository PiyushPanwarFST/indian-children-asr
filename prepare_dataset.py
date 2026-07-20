"""
ASER Dataset Preparation Script
- Extracts all 5301 zip files
- Builds a master manifest CSV (audio_path, transcript, language, age_group, etc.)
- Computes full dataset statistics
"""

import os
import json
import zipfile
import csv
import re
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
ASER_ROOT   = Path("/home/hp/Indain_children_spech/ASER-Dataset/Data")
EXTRACT_DIR = Path("/home/hp/Indain_children_spech/ASER-Dataset/extracted")
MANIFEST    = Path("/home/hp/Indain_children_spech/ASER-Dataset/manifest.csv")

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Language folder mapping ────────────────────────────────────────────────────
FOLDER_LANG = {
    "Hindi RJ": "Hindi",
    "Hindi UP": "Hindi",
    "Marathi MH": "Marathi",
}

# ── Reading level mapping (from que_id prefix) ─────────────────────────────────
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
    """Derive reading level and script language from que_id like HI_S1_ST_0"""
    parts = que_id.split("_")
    if len(parts) < 3:
        return "Unknown", "Unknown"
    lang_code = parts[0]   # HI or MR
    level_code = parts[2]  # ST, P, WD, L, CL, SL, W, S

    script_lang = "Hindi" if lang_code == "HI" else "Marathi" if lang_code == "MR" else "Unknown"
    # English items exist inside Hindi/Marathi sessions
    if level_code in ("CL", "SL", "W", "S"):
        script_lang = "English"

    return LEVEL_MAP.get(level_code, level_code), script_lang

records = []
errors  = []

zip_files = sorted(ASER_ROOT.rglob("*.zip"))
total = len(zip_files)
print(f"Found {total} zip files. Starting extraction...")

for i, zip_path in enumerate(zip_files, 1):
    if i % 500 == 0 or i == 1:
        print(f"  Processing {i}/{total} ...")

    folder_name = zip_path.parent.name
    region      = folder_name  # e.g. "Hindi RJ"

    child_id    = zip_path.stem  # e.g. "3439"
    child_out   = EXTRACT_DIR / folder_name / child_id
    child_out.mkdir(parents=True, exist_ok=True)

    # Extract zip
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(child_out)
    except Exception as e:
        errors.append((str(zip_path), str(e)))
        continue

    # Find summary JSON
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

    age_group          = meta.get("ageGroup", "Unknown")
    stud_class         = meta.get("studClass", "Unknown")
    native_proficiency = meta.get("nativeProficiency", "Unknown")
    english_proficiency= meta.get("englishProficiency", "Unknown")
    device_id          = meta.get("deviceID", "Unknown")
    date_str           = meta.get("date", "")

    for item in meta.get("sequenceList", []):
        que_text    = item.get("que_text", "").strip()
        rec_name    = item.get("recordingName", "")
        que_id      = item.get("que_id", "")
        is_correct  = item.get("isCorrect", None)
        n_mistakes  = item.get("noOfMistakes", "0")

        audio_path = child_out / rec_name
        if not audio_path.exists():
            errors.append((str(zip_path), f"Missing audio: {rec_name}"))
            continue

        reading_level, script_lang = get_level_and_lang(que_id)

        records.append({
            "audio_path":          str(audio_path),
            "transcript":          que_text,
            "script_language":     script_lang,
            "region":              region,
            "reading_level":       reading_level,
            "que_id":              que_id,
            "is_correct":          is_correct,
            "num_mistakes":        n_mistakes,
            "age_group":           age_group,
            "student_class":       stud_class,
            "native_proficiency":  native_proficiency,
            "english_proficiency": english_proficiency,
            "child_id":            child_id,
            "date":                date_str,
        })

print(f"\nExtraction complete. Total records: {len(records)}")
if errors:
    print(f"Errors encountered: {len(errors)}")
    for e in errors[:10]:
        print(" ", e)

# ── Write manifest CSV ─────────────────────────────────────────────────────────
fieldnames = list(records[0].keys()) if records else []
with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nManifest saved → {MANIFEST}")

# ── Statistics ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ASER DATASET STATISTICS")
print("="*60)

print(f"\nTotal audio clips : {len(records)}")
print(f"Total children    : {len({r['child_id'] for r in records})}")

# Language distribution
lang_counts = defaultdict(int)
for r in records:
    lang_counts[r["script_language"]] += 1
print("\nLanguage / Script distribution:")
for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"  {lang:<25} {cnt:>6} clips")

# Region
region_counts = defaultdict(int)
for r in records:
    region_counts[r["region"]] += 1
print("\nRegion distribution:")
for reg, cnt in sorted(region_counts.items(), key=lambda x: -x[1]):
    print(f"  {reg:<25} {cnt:>6} clips")

# Reading level
level_counts = defaultdict(int)
for r in records:
    level_counts[r["reading_level"]] += 1
print("\nReading level distribution:")
for lvl, cnt in sorted(level_counts.items(), key=lambda x: -x[1]):
    print(f"  {lvl:<30} {cnt:>6} clips")

# Age group
age_counts = defaultdict(int)
for r in records:
    age_counts[r["age_group"]] += 1
print("\nAge group distribution:")
for age, cnt in sorted(age_counts.items()):
    print(f"  {age:<20} {cnt:>6} clips")

# Correctness
correct   = sum(1 for r in records if r["is_correct"] is True)
incorrect = sum(1 for r in records if r["is_correct"] is False)
print(f"\nCorrect readings  : {correct}")
print(f"Incorrect readings: {incorrect}")

# Audio duration using ffprobe if available
print("\nEstimating audio duration (using ffprobe)...")
try:
    import subprocess
    total_seconds = 0
    sample_count  = 0
    for r in records[:200]:  # sample 200 files for speed estimate
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", r["audio_path"]],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            total_seconds += float(result.stdout.strip())
            sample_count  += 1

    if sample_count > 0:
        avg_dur        = total_seconds / sample_count
        est_total_hrs  = (avg_dur * len(records)) / 3600
        print(f"  Avg clip duration : {avg_dur:.2f} sec (from {sample_count} samples)")
        print(f"  Estimated total   : {est_total_hrs:.1f} hours")
except Exception as e:
    print(f"  ffprobe not available or error: {e}")
    print("  Install with: sudo apt install ffmpeg")

print("\n" + "="*60)
print("Dataset ready for use.")
print("="*60)
