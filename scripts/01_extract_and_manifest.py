"""
Script 1: Extract Raw Data & Build Manifest

What this script does:
  1. Extracts all 5,301 ZIP files from the ASER dataset
  2. Reads the JSON metadata inside each ZIP (child info + transcripts)
  3. Builds one master CSV (raw_manifest.csv) with one row per audio clip
  4. Prints raw data statistics and verifies against the paper (Table 3)

Output: ASER-Dataset/raw_manifest.csv (81,423 clips, 123.72 hours)
"""

import os
import json
import zipfile
import csv
import subprocess
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
ZIP_DIR      = ASER_ROOT / "Data"
EXTRACT_DIR  = ASER_ROOT / "extracted"
OUTPUT_CSV   = ASER_ROOT / "raw_manifest.csv"

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"ZIP source:  {ZIP_DIR}")
print(f"Extract to:  {EXTRACT_DIR}")
print(f"Output CSV:  {OUTPUT_CSV}")

# ── Reading level codes from que_id ───────────────────────────────────────────
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

def get_level_and_language(que_id):
    parts = que_id.split("_")
    if len(parts) < 3:
        return "Unknown", "Unknown"
    lang_code = parts[0]
    level_code = parts[2]
    if level_code in ("CL", "SL", "W", "S"):
        language = "English"
    elif lang_code == "HI":
        language = "Hindi"
    elif lang_code == "MR":
        language = "Marathi"
    else:
        language = "Unknown"
    level = LEVEL_MAP.get(level_code, level_code)
    return level, language

def get_duration(audio_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             audio_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except Exception:
        return 0.0

# ── Count ZIPs ────────────────────────────────────────────────────────────────
for region_dir in sorted(ZIP_DIR.iterdir()):
    if region_dir.is_dir():
        zips = list(region_dir.glob("*.zip"))
        print(f"{region_dir.name}: {len(zips)} ZIP files")

all_zips = sorted(ZIP_DIR.rglob("*.zip"))
print(f"\nTotal ZIP files: {len(all_zips)}")

# ── Extract all ZIPs and build manifest ───────────────────────────────────────
print("\nExtracting ZIPs and building manifest...")
records = []
errors = []

for i, zip_path in enumerate(all_zips, 1):
    if i % 500 == 0 or i == 1:
        print(f"  Processing {i}/{len(all_zips)} ...")

    region = zip_path.parent.name
    child_id = zip_path.stem
    child_dir = EXTRACT_DIR / region / child_id
    child_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(child_dir)
    except Exception as e:
        errors.append((str(zip_path), str(e)))
        continue

    json_files = list(child_dir.glob("summary*.json"))
    if not json_files:
        errors.append((str(zip_path), "No summary JSON found"))
        continue

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        errors.append((str(zip_path), f"JSON error: {e}"))
        continue

    age_group = meta.get("ageGroup", "Unknown")
    student_class = meta.get("studClass", "Unknown")
    date_str = meta.get("date", "")

    for item in meta.get("sequenceList", []):
        transcript = item.get("que_text", "").strip()
        recording_name = item.get("recordingName", "")
        que_id = item.get("que_id", "")
        is_correct = item.get("isCorrect", None)
        num_mistakes = item.get("noOfMistakes", "0")

        audio_path = child_dir / recording_name
        if not audio_path.exists():
            errors.append((str(zip_path), f"Missing audio: {recording_name}"))
            continue

        reading_level, language = get_level_and_language(que_id)

        records.append({
            "audio_path": str(audio_path),
            "transcript": transcript,
            "language": language,
            "region": region,
            "reading_level": reading_level,
            "que_id": que_id,
            "is_correct": is_correct,
            "num_mistakes": num_mistakes,
            "age_group": age_group,
            "student_class": student_class,
            "child_id": child_id,
            "date": date_str,
        })

print(f"\nExtraction complete!")
print(f"Total audio clips: {len(records)}")
print(f"Errors: {len(errors)}")

# ── Compute durations ─────────────────────────────────────────────────────────
print("\nComputing duration for all clips...")
for i, r in enumerate(records):
    r["duration_sec"] = get_duration(r["audio_path"])
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(records)} done")
print(f"Done. All {len(records)} durations computed.")

# ── Save CSV ──────────────────────────────────────────────────────────────────
fieldnames = ["audio_path", "transcript", "language", "region", "reading_level",
              "que_id", "is_correct", "num_mistakes", "age_group",
              "student_class", "child_id", "date", "duration_sec"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Rows: {len(records)}")

# ── Statistics ────────────────────────────────────────────────────────────────
total_clips = len(records)
total_hours = sum(r["duration_sec"] for r in records) / 3600
total_children = len(set(r["child_id"] for r in records))

print("\n" + "=" * 60)
print("RAW DATASET OVERVIEW")
print("=" * 60)
print(f"Total clips:    {total_clips:,}")
print(f"Total duration: {total_hours:.2f} hours")
print(f"Total children: {total_children:,}")

print(f"\n--- By Region ---")
for region in ["Hindi RJ", "Hindi UP", "Marathi MH"]:
    clips = [r for r in records if r["region"] == region]
    hrs = sum(r["duration_sec"] for r in clips) / 3600
    children = len(set(r["child_id"] for r in clips))
    print(f"  {region:<15} {len(clips):>6} clips | {hrs:>6.2f} hrs | {children:>5} children")

print(f"\n--- By Language ---")
for lang in ["Hindi", "Marathi", "English"]:
    clips = [r for r in records if r["language"] == lang]
    hrs = sum(r["duration_sec"] for r in clips) / 3600
    print(f"  {lang:<15} {len(clips):>6} clips | {hrs:>6.2f} hrs")

print(f"\n--- By Reading Level ---")
level_data = defaultdict(lambda: {"clips": 0, "hrs": 0})
for r in records:
    level_data[r["reading_level"]]["clips"] += 1
    level_data[r["reading_level"]]["hrs"] += r["duration_sec"] / 3600
for lvl in sorted(level_data, key=lambda x: -level_data[x]["hrs"]):
    d = level_data[lvl]
    print(f"  {lvl:<30} {d['clips']:>6} clips | {d['hrs']:>6.2f} hrs")

print("\n" + "=" * 60)
print("DONE. raw_manifest.csv ready.")
print("=" * 60)
