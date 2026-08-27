"""
Recover Long Clips (> 30 seconds)
==================================
The original pipeline deleted clips > 30s BEFORE applying VAD.
Many of those clips had substantial trailing silence — after VAD
they come under 30s and are perfectly usable.

This script:
  1. Finds all > 30s clips from the original manifest
  2. Applies VAD + volume normalization (same ffmpeg pipeline as clean_dataset.py)
  3. Clips now ≤ 30s → added to manifest_clean.csv and split files
  4. Clips still > 30s → saved to long_clips_manifest.csv for future chunking
     (chunking requires torchaudio forced alignment, done separately)

Run: python3 scripts/recover_long_clips.py
"""

import csv
import subprocess
import os
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = Path("/home/hp/Indain_children_spech/ASER-Dataset")
MANIFEST_ORIG = BASE / "manifest.csv"
MANIFEST_CLN  = BASE / "manifest_clean.csv"
AUDIO_CLEAN   = BASE / "audio_clean"
EXTRACTED     = BASE / "extracted"
LONG_MANIFEST = BASE / "long_clips_manifest.csv"

MIN_DUR = 0.5
MAX_DUR = 30.0

# Characters to remove (same as clean_dataset.py)
import re
REMOVE_CHARS = str.maketrans("", "", "।.,:;!?\"'()[]{}–—-/\\")

def normalize_text(text):
    text = text.strip()
    text = text.translate(REMOVE_CHARS)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ── Step 1: Find all > 30s clips from original manifest ───────────────────────
print("=" * 60)
print("RECOVERING LONG CLIPS (> 30 seconds)")
print("=" * 60)

print(f"\nLoading {MANIFEST_ORIG} ...")
with open(MANIFEST_ORIG, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
    fieldnames = reader.fieldnames

# Add duration_sec to fieldnames if not present
if "duration_sec" not in fieldnames:
    fieldnames = fieldnames + ["duration_sec"]

long_clips = []
for r in all_rows:
    dur = float(r["duration_sec"])
    # Same filters as clean_dataset.py Step 1, but KEEP > 30s clips
    if dur == 0.0:
        continue
    if dur < MIN_DUR:
        continue
    if r["reading_level"] == "Cl":
        continue
    if r["script_language"] == "Unknown" and r["reading_level"] == "Letter":
        r = dict(r)
        r["script_language"] = "Hindi"
    if r["script_language"] == "Unknown":
        continue
    if not r["transcript"].strip():
        continue
    if dur > MAX_DUR:
        long_clips.append(r)

print(f"  Found {len(long_clips)} clips > 30 seconds to process")

# ── Step 2: Normalize transcripts ─────────────────────────────────────────────
print("\n[Step 1] Normalizing transcripts...")
normalized = []
for r in long_clips:
    r = dict(r)
    r["transcript"] = normalize_text(r["transcript"])
    if r["transcript"]:
        normalized.append(r)
long_clips = normalized
print(f"  {len(long_clips)} clips after transcript normalization")

# ── Step 3: Apply VAD + volume normalization ──────────────────────────────────
print(f"\n[Step 2] Applying VAD + volume normalization to {len(long_clips)} clips...")
print(f"  Using {NUM_WORKERS} parallel workers...")

def process_one(args):
    src_str, dst_str = args
    src = Path(src_str)
    dst = Path(dst_str)
    if dst.exists():
        return src_str, dst_str, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_str,
        "-af",
        (
            "silenceremove="
            "start_periods=1:start_threshold=-40dB:start_duration=0.05"
            ":stop_periods=-1:stop_threshold=-40dB:stop_duration=0.2,"
            "dynaudnorm=f=150:g=15,"
            "aresample=16000"
        ),
        dst_str
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0 or not dst.exists():
        cmd_fallback = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", src_str,
            "-af", "dynaudnorm=f=150:g=15,aresample=16000",
            dst_str
        ]
        subprocess.run(cmd_fallback, capture_output=True, timeout=120)
        return src_str, dst_str, "fallback"
    return src_str, dst_str, "ok"

tasks = []
for r in long_clips:
    src = Path(r["audio_path"])
    try:
        rel = src.relative_to(EXTRACTED)
        dst = str(AUDIO_CLEAN / rel)
    except ValueError:
        dst = r["audio_path"]
    tasks.append((r["audio_path"], dst))

new_audio_paths = {}
fallbacks = 0
done = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_one, t): t for t in tasks}
    for future in as_completed(futures):
        src_str, dst_str, status = future.result()
        new_audio_paths[src_str] = dst_str
        if status == "fallback":
            fallbacks += 1
        done += 1
        if done % 500 == 0:
            pct = done / len(tasks) * 100
            print(f"  [{pct:5.1f}%] {done}/{len(tasks)} done  (fallbacks: {fallbacks})")

print(f"\n  Done. Processed: {done}, Fallbacks: {fallbacks}")

# Update audio paths
for r in long_clips:
    if r["audio_path"] in new_audio_paths:
        r["audio_path"] = new_audio_paths[r["audio_path"]]

# ── Step 4: Measure new durations and categorize ─────────────────────────────
print(f"\n[Step 3] Measuring post-VAD durations...")

recovered = []      # Clips now ≤ 30s → add to cleaned dataset
still_long = []     # Clips still > 30s → need chunking later
too_short = 0

for i, r in enumerate(long_clips):
    if i % 500 == 0:
        print(f"  {i}/{len(long_clips)} ...")
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         r["audio_path"]],
        capture_output=True, text=True, timeout=10
    )
    if result.stdout.strip():
        new_dur = float(result.stdout.strip())
        r = dict(r)
        r["duration_sec"] = str(round(new_dur, 4))
        if new_dur < MIN_DUR:
            too_short += 1
        elif new_dur <= MAX_DUR:
            recovered.append(r)
        else:
            still_long.append(r)

print(f"\n  Results:")
print(f"    Recovered (now ≤ 30s): {len(recovered)} clips")
print(f"    Still > 30s (need chunking): {len(still_long)} clips")
print(f"    Too short after VAD: {too_short} clips")

rec_hrs = sum(float(r["duration_sec"]) for r in recovered) / 3600
long_hrs = sum(float(r["duration_sec"]) for r in still_long) / 3600
print(f"    Recovered duration: {rec_hrs:.2f} hours")
print(f"    Still-long duration: {long_hrs:.2f} hours")

# ── Step 5: Add recovered clips to manifest_clean.csv ─────────────────────────
print(f"\n[Step 4] Adding {len(recovered)} recovered clips to manifest_clean.csv ...")

with open(MANIFEST_CLN, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    existing = list(reader)
    fieldnames = reader.fieldnames

existing.extend(recovered)

with open(MANIFEST_CLN, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing)
print(f"  manifest_clean.csv: {len(existing)} clips (was {len(existing)-len(recovered)})")

# ── Step 6: Save long clips manifest for future chunking ──────────────────────
if still_long:
    print(f"\n[Step 5] Saving long_clips_manifest.csv ({len(still_long)} clips) ...")
    with open(LONG_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(still_long)
    print(f"  Saved: {LONG_MANIFEST}")
    print(f"  These clips need chunking via forced alignment (torchaudio)")

# ── Step 7: Update split files ────────────────────────────────────────────────
print(f"\n[Step 6] Updating split files with recovered clips...")

# Build lookup: derive original path from cleaned path
clean_lookup = {}
for r in existing:
    try:
        rel = Path(r["audio_path"]).relative_to(AUDIO_CLEAN)
        old_path = str(EXTRACTED / rel)
    except ValueError:
        old_path = r["audio_path"]
    clean_lookup[old_path] = r

ASR_LEVELS = {"Paragraph", "Story", "Sentence (English)"}

splits = {
    "train":     BASE / "train.csv",
    "dev":       BASE / "dev.csv",
    "test":      BASE / "test.csv",
    "asr_train": BASE / "asr_train.csv",
    "asr_dev":   BASE / "asr_dev.csv",
    "asr_test":  BASE / "asr_test.csv",
}

for split_name, split_path in splits.items():
    with open(split_path, encoding="utf-8") as f:
        split_reader = csv.DictReader(f)
        split_rows = list(split_reader)

    cleaned_split = []
    for sr in split_rows:
        old_path = sr["audio_path"]
        if old_path in clean_lookup:
            row = clean_lookup[old_path]
            # For asr_ splits, only include ASR levels
            if split_name.startswith("asr_") and row["reading_level"] not in ASR_LEVELS:
                continue
            cleaned_split.append(dict(row))

    out_path = BASE / f"{split_name}_clean.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_split)

    print(f"  {split_name}_clean.csv: {len(cleaned_split)} clips")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOVERY SUMMARY")
print("=" * 60)
total_hrs = sum(float(r["duration_sec"]) for r in existing) / 3600
asr_clips = [r for r in existing if r["reading_level"] in ASR_LEVELS]
asr_hrs = sum(float(r["duration_sec"]) for r in asr_clips) / 3600
print(f"  Total clips now:     {len(existing)}")
print(f"  Total duration:      {total_hrs:.2f} hours")
print(f"  ASR clips now:       {len(asr_clips)}")
print(f"  ASR duration:        {asr_hrs:.2f} hours")
print(f"  Clips needing chunk: {len(still_long)} ({long_hrs:.2f} hours)")
print("=" * 60)
