"""
ASER Dataset Cleaning Script
=============================
What this script does:
  1. Filters bad clips (zero duration, too short, too long, bad categories)
  2. Fixes Unknown language tags
  3. Normalizes transcripts (remove punctuation that does not affect pronunciation)
  4. Applies VAD silence trimming + volume normalization via ffmpeg
  5. Recomputes durations on cleaned audio
  6. Saves manifest_clean.csv (all levels)
  7. Saves asr_manifest_clean.csv (Paragraph + Story + Sentence only — for ASR training)
  8. Updates all train/dev/test splits

Run: python3 clean_dataset.py
Resume: safe to re-run — already processed audio files are skipped
"""

import csv
import subprocess
import re
import sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # leave 1 core free

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = Path("/home/hp/Indain_children_spech/ASER-Dataset")
MANIFEST_IN   = BASE / "manifest.csv"
MANIFEST_OUT  = BASE / "manifest_clean.csv"
ASR_MANIFEST  = BASE / "asr_manifest_clean.csv"
AUDIO_CLEAN   = BASE / "audio_clean"
EXTRACTED     = BASE / "extracted"

AUDIO_CLEAN.mkdir(parents=True, exist_ok=True)

# ASR-useful reading levels (connected speech only — excludes single letters/words)
ASR_LEVELS = {"Paragraph", "Story", "Sentence (English)"}

# Duration thresholds
MIN_DUR = 0.5    # seconds
MAX_DUR = 30.0   # seconds

# ── Load manifest ──────────────────────────────────────────────────────────────
print("=" * 60)
print("ASER DATASET CLEANING")
print("=" * 60)
print(f"\nLoading {MANIFEST_IN} ...")
with open(MANIFEST_IN, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    original_rows = list(reader)
    fieldnames = reader.fieldnames

print(f"  Loaded {len(original_rows)} clips")

# ── Step 1: Filter bad clips ───────────────────────────────────────────────────
print("\n[Step 1] Filtering bad clips...")
removed = Counter()
kept = []

for r in original_rows:
    dur = float(r["duration_sec"])

    if dur == 0.0:
        removed["zero_duration"] += 1
        continue

    if dur < MIN_DUR:
        removed["too_short_under_0.5s"] += 1
        continue

    if dur > MAX_DUR:
        removed["too_long_over_30s"] += 1
        continue

    if r["reading_level"] == "Cl":
        removed["bad_Cl_category"] += 1
        continue

    # Remap Devanagari Letter clips tagged Unknown → Hindi
    if r["script_language"] == "Unknown" and r["reading_level"] == "Letter":
        r = dict(r)
        r["script_language"] = "Hindi"

    if r["script_language"] == "Unknown":
        removed["unknown_language_other"] += 1
        continue

    if not r["transcript"].strip():
        removed["empty_transcript"] += 1
        continue

    kept.append(r)

print(f"  Removal breakdown:")
for reason, count in removed.most_common():
    print(f"    {reason:<35}: {count}")
print(f"  Kept: {len(kept)} clips  (removed {len(original_rows)-len(kept)})")

# ── Step 2: Normalize transcripts ─────────────────────────────────────────────
print("\n[Step 2] Normalizing transcripts...")

# Characters to remove — punctuation that has no phonetic value
REMOVE_CHARS = str.maketrans("", "", "।.,:;!?\"'()[]{}–—-/\\")

def normalize_text(text):
    text = text.strip()
    text = text.translate(REMOVE_CHARS)
    text = re.sub(r"\s+", " ", text)   # collapse multiple spaces
    text = text.strip()
    return text

empty_after_norm = 0
normalized_kept = []
for r in kept:
    r = dict(r)
    r["transcript"] = normalize_text(r["transcript"])
    if not r["transcript"]:
        empty_after_norm += 1
        continue
    normalized_kept.append(r)

kept = normalized_kept
print(f"  Removed {empty_after_norm} clips with empty transcript after normalization")
print(f"  Kept: {len(kept)} clips")

# ── Step 3: VAD silence trimming + volume normalization ────────────────────────
print("\n[Step 3] VAD silence trimming + volume normalization via ffmpeg...")
print(f"  Output directory: {AUDIO_CLEAN}")
print(f"  Processing {len(kept)} clips using {NUM_WORKERS} parallel workers...")
print(f"  (Skips already processed files for resume support)\n")

def process_one(args):
    """Process a single audio file — designed to run in a subprocess worker."""
    src_str, dst_str = args
    src = Path(src_str)
    dst = Path(dst_str)

    if dst.exists():
        return src_str, dst_str, "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)

    # silenceremove: trim silence from start + end
    # dynaudnorm: fast single-pass volume normalization (loudnorm needs 2 passes = slow)
    # aresample=16000: ensure 16kHz
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
    result = subprocess.run(cmd, capture_output=True, timeout=60)

    if result.returncode != 0 or not dst.exists():
        # Fallback: just resample, skip VAD
        cmd_fallback = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", src_str,
            "-af", "dynaudnorm=f=150:g=15,aresample=16000",
            dst_str
        ]
        subprocess.run(cmd_fallback, capture_output=True, timeout=60)
        return src_str, dst_str, "fallback"

    return src_str, dst_str, "ok"

# Build task list
tasks = []
for r in kept:
    src = Path(r["audio_path"])
    try:
        rel = src.relative_to(EXTRACTED)
        dst = str(AUDIO_CLEAN / rel)
    except ValueError:
        dst = r["audio_path"]  # not under extracted/, keep original
    tasks.append((r["audio_path"], dst))

# Run in parallel
failed = []
new_audio_paths = {}
done_count = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_one, t): t for t in tasks}
    for future in as_completed(futures):
        src_str, dst_str, status = future.result()
        new_audio_paths[src_str] = dst_str
        if status == "fallback":
            failed.append(src_str)
        done_count += 1
        if done_count % 5000 == 0:
            pct = done_count / len(tasks) * 100
            print(f"  [{pct:5.1f}%] {done_count}/{len(tasks)} done  "
                  f"(fallbacks: {len(failed)})", flush=True)

print(f"\n  Done. Fallbacks (VAD skipped): {len(failed)}")
if failed:
    fail_log = BASE / "vad_failures.txt"
    with open(fail_log, "w") as f:
        f.write("\n".join(failed))
    print(f"  Failure list saved to: {fail_log}")

# Update audio paths
for r in kept:
    if r["audio_path"] in new_audio_paths:
        r["audio_path"] = new_audio_paths[r["audio_path"]]

# ── Step 4: Recompute durations after VAD ─────────────────────────────────────
print("\n[Step 4] Recomputing durations after VAD trimming...")

final = []
too_short_after_vad = 0

for i, r in enumerate(kept):
    if i % 5000 == 0:
        print(f"  {i}/{len(kept)} ...")

    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         r["audio_path"]],
        capture_output=True, text=True, timeout=10
    )

    if result.stdout.strip():
        new_dur = float(result.stdout.strip())
        if new_dur < MIN_DUR:
            too_short_after_vad += 1
            continue
        r = dict(r)
        r["duration_sec"] = str(round(new_dur, 4))
        final.append(r)
    else:
        # ffprobe failed — keep original duration, keep the clip
        final.append(r)

print(f"  Removed {too_short_after_vad} clips that became too short after VAD")
print(f"  Final clip count: {len(final)}")

# ── Step 5: Save manifest_clean.csv (all levels) ──────────────────────────────
print("\n[Step 5] Saving manifest_clean.csv ...")
with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final)
print(f"  Saved: {MANIFEST_OUT}  ({len(final)} clips)")

# ── Step 6: Save asr_manifest_clean.csv (Paragraph + Story + Sentence only) ───
print("\n[Step 6] Saving asr_manifest_clean.csv (connected speech only)...")
asr_rows = [r for r in final if r["reading_level"] in ASR_LEVELS]
with open(ASR_MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asr_rows)
print(f"  Saved: {ASR_MANIFEST}  ({len(asr_rows)} clips)")

# ── Step 7: Update train/dev/test splits ──────────────────────────────────────
print("\n[Step 7] Updating train/dev/test splits...")

# Build lookup: old_audio_path → cleaned row
old_to_row = {}
for r in final:
    # derive original path from cleaned path
    try:
        rel = Path(r["audio_path"]).relative_to(AUDIO_CLEAN)
        old_path = str(EXTRACTED / rel)
    except ValueError:
        old_path = r["audio_path"]
    old_to_row[old_path] = r

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
        if old_path in old_to_row:
            updated_row = dict(old_to_row[old_path])
            cleaned_split.append(updated_row)

    out_path = BASE / f"{split_name}_clean.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_split)

    print(f"  {split_name:<12}: {len(split_rows):>6} → {len(cleaned_split):>6} clips  →  {out_path.name}")

# ── Step 8: Final statistics ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL CLEANING SUMMARY")
print("=" * 60)

print(f"\nOriginal clips   : {len(original_rows)}")
print(f"Final clips      : {len(final)}")
print(f"Total removed    : {len(original_rows) - len(final)}")

total_hrs = sum(float(r["duration_sec"]) for r in final) / 3600
print(f"Total duration   : {total_hrs:.2f} hrs")

print(f"\nBy language:")
lang_counts = Counter(r["script_language"] for r in final)
for lang, cnt in lang_counts.most_common():
    hrs = sum(float(r["duration_sec"]) for r in final if r["script_language"] == lang) / 3600
    print(f"  {lang:<12}: {cnt:>6} clips   {hrs:.2f} hrs")

print(f"\nBy reading level:")
level_counts = Counter(r["reading_level"] for r in final)
for level, cnt in level_counts.most_common():
    tag = " ← ASR" if level in ASR_LEVELS else ""
    print(f"  {level:<35}: {cnt:>6}{tag}")

print(f"\nASR-only clips   : {len(asr_rows)}")
asr_hrs = sum(float(r["duration_sec"]) for r in asr_rows) / 3600
print(f"ASR duration     : {asr_hrs:.2f} hrs")

unique_children = len(set(r["child_id"] for r in final))
print(f"\nUnique children  : {unique_children}")

print("\n" + "=" * 60)
print("Done. Files ready for training:")
print(f"  Full manifest  : {MANIFEST_OUT}")
print(f"  ASR manifest   : {ASR_MANIFEST}")
print(f"  ASR splits     : asr_train_clean.csv / asr_dev_clean.csv / asr_test_clean.csv")
print(f"  Full splits    : train_clean.csv / dev_clean.csv / test_clean.csv")
print(f"  Clean audio    : {AUDIO_CLEAN}/")
print("=" * 60)
