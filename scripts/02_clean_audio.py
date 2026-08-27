"""
Script 2: Clean Audio (Silero VAD + Normalization + MMS_FA Chunking)

What this script does:
  1. Loads raw_manifest.csv from Script 1
  2. Removes invalid clips (zero duration, bad labels)
  3. Normalizes transcripts (remove punctuation)
  4. Silero VAD detects speech boundaries (start/end trim points + internal gaps)
  5. ffmpeg trims start/end silence + volume normalization (NO middle silence removal)
  6. Removes clips that became too short after trimming (<0.5s)
  7. MMS_FA forced alignment chunks clips >30s OR with internal gaps >5s
  8. Saves processed_manifest.csv

Input:  ASER-Dataset/raw_manifest.csv
Output: ASER-Dataset/processed_manifest.csv + ASER-Dataset/audio_processed/

Why Silero VAD instead of ffmpeg silenceremove?
  - ffmpeg silenceremove with stop_periods=-1 removes ALL internal silence and
    concatenates remaining speech — creates splice artifacts (waveform discontinuities)
  - Silero VAD is a neural network detector that gives precise speech timestamps
  - We only trim start/end silence, never remove middle silence

Why keep internal silence?
  - Natural pauses are part of speech prosody
  - Standard practice (Kaldi, ESPnet): keep silence <5s, only chunk at gaps >5s
"""

import csv
import json
import os
import re
import shutil
import subprocess
import multiprocessing
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchaudio

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
RAW_CSV      = ASER_ROOT / "raw_manifest.csv"
EXTRACTED    = ASER_ROOT / "extracted"
AUDIO_OUT    = ASER_ROOT / "audio_processed"
OUTPUT_CSV   = ASER_ROOT / "processed_manifest.csv"
VAD_CACHE    = ASER_ROOT / "vad_cache.json"

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# ══════════════════════════════════════════════════════════════════════════════
# Step 0: Clean up previous run outputs
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 0: Cleaning up previous outputs...")
print("=" * 60)

cleanup_targets = [
    ASER_ROOT / "audio_processed",
    ASER_ROOT / "processed_manifest.csv",
    ASER_ROOT / "vad_cache.json",
]

for target in cleanup_targets:
    if target.is_dir():
        shutil.rmtree(target)
        print(f"  Deleted directory: {target}")
    elif target.is_file():
        target.unlink()
        print(f"  Deleted file: {target}")
    else:
        print(f"  Already clean: {target}")

AUDIO_OUT.mkdir(parents=True, exist_ok=True)
print("\nCleanup complete. Starting fresh.\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load raw manifest
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading raw manifest...")
print("=" * 60)

def get_duration(audio_path):
    """Get audio duration in seconds using ffprobe."""
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

with open(RAW_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_clips = list(reader)

raw_count = len(all_clips)
raw_hours = sum(float(r["duration_sec"]) for r in all_clips) / 3600

print(f"Loaded: {raw_count:,} clips | {raw_hours:.2f} hours")
print(f"Workers available: {NUM_WORKERS}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Remove invalid clips
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: Removing invalid clips...")
print("=" * 60)

valid_clips = []
removed = {"zero_duration": 0, "bad_label": 0, "empty_transcript": 0, "unknown_lang": 0}

for r in all_clips:
    dur = float(r["duration_sec"])

    if dur == 0.0:
        removed["zero_duration"] += 1
        continue

    if r["reading_level"] == "Cl":
        removed["bad_label"] += 1
        continue

    if not r["transcript"].strip():
        removed["empty_transcript"] += 1
        continue

    # Fix unknown language for Letter level
    if r["language"] == "Unknown" and r["reading_level"] == "Letter":
        r["language"] = "Hindi"

    if r["language"] == "Unknown":
        removed["unknown_lang"] += 1
        continue

    valid_clips.append(r)

removed_count = raw_count - len(valid_clips)
removed_hours = raw_hours - sum(float(r["duration_sec"]) for r in valid_clips) / 3600

print(f"Removed {removed_count} invalid clips ({removed_hours:.2f} hours):")
for reason, count in removed.items():
    print(f"  {reason}: {count}")
print(f"\nRemaining: {len(valid_clips):,} clips | {sum(float(r['duration_sec']) for r in valid_clips)/3600:.2f} hours\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Normalize transcripts
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: Normalizing transcripts...")
print("=" * 60)

REMOVE_CHARS = str.maketrans("", "", "।.,;:!?\"'()[]{}\u2013\u2014-/\\")

def normalize_transcript(text):
    text = text.strip()
    text = text.translate(REMOVE_CHARS)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

empty_after = 0
for r in valid_clips:
    r["transcript"] = normalize_transcript(r["transcript"])
    if not r["transcript"]:
        empty_after += 1

valid_clips = [r for r in valid_clips if r["transcript"]]

print(f"Transcripts normalized. Empty after normalization: {empty_after}")
print(f"Remaining: {len(valid_clips):,} clips")

ex = [r for r in valid_clips if r["reading_level"] == "Paragraph"]
if ex:
    print(f"\nExample transcript: {ex[0]['transcript'][:80]}...")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Silero VAD — Detect speech boundaries for ALL clips
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4: Running Silero VAD on all clips...")
print("=" * 60)

silero_model, silero_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, _, _) = silero_utils
print("Silero VAD model loaded.")

INTERNAL_GAP_THRESHOLD = 5.0   # seconds — gaps longer than this trigger chunking
SILERO_SR = 16000               # Silero VAD requires 16kHz

# Load cached VAD results if they exist (crash recovery)
vad_cache = {}
if VAD_CACHE.exists():
    with open(VAD_CACHE, "r") as f:
        vad_cache = json.load(f)
    print(f"Loaded {len(vad_cache)} cached VAD results from previous run.")

print(f"\nRunning Silero VAD on {len(valid_clips):,} clips...")
print(f"Internal gap threshold: {INTERNAL_GAP_THRESHOLD}s")

vad_results = {}
done = 0
errors = 0

for r in valid_clips:
    audio_path = r["audio_path"]

    # Use cached result if available
    if audio_path in vad_cache:
        vad_results[audio_path] = vad_cache[audio_path]
        done += 1
        continue

    try:
        wav = read_audio(audio_path, sampling_rate=SILERO_SR)
        speech_timestamps = get_speech_timestamps(wav, silero_model, sampling_rate=SILERO_SR)

        if not speech_timestamps:
            vad_results[audio_path] = {
                "trim_start": 0.0,
                "trim_end": 0.0,
                "has_long_gap": False,
                "gaps": [],
                "no_speech": True
            }
        else:
            trim_start = speech_timestamps[0]["start"] / SILERO_SR
            trim_end = speech_timestamps[-1]["end"] / SILERO_SR

            gaps = []
            for i in range(1, len(speech_timestamps)):
                gap_start = speech_timestamps[i-1]["end"] / SILERO_SR
                gap_end = speech_timestamps[i]["start"] / SILERO_SR
                gap_dur = gap_end - gap_start
                if gap_dur > INTERNAL_GAP_THRESHOLD:
                    gaps.append({"start": gap_start, "end": gap_end, "duration": gap_dur})

            vad_results[audio_path] = {
                "trim_start": round(trim_start, 4),
                "trim_end": round(trim_end, 4),
                "has_long_gap": len(gaps) > 0,
                "gaps": gaps,
                "no_speech": False
            }
    except Exception as e:
        vad_results[audio_path] = {
            "trim_start": 0.0,
            "trim_end": float(r["duration_sec"]),
            "has_long_gap": False,
            "gaps": [],
            "no_speech": False,
            "error": str(e)
        }
        errors += 1

    done += 1
    if done % 5000 == 0:
        print(f"  [{done:,}/{len(valid_clips):,}] done | errors: {errors}")
        # Save checkpoint
        with open(VAD_CACHE, "w") as f:
            json.dump({**vad_cache, **vad_results}, f)

# Final save of all VAD results
with open(VAD_CACHE, "w") as f:
    json.dump(vad_results, f)

no_speech = sum(1 for v in vad_results.values() if v.get("no_speech", False))
has_gaps = sum(1 for v in vad_results.values() if v.get("has_long_gap", False))

print(f"\nSilero VAD complete!")
print(f"  Processed: {done:,} clips | Errors: {errors}")
print(f"  No speech detected: {no_speech} clips (will be removed)")
print(f"  Clips with internal gaps >{INTERNAL_GAP_THRESHOLD}s: {has_gaps} (will be chunked)")
print(f"  VAD cache saved to: {VAD_CACHE}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: ffmpeg — Trim start/end silence + Volume normalize
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: Trimming + normalizing audio with ffmpeg...")
print("=" * 60)

def process_audio_trimmed(args):
    """Trim start/end using VAD timestamps + volume normalize with ffmpeg."""
    src, dst, trim_start, trim_end = args
    if os.path.exists(dst):
        return src, dst, "skip"

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    duration = trim_end - trim_start
    if duration < 0.1:
        return src, dst, "too_short"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-ss", str(trim_start),
        "-t", str(duration),
        "-af", "dynaudnorm=f=150:g=15,aresample=16000",
        dst
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not os.path.exists(dst):
            return src, dst, "failed"
        return src, dst, "ok"
    except Exception:
        return src, dst, "failed"

# Build task list using VAD results
tasks = []
no_speech_clips = []

for r in valid_clips:
    src = r["audio_path"]
    vad = vad_results.get(src, {})

    if vad.get("no_speech", False):
        no_speech_clips.append(r)
        continue

    trim_start = vad.get("trim_start", 0.0)
    trim_end = vad.get("trim_end", float(r["duration_sec"]))

    rel = Path(src).relative_to(EXTRACTED)
    dst = str(AUDIO_OUT / rel.with_suffix(".wav"))
    tasks.append((src, dst, trim_start, trim_end))

print(f"Clips to process: {len(tasks):,}")
print(f"Clips with no speech (skipped): {len(no_speech_clips)}")
print(f"Processing with {NUM_WORKERS} workers...")

# Run ffmpeg trim + normalize in parallel
path_mapping = {}
done = 0
fallbacks = 0
failures = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_audio_trimmed, t): t for t in tasks}
    for future in as_completed(futures):
        try:
            src, dst, status = future.result()
            if status in ("ok", "skip"):
                path_mapping[src] = dst
            elif status == "too_short":
                fallbacks += 1
            else:
                failures += 1
        except Exception:
            failures += 1
        done += 1
        if done % 5000 == 0:
            pct = done / len(tasks) * 100
            print(f"  [{pct:.0f}%] {done:,}/{len(tasks):,} done | failures: {failures}")

print(f"\nDone! Processed: {done:,} | Too-short: {fallbacks} | Failures: {failures}")

# Update audio paths and remove clips that failed processing
processed_clips = []
for r in valid_clips:
    if r["audio_path"] in path_mapping:
        r["audio_path"] = path_mapping[r["audio_path"]]
        processed_clips.append(r)

print(f"Clips with processed audio: {len(processed_clips):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Recompute durations and categorize clips
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 6: Recomputing durations and categorizing...")
print("=" * 60)

reverse_mapping = {v: k for k, v in path_mapping.items()}

short_clips = []
normal_clips = []
needs_chunking = []

for i, r in enumerate(processed_clips):
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(processed_clips)}...")

    new_dur = get_duration(r["audio_path"])
    r["duration_sec"] = str(round(new_dur, 4))

    if new_dur < 0.5:
        short_clips.append(r)
        continue

    orig_path = reverse_mapping.get(r["audio_path"], r["audio_path"])
    vad = vad_results.get(orig_path, {})
    has_long_gap = vad.get("has_long_gap", False)

    if new_dur > 30.0 or has_long_gap:
        r["_gaps"] = vad.get("gaps", [])
        r["_trim_start"] = vad.get("trim_start", 0.0)
        needs_chunking.append(r)
    else:
        normal_clips.append(r)

short_hrs = sum(float(r["duration_sec"]) for r in short_clips) / 3600
normal_hrs = sum(float(r["duration_sec"]) for r in normal_clips) / 3600
chunk_hrs = sum(float(r["duration_sec"]) for r in needs_chunking) / 3600

print(f"\nAfter trimming:")
print(f"  Too short (<0.5s):           {len(short_clips):,} clips | {short_hrs:.2f} hrs -> REMOVED")
print(f"  Normal (0.5-30s, no gap):    {len(normal_clips):,} clips | {normal_hrs:.2f} hrs -> KEPT")
print(f"  Needs chunking (>30s/gap):   {len(needs_chunking):,} clips | {chunk_hrs:.2f} hrs -> CHUNK")

long_only = [r for r in needs_chunking if float(r["duration_sec"]) > 30.0 and not r.get("_gaps")]
gap_only = [r for r in needs_chunking if float(r["duration_sec"]) <= 30.0 and r.get("_gaps")]
both = [r for r in needs_chunking if float(r["duration_sec"]) > 30.0 and r.get("_gaps")]
print(f"\n  Chunking breakdown:")
print(f"    >30s only:         {len(long_only)}")
print(f"    Gap >5s only:      {len(gap_only)}")
print(f"    Both >30s + gap:   {len(both)}")

print(f"\n  Too-short clips by reading level:")
for lvl, cnt in Counter(r["reading_level"] for r in short_clips).most_common():
    print(f"    {lvl}: {cnt}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Chunk clips using MMS_FA CTC Forced Alignment
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 7: Chunking with MMS_FA forced alignment...")
print("=" * 60)

import uroman as ur_module
from torchaudio.pipelines import MMS_FA as bundle

print("Loading MMS Forced Alignment model...")
fa_model = bundle.get_model()
fa_tokenizer = bundle.get_tokenizer()
fa_aligner = bundle.get_aligner()
ur = ur_module.Uroman()
print("Model loaded.")

MAX_CHUNK_SEC = 25.0
MIN_CHUNK_SEC = 0.5
CHUNKS_DIR = ASER_ROOT / "audio_processed" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Buffer: 5 frames × 20ms/frame = 100ms = 1600 samples at 16kHz
# CTC alignment gives the frame where the last character PEAKS, but the
# actual acoustic energy may trail a few frames beyond. Adding a small
# buffer prevents clipping the tail of the last word. (Professor suggestion)
FRAME_BUFFER = 5
SAMPLE_BUFFER = int(FRAME_BUFFER * (bundle.sample_rate / 50))  # 50 fps = 20ms/frame → 1600 samples

ASR_LEVELS = {"Paragraph", "Story", "Sentence (English)"}

asr_to_chunk = [r for r in needs_chunking if r["reading_level"] in ASR_LEVELS]
non_asr_skipped = [r for r in needs_chunking if r["reading_level"] not in ASR_LEVELS]

print(f"\nClips needing chunking: {len(needs_chunking)}")
print(f"  ASR levels (will chunk): {len(asr_to_chunk)}")
print(f"  Non-ASR (skipped):       {len(non_asr_skipped)}")
print(f"  Frame buffer: {FRAME_BUFFER} frames = {SAMPLE_BUFFER/bundle.sample_rate*1000:.0f}ms")

chunked_clips = []
failed_chunks = []

for idx, clip in enumerate(asr_to_chunk):
    if idx % 100 == 0:
        print(f"  [{idx}/{len(asr_to_chunk)}] Chunking... ({len(chunked_clips)} chunks so far)")

    audio_path = clip["audio_path"]
    transcript = clip["transcript"].strip()

    if not transcript or not os.path.exists(audio_path):
        failed_chunks.append(clip)
        continue

    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        total_samples = waveform.shape[1]

        words_orig = transcript.split()
        romanized = ur.romanize_string(transcript)
        words_roman = romanized.split()

        if len(words_orig) != len(words_roman):
            words_roman = [ur.romanize_string(w) for w in words_orig]

        if len(words_orig) < 2:
            if float(clip["duration_sec"]) <= 30.0:
                row = dict(clip)
                row.pop("_gaps", None)
                row.pop("_trim_start", None)
                normal_clips.append(row)
            else:
                failed_chunks.append(clip)
            continue

        tokens = fa_tokenizer(words_roman)
        with torch.inference_mode():
            emission, _ = fa_model(waveform)
        token_spans = fa_aligner(emission[0], tokens)
        ratio = waveform.shape[1] / emission.shape[1] / bundle.sample_rate

        word_times = []
        for i, spans in enumerate(token_spans):
            ws = spans[0].start * ratio
            we = spans[-1].end * ratio
            word_times.append((ws, we, words_orig[i]))

        trim_offset = clip.get("_trim_start", 0.0)
        gap_boundaries = set()
        for gap in clip.get("_gaps", []):
            gap_mid = ((gap["start"] + gap["end"]) / 2.0) - trim_offset
            if gap_mid > 0:
                gap_boundaries.add(gap_mid)

        chunks = []
        chunk_words = []
        chunk_word_times = []
        chunk_start_time = None

        for i, (ws, we, word) in enumerate(word_times):
            if chunk_start_time is None:
                chunk_start_time = ws

            chunk_duration = we - chunk_start_time

            should_cut = False

            if chunk_duration > MAX_CHUNK_SEC and chunk_words:
                should_cut = True

            if i > 0 and chunk_words:
                prev_end = word_times[i-1][1]
                for gb in gap_boundaries:
                    if prev_end <= gb <= ws:
                        should_cut = True
                        break

            if should_cut:
                prev_ws, prev_we, _ = word_times[i-1]
                start_sample = int(chunk_word_times[0][0] * bundle.sample_rate)
                end_sample = min(int(prev_we * bundle.sample_rate) + SAMPLE_BUFFER, total_samples)
                dur = (end_sample - start_sample) / bundle.sample_rate
                if dur >= MIN_CHUNK_SEC:
                    text = " ".join(chunk_words)
                    chunks.append((start_sample, end_sample, text, dur))

                chunk_words = [word]
                chunk_word_times = [(ws, we)]
                chunk_start_time = ws
            else:
                chunk_words.append(word)
                chunk_word_times.append((ws, we))

        # Save last chunk
        if chunk_words and chunk_word_times:
            start_sample = int(chunk_word_times[0][0] * bundle.sample_rate)
            end_sample = min(int(chunk_word_times[-1][1] * bundle.sample_rate) + SAMPLE_BUFFER, total_samples)
            dur = (end_sample - start_sample) / bundle.sample_rate
            if dur >= MIN_CHUNK_SEC:
                text = " ".join(chunk_words)
                chunks.append((start_sample, end_sample, text, dur))

        # Filter out bad chunks: single-word chunks >30s are MMS_FA alignment
        # failures (CTC assigns long silence to adjacent words). These contain
        # mostly silence and are not useful training data.
        chunks = [c for c in chunks if not (len(c[2].split()) <= 1 and c[3] > 30.0)]

        # If no valid chunks remain after filtering, skip this clip
        if not chunks:
            failed_chunks.append(clip)
            continue

        # If only 1 chunk and <=30s, save it as the trimmed version
        # (NOT the original file — the original may have trailing silence
        # that makes it >30s, while the chunk is the speech-only portion)
        base_name = Path(audio_path).stem
        chunk_dir = CHUNKS_DIR / clip["region"] / clip["child_id"]
        chunk_dir.mkdir(parents=True, exist_ok=True)

        if len(chunks) == 1 and chunks[0][3] <= 30.0:
            start_s, end_s, text, dur = chunks[0]
            chunk_audio = waveform[:, start_s:end_s]
            chunk_path = chunk_dir / f"{base_name}_trimmed.wav"
            torchaudio.save(str(chunk_path), chunk_audio, bundle.sample_rate)

            row = dict(clip)
            row["audio_path"] = str(chunk_path)
            row["transcript"] = text
            row["duration_sec"] = str(round(dur, 4))
            row.pop("_gaps", None)
            row.pop("_trim_start", None)
            normal_clips.append(row)
            continue

        # Save each chunk as a separate audio file
        for ci, (start_s, end_s, text, dur) in enumerate(chunks):
            # Skip individual chunks >30s (alignment failures)
            if dur > 30.0:
                continue
            chunk_audio = waveform[:, start_s:end_s]
            chunk_path = chunk_dir / f"{base_name}_chunk{ci}.wav"
            torchaudio.save(str(chunk_path), chunk_audio, bundle.sample_rate)

            row = dict(clip)
            row["audio_path"] = str(chunk_path)
            row["transcript"] = text
            row["duration_sec"] = str(round(dur, 4))
            row.pop("_gaps", None)
            row.pop("_trim_start", None)
            chunked_clips.append(row)

    except Exception as e:
        failed_chunks.append(clip)
        continue

chunk_hrs = sum(float(c["duration_sec"]) for c in chunked_clips) / 3600
print(f"\nChunking complete!")
print(f"  Chunks created: {len(chunked_clips):,} | {chunk_hrs:.2f} hours")
print(f"  Failed: {len(failed_chunks)}")
print(f"  Skipped (non-ASR): {len(non_asr_skipped)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Combine everything and save processed_manifest.csv
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 8: Saving processed_manifest.csv...")
print("=" * 60)

for r in normal_clips:
    r.pop("_gaps", None)
    r.pop("_trim_start", None)

final_clips = normal_clips + chunked_clips

# Remove clips still >30s (MMS_FA alignment failures / 1-chunk fallback edge cases)
over30 = [r for r in final_clips if float(r["duration_sec"]) > 30.0]
final_clips = [r for r in final_clips if float(r["duration_sec"]) <= 30.0]
over30_hrs = sum(float(r["duration_sec"]) for r in over30) / 3600
print(f"Removed {len(over30)} clips still >30s ({over30_hrs:.2f}h) — alignment edge cases")

# Remove duplicate rows (same audio_path appearing more than once)
seen_paths = set()
deduped = []
dupes_removed = 0
for r in final_clips:
    if r["audio_path"] not in seen_paths:
        seen_paths.add(r["audio_path"])
        deduped.append(r)
    else:
        dupes_removed += 1
final_clips = deduped
print(f"Removed {dupes_removed} duplicate rows")

fieldnames = ["audio_path", "transcript", "language", "region", "reading_level",
              "que_id", "is_correct", "num_mistakes", "age_group",
              "student_class", "child_id", "date", "duration_sec"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_clips)

final_hrs = sum(float(r["duration_sec"]) for r in final_clips) / 3600
print(f"Saved: {OUTPUT_CSV}")
print(f"Total clips: {len(final_clips):,}")
print(f"Total duration: {final_hrs:.2f} hours\n")

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Duration accounting
# ══════════════════════════════════════════════════════════════════════════════
final_hrs = sum(float(r["duration_sec"]) for r in final_clips) / 3600
invalid_hrs = removed_hours
no_speech_hrs = sum(float(r["duration_sec"]) for r in no_speech_clips) / 3600
short_hrs_val = sum(float(r["duration_sec"]) for r in short_clips) / 3600
skipped_long_hrs = sum(float(r["duration_sec"]) for r in non_asr_skipped) / 3600
failed_hrs = sum(float(r["duration_sec"]) for r in failed_chunks) / 3600
silence_trimmed = raw_hours - final_hrs - invalid_hrs - no_speech_hrs - short_hrs_val - skipped_long_hrs - failed_hrs

print("=" * 65)
print("  DURATION ACCOUNTING")
print("=" * 65)
print(f"  Raw data:                       {raw_hours:.2f} hours")
print()
print(f"  KEPT (processed speech):        {final_hrs:.2f} hours")
print(f"    Normal clips (0.5-30s):       {sum(float(r['duration_sec']) for r in normal_clips)/3600:.2f} hrs")
print(f"    Chunked clips:                {chunk_hrs:.2f} hrs")
print()
print(f"  REMOVED:                        {raw_hours - final_hrs:.2f} hours")
print(f"    Start/end silence trimmed:    {silence_trimmed:.2f} hrs")
print(f"    Too short after trim (<0.5s): {short_hrs_val:.2f} hrs ({len(short_clips)} clips)")
print(f"    No speech detected (VAD):     {no_speech_hrs:.2f} hrs ({len(no_speech_clips)} clips)")
print(f"    Invalid clips (Step 2):       {invalid_hrs:.2f} hrs ({removed_count} clips)")
print(f"    Skipped long non-ASR:         {skipped_long_hrs:.2f} hrs ({len(non_asr_skipped)} clips)")
print(f"    Failed chunking:              {failed_hrs:.2f} hrs ({len(failed_chunks)} clips)")
print()
check = final_hrs + silence_trimmed + short_hrs_val + no_speech_hrs + invalid_hrs + skipped_long_hrs + failed_hrs
print(f"  VERIFY: {check:.2f} hours (should equal {raw_hours:.2f})")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# Step 10: Statistics of processed dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PROCESSED DATASET STATISTICS")
print("=" * 65)
print(f"  Total clips:    {len(final_clips):,}")
print(f"  Total duration: {final_hrs:.2f} hours")
print(f"  Total children: {len(set(r['child_id'] for r in final_clips)):,}")

print(f"\n  --- By Language ---")
for lang in ["Hindi", "Marathi", "English"]:
    clips = [r for r in final_clips if r["language"] == lang]
    hrs = sum(float(r["duration_sec"]) for r in clips) / 3600
    print(f"    {lang:<15} {len(clips):>6,} clips | {hrs:>6.2f} hrs")

print(f"\n  --- By Reading Level ---")
levels = defaultdict(lambda: {"clips": 0, "hrs": 0.0})
for r in final_clips:
    levels[r["reading_level"]]["clips"] += 1
    levels[r["reading_level"]]["hrs"] += float(r["duration_sec"]) / 3600
for lvl in sorted(levels, key=lambda x: -levels[x]["hrs"]):
    d = levels[lvl]
    asr_tag = " <-- ASR" if lvl in ASR_LEVELS else ""
    print(f"    {lvl:<30} {d['clips']:>6,} clips | {d['hrs']:>6.2f} hrs{asr_tag}")

print(f"\n  --- By Region ---")
for reg in sorted(set(r["region"] for r in final_clips)):
    clips = [r for r in final_clips if r["region"] == reg]
    hrs = sum(float(r["duration_sec"]) for r in clips) / 3600
    print(f"    {reg:<30} {len(clips):>6,} clips | {hrs:>6.2f} hrs")

asr = [r for r in final_clips if r["reading_level"] in ASR_LEVELS]
asr_hrs = sum(float(r["duration_sec"]) for r in asr) / 3600
print(f"\n  --- ASR Subset (connected speech only) ---")
print(f"    Total: {len(asr):,} clips | {asr_hrs:.2f} hours")
for lang in ["Hindi", "Marathi", "English"]:
    clips = [r for r in asr if r["language"] == lang]
    hrs = sum(float(r["duration_sec"]) for r in clips) / 3600
    print(f"      {lang:<15} {len(clips):>5,} clips | {hrs:>5.2f} hrs")

print("\n" + "=" * 65)
print("  DONE! Run 03_create_splits.py next.")
print("=" * 65)
