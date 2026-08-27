"""
Chunk Long Clips (> 30 seconds) Using Forced Alignment
========================================================
Uses torchaudio MMS Forced Aligner to get word-level timestamps,
then splits audio at ~25s boundaries with matching transcripts.

Requires: torch, torchaudio, uroman, torchcodec
Run: source venv/bin/activate && python3 scripts/chunk_long_clips.py
"""

import csv
import os
import subprocess
import torch
import torchaudio
import uroman
from torchaudio.pipelines import MMS_FA as bundle
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
BASE           = Path("/home/hp/Indain_children_spech/ASER-Dataset")
LONG_MANIFEST  = BASE / "long_clips_manifest.csv"
MANIFEST_CLN   = BASE / "manifest_clean.csv"
AUDIO_CLEAN    = BASE / "audio_clean"
CHUNKS_DIR     = BASE / "audio_chunks"
MAX_CHUNK_SEC  = 25.0   # chunk at 25s to leave margin from 30s model limit
MIN_CHUNK_SEC  = 0.5    # discard chunks shorter than this
ASR_LEVELS     = {"Paragraph", "Story", "Sentence (English)"}

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load MMS FA model ─────────────────────────────────────────────────────────
print("Loading MMS Forced Alignment model...")
model = bundle.get_model()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
ur = uroman.Uroman()
print("  Model loaded.")

# ── Load long clips ───────────────────────────────────────────────────────────
with open(LONG_MANIFEST, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    long_clips = list(reader)
    fieldnames = reader.fieldnames

print(f"\nTotal long clips to chunk: {len(long_clips)}")

# ── Process each clip ─────────────────────────────────────────────────────────
all_chunks = []
failed = []
skipped = 0

for idx, clip in enumerate(long_clips):
    if idx % 100 == 0:
        print(f"\n[{idx}/{len(long_clips)}] Processing...")

    audio_path = clip["audio_path"]
    transcript = clip["transcript"].strip()

    # Skip non-ASR clips (only 5 English letters/words, not worth chunking)
    if clip["reading_level"] not in ASR_LEVELS:
        skipped += 1
        continue

    if not transcript or not os.path.exists(audio_path):
        failed.append((audio_path, "missing file or empty transcript"))
        continue

    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        # Romanize transcript for MMS FA
        words_orig = transcript.split()
        romanized = ur.romanize_string(transcript)
        words_roman = romanized.split()

        # Word count mismatch after romanization — skip
        if len(words_orig) != len(words_roman):
            failed.append((audio_path, f"word count mismatch: {len(words_orig)} vs {len(words_roman)}"))
            continue

        if len(words_orig) < 2:
            failed.append((audio_path, "too few words"))
            continue

        # Tokenize and align
        tokens = tokenizer(words_roman)

        with torch.inference_mode():
            emission, _ = model(waveform)

        word_spans = aligner(emission[0], tokens)
        ratio = waveform.shape[1] / emission.shape[1] / bundle.sample_rate

        # Build chunks at MAX_CHUNK_SEC boundaries
        chunks = []
        chunk_words = []
        chunk_start_sample = None
        chunk_start_time = None

        for i in range(len(words_orig)):
            ws = word_spans[i][0].start * ratio
            we = word_spans[i][-1].end * ratio
            ws_sample = int(word_spans[i][0].start * ratio * bundle.sample_rate)
            we_sample = int(word_spans[i][-1].end * ratio * bundle.sample_rate)

            if chunk_start_time is None:
                chunk_start_time = ws
                chunk_start_sample = ws_sample

            if we - chunk_start_time > MAX_CHUNK_SEC and chunk_words:
                # End current chunk at previous word
                prev_end_sample = int(word_spans[i-1][-1].end * ratio * bundle.sample_rate)
                chunk_text = " ".join(chunk_words)
                chunk_dur = (prev_end_sample - chunk_start_sample) / bundle.sample_rate
                if chunk_dur >= MIN_CHUNK_SEC:
                    chunks.append({
                        "start_sample": chunk_start_sample,
                        "end_sample": prev_end_sample,
                        "text": chunk_text,
                        "duration": chunk_dur,
                    })
                # Start new chunk from current word
                chunk_words = [words_orig[i]]
                chunk_start_time = ws
                chunk_start_sample = ws_sample
            else:
                chunk_words.append(words_orig[i])

        # Last chunk
        if chunk_words:
            last_end_sample = int(word_spans[len(words_orig)-1][-1].end * ratio * bundle.sample_rate)
            chunk_text = " ".join(chunk_words)
            chunk_dur = (last_end_sample - chunk_start_sample) / bundle.sample_rate
            if chunk_dur >= MIN_CHUNK_SEC:
                chunks.append({
                    "start_sample": chunk_start_sample,
                    "end_sample": last_end_sample,
                    "text": chunk_text,
                    "duration": chunk_dur,
                })

        # Save each chunk as a separate audio file
        child_id = clip["child_id"]
        region = clip["region"]
        base_name = Path(audio_path).stem

        for ci, chunk in enumerate(chunks):
            chunk_audio = waveform[:, chunk["start_sample"]:chunk["end_sample"]]
            chunk_filename = f"{base_name}_chunk{ci}.wav"
            chunk_dir = CHUNKS_DIR / region / child_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / chunk_filename

            torchaudio.save(str(chunk_path), chunk_audio, bundle.sample_rate)

            # Build manifest row
            row = dict(clip)
            row["audio_path"] = str(chunk_path)
            row["transcript"] = chunk["text"]
            row["duration_sec"] = str(round(chunk["duration"], 4))
            all_chunks.append(row)

    except Exception as e:
        failed.append((audio_path, str(e)[:100]))
        continue

print(f"\n{'='*60}")
print(f"CHUNKING COMPLETE")
print(f"{'='*60}")
print(f"  Clips processed: {len(long_clips)}")
print(f"  Skipped (non-ASR): {skipped}")
print(f"  Failed: {len(failed)}")
print(f"  Chunks created: {len(all_chunks)}")

chunk_hrs = sum(float(c["duration_sec"]) for c in all_chunks) / 3600
print(f"  Chunk duration: {chunk_hrs:.2f} hours")

if failed:
    print(f"\n  First 10 failures:")
    for path, err in failed[:10]:
        print(f"    {Path(path).name}: {err}")

# ── Add chunks to manifest_clean.csv ──────────────────────────────────────────
print(f"\nAdding {len(all_chunks)} chunks to manifest_clean.csv ...")

with open(MANIFEST_CLN, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    existing = list(reader)
    mf = reader.fieldnames

existing.extend(all_chunks)

with open(MANIFEST_CLN, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=mf)
    writer.writeheader()
    writer.writerows(existing)

print(f"  manifest_clean.csv: {len(existing)} clips")

# ── Update split files ────────────────────────────────────────────────────────
print("\nUpdating split files...")

# Load original split assignments to know which child is in which split
split_children = {}
for split_name in ["train", "dev", "test"]:
    with open(BASE / f"{split_name}.csv", encoding="utf-8") as f:
        children = set(r["child_id"] for r in csv.DictReader(f))
    split_children[split_name] = children

# Assign chunks to splits based on child_id
chunk_splits = {"train": [], "dev": [], "test": []}
for chunk in all_chunks:
    cid = chunk["child_id"]
    for sname, children in split_children.items():
        if cid in children:
            chunk_splits[sname].append(chunk)
            break

# Update *_clean.csv files
for split_name in ["train", "dev", "test"]:
    # Full split
    clean_path = BASE / f"{split_name}_clean.csv"
    with open(clean_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.extend(chunk_splits[split_name])
    with open(clean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mf)
        writer.writeheader()
        writer.writerows(rows)

    # ASR split (only ASR levels)
    asr_path = BASE / f"asr_{split_name}_clean.csv"
    with open(asr_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        asr_rows = list(reader)
    asr_chunks = [c for c in chunk_splits[split_name] if c["reading_level"] in ASR_LEVELS]
    asr_rows.extend(asr_chunks)
    with open(asr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mf)
        writer.writeheader()
        writer.writerows(asr_rows)

    full_hrs = sum(float(r["duration_sec"]) for r in rows) / 3600
    asr_hrs_s = sum(float(r["duration_sec"]) for r in asr_rows) / 3600
    print(f"  {split_name}_clean.csv: {len(rows)} clips ({full_hrs:.2f} hrs)")
    print(f"  asr_{split_name}_clean.csv: {len(asr_rows)} clips ({asr_hrs_s:.2f} hrs)")

# ── Final summary ─────────────────────────────────────────────────────────────
total_clips = len(existing)
total_hrs = sum(float(r["duration_sec"]) for r in existing) / 3600
asr_all = [r for r in existing if r["reading_level"] in ASR_LEVELS]
asr_total_hrs = sum(float(r["duration_sec"]) for r in asr_all) / 3600

print(f"\n{'='*60}")
print(f"FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Total clips:    {total_clips}")
print(f"  Total duration: {total_hrs:.2f} hours")
print(f"  ASR clips:      {len(asr_all)}")
print(f"  ASR duration:   {asr_total_hrs:.2f} hours")
print(f"{'='*60}")
