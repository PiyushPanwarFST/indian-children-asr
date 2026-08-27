"""
alignment_visualizer.py — Show MMS_FA CTC alignment output for any clip

Usage:
    python scripts/alignment_visualizer.py
    python scripts/alignment_visualizer.py --audio path/to/clip.wav --transcript "text here"
    python scripts/alignment_visualizer.py --save  # also save JSON output

What this shows:
  - For each word in the transcript: start time, end time, duration, confidence score
  - This is the raw output that 02_clean_audio.py used internally to find chunk split points
  - The alignment was NOT saved during preprocessing (only chunk audio files were saved)
  - This script re-runs alignment on any clip on demand for verification/inspection

Why CTC forced alignment works:
  MMS_FA is a CTC-based model trained to match audio frames to text tokens.
  Unlike a pure ASR model (which predicts text from audio), forced alignment
  goes in both directions: given the TRANSCRIPT and AUDIO, find WHERE each
  word occurs in time. It cannot fail to align — it must assign every frame
  to some token (which is why we filter single-word chunks >30s: that's silence
  being forced onto the nearest word).
"""

import argparse
import json
import torch
import torchaudio
import uroman as ur_module
from pathlib import Path

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--audio",      default=None, help="Path to audio file")
parser.add_argument("--transcript", default=None, help="Transcript text")
parser.add_argument("--save",       action="store_true", help="Save alignment to JSON")
args = parser.parse_args()

# ── Default: pick a known chunked Story clip for demo ─────────────────────────
DEFAULT_AUDIO = "/home/hp/Indain_children_spech/ASER-Dataset/extracted/Hindi RJ/3439/HI_S1_ST_0.wav"
DEFAULT_TEXT  = None  # will auto-read from processed_manifest

if args.audio is None:
    # Find the original file from the manifest (before chunking)
    import csv
    MANIFEST = "/home/hp/Indain_children_spech/ASER-Dataset/raw_manifest.csv"
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if "ST" in row.get("que_id", "") and row["language"] == "Hindi":
                audio_path = row["audio_path"]
                transcript = row["transcript"]
                # Only use clips that were actually chunked
                if Path(audio_path).exists():
                    dur = float(row["duration_sec"])
                    if 15.0 < dur < 60.0:  # likely to have been chunked
                        DEFAULT_AUDIO = audio_path
                        DEFAULT_TEXT  = transcript
                        break
    AUDIO_PATH = DEFAULT_AUDIO
    TRANSCRIPT = DEFAULT_TEXT
else:
    AUDIO_PATH = args.audio
    TRANSCRIPT = args.transcript

if TRANSCRIPT is None:
    print("No transcript provided. Use --transcript 'text here'")
    exit(1)

print(f"\nAudio:      {AUDIO_PATH}")
print(f"Transcript: {TRANSCRIPT[:80]}...")
print(f"Duration:   ", end="")

# Load audio
wav, sr = torchaudio.load(AUDIO_PATH)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)

duration = wav.shape[1] / 16000
print(f"{duration:.2f}s")

# ── Load MMS_FA ───────────────────────────────────────────────────────────────
print("\nLoading MMS_FA model...")
bundle    = torchaudio.pipelines.MMS_FA
fa_model  = bundle.get_model()
fa_tokenizer = bundle.get_tokenizer()
fa_aligner   = bundle.get_aligner()
ur = ur_module.Uroman()
print("Model loaded.")

# ── Romanize transcript (MMS_FA operates on romanized text) ───────────────────
words       = TRANSCRIPT.split()
words_roman = [ur.romanize_string(w) for w in words]

print(f"\nOriginal words ({len(words)}):  {' | '.join(words[:8])}{'...' if len(words) > 8 else ''}")
print(f"Romanized words ({len(words_roman)}): {' | '.join(words_roman[:8])}{'...' if len(words_roman) > 8 else ''}")

# ── Run CTC forced alignment ───────────────────────────────────────────────────
print("\nRunning CTC forced alignment...")
tokens = fa_tokenizer(words_roman)
with torch.inference_mode():
    emission, _ = fa_model(wav)    # emission: (1, T_frames, vocab_size)

print(f"Emission shape: {list(emission.shape)}  (T={emission.shape[1]} frames × vocab={emission.shape[2]})")

token_spans = fa_aligner(emission[0], tokens)

# Each token_span gives: (start_frame, end_frame, score) per character token
# We aggregate character spans → word spans
ratio = wav.shape[1] / emission.shape[1] / bundle.sample_rate  # seconds per frame

# ── Build word-level alignment ─────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"  {'#':<4} {'Word':<20} {'Romanized':<20} {'Start':>7} {'End':>7} {'Dur':>6}  {'Score':>7}")
print(f"  {'-'*4} {'-'*20} {'-'*20} {'-'*7} {'-'*7} {'-'*6}  {'-'*7}")

alignment_records = []
word_idx = 0
for span_group, orig_word, roman_word in zip(token_spans, words, words_roman):
    if not span_group:
        continue
    start_frame = span_group[0].start
    end_frame   = span_group[-1].end
    score       = sum(s.score for s in span_group) / len(span_group)

    start_sec = start_frame * ratio
    end_sec   = end_frame   * ratio
    dur_sec   = end_sec - start_sec

    print(f"  {word_idx:<4} {orig_word:<20} {roman_word:<20} {start_sec:>6.3f}s {end_sec:>6.3f}s {dur_sec:>5.3f}s  {score:>7.4f}")

    alignment_records.append({
        "word_index":  word_idx,
        "word":        orig_word,
        "romanized":   roman_word,
        "start_sec":   round(start_sec, 4),
        "end_sec":     round(end_sec, 4),
        "duration_sec": round(dur_sec, 4),
        "confidence":  round(float(score), 4),
        "start_frame": int(start_frame),
        "end_frame":   int(end_frame),
    })
    word_idx += 1

print(f"\n  Total words aligned: {len(alignment_records)}")
print(f"  Clip duration: {duration:.3f}s")
print(f"  Alignment coverage: {alignment_records[-1]['end_sec']:.3f}s (should be close to clip duration)")

# ── Show gap detection (what 02_clean_audio.py used) ──────────────────────────
GAP_THRESHOLD = 5.0  # seconds
print(f"\n  Gap detection (threshold = {GAP_THRESHOLD}s):")
gaps_found = 0
for i in range(len(alignment_records) - 1):
    gap_start = alignment_records[i]["end_sec"]
    gap_end   = alignment_records[i+1]["start_sec"]
    gap_dur   = gap_end - gap_start
    if gap_dur > GAP_THRESHOLD:
        print(f"    GAP found between word {i} ('{alignment_records[i]['word']}') and word {i+1} ('{alignment_records[i+1]['word']}'): {gap_dur:.2f}s")
        gaps_found += 1

if gaps_found == 0:
    print(f"    No gaps >{GAP_THRESHOLD}s found in this clip.")

# ── Save to JSON ──────────────────────────────────────────────────────────────
if args.save:
    out = {
        "audio_path": AUDIO_PATH,
        "transcript": TRANSCRIPT,
        "duration_sec": round(duration, 4),
        "num_frames": int(emission.shape[1]),
        "seconds_per_frame": round(ratio, 6),
        "word_alignments": alignment_records,
    }
    save_path = Path(AUDIO_PATH).stem + "_alignment.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nAlignment saved to: {save_path}")

print("\nDone.")
