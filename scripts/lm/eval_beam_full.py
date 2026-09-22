"""
Full Test Set: Greedy vs Beam+LM CTC Decoding
==============================================
Runs ONLY our encoder + CTC head (no baseline, no teacher).
Compares greedy vs beam+LM decode on the full test set.
Fast enough for CPU.

Usage:
    python scripts/lm/eval_beam_full.py
    python scripts/lm/eval_beam_full.py --alpha 0.7 --beta 1.5
    python scripts/lm/eval_beam_full.py --split dev
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.wer import normalize_text, compute_corpus_wer
from scripts.utils.ctc_decode import ctc_greedy_decode, ctc_beam_decode, build_ctc_decoder

# ── Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="checkpoints/joint/joint_best_wer.pt")
parser.add_argument("--lm_path", type=str, default="models/lm/5gram.arpa")
parser.add_argument("--no_lm", action="store_true")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--beam_width", type=int, default=100)
parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
args = parser.parse_args()

# ── Paths ────────────────────────────────────────────────────────────────
ASER_ROOT = PROJECT_ROOT / "ASER-Dataset"
VOCAB_PATH = ASER_ROOT / "vocab.json"
SPLIT_CSV = ASER_ROOT / "splits" / f"asr_{args.split}.csv"
SAMPLE_RATE = 16000
MAX_DURATION = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

print("=" * 70)
print("  Full Evaluation: Greedy vs Beam+LM")
print("=" * 70)
print(f"  Device:     {DEVICE}")
print(f"  Checkpoint: {args.checkpoint}")
print(f"  LM:         {'None' if args.no_lm else args.lm_path}")
print(f"  Alpha={args.alpha}, Beta={args.beta}, Beam={args.beam_width}")
print(f"  Split:      {args.split}")

# ── Load vocab ───────────────────────────────────────────────────────────
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)
BLANK_IDX = char_to_idx["<blank>"]

# ── Load encoder + CTC head ─────────────────────────────────────────────
print("\nLoading model...")
from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)

ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

ckpt_path = PROJECT_ROOT / args.checkpoint
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"])
ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
print(f"  Loaded epoch {ckpt.get('epoch', '?')}, dev WER: {ckpt.get('dev_wer', 0)*100:.1f}%")

encoder.eval()
ctc_head.eval()
del whisper_model, ckpt

# ── Build beam decoder ───────────────────────────────────────────────────
print("Building beam decoder...")
lm_path = None if args.no_lm else str(PROJECT_ROOT / args.lm_path)
beam_decoder = build_ctc_decoder(
    vocab_path=str(VOCAB_PATH), lm_path=lm_path,
    alpha=args.alpha, beta=args.beta,
)
print(f"  Ready (LM: {'OFF' if args.no_lm else 'ON'})")

# ── Load all clips ───────────────────────────────────────────────────────
print(f"\nLoading clips from {args.split} set...")
clips = []
with open(SPLIT_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = row.get("language", "")
        lang_code = LANG_MAP.get(lang)
        if lang_code is None:
            continue
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ASER_ROOT / audio_path)
        if float(row.get("duration_sec", 0)) > MAX_DURATION:
            continue
        gt = row.get("transcript", row.get("que_text", row.get("text", "")))
        clips.append({
            "audio_path": audio_path, "language": lang,
            "lang_code": lang_code, "ground_truth": gt,
        })

print(f"  Total clips: {len(clips)}")
for lc in ["hi", "mr", "en"]:
    n = sum(1 for c in clips if c["lang_code"] == lc)
    print(f"    {lc.upper()}: {n}")


def compute_real_frames(num_samples):
    return min(num_samples // 160 // 2, 1500)


# ── Run ──────────────────────────────────────────────────────────────────
lang_data = {}  # lang_code -> {greedy_refs, greedy_hyps, beam_refs, beam_hyps}
errors = 0
t_start = time.time()

for clip in tqdm(clips, desc="Evaluating", bar_format="{l_bar}{bar:30}{r_bar}"):
    try:
        wav, sr = torchaudio.load(clip["audio_path"])
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze()
        if len(wav) < 4000:
            continue

        mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        real_frames = compute_real_frames(len(wav))

        with torch.no_grad():
            enc_out = encoder(mel.input_features.to(DEVICE))
            logits = ctc_head(enc_out.last_hidden_state)
            logits_slice = logits[0, :real_frames, :]

        greedy_text = ctc_greedy_decode(logits_slice, idx_to_char, BLANK_IDX)
        beam_text = ctc_beam_decode(logits_slice, beam_decoder, args.beam_width)

        ref = normalize_text(clip["ground_truth"])
        if not ref:
            continue

        lc = clip["lang_code"]
        if lc not in lang_data:
            lang_data[lc] = {"refs": [], "greedy": [], "beam": []}
        lang_data[lc]["refs"].append(ref)
        lang_data[lc]["greedy"].append(normalize_text(greedy_text))
        lang_data[lc]["beam"].append(normalize_text(beam_text))

    except Exception as e:
        errors += 1
        continue

elapsed = time.time() - t_start

# ── Results ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  RESULTS — {args.split.upper()} SET")
print(f"{'='*70}")
print(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
if errors:
    print(f"  Errors: {errors}")

all_refs, all_greedy, all_beam = [], [], []

print(f"\n  {'Language':<15} {'N':>5} {'Greedy WER':>12} {'Beam+LM WER':>13} {'Δ':>8}")
print(f"  {'─'*53}")

for lc in ["hi", "mr", "en"]:
    if lc not in lang_data:
        continue
    d = lang_data[lc]
    n = len(d["refs"])
    g_wer = compute_corpus_wer(d["refs"], d["greedy"]) * 100
    b_wer = compute_corpus_wer(d["refs"], d["beam"]) * 100
    diff = g_wer - b_wer
    print(f"  {lc.upper():<15} {n:>5} {g_wer:>11.2f}% {b_wer:>12.2f}% {diff:>+7.2f}%")
    all_refs.extend(d["refs"])
    all_greedy.extend(d["greedy"])
    all_beam.extend(d["beam"])

print(f"  {'─'*53}")
n_total = len(all_refs)
g_total = compute_corpus_wer(all_refs, all_greedy) * 100
b_total = compute_corpus_wer(all_refs, all_beam) * 100
diff_total = g_total - b_total
print(f"  {'Overall':<15} {n_total:>5} {g_total:>11.2f}% {b_total:>12.2f}% {diff_total:>+7.2f}%")

print(f"\n  Settings: alpha={args.alpha}, beta={args.beta}, beam_width={args.beam_width}")
print(f"  LM: {'OFF' if args.no_lm else args.lm_path}")
print(f"\n{'='*70}")
