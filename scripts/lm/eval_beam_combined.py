"""
Combined Model: Greedy vs Beam+LM CTC Decoding — Full Test Set
===============================================================
Runs dual-encoder gated fusion model (no baseline, no teacher).
Compares greedy vs beam+LM decode on the full test set.

Usage:
    python scripts/lm/eval_beam_combined.py
    python scripts/lm/eval_beam_combined.py --alpha 0.7 --beta 1.5
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
parser.add_argument("--checkpoint", type=str, default="checkpoints/combined_unfrozen/best_wer_e21.pt")
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
WHISPER_ID = "openai/whisper-small"
SAMPLE_RATE = 16000
MAX_DURATION = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

print("=" * 70)
print("  Combined Model: Greedy vs Beam+LM")
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

# ── GatedFusion (must match training) ────────────────────────────────────
class GatedFusion(nn.Module):
    def __init__(self, dim=768, dropout=0.0):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic_feat, semantic_feat):
        concat = torch.cat([acoustic_feat, semantic_feat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        fused = gate * acoustic_feat + (1 - gate) * semantic_feat
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused

# ── Load combined model ──────────────────────────────────────────────────
print("\nLoading combined model...")
from transformers import WhisperModel, WhisperFeatureExtractor

feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
ckpt_path = PROJECT_ROOT / args.checkpoint
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
print(f"  Epoch {ckpt.get('epoch', '?')}, Dev WER: {ckpt.get('dev_wer', 0):.2f}%")

# Acoustic encoder
whisper_a = WhisperModel.from_pretrained(WHISPER_ID)
acoustic_enc = whisper_a.encoder.to(DEVICE)
acoustic_enc.load_state_dict(ckpt["acoustic_encoder_state_dict"])
acoustic_enc.eval()
del whisper_a

# Semantic encoder
whisper_s = WhisperModel.from_pretrained(WHISPER_ID)
semantic_enc = whisper_s.encoder.to(DEVICE)
semantic_enc.load_state_dict(ckpt["semantic_encoder_state_dict"])
semantic_enc.eval()
del whisper_s

# Fusion + CTC head
fusion = GatedFusion(dim=768, dropout=0.0).to(DEVICE)
fusion.load_state_dict(ckpt["fusion_state_dict"])
fusion.eval()

ctc_head = nn.Linear(768, vocab_size).to(DEVICE)
ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
ctc_head.eval()

del ckpt
print("  Model loaded")

# ── Build beam decoder ───────────────────────────────────────────────────
print("Building beam decoder...")
lm_path = None if args.no_lm else str(PROJECT_ROOT / args.lm_path)
beam_decoder = build_ctc_decoder(
    vocab_path=str(VOCAB_PATH), lm_path=lm_path,
    alpha=args.alpha, beta=args.beta,
)
print(f"  Ready (LM: {'OFF' if args.no_lm else 'ON'})")

# ── Load clips ───────────────────────────────────────────────────────────
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

print(f"  Total: {len(clips)}")
for lc in ["hi", "mr", "en"]:
    print(f"    {lc.upper()}: {sum(1 for c in clips if c['lang_code'] == lc)}")


def compute_real_frames(num_samples):
    return min(num_samples // 160 // 2, 1500)


# ── Evaluate ─────────────────────────────────────────────────────────────
lang_data = {}
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
        mel_gpu = mel.input_features.to(DEVICE)

        with torch.no_grad():
            a_feat = acoustic_enc(mel_gpu).last_hidden_state[:, :real_frames, :]
            s_feat = semantic_enc(mel_gpu).last_hidden_state[:, :real_frames, :]
            fused = fusion(a_feat, s_feat)
            logits = ctc_head(fused)
            logits_slice = logits[0]  # (T, 85)

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
print(f"  RESULTS — COMBINED MODEL — {args.split.upper()} SET")
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
