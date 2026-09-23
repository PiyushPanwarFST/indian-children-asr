"""
End-to-End Dual-Encoder Evaluation
====================================
Evaluate the E2E checkpoint on test set with greedy or beam+LM decoding.

Usage:
  # Greedy:
  python e2e_evaluate.py --checkpoint checkpoints/e2e/best_wer.pt

  # Beam + LM:
  python e2e_evaluate.py --checkpoint checkpoints/e2e/best_wer.pt \
      --beam_size 10 --lm_path models/lm/aser_3gram.bin --lm_alpha 0.5 --lm_beta 1.0
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

parser = argparse.ArgumentParser(description="E2E Dual-Encoder Evaluation")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Checkpoint path (relative to project root)")
parser.add_argument("--split", choices=["dev", "test"], default="test")
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--beam_size", type=int, default=0,
                    help="Beam search width (0 = greedy)")
parser.add_argument("--lm_path", type=str, default=None,
                    help="Path to KenLM language model (.bin)")
parser.add_argument("--lm_alpha", type=float, default=0.5)
parser.add_argument("--lm_beta", type=float, default=1.0)
parser.add_argument("--output_csv", type=str, default=None,
                    help="Save predictions to CSV")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_ROOT = PROJECT_ROOT / "ASER-Dataset"
VOCAB_PATH = ASER_ROOT / "vocab.json"

if args.split == "test":
    EVAL_CSV = ASER_ROOT / "splits" / "asr_test.csv"
else:
    EVAL_CSV = ASER_ROOT / "splits" / "asr_dev.csv"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Vocab ─────────────────────────────────────────────────────────────────────
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)
BLANK_IDX = char_to_idx["<blank>"]
print(f"Vocab: {vocab_size} tokens")

# ── Load data ─────────────────────────────────────────────────────────────────
clips = []
with open(EVAL_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = row.get("language", "")
        lang_code = LANG_MAP.get(lang)
        if lang_code is None:
            continue
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ASER_ROOT / audio_path)
        gt = row.get("transcript", row.get("que_text", row.get("text", "")))
        clips.append({
            "audio_path": audio_path,
            "language": lang,
            "lang_code": lang_code,
            "ground_truth": gt,
        })
print(f"Loaded {len(clips)} {args.split} clips")

# ── Load model ────────────────────────────────────────────────────────────────
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

# GatedFusion (must match training)
class GatedFusion(nn.Module):
    def __init__(self, dim=768, dropout=0.1):
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
        return fused, gate

# Initialize architectures
encoder_a = WhisperModel.from_pretrained(WHISPER_ID).encoder.to(DEVICE)
encoder_b = WhisperModel.from_pretrained(WHISPER_ID).encoder.to(DEVICE)
fusion = GatedFusion(dim=768, dropout=0.0).to(DEVICE)  # No dropout at eval
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# Load checkpoint
ckpt_path = PROJECT_ROOT / args.checkpoint
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
encoder_a.load_state_dict(ckpt["encoder_a_state_dict"])
encoder_b.load_state_dict(ckpt["encoder_b_state_dict"])
fusion.load_state_dict(ckpt["fusion_state_dict"], strict=False)
ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
ep = ckpt.get("epoch", "?")
train_wer = ckpt.get("dev_wer", "?")
print(f"  Epoch {ep}, train dev WER: {train_wer}")
del ckpt

encoder_a.eval()
encoder_b.eval()
fusion.eval()
ctc_head.eval()

# ── Decode functions ──────────────────────────────────────────────────────────
def ctc_greedy_decode(logits, idx_to_char):
    indices = torch.argmax(logits, dim=-1)
    collapsed = torch.unique_consecutive(indices)
    chars = []
    for idx in collapsed:
        idx = idx.item()
        if idx == BLANK_IDX:
            continue
        token = idx_to_char.get(idx, "")
        if token == "<space>":
            chars.append(" ")
        elif token in ("<blank>", "<unk>"):
            continue
        else:
            chars.append(token)
    return "".join(chars).strip()


# Beam search + LM (optional)
beam_decoder = None
if args.beam_size > 0 and args.lm_path:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
        from ctc_decode import build_beam_decoder
        beam_decoder = build_beam_decoder(
            vocab_path=str(VOCAB_PATH),
            lm_path=args.lm_path,
            beam_size=args.beam_size,
            alpha=args.lm_alpha,
            beta=args.lm_beta,
        )
        print(f"Beam decoder: beam={args.beam_size}, alpha={args.lm_alpha}, beta={args.lm_beta}")
    except Exception as e:
        print(f"WARNING: Could not load beam decoder: {e}")
        print("  Falling back to greedy decode")


def compute_real_frames(num_samples):
    return min(num_samples // 160 // 2, 1500)


# ── Evaluate ──────────────────────────────────────────────────────────────────
print(f"\nEvaluating on {args.split} set ({len(clips)} clips)...")

all_refs, all_hyps = [], []
lang_refs = {"hi": [], "mr": [], "en": []}
lang_hyps = {"hi": [], "mr": [], "en": []}
results = []

with torch.no_grad():
    for clip in tqdm(clips, desc="Eval", bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            wav, sr = torchaudio.load(clip["audio_path"])
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze()
            if args.max_audio_sec and len(wav) > int(args.max_audio_sec * SAMPLE_RATE):
                wav = wav[:int(args.max_audio_sec * SAMPLE_RATE)]
            if len(wav) < 8000:
                continue

            mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            mel_gpu = mel.input_features.to(DEVICE)

            a_out = encoder_a(mel_gpu).last_hidden_state
            b_out = encoder_b(mel_gpu).last_hidden_state
            real_frames = compute_real_frames(len(wav))
            a_feat = a_out[:, :real_frames, :]
            b_feat = b_out[:, :real_frames, :]

            fused, gate = fusion(a_feat, b_feat)
            logits = ctc_head(fused)  # (1, T, 85)

            # Decode
            if beam_decoder is not None:
                log_probs = logits.log_softmax(dim=-1).squeeze(0).cpu().numpy()
                predicted = beam_decoder(log_probs)
            else:
                predicted = ctc_greedy_decode(logits.squeeze(0), idx_to_char)

            ref = normalize_text(clip["ground_truth"])
            if ref:
                all_refs.append(ref)
                all_hyps.append(predicted)
                lc = clip["lang_code"]
                if lc in lang_refs:
                    lang_refs[lc].append(ref)
                    lang_hyps[lc].append(predicted)

                results.append({
                    "language": clip["language"],
                    "reference": ref,
                    "hypothesis": predicted,
                    "gate_mean": f"{gate.mean().item():.3f}",
                })

            if DEVICE == "cuda":
                del a_out, b_out, a_feat, b_feat, fused, logits
                torch.cuda.empty_cache()

        except Exception as e:
            continue

# ── Results ───────────────────────────────────────────────────────────────────
overall_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0
decode_mode = f"beam={args.beam_size}+LM" if beam_decoder else "greedy"

print(f"\n{'='*60}")
print(f"  E2E Dual-Encoder Results ({args.split}, {decode_mode})")
print(f"{'='*60}")
print(f"  Overall WER: {overall_wer:.2f}% ({len(all_refs)} clips)")

for lc, lname in [("hi", "Hindi"), ("mr", "Marathi"), ("en", "English")]:
    if lang_refs[lc]:
        wer = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
        print(f"  {lname}: {wer:.2f}% ({len(lang_refs[lc])} clips)")

print(f"{'='*60}")

# Save CSV
if args.output_csv:
    out_path = PROJECT_ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "reference", "hypothesis", "gate_mean"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {out_path}")
