"""
Step 4: Evaluate Semantic Joint Model — Decode from CTC Head 1 and compute WER
===============================================================================

WHAT THIS DOES:
    Loads the jointly trained encoder + CTC Head 1 (768→85, our char vocab).
    For each test clip (Hindi, Marathi, AND English):
        Audio → Whisper encoder → CTC Head 1 → logits (T, 85)
        → greedy CTC decode → character sequence → text
        → compute WER vs ground truth

    Unlike step2 which decoded from CTC Head 2 (teacher's BPE vocab, 257 tokens),
    this decodes from CTC Head 1 (our char vocab, 85 tokens).
    CTC Head 1 supports ALL languages (Hindi, Marathi, English).

EXPECTED OUTCOME:
    Hindi:   < 39.72% (step2 via Head 2)
    Marathi: < 68.11% (step2 via Head 2)
    English: NOW TESTABLE (Head 1 has English chars, Head 2 didn't)

Usage:
    python scripts/semantic/step4_semantic_joint_evaluate.py
    python scripts/semantic/step4_semantic_joint_evaluate.py --checkpoint checkpoints/semantic_joint/best_dev.pt
    python scripts/semantic/step4_semantic_joint_evaluate.py --split dev
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate semantic joint model WER")
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/semantic_joint/best_dev.pt",
                    help="Path to trained checkpoint")
parser.add_argument("--split", type=str, default="test",
                    choices=["dev", "test"],
                    help="Which split to evaluate on")
parser.add_argument("--max_clips", type=int, default=None,
                    help="Max clips to evaluate (for quick tests)")
parser.add_argument("--save_predictions", action="store_true", default=True,
                    help="Save predictions to experiments/ folder")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
SPLIT_CSV    = {"dev": ASER_ROOT / "splits" / "asr_dev.csv",
                "test": ASER_ROOT / "splits" / "asr_test.csv"}
RESULTS_DIR  = PROJECT_ROOT / "results" / "semantic_branch" / "04_joint_training"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add project root for utils
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load vocabulary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading character vocabulary")
print("=" * 70)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)
BLANK_IDX = char_to_idx["<blank>"]  # 0
print(f"  Vocab: {vocab_size} tokens, blank={BLANK_IDX}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load trained model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading trained model")
print("=" * 70)

from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)

# CTC Head 1 (our char vocab)
ctc_head_char = nn.Linear(768, vocab_size).to(DEVICE)

# Load checkpoint
ckpt_path = PROJECT_ROOT / args.checkpoint
print(f"  Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
ctc_head_char.load_state_dict(checkpoint["ctc_head_char_state_dict"])
print(f"  Epoch: {checkpoint.get('epoch', '?')}")
print(f"  Dev WER: {checkpoint.get('dev_wer', '?')}")
print(f"  Train MSE: {checkpoint.get('train_mse', '?')}")
print(f"  Train CTC: {checkpoint.get('train_ctc', '?')}")
print(f"  Alpha: {checkpoint.get('alpha', '?')}")

encoder.eval()
ctc_head_char.eval()

import gc
del whisper_model, checkpoint
gc.collect()
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load test clips (ALL languages — Head 1 supports Hi+Mr+En)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 3: Loading {args.split} clips")
print("=" * 70)

LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
csv_path = SPLIT_CSV[args.split]
clips = []

with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = row.get("language", "")
        lang_code = LANG_MAP.get(lang)
        if lang_code is None:
            continue

        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ASER_ROOT / audio_path)

        child_id = row.get("child_id", "")
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        clip_uid = f"{child_id}_{basename}" if child_id else basename

        gt = row.get("transcript", row.get("que_text", row.get("text", "")))

        clips.append({
            "audio_path": audio_path,
            "language": lang,
            "lang_code": lang_code,
            "clip_name": clip_uid,
            "ground_truth": gt,
        })

if args.max_clips and args.max_clips < len(clips):
    clips = clips[:args.max_clips]

hi_count = sum(1 for c in clips if c["lang_code"] == "hi")
mr_count = sum(1 for c in clips if c["lang_code"] == "mr")
en_count = sum(1 for c in clips if c["lang_code"] == "en")
print(f"  Total: {len(clips)} clips ({hi_count} Hindi, {mr_count} Marathi, {en_count} English)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Evaluate
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Running evaluation")
print("=" * 70)


def compute_real_frames(num_samples):
    return min(num_samples // 160 // 2, 1500)


def ctc_greedy_decode(logits, idx_to_char):
    """Greedy CTC decode from logits (T, vocab_size) → text."""
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


results = []
start_time = time.time()

with torch.no_grad():
    for i, clip in enumerate(clips):
        try:
            # Load audio
            wav, sr = torchaudio.load(clip["audio_path"])
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze()

            # Mel
            mel = feat_extractor(
                wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            input_features = mel.input_features.to(DEVICE)

            # Encoder
            enc_out = encoder(input_features)
            features = enc_out.last_hidden_state  # (1, 1500, 768)

            # Slice real frames
            real_frames = compute_real_frames(len(wav))
            real_features = features[:, :real_frames, :]

            # CTC Head 1
            char_logits = ctc_head_char(real_features)  # (1, T, 85)
            predicted = ctc_greedy_decode(char_logits.squeeze(0), idx_to_char)

            # WER
            gt = normalize_text(clip["ground_truth"])
            pred_norm = normalize_text(predicted)

            if gt:
                clip_wer = compute_corpus_wer([gt], [pred_norm])
            else:
                clip_wer = 0.0

            results.append({
                "clip_name": clip["clip_name"],
                "language": clip["language"],
                "lang_code": clip["lang_code"],
                "ground_truth": clip["ground_truth"],
                "predicted": predicted,
                "wer": clip_wer,
            })

            if (i + 1) % 100 == 0 or i < 5:
                print(f"  [{i+1}/{len(clips)}] {clip['clip_name']} "
                      f"lang={clip['lang_code']} WER={clip_wer*100:.1f}%")

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {clip['clip_name']}: {e}")
            continue

elapsed = time.time() - start_time

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Results")
print("=" * 70)

hi_results = [r for r in results if r["lang_code"] == "hi"]
mr_results = [r for r in results if r["lang_code"] == "mr"]
en_results = [r for r in results if r["lang_code"] == "en"]


def compute_lang_wer(lang_results):
    refs = [normalize_text(r["ground_truth"]) for r in lang_results if normalize_text(r["ground_truth"])]
    hyps = [normalize_text(r["predicted"]) for r in lang_results if normalize_text(r["ground_truth"])]
    if refs:
        return compute_corpus_wer(refs, hyps) * 100
    return 0.0


hi_wer = compute_lang_wer(hi_results)
mr_wer = compute_lang_wer(mr_results)
en_wer = compute_lang_wer(en_results)

all_refs = [normalize_text(r["ground_truth"]) for r in results if normalize_text(r["ground_truth"])]
all_hyps = [normalize_text(r["predicted"]) for r in results if normalize_text(r["ground_truth"])]
overall_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 0.0

print(f"\n  {'='*60}")
print(f"  {'Language':<12} {'Clips':>8} {'WER':>10}")
print(f"  {'-'*60}")
print(f"  {'Hindi':<12} {len(hi_results):>8} {hi_wer:>9.2f}%")
print(f"  {'Marathi':<12} {len(mr_results):>8} {mr_wer:>9.2f}%")
print(f"  {'English':<12} {len(en_results):>8} {en_wer:>9.2f}%")
print(f"  {'-'*60}")
print(f"  {'Overall':<12} {len(results):>8} {overall_wer:>9.2f}%")
print(f"  {'='*60}")

print(f"\n  Comparison:")
print(f"    Whisper Small baseline:       Hindi {161.81:.2f}%  Marathi {170.78:.2f}%  English {47.42:.2f}%")
print(f"    IndicConformer (teacher):     Hindi  {39.01:.2f}%  Marathi  {44.61:.2f}%  English     N/A")
print(f"    Step2 semantic (Head 2 BPE):  Hindi {39.72:.2f}%  Marathi {68.11:.2f}%  English     N/A")
print(f"    Step4 joint (Head 1 char):    Hindi {hi_wer:.2f}%  Marathi {mr_wer:.2f}%  English {en_wer:.2f}%")
print(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# ── Save predictions ─────────────────────────────────────────────────────────
if args.save_predictions:
    # results_summary.txt
    with open(RESULTS_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Semantic Joint Student (244M + CTC Head 1)\n")
        f.write(f"# Model: Whisper Small encoder + Linear(768→{vocab_size})\n")
        f.write(f"# Training: MSE (IndicConformer logits) + CTC (ground truth)\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        f.write(f"# {args.split} set: {len(results)} clips "
                f"({len(hi_results)} Hindi, {len(mr_results)} Marathi, {len(en_results)} English)\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Language':<12} {'Clips':>8} {'WER':>10}\n")
        f.write(f"  {'-'*40}\n")
        f.write(f"  {'Hindi':<12} {len(hi_results):>8} {hi_wer:>9.2f}%\n")
        f.write(f"  {'Marathi':<12} {len(mr_results):>8} {mr_wer:>9.2f}%\n")
        f.write(f"  {'English':<12} {len(en_results):>8} {en_wer:>9.2f}%\n")
        f.write(f"  {'-'*40}\n")
        f.write(f"  {'Overall':<12} {len(results):>8} {overall_wer:>9.2f}%\n")
        f.write(f"{'='*60}\n")
    print(f"  Saved: {RESULTS_DIR / 'results_summary.txt'}")

    # predictions_all.txt
    with open(RESULTS_DIR / "predictions_all.txt", "w", encoding="utf-8") as f:
        f.write(f"# Semantic Joint Student — All Predictions ({args.split} set)\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        f.write(f"# Hindi WER: {hi_wer:.2f}% | Marathi WER: {mr_wer:.2f}% | "
                f"English WER: {en_wer:.2f}% | Overall: {overall_wer:.2f}%\n\n")
        for r in results:
            f.write(f"[{r['clip_name']}] ({r['language']}) WER={r['wer']*100:.1f}%\n")
            f.write(f"  REF: {r['ground_truth']}\n")
            f.write(f"  HYP: {r['predicted']}\n\n")
    print(f"  Saved: {RESULTS_DIR / 'predictions_all.txt'}")

    # predictions_mismatches.txt
    mismatches = [r for r in results if r["wer"] > 0]
    with open(RESULTS_DIR / "predictions_mismatches.txt", "w", encoding="utf-8") as f:
        f.write(f"# Semantic Joint Student — Mismatches Only ({args.split} set)\n")
        f.write(f"# {len(mismatches)}/{len(results)} clips with WER > 0\n\n")
        for r in sorted(mismatches, key=lambda x: -x["wer"]):
            f.write(f"[{r['clip_name']}] ({r['language']}) WER={r['wer']*100:.1f}%\n")
            f.write(f"  REF: {r['ground_truth']}\n")
            f.write(f"  HYP: {r['predicted']}\n\n")
    print(f"  Saved: {RESULTS_DIR / 'predictions_mismatches.txt'}")

print(f"\n{'=' * 70}")
