"""
Step 2: Evaluate Semantic Branch — Decode from CTC Head 2 and compute WER
==========================================================================

WHAT THIS DOES:
    Loads the trained encoder + CTC Head 2 (from step1).
    For each Hindi/Marathi test clip:
        Audio → Whisper encoder → CTC Head 2 → logits (T, 257)
        → greedy CTC decode → BPE tokens → text
        → compute WER vs ground truth

    Decoding uses IndicConformer's BPE vocabulary (257 tokens per language).
    Same decode logic as IndicConformer itself uses.

EXPECTED OUTCOME:
    If semantic distillation worked, WER should be:
        - Better than Whisper Small baseline (Hindi 161.81%, Marathi 170.78%)
        - Closer to IndicConformer (Hindi 39.01%, Marathi 44.61%)

Usage:
    python scripts/semantic/step2_semantic_evaluate.py
    python scripts/semantic/step2_semantic_evaluate.py --checkpoint checkpoints/semantic_mse/best_dev.pt
    python scripts/semantic/step2_semantic_evaluate.py --split dev
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate semantic branch WER")
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/semantic_mse/best_dev.pt",
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
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
SPLIT_CSV    = {"dev": ASER_ROOT / "splits" / "asr_dev.csv",
                "test": ASER_ROOT / "splits" / "asr_test.csv"}
RESULTS_DIR  = PROJECT_ROOT / "experiments" / "semantic_mse"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr"}

print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load IndicConformer vocabulary (for decoding BPE tokens)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading IndicConformer vocabulary")
print("=" * 70)

from transformers import AutoModel

print("  Loading ai4bharat/indic-conformer-600m-multilingual for vocab...")
indic_model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True,
)
vocab_hi = indic_model.vocab["hi"]  # 257 tokens
vocab_mr = indic_model.vocab["mr"]  # 257 tokens
BLANK_ID = indic_model.config.BLANK_ID  # 256
print(f"  Hindi vocab: {len(vocab_hi)} tokens")
print(f"  Marathi vocab: {len(vocab_mr)} tokens")
print(f"  BLANK_ID: {BLANK_ID}")

# Free the model — we only need the vocab
del indic_model
import gc
gc.collect()
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load trained model
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 2: Loading trained model")
print("=" * 70)

from transformers import WhisperModel, WhisperFeatureExtractor

# Load Whisper encoder
WHISPER_ID = "openai/whisper-small"
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)

# Load CTC heads
ctc_head_hi = nn.Linear(768, 257).to(DEVICE)
ctc_head_mr = nn.Linear(768, 257).to(DEVICE)

# Load checkpoint
ckpt_path = PROJECT_ROOT / args.checkpoint
print(f"  Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
ctc_head_hi.load_state_dict(checkpoint["ctc_head_hi_state_dict"])
ctc_head_mr.load_state_dict(checkpoint["ctc_head_mr_state_dict"])
print(f"  Epoch: {checkpoint.get('epoch', '?')}")
print(f"  Train loss: {checkpoint.get('train_loss', '?')}")
print(f"  Dev loss: {checkpoint.get('dev_loss', '?')}")

encoder.eval()
ctc_head_hi.eval()
ctc_head_mr.eval()

del whisper_model
gc.collect()
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load test clips
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 3: Loading {args.split} clips")
print("=" * 70)

csv_path = SPLIT_CSV[args.split]
clips = []
skipped = 0

with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = row.get("language", "")
        lang_code = LANG_MAP.get(lang)
        if lang_code is None:
            skipped += 1
            continue

        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ASER_ROOT / audio_path)

        clips.append({
            "audio_path": audio_path,
            "language": lang,
            "lang_code": lang_code,
            "clip_name": os.path.splitext(os.path.basename(audio_path))[0],
            "ground_truth": row.get("que_text", row.get("text", "")),
        })

if args.max_clips and args.max_clips < len(clips):
    clips = clips[:args.max_clips]

hi_count = sum(1 for c in clips if c["lang_code"] == "hi")
mr_count = sum(1 for c in clips if c["lang_code"] == "mr")
print(f"  Total: {len(clips)} clips ({hi_count} Hindi, {mr_count} Marathi)")
print(f"  Skipped: {skipped} English clips")
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Evaluate
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Running evaluation")
print("=" * 70)

try:
    from jiwer import wer as compute_wer
except ImportError:
    print("  ERROR: jiwer not installed. Run: pip install jiwer")
    exit(1)


def greedy_ctc_decode(logits, vocab):
    """
    Greedy CTC decode from logits.
    logits: (T, 257)
    Returns decoded text string.
    """
    log_probs = logits.log_softmax(dim=-1)
    indices = torch.argmax(log_probs, dim=-1)  # (T,)
    collapsed = torch.unique_consecutive(indices)
    tokens = [vocab[idx] for idx in collapsed if idx != BLANK_ID]
    text = ''.join(tokens).replace('▁', ' ').strip()
    return text


def compute_real_frames(num_samples):
    return min(num_samples // 160 // 2, 1500)


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
            wav = wav.squeeze()  # (samples,)

            # Mel spectrogram
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

            # CTC Head
            if clip["lang_code"] == "hi":
                logits = ctc_head_hi(real_features)  # (1, T, 257)
                vocab = vocab_hi
            else:
                logits = ctc_head_mr(real_features)
                vocab = vocab_mr

            # Decode
            predicted = greedy_ctc_decode(logits.squeeze(0), vocab)

            # WER
            gt = clip["ground_truth"]
            if gt.strip() and predicted.strip():
                clip_wer = compute_wer(gt, predicted)
            elif gt.strip() and not predicted.strip():
                clip_wer = 1.0  # all deletions
            else:
                clip_wer = 0.0

            results.append({
                "clip_name": clip["clip_name"],
                "language": clip["language"],
                "lang_code": clip["lang_code"],
                "ground_truth": gt,
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

# Compute WER properly (concatenated, not average of per-clip)
if hi_results:
    hi_refs = [r["ground_truth"] for r in hi_results if r["ground_truth"].strip()]
    hi_hyps = [r["predicted"] for r in hi_results if r["ground_truth"].strip()]
    hi_wer = compute_wer(hi_refs, hi_hyps) * 100
else:
    hi_wer = 0

if mr_results:
    mr_refs = [r["ground_truth"] for r in mr_results if r["ground_truth"].strip()]
    mr_hyps = [r["predicted"] for r in mr_results if r["ground_truth"].strip()]
    mr_wer = compute_wer(mr_refs, mr_hyps) * 100
else:
    mr_wer = 0

all_refs = [r["ground_truth"] for r in results if r["ground_truth"].strip()]
all_hyps = [r["predicted"] for r in results if r["ground_truth"].strip()]
overall_wer = compute_wer(all_refs, all_hyps) * 100

print(f"\n  {'='*50}")
print(f"  {'Language':<12} {'Clips':>8} {'WER':>10}")
print(f"  {'-'*50}")
print(f"  {'Hindi':<12} {len(hi_results):>8} {hi_wer:>9.2f}%")
print(f"  {'Marathi':<12} {len(mr_results):>8} {mr_wer:>9.2f}%")
print(f"  {'-'*50}")
print(f"  {'Overall':<12} {len(results):>8} {overall_wer:>9.2f}%")
print(f"  {'='*50}")

print(f"\n  Comparison:")
print(f"    Whisper Small baseline:   Hindi {161.81:.2f}%  Marathi {170.78:.2f}%")
print(f"    IndicConformer (teacher): Hindi  {39.01:.2f}%  Marathi  {44.61:.2f}%")
print(f"    Our semantic student:     Hindi {hi_wer:.2f}%  Marathi {mr_wer:.2f}%")
print(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# ── Save predictions ─────────────────────────────────────────────────────────
if args.save_predictions:
    # predictions_all.txt
    with open(RESULTS_DIR / "predictions_all.txt", "w", encoding="utf-8") as f:
        f.write(f"# Semantic MSE Student — All Predictions ({args.split} set)\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        f.write(f"# Hindi WER: {hi_wer:.2f}% | Marathi WER: {mr_wer:.2f}% | Overall: {overall_wer:.2f}%\n\n")
        for r in results:
            f.write(f"[{r['clip_name']}] ({r['language']}) WER={r['wer']*100:.1f}%\n")
            f.write(f"  REF: {r['ground_truth']}\n")
            f.write(f"  HYP: {r['predicted']}\n\n")
    print(f"  Saved: {RESULTS_DIR / 'predictions_all.txt'}")

    # predictions_mismatches.txt (WER > 0)
    mismatches = [r for r in results if r["wer"] > 0]
    with open(RESULTS_DIR / "predictions_mismatches.txt", "w", encoding="utf-8") as f:
        f.write(f"# Semantic MSE Student — Mismatches Only ({args.split} set)\n")
        f.write(f"# {len(mismatches)}/{len(results)} clips with WER > 0\n\n")
        for r in sorted(mismatches, key=lambda x: -x["wer"]):
            f.write(f"[{r['clip_name']}] ({r['language']}) WER={r['wer']*100:.1f}%\n")
            f.write(f"  REF: {r['ground_truth']}\n")
            f.write(f"  HYP: {r['predicted']}\n\n")
    print(f"  Saved: {RESULTS_DIR / 'predictions_mismatches.txt'}")

    # results_summary.txt
    with open(RESULTS_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Semantic MSE Student (244M + CTC Head 2)\n")
        f.write(f"# Model: Whisper Small encoder + Linear(768→257) per language\n")
        f.write(f"# Teacher: IndicConformer 600M (MSE on logits)\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        f.write(f"# {args.split} set: {len(results)} clips ({len(hi_results)} Hindi, {len(mr_results)} Marathi)\n\n")
        f.write(f"{'='*50}\n")
        f.write(f"  {'Language':<12} {'Clips':>8} {'WER':>10}\n")
        f.write(f"  {'-'*35}\n")
        f.write(f"  {'Hindi':<12} {len(hi_results):>8} {hi_wer:>9.2f}%\n")
        f.write(f"  {'Marathi':<12} {len(mr_results):>8} {mr_wer:>9.2f}%\n")
        f.write(f"  {'-'*35}\n")
        f.write(f"  {'Overall':<12} {len(results):>8} {overall_wer:>9.2f}%\n")
        f.write(f"{'='*50}\n")
    print(f"  Saved: {RESULTS_DIR / 'results_summary.txt'}")

print(f"\n{'=' * 70}")
