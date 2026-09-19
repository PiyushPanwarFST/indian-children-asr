"""
Step 3: Evaluate MuRIL Cross-Modal Model on Test Set
====================================================

WHAT THIS DOES:
    Loads the trained encoder + CTC Head 1 (from step2_muril_training.py).
    For each test clip (Hindi, Marathi, English):
        Audio -> Whisper encoder -> CTC Head 1 -> logits (T, 85)
        -> greedy CTC decode -> character sequence -> text
        -> compute WER vs ground truth

    The encoder was trained with MuRIL cross-modal knowledge distillation.
    CTC Head 1 (Linear(768->85), our character vocab) is used for decoding,
    same as all other branch evaluations for fair comparison.

EXPECTED OUTCOME:
    If MuRIL cross-modal distillation helped, WER should improve over
    the baseline semantic branches (MSE, sequential CTC, joint).

Usage:
    python step3_muril_evaluate.py
    python step3_muril_evaluate.py --checkpoint checkpoints/semantic_muril/best_wer.pt
    python step3_muril_evaluate.py --split dev
    python step3_muril_evaluate.py --max_clips 50
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate MuRIL cross-modal model WER")
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/semantic_muril/best_wer.pt",
                    help="Path to trained checkpoint")
parser.add_argument("--split", type=str, default="test",
                    choices=["dev", "test"],
                    help="Which split to evaluate on")
parser.add_argument("--max_clips", type=int, default=None,
                    help="Max clips to evaluate (for quick tests)")
parser.add_argument("--save_predictions", action="store_true", default=True,
                    help="Save predictions to results/ folder")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
SPLIT_CSV    = {"dev": ASER_ROOT / "splits" / "asr_dev.csv",
                "test": ASER_ROOT / "splits" / "asr_test.csv"}
RESULTS_DIR  = PROJECT_ROOT / "results" / "semantic_branch" / "05_muril_cross_modal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_ID  = "openai/whisper-small"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

# Add project root for utils
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
print(f"Split: {args.split}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load vocabulary (vocab.json, 85 tokens)
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


# ── Utility functions ────────────────────────────────────────────────────────

def load_audio(path, max_sec=None):
    """
    Load audio file and resample to 16kHz mono.

    Args:
        path: Path to the audio file.
        max_sec: Optional max duration in seconds (truncate if longer).

    Returns:
        wav: 1D tensor of audio samples at 16kHz.
    """
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()
    if max_sec and len(wav) > int(max_sec * SAMPLE_RATE):
        wav = wav[:int(max_sec * SAMPLE_RATE)]
    return wav


def compute_real_frames(num_samples):
    """
    Compute the number of real encoder output frames for a given audio length.
    Whisper: mel stride=160, conv stride=2 -> frames = samples // 320.
    Capped at 1500 (Whisper max).

    Args:
        num_samples: Number of audio samples at 16kHz.

    Returns:
        int: Number of real encoder frames.
    """
    return min(num_samples // 160 // 2, 1500)


def text_to_indices(text, char_to_idx):
    """
    Convert ground truth text to character indices.
    Handles multi-codepoint Hindi/Marathi characters (try 2-char match before 1-char).

    Args:
        text: Ground truth text string.
        char_to_idx: Dictionary mapping characters to indices.

    Returns:
        List of integer indices.
    """
    indices = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            if "<space>" in char_to_idx:
                indices.append(char_to_idx["<space>"])
            i += 1
            continue
        # Try 2-char match first (for multi-codepoint characters)
        if i + 1 < len(text) and text[i:i+2] in char_to_idx:
            indices.append(char_to_idx[text[i:i+2]])
            i += 2
        elif text[i] in char_to_idx:
            indices.append(char_to_idx[text[i]])
            i += 1
        else:
            # Unknown character — use <unk> if available, else skip
            if "<unk>" in char_to_idx:
                indices.append(char_to_idx["<unk>"])
            i += 1
    return indices


def ctc_greedy_decode(logits, idx_to_char):
    """
    Greedy CTC decode from logits to text string.
    Collapses repeated tokens and removes blanks.

    Args:
        logits: Tensor of shape (T, vocab_size) — raw logits or log-probs.
        idx_to_char: Dictionary mapping indices to characters.

    Returns:
        Decoded text string.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load trained model (encoder + CTC Head 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading trained model")
print("=" * 70)

from transformers import WhisperModel, WhisperFeatureExtractor

feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

# Load Whisper encoder
print(f"\n  Loading Whisper Small encoder...")
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)
del whisper_model

# CTC Head 1 (our char vocab, 85 tokens)
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# Load checkpoint
ckpt_path = PROJECT_ROOT / args.checkpoint
print(f"  Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

# Load encoder weights — handle possible 'encoder.' prefix in keys
if "encoder_state_dict" in checkpoint:
    enc_state = checkpoint["encoder_state_dict"]
    # Check if keys have 'encoder.' prefix and strip it
    first_key = next(iter(enc_state))
    if first_key.startswith("encoder."):
        enc_state = {k[len("encoder."):]: v for k, v in enc_state.items()}
    encoder.load_state_dict(enc_state)
elif "model_state_dict" in checkpoint:
    state = {}
    for k, v in checkpoint["model_state_dict"].items():
        if k.startswith("encoder."):
            state[k[len("encoder."):]] = v
    encoder.load_state_dict(state)
else:
    print(f"  ERROR: Unknown checkpoint format. Keys: {list(checkpoint.keys())[:10]}")
    sys.exit(1)

# Load CTC Head 1 weights
if "ctc_head_state_dict" in checkpoint:
    ctc_head.load_state_dict(checkpoint["ctc_head_state_dict"])
elif "ctc_head_char_state_dict" in checkpoint:
    ctc_head.load_state_dict(checkpoint["ctc_head_char_state_dict"])
else:
    print(f"  ERROR: No CTC head weights found. Keys: {list(checkpoint.keys())[:10]}")
    sys.exit(1)

print(f"  Epoch: {checkpoint.get('epoch', '?')}")
print(f"  Dev WER: {checkpoint.get('dev_wer', '?')}")
if "lang_wers" in checkpoint:
    print(f"  Lang WERs: {checkpoint['lang_wers']}")
if "args" in checkpoint:
    ckpt_args = checkpoint["args"]
    if hasattr(ckpt_args, "__dict__"):
        print(f"  Training args: lr={getattr(ckpt_args, 'lr', '?')}, "
              f"epochs={getattr(ckpt_args, 'epochs', '?')}")

encoder.eval()
ctc_head.eval()

del checkpoint
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load test clips from CSV
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 3: Loading {args.split} clips")
print("=" * 70)

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
# STEP 4: Evaluate (loop through clips, decode, compute WER)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 4: Evaluating on {args.split} set ({len(clips)} clips)")
print("=" * 70)

results = []
errors = 0
start_time = time.time()

with torch.no_grad():
    for i, clip in enumerate(clips):
        try:
            # Load audio
            wav = load_audio(clip["audio_path"])
            if len(wav) < 8000:  # Skip clips shorter than 0.5s
                continue

            # Mel spectrogram (NO SpecAugment during eval)
            mel = feat_extractor(
                wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            input_features = mel.input_features.to(DEVICE)

            # Encoder forward -> (1, T, 768)
            enc_out = encoder(input_features)
            features = enc_out.last_hidden_state  # (1, 1500, 768)

            # Slice to real frames
            real_frames = compute_real_frames(len(wav))
            real_features = features[:, :real_frames, :]

            # CTC Head 1 -> logits (1, T, 85)
            logits = ctc_head(real_features)

            # Greedy CTC decode -> text
            predicted = ctc_greedy_decode(logits.squeeze(0), idx_to_char)

            # Normalize and compute WER
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

            # GPU memory cleanup
            if DEVICE == "cuda":
                del enc_out, features, logits, real_features
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            print(f"  ERROR [{i+1}/{len(clips)}] {clip['clip_name']}: {e}")
            continue

elapsed = time.time() - start_time

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Results summary (overall + per-language + comparison table)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 5: Results -- MuRIL Cross-Modal ({args.split} set)")
print("=" * 70)

hi_results = [r for r in results if r["lang_code"] == "hi"]
mr_results = [r for r in results if r["lang_code"] == "mr"]
en_results = [r for r in results if r["lang_code"] == "en"]


def compute_lang_wer(lang_results):
    """
    Compute corpus-level WER for a set of results from one language.

    Args:
        lang_results: List of result dicts with 'ground_truth' and 'predicted' keys.

    Returns:
        float: WER as a percentage (0-100+), or None if no valid clips.
    """
    refs = [normalize_text(r["ground_truth"]) for r in lang_results
            if normalize_text(r["ground_truth"])]
    hyps = [normalize_text(r["predicted"]) for r in lang_results
            if normalize_text(r["ground_truth"])]
    if refs:
        return compute_corpus_wer(refs, hyps) * 100
    return None


hi_wer = compute_lang_wer(hi_results)
mr_wer = compute_lang_wer(mr_results)
en_wer = compute_lang_wer(en_results)

all_refs = [normalize_text(r["ground_truth"]) for r in results
            if normalize_text(r["ground_truth"])]
all_hyps = [normalize_text(r["predicted"]) for r in results
            if normalize_text(r["ground_truth"])]
overall_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 0.0

# ── Per-language table ────────────────────────────────────────────────────────
print(f"\n  {'='*60}")
print(f"  {'Language':<12} {'Clips':>8} {'WER':>10}")
print(f"  {'-'*60}")
if hi_wer is not None:
    print(f"  {'Hindi':<12} {len(hi_results):>8} {hi_wer:>9.2f}%")
else:
    print(f"  {'Hindi':<12} {len(hi_results):>8} {'N/A':>10}")
if mr_wer is not None:
    print(f"  {'Marathi':<12} {len(mr_results):>8} {mr_wer:>9.2f}%")
else:
    print(f"  {'Marathi':<12} {len(mr_results):>8} {'N/A':>10}")
if en_wer is not None:
    print(f"  {'English':<12} {len(en_results):>8} {en_wer:>9.2f}%")
else:
    print(f"  {'English':<12} {len(en_results):>8} {'N/A':>10}")
print(f"  {'-'*60}")
print(f"  {'Overall':<12} {len(results):>8} {overall_wer:>9.2f}%")
print(f"  {'='*60}")
print(f"  Errors: {errors} | Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# ── Format WER strings for comparison table ──────────────────────────────────
hi_str = f"{hi_wer:.2f}%" if hi_wer is not None else "N/A"
mr_str = f"{mr_wer:.2f}%" if mr_wer is not None else "N/A"
en_str = f"{en_wer:.2f}%" if en_wer is not None else "N/A"

# ── Comparison table ─────────────────────────────────────────────────────────
print(f"\n  Comparison with other systems ({args.split} set):")
print(f"  {'System':<35}| {'Hindi':>8} | {'Marathi':>9} | {'English':>9} | {'Overall':>9}")
print(f"  {'─'*35}+{'─'*10}+{'─'*11}+{'─'*11}+{'─'*10}")
print(f"  {'Acoustic (joint_best_wer)':<35}| {'15.88%':>8} | {'29.96%':>9} | {'21.63%':>9} | {'19.96%':>9}")
print(f"  {'Semantic MSE (IndicConformer)':<35}| {'47.86%':>8} | {'---':>9} | {'---':>9} | {'47.86%':>9}")
print(f"  {'Semantic CTC (sequential)':<35}| {'37.95%':>8} | {'74.36%':>9} | {'56.82%':>9} | {'49.02%':>9}")
print(f"  {'Semantic Joint (MSE+CTC)':<35}| {'---':>8} | {'---':>9} | {'---':>9} | {'52.45%':>9}")
overall_str = f"{overall_wer:.2f}%"
print(f"  {'MuRIL Cross-Modal (this run)':<35}| {hi_str:>8} | {mr_str:>9} | {en_str:>9} | {overall_str:>9}")
print(f"  {'Combined (gated fusion)':<35}| {'10.55%':>8} | {'25.27%':>9} | {'17.04%':>9} | {'14.72%':>9}")

# ── Sample predictions ───────────────────────────────────────────────────────
print(f"\n  Sample predictions:")
n_show = min(5, len(results))
for r in results[:n_show]:
    print(f"    [{r['language']}] REF: {r['ground_truth']}")
    print(f"    [{r['language']}] HYP: {r['predicted']}")
    print()

# ── Save predictions ─────────────────────────────────────────────────────────
if args.save_predictions and results:
    # test_predictions_all.txt
    pred_file = RESULTS_DIR / f"{args.split}_predictions_all.txt"
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write(f"=== MuRIL Cross-Modal: {args.split.capitalize()} Predictions ===\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Overall WER: {overall_wer:.2f}%\n")
        f.write(f"Per-language: Hindi {hi_str}, Marathi {mr_str}, English {en_str}\n\n")
        for j, r in enumerate(results, 1):
            clip_wer_pct = r["wer"] * 100
            f.write(f"[{j}] LANG={r['lang_code']} WER={clip_wer_pct:.2f}%\n")
            f.write(f"  GT:   {r['ground_truth']}\n")
            f.write(f"  PRED: {r['predicted']}\n\n")
    print(f"\n  Saved: {pred_file}")

    # test_results.txt (summary stats)
    summary_file = RESULTS_DIR / f"{args.split}_results.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# MuRIL Cross-Modal Evaluation Results\n")
        f.write(f"# Model: Whisper Small encoder + Linear(768->{vocab_size}) CTC Head 1\n")
        f.write(f"# Training: MuRIL cross-modal knowledge distillation\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        f.write(f"# {args.split} set: {len(results)} clips "
                f"({len(hi_results)} Hindi, {len(mr_results)} Marathi, {len(en_results)} English)\n")
        f.write(f"# Errors: {errors}\n")
        f.write(f"# Eval time: {elapsed:.0f}s\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Language':<12} {'Clips':>8} {'WER':>10}\n")
        f.write(f"  {'-'*40}\n")
        if hi_wer is not None:
            f.write(f"  {'Hindi':<12} {len(hi_results):>8} {hi_wer:>9.2f}%\n")
        if mr_wer is not None:
            f.write(f"  {'Marathi':<12} {len(mr_results):>8} {mr_wer:>9.2f}%\n")
        if en_wer is not None:
            f.write(f"  {'English':<12} {len(en_results):>8} {en_wer:>9.2f}%\n")
        f.write(f"  {'-'*40}\n")
        f.write(f"  {'Overall':<12} {len(results):>8} {overall_wer:>9.2f}%\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Comparison:\n")
        f.write(f"  {'System':<35}| {'Hindi':>8} | {'Marathi':>9} | {'English':>9} | {'Overall':>9}\n")
        f.write(f"  {'─'*35}+{'─'*10}+{'─'*11}+{'─'*11}+{'─'*10}\n")
        f.write(f"  {'Acoustic (joint_best_wer)':<35}| {'15.88%':>8} | {'29.96%':>9} | {'21.63%':>9} | {'19.96%':>9}\n")
        f.write(f"  {'Semantic MSE (IndicConformer)':<35}| {'47.86%':>8} | {'---':>9} | {'---':>9} | {'47.86%':>9}\n")
        f.write(f"  {'Semantic CTC (sequential)':<35}| {'37.95%':>8} | {'74.36%':>9} | {'56.82%':>9} | {'49.02%':>9}\n")
        f.write(f"  {'Semantic Joint (MSE+CTC)':<35}| {'---':>8} | {'---':>9} | {'---':>9} | {'52.45%':>9}\n")
        f.write(f"  {'MuRIL Cross-Modal (this run)':<35}| {hi_str:>8} | {mr_str:>9} | {en_str:>9} | {overall_wer:>9.2f}%\n")
        f.write(f"  {'Combined (gated fusion)':<35}| {'10.55%':>8} | {'25.27%':>9} | {'17.04%':>9} | {'14.72%':>9}\n")
    print(f"  Saved: {summary_file}")

    # Mismatches
    mismatches = [r for r in results if r["wer"] > 0]
    mismatch_file = RESULTS_DIR / f"{args.split}_predictions_mismatches.txt"
    with open(mismatch_file, "w", encoding="utf-8") as f:
        f.write(f"# MuRIL Cross-Modal -- Mismatches Only ({args.split} set)\n")
        f.write(f"# Overall WER: {overall_wer:.2f}%\n")
        f.write(f"# Total mismatches: {len(mismatches)}/{len(results)}\n\n")
        for r in sorted(mismatches, key=lambda x: -x["wer"]):
            clip_wer_pct = r["wer"] * 100
            f.write(f"[{r['language']}] {r['clip_name']} WER={clip_wer_pct:.1f}%\n")
            f.write(f"  REF: {r['ground_truth']}\n")
            f.write(f"  HYP: {r['predicted']}\n\n")
    print(f"  Saved: {mismatch_file}")

print(f"\n{'=' * 70}")
