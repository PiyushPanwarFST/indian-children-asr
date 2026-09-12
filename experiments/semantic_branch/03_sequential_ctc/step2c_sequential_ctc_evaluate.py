"""
Step 2c: Evaluate Semantic Sequential CTC on Test Set
=====================================================

WHAT THIS DOES:
    Evaluates the sequential CTC model (frozen semantic encoder + trained CTC head)
    on the test set to get real test WER numbers.

    Architecture: Audio → Mel → Frozen Semantic Encoder (768) → CTC Head (768→85) → Decode

    The encoder was trained with IndicConformer MSE distillation (step1).
    The CTC head was trained on top of the frozen encoder (step2b).

Prerequisites:
    - Semantic encoder: checkpoints/semantic_mse/best_dev.pt
    - CTC head: checkpoints/semantic_ctc/best_wer.pt
    - Vocabulary: ASER-Dataset/vocab.json (85 tokens)
    - Splits: ASER-Dataset/splits/asr_test.csv

Usage:
    # Evaluate on test set
    python step2c_sequential_ctc_evaluate.py --split test

    # Evaluate on dev set
    python step2c_sequential_ctc_evaluate.py --split dev

    # Custom checkpoint paths
    python step2c_sequential_ctc_evaluate.py --split test \
        --encoder_ckpt checkpoints/semantic_mse/best_dev.pt \
        --ctc_ckpt checkpoints/semantic_ctc/best_wer.pt
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
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate semantic sequential CTC on test set")
parser.add_argument("--split", type=str, default="test", choices=["dev", "test"],
                    help="Which split to evaluate on")
parser.add_argument("--encoder_ckpt", type=str,
                    default="checkpoints/semantic_mse/best_dev.pt",
                    help="Semantic encoder checkpoint (MSE-trained)")
parser.add_argument("--ctc_ckpt", type=str,
                    default="checkpoints/semantic_ctc/best_wer.pt",
                    help="CTC head checkpoint (sequential CTC-trained)")
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--save_predictions", action="store_true", default=True)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
SPLIT_CSV    = {"dev": ASER_ROOT / "splits" / "asr_dev.csv",
                "test": ASER_ROOT / "splits" / "asr_test.csv"}
RESULTS_DIR  = PROJECT_ROOT / "results" / "semantic_branch" / "03_sequential_ctc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
print(f"Split: {args.split}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load vocabulary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading vocabulary")
print("=" * 70)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)
BLANK_IDX = char_to_idx["<blank>"]
print(f"  Vocab: {vocab_size} tokens")


def ctc_greedy_decode(logits, idx_to_char):
    """Greedy CTC decode from logits (T, vocab_size) → text string."""
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
# STEP 2: Load clips
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 2: Loading {args.split} clips")
print("=" * 70)

clips = []
csv_path = SPLIT_CSV[args.split]
with open(csv_path, encoding="utf-8") as f:
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
            "clip_name": os.path.splitext(os.path.basename(audio_path))[0],
        })

hi_n = sum(1 for c in clips if c["lang_code"] == "hi")
mr_n = sum(1 for c in clips if c["lang_code"] == "mr")
en_n = sum(1 for c in clips if c["lang_code"] == "en")
print(f"  Total: {len(clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Loading semantic encoder + CTC head")
print("=" * 70)

from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

# Load encoder
print(f"\n  Loading Whisper Small encoder...")
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)
del whisper_model

# Load semantic MSE weights into encoder
encoder_ckpt_path = PROJECT_ROOT / args.encoder_ckpt
print(f"  Loading semantic encoder from: {encoder_ckpt_path}")
enc_ckpt = torch.load(encoder_ckpt_path, map_location=DEVICE, weights_only=False)

if "encoder_state_dict" in enc_ckpt:
    encoder.load_state_dict(enc_ckpt["encoder_state_dict"])
elif "model_state_dict" in enc_ckpt:
    state = {}
    for k, v in enc_ckpt["model_state_dict"].items():
        if k.startswith("encoder."):
            state[k[len("encoder."):]] = v
    encoder.load_state_dict(state)
else:
    print(f"  ERROR: Unknown checkpoint format. Keys: {list(enc_ckpt.keys())[:5]}")
    exit(1)

enc_epoch = enc_ckpt.get("epoch", "?")
print(f"  Encoder loaded from epoch {enc_epoch}")
del enc_ckpt

encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# Load CTC head
ctc_ckpt_path = PROJECT_ROOT / args.ctc_ckpt
print(f"\n  Loading CTC head from: {ctc_ckpt_path}")
ctc_ckpt = torch.load(ctc_ckpt_path, map_location=DEVICE, weights_only=False)

ctc_head = nn.Linear(768, vocab_size).to(DEVICE)
ctc_head.load_state_dict(ctc_ckpt["ctc_head_state_dict"])
ctc_head.eval()

ctc_epoch = ctc_ckpt.get("epoch", "?")
ctc_dev_wer = ctc_ckpt.get("dev_wer", "?")
if isinstance(ctc_dev_wer, float) and ctc_dev_wer < 1:
    ctc_dev_wer = f"{ctc_dev_wer * 100:.2f}%"
print(f"  CTC head from epoch {ctc_epoch}, dev WER: {ctc_dev_wer}")
del ctc_ckpt

gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()


def load_audio(path, max_sec=None):
    """Load audio → 16kHz mono tensor."""
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
    """Whisper: mel stride=160, conv stride=2 → frames = samples // 320."""
    return min(num_samples // 160 // 2, 1500)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Evaluate
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 4: Evaluating on {args.split} set ({len(clips)} clips)")
print("=" * 70)

all_refs = []
all_hyps = []
lang_refs = {"hi": [], "mr": [], "en": []}
lang_hyps = {"hi": [], "mr": [], "en": []}
results = []
errors = 0

start_time = time.time()

with torch.no_grad():
    for clip in tqdm(clips, desc=f"  {args.split} eval", unit="clip",
                     bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:
                continue

            mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            enc_out = encoder(mel.input_features.to(DEVICE)).last_hidden_state
            real_frames = compute_real_frames(len(wav))
            features = enc_out[:, :real_frames, :]
            logits = ctc_head(features)

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
                    "clip": clip["clip_name"],
                    "language": clip["language"],
                    "lang_code": lc,
                    "ref": ref,
                    "hyp": predicted,
                })

            if DEVICE == "cuda":
                del enc_out, features, logits
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            continue

eval_time = time.time() - start_time

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 5: Results — Semantic Sequential CTC ({args.split} set)")
print("=" * 70)

overall_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0

lang_wers = {}
for lc in ("hi", "mr", "en"):
    if lang_refs[lc]:
        lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
    else:
        lang_wers[lc] = None

lang_names = {"hi": "Hindi", "mr": "Marathi", "en": "English"}

print(f"\n  ┌─ Semantic Sequential CTC — {args.split.upper()} Set ─────────────────────")
print(f"  │ Overall WER: {overall_wer:.2f}%")
for lc in ("hi", "mr", "en"):
    if lang_wers.get(lc) is not None:
        n = len(lang_refs[lc])
        print(f"  │   {lang_names[lc]:>8}: {lang_wers[lc]:.2f}% ({n} clips)")
print(f"  │ Total clips: {len(all_refs)} | Errors: {errors}")
print(f"  │ Time: {eval_time:.0f}s ({eval_time/60:.1f}min)")
print(f"  └──────────────────────────────────────────────────────────────")

# ── Comparison ───────────────────────────────────────────────────────────────
print(f"\n  Comparison (all on {args.split} set):")
print(f"    Semantic MSE (CTC Head 2, BPE):    47.86% (test, Hi+Mr only)")
print(f"    Semantic Seq CTC (this eval):       {overall_wer:.2f}% ({args.split}, all 3 languages)")
print(f"    Acoustic Joint Training:            19.97% (test)")

# ── Save predictions ─────────────────────────────────────────────────────────
if args.save_predictions and results:
    # All predictions
    pred_file = RESULTS_DIR / f"predictions_all_{args.split}.txt"
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write(f"# Semantic Sequential CTC — All Predictions ({args.split} set)\n")
        f.write(f"# Overall WER: {overall_wer:.2f}%\n\n")
        for r in results:
            f.write(f"[{r['language']}] {r['clip']}\n")
            f.write(f"  REF: {r['ref']}\n")
            f.write(f"  HYP: {r['hyp']}\n\n")
    print(f"\n  Saved: {pred_file}")

    # Mismatches only
    mismatch_file = RESULTS_DIR / f"predictions_mismatches_{args.split}.txt"
    mismatches = [r for r in results if r["ref"] != r["hyp"]]
    with open(mismatch_file, "w", encoding="utf-8") as f:
        f.write(f"# Semantic Sequential CTC — Mismatches ({args.split} set)\n")
        f.write(f"# Overall WER: {overall_wer:.2f}%\n")
        f.write(f"# Total mismatches: {len(mismatches)}/{len(results)}\n\n")
        for r in mismatches:
            f.write(f"[{r['language']}] {r['clip']}\n")
            f.write(f"  REF: {r['ref']}\n")
            f.write(f"  HYP: {r['hyp']}\n\n")
    print(f"  Saved: {mismatch_file}")

    # Summary
    summary_file = RESULTS_DIR / f"results_summary_{args.split}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Semantic Sequential CTC (244M + CTC Head)\n")
        f.write(f"# Encoder: Whisper Small, MSE-trained with IndicConformer\n")
        f.write(f"# CTC Head: Linear(768→85), trained on frozen encoder\n")
        f.write(f"# Encoder ckpt: {args.encoder_ckpt}\n")
        f.write(f"# CTC ckpt: {args.ctc_ckpt}\n")
        f.write(f"# {args.split} set: {len(all_refs)} clips\n\n")
        f.write(f"==================================================\n")
        f.write(f"  Language      Clips        WER\n")
        f.write(f"  -----------------------------------\n")
        for lc in ("hi", "mr", "en"):
            if lang_wers.get(lc) is not None:
                f.write(f"  {lang_names[lc]:<12} {len(lang_refs[lc]):>5}   {lang_wers[lc]:>8.2f}%\n")
        f.write(f"  -----------------------------------\n")
        f.write(f"  Overall      {len(all_refs):>5}   {overall_wer:>8.2f}%\n")
        f.write(f"==================================================\n")
    print(f"  Saved: {summary_file}")

# ── Sample predictions ───────────────────────────────────────────────────────
print(f"\n  Sample predictions:")
n_show = min(5, len(results))
for r in results[:n_show]:
    print(f"    [{r['language']}] REF: {r['ref']}")
    print(f"    [{r['language']}] HYP: {r['hyp']}")
    print()

print(f"{'=' * 70}")
