"""
Baseline Fine-tuning Evaluation — Test Set WER
================================================

Evaluates a fine-tuned baseline model (Whisper Small or Kid-Whisper Medium)
on the ASER test set and compares against all other systems.

Loads the encoder + CTC head from a training checkpoint and runs CTC
greedy decode on asr_test.csv with per-language WER breakdown.

Usage:
    # Evaluate fine-tuned Whisper Small
    python baseline_finetune_evaluate.py --model whisper-small \
        --checkpoint checkpoints/baselines/whisper-small/best_wer.pt

    # Evaluate fine-tuned Kid-Whisper Medium
    python baseline_finetune_evaluate.py --model kid-whisper \
        --checkpoint checkpoints/baselines/kid-whisper/best_wer.pt
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

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate fine-tuned baseline on test set")
parser.add_argument("--model", type=str, required=True,
                    choices=["whisper-small", "kid-whisper"],
                    help="Which model was fine-tuned")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best_wer.pt checkpoint")
parser.add_argument("--max_audio_sec", type=float, default=30.0)
args = parser.parse_args()

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "whisper-small": {
        "hf_id": "openai/whisper-small",
        "encoder_dim": 768,
    },
    "kid-whisper": {
        "hf_id": "aadel4/kid-whisper-medium-en-myst",
        "encoder_dim": 1024,
    },
}

config = MODEL_CONFIGS[args.model]
HF_ID = config["hf_id"]
ENCODER_DIM = config["encoder_dim"]

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TEST_CSV     = ASER_ROOT / "splits" / "asr_test.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
print(f"Model: {args.model} ({HF_ID})")

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
# STEP 2: Load test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading test set")
print("=" * 70)

test_clips = []
with open(TEST_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = row.get("language", "")
        lang_code = LANG_MAP.get(lang)
        if lang_code is None:
            continue
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ASER_ROOT / audio_path)
        gt = row.get("transcript", row.get("que_text", row.get("text", "")))
        test_clips.append({
            "audio_path": audio_path,
            "language": lang,
            "lang_code": lang_code,
            "ground_truth": gt,
        })

hi_n = sum(1 for c in test_clips if c["lang_code"] == "hi")
mr_n = sum(1 for c in test_clips if c["lang_code"] == "mr")
en_n = sum(1 for c in test_clips if c["lang_code"] == "en")
print(f"  Total: {len(test_clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load model + checkpoint
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Loading model + checkpoint")
print("=" * 70)

import gc
from transformers import WhisperModel, WhisperFeatureExtractor

feat_extractor = WhisperFeatureExtractor.from_pretrained(HF_ID)

print(f"  Loading encoder: {HF_ID}")
whisper_model = WhisperModel.from_pretrained(HF_ID)
encoder = whisper_model.encoder.to(DEVICE)
del whisper_model
gc.collect()

ctc_head = nn.Linear(ENCODER_DIM, vocab_size).to(DEVICE)

# Load checkpoint
ckpt_path = PROJECT_ROOT / args.checkpoint
print(f"  Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
encoder.load_state_dict(ckpt["encoder_state_dict"])
ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
epoch = ckpt.get("epoch", "?")
dev_wer = ckpt.get("dev_wer", "?")
print(f"  Checkpoint from epoch {epoch}, dev WER: {dev_wer}%")
del ckpt
gc.collect()

encoder.eval()
ctc_head.eval()


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
# STEP 4: Evaluate on test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Evaluating on test set")
print("=" * 70)

all_refs = []
all_hyps = []
lang_refs = {"hi": [], "mr": [], "en": []}
lang_hyps = {"hi": [], "mr": [], "en": []}
errors = 0

start_time = time.time()

with torch.no_grad():
    for clip in tqdm(test_clips, desc="  Test eval", unit="clip",
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
print(f"STEP 5: Test Results — {args.model} fine-tuned")
print("=" * 70)

overall_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0

lang_wers = {}
for lc in ("hi", "mr", "en"):
    if lang_refs[lc]:
        lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
    else:
        lang_wers[lc] = None

print(f"\n  ┌─ {args.model} Fine-tuned — Test Set ───────────────────────────")
print(f"  │ Overall WER: {overall_wer:.2f}%")
print(f"  │   Hindi:   {lang_wers.get('hi', 'N/A'):.2f}%" if lang_wers.get('hi') is not None else "  │   Hindi:   N/A")
print(f"  │   Marathi: {lang_wers.get('mr', 'N/A'):.2f}%" if lang_wers.get('mr') is not None else "  │   Marathi: N/A")
print(f"  │   English: {lang_wers.get('en', 'N/A'):.2f}%" if lang_wers.get('en') is not None else "  │   English: N/A")
print(f"  │ Clips evaluated: {len(all_refs)} | Errors: {errors}")
print(f"  │ Time: {eval_time:.0f}s ({eval_time/60:.1f}min)")
print(f"  └──────────────────────────────────────────────────────────────")

# ── Comparison table ─────────────────────────────────────────────────────────
print(f"\n  ┌─ System Comparison ─────────────────────────────────────────────")
print(f"  │ {'System':<35} {'Overall':>8} {'Hindi':>8} {'Marathi':>8} {'English':>8}")
print(f"  │ {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
print(f"  │ {'Whisper Small (zero-shot)':<35} {'~45%':>8} {'—':>8} {'—':>8} {'—':>8}")
print(f"  │ {'Acoustic (Kid-Whisper dist.)':<35} {'19.97%':>8} {'15.72%':>8} {'26.37%':>8} {'30.03%':>8}")
print(f"  │ {'Semantic (IndicConf. dist.)':<35} {'49.02%':>8} {'37.95%':>8} {'74.36%':>8} {'56.82%':>8}")
print(f"  │ {'Combined (gated fusion)':<35} {'17.10%':>8} {'—':>8} {'—':>8} {'—':>8}")

hi_str = f"{lang_wers['hi']:.2f}%" if lang_wers.get('hi') is not None else "—"
mr_str = f"{lang_wers['mr']:.2f}%" if lang_wers.get('mr') is not None else "—"
en_str = f"{lang_wers['en']:.2f}%" if lang_wers.get('en') is not None else "—"
print(f"  │ {f'{args.model} fine-tuned (this)':<35} {f'{overall_wer:.2f}%':>8} {hi_str:>8} {mr_str:>8} {en_str:>8}")
print(f"  └──────────────────────────────────────────────────────────────────")

# ── Save results ─────────────────────────────────────────────────────────────
results_file = RESULTS_DIR / f"{args.model}_test_results.txt"
with open(results_file, "w") as f:
    f.write(f"Baseline Fine-tuning Test Results: {args.model}\n")
    f.write(f"{'=' * 50}\n")
    f.write(f"Model: {HF_ID}\n")
    f.write(f"Checkpoint: {args.checkpoint}\n\n")
    f.write(f"Overall WER: {overall_wer:.2f}%\n")
    f.write(f"  Hindi:   {lang_wers.get('hi', 'N/A')}\n")
    f.write(f"  Marathi: {lang_wers.get('mr', 'N/A')}\n")
    f.write(f"  English: {lang_wers.get('en', 'N/A')}\n")
    f.write(f"\nClips evaluated: {len(all_refs)}\n")
    f.write(f"Errors: {errors}\n")

print(f"\n  Results saved: {results_file}")

# ── Sample predictions ───────────────────────────────────────────────────────
print(f"\n  Sample predictions:")
n_show = min(5, len(all_refs))
for i in range(n_show):
    print(f"    REF: {all_refs[i]}")
    print(f"    HYP: {all_hyps[i]}")
    print()

print(f"{'=' * 70}")
