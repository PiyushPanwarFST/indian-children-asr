"""
Step 2: MuRIL Cross-Modal Knowledge Distillation Training
=========================================================

WHAT:
    Trains the Whisper Small encoder to produce word-level speech features
    that match MuRIL (google/muril-base-cased) contextual text embeddings,
    while simultaneously maintaining CTC decodability. The result is an
    encoder whose internal representations carry meaning-level information
    (from MuRIL) without sacrificing transcription accuracy (from CTC).

WHY:
    Standard ASR encoders learn acoustic features optimized for character
    prediction, but these features lack semantic grounding. By distilling
    MuRIL's multilingual text understanding into the encoder, we get
    features that are aware of word meaning — useful for downstream tasks
    and for improving recognition of semantically similar words.

    Unlike the IndicConformer approach (step3 joint training), MuRIL and
    Whisper both have 768-dim hidden states, so NO bridge layer (CTC Head 2)
    is needed. We directly compare word-level averages of encoder frames
    against MuRIL word embeddings via MSE loss.

ARCHITECTURE:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Audio (.wav) — Hindi + Marathi + English                        │
    │       ↓                                                          │
    │  Mel Spectrogram → SpecAugment (freq+time masks)                 │
    │       ↓                                                          │
    │  Whisper Encoder (layers 0-5 FROZEN, 6-11 TRAINABLE)             │
    │       ↓                                                          │
    │  Encoder features (1, T, 768)                                    │
    │       ↓                                                          │
    │  ┌─────────────────────┐    ┌──────────────────────────────────┐  │
    │  │  BRANCH A: MSE      │    │  BRANCH B: CTC                  │  │
    │  │  For each word:      │    │  Dropout(0.1)                   │  │
    │  │    avg frames in     │    │       ↓                         │  │
    │  │    word span → (768) │    │  CTC Head 1 (768→85)            │  │
    │  │       ↓              │    │       ↓                         │  │
    │  │  MSE vs MuRIL emb   │    │  CTC Loss ← ground truth text   │  │
    │  └─────────────────────┘    └──────────────────────────────────┘  │
    │                                                                   │
    │  total_loss = α × MSE_word + (1-α) × CTC_loss                    │
    │  α schedule: epochs 1-10 → 0.3, 11-20 → 0.2, 21-30 → 0.1        │
    └───────────────────────────────────────────────────────────────────┘

WARM START:
    Encoder: from checkpoints/semantic_mse/best_dev.pt (step1 MSE distillation)
    CTC Head 1: from --ctc_checkpoint (e.g., checkpoints/semantic_ctc/best_wer.pt)
                OR random init if --ctc_checkpoint not provided
    The checkpoint format has keys: encoder_state_dict, ctc_head_hi_state_dict, etc.

Prerequisites:
    - Step 0: Pre-computed MuRIL embeddings in muril_embeddings/{split}/{uid}.pt
    - Step 1: Pre-computed CTC alignments in ctc_alignments/{split}/{uid}.pt
    - Step 1 encoder checkpoint: checkpoints/semantic_mse/best_dev.pt
    - Vocabulary: ASER-Dataset/vocab.json (85 tokens)
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Quick verify — 100 clips, component test only
    python step2_muril_training.py --verify --test_clips 100 --epochs 3

    # Full training with CTC warm start
    python step2_muril_training.py --epochs 30 --patience 7 \\
        --ctc_checkpoint checkpoints/semantic_ctc/best_wer.pt

    # Resume from checkpoint
    python step2_muril_training.py --resume checkpoints/semantic_muril/epoch_10.pt
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="MuRIL cross-modal knowledge distillation (MSE_word + CTC)"
)
parser.add_argument("--epochs", type=int, default=30,
                    help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-5,
                    help="Peak learning rate")
parser.add_argument("--warmup_steps", type=int, default=500,
                    help="Linear warmup steps before cosine decay")
parser.add_argument("--patience", type=int, default=7,
                    help="Early stopping patience on dev WER (0=disabled)")
parser.add_argument("--alpha_start", type=float, default=0.3,
                    help="Initial MSE loss weight (CTC weight = 1 - alpha)")
parser.add_argument("--freeze_layers", type=int, default=6,
                    help="Number of bottom encoder layers to freeze")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout before CTC head")
parser.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch size)")
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/semantic_mse/best_dev.pt",
                    help="Step 1 checkpoint to warm-start encoder from")
parser.add_argument("--ctc_checkpoint", type=str, default=None,
                    help="Warm-start CTC Head 1 from this checkpoint")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume training from a MuRIL training checkpoint")
parser.add_argument("--test_clips", type=int, default=None,
                    help="Limit training clips (works with or without --verify)")
parser.add_argument("--verify", action="store_true",
                    help="Test run with subset of clips (skips checkpoint saving)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--log_every", type=int, default=10,
                    help="Log training stats every N steps")
parser.add_argument("--max_audio_sec", type=float, default=30.0,
                    help="Maximum audio duration in seconds (truncate longer)")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
# 4 levels up: 05_muril_cross_modal → semantic_branch → experiments → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"

# Pre-computed data directories
MURIL_EMB_DIR   = Path(__file__).resolve().parent / "muril_embeddings"
CTC_ALIGN_DIR   = PROJECT_ROOT / "ctc_alignments"

# Checkpoint output
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "semantic_muril"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

# Minimum alignment quality to include a clip in training.
# Clips with alignment_quality below this are skipped.
MIN_ALIGNMENT_QUALITY = 0.3

# ── Add project root for utils import ────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

print(f"\nPaths:")
print(f"  PROJECT_ROOT:    {PROJECT_ROOT}")
print(f"  ASER_ROOT:       {ASER_ROOT}")
print(f"  MURIL_EMB_DIR:   {MURIL_EMB_DIR}")
print(f"  CTC_ALIGN_DIR:   {CTC_ALIGN_DIR}")
print(f"  CKPT_DIR:        {CKPT_DIR}")


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

print(f"  Vocab: {vocab_size} tokens (from {VOCAB_PATH.name})")
print(f"  Blank index: {BLANK_IDX}")


def text_to_indices(text, char_to_idx):
    """
    Convert normalized text to a list of vocabulary indices for CTC targets.

    Hindi/Marathi characters may be multi-codepoint (e.g., "ख़" = ख + ़).
    We try longest-match-first: check 2-char substring before falling back
    to single char. Spaces become <space>, unknowns become <unk>.

    Args:
        text (str): Normalized text (lowercase, no punctuation).
        char_to_idx (dict): Character-to-index vocabulary mapping.

    Returns:
        list[int]: Vocabulary indices for each character/token.
    """
    indices = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            indices.append(char_to_idx["<space>"])
            i += 1
        # Try 2-char match first (for nukta combinations like ख़, ग़, etc.)
        elif i + 1 < len(text) and text[i:i+2] in char_to_idx:
            indices.append(char_to_idx[text[i:i+2]])
            i += 2
        elif text[i] in char_to_idx:
            indices.append(char_to_idx[text[i]])
            i += 1
        else:
            indices.append(char_to_idx.get("<unk>", 2))
            i += 1
    return indices


def ctc_greedy_decode(logits, idx_to_char):
    """
    Greedy CTC decode from logits tensor of shape (T, vocab_size).

    Applies argmax per frame, collapses consecutive duplicates, removes
    blanks, and converts indices back to characters.

    Args:
        logits (torch.Tensor): Frame-level logits, shape (T, vocab_size).
        idx_to_char (dict): Index-to-character mapping.

    Returns:
        str: Decoded text string.
    """
    indices = torch.argmax(logits, dim=-1)  # (T,)
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
# STEP 2: Load training data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading training data")
print("=" * 70)


def load_clips_from_csv(csv_path, split="train", max_clips=None):
    """
    Load clips from ASER split CSV file.

    For each clip, checks that both MuRIL embeddings and CTC alignments
    exist on disk. Clips missing either are skipped.

    All 3 languages (Hindi, Marathi, English) are included — MuRIL supports
    all of them.

    Args:
        csv_path (str or Path): Path to the CSV file (asr_train.csv, etc.).
        split (str): Split name ("train", "dev", "test") — used to locate
            pre-computed files in muril_embeddings/{split}/ and
            ctc_alignments/{split}/.
        max_clips (int or None): If set, return at most this many clips.

    Returns:
        clips (list[dict]): List of clip dicts with keys:
            audio_path, language, lang_code, clip_name, ground_truth,
            duration_sec, muril_path, align_path
        skipped_lang (int): Clips with unknown language.
        skipped_no_muril (int): Clips missing MuRIL embeddings.
        skipped_no_align (int): Clips missing CTC alignments.
    """
    clips = []
    skipped_lang = 0
    skipped_no_muril = 0
    skipped_no_align = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row.get("language", "")
            lang_code = LANGUAGE_MAP.get(lang)

            if lang_code is None:
                skipped_lang += 1
                continue

            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = str(ASER_ROOT / audio_path)

            # Use child_id + basename as unique identifier (matches step0/step1)
            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            # Check for pre-computed MuRIL embeddings
            muril_path = MURIL_EMB_DIR / split / f"{clip_uid}.pt"
            if not muril_path.exists():
                skipped_no_muril += 1
                continue

            # Check for pre-computed CTC alignments
            align_path = CTC_ALIGN_DIR / split / f"{clip_uid}.pt"
            if not align_path.exists():
                skipped_no_align += 1
                continue

            gt = row.get("transcript", row.get("que_text", row.get("text", "")))

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "clip_name": clip_uid,
                "ground_truth": gt,
                "duration_sec": float(row.get("duration_sec", 0)),
                "muril_path": str(muril_path),
                "align_path": str(align_path),
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips, skipped_lang, skipped_no_muril, skipped_no_align


print(f"  Mode: MuRIL Cross-Modal KD (MSE_word + CTC)")

train_clips, skip_lang, skip_muril, skip_align = load_clips_from_csv(
    TRAIN_CSV, split="train")
hi_clips = [c for c in train_clips if c["lang_code"] == "hi"]
mr_clips = [c for c in train_clips if c["lang_code"] == "mr"]
en_clips = [c for c in train_clips if c["lang_code"] == "en"]
total_hrs = sum(c["duration_sec"] for c in train_clips) / 3600

print(f"  Total: {len(train_clips)} clips ({total_hrs:.1f}h)")
print(f"    Hindi:   {len(hi_clips)} clips")
print(f"    Marathi: {len(mr_clips)} clips")
print(f"    English: {len(en_clips)} clips")
print(f"  Skipped: {skip_lang} excluded lang, "
      f"{skip_muril} missing MuRIL emb, {skip_align} missing alignments")

if len(train_clips) == 0:
    print("\n  ERROR: No training clips found!")
    print("  Make sure step0 (MuRIL embeddings) and step1 (CTC alignments) "
          "have been run.")
    exit(1)

# Dev set — all languages
dev_clips = []
if DEV_CSV.exists():
    dev_clips, _, _, _ = load_clips_from_csv(DEV_CSV, split="dev")
    dev_hi = sum(1 for c in dev_clips if c["lang_code"] == "hi")
    dev_mr = sum(1 for c in dev_clips if c["lang_code"] == "mr")
    dev_en = sum(1 for c in dev_clips if c["lang_code"] == "en")
    print(f"  Dev set: {len(dev_clips)} clips "
          f"({dev_hi} hi, {dev_mr} mr, {dev_en} en)")

# Clip limiting (--test_clips or --verify)
N = args.test_clips
if N is None and args.verify:
    N = 100  # default for verify mode

if N and N < len(train_clips):
    random.seed(args.seed)
    total_all = len(train_clips)
    # Proportional sampling across languages
    lang_clips = {"hi": hi_clips, "mr": mr_clips, "en": en_clips}
    sampled = []
    remaining = N
    for lc, lclips in lang_clips.items():
        n_lang = min(int(N * len(lclips) / total_all), len(lclips))
        if lclips:
            sampled.extend(random.sample(lclips, n_lang))
            remaining -= n_lang
    # Fill remainder from any language
    if remaining > 0:
        leftover = [c for c in train_clips if c not in sampled]
        sampled.extend(random.sample(leftover, min(remaining, len(leftover))))
    train_clips = sampled
    random.shuffle(train_clips)
    hi_n = sum(1 for c in train_clips if c["lang_code"] == "hi")
    mr_n = sum(1 for c in train_clips if c["lang_code"] == "mr")
    en_n = sum(1 for c in train_clips if c["lang_code"] == "en")
    label = "VERIFY MODE" if args.verify else "LIMITED"
    print(f"\n  {label}: {len(train_clips)} clips "
          f"({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

    if args.verify and dev_clips:
        dev_n = min(100, len(dev_clips))
        random.seed(args.seed + 1)
        dev_clips = random.sample(dev_clips, dev_n)
        print(f"    Dev subset: {len(dev_clips)} clips")

print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load models (warm start from step1)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Loading models")
print("=" * 70)

import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"

# ── Feature extractor ────────────────────────────────────────────────────────
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

# ── Whisper encoder ──────────────────────────────────────────────────────────
print("\n  [STUDENT] Loading Whisper Small encoder...")
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)

# ── CTC Head 1: Linear(768→85) — character-level decoding head ──────────────
print(f"  [CTC HEAD 1] Decoding head: Linear(768 -> {vocab_size})...")
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# ── Dropout ──────────────────────────────────────────────────────────────────
head_dropout = nn.Dropout(args.dropout)

# ── Load step1 checkpoint (warm start) ───────────────────────────────────────
start_epoch = 0
best_dev_wer = float("inf")
global_step = 0

if args.resume:
    # Resume from a MuRIL training checkpoint
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Load encoder weights (handle possible 'encoder.' prefix)
    enc_sd = ckpt["encoder_state_dict"]
    cleaned_enc = {}
    for k, v in enc_sd.items():
        key = k.replace("encoder.", "") if k.startswith("encoder.") else k
        cleaned_enc[key] = v
    encoder.load_state_dict(cleaned_enc)

    # Load CTC head
    ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))
    global_step = ckpt.get("global_step", 0)
    print(f"    Resuming from epoch {start_epoch}, "
          f"best dev WER: {best_dev_wer:.2f}%, global_step: {global_step}")
else:
    # Warm start: encoder from step1, CTC head optionally from a CTC checkpoint
    ckpt_path = PROJECT_ROOT / args.checkpoint
    print(f"\n  [WARM START] Loading step1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Load encoder — handle possible 'encoder.' prefix in state dict keys
    enc_sd = ckpt["encoder_state_dict"]
    cleaned_enc = {}
    for k, v in enc_sd.items():
        key = k.replace("encoder.", "") if k.startswith("encoder.") else k
        cleaned_enc[key] = v
    encoder.load_state_dict(cleaned_enc)
    print(f"    Encoder loaded from epoch {ckpt.get('epoch', '?')}")
    print(f"    Step1 train MSE: {ckpt.get('train_loss', '?')}")
    print(f"    Step1 dev MSE:   {ckpt.get('dev_loss', '?')}")

    # CTC Head 1: warm start from --ctc_checkpoint if provided
    if args.ctc_checkpoint:
        ctc_ckpt_path = PROJECT_ROOT / args.ctc_checkpoint
        print(f"\n  [CTC HEAD 1] Warm-starting from: {ctc_ckpt_path}")
        ctc_ckpt = torch.load(ctc_ckpt_path, map_location=DEVICE, weights_only=False)

        # Try multiple possible key names for the CTC head state dict
        ctc_sd = None
        for key_name in ("ctc_head_state_dict", "ctc_head_char_state_dict"):
            if key_name in ctc_ckpt:
                ctc_sd = ctc_ckpt[key_name]
                print(f"    Found CTC head weights under key: '{key_name}'")
                break

        if ctc_sd is not None:
            ctc_head.load_state_dict(ctc_sd)
            print(f"    CTC Head loaded. "
                  f"Source dev WER: {ctc_ckpt.get('dev_wer', '?')}")
        else:
            print(f"    WARNING: No CTC head state dict found in checkpoint!")
            print(f"    Available keys: {list(ctc_ckpt.keys())}")
            print(f"    CTC Head 1 will use random init.")

        del ctc_ckpt
    else:
        print(f"\n  [CTC HEAD 1] No --ctc_checkpoint provided → random init")

del ckpt
gc.collect()

# ── Freeze bottom encoder layers ────────────────────────────────────────────
n_freeze = args.freeze_layers
n_layers = len(encoder.layers)
print(f"\n  [FREEZE] Freezing bottom {n_freeze}/{n_layers} encoder layers...")

# Freeze conv layers (always freeze — they're low-level mel processors)
for param in encoder.conv1.parameters():
    param.requires_grad = False
for param in encoder.conv2.parameters():
    param.requires_grad = False

# Freeze bottom N transformer layers
for layer in encoder.layers[:n_freeze]:
    for param in layer.parameters():
        param.requires_grad = False

# Count params
total_params = sum(p.numel() for p in encoder.parameters())
frozen_params = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
trainable_enc = total_params - frozen_params
head_params = sum(p.numel() for p in ctc_head.parameters())

print(f"    Encoder: {total_params:,} total, {trainable_enc:,} trainable, "
      f"{frozen_params:,} frozen")
print(f"    CTC Head 1 (char): {head_params:,} params")
print(f"    Total trainable: {trainable_enc + head_params:,}")

# Enable gradient checkpointing for memory efficiency
encoder.gradient_checkpointing_enable()
print(f"    Gradient checkpointing: ENABLED")

encoder.train()
ctc_head.train()

# Clean up
del whisper_model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  GPU Memory: {allocated:.0f} MB allocated")

print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Helper functions
# ══════════════════════════════════════════════════════════════════════════════


def load_audio(path, max_sec=None):
    """
    Load audio file and convert to 16kHz mono waveform tensor.

    Resamples if the source sample rate differs from 16kHz. Multi-channel
    audio is averaged to mono. Optionally truncates to max_sec seconds.

    Args:
        path (str): Path to audio file (.wav, .mp3, etc.).
        max_sec (float or None): Maximum duration in seconds. None for no limit.

    Returns:
        torch.Tensor: 1-D waveform tensor at 16kHz.
    """
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()  # (samples,)
    if max_sec and len(wav) > int(max_sec * SAMPLE_RATE):
        wav = wav[:int(max_sec * SAMPLE_RATE)]
    return wav


def compute_real_frames(num_samples):
    """
    Compute number of real encoder output frames for a given audio length.

    Whisper's mel spectrogram uses 10ms hop (160 samples per frame), then
    the convolutional frontend has stride 2, so encoder output has
    num_samples // 160 // 2 frames, capped at 1500 (Whisper's max for 30s).

    Args:
        num_samples (int): Number of audio samples at 16kHz.

    Returns:
        int: Number of encoder frames (at most 1500).
    """
    return min(num_samples // 160 // 2, 1500)


def apply_spec_augment(mel_features):
    """
    Apply SpecAugment to mel spectrogram for training regularization.

    Applies 2 frequency masks (max 15 bins each) and 2 time masks
    (max 50 frames each) by zeroing out random contiguous bands.

    Args:
        mel_features (torch.Tensor): Shape (1, 80, T) mel spectrogram.

    Returns:
        torch.Tensor: Augmented mel spectrogram, same shape.
    """
    _, n_freq, n_time = mel_features.shape

    # Frequency masking
    for _ in range(2):
        f = random.randint(0, 15)
        f0 = random.randint(0, max(n_freq - f, 1) - 1)
        mel_features[:, f0:f0 + f, :] = 0.0

    # Time masking
    for _ in range(2):
        t = random.randint(0, 50)
        t0 = random.randint(0, max(n_time - t, 1) - 1)
        mel_features[:, :, t0:t0 + t] = 0.0

    return mel_features


def get_alpha(epoch):
    """
    Get MSE loss weight (alpha) for the given epoch.

    Alpha schedule decays over training to let CTC loss dominate later:
        epochs  1-10: alpha_start (default 0.3)
        epochs 11-20: alpha_start - 0.1 (default 0.2)
        epochs 21-30: alpha_start - 0.2 (default 0.1)

    Args:
        epoch (int): Current epoch number (1-indexed).

    Returns:
        float: MSE loss weight (0.0 to alpha_start).
    """
    if epoch <= 10:
        return args.alpha_start            # 0.3
    elif epoch <= 20:
        return max(args.alpha_start - 0.1, 0.0)  # 0.2
    else:
        return max(args.alpha_start - 0.2, 0.0)  # 0.1


def load_precomputed_data(clip):
    """
    Load pre-computed MuRIL embeddings and CTC alignments for a clip.

    MuRIL embeddings (from step0):
        word_embeddings: (N_words, 768) contextual word embeddings
        words: list[str] of whitespace-delimited words
        num_words: int

    CTC alignments (from step1):
        word_boundaries: list of dicts with word, start_frame, end_frame,
            chars_matched, chars_total
        num_frames: int
        num_words_aligned: int
        num_words_total: int
        alignment_quality: float (0.0 to 1.0)

    Args:
        clip (dict): Clip dict with muril_path and align_path keys.

    Returns:
        muril_data (dict): MuRIL embedding data.
        align_data (dict): CTC alignment data.
    """
    muril_data = torch.load(clip["muril_path"], weights_only=True)
    align_data = torch.load(clip["align_path"], weights_only=True)
    return muril_data, align_data


def compute_word_mse_loss(encoder_features, muril_data, align_data):
    """
    Compute word-level MSE loss between encoder frame averages and MuRIL embeddings.

    For each aligned word:
        1. Get the word's frame span (start_frame, end_frame) from CTC alignment.
        2. Average encoder features in that span → (768,) speech embedding.
        3. Get the corresponding MuRIL embedding by word INDEX.
           Both come from the same ground-truth word list, so index i in
           word_boundaries corresponds to the i-th word that was aligned.
           We find the matching MuRIL word by looking up the word string.
        4. Compute MSE between speech embedding and MuRIL embedding.

    The word INDEX matching works as follows:
        - muril_data["words"] = ["यह", "एक", "किताब", "है"]  (all GT words)
        - align_data["word_boundaries"] = [
              {"word": "यह", "start_frame": 5, "end_frame": 15, ...},
              {"word": "किताब", "start_frame": 25, "end_frame": 45, ...},
          ]
        - For boundary "यह" we find its index in muril words list → 0
        - For boundary "किताब" we find its index → 2
        - Safety: skip if word index >= muril num_words

    Args:
        encoder_features (torch.Tensor): Shape (1, T, 768) encoder output,
            already sliced to real frames.
        muril_data (dict): Pre-computed MuRIL embeddings with word_embeddings
            (N_words, 768) and words (list[str]).
        align_data (dict): Pre-computed CTC alignments with word_boundaries
            (list of dicts).

    Returns:
        mse_loss (torch.Tensor or None): Scalar MSE loss if >= 1 valid word
            pair was found, else None.
        num_words_used (int): Number of word pairs used in the loss.
    """
    word_boundaries = align_data["word_boundaries"]
    muril_embeddings = muril_data["word_embeddings"]  # (N_muril, 768)
    muril_words = muril_data["words"]                 # list[str]
    num_muril_words = muril_data["num_words"]

    if num_muril_words == 0 or len(word_boundaries) == 0:
        return None, 0

    # Build a word-to-index mapping from MuRIL words for fast lookup.
    # If duplicate words exist, we track all indices and consume them
    # in order (left-to-right greedy matching).
    muril_word_indices = {}
    for idx, w in enumerate(muril_words):
        if w not in muril_word_indices:
            muril_word_indices[w] = []
        muril_word_indices[w].append(idx)

    # Track which muril indices have been used (for duplicate words)
    muril_used = set()

    speech_embeds = []
    muril_embeds = []
    T = encoder_features.shape[1]

    for boundary in word_boundaries:
        word = boundary["word"]
        start_frame = boundary["start_frame"]
        end_frame = boundary["end_frame"]

        # Bounds check on frames
        if start_frame >= T or end_frame >= T:
            continue
        if start_frame > end_frame:
            continue

        # Find matching MuRIL word index
        if word not in muril_word_indices:
            continue

        # Pick the first unused index for this word
        muril_idx = None
        for candidate_idx in muril_word_indices[word]:
            if candidate_idx not in muril_used:
                muril_idx = candidate_idx
                break

        if muril_idx is None:
            continue

        # Safety check: index must be within muril embeddings
        if muril_idx >= num_muril_words:
            continue

        muril_used.add(muril_idx)

        # Average encoder frames in the word's span → (768,)
        # end_frame is inclusive, so we use end_frame + 1
        word_frames = encoder_features[0, start_frame:end_frame + 1, :]  # (span, 768)
        speech_embed = word_frames.mean(dim=0)  # (768,)

        # Get MuRIL embedding for this word → (768,)
        muril_embed = muril_embeddings[muril_idx].to(encoder_features.device)  # (768,)

        speech_embeds.append(speech_embed)
        muril_embeds.append(muril_embed)

    if len(speech_embeds) == 0:
        return None, 0

    # Stack all word embeddings → (N_aligned, 768)
    speech_stack = torch.stack(speech_embeds)   # (N_aligned, 768)
    muril_stack = torch.stack(muril_embeds)     # (N_aligned, 768)

    # MSE loss between speech and MuRIL word embeddings
    mse_loss = F.mse_loss(speech_stack, muril_stack)

    return mse_loss, len(speech_embeds)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Component verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Component verification")
print("=" * 70)

test_clip = train_clips[0]
print(f"\n  Test clip: {test_clip['clip_name']} "
      f"({test_clip['language']})")

# ── Load pre-computed data ──────────────────────────────────────────────────
test_muril, test_align = load_precomputed_data(test_clip)
print(f"  MuRIL embeddings: {tuple(test_muril['word_embeddings'].shape)} "
      f"({test_muril['num_words']} words)")
print(f"  MuRIL words: {test_muril['words'][:5]}{'...' if len(test_muril['words']) > 5 else ''}")
print(f"  CTC alignment: {test_align['num_words_aligned']}/{test_align['num_words_total']} words "
      f"(quality={test_align['alignment_quality']:.2f})")
if test_align['word_boundaries']:
    wb0 = test_align['word_boundaries'][0]
    print(f"  First word boundary: '{wb0['word']}' frames {wb0['start_frame']}-{wb0['end_frame']}")

# ── Load audio and run encoder ──────────────────────────────────────────────
test_wav = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"\n  Audio: {len(test_wav)/SAMPLE_RATE:.1f}s")

# Mel + SpecAugment
mel = feat_extractor(test_wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
mel_aug = apply_spec_augment(mel.input_features.clone())
print(f"  Mel: {tuple(mel.input_features.shape)} -> SpecAugment applied")

# Encoder forward
enc_out = encoder(mel_aug.to(DEVICE))
features = enc_out.last_hidden_state
real_frames = compute_real_frames(len(test_wav))
real_features = features[:, :real_frames, :]
print(f"  Encoder: (1, 1500, 768) -> slice({real_frames}) -> {tuple(real_features.shape)}")

# ── Word-level MSE loss ─────────────────────────────────────────────────────
mse_loss, num_words_mse = compute_word_mse_loss(real_features, test_muril, test_align)
if mse_loss is not None:
    print(f"\n  Word MSE loss: {mse_loss.item():.4f} ({num_words_mse} words used)")
else:
    print(f"\n  Word MSE loss: N/A (no valid word pairs found)")

# ── CTC loss ────────────────────────────────────────────────────────────────
char_logits = ctc_head(head_dropout(real_features))  # (1, T, 85)
print(f"  CTC Head 1: {tuple(char_logits.shape)} (our char vocab)")

gt = normalize_text(test_clip["ground_truth"])
target_indices = text_to_indices(gt, char_to_idx)

ctc_loss = None
if target_indices and real_frames > len(target_indices):
    log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)
    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    print(f"  CTC loss: {ctc_loss.item():.4f}")
    print(f"  Ground truth: '{gt[:80]}' ({len(target_indices)} chars)")
else:
    print(f"  CTC loss: skipped (empty/long target)")

# ── Combined loss + backward ───────────────────────────────────────────────
alpha = get_alpha(1)
if mse_loss is not None and ctc_loss is not None:
    total_loss = alpha * mse_loss + (1 - alpha) * ctc_loss
    total_loss.backward()
    print(f"\n  Combined: {alpha:.1f} x MSE + {1 - alpha:.1f} x CTC = {total_loss.item():.4f}")
elif ctc_loss is not None:
    ctc_loss.backward()
    print(f"\n  CTC-only loss: {ctc_loss.item():.4f} (no valid word pairs for MSE)")
elif mse_loss is not None:
    mse_loss.backward()
    print(f"\n  MSE-only loss: {mse_loss.item():.4f} (CTC skipped)")
else:
    print(f"\n  No loss computed — both MSE and CTC were skipped")

# ── Verify gradient flow ───────────────────────────────────────────────────
frozen_grads = sum(
    1 for p in encoder.layers[:n_freeze].parameters()
    if p.grad is not None and p.grad.abs().sum() > 0
)
unfrozen_grads = sum(
    1 for p in encoder.layers[n_freeze:].parameters()
    if p.grad is not None and p.grad.abs().sum() > 0
)
head_grads = sum(1 for p in ctc_head.parameters() if p.grad is not None)

print(f"  Gradients: frozen_layers={frozen_grads}(should=0) | "
      f"unfrozen_layers={unfrozen_grads}>0 | head1={head_grads}>0")

if frozen_grads > 0:
    print(f"  WARNING: Frozen layers have gradients! Check freeze logic.")
if unfrozen_grads == 0:
    print(f"  WARNING: No gradients in unfrozen layers!")
if head_grads == 0:
    print(f"  WARNING: No gradients in CTC head!")

encoder.zero_grad()
ctc_head.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# Cleanup test data
del test_muril, test_align, test_wav, mel, mel_aug, features, real_features
del enc_out, char_logits, mse_loss, ctc_loss
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

print(f"\n  ALL VERIFIED\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Training setup")
print("=" * 70)

# Collect trainable parameters: unfrozen encoder layers + CTC head
trainable_params = [p for p in encoder.parameters() if p.requires_grad]
trainable_params += list(ctc_head.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

num_epochs = args.epochs
total_steps_est = (len(train_clips) * num_epochs) // args.grad_accum

# Warmup + cosine decay scheduler
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    """
    Learning rate schedule: linear warmup then cosine decay.

    Args:
        step (int): Current global step.

    Returns:
        float: LR multiplier (0.0 to 1.0).
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Load optimizer/scheduler state if resuming
if args.resume:
    resume_path = PROJECT_ROOT / args.resume
    resume_ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print(f"  Optimizer state restored from checkpoint")
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        print(f"  Scheduler state restored from checkpoint")
    del resume_ckpt
    gc.collect()

ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) + cosine decay")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)} | Grad accum: {args.grad_accum}")
print(f"  Effective batch size: {args.grad_accum}")
print(f"  Alpha schedule: {args.alpha_start} -> "
      f"{max(args.alpha_start - 0.1, 0.0)} -> "
      f"{max(args.alpha_start - 0.2, 0.0)}")
print(f"  Dropout: {args.dropout}")
print(f"  Frozen layers: {n_freeze}/{n_layers}")
print(f"  Min alignment quality: {MIN_ALIGNMENT_QUALITY}")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience} (on dev WER)")
print()


# ── Dev evaluation function ─────────────────────────────────────────────────

def evaluate_dev(dev_clips_list):
    """
    Evaluate on dev set using CTC Head 1 (greedy decode) for WER,
    and word-level MSE as a secondary metric.

    WER from CTC Head 1 is the PRIMARY metric used for checkpointing
    and early stopping. The word-level MSE is reported for monitoring
    but does not drive decisions.

    Per-language WER breakdown is computed for Hindi, Marathi, English.

    Args:
        dev_clips_list (list[dict]): Dev clips to evaluate.

    Returns:
        dev_wer (float): Corpus-level WER in percent (0-100).
        dev_mse (float): Average word-level MSE on dev set.
        all_preds (list[tuple]): List of (reference, hypothesis) string pairs.
        lang_wers (dict): Per-language WER, e.g. {"hi": 35.2, "mr": 70.1, "en": 55.0}.
            Value is None if no clips for that language.
    """
    encoder.eval()
    ctc_head.eval()

    all_refs = []
    all_hyps = []
    mse_losses = []
    # Per-language tracking
    lang_refs = {"hi": [], "mr": [], "en": []}
    lang_hyps = {"hi": [], "mr": [], "en": []}

    with torch.no_grad():
        for clip in tqdm(dev_clips_list, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(wav) < 8000:
                    continue

                # Mel (no SpecAugment for eval)
                mel = feat_extractor(
                    wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
                )
                enc_out = encoder(mel.input_features.to(DEVICE))
                features = enc_out.last_hidden_state
                real_frames = compute_real_frames(len(wav))
                real_features = features[:, :real_frames, :]

                # ── Word-level MSE (secondary metric) ──
                try:
                    muril_data, align_data = load_precomputed_data(clip)
                    if (align_data["alignment_quality"] >= MIN_ALIGNMENT_QUALITY
                            and align_data["num_words_aligned"] >= 2):
                        mse_val, n_words = compute_word_mse_loss(
                            real_features, muril_data, align_data
                        )
                        if mse_val is not None:
                            mse_losses.append(mse_val.item())
                except Exception:
                    pass  # MSE is secondary — don't fail dev eval over it

                # ── WER from CTC Head 1 (primary metric) ──
                char_logits = ctc_head(real_features)  # (1, T, 85)
                predicted = ctc_greedy_decode(char_logits.squeeze(0), idx_to_char)

                ref = normalize_text(clip["ground_truth"])
                if ref:
                    all_refs.append(ref)
                    all_hyps.append(predicted)
                    lc = clip["lang_code"]
                    if lc in lang_refs:
                        lang_refs[lc].append(ref)
                        lang_hyps[lc].append(predicted)

                # Memory cleanup
                if DEVICE == "cuda":
                    del features, real_features, char_logits
                    torch.cuda.empty_cache()

            except Exception:
                continue

    encoder.train()
    ctc_head.train()

    dev_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0
    dev_mse = np.mean(mse_losses) if mse_losses else 0.0

    # Per-language WER
    lang_wers = {}
    for lc in ("hi", "mr", "en"):
        if lang_refs[lc]:
            lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
        else:
            lang_wers[lc] = None

    return dev_wer, dev_mse, list(zip(all_refs, all_hyps)), lang_wers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
mode_str = (f"VERIFICATION ({len(train_clips)} clips x {min(3, num_epochs)} epochs)"
            if args.verify
            else f"Full training ({len(train_clips)} clips x {num_epochs} epochs)")
print(f"STEP 7: {mode_str}")
print("=" * 70)

random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

# Training state
epoch_stats = []
patience_counter = 0
training_start = time.time()

# If verify mode, cap epochs to 3
effective_epochs = min(3, num_epochs) if args.verify else num_epochs

for epoch in range(start_epoch + 1, start_epoch + effective_epochs + 1):
    alpha = get_alpha(epoch)
    random.shuffle(train_clips)

    ep_mse_losses = []
    ep_ctc_losses = []
    ep_combined_losses = []
    ep_words_used = []
    ep_start = time.time()
    skipped = 0
    skipped_quality = 0
    skipped_few_words = 0

    optimizer.zero_grad()

    pbar = tqdm(train_clips, desc=f"Epoch {epoch} (a={alpha:.1f})",
                unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # ── Load pre-computed MuRIL embeddings and CTC alignments ──
            muril_data, align_data = load_precomputed_data(clip)

            # Skip clips with poor alignment quality
            if align_data["alignment_quality"] < MIN_ALIGNMENT_QUALITY:
                skipped_quality += 1
                skipped += 1
                continue

            # Skip clips with too few aligned words (need >= 2 for meaningful MSE)
            if align_data["num_words_aligned"] < 2:
                skipped_few_words += 1
                skipped += 1
                continue

            # ── Load audio ──
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:
                skipped += 1
                continue

            num_samples = len(wav)

            # ── Mel + SpecAugment ──
            mel = feat_extractor(
                wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            mel_features = apply_spec_augment(mel.input_features.clone())

            # ── Encoder forward ──
            enc_out = encoder(mel_features.to(DEVICE))
            features = enc_out.last_hidden_state  # (1, 1500, 768)
            real_frames = compute_real_frames(num_samples)
            real_features = features[:, :real_frames, :]  # (1, T, 768)

            # ── BRANCH A: Word-level MSE loss (encoder vs MuRIL) ──
            mse_loss, num_words = compute_word_mse_loss(
                real_features, muril_data, align_data
            )

            # ── BRANCH B: CTC loss (CTC Head 1 vs ground truth) ──
            gt = normalize_text(clip["ground_truth"])
            target_indices = text_to_indices(gt, char_to_idx)

            ctc_loss = None
            if target_indices and real_frames > len(target_indices):
                char_logits = ctc_head(head_dropout(real_features))  # (1, T, 85)
                log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)

                targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
                input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
                target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

                ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

                # Check for NaN/Inf
                if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                    ctc_loss = None

            # ── Combine losses ──
            if mse_loss is not None and ctc_loss is not None:
                # Both losses available — full combined
                total_loss = (alpha * mse_loss + (1 - alpha) * ctc_loss) / args.grad_accum
                total_loss.backward()
                ep_mse_losses.append(mse_loss.item())
                ep_ctc_losses.append(ctc_loss.item())
                ep_combined_losses.append(
                    alpha * mse_loss.item() + (1 - alpha) * ctc_loss.item()
                )
                ep_words_used.append(num_words)
            elif ctc_loss is not None:
                # Only CTC available (MSE had no valid word pairs)
                total_loss = ctc_loss / args.grad_accum
                total_loss.backward()
                ep_ctc_losses.append(ctc_loss.item())
                ep_combined_losses.append(ctc_loss.item())
            elif mse_loss is not None:
                # Only MSE available (CTC target was empty/too long)
                total_loss = mse_loss / args.grad_accum
                total_loss.backward()
                ep_mse_losses.append(mse_loss.item())
                ep_combined_losses.append(mse_loss.item())
                ep_words_used.append(num_words)
            else:
                # Neither loss could be computed — skip
                skipped += 1
                continue

            # ── Gradient accumulation step ──
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_clips):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ── Memory cleanup ──
            if DEVICE == "cuda":
                del features, real_features
                if ctc_loss is not None:
                    del char_logits, log_probs
                torch.cuda.empty_cache()

            # ── Progress bar update ──
            if ep_combined_losses and (i + 1) % args.log_every == 0:
                avg_recent = np.mean(ep_combined_losses[-20:])
                postfix = {
                    "comb": f"{avg_recent:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                }
                if ep_mse_losses:
                    postfix["mse"] = f"{ep_mse_losses[-1]:.3f}"
                if ep_ctc_losses:
                    postfix["ctc"] = f"{ep_ctc_losses[-1]:.3f}"
                if ep_words_used:
                    postfix["wds"] = f"{ep_words_used[-1]}"
                pbar.set_postfix(postfix)

        except Exception as e:
            if (i + 1) <= 5 or (i + 1) % 500 == 0:
                print(f"\n  ERROR step {global_step}: {clip['clip_name']}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            skipped += 1
            continue

    pbar.close()

    # ── Epoch summary ────────────────────────────────────────────────────
    ep_time = time.time() - ep_start
    avg_mse = np.mean(ep_mse_losses) if ep_mse_losses else 0.0
    avg_ctc = np.mean(ep_ctc_losses) if ep_ctc_losses else 0.0
    avg_combined = np.mean(ep_combined_losses) if ep_combined_losses else 0.0
    avg_words = np.mean(ep_words_used) if ep_words_used else 0.0

    print(f"\n  ┌─ Epoch {epoch} ─────────────────────────────────────────────────")
    print(f"  │ MSE loss (word):  {avg_mse:.4f} ({len(ep_mse_losses)} clips, "
          f"avg {avg_words:.1f} words/clip)")
    print(f"  │ CTC loss:         {avg_ctc:.4f} ({len(ep_ctc_losses)} clips)")
    print(f"  │ Combined:         {avg_combined:.4f} (alpha={alpha:.1f})")
    print(f"  │ Processed: {len(ep_combined_losses)} clips | "
          f"Skipped: {skipped} ({skipped_quality} quality, "
          f"{skipped_few_words} few_words)")
    print(f"  │ Steps: {global_step} | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
    print(f"  │ LR: {scheduler.get_last_lr()[0]:.2e}")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  │ Peak GPU: {peak:.0f} MB")
    print(f"  └──────────────────────────────────────────────────────────────")

    # ── Dev evaluation ───────────────────────────────────────────────────
    dev_wer, dev_mse, dev_preds, lang_wers = 999.0, 0.0, [], {}
    if dev_clips:
        dev_wer, dev_mse, dev_preds, lang_wers = evaluate_dev(dev_clips)
        print(f"\n  ┌─ Dev Results ───────────────────────────────────────────────")
        print(f"  │ WER:  {dev_wer:.2f}% (CTC Head 1 — primary metric)")
        for lc, lname in [("hi", "Hindi"), ("mr", "Marathi"), ("en", "English")]:
            if lang_wers.get(lc) is not None:
                print(f"  │   {lname}: {lang_wers[lc]:.2f}%")
        print(f"  │ MSE:  {dev_mse:.4f} (word-level — secondary metric)")
        print(f"  └──────────────────────────────────────────────────────────────")

        # Show some predictions
        if dev_preds:
            n_show = min(3, len(dev_preds))
            print(f"\n  Sample predictions:")
            for ref, hyp in dev_preds[:n_show]:
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
                print()

    # Save stats
    epoch_stats.append({
        "epoch": epoch, "alpha": alpha,
        "train_mse": avg_mse, "train_ctc": avg_ctc, "train_combined": avg_combined,
        "dev_wer": dev_wer, "dev_mse": dev_mse, "avg_words": avg_words,
    })

    # ── Checkpoint ───────────────────────────────────────────────────────
    if not args.verify:
        ckpt_data = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_mse": avg_mse,
            "train_ctc": avg_ctc,
            "train_combined": avg_combined,
            "dev_wer": dev_wer,
            "dev_mse": dev_mse,
            "lang_wers": lang_wers,
            "alpha": alpha,
            "vocab_size": vocab_size,
            "global_step": global_step,
            "args": vars(args),
        }

        # Best dev WER (primary metric)
        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            patience_counter = 0
            torch.save(ckpt_data, CKPT_DIR / "best_dev.pt")
            print(f"  * NEW BEST DEV WER: {dev_wer:.2f}% -> saved best_dev.pt")
        else:
            patience_counter += 1
            print(f"  Early stopping: {patience_counter}/{args.patience} "
                  f"(best: {best_dev_wer:.2f}%)")
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n  EARLY STOPPING at epoch {epoch}")
                # Save final epoch before stopping
                torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch}.pt")
                break

        # Save every epoch
        torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch}.pt")
        print(f"  Saved: {CKPT_DIR / f'epoch_{epoch}.pt'}")

    print()

total_time = time.time() - training_start


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Final results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 8: Results")
print("=" * 70)

if epoch_stats:
    print(f"\n  {'Epoch':<8} {'a':>4} {'MSE':>8} {'CTC':>8} {'Comb':>8} "
          f"{'DevWER':>10} {'DevMSE':>8} {'Words':>6}")
    print(f"  {'─'*8} {'─'*4} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*10} {'─'*8} {'─'*6}")
    for s in epoch_stats:
        wer_str = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        print(f"  {s['epoch']:<8} {s['alpha']:>4.1f} {s['train_mse']:>8.4f} "
              f"{s['train_ctc']:>8.4f} {s['train_combined']:>8.4f} "
              f"{wer_str:>10} {s['dev_mse']:>8.4f} {s['avg_words']:>6.1f}")

    # Learning check
    if len(epoch_stats) >= 2:
        first_comb = epoch_stats[0]["train_combined"]
        last_comb = epoch_stats[-1]["train_combined"]
        if first_comb > 0:
            reduction = (first_comb - last_comb) / first_comb * 100
            print(f"\n  Combined loss reduction: {reduction:.1f}% "
                  f"({first_comb:.4f} -> {last_comb:.4f})")
            if reduction > 0:
                print(f"  LEARNING CONFIRMED")
            else:
                print(f"  WARNING — loss not decreasing")

    if best_dev_wer < 999:
        print(f"\n  Best dev WER: {best_dev_wer:.2f}%")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
print(f"  MuRIL cross-modal training complete.")
print(f"{'=' * 70}")
