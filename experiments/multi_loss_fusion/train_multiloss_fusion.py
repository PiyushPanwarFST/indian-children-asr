"""
Multi-Loss Dual-Encoder Fusion Training with PRE-TRAINED Encoders
=================================================================

This script trains a dual-encoder fusion model using the FULL multi-loss objective
(CTC + MSE acoustic + MSE semantic + orthogonality) starting from PRE-TRAINED
encoder checkpoints. It supports two fusion strategies via --fusion_type:

  1. "gated"           -- GatedFusion: per-frame sigmoid gating (fast, ~1.2M params)
  2. "cross_attention"  -- CrossAttentionFusion: bidirectional cross-attention (~3.5M params)

HOW THIS DIFFERS FROM EXISTING SCRIPTS:
  - experiments/end_to_end/e2e_training.py:
      Same multi-loss but starts from VANILLA Whisper Small (no pre-training).
  - experiments/cross_attention_fusion/cross_attention_training.py:
      Starts from pre-trained encoders but uses CTC-ONLY loss (no KD).
  - THIS SCRIPT:
      Pre-trained encoders + multi-loss = best of both worlds.
      Encoders start with KD-learned representations and continue learning
      with KD supervision, preventing catastrophic forgetting.

ARCHITECTURE:
    +----------------------------------------------------------------------+
    |                                                                      |
    |  INPUT:                                                              |
    |    Raw audio (.mp3/.wav) -> resample to 16kHz mono                   |
    |    -> WhisperFeatureExtractor -> 80-bin log-Mel spectrogram (1,80,3000)
    |    -> SpecAugment: 2 freq masks (width<=15), 2 time masks (width<=50)|
    |                                                                      |
    |  DUAL ENCODERS (both UNFROZEN, from pre-trained KD checkpoints):     |
    |    Encoder A (acoustic, 768-dim):                                    |
    |      - From: checkpoints/joint/joint_best_wer.pt                     |
    |      - Pre-trained via Kid-Whisper KD + CTC joint training           |
    |      - Input: mel (1, 80, 3000) -> Output: feat_a (1, T, 768)       |
    |                                                                      |
    |    Encoder B (semantic, 768-dim):                                    |
    |      - From: checkpoints/semantic_ctc_unfrozen/best_dev.pt           |
    |      - Pre-trained via IndicConformer MSE distillation               |
    |      - Input: mel (1, 80, 3000) -> Output: feat_b (1, T, 768)       |
    |                                                                      |
    |  FUSION MODULE (--fusion_type gated|cross_attention):                |
    |    GatedFusion:                                                      |
    |      gate = sigmoid(W_g . [feat_a; feat_b])    (1, T, 768)          |
    |      fused = gate * feat_a + (1-gate) * feat_b  (1, T, 768)         |
    |      -> LayerNorm -> Dropout                                         |
    |      Returns: (fused, gate)  -- gate used for monitoring             |
    |                                                                      |
    |    CrossAttentionFusion:                                             |
    |      Dir1: acoustic queries semantic (Q=feat_a, K,V=feat_b)          |
    |        enriched_a = feat_a + softmax(Q@K^T/sqrt(768)) @ V            |
    |      Dir2: semantic queries acoustic (Q=feat_b, K,V=feat_a)          |
    |        enriched_b = feat_b + softmax(Q@K^T/sqrt(768)) @ V            |
    |      fused = LayerNorm(Dropout(enriched_a + enriched_b))             |
    |      Returns: fused  -- no gate stats                                |
    |                                                                      |
    |  LOSS HEADS:                                                         |
    |    1. CTC Head: Linear(768->85) on fused features                    |
    |       -> CTC loss vs ground truth characters                         |
    |    2. Projection: Linear(768->1024) on Encoder A features            |
    |       -> MSE loss vs Kid-Whisper teacher encoder (frozen, online)     |
    |    3. Bridge heads: Linear(768->257) on Encoder B features           |
    |       -> MSE loss vs IndicConformer logits (pre-computed, per-lang)   |
    |    4. Orthogonality: |cos_sim(mean(feat_a), mean(feat_b))|           |
    |       -> Encourages encoder specialization                           |
    |                                                                      |
    |  TOTAL LOSS:                                                         |
    |    L = w_ctc * CTC + w_acou * MSE_acoustic + w_sem * MSE_semantic    |
    |        + w_ortho * orthogonality                                     |
    |                                                                      |
    |  LOSS WEIGHT SCHEDULE (unless --fixed_weights):                      |
    |    Phase 1 (epochs 0-25%):   KD-heavy   (CTC=0.3, Acou=0.5, Sem=0.3)|
    |    Phase 2 (epochs 25-62%):  Balanced    (CTC=0.5, Acou=0.3, Sem=0.2)|
    |    Phase 3 (epochs 62-100%): CTC-heavy   (CTC=0.7, Acou=0.2, Sem=0.1)|
    |                                                                      |
    |  OPTIMIZER:                                                          |
    |    AdamW with differential LR:                                       |
    |      - Encoders: --encoder_lr (default 1e-5)                         |
    |      - Everything else: --fusion_lr (default 1e-3)                   |
    |    Scheduler: linear warmup -> cosine decay with floor                |
    +----------------------------------------------------------------------+

Usage:
    # Quick verify (100 clips, 2 epochs, gated fusion):
    python train_multiloss_fusion.py --fusion_type gated --verify

    # Test run (3000 clips, 10 epochs, cross-attention):
    python train_multiloss_fusion.py --fusion_type cross_attention --test_clips 3000 --epochs 10

    # Full gated training (30 epochs):
    python train_multiloss_fusion.py --fusion_type gated --epochs 30 --patience 10

    # Full cross-attention training (30 epochs):
    python train_multiloss_fusion.py --fusion_type cross_attention --epochs 30 --patience 10

    # Resume from checkpoint:
    python train_multiloss_fusion.py --fusion_type gated --resume checkpoints/multiloss_gated/epoch_5.pt
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm as _tqdm
# Force tqdm to stdout so it appears in PBS log files (PBS captures stdout, not stderr)
tqdm = partial(_tqdm, file=sys.stdout)


# ==============================================================================
# ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(
    description="Multi-Loss Dual-Encoder Fusion Training with Pre-trained Encoders"
)

# Fusion type — REQUIRED
parser.add_argument("--fusion_type", type=str, required=True,
                    choices=["gated", "cross_attention"],
                    help="Fusion strategy: 'gated' (GatedFusion) or 'cross_attention' (CrossAttentionFusion)")

# Training hyperparameters
parser.add_argument("--epochs", type=int, default=30,
                    help="Number of training epochs (default: 30)")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience on dev WER; 0 = disabled (default: 10)")
parser.add_argument("--encoder_lr", type=float, default=1e-5,
                    help="Learning rate for BOTH encoder param groups (default: 1e-5)")
parser.add_argument("--fusion_lr", type=float, default=1e-3,
                    help="Learning rate for fusion/CTC/projection/bridge heads (default: 1e-3)")
parser.add_argument("--warmup_steps", type=int, default=500,
                    help="Linear warmup steps before cosine decay (default: 500)")
parser.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps; effective batch = grad_accum (default: 4)")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout rate in fusion module (default: 0.1)")
parser.add_argument("--max_audio_sec", type=float, default=30.0,
                    help="Max audio length in seconds; longer clips truncated (default: 30.0)")

# Loss weights (initial values -- can be scheduled during training)
parser.add_argument("--w_ctc", type=float, default=0.3,
                    help="Initial CTC loss weight (default: 0.3)")
parser.add_argument("--w_acoustic", type=float, default=0.5,
                    help="Initial acoustic KD (MSE) weight (default: 0.5)")
parser.add_argument("--w_semantic", type=float, default=0.3,
                    help="Initial semantic KD (MSE) weight (default: 0.3)")
parser.add_argument("--w_ortho", type=float, default=0.01,
                    help="Orthogonality loss weight, constant (default: 0.01)")
parser.add_argument("--fixed_weights", action="store_true",
                    help="Disable loss weight schedule -- use --w_ctc/w_acoustic/w_semantic as constants")

# Encoder checkpoints (pre-trained via KD)
parser.add_argument("--acoustic_ckpt", type=str,
                    default="checkpoints/joint/joint_best_wer.pt",
                    help="Acoustic encoder checkpoint (has encoder_state_dict + ctc_head_state_dict)")
parser.add_argument("--semantic_ckpt", type=str,
                    default="checkpoints/semantic_ctc_unfrozen/best_dev.pt",
                    help="Semantic encoder checkpoint (has encoder_state_dict)")

# Teacher models for KD losses
parser.add_argument("--teacher_model", type=str,
                    default="aadel4/kid-whisper-medium-en-myst",
                    help="Kid-Whisper teacher model for acoustic MSE (online, frozen)")
parser.add_argument("--teacher_logits_dir", type=str, default="teacher_logits",
                    help="Directory with pre-computed IndicConformer logits for semantic MSE")

# SpecAugment
parser.add_argument("--specaugment", action="store_true", default=True,
                    help="Apply SpecAugment data augmentation (default: on)")
parser.add_argument("--spec_freq_mask", type=int, default=15,
                    help="Max frequency mask width for SpecAugment (default: 15)")
parser.add_argument("--spec_time_mask", type=int, default=50,
                    help="Max time mask width for SpecAugment (default: 50)")

# Modes
parser.add_argument("--verify", action="store_true",
                    help="Quick verification: 100 clips, 2 epochs, no checkpointing")
parser.add_argument("--test_clips", type=int, default=None,
                    help="Limit training to N clips (for test runs)")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path (relative to project root)")
parser.add_argument("--fresh_optimizer", action="store_true",
                    help="When resuming, create fresh optimizer+scheduler (warm restart LR)")
parser.add_argument("--grad_checkpoint", action="store_true",
                    help="Enable gradient checkpointing (saves ~40%% GPU memory)")
parser.add_argument("--lr_floor", type=float, default=0.03,
                    help="Minimum LR as fraction of peak for cosine floor (default: 0.03)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")

args = parser.parse_args()

# In verify mode, limit epochs and clips
if args.verify:
    args.epochs = min(args.epochs, 2)
    if args.test_clips is None:
        args.test_clips = 100


# ==============================================================================
# PATHS AND CONSTANTS
# ==============================================================================
# this file -> multi_loss_fusion/ -> experiments/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
TEACHER_LOGITS_DIR = PROJECT_ROOT / args.teacher_logits_dir

# Output directory depends on fusion type
if args.fusion_type == "gated":
    CKPT_DIR = PROJECT_ROOT / "checkpoints" / "multiloss_gated"
else:
    CKPT_DIR = PROJECT_ROOT / "checkpoints" / "multiloss_cross_attn"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

# Add project root to path so we can import utils
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
print(f"Fusion type: {args.fusion_type}")
print(f"Checkpoint dir: {CKPT_DIR}")


# ==============================================================================
# STEP 1: Load character vocabulary (85 tokens)
# ==============================================================================
# The vocabulary maps each character (Devanagari + English + special tokens)
# to an integer index used by the CTC loss and decoding.
print("\n" + "=" * 70)
print("STEP 1: Loading character vocabulary")
print("=" * 70)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)  # 85 tokens: 3 special + 22 English + 60 Devanagari
BLANK_IDX = char_to_idx["<blank>"]  # 0 -- CTC blank token

print(f"  Vocab: {vocab_size} tokens (from {VOCAB_PATH.name})")
print(f"  Blank index: {BLANK_IDX}")


# ==============================================================================
# STEP 1b: Helper functions
# ==============================================================================
# These utility functions are shared across all training scripts in the project.
# They handle audio loading, mel feature computation, SpecAugment, CTC encoding/
# decoding, teacher logit alignment, and loss weight scheduling.


def text_to_indices(text, char_to_idx):
    """
    Convert normalized text string to a list of vocabulary indices for CTC targets.

    Each character is mapped to its index in the vocabulary. Spaces become the
    <space> token. Unknown characters become <unk>.

    Args:
        text (str): Normalized ground truth text (e.g., "raadha ke paas")
        char_to_idx (dict): Character-to-index mapping from vocab.json

    Returns:
        list[int]: Token indices, e.g., [25, 42, 1, 30, ...]
    """
    indices = []
    for ch in text:
        if ch == " ":
            indices.append(char_to_idx["<space>"])
        elif ch in char_to_idx:
            indices.append(char_to_idx[ch])
        else:
            indices.append(char_to_idx["<unk>"])
    return indices


def ctc_greedy_decode(logits, idx_to_char):
    """
    Greedy CTC decoding: argmax at each frame -> collapse repeats -> remove blanks.

    This is the simplest decoding strategy. For better results, use beam search
    with a language model (see scripts/utils/ctc_decode.py).

    Args:
        logits (Tensor): Shape (T, vocab_size) -- raw logits from CTC head
        idx_to_char (dict): Index-to-character mapping (reverse of vocab)

    Returns:
        str: Decoded text string
    """
    indices = torch.argmax(logits, dim=-1)           # (T,) -- best token per frame
    collapsed = torch.unique_consecutive(indices)     # Remove repeated tokens
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


def load_audio(path, max_sec=None):
    """
    Load an audio file and convert to 16kHz mono waveform tensor.

    Handles resampling from any sample rate and stereo-to-mono conversion.
    Optionally truncates to max_sec seconds to prevent OOM on long clips.

    Args:
        path (str): Path to audio file (.mp3 or .wav)
        max_sec (float, optional): Maximum duration in seconds

    Returns:
        Tensor: 1D waveform tensor, shape (num_samples,), at 16kHz
    """
    import torchaudio
    wav, sr = torchaudio.load(path)
    # Resample to 16kHz if needed (Whisper requirement)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # Convert stereo to mono by averaging channels
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()                              # (num_samples,)
    # Truncate if too long
    if max_sec and len(wav) > int(max_sec * SAMPLE_RATE):
        wav = wav[:int(max_sec * SAMPLE_RATE)]
    return wav


def compute_real_frames(num_samples):
    """
    Compute the number of real encoder output frames for a given audio length.

    Whisper's mel spectrogram uses hop_length=160 samples, and the two conv layers
    further downsample by 2x, giving: frames = num_samples // 160 // 2.
    This is capped at 1500 (Whisper's max sequence length for 30s audio).

    Example: 10s audio = 160,000 samples -> 160000 // 160 // 2 = 500 frames.

    Args:
        num_samples (int): Number of audio samples at 16kHz

    Returns:
        int: Number of encoder output frames (max 1500)
    """
    return min(num_samples // 160 // 2, 1500)


def apply_spec_augment(mel_features):
    """
    Apply SpecAugment data augmentation to mel spectrogram features.

    SpecAugment (Park et al., 2019) randomly masks frequency bands and time
    steps to improve generalization. The same augmented mel is fed to BOTH
    encoders so they see identical input.

    Config: 2 freq masks (width from --spec_freq_mask), 2 time masks (from --spec_time_mask)

    Args:
        mel_features (Tensor): Shape (1, 80, 3000) -- mel spectrogram

    Returns:
        Tensor: Shape (1, 80, 3000) -- augmented mel (modified in-place)
    """
    _, n_freq, n_time = mel_features.shape           # (1, 80, 3000)
    freq_mask = args.spec_freq_mask
    time_mask = args.spec_time_mask

    # Frequency masking: zero out random frequency bands
    for _ in range(2):
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, max(n_freq - f, 1) - 1)
        mel_features[:, f0:f0 + f, :] = 0.0

    # Time masking: zero out random time segments
    for _ in range(2):
        t = random.randint(0, time_mask)
        t0 = random.randint(0, max(n_time - t, 1) - 1)
        mel_features[:, :, t0:t0 + t] = 0.0

    return mel_features


def align_teacher_to_student(teacher_logits, student_frames):
    """
    Interpolate pre-computed teacher logits to match student encoder frame count.

    The IndicConformer teacher and Whisper student have different frame rates,
    so we use linear interpolation to align them temporally. This is necessary
    for computing the semantic MSE loss between student bridge-head output and
    teacher logits.

    Args:
        teacher_logits (Tensor): Shape (T_teacher, 257) -- IndicConformer output logits
        student_frames (int): Number of student encoder frames (T_student)

    Returns:
        Tensor: Shape (1, T_student, 257) -- aligned teacher logits
    """
    # Transpose to (1, 257, T_teacher) for F.interpolate (expects channel-first)
    t = teacher_logits.T.unsqueeze(0)                # (1, 257, T_teacher)
    aligned = F.interpolate(t, size=student_frames, mode='linear', align_corners=False)
    return aligned.permute(0, 2, 1)                  # (1, T_student, 257)


def get_loss_weights(epoch, total_epochs):
    """
    Compute scheduled loss weights based on training progress.

    The schedule transitions from KD-heavy (early) to CTC-heavy (late):
      - Phase 1 (0-25% of training):   w_ctc=0.3, w_acou=0.5, w_sem=0.3
        -> Focus on learning good representations from teachers
      - Phase 2 (25-62.5%):            w_ctc=0.5, w_acou=0.3, w_sem=0.2
        -> Balance KD with CTC task loss
      - Phase 3 (62.5-100%):           w_ctc=0.7, w_acou=0.2, w_sem=0.1
        -> Focus on the actual ASR objective

    The orthogonality weight (w_ortho) is kept constant throughout.

    If --fixed_weights is set, always returns the CLI values (useful for
    resuming runs where the schedule would restart incorrectly).

    Args:
        epoch (int): Current epoch number
        total_epochs (int): Total number of epochs

    Returns:
        tuple: (w_ctc, w_acoustic, w_semantic) -- scheduled loss weights
    """
    if args.fixed_weights:
        return args.w_ctc, args.w_acoustic, args.w_semantic

    progress = epoch / total_epochs
    if progress <= 0.25:
        # Phase 1: KD-heavy -- teachers guide encoder specialization
        return args.w_ctc, args.w_acoustic, args.w_semantic
    elif progress <= 0.625:
        # Phase 2: Balanced -- transitioning to task loss
        return 0.5, 0.3, 0.2
    else:
        # Phase 3: CTC-heavy -- fine-tune for ASR performance
        return 0.7, 0.2, 0.1


# ==============================================================================
# STEP 2: Load training and dev data
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Loading training data")
print("=" * 70)


def load_clips_from_csv(csv_path, split="train", max_clips=None):
    """
    Load clip metadata from CSV. For Hindi/Marathi, also resolves teacher logit paths.

    Each clip gets:
      - audio_path: absolute path to the audio file
      - lang_code: "hi", "mr", or "en"
      - logit_path: path to pre-computed IndicConformer logits (Hindi/Marathi only)
      - ground_truth: normalized transcript text

    Args:
        csv_path (str or Path): Path to asr_train.csv or asr_dev.csv
        split (str): "train" or "dev" -- determines logit subdirectory
        max_clips (int, optional): Only return first N clips

    Returns:
        tuple: (clips_list, num_skipped)
    """
    clips = []
    skipped = 0
    skipped_no_logits = 0

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

            # Create unique clip ID from child_id + filename
            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            # Teacher logits for IndicConformer KD (Hindi/Marathi only)
            # These are pre-computed and stored as .pt files
            logit_path = None
            if lang_code in ("hi", "mr"):
                lp = TEACHER_LOGITS_DIR / split / f"{clip_uid}.pt"
                if lp.exists():
                    logit_path = str(lp)
                else:
                    # Don't skip -- clip still benefits from acoustic KD + CTC
                    pass

            # Ground truth text (try multiple column names for CSV compatibility)
            gt = row.get("transcript", row.get("que_text", row.get("text", "")))

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "clip_name": clip_uid,
                "logit_path": logit_path,
                "duration_sec": float(row.get("duration_sec", 0)),
                "ground_truth": gt,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips, skipped


# Load training and dev sets
train_clips, skip_other = load_clips_from_csv(TRAIN_CSV, split="train")
hi_clips = [c for c in train_clips if c["lang_code"] == "hi"]
mr_clips = [c for c in train_clips if c["lang_code"] == "mr"]
en_clips = [c for c in train_clips if c["lang_code"] == "en"]
logit_clips = sum(1 for c in train_clips if c["logit_path"] is not None)
total_hrs = sum(c["duration_sec"] for c in train_clips) / 3600

print(f"  Total: {len(train_clips)} clips ({total_hrs:.1f}h)")
print(f"    Hindi:   {len(hi_clips)} clips")
print(f"    Marathi: {len(mr_clips)} clips")
print(f"    English: {len(en_clips)} clips")
print(f"  Teacher logits available: {logit_clips}/{len(hi_clips)+len(mr_clips)} (Hi+Mr)")
print(f"  Skipped: {skip_other} unknown language")

if len(train_clips) == 0:
    print("\n  ERROR: No training clips found!")
    sys.exit(1)

dev_clips, _ = load_clips_from_csv(DEV_CSV, split="dev")
print(f"  Dev set: {len(dev_clips)} clips")

# -- Optionally limit clips for test/verify runs --
N = args.test_clips
if N and N < len(train_clips):
    random.seed(args.seed)
    total_all = len(train_clips)
    lang_clips = {"hi": hi_clips, "mr": mr_clips, "en": en_clips}
    sampled = []
    remaining = N

    # Proportional sampling: maintain language ratio from full dataset
    for lc, lclips in lang_clips.items():
        n_lang = min(int(N * len(lclips) / total_all), len(lclips))
        if lclips:
            sampled.extend(random.sample(lclips, n_lang))
            remaining -= n_lang
    if remaining > 0:
        leftover = [c for c in train_clips if c not in sampled]
        sampled.extend(random.sample(leftover, min(remaining, len(leftover))))

    train_clips = sampled
    random.shuffle(train_clips)
    hi_n = sum(1 for c in train_clips if c["lang_code"] == "hi")
    mr_n = sum(1 for c in train_clips if c["lang_code"] == "mr")
    en_n = sum(1 for c in train_clips if c["lang_code"] == "en")
    label = "VERIFY MODE" if args.verify else "TEST RUN"
    print(f"\n  {label}: {len(train_clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

    if args.verify and dev_clips:
        dev_n = min(100, len(dev_clips))
        random.seed(args.seed + 1)
        dev_clips = random.sample(dev_clips, dev_n)
        print(f"    Dev subset: {len(dev_clips)} clips")

print()


# ==============================================================================
# STEP 3: Initialize models from PRE-TRAINED checkpoints
# ==============================================================================
print("=" * 70)
print("STEP 3: Initializing models (pre-trained encoders + multi-loss heads)")
print("=" * 70)

import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration

WHISPER_ID = "openai/whisper-small"

# Feature extractor: converts raw audio waveform -> 80-bin log-Mel spectrogram
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)


# -- Helper to load a pre-trained encoder from checkpoint --
def load_encoder(checkpoint_path, label):
    """
    Load a Whisper Small encoder and restore pre-trained weights from checkpoint.

    Creates a fresh Whisper Small model, extracts the encoder, then loads our
    KD-trained weights. The encoder is set to train mode (unfrozen) so gradients
    flow through during fine-tuning.

    Handles two checkpoint key formats:
      - 'encoder_state_dict': Direct encoder weights (from joint/semantic training)
      - 'model_state_dict': Full model with 'encoder.' prefix (from MSE training)

    Args:
        checkpoint_path (str): Relative path from PROJECT_ROOT to .pt checkpoint
        label (str): Display name for logging (e.g., "ENCODER A (acoustic)")

    Returns:
        nn.Module: Whisper encoder on DEVICE, in train mode, with restored weights
    """
    print(f"\n  [{label}] Loading Whisper Small encoder...")

    # Load fresh Whisper Small, extract encoder
    whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
    encoder = whisper_model.encoder.to(DEVICE)

    # Resolve checkpoint path (try alternates if primary not found)
    ckpt_path = PROJECT_ROOT / checkpoint_path
    if not ckpt_path.exists():
        alt_paths = [
            PROJECT_ROOT / "checkpoints" / "semantic" / "best_dev_model.pt",
            PROJECT_ROOT / "checkpoints" / "semantic_ctc_unfrozen" / "best_dev.pt",
            PROJECT_ROOT / "checkpoints" / "joint" / "joint_best_wer.pt",
        ]
        for alt in alt_paths:
            if alt.exists():
                if "semantic" in checkpoint_path.lower() and "semantic" in str(alt).lower():
                    ckpt_path = alt
                    break
                elif "acoustic" in label.lower() and "joint" in str(alt).lower():
                    ckpt_path = alt
                    break
        if not ckpt_path.exists():
            print(f"    ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)

    print(f"    Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Load weights -- handle different checkpoint key formats
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    elif "model_state_dict" in ckpt:
        state = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("encoder."):
                state[k[len("encoder."):]] = v
        encoder.load_state_dict(state)
    else:
        print(f"    ERROR: Unknown checkpoint format. Keys: {list(ckpt.keys())[:5]}")
        sys.exit(1)

    epoch_info = ckpt.get("epoch", "?")
    print(f"    Loaded from epoch {epoch_info}")

    # Set to train mode (unfrozen) -- gradients will flow through all layers
    encoder.train()

    # Gradient checkpointing: recomputes activations during backward pass
    # instead of storing them. Saves ~40% GPU memory at ~20% speed cost.
    if args.grad_checkpoint:
        encoder.gradient_checkpointing_enable()
        print(f"    Gradient checkpointing ENABLED")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"    UNFROZEN ({n_params:,} params, lr={args.encoder_lr})")

    del whisper_model, ckpt
    gc.collect()
    return encoder


# -- Load Encoder A (acoustic branch, from Kid-Whisper KD + CTC joint training) --
encoder_a = load_encoder(args.acoustic_ckpt, "ENCODER A (acoustic)")
n_params_a = sum(p.numel() for p in encoder_a.parameters())

# -- Load Encoder B (semantic branch, from IndicConformer MSE distillation) --
encoder_b = load_encoder(args.semantic_ckpt, "ENCODER B (semantic)")
n_params_b = sum(p.numel() for p in encoder_b.parameters())


# -- Kid-Whisper Teacher (frozen, online) --
# This teacher provides acoustic targets for MSE loss. It runs online
# (forward pass each clip) because its encoder output is high-dimensional
# (1024-dim) and storing all outputs would be too large.
print(f"\n  [TEACHER] Loading Kid-Whisper: {args.teacher_model}")
teacher_full = WhisperForConditionalGeneration.from_pretrained(args.teacher_model)
teacher_encoder = teacher_full.model.encoder.to(DEVICE)
teacher_dim = teacher_full.config.d_model  # 1024 for Kid-Whisper Medium
teacher_encoder.eval()
for param in teacher_encoder.parameters():
    param.requires_grad = False  # FROZEN -- never updated
teacher_params = sum(p.numel() for p in teacher_encoder.parameters())
print(f"    Teacher dim: {teacher_dim}")
print(f"    Teacher params: {teacher_params:,} (all FROZEN)")
# Free decoder memory -- we only need the encoder
del teacher_full.proj_out
del teacher_full.model.decoder
del teacher_full
gc.collect()


# -- Projection head: student (768) -> teacher (1024) for acoustic MSE --
# This linear layer maps Encoder A's 768-dim features to the teacher's 1024-dim
# space so we can compute MSE between student and teacher representations.
print(f"\n  [PROJECTION] Linear(768 -> {teacher_dim})")
projection_a = nn.Linear(768, teacher_dim).to(DEVICE)

# -- Bridge heads: student (768) -> IndicConformer vocab (257) for semantic MSE --
# These project Encoder B's features to IndicConformer's output space (257 classes).
# Separate heads for Hindi and Marathi because the teacher was trained on each
# language independently. English clips skip this loss component.
print(f"  [BRIDGE] Linear(768 -> 257) x 2 (Hindi + Marathi)")
bridge_hi = nn.Linear(768, 257).to(DEVICE)
bridge_mr = nn.Linear(768, 257).to(DEVICE)


# ==============================================================================
# Fusion Modules
# ==============================================================================

class GatedFusion(nn.Module):
    """
    Learnable gated fusion: per-frame, per-dimension sigmoid gating.

    For each time frame t and dimension d, the gate learns a weight in [0, 1]:
      gate[t,d] = sigmoid(W_g . [feat_a[t,:]; feat_b[t,:]] + b_g)[d]
      fused[t,d] = gate[t,d] * feat_a[t,d] + (1 - gate[t,d]) * feat_b[t,d]

    gate ~ 1 means "trust Encoder A (acoustic)" at this frame/dimension.
    gate ~ 0 means "trust Encoder B (semantic)" at this frame/dimension.

    Monitoring the gate mean tells us if one encoder dominates. A healthy
    gate mean is around 0.4-0.6. If it's < 0.1 or > 0.9, one encoder
    is being ignored (potential problem).

    Parameters:
      gate_linear: Linear(1536, 768) = 1,180,416 params
      layer_norm:  LayerNorm(768)     = 1,536 params
      Total: ~1.18M params

    Args:
        dim (int): Feature dimension (768 for Whisper Small)
        dropout (float): Dropout rate after LayerNorm
    """
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)   # (1536 -> 768)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_a, feat_s):
        """
        Args:
            feat_a: (B, T, 768) -- acoustic encoder output
            feat_s: (B, T, 768) -- semantic encoder output

        Returns:
            tuple: (fused, gate) where:
              fused: (B, T, 768) -- fused features
              gate:  (B, T, 768) -- gate values for monitoring
        """
        concat = torch.cat([feat_a, feat_s], dim=-1)  # (B, T, 1536)
        gate = torch.sigmoid(self.gate_linear(concat)) # (B, T, 768), values in [0,1]
        fused = gate * feat_a + (1 - gate) * feat_s    # (B, T, 768)
        fused = self.layer_norm(fused)                  # (B, T, 768)
        fused = self.dropout(fused)                     # (B, T, 768)
        return fused, gate  # Return gate for monitoring


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion for dual-encoder ASR.

    Unlike GatedFusion which operates frame-by-frame (frame i only sees frame i),
    cross-attention lets each encoder attend to ALL frames of the other encoder.
    This enables temporal reasoning: e.g., the acoustic encoder's representation
    of a child's hesitation at frame 40 can incorporate semantic context from
    the Hindi phoneme sequence at frames 10-20.

    Architecture:
        Direction 1 (acoustic queries semantic):
            Q = feat_a @ W_Q1            (B, T, 768)
            K = feat_s @ W_K1            (B, T, 768)
            V = feat_s @ W_V1            (B, T, 768)
            attn = softmax(Q @ K^T / sqrt(768)) @ V    (B, T, 768)
            enriched_a = feat_a + attn   (residual connection)

        Direction 2 (semantic queries acoustic):
            Q = feat_s @ W_Q2            (B, T, 768)
            K = feat_a @ W_K2            (B, T, 768)
            V = feat_a @ W_V2            (B, T, 768)
            attn = softmax(Q @ K^T / sqrt(768)) @ V    (B, T, 768)
            enriched_s = feat_s + attn   (residual connection)

        Combine:
            combined = enriched_a + enriched_s    (B, T, 768)
            output = Dropout(LayerNorm(combined)) (B, T, 768)

    Parameters:
        6 x Linear(768, 768) = 6 x 590,592 = 3,543,552
        LayerNorm(768) = 1,536
        Total: ~3.5M params
    """

    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5    # 1/sqrt(768) = 0.0361

        # Direction 1: acoustic attends to semantic
        # "What semantic context is relevant for each acoustic frame?"
        self.W_Q1 = nn.Linear(dim, dim)    # (768 -> 768) queries from acoustic
        self.W_K1 = nn.Linear(dim, dim)    # (768 -> 768) keys from semantic
        self.W_V1 = nn.Linear(dim, dim)    # (768 -> 768) values from semantic

        # Direction 2: semantic attends to acoustic
        # "What acoustic evidence supports each semantic frame?"
        self.W_Q2 = nn.Linear(dim, dim)    # (768 -> 768) queries from semantic
        self.W_K2 = nn.Linear(dim, dim)    # (768 -> 768) keys from acoustic
        self.W_V2 = nn.Linear(dim, dim)    # (768 -> 768) values from acoustic

        # Output normalization and regularization
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_a, feat_s):
        """
        Bidirectional cross-attention fusion.

        Args:
            feat_a: (B, T, 768) -- acoustic encoder output
            feat_s: (B, T, 768) -- semantic encoder output

        Returns:
            (B, T, 768) -- fused features (no gate stats for cross-attention)
        """
        # -- Direction 1: acoustic queries semantic --
        Q1 = self.W_Q1(feat_a)          # (B, T, 768)
        K1 = self.W_K1(feat_s)          # (B, T, 768)
        V1 = self.W_V1(feat_s)          # (B, T, 768)

        # Scaled dot-product attention: (B, T, T) attention matrix
        # At T=1500: 1500*1500*4 bytes = 9 MB per sample (manageable on V100-32GB)
        attn_weights_1 = torch.matmul(Q1, K1.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn_weights_1 = F.softmax(attn_weights_1, dim=-1)                     # (B, T, T)
        attn_out_1 = torch.matmul(attn_weights_1, V1)                          # (B, T, 768)

        # Residual connection: keep original + add cross-encoder context
        enriched_a = feat_a + attn_out_1    # (B, T, 768)

        # -- Direction 2: semantic queries acoustic --
        Q2 = self.W_Q2(feat_s)          # (B, T, 768)
        K2 = self.W_K2(feat_a)          # (B, T, 768)
        V2 = self.W_V2(feat_a)          # (B, T, 768)

        attn_weights_2 = torch.matmul(Q2, K2.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn_weights_2 = F.softmax(attn_weights_2, dim=-1)                     # (B, T, T)
        attn_out_2 = torch.matmul(attn_weights_2, V2)                          # (B, T, 768)

        enriched_s = feat_s + attn_out_2    # (B, T, 768)

        # -- Combine: both enriched representations contain full info --
        combined = enriched_a + enriched_s  # (B, T, 768)
        combined = self.layer_norm(combined) # (B, T, 768)
        combined = self.dropout(combined)    # (B, T, 768)

        return combined  # No gate stats for cross-attention


# -- Create the fusion module based on --fusion_type --
if args.fusion_type == "gated":
    print(f"\n  [GATED FUSION] Creating fusion module...")
    fusion = GatedFusion(dim=768, dropout=args.dropout).to(DEVICE)
elif args.fusion_type == "cross_attention":
    print(f"\n  [CROSS-ATTENTION FUSION] Creating fusion module...")
    fusion = CrossAttentionFusion(dim=768, dropout=args.dropout).to(DEVICE)

fusion_params = sum(p.numel() for p in fusion.parameters())
print(f"    Params: {fusion_params:,}")

# -- CTC Head: fused features -> character logits --
print(f"\n  [CTC HEAD] Linear(768 -> {vocab_size})")
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# -- Warm-start CTC head from acoustic checkpoint --
# The acoustic branch CTC head already knows the char->CTC mapping from joint
# training, so reusing it gives a much better starting point than random init.
ctc_ckpt_path = PROJECT_ROOT / args.acoustic_ckpt
if ctc_ckpt_path.exists() and not args.resume:
    print(f"    CTC head warm-start from: {ctc_ckpt_path}")
    ctc_ckpt = torch.load(ctc_ckpt_path, map_location=DEVICE, weights_only=False)
    if "ctc_head_state_dict" in ctc_ckpt:
        ctc_head.load_state_dict(ctc_ckpt["ctc_head_state_dict"])
        print(f"    CTC head loaded (warm-started)")
    elif "ctc_head_char_state_dict" in ctc_ckpt:
        ctc_head.load_state_dict(ctc_ckpt["ctc_head_char_state_dict"])
        print(f"    CTC head loaded (warm-started, alt key)")
    else:
        print(f"    WARNING: No CTC head found in checkpoint -- random init")
    del ctc_ckpt
    gc.collect()


# -- Resume from checkpoint (loads ALL components) --
start_epoch = 0
best_dev_wer = float("inf")

if args.resume:
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Restore all model components
    encoder_a.load_state_dict(ckpt["acoustic_encoder_state_dict"])
    encoder_b.load_state_dict(ckpt["semantic_encoder_state_dict"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
    projection_a.load_state_dict(ckpt["projection_a_state_dict"])
    bridge_hi.load_state_dict(ckpt["bridge_hi_state_dict"])
    bridge_mr.load_state_dict(ckpt["bridge_mr_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))

    # Verify fusion type matches
    saved_fusion = ckpt.get("fusion_type", "unknown")
    if saved_fusion != args.fusion_type:
        print(f"    WARNING: Checkpoint fusion_type='{saved_fusion}' != args.fusion_type='{args.fusion_type}'")
        print(f"    This may cause errors if architectures differ!")

    print(f"    Resumed from epoch {start_epoch}, best dev WER: {best_dev_wer:.2f}%")
    del ckpt
    gc.collect()


# -- Parameter summary --
proj_params = sum(p.numel() for p in projection_a.parameters())
bridge_params = sum(p.numel() for p in bridge_hi.parameters()) + sum(p.numel() for p in bridge_mr.parameters())
ctc_params = sum(p.numel() for p in ctc_head.parameters())
total_trainable = n_params_a + n_params_b + fusion_params + proj_params + bridge_params + ctc_params

fusion_label = "Gated fusion" if args.fusion_type == "gated" else "Cross-attn fusion"

print(f"\n  +-  Parameter Summary ----------------------------------------")
print(f"  | Encoder A (acoustic): {n_params_a:,} (unfrozen, lr={args.encoder_lr})")
print(f"  | Encoder B (semantic): {n_params_b:,} (unfrozen, lr={args.encoder_lr})")
print(f"  | Projection (768->{teacher_dim}): {proj_params:,} (lr={args.fusion_lr})")
print(f"  | Bridge heads (768->257): {bridge_params:,} (lr={args.fusion_lr})")
print(f"  | {fusion_label}:       {fusion_params:,} (lr={args.fusion_lr})")
print(f"  | CTC head (768->{vocab_size}): {ctc_params:,} (lr={args.fusion_lr})")
print(f"  | Teacher (FROZEN):    {teacher_params:,}")
print(f"  | Total trainable:     {total_trainable:,}")
print(f"  +-------------------------------------------------------------")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  GPU Memory: {allocated:.0f} MB allocated")

print()


# ==============================================================================
# STEP 4: Component verification
# ==============================================================================
# Run one forward + backward pass on a single clip to verify that:
#   1. All model components produce correct output shapes
#   2. Gradients flow through all trainable parameters
#   3. All loss terms can be computed
print("=" * 70)
print("STEP 4: Component verification (forward + backward on 1 clip)")
print("=" * 70)

test_clip = train_clips[0]
test_wav = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"  Test: {test_clip['clip_name']} ({len(test_wav)/SAMPLE_RATE:.1f}s, {test_clip['language']})")

# Mel spectrogram + SpecAugment
mel = feat_extractor(test_wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
mel_aug = apply_spec_augment(mel.input_features.clone())
mel_gpu = mel_aug.to(DEVICE)
print(f"  Mel: {tuple(mel.input_features.shape)} -> SpecAugment applied")

# Forward through both student encoders
a_out = encoder_a(mel_gpu).last_hidden_state    # (1, 1500, 768)
b_out = encoder_b(mel_gpu).last_hidden_state    # (1, 1500, 768)
real_frames = compute_real_frames(len(test_wav))
a_feat = a_out[:, :real_frames, :]               # (1, T, 768)
b_feat = b_out[:, :real_frames, :]               # (1, T, 768)
print(f"  Encoder A: {tuple(a_feat.shape)}")
print(f"  Encoder B: {tuple(b_feat.shape)}")

# Teacher forward (frozen, no gradients)
with torch.no_grad():
    t_out = teacher_encoder(mel_gpu).last_hidden_state  # (1, 1500, 1024)
    t_feat = t_out[:, :real_frames, :]                   # (1, T, 1024)
print(f"  Teacher:   {tuple(t_feat.shape)}")

# Projection A -> teacher space for acoustic MSE
proj_a = projection_a(a_feat)                     # (1, T, 1024)
mse_acou = F.mse_loss(proj_a, t_feat)
print(f"  Projection A: {tuple(proj_a.shape)} -> MSE vs teacher: {mse_acou.item():.4f}")

# Bridge head (if Hindi/Marathi with logits)
if test_clip["logit_path"]:
    teacher_data = torch.load(test_clip["logit_path"], weights_only=True)
    t_logits = teacher_data["logits"].float().to(DEVICE)  # (T_teacher, 257)
    t_aligned = align_teacher_to_student(t_logits, real_frames)  # (1, T, 257)
    if test_clip["lang_code"] == "hi":
        b_logits = bridge_hi(b_feat)              # (1, T, 257)
    else:
        b_logits = bridge_mr(b_feat)              # (1, T, 257)
    mse_sem = F.mse_loss(b_logits, t_aligned)
    print(f"  Bridge:    {tuple(b_logits.shape)} -> MSE vs IndicConf: {mse_sem.item():.4f}")

# Fusion (output format differs by fusion type)
if args.fusion_type == "gated":
    fused, gate = fusion(a_feat, b_feat)           # (1, T, 768), (1, T, 768)
    print(f"  Fusion:    {tuple(fused.shape)} (gate mean={gate.mean().item():.3f})")
else:
    fused = fusion(a_feat, b_feat)                 # (1, T, 768)
    print(f"  Fusion:    {tuple(fused.shape)}")

# CTC head
char_logits = ctc_head(fused)                      # (1, T, 85)
print(f"  CTC head:  {tuple(char_logits.shape)}")

# Orthogonality loss: penalize encoder similarity to encourage specialization
a_mean = a_feat.mean(dim=1)                        # (1, 768) -- mean-pooled
b_mean = b_feat.mean(dim=1)                        # (1, 768)
cos_sim = F.cosine_similarity(a_mean, b_mean).abs().mean()
print(f"  Encoder cosine sim: {cos_sim.item():.4f} (will decrease as encoders specialize)")

# CTC loss + backward test
gt = normalize_text(test_clip["ground_truth"])
target_indices = text_to_indices(gt, char_to_idx)
if target_indices and real_frames > len(target_indices):
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)
    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
    ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    # Total test loss
    total_loss = 0.3 * ctc_loss + 0.5 * mse_acou + 0.01 * cos_sim
    if test_clip["logit_path"]:
        total_loss += 0.3 * mse_sem

    total_loss.backward()

    # Verify gradients reach all trainable components
    a_grads = sum(1 for p in encoder_a.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    b_grads = sum(1 for p in encoder_b.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    f_grads = sum(1 for p in fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    p_grads = sum(1 for p in projection_a.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  CTC loss: {ctc_loss.item():.4f} | Total: {total_loss.item():.4f}")
    print(f"  Gradients: enc_a={a_grads}>0 | enc_b={b_grads}>0 | fusion={f_grads}>0 | proj={p_grads}>0")

    pred = ctc_greedy_decode(char_logits.squeeze(0).detach(), idx_to_char)
    print(f"  GT:      '{gt[:80]}'")
    print(f"  Decoded: '{pred[:80]}'")

# Zero all gradients after verification
for m in [encoder_a, encoder_b, fusion, ctc_head, projection_a, bridge_hi, bridge_mr]:
    m.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ALL VERIFIED\n")


# ==============================================================================
# STEP 5: Training setup (optimizer, scheduler, loss)
# ==============================================================================
print("=" * 70)
print("STEP 5: Training setup")
print("=" * 70)

# Differential LR: encoders train slowly to preserve pre-trained features,
# while new modules (fusion, heads) train faster since they start from scratch.
encoder_params = list(encoder_a.parameters()) + list(encoder_b.parameters())
new_params = (list(fusion.parameters()) + list(ctc_head.parameters()) +
              list(projection_a.parameters()) + list(bridge_hi.parameters()) +
              list(bridge_mr.parameters()))
all_trainable = encoder_params + new_params

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.encoder_lr},    # 1e-5 for encoders
    {"params": new_params,     "lr": args.fusion_lr},      # 1e-3 for fusion/heads
], betas=(0.9, 0.98), weight_decay=0.01)

num_epochs = args.epochs
# Steps per epoch accounts for gradient accumulation
steps_per_epoch = len(train_clips) // args.grad_accum
total_steps_est = steps_per_epoch * num_epochs
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    """
    LR multiplier: linear warmup -> cosine decay with floor.

    During warmup (0 to warmup_steps): LR ramps linearly from 0 to base_lr.
    After warmup: LR follows cosine curve from base_lr down to lr_floor * base_lr.

    Args:
        step (int): Current optimizer step (post-accumulation)

    Returns:
        float: LR multiplier in [lr_floor, 1.0]
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(args.lr_floor, 0.5 * (1.0 + math.cos(math.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Restore optimizer/scheduler state if resuming (unless --fresh_optimizer)
if args.resume and not args.fresh_optimizer:
    resume_ckpt = torch.load(PROJECT_ROOT / args.resume, map_location=DEVICE, weights_only=False)
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    del resume_ckpt
    gc.collect()
elif args.resume and args.fresh_optimizer:
    print(f"  FRESH OPTIMIZER: warm restart LR (encoder={args.encoder_lr}, fusion={args.fusion_lr})")

ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

print(f"  Optimizer: AdamW (encoder_lr={args.encoder_lr}, fusion_lr={args.fusion_lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) -> cosine decay (floor={args.lr_floor})")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)} | Grad accum: {args.grad_accum}")
print(f"  Steps/epoch: ~{steps_per_epoch} | Total: ~{total_steps_est}")
if args.fixed_weights:
    print(f"  Loss weights (FIXED): CTC={args.w_ctc} | Acoustic={args.w_acoustic} | Semantic={args.w_semantic} | Ortho={args.w_ortho}")
else:
    print(f"  Loss weights (scheduled): CTC={args.w_ctc} | Acoustic={args.w_acoustic} | Semantic={args.w_semantic} | Ortho={args.w_ortho}")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience}")
print()


# ==============================================================================
# Dev evaluation function
# ==============================================================================

def evaluate_dev(dev_clips_list):
    """
    Evaluate model on dev set: greedy CTC decode -> per-language WER.

    Runs inference (no gradients, no SpecAugment) on all dev clips,
    performs greedy CTC decoding, and computes Word Error Rate per language
    and overall.

    Args:
        dev_clips_list (list[dict]): Dev clip metadata from load_clips_from_csv

    Returns:
        tuple: (dev_wer, dev_loss, predictions, lang_wers, avg_gate) where:
          - dev_wer (float): Overall WER as percentage (e.g., 18.5)
          - dev_loss (float): Average CTC loss on dev set
          - predictions (list[tuple]): (reference, hypothesis) pairs
          - lang_wers (dict): {"hi": 15.2, "mr": 25.0, "en": 16.0}
          - avg_gate (float): Mean gate value (only meaningful for gated fusion)
    """
    # Set all modules to eval mode (disables dropout)
    fusion.eval()
    ctc_head.eval()
    encoder_a.eval()
    encoder_b.eval()

    all_refs, all_hyps = [], []
    losses = []
    lang_refs = {"hi": [], "mr": [], "en": []}
    lang_hyps = {"hi": [], "mr": [], "en": []}
    gate_vals = []

    with torch.no_grad():
        for clip in tqdm(dev_clips_list, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(wav) < 8000:  # Skip clips shorter than 0.5s
                    continue

                # Mel spectrogram -- NO SpecAugment during evaluation
                mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                mel_gpu = mel.input_features.to(DEVICE)

                # Forward through both encoders
                a_out = encoder_a(mel_gpu).last_hidden_state   # (1, 1500, 768)
                b_out = encoder_b(mel_gpu).last_hidden_state   # (1, 1500, 768)
                real_frames = compute_real_frames(len(wav))
                a_feat = a_out[:, :real_frames, :]              # (1, T, 768)
                b_feat = b_out[:, :real_frames, :]              # (1, T, 768)

                # Fusion (handle both types)
                if args.fusion_type == "gated":
                    fused, gate = fusion(a_feat, b_feat)
                    gate_vals.append(gate.mean().item())
                else:
                    fused = fusion(a_feat, b_feat)

                logits = ctc_head(fused)                       # (1, T, 85)

                # CTC loss for monitoring
                gt = normalize_text(clip["ground_truth"])
                target_indices = text_to_indices(gt, char_to_idx)
                if target_indices and real_frames > len(target_indices):
                    log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
                    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
                    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
                    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
                    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        losses.append(loss.item())

                # Greedy CTC decode
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
                    del a_out, b_out, a_feat, b_feat, fused, logits
                    torch.cuda.empty_cache()

            except Exception:
                continue

    # Restore training mode
    fusion.train()
    ctc_head.train()
    encoder_a.train()
    encoder_b.train()

    dev_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0
    dev_loss = np.mean(losses) if losses else 0.0
    avg_gate = np.mean(gate_vals) if gate_vals else 0.5

    lang_wers = {}
    for lc in ("hi", "mr", "en"):
        if lang_refs[lc]:
            lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
        else:
            lang_wers[lc] = None

    return dev_wer, dev_loss, list(zip(all_refs, all_hyps)), lang_wers, avg_gate


# ==============================================================================
# STEP 6: Training loop
# ==============================================================================
print("=" * 70)
mode_str = f"VERIFICATION ({len(train_clips)} clips)" if args.verify else f"Training ({len(train_clips)} clips)"
print(f"STEP 6: {mode_str}")
print("=" * 70)

# Set seeds for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

epoch_stats = []
global_step = 0
patience_counter = 0
training_start = time.time()

# Set all trainable modules to train mode
for m in [encoder_a, encoder_b, fusion, ctc_head, projection_a, bridge_hi, bridge_mr]:
    m.train()

for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
    random.shuffle(train_clips)

    # Get scheduled loss weights for this epoch
    # These change over training: KD-heavy early -> CTC-heavy late
    w_ctc, w_acou, w_sem = get_loss_weights(epoch, start_epoch + num_epochs)

    # Per-epoch loss accumulators
    ep_ctc_losses = []
    ep_mse_acou_losses = []
    ep_mse_sem_losses = []
    ep_ortho_losses = []
    ep_total_losses = []
    ep_gate_vals = []      # Only used for gated fusion
    ep_cos_sims = []
    ep_start = time.time()
    skipped = 0

    optimizer.zero_grad()  # Zero grads at start (for gradient accumulation)

    pbar = tqdm(train_clips, desc=f"Epoch {epoch}",
                unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # ================================================================
            # 1. Load audio waveform
            # ================================================================
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:  # Skip clips shorter than 0.5s
                skipped += 1
                continue

            num_samples = len(wav)

            # ================================================================
            # 2. Mel spectrogram + SpecAugment
            # ================================================================
            mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            if args.specaugment:
                mel_features = apply_spec_augment(mel.input_features.clone())
            else:
                mel_features = mel.input_features
            mel_gpu = mel_features.to(DEVICE)
            # mel_gpu shape: (1, 80, 3000) -- same mel fed to BOTH encoders

            # ================================================================
            # 3. Forward through both student encoders (WITH gradients)
            # ================================================================
            a_out = encoder_a(mel_gpu).last_hidden_state   # (1, 1500, 768)
            b_out = encoder_b(mel_gpu).last_hidden_state   # (1, 1500, 768)

            real_frames = compute_real_frames(num_samples)
            a_feat = a_out[:, :real_frames, :]              # (1, T, 768)
            b_feat = b_out[:, :real_frames, :]              # (1, T, 768)

            # ================================================================
            # 4. Acoustic MSE: Encoder A projection vs Kid-Whisper teacher
            # ================================================================
            # Teacher runs frozen (no_grad) -- we only learn the student
            with torch.no_grad():
                t_out = teacher_encoder(mel_gpu).last_hidden_state  # (1, 1500, 1024)
                t_feat = t_out[:, :real_frames, :]                   # (1, T, 1024)

            proj_a = projection_a(a_feat)                            # (1, T, 1024)
            mse_acou = F.mse_loss(proj_a, t_feat)
            # mse_acou: scalar -- how far Encoder A is from Kid-Whisper teacher

            # ================================================================
            # 5. Semantic MSE: Encoder B bridge vs IndicConformer (Hi/Mr only)
            # ================================================================
            # English clips don't have IndicConformer logits, so semantic MSE
            # is skipped for them. The clip still gets acoustic MSE + CTC.
            mse_sem = torch.tensor(0.0, device=DEVICE)
            has_semantic = False
            if clip["logit_path"] and clip["lang_code"] in ("hi", "mr"):
                try:
                    teacher_data = torch.load(clip["logit_path"], weights_only=True)
                    t_logits = teacher_data["logits"].float().to(DEVICE)  # (T_teacher, 257)
                    t_aligned = align_teacher_to_student(t_logits, real_frames)  # (1, T, 257)

                    # Use language-specific bridge head
                    if clip["lang_code"] == "hi":
                        b_logits = bridge_hi(b_feat)   # (1, T, 257)
                    else:
                        b_logits = bridge_mr(b_feat)   # (1, T, 257)
                    mse_sem = F.mse_loss(b_logits, t_aligned)
                    has_semantic = True
                except Exception:
                    pass

            # ================================================================
            # 6. Fusion -> CTC head -> CTC loss
            # ================================================================
            if args.fusion_type == "gated":
                fused, gate = fusion(a_feat, b_feat)   # (1, T, 768), (1, T, 768)
                ep_gate_vals.append(gate.mean().item())
            else:
                fused = fusion(a_feat, b_feat)         # (1, T, 768)

            logits = ctc_head(fused)                   # (1, T, 85)

            gt = normalize_text(clip["ground_truth"])
            target_indices = text_to_indices(gt, char_to_idx)

            if not target_indices or real_frames <= len(target_indices):
                skipped += 1
                continue

            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)
            targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

            ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            # ================================================================
            # 7. Orthogonality loss: encourage encoder specialization
            # ================================================================
            # Mean-pool over time, then compute cosine similarity.
            # We want encoders to produce DIFFERENT representations (low similarity)
            # so we penalize high |cos_sim|.
            a_mean = a_feat.mean(dim=1)                 # (1, 768)
            b_mean = b_feat.mean(dim=1)                 # (1, 768)
            ortho_loss = F.cosine_similarity(a_mean, b_mean).abs().mean()

            # ================================================================
            # 8. Combined loss with scheduled weights
            # ================================================================
            total_loss = w_ctc * ctc_loss + w_acou * mse_acou + args.w_ortho * ortho_loss
            if has_semantic:
                total_loss += w_sem * mse_sem

            # Skip bad losses (NaN/Inf can occur with extreme inputs)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                skipped += 1
                continue

            # Scale for gradient accumulation: we average over grad_accum steps
            (total_loss / args.grad_accum).backward()

            # ================================================================
            # 9. Optimizer step every grad_accum clips
            # ================================================================
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Track losses for epoch summary
            ep_ctc_losses.append(ctc_loss.item())
            ep_mse_acou_losses.append(mse_acou.item())
            if has_semantic:
                ep_mse_sem_losses.append(mse_sem.item())
            ep_ortho_losses.append(ortho_loss.item())
            ep_total_losses.append(total_loss.item())
            ep_cos_sims.append(ortho_loss.item())

            # ================================================================
            # 10. Free GPU memory
            # ================================================================
            if DEVICE == "cuda":
                del a_out, b_out, a_feat, b_feat, t_out, t_feat, proj_a
                del fused, logits, mel_gpu
                if has_semantic:
                    del t_logits, t_aligned, b_logits
                torch.cuda.empty_cache()

            # Update progress bar with recent loss values
            if ep_total_losses:
                pbar_dict = {
                    "total": f"{ep_total_losses[-1]:.3f}",
                    "ctc": f"{ep_ctc_losses[-1]:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                }
                if args.fusion_type == "gated" and ep_gate_vals:
                    pbar_dict["gate"] = f"{ep_gate_vals[-1]:.2f}"
                pbar.set_postfix(pbar_dict)

        except Exception as e:
            print(f"\n  ERROR step {global_step}: {clip['clip_name']}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            skipped += 1
            continue

    pbar.close()

    # Handle remaining accumulated gradients at end of epoch
    if len(train_clips) % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    # -- Epoch summary --
    ep_time = time.time() - ep_start
    avg_ctc = np.mean(ep_ctc_losses) if ep_ctc_losses else 0
    avg_mse_acou = np.mean(ep_mse_acou_losses) if ep_mse_acou_losses else 0
    avg_mse_sem = np.mean(ep_mse_sem_losses) if ep_mse_sem_losses else 0
    avg_ortho = np.mean(ep_ortho_losses) if ep_ortho_losses else 0
    avg_total = np.mean(ep_total_losses) if ep_total_losses else 0
    avg_gate = np.mean(ep_gate_vals) if ep_gate_vals else 0.5
    avg_cos = np.mean(ep_cos_sims) if ep_cos_sims else 0

    print(f"\n  +-- Epoch {epoch} -------------------------------------------------")
    print(f"  | Total loss:  {avg_total:.4f}")
    print(f"  | CTC loss:    {avg_ctc:.4f} (weight={w_ctc:.1f})")
    print(f"  | MSE acoustic:{avg_mse_acou:.4f} (weight={w_acou:.1f})")
    print(f"  | MSE semantic:{avg_mse_sem:.4f} (weight={w_sem:.1f})")
    print(f"  | Ortho loss:  {avg_ortho:.4f} (weight={args.w_ortho})")
    if args.fusion_type == "gated":
        print(f"  | Gate mean:   {avg_gate:.3f} (std={np.std(ep_gate_vals):.3f})")
    print(f"  | Encoder cos: {avg_cos:.4f}")
    print(f"  | Steps: {len(ep_total_losses)} ({skipped} skipped) | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
    print(f"  | LR: encoder={scheduler.get_last_lr()[0]:.2e}, fusion={scheduler.get_last_lr()[1]:.2e}")
    print(f"  | Weights: CTC={w_ctc:.1f} | Acou={w_acou:.1f} | Sem={w_sem:.1f}")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  | Peak GPU: {peak:.0f} MB")
    print(f"  +--------------------------------------------------------------")

    # -- Warning signs --
    if args.fusion_type == "gated":
        if avg_gate < 0.1 or avg_gate > 0.9:
            print(f"  WARNING: Gate mean={avg_gate:.3f} -- one encoder may be ignored!")
    if avg_cos > 0.9:
        print(f"  WARNING: Encoder cos_sim={avg_cos:.3f} -- possible mode collapse!")

    # -- Dev evaluation --
    dev_wer, dev_loss, dev_preds, lang_wers, dev_gate = 999.0, 0.0, [], {}, 0.5
    if dev_clips:
        dev_wer, dev_loss, dev_preds, lang_wers, dev_gate = evaluate_dev(dev_clips)
        print(f"\n  +-- Dev Results -----------------------------------------------")
        print(f"  | WER:  {dev_wer:.2f}%")
        for lc, lname in [("hi", "Hindi"), ("mr", "Marathi"), ("en", "English")]:
            if lang_wers.get(lc) is not None:
                print(f"  |   {lname}: {lang_wers[lc]:.2f}%")
        print(f"  | Loss: {dev_loss:.4f}", end="")
        if args.fusion_type == "gated":
            print(f" | Gate: {dev_gate:.3f}")
        else:
            print()
        print(f"  +--------------------------------------------------------------")

        if dev_preds:
            n_show = min(3, len(dev_preds))
            print(f"\n  Sample predictions:")
            for ref, hyp in dev_preds[:n_show]:
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
                print()

    # Save epoch stats for final summary table
    epoch_stats.append({
        "epoch": epoch,
        "train_loss": avg_total,
        "ctc_loss": avg_ctc,
        "mse_acou": avg_mse_acou,
        "mse_sem": avg_mse_sem,
        "ortho": avg_ortho,
        "gate_mean": avg_gate,
        "cos_sim": avg_cos,
        "dev_wer": dev_wer,
        "dev_loss": dev_loss,
        "lang_wers": lang_wers,
        "w_ctc": w_ctc, "w_acou": w_acou, "w_sem": w_sem,
    })

    # -- Checkpoint saving --
    if not args.verify:
        ckpt_data = {
            # Metadata
            "epoch": epoch,
            "global_step": global_step,
            "fusion_type": args.fusion_type,            # "gated" or "cross_attention"
            # Model state dicts (ALL components saved)
            "fusion_state_dict": fusion.state_dict(),    # Works for both GatedFusion and CrossAttentionFusion
            "ctc_head_state_dict": ctc_head.state_dict(),
            "acoustic_encoder_state_dict": encoder_a.state_dict(),
            "semantic_encoder_state_dict": encoder_b.state_dict(),
            "projection_a_state_dict": projection_a.state_dict(),
            "bridge_hi_state_dict": bridge_hi.state_dict(),
            "bridge_mr_state_dict": bridge_mr.state_dict(),
            # Optimizer state (for --resume)
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # Metrics
            "dev_wer": dev_wer,
            "train_loss": avg_total,
            "dev_loss": dev_loss,
            "lang_wers": lang_wers,
            # Config (for reproducibility)
            "vocab_size": vocab_size,
            "args": vars(args),
        }

        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            patience_counter = 0
            torch.save(ckpt_data, CKPT_DIR / "best_wer.pt")
            print(f"  NEW BEST DEV WER: {dev_wer:.2f}% -> saved best_wer.pt")
        else:
            patience_counter += 1
            print(f"  Early stopping: {patience_counter}/{args.patience} "
                  f"(best: {best_dev_wer:.2f}%)")
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n  EARLY STOPPING at epoch {epoch}")
                torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch}.pt")
                break

        torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch}.pt")
        print(f"  Saved: {CKPT_DIR / f'epoch_{epoch}.pt'}")

    print()

total_time = time.time() - training_start


# ==============================================================================
# STEP 7: Final results
# ==============================================================================
print("=" * 70)
print("STEP 7: Final Results")
print("=" * 70)

if epoch_stats:
    # Print epoch-by-epoch summary table
    header = f"  {'Ep':<4} {'Total':>7} {'CTC':>7} {'Acou':>7} {'Sem':>7} {'Ortho':>6} "
    if args.fusion_type == "gated":
        header += f"{'Gate':>5} "
    header += f"{'CosSim':>6} {'WER':>8} {'Hi':>7} {'Mr':>7} {'En':>7}"
    print(f"\n{header}")

    divider = f"  {'---':<4} {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'---':>6} "
    if args.fusion_type == "gated":
        divider += f"{'---':>5} "
    divider += f"{'---':>6} {'---':>8} {'---':>7} {'---':>7} {'---':>7}"
    print(divider)

    for s in epoch_stats:
        wer = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        hi = f"{s['lang_wers'].get('hi', 0):.2f}%" if s['lang_wers'].get('hi') is not None else "N/A"
        mr = f"{s['lang_wers'].get('mr', 0):.2f}%" if s['lang_wers'].get('mr') is not None else "N/A"
        en = f"{s['lang_wers'].get('en', 0):.2f}%" if s['lang_wers'].get('en') is not None else "N/A"
        line = (f"  {s['epoch']:<4} {s['train_loss']:>7.4f} {s['ctc_loss']:>7.4f} {s['mse_acou']:>7.4f} "
                f"{s['mse_sem']:>7.4f} {s['ortho']:>6.4f} ")
        if args.fusion_type == "gated":
            line += f"{s['gate_mean']:>5.3f} "
        line += f"{s['cos_sim']:>6.4f} {wer:>8} {hi:>7} {mr:>7} {en:>7}"
        print(line)

    if len(epoch_stats) >= 2:
        first = epoch_stats[0]["train_loss"]
        last = epoch_stats[-1]["train_loss"]
        if first > 0:
            reduction = (first - last) / first * 100
            print(f"\n  Loss reduction: {reduction:.1f}% ({first:.4f} -> {last:.4f})")

    if best_dev_wer < 999:
        print(f"\n  Best dev WER: {best_dev_wer:.2f}%")

print(f"\n  Fusion type: {args.fusion_type}")
print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
print("Done.")
