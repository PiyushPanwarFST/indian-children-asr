"""
Combined Branch — Dual-Encoder Gated Fusion Training (Unfrozen Encoders)
========================================================================

WHAT THIS SCRIPT DOES:
    Trains a gated fusion model that combines TWO Whisper Small encoders:
      1. Acoustic encoder  — distilled from Kid-Whisper (children's speech expert)
      2. Semantic encoder   — distilled from IndicConformer (Indian language expert)

    Both encoders are UNFROZEN and fine-tuned with a small learning rate (1e-5),
    while the fusion module and CTC head use a larger learning rate (1e-3).
    This differential LR lets encoders co-adapt to produce complementary features
    without destroying their pre-trained representations.

WHY THIS WORKS:
    The acoustic encoder captures children's voice patterns (pitch, rate, disfluency).
    The semantic encoder captures Indian language phonetics (Hindi, Marathi phonemes).
    Neither alone has both. The gated fusion learns per-frame, per-dimension weighting
    to combine them optimally. Unfreezing allows the encoders to specialize further —
    each encoder learns to focus on what the other cannot provide.

DETAILED ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  INPUT:                                                              │
    │    Raw audio (.mp3/.wav) → resample to 16kHz mono                    │
    │    → WhisperFeatureExtractor → 80-bin log-Mel spectrogram (1,80,3000)│
    │    → SpecAugment: 2 freq masks (width≤15), 2 time masks (width≤50)  │
    │                                                                      │
    │  DUAL ENCODERS (both unfrozen, lr=1e-5):                             │
    │    Acoustic Encoder (Whisper Small, 12 layers, 768-dim):             │
    │      - Initialized from: Kid-Whisper KD + CTC joint training         │
    │      - Checkpoint: checkpoints/joint/joint_best_wer.pt               │
    │      - Input: mel (1, 80, 3000) → Output: feat_a (1, T, 768)        │
    │                                                                      │
    │    Semantic Encoder (Whisper Small, 12 layers, 768-dim):             │
    │      - Initialized from: IndicConformer MSE distillation             │
    │      - Checkpoint: checkpoints/semantic_mse/best_dev.pt              │
    │      - Input: mel (1, 80, 3000) → Output: feat_s (1, T, 768)        │
    │                                                                      │
    │    Both encoders process the SAME augmented mel spectrogram.          │
    │    T = real_frames = min(num_samples // 160 // 2, 1500)              │
    │    Padding frames beyond actual audio length are trimmed off.         │
    │                                                                      │
    │  GATED FUSION MODULE (lr=1e-3, ~1.18M params):                       │
    │    Step 1: Concatenate features                                      │
    │      concat = [feat_a; feat_s]              → (1, T, 1536)           │
    │    Step 2: Compute sigmoid gate                                      │
    │      gate = σ(W_g · concat + b_g)           → (1, T, 768)            │
    │      W_g: Linear(1536→768), b_g: bias(768)                           │
    │      gate values ∈ [0,1] per frame per dimension                     │
    │      gate≈1 → trust acoustic, gate≈0 → trust semantic               │
    │    Step 3: Weighted combination                                      │
    │      fused = gate ⊙ feat_a + (1-gate) ⊙ feat_s  → (1, T, 768)      │
    │    Step 4: Normalize and regularize                                  │
    │      fused = LayerNorm(fused)                                        │
    │      fused = Dropout(fused, p=0.1)                                   │
    │                                                                      │
    │  CTC HEAD (lr=1e-3, ~65K params):                                    │
    │    logits = Linear(768→85)(fused)            → (1, T, 85)            │
    │    Warm-started from acoustic branch CTC head.                       │
    │    85 = 3 special (<blank>, <space>, <unk>) + 22 English + 60 Devanag│
    │                                                                      │
    │  LOSS AND OPTIMIZATION:                                              │
    │    CTC Loss (blank=0, zero_infinity=True)                            │
    │    Target: character indices from ground truth text                   │
    │    Optimizer: AdamW with differential LR:                            │
    │      - Encoder params: lr=1e-5 (small, preserve pre-trained features)│
    │      - Fusion + CTC params: lr=1e-3 (large, learn fast)             │
    │    Scheduler: linear warmup (500 steps) → cosine decay to zero       │
    │    Gradient clipping: max_norm=1.0                                   │
    │                                                                      │
    │  GRADIENT FLOW (backward pass):                                      │
    │    CTC Loss                                                          │
    │      ↓                                                               │
    │    CTC Head (lr=1e-3)                                                │
    │      ↓                                                               │
    │    GatedFusion gate + LayerNorm (lr=1e-3)                            │
    │      ↓ splits via gate weighting                                     │
    │      ├→ Acoustic Encoder all 12 layers (lr=1e-5)                     │
    │      └→ Semantic Encoder all 12 layers (lr=1e-5)                     │
    │    Gradients encourage encoders to SPECIALIZE:                       │
    │    the fusion gate tells each encoder what features are missing.     │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

PSEUDOCODE:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ STEP 1: Load vocabulary (85 characters from vocab.json)             │
    │                                                                      │
    │ STEP 2: Load training data from asr_train.csv (~13,765 clips)       │
    │         Load dev data from asr_dev.csv (~1,775 clips)               │
    │         Optionally limit clips with --test_clips N                   │
    │                                                                      │
    │ STEP 3: Load models                                                  │
    │   3a. Load Whisper Small encoder → restore acoustic weights          │
    │       from joint_best_wer.pt → set .train() mode                     │
    │   3b. Load Whisper Small encoder → restore semantic weights           │
    │       from best_dev.pt → set .train() mode                           │
    │   3c. Create GatedFusion(dim=768, dropout=0.1) → random init         │
    │   3d. Create CTC Head Linear(768→85) → warm-start from acoustic ckpt │
    │   3e. If --resume: load ALL weights from combined checkpoint          │
    │       (fusion + CTC + both encoders + optimizer + scheduler)          │
    │                                                                      │
    │ STEP 4: Define helper functions                                      │
    │   load_audio(path) → 16kHz mono tensor                               │
    │   compute_real_frames(samples) → encoder output length               │
    │   apply_spec_augment(mel) → augmented mel spectrogram                │
    │                                                                      │
    │ STEP 5: Verify forward + backward pass on 1 clip                     │
    │   Check: mel shape, encoder outputs, fusion output, CTC loss,        │
    │   gradients flow to ALL components (encoders, fusion, CTC head)      │
    │                                                                      │
    │ STEP 6: Setup optimizer and scheduler                                │
    │   AdamW with 2 param groups:                                         │
    │     Group 1: encoder params → lr=1e-5                                │
    │     Group 2: fusion + CTC params → lr=1e-3                           │
    │   LR schedule: warmup 500 steps → cosine decay                       │
    │                                                                      │
    │ STEP 7: Training loop                                                │
    │   FOR each epoch (1 to 30):                                          │
    │     Shuffle training clips                                           │
    │     FOR each clip:                                                   │
    │       1. Load audio → 16kHz mono waveform                            │
    │       2. Compute mel spectrogram → apply SpecAugment                 │
    │       3. Forward through BOTH encoders (with gradients)              │
    │       4. Trim to real frames (remove padding)                        │
    │       5. Gated fusion: gate = σ(W·[a;s]+b), fused = gate·a+(1-g)·s  │
    │       6. CTC head: logits = Linear(fused)                            │
    │       7. CTC loss on log_softmax(logits) vs target character indices │
    │       8. Backward pass → clip gradients → optimizer step → scheduler │
    │     END FOR                                                          │
    │     Evaluate on dev set → per-language WER                           │
    │     Save checkpoint (fusion + CTC + encoders + optimizer + scheduler)│
    │     If dev WER improved → save best_wer.pt                           │
    │     If no improvement for 10 epochs → EARLY STOP                     │
    │   END FOR                                                            │
    │                                                                      │
    │ STEP 8: Print final epoch-by-epoch results table                     │
    │         Compare with individual encoder baselines                    │
    └──────────────────────────────────────────────────────────────────────┘

WARM START:
    Acoustic encoder: checkpoints/joint/joint_best_wer.pt → encoder_state_dict
    Semantic encoder: checkpoints/semantic_mse/best_dev.pt → encoder_state_dict
    CTC head:         checkpoints/joint/joint_best_wer.pt → ctc_head_state_dict
                      (warm from acoustic branch — already knows char→CTC mapping)

Prerequisites:
    - Acoustic checkpoint: checkpoints/joint/joint_best_wer.pt
    - Semantic checkpoint: checkpoints/semantic_mse/best_dev.pt
    - Vocabulary: ASER-Dataset/vocab.json (85 tokens)
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Test run (3000 clips, 10 epochs)
    python step1_combined_training.py --test_clips 3000 --epochs 10

    # Full training (all clips, 30 epochs)
    python step1_combined_training.py --epochs 30 --patience 10

    # Resume from checkpoint
    python step1_combined_training.py --resume checkpoints/combined_unfrozen/epoch_5.pt --epochs 30
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

# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Combined branch: dual-encoder gated fusion training")
parser.add_argument("--verify", action="store_true",
                    help="Quick sanity check with 100 clips, no checkpointing")
parser.add_argument("--test_clips", type=int, default=None,
                    help="Limit number of training clips (for test runs)")
parser.add_argument("--epochs", type=int, default=30,
                    help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate for fusion module + CTC head")
parser.add_argument("--encoder_lr", type=float, default=1e-5,
                    help="Learning rate for encoder params (100x smaller than fusion LR)")
parser.add_argument("--warmup_steps", type=int, default=500,
                    help="Linear warmup steps before cosine decay")
parser.add_argument("--max_audio_sec", type=float, default=30.0,
                    help="Max audio length in seconds (longer clips are truncated)")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience on dev WER (0 = disabled)")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout rate in fusion module")
parser.add_argument("--acoustic_ckpt", type=str,
                    default="checkpoints/joint/joint_best_wer.pt",
                    help="Path to acoustic encoder checkpoint (from Kid-Whisper KD + CTC)")
parser.add_argument("--semantic_ckpt", type=str,
                    default="checkpoints/semantic_mse/best_dev.pt",
                    help="Path to semantic encoder checkpoint (from IndicConformer KD)")
parser.add_argument("--ctc_ckpt", type=str, default=None,
                    help="Optional separate CTC head checkpoint (default: from acoustic_ckpt)")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume training from a combined checkpoint (e.g. epoch_5.pt)")
parser.add_argument("--grad_checkpoint", action="store_true",
                    help="Enable gradient checkpointing to reduce GPU memory usage")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--log_every", type=int, default=10,
                    help="Log frequency (every N steps)")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# PATHS AND CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Project root: 4 levels up from this script
# this file → 01_feature_fusion/ → combined_branch/ → experiments/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"

# Checkpoints saved to combined_unfrozen/ directory
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "combined_unfrozen"
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load character vocabulary (85 tokens)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading character vocabulary")
print("=" * 70)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)  # 85 tokens: 3 special + 22 English + 60 Devanagari
BLANK_IDX = char_to_idx["<blank>"]  # 0 — CTC blank token

print(f"  Vocab: {vocab_size} tokens (from {VOCAB_PATH.name})")
print(f"  Blank index: {BLANK_IDX}")


def text_to_indices(text, char_to_idx):
    """
    Convert normalized text string to a list of vocabulary indices for CTC targets.

    Each character is mapped to its index in the vocabulary. Spaces are mapped
    to the <space> token. Unknown characters are mapped to <unk>.

    Args:
        text (str): Normalized ground truth text (e.g., "राधा के पास")
        char_to_idx (dict): Character-to-index mapping from vocab.json

    Returns:
        list[int]: List of token indices, e.g., [25, 42, 1, 30, ...]
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
    Perform greedy CTC decoding from model logits.

    Takes the argmax at each time step, collapses consecutive repeated tokens,
    and removes blank tokens to produce the final decoded string.

    Args:
        logits (Tensor): Shape (T, vocab_size) — raw logits from CTC head
        idx_to_char (dict): Index-to-character mapping (reverse of vocab)

    Returns:
        str: Decoded text string (e.g., "राधा के पास")
    """
    indices = torch.argmax(logits, dim=-1)  # (T,) — best token at each frame
    collapsed = torch.unique_consecutive(indices)  # Remove repeated tokens
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
# STEP 2: Load training and dev data (Hindi, Marathi, English)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading training data (all languages)")
print("=" * 70)


def load_clips_from_csv(csv_path, max_clips=None):
    """
    Load audio clip metadata from a CSV split file.

    Reads asr_train.csv or asr_dev.csv and returns a list of clip dictionaries.
    Each clip has audio path, language, ground truth transcript, and duration.

    Args:
        csv_path (str or Path): Path to the CSV file (asr_train.csv or asr_dev.csv)
        max_clips (int, optional): If set, only return the first N clips

    Returns:
        tuple: (clips, skipped) where:
            - clips (list[dict]): List of clip dicts with keys:
                audio_path, language, lang_code, clip_name, duration_sec, ground_truth
            - skipped (int): Number of clips skipped due to unknown language
    """
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

            # Create unique clip ID from child_id + filename
            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            # Ground truth text (try multiple column names for compatibility)
            gt = row.get("transcript", row.get("que_text", row.get("text", "")))

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "clip_name": clip_uid,
                "duration_sec": float(row.get("duration_sec", 0)),
                "ground_truth": gt,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips, skipped


# Load training set
train_clips, skip_other = load_clips_from_csv(TRAIN_CSV)
hi_clips = [c for c in train_clips if c["lang_code"] == "hi"]
mr_clips = [c for c in train_clips if c["lang_code"] == "mr"]
en_clips = [c for c in train_clips if c["lang_code"] == "en"]
total_hrs = sum(c["duration_sec"] for c in train_clips) / 3600

print(f"  Total: {len(train_clips)} clips ({total_hrs:.1f}h)")
print(f"    Hindi:   {len(hi_clips)} clips")
print(f"    Marathi: {len(mr_clips)} clips")
print(f"    English: {len(en_clips)} clips")
print(f"  Skipped: {skip_other} unknown language")

if len(train_clips) == 0:
    print("\n  ERROR: No training clips found!")
    exit(1)

# Load dev set
dev_clips, _ = load_clips_from_csv(DEV_CSV)
print(f"  Dev set: {len(dev_clips)} clips")

# ── Optionally limit clips for test runs ────────────────────────────────────
N = args.test_clips
if N is None and args.verify:
    N = 100

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
    label = "VERIFY MODE" if args.verify else "LIMITED"
    print(f"\n  {label}: {len(train_clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

    if args.verify and dev_clips:
        dev_n = min(100, len(dev_clips))
        random.seed(args.seed + 1)
        dev_clips = random.sample(dev_clips, dev_n)
        print(f"    Dev subset: {len(dev_clips)} clips")

print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load models (two unfrozen encoders + trainable fusion + CTC head)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Loading models")
print("=" * 70)

import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"

# Feature extractor: converts raw audio waveform → 80-bin log-Mel spectrogram
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)


def load_encoder(checkpoint_path, label):
    """
    Load a Whisper Small encoder and restore trained weights from checkpoint.

    Initializes a fresh Whisper Small model, extracts just the encoder, then
    loads our distilled weights from a checkpoint file. The encoder is kept
    trainable (unfrozen) with gradient checkpointing enabled if requested.

    Handles two checkpoint formats:
      - 'encoder_state_dict': Direct encoder weights (from joint/semantic training)
      - 'model_state_dict': Full model weights with 'encoder.' prefix (from MSE training)

    Args:
        checkpoint_path (str): Relative path from PROJECT_ROOT to the .pt checkpoint
        label (str): Display name for logging (e.g., "ACOUSTIC ENCODER")

    Returns:
        nn.Module: Whisper encoder with restored weights, in train mode, on DEVICE
    """
    print(f"\n  [{label}] Loading Whisper Small encoder...")

    # Load a fresh Whisper Small model, extract just the encoder
    whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
    encoder = whisper_model.encoder.to(DEVICE)

    # Resolve checkpoint path
    ckpt_path = PROJECT_ROOT / checkpoint_path
    if not ckpt_path.exists():
        # Try known alternate paths
        alt_paths = [
            PROJECT_ROOT / "checkpoints" / "semantic" / "best_dev_model.pt",
            PROJECT_ROOT / "checkpoints" / "joint" / "joint_best_wer.pt",
        ]
        for alt in alt_paths:
            if alt.exists() and "semantic" in checkpoint_path.lower():
                ckpt_path = alt
                break
        if not ckpt_path.exists():
            print(f"    ERROR: Checkpoint not found: {ckpt_path}")
            exit(1)

    print(f"    Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Load weights — handle different checkpoint key formats
    if "encoder_state_dict" in ckpt:
        # Format from joint training / semantic training: keys are direct encoder params
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    elif "model_state_dict" in ckpt:
        # Format from acoustic MSE training: keys have 'encoder.' prefix, strip it
        state = {}
        for k, v in ckpt["model_state_dict"].items():
            if k.startswith("encoder."):
                state[k[len("encoder."):]] = v
        encoder.load_state_dict(state)
    else:
        print(f"    ERROR: Unknown checkpoint format. Keys: {list(ckpt.keys())[:5]}")
        exit(1)

    epoch_info = ckpt.get("epoch", "?")
    print(f"    Loaded from epoch {epoch_info}")

    # Set encoder to training mode (BatchNorm and Dropout active)
    encoder.train()

    # Enable gradient checkpointing if requested — trades compute for memory
    # Instead of storing all intermediate activations, recomputes them during backward pass
    # Reduces GPU memory by ~40% at the cost of ~20% slower training
    if args.grad_checkpoint:
        encoder.gradient_checkpointing_enable()
        print(f"    Gradient checkpointing ENABLED")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"    UNFROZEN ({n_params:,} params, requires_grad=True, lr={args.encoder_lr})")

    # Clean up the full Whisper model (we only keep the encoder)
    del whisper_model, ckpt
    gc.collect()

    return encoder


# ── Load both encoders (both unfrozen, will be fine-tuned) ──────────────────
acoustic_encoder = load_encoder(args.acoustic_ckpt, "ACOUSTIC ENCODER")
semantic_encoder = load_encoder(args.semantic_ckpt, "SEMANTIC ENCODER")


# ── Gated Fusion Module ────────────────────────────────────────────────────
class GatedFusion(nn.Module):
    """
    Learnable gated fusion that combines acoustic and semantic encoder features.

    For each time frame and each feature dimension, the gate learns how much to
    trust the acoustic encoder vs the semantic encoder. The gate is a sigmoid
    function, so values near 1 mean "use acoustic" and near 0 mean "use semantic".

    Equation:
        gate = σ(W_g · [feat_a; feat_s] + b_g)           # (B, T, 768)
        fused = gate ⊙ feat_a + (1 - gate) ⊙ feat_s      # (B, T, 768)
        output = Dropout(LayerNorm(fused))

    Args:
        dim (int): Feature dimension of each encoder output (768 for Whisper Small)
        dropout (float): Dropout rate applied after LayerNorm
    """
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        # Input: concatenated features from both encoders (dim * 2 = 1536)
        # Output: gate values per dimension (dim = 768)
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic_feat, semantic_feat):
        """
        Forward pass: compute gated fusion of two feature tensors.

        Args:
            acoustic_feat (Tensor): Shape (B, T, 768) — acoustic encoder output
            semantic_feat (Tensor): Shape (B, T, 768) — semantic encoder output

        Returns:
            Tensor: Shape (B, T, 768) — fused features after gating + LayerNorm + Dropout
        """
        # Concatenate along feature dimension: (B, T, 1536)
        concat = torch.cat([acoustic_feat, semantic_feat], dim=-1)

        # Compute gate: sigmoid squashes to [0, 1] per dimension
        gate = torch.sigmoid(self.gate_linear(concat))  # (B, T, 768)

        # Weighted combination: gate=1 means full acoustic, gate=0 means full semantic
        fused = gate * acoustic_feat + (1 - gate) * semantic_feat  # (B, T, 768)

        # Normalize and regularize
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused


print(f"\n  [GATED FUSION] Creating fusion module...")
fusion = GatedFusion(dim=768, dropout=args.dropout).to(DEVICE)
fusion_params = sum(p.numel() for p in fusion.parameters())
print(f"    Params: {fusion_params:,}")

# ── CTC Head: maps fused features to character probabilities ────────────────
print(f"\n  [CTC HEAD] Linear(768→{vocab_size})...")
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# ── Load checkpoint (resume or warm-start) ──────────────────────────────────
start_epoch = 0
best_dev_wer = float("inf")

if args.resume:
    # RESUME: Load everything from a previous combined training checkpoint
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading combined checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Restore fusion module and CTC head
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])

    # Restore encoder weights (critical — encoders change during unfrozen training)
    if "acoustic_encoder_state_dict" in ckpt:
        acoustic_encoder.load_state_dict(ckpt["acoustic_encoder_state_dict"])
        print(f"    Acoustic encoder restored from checkpoint")
    if "semantic_encoder_state_dict" in ckpt:
        semantic_encoder.load_state_dict(ckpt["semantic_encoder_state_dict"])
        print(f"    Semantic encoder restored from checkpoint")

    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))
    print(f"    Resuming from epoch {start_epoch}, best dev WER: {best_dev_wer:.2f}%")
    del ckpt
else:
    # WARM-START: Initialize CTC head from acoustic branch checkpoint
    # The acoustic branch CTC head already knows the char→CTC mapping,
    # so we reuse it instead of random initialization
    ctc_ckpt_path = args.ctc_ckpt or args.acoustic_ckpt
    ctc_ckpt_full = PROJECT_ROOT / ctc_ckpt_path
    if ctc_ckpt_full.exists():
        print(f"    CTC head warm-start from: {ctc_ckpt_full}")
        ctc_ckpt = torch.load(ctc_ckpt_full, map_location=DEVICE, weights_only=False)
        if "ctc_head_state_dict" in ctc_ckpt:
            ctc_head.load_state_dict(ctc_ckpt["ctc_head_state_dict"])
            print(f"    CTC head loaded (warm-started from acoustic branch)")
        elif "ctc_head_char_state_dict" in ctc_ckpt:
            ctc_head.load_state_dict(ctc_ckpt["ctc_head_char_state_dict"])
            print(f"    CTC head loaded (warm-started from semantic branch)")
        else:
            print(f"    WARNING: No CTC head found in checkpoint — random init")
        del ctc_ckpt
    else:
        print(f"    CTC head: random init (no checkpoint at {ctc_ckpt_full})")

gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ── Print parameter summary ────────────────────────────────────────────────
acoustic_params = sum(p.numel() for p in acoustic_encoder.parameters())
semantic_params = sum(p.numel() for p in semantic_encoder.parameters())
ctc_params = sum(p.numel() for p in ctc_head.parameters())
total_trainable = acoustic_params + semantic_params + fusion_params + ctc_params

print(f"\n  ┌─ Parameter Summary ──────────────────────────────────────────")
print(f"  │ Acoustic encoder: {acoustic_params:,} (unfrozen, lr={args.encoder_lr})")
print(f"  │ Semantic encoder: {semantic_params:,} (unfrozen, lr={args.encoder_lr})")
print(f"  │ Gated fusion:     {fusion_params:,} (trainable, lr={args.lr})")
print(f"  │ CTC head:         {ctc_params:,} (trainable, lr={args.lr})")
print(f"  │ Total trainable:  {total_trainable:,}")
print(f"  │ Encoder LR: {args.encoder_lr} | Fusion+CTC LR: {args.lr}")
print(f"  └──────────────────────────────────────────────────────────────")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  GPU Memory: {allocated:.0f} MB allocated")

print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Helper functions for audio processing
# ══════════════════════════════════════════════════════════════════════════════


def load_audio(path, max_sec=None):
    """
    Load an audio file and convert to 16kHz mono waveform.

    Args:
        path (str): Path to audio file (mp3 or wav)
        max_sec (float, optional): Truncate audio to this many seconds

    Returns:
        Tensor: 1D tensor of audio samples at 16kHz
    """
    wav, sr = torchaudio.load(path)
    # Resample to 16kHz if needed (Whisper requirement)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # Convert stereo to mono by averaging channels
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()
    # Truncate if too long
    if max_sec and len(wav) > int(max_sec * SAMPLE_RATE):
        wav = wav[:int(max_sec * SAMPLE_RATE)]
    return wav


def compute_real_frames(num_samples):
    """
    Compute the number of real encoder output frames for a given audio length.

    Whisper's mel spectrogram uses stride=160 samples, and the conv layers
    further downsample by 2x, so: frames = num_samples // 160 // 2.
    Capped at 1500 (Whisper's max sequence length for 30s audio).

    Args:
        num_samples (int): Number of audio samples at 16kHz

    Returns:
        int: Number of encoder output frames (max 1500)
    """
    return min(num_samples // 160 // 2, 1500)


def apply_spec_augment(mel_features):
    """
    Apply SpecAugment data augmentation to mel spectrogram features.

    Randomly masks frequency bands and time steps in the mel spectrogram
    to improve model robustness. The same augmented mel is fed to BOTH
    encoders so they see identical input.

    Config: 2 frequency masks (max width 15 bins), 2 time masks (max width 50 frames)

    Args:
        mel_features (Tensor): Shape (1, 80, T) — mel spectrogram

    Returns:
        Tensor: Shape (1, 80, T) — augmented mel spectrogram (modified in-place)
    """
    _, n_freq, n_time = mel_features.shape

    # Frequency masking: zero out random frequency bands
    for _ in range(2):
        f = random.randint(0, 15)  # mask width
        f0 = random.randint(0, max(n_freq - f, 1) - 1)  # mask start
        mel_features[:, f0:f0 + f, :] = 0.0

    # Time masking: zero out random time segments
    for _ in range(2):
        t = random.randint(0, 50)  # mask width
        t0 = random.randint(0, max(n_time - t, 1) - 1)  # mask start
        mel_features[:, :, t0:t0 + t] = 0.0

    return mel_features


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Component verification (quick sanity check before training)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Component verification")
print("=" * 70)

# Run one forward + backward pass to verify everything works
test_clip = train_clips[0]
test_wav = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"  Test: {test_clip['clip_name']} ({len(test_wav)/SAMPLE_RATE:.1f}s, {test_clip['language']})")

# Mel spectrogram + SpecAugment
mel = feat_extractor(test_wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
mel_aug = apply_spec_augment(mel.input_features.clone())
print(f"  Mel: {tuple(mel.input_features.shape)} → SpecAugment applied")

# Forward pass through both encoders (WITH gradients — encoders are unfrozen)
acoustic_out = acoustic_encoder(mel_aug.to(DEVICE)).last_hidden_state
semantic_out = semantic_encoder(mel_aug.to(DEVICE)).last_hidden_state

# Trim to real frames (ignore padding frames beyond actual audio length)
real_frames = compute_real_frames(len(test_wav))
acoustic_feat = acoustic_out[:, :real_frames, :]
semantic_feat = semantic_out[:, :real_frames, :]
print(f"  Acoustic encoder: {tuple(acoustic_feat.shape)}")
print(f"  Semantic encoder: {tuple(semantic_feat.shape)}")

# Fusion forward
fused = fusion(acoustic_feat, semantic_feat)
print(f"  Gated fusion: {tuple(fused.shape)}")

# CTC Head forward
char_logits = ctc_head(fused)
print(f"  CTC head: {tuple(char_logits.shape)}")

# CTC loss + backward pass to verify gradients
gt = normalize_text(test_clip["ground_truth"])
target_indices = text_to_indices(gt, char_to_idx)
if target_indices and real_frames > len(target_indices):
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)
    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
    ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    print(f"  CTC loss: {ctc_loss.item():.4f}")
    print(f"  Ground truth: '{gt}' ({len(target_indices)} chars)")

    # Verify gradients flow through ALL components
    ctc_loss.backward()
    fusion_grads = sum(1 for p in fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    ctc_grads = sum(1 for p in ctc_head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    acoustic_grads = sum(1 for p in acoustic_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    semantic_grads = sum(1 for p in semantic_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Gradients: fusion={fusion_grads}>0 | ctc_head={ctc_grads}>0 | "
          f"acoustic_enc={acoustic_grads}>0 | semantic_enc={semantic_grads}>0")

    # Quick decode check
    pred = ctc_greedy_decode(char_logits.squeeze(0).detach(), idx_to_char)
    print(f"  Decode test: '{pred[:80]}...'")

fusion.zero_grad()
ctc_head.zero_grad()
acoustic_encoder.zero_grad()
semantic_encoder.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ALL VERIFIED\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Training setup (optimizer, scheduler, loss)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Training setup")
print("=" * 70)

# Collect parameters into two groups with DIFFERENT learning rates:
#   - Encoder params: small LR (1e-5) — don't destroy pre-trained features
#   - Fusion + CTC params: large LR (1e-3) — these need to learn fast
encoder_params = list(acoustic_encoder.parameters()) + list(semantic_encoder.parameters())
fusion_ctc_params = list(fusion.parameters()) + list(ctc_head.parameters())
trainable_params = encoder_params + fusion_ctc_params

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.encoder_lr},    # 1e-5 for encoders
    {"params": fusion_ctc_params, "lr": args.lr},          # 1e-3 for fusion + CTC
], weight_decay=0.01)

num_epochs = args.epochs
total_steps_est = len(train_clips) * num_epochs
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


# Learning rate schedule: linear warmup → cosine decay to zero
def lr_lambda(step):
    """Compute LR multiplier for current step: warmup then cosine decay."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)  # Linear warmup from 0 to 1
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))  # Cosine decay


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Restore optimizer/scheduler state if resuming
if args.resume:
    resume_ckpt = torch.load(PROJECT_ROOT / args.resume, map_location=DEVICE, weights_only=False)
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    del resume_ckpt
    gc.collect()

ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

print(f"  Optimizer: AdamW (encoder_lr={args.encoder_lr}, fusion_lr={args.lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) + cosine decay")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)}")
print(f"  Trainable: {total_trainable:,} params (encoders + fusion + CTC head)")
print(f"  Dropout: {args.dropout}")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience} (on dev WER)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Dev evaluation function
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dev(dev_clips_list):
    """
    Evaluate the fused model on the dev set and compute WER.

    Runs inference (no gradients) on all dev clips, computes CTC loss and
    performs greedy decoding to get predicted text, then computes per-language
    and overall Word Error Rate.

    Args:
        dev_clips_list (list[dict]): List of dev clip dicts from load_clips_from_csv

    Returns:
        tuple: (dev_wer, dev_loss, all_preds, lang_wers) where:
            - dev_wer (float): Overall WER as percentage (e.g., 18.5)
            - dev_loss (float): Average CTC loss on dev set
            - all_preds (list[tuple]): List of (reference, hypothesis) pairs
            - lang_wers (dict): Per-language WER: {"hi": 15.2, "mr": 25.0, "en": 16.0}
    """
    # Set all modules to eval mode (disables dropout, uses running stats for BatchNorm)
    fusion.eval()
    ctc_head.eval()
    acoustic_encoder.eval()
    semantic_encoder.eval()

    all_refs = []
    all_hyps = []
    losses = []
    lang_refs = {"hi": [], "mr": [], "en": []}
    lang_hyps = {"hi": [], "mr": [], "en": []}

    with torch.no_grad():  # No gradients needed during evaluation
        for clip in tqdm(dev_clips_list, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(wav) < 8000:  # Skip clips shorter than 0.5s
                    continue

                # Mel spectrogram (NO SpecAugment during eval)
                mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")

                # Forward pass through both encoders
                a_out = acoustic_encoder(mel.input_features.to(DEVICE)).last_hidden_state
                s_out = semantic_encoder(mel.input_features.to(DEVICE)).last_hidden_state
                real_frames = compute_real_frames(len(wav))
                a_feat = a_out[:, :real_frames, :]
                s_feat = s_out[:, :real_frames, :]

                # Fusion + CTC head
                fused = fusion(a_feat, s_feat)
                logits = ctc_head(fused)  # (1, T, 85)

                # Compute CTC loss for monitoring
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
                    del a_out, s_out, a_feat, s_feat, fused, logits
                    torch.cuda.empty_cache()

            except Exception:
                continue

    # Restore training mode for all modules
    fusion.train()
    ctc_head.train()
    acoustic_encoder.train()
    semantic_encoder.train()

    dev_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0
    dev_loss = np.mean(losses) if losses else 0.0

    lang_wers = {}
    for lc in ("hi", "mr", "en"):
        if lang_refs[lc]:
            lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
        else:
            lang_wers[lc] = None

    return dev_wer, dev_loss, list(zip(all_refs, all_hyps)), lang_wers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
mode_str = f"VERIFICATION ({len(train_clips)} clips)" if args.verify else "Full training"
print(f"STEP 7: {mode_str}")
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

# Ensure all modules are in training mode
fusion.train()
ctc_head.train()
acoustic_encoder.train()
semantic_encoder.train()

for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
    random.shuffle(train_clips)  # Shuffle data each epoch

    ep_losses = []
    ep_start = time.time()
    skipped = 0

    pbar = tqdm(train_clips, desc=f"Epoch {epoch}",
                unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # ── 1. Load audio waveform ──
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:  # Skip clips shorter than 0.5s
                skipped += 1
                continue

            num_samples = len(wav)

            # ── 2. Compute mel spectrogram + apply SpecAugment ──
            mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            mel_features = apply_spec_augment(mel.input_features.clone())
            mel_gpu = mel_features.to(DEVICE)

            # ── 3. Forward pass through both encoders (WITH gradients) ──
            a_out = acoustic_encoder(mel_gpu).last_hidden_state  # (1, 1500, 768)
            s_out = semantic_encoder(mel_gpu).last_hidden_state  # (1, 1500, 768)

            # Trim to real frames (drop padding beyond actual audio)
            real_frames = compute_real_frames(num_samples)
            a_feat = a_out[:, :real_frames, :]  # (1, T, 768)
            s_feat = s_out[:, :real_frames, :]  # (1, T, 768)

            # ── 4. Gated fusion + CTC head ──
            fused = fusion(a_feat, s_feat)  # (1, T, 768)
            logits = ctc_head(fused)  # (1, T, 85)

            # ── 5. Compute CTC loss ──
            gt = normalize_text(clip["ground_truth"])
            target_indices = text_to_indices(gt, char_to_idx)

            # Skip if target is empty or longer than available frames
            if not target_indices or real_frames <= len(target_indices):
                skipped += 1
                continue

            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85) — CTC format
            targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            # Skip NaN/Inf losses (rare edge cases)
            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            # ── 6. Backward pass + parameter update ──
            optimizer.zero_grad()
            loss.backward()  # Gradients flow through: CTC head → fusion → BOTH encoders
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  # Prevent explosion
            optimizer.step()
            scheduler.step()
            global_step += 1

            ep_losses.append(loss.item())

            # ── 7. Free GPU memory ──
            if DEVICE == "cuda":
                del a_out, s_out, a_feat, s_feat, fused, logits, mel_gpu
                torch.cuda.empty_cache()

            # Update progress bar
            if ep_losses:
                avg_recent = np.mean(ep_losses[-20:])
                pbar.set_postfix({
                    "loss": f"{ep_losses[-1]:.3f}",
                    "avg": f"{avg_recent:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                })

        except Exception as e:
            print(f"\n  ERROR step {global_step}: {clip['clip_name']}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

    pbar.close()

    # ── Epoch summary ────────────────────────────────────────────────────
    ep_time = time.time() - ep_start
    avg_loss = np.mean(ep_losses) if ep_losses else 0

    print(f"\n  ┌─ Epoch {epoch} ─────────────────────────────────────────────────")
    print(f"  │ CTC loss:   {avg_loss:.4f}")
    print(f"  │ Steps: {len(ep_losses)} ({skipped} skipped) | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
    print(f"  │ LR: encoder={scheduler.get_last_lr()[0]:.2e}, fusion={scheduler.get_last_lr()[1]:.2e}")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  │ Peak GPU: {peak:.0f} MB")
    print(f"  └──────────────────────────────────────────────────────────────")

    # ── Dev evaluation ───────────────────────────────────────────────────
    dev_wer, dev_loss, dev_preds, lang_wers = 999.0, 0.0, [], {}
    if dev_clips:
        dev_wer, dev_loss, dev_preds, lang_wers = evaluate_dev(dev_clips)
        print(f"\n  ┌─ Dev Results ───────────────────────────────────────────────")
        print(f"  │ WER:  {dev_wer:.2f}%")
        for lc, lname in [("hi", "Hindi"), ("mr", "Marathi"), ("en", "English")]:
            if lang_wers.get(lc) is not None:
                print(f"  │   {lname}: {lang_wers[lc]:.2f}%")
        print(f"  │ Loss: {dev_loss:.4f}")
        print(f"  └──────────────────────────────────────────────────────────────")

        # Show sample predictions
        if dev_preds:
            n_show = min(3, len(dev_preds))
            print(f"\n  Sample predictions:")
            for ref, hyp in dev_preds[:n_show]:
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
                print()

    # Save epoch stats
    epoch_stats.append({
        "epoch": epoch,
        "train_loss": avg_loss,
        "dev_wer": dev_wer, "dev_loss": dev_loss,
        "lang_wers": lang_wers,
    })

    # ── Checkpoint saving ────────────────────────────────────────────────
    if not args.verify:
        ckpt_data = {
            "epoch": epoch,
            # Model weights
            "fusion_state_dict": fusion.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "acoustic_encoder_state_dict": acoustic_encoder.state_dict(),
            "semantic_encoder_state_dict": semantic_encoder.state_dict(),
            # Optimizer state (for resume)
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # Metrics
            "train_loss": avg_loss,
            "dev_wer": dev_wer,
            "dev_loss": dev_loss,
            "lang_wers": lang_wers,
            "vocab_size": vocab_size,
            "global_step": global_step,
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Final results summary
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 8: Results")
print("=" * 70)

if epoch_stats:
    # Print epoch-by-epoch table
    print(f"\n  {'Epoch':<8} {'Loss':>8} {'DevWER':>10} {'DevLoss':>8} {'Hindi':>8} {'Marathi':>8} {'English':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for s in epoch_stats:
        wer_str = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        hi_str = f"{s['lang_wers'].get('hi', 0):.2f}%" if s['lang_wers'].get('hi') is not None else "N/A"
        mr_str = f"{s['lang_wers'].get('mr', 0):.2f}%" if s['lang_wers'].get('mr') is not None else "N/A"
        en_str = f"{s['lang_wers'].get('en', 0):.2f}%" if s['lang_wers'].get('en') is not None else "N/A"
        print(f"  {s['epoch']:<8} {s['train_loss']:>8.4f} {wer_str:>10} {s['dev_loss']:>8.4f} "
              f"{hi_str:>8} {mr_str:>8} {en_str:>8}")

    # Print learning summary
    if len(epoch_stats) >= 2:
        first_loss = epoch_stats[0]["train_loss"]
        last_loss = epoch_stats[-1]["train_loss"]
        if first_loss > 0:
            reduction = (first_loss - last_loss) / first_loss * 100
            print(f"\n  Loss reduction: {reduction:.1f}% ({first_loss:.4f} -> {last_loss:.4f})")
            if reduction > 0:
                print(f"  LEARNING CONFIRMED")
            else:
                print(f"  WARNING -- loss not decreasing")

    # Print comparison with individual encoders
    if best_dev_wer < 999:
        print(f"\n  Best dev WER: {best_dev_wer:.2f}%")
        print(f"\n  Comparison:")
        print(f"    Acoustic branch (Kid-Whisper):     19.97% (Hi 15.72% | Mr 26.37% | En 30.03%)")
        print(f"    Semantic branch (IndicConformer):   49.02% (Hi 37.95% | Mr 74.36% | En 56.82%)")
        print(f"    Combined (this run):                {best_dev_wer:.2f}%")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
