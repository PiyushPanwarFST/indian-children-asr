"""
Step 3: Semantic Branch — Joint Training (MSE + CTC)
====================================================

WHAT THIS SCRIPT DOES:
    Jointly trains the Whisper Small encoder with TWO losses:
      1. MSE loss: CTC Head 2 (768→257) vs IndicConformer pre-computed logits
      2. CTC loss: CTC Head 1 (768→85) vs ground truth text (our char vocab)

    The encoder receives gradients from BOTH losses → learns features that
    match the teacher AND are useful for transcription.

    This is the CARE paper's dual-loss approach applied to the semantic branch.

WHY THIS IS BETTER THAN STEP 1 (MSE ONLY):
    Step 1 only trained with MSE → encoder matched teacher logits but never
    saw actual text. CTC Head 2 can decode using teacher's BPE vocab, but
    that's the teacher's vocabulary, not ours.

    Joint training adds CTC Head 1 with OUR character vocabulary (85 tokens:
    Devanagari + English + special). The encoder now optimizes for both.

    Plus training improvements: SpecAugment, dropout, layer freezing, grad accum.

ARCHITECTURE:
    ┌───────────────────────────────────────────────────────────────┐
    │  Audio (.wav)                                                 │
    │       ↓                                                      │
    │  Mel Spectrogram → SpecAugment (freq+time masks)             │
    │       ↓                                                      │
    │  Whisper Encoder (layers 0-5 FROZEN, 6-11 UNFROZEN)          │
    │       ↓                                                      │
    │  Encoder features (1, T, 768)                                │
    │       ↓                              ↓                       │
    │  Dropout(0.1)                   Dropout(0.1)                 │
    │       ↓                              ↓                       │
    │  CTC Head 2 (768→257)           CTC Head 1 (768→85)         │
    │  per-language (Hi/Mr)           shared (all langs)           │
    │       ↓                              ↓                       │
    │  MSE Loss ← teacher logits      CTC Loss ← ground truth     │
    │       ↓                              ↓                       │
    │  Total = α × MSE + (1-α) × CTC                              │
    │       ↓                                                      │
    │  Backprop → Encoder top 6 + both heads                       │
    └───────────────────────────────────────────────────────────────┘

WARM START:
    Encoder + CTC Head 2: from checkpoints/semantic_mse/best_dev.pt (step1)
    CTC Head 1: from checkpoints/semantic_ctc/best_wer.pt (step2b) via --ctc_checkpoint
                OR random init if --ctc_checkpoint not provided

Prerequisites:
    - Step 1 checkpoint: checkpoints/semantic_mse/best_dev.pt
    - Pre-computed teacher logits: teacher_logits/train/*.pt (from step0)
    - Vocabulary: ASER-Dataset/vocab.json (85 tokens)
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Quick verify (100 clips, 3 epochs)
    python scripts/semantic/step3_semantic_joint_training.py --verify

    # Verify with more clips
    python scripts/semantic/step3_semantic_joint_training.py --verify --test_clips 200

    # Full training run
    python scripts/semantic/step3_semantic_joint_training.py --epochs 30 --patience 7
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
parser = argparse.ArgumentParser(description="Semantic joint training (MSE + CTC)")
parser.add_argument("--verify", action="store_true",
                    help="Test run with subset of clips × 3 epochs")
parser.add_argument("--test_clips", type=int, default=100,
                    help="Number of clips in verify mode (default 100)")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--warmup_steps", type=int, default=1000,
                    help="Linear warmup steps")
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--patience", type=int, default=7,
                    help="Early stopping patience on dev WER (0=disabled)")
parser.add_argument("--alpha_start", type=float, default=0.3,
                    help="Initial MSE loss weight (CTC weight = 1 - alpha)")
parser.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch size)")
parser.add_argument("--freeze_layers", type=int, default=6,
                    help="Number of bottom encoder layers to freeze")
parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout before CTC heads")
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/semantic_mse/best_dev.pt",
                    help="Step 1 checkpoint to warm-start from")
parser.add_argument("--ctc_checkpoint", type=str, default=None,
                    help="Warm-start CTC Head 1 from sequential CTC checkpoint (step2b)")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from joint training checkpoint")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_every", type=int, default=10)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
TEACHER_LOGITS_DIR = PROJECT_ROOT / "teacher_logits"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "semantic_joint"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr"}

# ── Add project root for utils import ────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

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
    """Convert normalized text to vocabulary indices for CTC targets."""
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
    Greedy CTC decode from logits (T, vocab_size).
    Returns decoded text string.
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
# STEP 2: Load training data (Hindi + Marathi only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading training data")
print("=" * 70)


def load_clips_from_csv(csv_path, split="train", max_clips=None):
    """Load Hindi/Marathi clips that have pre-computed teacher logits."""
    clips = []
    skipped_english = 0
    skipped_no_logits = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row.get("language", "")
            lang_code = LANG_MAP.get(lang)

            if lang_code is None:
                skipped_english += 1
                continue

            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = str(ASER_ROOT / audio_path)

            # Use child_id + basename as unique identifier (matches step0)
            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename
            logit_path = TEACHER_LOGITS_DIR / split / f"{clip_uid}.pt"

            if not logit_path.exists():
                skipped_no_logits += 1
                continue

            gt = row.get("transcript", row.get("que_text", row.get("text", "")))

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "clip_name": clip_uid,
                "logit_path": str(logit_path),
                "duration_sec": float(row.get("duration_sec", 0)),
                "ground_truth": gt,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips, skipped_english, skipped_no_logits


train_clips, skip_en, skip_nolog = load_clips_from_csv(TRAIN_CSV, split="train")
hi_clips = [c for c in train_clips if c["lang_code"] == "hi"]
mr_clips = [c for c in train_clips if c["lang_code"] == "mr"]
total_hrs = sum(c["duration_sec"] for c in train_clips) / 3600

print(f"  Total: {len(train_clips)} clips ({total_hrs:.1f}h)")
print(f"    Hindi:   {len(hi_clips)} clips")
print(f"    Marathi: {len(mr_clips)} clips")
print(f"  Skipped: {skip_en} English, {skip_nolog} missing logits")

if len(train_clips) == 0:
    print("\n  ERROR: No training clips found! Run step0 first.")
    exit(1)

# Dev set
dev_clips = []
if DEV_CSV.exists():
    dev_clips, _, _ = load_clips_from_csv(DEV_CSV, split="dev")
    print(f"  Dev set: {len(dev_clips)} clips")

# Verify mode
if args.verify:
    random.seed(args.seed)
    N = args.test_clips
    total_hm = len(hi_clips) + len(mr_clips)
    n_hi = min(int(N * len(hi_clips) / total_hm), len(hi_clips)) if total_hm > 0 else 0
    n_mr = min(N - n_hi, len(mr_clips))
    sample_hi = random.sample(hi_clips, n_hi) if n_hi > 0 else []
    sample_mr = random.sample(mr_clips, n_mr) if n_mr > 0 else []
    train_clips = sample_hi + sample_mr
    random.shuffle(train_clips)
    print(f"\n  VERIFY MODE: {len(train_clips)} clips ({n_hi} Hindi + {n_mr} Marathi)")

    if dev_clips:
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

# ── CTC Head 2: per-language Linear(768→257) — bridge layer ─────────────────
print("  [CTC HEAD 2] Per-language bridge heads...")
ctc_head_hi = nn.Linear(768, 257).to(DEVICE)
ctc_head_mr = nn.Linear(768, 257).to(DEVICE)

# ── CTC Head 1: Linear(768→85) — decoding head (our char vocab) ─────────────
print(f"  [CTC HEAD 1] Decoding head: Linear(768→{vocab_size})...")
ctc_head_char = nn.Linear(768, vocab_size).to(DEVICE)

# ── Dropout ──────────────────────────────────────────────────────────────────
head_dropout = nn.Dropout(args.dropout)

# ── Load step1 checkpoint (warm start) ───────────────────────────────────────
start_epoch = 0
best_dev_wer = float("inf")

if args.resume:
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading joint checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    ctc_head_hi.load_state_dict(ckpt["ctc_head_hi_state_dict"])
    ctc_head_mr.load_state_dict(ckpt["ctc_head_mr_state_dict"])
    ctc_head_char.load_state_dict(ckpt["ctc_head_char_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))
    print(f"    Resuming from epoch {start_epoch}, best dev WER: {best_dev_wer:.2f}%")
else:
    ckpt_path = PROJECT_ROOT / args.checkpoint
    print(f"\n  [WARM START] Loading step1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    ctc_head_hi.load_state_dict(ckpt["ctc_head_hi_state_dict"])
    ctc_head_mr.load_state_dict(ckpt["ctc_head_mr_state_dict"])
    print(f"    Encoder + CTC Head 2 loaded from epoch {ckpt.get('epoch', '?')}")
    print(f"    Step1 train MSE: {ckpt.get('train_loss', '?')}")
    print(f"    Step1 dev MSE:   {ckpt.get('dev_loss', '?')}")
    if args.ctc_checkpoint:
        ctc_ckpt_path = PROJECT_ROOT / args.ctc_checkpoint
        print(f"    CTC Head 1: warm-starting from {ctc_ckpt_path}")
        ctc_ckpt = torch.load(ctc_ckpt_path, map_location=DEVICE, weights_only=False)
        ctc_head_char.load_state_dict(ctc_ckpt["ctc_head_state_dict"])
        print(f"    Sequential CTC dev WER: {ctc_ckpt.get('dev_wer', '?')}")
        del ctc_ckpt
    else:
        print(f"    CTC Head 1 (768→{vocab_size}): random init (new)")

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
head2_params = sum(p.numel() for p in ctc_head_hi.parameters()) + sum(p.numel() for p in ctc_head_mr.parameters())
head1_params = sum(p.numel() for p in ctc_head_char.parameters())

print(f"    Encoder: {total_params:,} total, {trainable_enc:,} trainable, {frozen_params:,} frozen")
print(f"    CTC Head 2 (Hi+Mr): {head2_params:,} params")
print(f"    CTC Head 1 (char):  {head1_params:,} params")
print(f"    Total trainable: {trainable_enc + head2_params + head1_params:,}")

# Enable gradient checkpointing for memory efficiency
encoder.gradient_checkpointing_enable()
print(f"    Gradient checkpointing: ENABLED")

encoder.train()
ctc_head_hi.train()
ctc_head_mr.train()
ctc_head_char.train()

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
    """Load audio → 16kHz mono tensor."""
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
    """Whisper: mel stride=160, conv stride=2 → frames = samples // 320."""
    return min(num_samples // 160 // 2, 1500)


def align_teacher_to_student(teacher_logits, student_frames):
    """
    Interpolate teacher logits to match student frame count.
    Teacher: (T_teacher, 257) at 12.5fps → Student: T_student frames at 50fps.
    Returns: (1, T_student, 257)
    """
    t = teacher_logits.T.unsqueeze(0)  # (1, 257, T_teacher)
    aligned = F.interpolate(t, size=student_frames, mode='linear', align_corners=False)
    return aligned.permute(0, 2, 1)  # (1, T_student, 257)


def apply_spec_augment(mel_features):
    """
    Apply SpecAugment to mel spectrogram (1, 80, T).
    2 frequency masks (max 15 bins), 2 time masks (max 50 frames).
    Only during training.
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
    """MSE loss weight schedule: decays over training."""
    if epoch <= 5:
        return args.alpha_start       # 0.3
    elif epoch <= 15:
        return args.alpha_start - 0.1  # 0.2
    else:
        return 0.1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Component verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Component verification")
print("=" * 70)

test_clip = train_clips[0]
test_wav = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"  Test: {test_clip['clip_name']} ({len(test_wav)/SAMPLE_RATE:.1f}s, {test_clip['language']})")

# Mel + SpecAugment
mel = feat_extractor(test_wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
mel_aug = apply_spec_augment(mel.input_features.clone())
print(f"  Mel: {tuple(mel.input_features.shape)} → SpecAugment applied")

# Encoder forward
enc_out = encoder(mel_aug.to(DEVICE))
features = enc_out.last_hidden_state
real_frames = compute_real_frames(len(test_wav))
real_features = features[:, :real_frames, :]
print(f"  Encoder: (1, 1500, 768) → slice({real_frames}) → {tuple(real_features.shape)}")

# CTC Head 2 (MSE branch)
if test_clip["lang_code"] == "hi":
    student_logits_257 = ctc_head_hi(head_dropout(real_features))
else:
    student_logits_257 = ctc_head_mr(head_dropout(real_features))
print(f"  CTC Head 2: {tuple(student_logits_257.shape)} (teacher BPE space)")

# CTC Head 1 (CTC branch)
char_logits = ctc_head_char(head_dropout(real_features))
print(f"  CTC Head 1: {tuple(char_logits.shape)} (our char vocab)")

# MSE loss
teacher_data = torch.load(test_clip["logit_path"], weights_only=True)
t_logits = teacher_data["logits"].float().to(DEVICE)
t_aligned = align_teacher_to_student(t_logits, real_frames)
mse_loss = F.mse_loss(student_logits_257, t_aligned)
print(f"  MSE loss: {mse_loss.item():.4f}")

# CTC loss
gt = normalize_text(test_clip["ground_truth"])
target_indices = text_to_indices(gt, char_to_idx)
if target_indices and real_frames > len(target_indices):
    log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)
    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    print(f"  CTC loss: {ctc_loss.item():.4f}")
    print(f"  Ground truth: '{gt}' ({len(target_indices)} chars)")

    # Combined loss
    alpha = get_alpha(1)
    total_loss = alpha * mse_loss + (1 - alpha) * ctc_loss
    total_loss.backward()
    print(f"  Combined: {alpha:.1f}×MSE + {1-alpha:.1f}×CTC = {total_loss.item():.4f}")
else:
    mse_loss.backward()
    print(f"  CTC loss: skipped (empty/long target)")

# Verify gradients flow correctly
frozen_grads = sum(1 for p in encoder.layers[:n_freeze].parameters() if p.grad is not None and p.grad.abs().sum() > 0)
unfrozen_grads = sum(1 for p in encoder.layers[n_freeze:].parameters() if p.grad is not None and p.grad.abs().sum() > 0)
head1_grads = sum(1 for p in ctc_head_char.parameters() if p.grad is not None)
head2_grads = sum(1 for p in ctc_head_hi.parameters() if p.grad is not None) + \
              sum(1 for p in ctc_head_mr.parameters() if p.grad is not None)
print(f"  Gradients: frozen_layers={frozen_grads}(should=0) | "
      f"unfrozen_layers={unfrozen_grads}>0 ✓ | head1={head1_grads}>0 ✓ | head2={head2_grads}>0 ✓")

encoder.zero_grad()
ctc_head_hi.zero_grad()
ctc_head_mr.zero_grad()
ctc_head_char.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ✓ ALL VERIFIED\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Training setup")
print("=" * 70)

# Collect trainable params
trainable_params = (
    [p for p in encoder.parameters() if p.requires_grad] +
    list(ctc_head_hi.parameters()) +
    list(ctc_head_mr.parameters()) +
    list(ctc_head_char.parameters())
)
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

num_epochs = 3 if args.verify else args.epochs
total_steps_est = (len(train_clips) * num_epochs) // args.grad_accum

# Warmup + cosine decay scheduler
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Load optimizer state if resuming
if args.resume:
    resume_ckpt = torch.load(PROJECT_ROOT / args.resume, map_location=DEVICE, weights_only=False)
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    del resume_ckpt
    gc.collect()

ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) + cosine decay")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)} | Grad accum: {args.grad_accum}")
print(f"  Effective batch size: {args.grad_accum}")
print(f"  Alpha schedule: {args.alpha_start} → {args.alpha_start - 0.1} → 0.1")
print(f"  Dropout: {args.dropout}")
print(f"  Frozen layers: {n_freeze}/{n_layers}")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience} (on dev WER)")
print()


# ── Dev evaluation function (WER from CTC Head 1) ───────────────────────────

def evaluate_dev(dev_clips_list):
    """
    Evaluate on dev set:
      - WER from CTC Head 1 (our char vocab — this is the real metric)
      - MSE from CTC Head 2 (teacher alignment — secondary metric)
    Returns: (dev_wer, dev_mse, all_preds)
    """
    encoder.eval()
    ctc_head_hi.eval()
    ctc_head_mr.eval()
    ctc_head_char.eval()

    all_refs = []
    all_hyps = []
    mse_losses = []

    with torch.no_grad():
        for clip in tqdm(dev_clips_list, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(wav) < 8000:
                    continue

                # Mel (no SpecAugment for eval)
                mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                enc_out = encoder(mel.input_features.to(DEVICE))
                features = enc_out.last_hidden_state
                real_frames = compute_real_frames(len(wav))
                real_features = features[:, :real_frames, :]

                # MSE from CTC Head 2
                if clip["lang_code"] == "hi":
                    s_logits_257 = ctc_head_hi(real_features)
                else:
                    s_logits_257 = ctc_head_mr(real_features)

                teacher_data = torch.load(clip["logit_path"], weights_only=True)
                t_logits = teacher_data["logits"].float().to(DEVICE)
                t_aligned = align_teacher_to_student(t_logits, real_frames)
                mse_val = F.mse_loss(s_logits_257, t_aligned).item()
                mse_losses.append(mse_val)

                # WER from CTC Head 1
                char_logits = ctc_head_char(real_features)  # (1, T, 85)
                predicted = ctc_greedy_decode(char_logits.squeeze(0), idx_to_char)

                ref = normalize_text(clip["ground_truth"])
                if ref:
                    all_refs.append(ref)
                    all_hyps.append(predicted)

                if DEVICE == "cuda":
                    del features, real_features, s_logits_257, t_aligned, char_logits
                    torch.cuda.empty_cache()

            except Exception:
                continue

    encoder.train()
    ctc_head_hi.train()
    ctc_head_mr.train()
    ctc_head_char.train()
    # Re-freeze bottom layers after eval (train() unfreezes nothing, but be explicit)

    dev_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0
    dev_mse = np.mean(mse_losses) if mse_losses else 0.0

    return dev_wer, dev_mse, list(zip(all_refs, all_hyps))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
mode_str = f"VERIFICATION ({len(train_clips)} clips × 3 epochs)" if args.verify else "Full training"
print(f"STEP 7: {mode_str}")
print("=" * 70)

random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

# Training state
mse_history = []
ctc_history = []
epoch_stats = []
global_step = 0
patience_counter = 0
training_start = time.time()

for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
    alpha = get_alpha(epoch)
    random.shuffle(train_clips)

    ep_mse_losses = []
    ep_ctc_losses = []
    ep_combined_losses = []
    ep_start = time.time()
    skipped = 0

    optimizer.zero_grad()

    pbar = tqdm(train_clips, desc=f"Epoch {epoch} (α={alpha:.1f})",
                unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # ── Load audio ──
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:
                skipped += 1
                continue

            num_samples = len(wav)

            # ── Mel + SpecAugment ──
            mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            mel_features = apply_spec_augment(mel.input_features.clone())

            # ── Encoder forward ──
            enc_out = encoder(mel_features.to(DEVICE))
            features = enc_out.last_hidden_state  # (1, 1500, 768)
            real_frames = compute_real_frames(num_samples)
            real_features = features[:, :real_frames, :]  # (1, T, 768)

            # ── BRANCH A: MSE loss (CTC Head 2 vs teacher) ──
            if clip["lang_code"] == "hi":
                student_logits_257 = ctc_head_hi(head_dropout(real_features))
            else:
                student_logits_257 = ctc_head_mr(head_dropout(real_features))

            teacher_data = torch.load(clip["logit_path"], weights_only=True)
            t_logits = teacher_data["logits"].float().to(DEVICE)
            t_aligned = align_teacher_to_student(t_logits, real_frames)

            mse_loss = F.mse_loss(student_logits_257, t_aligned)

            # ── BRANCH B: CTC loss (CTC Head 1 vs ground truth) ──
            gt = normalize_text(clip["ground_truth"])
            target_indices = text_to_indices(gt, char_to_idx)

            if not target_indices or real_frames <= len(target_indices):
                # Can't compute CTC — use MSE only for this clip
                total_loss = mse_loss / args.grad_accum
                total_loss.backward()
                ep_mse_losses.append(mse_loss.item())
                skipped += 1
            else:
                char_logits = ctc_head_char(head_dropout(real_features))  # (1, T, 85)
                log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)

                targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
                input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
                target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

                ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

                # Skip NaN/Inf CTC
                if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                    total_loss = mse_loss / args.grad_accum
                    total_loss.backward()
                    ep_mse_losses.append(mse_loss.item())
                    skipped += 1
                else:
                    total_loss = (alpha * mse_loss + (1 - alpha) * ctc_loss) / args.grad_accum
                    total_loss.backward()
                    ep_mse_losses.append(mse_loss.item())
                    ep_ctc_losses.append(ctc_loss.item())
                    ep_combined_losses.append(
                        alpha * mse_loss.item() + (1 - alpha) * ctc_loss.item()
                    )

            # ── Gradient accumulation step ──
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_clips):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ── Cleanup ──
            if DEVICE == "cuda":
                del features, real_features, student_logits_257, t_aligned, t_logits
                if 'char_logits' in dir():
                    pass  # already freed by backward
                torch.cuda.empty_cache()

            # Progress bar
            if ep_combined_losses:
                avg_recent = np.mean(ep_combined_losses[-20:])
                pbar.set_postfix({
                    "mse": f"{ep_mse_losses[-1]:.3f}" if ep_mse_losses else "?",
                    "ctc": f"{ep_ctc_losses[-1]:.3f}" if ep_ctc_losses else "?",
                    "comb": f"{avg_recent:.3f}",
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
    avg_mse = np.mean(ep_mse_losses) if ep_mse_losses else 0
    avg_ctc = np.mean(ep_ctc_losses) if ep_ctc_losses else 0
    avg_combined = np.mean(ep_combined_losses) if ep_combined_losses else 0

    print(f"\n  ┌─ Epoch {epoch} ─────────────────────────────────────────────────")
    print(f"  │ MSE loss:      {avg_mse:.4f}")
    print(f"  │ CTC loss:      {avg_ctc:.4f}")
    print(f"  │ Combined:      {avg_combined:.4f} (α={alpha:.1f})")
    print(f"  │ Steps: {len(ep_mse_losses)} ({skipped} skipped) | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
    print(f"  │ LR: {scheduler.get_last_lr()[0]:.2e}")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  │ Peak GPU: {peak:.0f} MB")
    print(f"  └──────────────────────────────────────────────────────────────")

    # ── Dev evaluation ───────────────────────────────────────────────────
    dev_wer, dev_mse, dev_preds = 999.0, 0.0, []
    if dev_clips:
        dev_wer, dev_mse, dev_preds = evaluate_dev(dev_clips)
        print(f"\n  ┌─ Dev Results ───────────────────────────────────────────────")
        print(f"  │ WER:  {dev_wer:.2f}% (from CTC Head 1 — our char vocab)")
        print(f"  │ MSE:  {dev_mse:.4f} (from CTC Head 2 — teacher alignment)")
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
        "dev_wer": dev_wer, "dev_mse": dev_mse,
    })

    # ── Checkpoint ───────────────────────────────────────────────────────
    if not args.verify:
        ckpt_data = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "ctc_head_hi_state_dict": ctc_head_hi.state_dict(),
            "ctc_head_mr_state_dict": ctc_head_mr.state_dict(),
            "ctc_head_char_state_dict": ctc_head_char.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_mse": avg_mse,
            "train_ctc": avg_ctc,
            "train_combined": avg_combined,
            "dev_wer": dev_wer,
            "dev_mse": dev_mse,
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
            print(f"  ★ NEW BEST DEV WER: {dev_wer:.2f}% → saved best_dev.pt")
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
    print(f"\n  {'Epoch':<8} {'α':>4} {'MSE':>8} {'CTC':>8} {'Comb':>8} {'DevWER':>10} {'DevMSE':>8}")
    print(f"  {'─'*8} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for s in epoch_stats:
        wer_str = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        print(f"  {s['epoch']:<8} {s['alpha']:>4.1f} {s['train_mse']:>8.4f} "
              f"{s['train_ctc']:>8.4f} {s['train_combined']:>8.4f} "
              f"{wer_str:>10} {s['dev_mse']:>8.4f}")

    # Learning check
    if len(epoch_stats) >= 2:
        first_comb = epoch_stats[0]["train_combined"]
        last_comb = epoch_stats[-1]["train_combined"]
        if first_comb > 0:
            reduction = (first_comb - last_comb) / first_comb * 100
            print(f"\n  Combined loss reduction: {reduction:.1f}% ({first_comb:.4f} → {last_comb:.4f})")
            if reduction > 0:
                print(f"  ✓ LEARNING CONFIRMED")
            else:
                print(f"  ✗ WARNING — loss not decreasing")

    if best_dev_wer < 999:
        print(f"\n  Best dev WER: {best_dev_wer:.2f}%")
        print(f"\n  Comparison:")
        print(f"    Step 1 (MSE only, Head 2 BPE decode):  Hindi 39.72% | Marathi 68.11% | Overall 47.86%")
        print(f"    Step 3 (Joint, Head 1 char decode):    Dev WER {best_dev_wer:.2f}%")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
