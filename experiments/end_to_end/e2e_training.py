"""
End-to-End Dual-Encoder Training from Vanilla Whisper Small
============================================================

Trains the FULL dual-encoder gated fusion architecture from scratch:
  - Both encoders initialized from vanilla openai/whisper-small (NO KD checkpoints)
  - Multi-teacher KD: Kid-Whisper (acoustic, online) + IndicConformer (semantic, pre-computed)
  - All layers UNFROZEN from the start
  - Loss = w_ctc * CTC + w_acou * MSE_acoustic + w_sem * MSE_semantic + w_orth * L_orthogonality
  - Loss weight schedule: KD-heavy early → CTC-heavy late

Usage:
  # Verify (100 clips, 2 epochs):
  python e2e_training.py --verify

  # Test run (3000 clips, 10 epochs):
  python e2e_training.py --test_clips 3000 --epochs 10

  # Full run (40 epochs, all clips):
  python e2e_training.py --epochs 40 --patience 10
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

# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="End-to-End Dual-Encoder Training")

# Training hyperparameters
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience on dev WER (0 = disabled)")
parser.add_argument("--encoder_lr", type=float, default=3e-5,
                    help="Learning rate for both encoders")
parser.add_argument("--fusion_lr", type=float, default=1e-3,
                    help="Learning rate for fusion/CTC/projection/bridge heads")
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = grad_accum)")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--max_audio_sec", type=float, default=30.0)

# Loss weights (initial values — scheduled during training)
parser.add_argument("--w_ctc", type=float, default=0.3,
                    help="Initial CTC loss weight")
parser.add_argument("--w_acoustic", type=float, default=0.5,
                    help="Initial acoustic KD (MSE) weight")
parser.add_argument("--w_semantic", type=float, default=0.3,
                    help="Initial semantic KD (MSE) weight")
parser.add_argument("--w_ortho", type=float, default=0.01,
                    help="Orthogonality loss weight (constant)")

# Teacher models
parser.add_argument("--teacher_model", type=str,
                    default="aadel4/kid-whisper-medium-en-myst",
                    help="Kid-Whisper teacher model (online, frozen)")
parser.add_argument("--teacher_logits_dir", type=str, default="teacher_logits",
                    help="Directory with pre-computed IndicConformer logits")

# Modes
parser.add_argument("--test_clips", type=int, default=None,
                    help="Limit training to N clips (for test runs)")
parser.add_argument("--verify", action="store_true",
                    help="Quick verification with 100 clips, 2 epochs")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path (relative to project root)")
parser.add_argument("--grad_checkpoint", action="store_true",
                    help="Enable gradient checkpointing (saves ~40% memory)")
parser.add_argument("--specaugment", action="store_true", default=True,
                    help="Apply SpecAugment (default: on)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.verify:
    args.epochs = min(args.epochs, 2)
    if args.test_clips is None:
        args.test_clips = 100

# ══════════════════════════════════════════════════════════════════════════════
# PATHS AND CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# this file → end_to_end/ → experiments/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
TEACHER_LOGITS_DIR = PROJECT_ROOT / args.teacher_logits_dir

CKPT_DIR = PROJECT_ROOT / "checkpoints" / "e2e"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

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
vocab_size = len(char_to_idx)
BLANK_IDX = char_to_idx["<blank>"]

print(f"  Vocab: {vocab_size} tokens (from {VOCAB_PATH.name})")
print(f"  Blank index: {BLANK_IDX}")


def text_to_indices(text, char_to_idx):
    """Convert normalized text to list of vocabulary indices for CTC targets."""
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
    """Greedy CTC decoding: argmax → collapse repeats → remove blanks."""
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
# STEP 2: Load training and dev data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading training data")
print("=" * 70)


def load_clips_from_csv(csv_path, split="train", max_clips=None):
    """Load clip metadata from CSV. For Hindi/Marathi, also resolves teacher logit paths."""
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

            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            # Teacher logits for IndicConformer KD (Hindi/Marathi only)
            logit_path = None
            if lang_code in ("hi", "mr"):
                lp = TEACHER_LOGITS_DIR / split / f"{clip_uid}.pt"
                if lp.exists():
                    logit_path = str(lp)
                else:
                    # Don't skip — clip still has acoustic KD + CTC
                    pass

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
    exit(1)

dev_clips, _ = load_clips_from_csv(DEV_CSV, split="dev")
print(f"  Dev set: {len(dev_clips)} clips")

# ── Optionally limit clips for test runs ──────────────────────────────────
N = args.test_clips
if N and N < len(train_clips):
    random.seed(args.seed)
    total_all = len(train_clips)
    lang_clips = {"hi": hi_clips, "mr": mr_clips, "en": en_clips}
    sampled = []
    remaining = N

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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Initialize models from VANILLA Whisper Small
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Initializing models (vanilla Whisper Small)")
print("=" * 70)

import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration

WHISPER_ID = "openai/whisper-small"

# Feature extractor
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

# ── Encoder A (acoustic branch) — vanilla Whisper Small ───────────────────
print(f"\n  [ENCODER A] Loading vanilla Whisper Small...")
whisper_a = WhisperModel.from_pretrained(WHISPER_ID)
encoder_a = whisper_a.encoder.to(DEVICE)
encoder_a.train()
if args.grad_checkpoint:
    encoder_a.gradient_checkpointing_enable()
    print(f"    Gradient checkpointing ENABLED")
n_params_a = sum(p.numel() for p in encoder_a.parameters())
print(f"    UNFROZEN ({n_params_a:,} params) — will specialize via Kid-Whisper KD")
del whisper_a
gc.collect()

# ── Encoder B (semantic branch) — vanilla Whisper Small ───────────────────
print(f"\n  [ENCODER B] Loading vanilla Whisper Small...")
whisper_b = WhisperModel.from_pretrained(WHISPER_ID)
encoder_b = whisper_b.encoder.to(DEVICE)
encoder_b.train()
if args.grad_checkpoint:
    encoder_b.gradient_checkpointing_enable()
    print(f"    Gradient checkpointing ENABLED")
n_params_b = sum(p.numel() for p in encoder_b.parameters())
print(f"    UNFROZEN ({n_params_b:,} params) — will specialize via IndicConformer KD")
del whisper_b
gc.collect()

# ── Kid-Whisper Teacher (frozen, online) ──────────────────────────────────
print(f"\n  [TEACHER] Loading Kid-Whisper: {args.teacher_model}")
teacher_full = WhisperForConditionalGeneration.from_pretrained(args.teacher_model)
teacher_encoder = teacher_full.model.encoder.to(DEVICE)
teacher_dim = teacher_full.config.d_model  # 1024
teacher_encoder.eval()
for param in teacher_encoder.parameters():
    param.requires_grad = False
teacher_params = sum(p.numel() for p in teacher_encoder.parameters())
print(f"    Teacher dim: {teacher_dim}")
print(f"    Teacher params: {teacher_params:,} (all FROZEN)")
# Free decoder memory
del teacher_full.proj_out
del teacher_full.model.decoder
del teacher_full
gc.collect()

# ── Projection: student (768) → teacher (1024) for acoustic MSE ──────────
print(f"\n  [PROJECTION] Linear(768 → {teacher_dim})")
projection_a = nn.Linear(768, teacher_dim).to(DEVICE)

# ── Bridge heads: student (768) → IndicConformer vocab (257) for semantic MSE ──
print(f"  [BRIDGE] Linear(768 → 257) × 2 (Hindi + Marathi)")
bridge_hi = nn.Linear(768, 257).to(DEVICE)
bridge_mr = nn.Linear(768, 257).to(DEVICE)

# ── Gated Fusion Module ───────────────────────────────────────────────────
class GatedFusion(nn.Module):
    """Learnable gated fusion: gate = σ(W·[a;b]), fused = gate*a + (1-gate)*b."""
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
        return fused, gate  # return gate for monitoring


print(f"\n  [GATED FUSION] Creating fusion module...")
fusion = GatedFusion(dim=768, dropout=args.dropout).to(DEVICE)
fusion_params = sum(p.numel() for p in fusion.parameters())
print(f"    Params: {fusion_params:,}")

# ── CTC Head: fused features → character logits ──────────────────────────
print(f"\n  [CTC HEAD] Linear(768 → {vocab_size})")
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# ── Resume from checkpoint ────────────────────────────────────────────────
start_epoch = 0
best_dev_wer = float("inf")

if args.resume:
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder_a.load_state_dict(ckpt["encoder_a_state_dict"])
    encoder_b.load_state_dict(ckpt["encoder_b_state_dict"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
    projection_a.load_state_dict(ckpt["projection_a_state_dict"])
    bridge_hi.load_state_dict(ckpt["bridge_hi_state_dict"])
    bridge_mr.load_state_dict(ckpt["bridge_mr_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))
    print(f"    Resumed from epoch {start_epoch}, best dev WER: {best_dev_wer:.2f}%")
    del ckpt
    gc.collect()

# ── Parameter summary ─────────────────────────────────────────────────────
proj_params = sum(p.numel() for p in projection_a.parameters())
bridge_params = sum(p.numel() for p in bridge_hi.parameters()) + sum(p.numel() for p in bridge_mr.parameters())
ctc_params = sum(p.numel() for p in ctc_head.parameters())
total_trainable = n_params_a + n_params_b + fusion_params + proj_params + bridge_params + ctc_params

print(f"\n  ┌─ Parameter Summary ──────────────────────────────────────────")
print(f"  │ Encoder A (acoustic): {n_params_a:,} (unfrozen, lr={args.encoder_lr})")
print(f"  │ Encoder B (semantic): {n_params_b:,} (unfrozen, lr={args.encoder_lr})")
print(f"  │ Projection (768→{teacher_dim}): {proj_params:,} (lr={args.fusion_lr})")
print(f"  │ Bridge heads (768→257): {bridge_params:,} (lr={args.fusion_lr})")
print(f"  │ Gated fusion:        {fusion_params:,} (lr={args.fusion_lr})")
print(f"  │ CTC head (768→{vocab_size}): {ctc_params:,} (lr={args.fusion_lr})")
print(f"  │ Teacher (FROZEN):    {teacher_params:,}")
print(f"  │ Total trainable:     {total_trainable:,}")
print(f"  └──────────────────────────────────────────────────────────────")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  GPU Memory: {allocated:.0f} MB allocated")

print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Helper functions
# ══════════════════════════════════════════════════════════════════════════════


def load_audio(path, max_sec=None):
    """Load audio file → 16kHz mono waveform."""
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


def apply_spec_augment(mel_features):
    """SpecAugment: 2 freq masks (≤15), 2 time masks (≤50)."""
    _, n_freq, n_time = mel_features.shape
    for _ in range(2):
        f = random.randint(0, 15)
        f0 = random.randint(0, max(n_freq - f, 1) - 1)
        mel_features[:, f0:f0 + f, :] = 0.0
    for _ in range(2):
        t = random.randint(0, 50)
        t0 = random.randint(0, max(n_time - t, 1) - 1)
        mel_features[:, :, t0:t0 + t] = 0.0
    return mel_features


def align_teacher_to_student(teacher_logits, student_frames):
    """Interpolate teacher logits (T_teacher, 257) to student frame count → (1, T_student, 257)."""
    t = teacher_logits.T.unsqueeze(0)  # (1, 257, T_teacher)
    aligned = F.interpolate(t, size=student_frames, mode='linear', align_corners=False)
    return aligned.permute(0, 2, 1)  # (1, T_student, 257)


def get_loss_weights(epoch, total_epochs):
    """
    Scheduled loss weights: KD-heavy early → CTC-heavy late.

    Phase 1 (1-25%):  w_ctc=0.3, w_acou=0.5, w_sem=0.3  (KD-heavy)
    Phase 2 (25-62%): w_ctc=0.5, w_acou=0.3, w_sem=0.2  (balanced)
    Phase 3 (62-100%):w_ctc=0.7, w_acou=0.2, w_sem=0.1  (CTC-heavy)
    """
    progress = epoch / total_epochs
    if progress <= 0.25:
        return args.w_ctc, args.w_acoustic, args.w_semantic
    elif progress <= 0.625:
        return 0.5, 0.3, 0.2
    else:
        return 0.7, 0.2, 0.1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Component verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Component verification")
print("=" * 70)

test_clip = train_clips[0]
test_wav = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"  Test: {test_clip['clip_name']} ({len(test_wav)/SAMPLE_RATE:.1f}s, {test_clip['language']})")

mel = feat_extractor(test_wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
mel_aug = apply_spec_augment(mel.input_features.clone())
mel_gpu = mel_aug.to(DEVICE)
print(f"  Mel: {tuple(mel.input_features.shape)} → SpecAugment applied")

# Forward through both student encoders
a_out = encoder_a(mel_gpu).last_hidden_state
b_out = encoder_b(mel_gpu).last_hidden_state
real_frames = compute_real_frames(len(test_wav))
a_feat = a_out[:, :real_frames, :]
b_feat = b_out[:, :real_frames, :]
print(f"  Encoder A: {tuple(a_feat.shape)}")
print(f"  Encoder B: {tuple(b_feat.shape)}")

# Teacher forward (frozen)
with torch.no_grad():
    t_out = teacher_encoder(mel_gpu).last_hidden_state
    t_feat = t_out[:, :real_frames, :]
print(f"  Teacher:   {tuple(t_feat.shape)}")

# Projection A → teacher space
proj_a = projection_a(a_feat)
mse_acou = F.mse_loss(proj_a, t_feat)
print(f"  Projection A: {tuple(proj_a.shape)} → MSE vs teacher: {mse_acou.item():.4f}")

# Bridge head (if Hindi/Marathi with logits)
if test_clip["logit_path"]:
    teacher_data = torch.load(test_clip["logit_path"], weights_only=True)
    t_logits = teacher_data["logits"].float().to(DEVICE)
    t_aligned = align_teacher_to_student(t_logits, real_frames)
    if test_clip["lang_code"] == "hi":
        b_logits = bridge_hi(b_feat)
    else:
        b_logits = bridge_mr(b_feat)
    mse_sem = F.mse_loss(b_logits, t_aligned)
    print(f"  Bridge:    {tuple(b_logits.shape)} → MSE vs IndicConf: {mse_sem.item():.4f}")

# Fusion
fused, gate = fusion(a_feat, b_feat)
print(f"  Fusion:    {tuple(fused.shape)} (gate mean={gate.mean().item():.3f})")

# CTC
char_logits = ctc_head(fused)
print(f"  CTC head:  {tuple(char_logits.shape)}")

# Orthogonality
a_mean = a_feat[:, :real_frames, :].mean(dim=1)
b_mean = b_feat[:, :real_frames, :].mean(dim=1)
cos_sim = F.cosine_similarity(a_mean, b_mean).abs().mean()
print(f"  Encoder cosine sim: {cos_sim.item():.4f} (will decrease as encoders specialize)")

# CTC loss + backward
gt = normalize_text(test_clip["ground_truth"])
target_indices = text_to_indices(gt, char_to_idx)
if target_indices and real_frames > len(target_indices):
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    log_probs = char_logits.log_softmax(dim=-1).permute(1, 0, 2)
    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
    ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    # Total test loss
    total_loss = 0.3 * ctc_loss + 0.5 * mse_acou
    if test_clip["logit_path"]:
        total_loss += 0.3 * mse_sem
    total_loss += 0.01 * cos_sim

    total_loss.backward()
    a_grads = sum(1 for p in encoder_a.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    b_grads = sum(1 for p in encoder_b.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    fusion_grads = sum(1 for p in fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    proj_grads = sum(1 for p in projection_a.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  CTC loss: {ctc_loss.item():.4f} | Total: {total_loss.item():.4f}")
    print(f"  Gradients: enc_a={a_grads}>0 | enc_b={b_grads}>0 | fusion={fusion_grads}>0 | proj={proj_grads}>0")

    pred = ctc_greedy_decode(char_logits.squeeze(0).detach(), idx_to_char)
    print(f"  GT:      '{gt[:80]}'")
    print(f"  Decoded: '{pred[:80]}'")

# Zero grads
for m in [encoder_a, encoder_b, fusion, ctc_head, projection_a, bridge_hi, bridge_mr]:
    m.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ALL VERIFIED\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Training setup")
print("=" * 70)

# Differential LR: encoders = slow, new modules = fast
encoder_params = list(encoder_a.parameters()) + list(encoder_b.parameters())
new_params = (list(fusion.parameters()) + list(ctc_head.parameters()) +
              list(projection_a.parameters()) + list(bridge_hi.parameters()) +
              list(bridge_mr.parameters()))
all_trainable = encoder_params + new_params

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.encoder_lr},
    {"params": new_params, "lr": args.fusion_lr},
], betas=(0.9, 0.98), weight_decay=0.01)

num_epochs = args.epochs
# Steps per epoch = clips / grad_accum (since we accumulate)
steps_per_epoch = len(train_clips) // args.grad_accum
total_steps_est = steps_per_epoch * num_epochs
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    """Linear warmup → cosine decay to ~3% of peak."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(0.03, 0.5 * (1.0 + math.cos(math.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Restore optimizer/scheduler if resuming
if args.resume:
    resume_ckpt = torch.load(PROJECT_ROOT / args.resume, map_location=DEVICE, weights_only=False)
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    del resume_ckpt
    gc.collect()

ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

print(f"  Optimizer: AdamW (encoder_lr={args.encoder_lr}, fusion_lr={args.fusion_lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) → cosine decay")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)} | Grad accum: {args.grad_accum}")
print(f"  Steps/epoch: ~{steps_per_epoch} | Total: ~{total_steps_est}")
print(f"  Loss weights (initial): CTC={args.w_ctc} | Acoustic={args.w_acoustic} | Semantic={args.w_semantic} | Ortho={args.w_ortho}")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Dev evaluation function
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dev(dev_clips_list):
    """Evaluate model on dev set → (dev_wer, dev_loss, predictions, lang_wers)."""
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
                gate_vals.append(gate.mean().item())
                logits = ctc_head(fused)

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
mode_str = f"VERIFICATION ({len(train_clips)} clips)" if args.verify else f"Training ({len(train_clips)} clips)"
print(f"STEP 7: {mode_str}")
print("=" * 70)

random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

epoch_stats = []
global_step = 0
patience_counter = 0
training_start = time.time()

# Set all to train mode
for m in [encoder_a, encoder_b, fusion, ctc_head, projection_a, bridge_hi, bridge_mr]:
    m.train()

for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
    random.shuffle(train_clips)

    # Get scheduled loss weights for this epoch
    w_ctc, w_acou, w_sem = get_loss_weights(epoch, start_epoch + num_epochs)

    ep_ctc_losses = []
    ep_mse_acou_losses = []
    ep_mse_sem_losses = []
    ep_ortho_losses = []
    ep_total_losses = []
    ep_gate_vals = []
    ep_cos_sims = []
    ep_start = time.time()
    skipped = 0

    optimizer.zero_grad()  # Zero grads at start (for accumulation)

    pbar = tqdm(train_clips, desc=f"Epoch {epoch}",
                unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # ── 1. Load audio ──
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:
                skipped += 1
                continue

            num_samples = len(wav)

            # ── 2. Mel spectrogram + SpecAugment ──
            mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            if args.specaugment:
                mel_features = apply_spec_augment(mel.input_features.clone())
            else:
                mel_features = mel.input_features
            mel_gpu = mel_features.to(DEVICE)

            # ── 3. Forward through both student encoders ──
            a_out = encoder_a(mel_gpu).last_hidden_state
            b_out = encoder_b(mel_gpu).last_hidden_state

            real_frames = compute_real_frames(num_samples)
            a_feat = a_out[:, :real_frames, :]
            b_feat = b_out[:, :real_frames, :]

            # ── 4. Acoustic MSE: Encoder A vs Kid-Whisper teacher ──
            with torch.no_grad():
                t_out = teacher_encoder(mel_gpu).last_hidden_state
                t_feat = t_out[:, :real_frames, :]

            proj_a = projection_a(a_feat)
            mse_acou = F.mse_loss(proj_a, t_feat)

            # ── 5. Semantic MSE: Encoder B vs IndicConformer (Hi/Mr only) ──
            mse_sem = torch.tensor(0.0, device=DEVICE)
            has_semantic = False
            if clip["logit_path"] and clip["lang_code"] in ("hi", "mr"):
                try:
                    teacher_data = torch.load(clip["logit_path"], weights_only=True)
                    t_logits = teacher_data["logits"].float().to(DEVICE)
                    t_aligned = align_teacher_to_student(t_logits, real_frames)

                    if clip["lang_code"] == "hi":
                        b_logits = bridge_hi(b_feat)
                    else:
                        b_logits = bridge_mr(b_feat)
                    mse_sem = F.mse_loss(b_logits, t_aligned)
                    has_semantic = True
                except Exception:
                    pass

            # ── 6. Gated fusion → CTC ──
            fused, gate = fusion(a_feat, b_feat)
            logits = ctc_head(fused)

            gt = normalize_text(clip["ground_truth"])
            target_indices = text_to_indices(gt, char_to_idx)

            if not target_indices or real_frames <= len(target_indices):
                skipped += 1
                continue

            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

            ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            # ── 7. Orthogonality loss ──
            a_mean = a_feat.mean(dim=1)  # (1, 768)
            b_mean = b_feat.mean(dim=1)  # (1, 768)
            ortho_loss = F.cosine_similarity(a_mean, b_mean).abs().mean()

            # ── 8. Combined loss ──
            total_loss = w_ctc * ctc_loss + w_acou * mse_acou + args.w_ortho * ortho_loss
            if has_semantic:
                total_loss += w_sem * mse_sem

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                skipped += 1
                continue

            # Scale for gradient accumulation
            (total_loss / args.grad_accum).backward()

            # ── 9. Optimizer step every grad_accum clips ──
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Track losses
            ep_ctc_losses.append(ctc_loss.item())
            ep_mse_acou_losses.append(mse_acou.item())
            if has_semantic:
                ep_mse_sem_losses.append(mse_sem.item())
            ep_ortho_losses.append(ortho_loss.item())
            ep_total_losses.append(total_loss.item())
            ep_gate_vals.append(gate.mean().item())
            ep_cos_sims.append(ortho_loss.item())

            # ── 10. Free GPU memory ──
            if DEVICE == "cuda":
                del a_out, b_out, a_feat, b_feat, t_out, t_feat, proj_a
                del fused, logits, mel_gpu
                if has_semantic:
                    del t_logits, t_aligned, b_logits
                torch.cuda.empty_cache()

            # Update progress bar
            if ep_total_losses:
                avg_recent = np.mean(ep_total_losses[-20:])
                pbar.set_postfix({
                    "total": f"{ep_total_losses[-1]:.3f}",
                    "ctc": f"{ep_ctc_losses[-1]:.3f}",
                    "gate": f"{ep_gate_vals[-1]:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                })

        except Exception as e:
            print(f"\n  ERROR step {global_step}: {clip['clip_name']}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            skipped += 1
            continue

    pbar.close()

    # Handle remaining accumulated gradients
    if len(train_clips) % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    # ── Epoch summary ────────────────────────────────────────────────────
    ep_time = time.time() - ep_start
    avg_ctc = np.mean(ep_ctc_losses) if ep_ctc_losses else 0
    avg_mse_acou = np.mean(ep_mse_acou_losses) if ep_mse_acou_losses else 0
    avg_mse_sem = np.mean(ep_mse_sem_losses) if ep_mse_sem_losses else 0
    avg_ortho = np.mean(ep_ortho_losses) if ep_ortho_losses else 0
    avg_total = np.mean(ep_total_losses) if ep_total_losses else 0
    avg_gate = np.mean(ep_gate_vals) if ep_gate_vals else 0.5
    avg_cos = np.mean(ep_cos_sims) if ep_cos_sims else 0

    print(f"\n  ┌─ Epoch {epoch} ─────────────────────────────────────────────────")
    print(f"  │ Total loss:  {avg_total:.4f}")
    print(f"  │ CTC loss:    {avg_ctc:.4f} (weight={w_ctc:.1f})")
    print(f"  │ MSE acoustic:{avg_mse_acou:.4f} (weight={w_acou:.1f})")
    print(f"  │ MSE semantic:{avg_mse_sem:.4f} (weight={w_sem:.1f})")
    print(f"  │ Ortho loss:  {avg_ortho:.4f} (weight={args.w_ortho})")
    print(f"  │ Gate mean:   {avg_gate:.3f} (std={np.std(ep_gate_vals):.3f})")
    print(f"  │ Encoder cos: {avg_cos:.4f}")
    print(f"  │ Steps: {len(ep_total_losses)} ({skipped} skipped) | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
    print(f"  │ LR: encoder={scheduler.get_last_lr()[0]:.2e}, fusion={scheduler.get_last_lr()[1]:.2e}")
    print(f"  │ Weights: CTC={w_ctc:.1f} | Acou={w_acou:.1f} | Sem={w_sem:.1f}")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  │ Peak GPU: {peak:.0f} MB")
    print(f"  └──────────────────────────────────────────────────────────────")

    # ── Warning signs ────────────────────────────────────────────────────
    if avg_gate < 0.1 or avg_gate > 0.9:
        print(f"  ⚠ WARNING: Gate mean={avg_gate:.3f} — one encoder may be ignored!")
    if avg_cos > 0.9:
        print(f"  ⚠ WARNING: Encoder cos_sim={avg_cos:.3f} — possible mode collapse!")

    # ── Dev evaluation ───────────────────────────────────────────────────
    dev_wer, dev_loss, dev_preds, lang_wers, dev_gate = 999.0, 0.0, [], {}, 0.5
    if dev_clips:
        dev_wer, dev_loss, dev_preds, lang_wers, dev_gate = evaluate_dev(dev_clips)
        print(f"\n  ┌─ Dev Results ───────────────────────────────────────────────")
        print(f"  │ WER:  {dev_wer:.2f}%")
        for lc, lname in [("hi", "Hindi"), ("mr", "Marathi"), ("en", "English")]:
            if lang_wers.get(lc) is not None:
                print(f"  │   {lname}: {lang_wers[lc]:.2f}%")
        print(f"  │ Loss: {dev_loss:.4f} | Gate: {dev_gate:.3f}")
        print(f"  └──────────────────────────────────────────────────────────────")

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

    # ── Checkpoint saving ────────────────────────────────────────────────
    if not args.verify:
        ckpt_data = {
            "epoch": epoch,
            "encoder_a_state_dict": encoder_a.state_dict(),
            "encoder_b_state_dict": encoder_b.state_dict(),
            "fusion_state_dict": fusion.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "projection_a_state_dict": projection_a.state_dict(),
            "bridge_hi_state_dict": bridge_hi.state_dict(),
            "bridge_mr_state_dict": bridge_mr.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_total,
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
            print(f"  NEW BEST DEV WER: {dev_wer:.2f}% → saved best_wer.pt")
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
# STEP 8: Final results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 8: Results")
print("=" * 70)

if epoch_stats:
    print(f"\n  {'Ep':<4} {'Total':>7} {'CTC':>7} {'Acou':>7} {'Sem':>7} {'Ortho':>6} {'Gate':>5} {'CosSim':>6} {'WER':>8} {'Hi':>7} {'Mr':>7} {'En':>7}")
    print(f"  {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*5} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")
    for s in epoch_stats:
        wer = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        hi = f"{s['lang_wers'].get('hi', 0):.2f}%" if s['lang_wers'].get('hi') is not None else "N/A"
        mr = f"{s['lang_wers'].get('mr', 0):.2f}%" if s['lang_wers'].get('mr') is not None else "N/A"
        en = f"{s['lang_wers'].get('en', 0):.2f}%" if s['lang_wers'].get('en') is not None else "N/A"
        print(f"  {s['epoch']:<4} {s['train_loss']:>7.4f} {s['ctc_loss']:>7.4f} {s['mse_acou']:>7.4f} "
              f"{s['mse_sem']:>7.4f} {s['ortho']:>6.4f} {s['gate_mean']:>5.3f} {s['cos_sim']:>6.4f} "
              f"{wer:>8} {hi:>7} {mr:>7} {en:>7}")

    if len(epoch_stats) >= 2:
        first = epoch_stats[0]["train_loss"]
        last = epoch_stats[-1]["train_loss"]
        if first > 0:
            reduction = (first - last) / first * 100
            print(f"\n  Loss reduction: {reduction:.1f}% ({first:.4f} → {last:.4f})")

    if best_dev_wer < 999:
        print(f"\n  Best dev WER: {best_dev_wer:.2f}%")

print(f"\n  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
