"""
Combined Branch — Dual-Encoder Feature Fusion Training
=======================================================

WHAT THIS SCRIPT DOES:
    Fuses features from TWO frozen Whisper Small encoders:
      1. Acoustic encoder (trained with Kid-Whisper MSE + CTC, 19.97% WER)
      2. Semantic encoder (trained with IndicConformer MSE, 49.02% WER)

    A trainable GatedFusion module learns per-frame weighting between the two
    encoders, and a warm-started CTC head decodes the fused features.

    Both encoders are FROZEN — only the fusion layer + CTC head train.
    Total trainable: ~1.25M params (vs ~89M in joint training).

WHY THIS SHOULD WORK:
    The acoustic encoder captures children's voice patterns (Kid-Whisper).
    The semantic encoder captures Indian language phonetics (IndicConformer).
    Neither alone has both. Gated fusion learns which encoder to trust per frame.

ARCHITECTURE:
    ┌───────────────────────────────────────────────────────────────┐
    │  Audio → Mel → SpecAugment (shared)                          │
    │       ├→ Acoustic Encoder (FROZEN) → feat_a (T, 768)         │
    │       └→ Semantic Encoder (FROZEN) → feat_s (T, 768)         │
    │            ↓ concat → (T, 1536)                              │
    │       GatedFusion (TRAINABLE):                               │
    │         gate = σ(Linear(1536→768))                           │
    │         fused = gate * feat_a + (1-gate) * feat_s            │
    │         → LayerNorm → Dropout                                │
    │            ↓                                                 │
    │       CTC Head: Linear(768→85) — warm from acoustic          │
    │            ↓                                                 │
    │       CTC Loss ← ground truth text                           │
    └───────────────────────────────────────────────────────────────┘

Usage:
    # Test run (3000 clips, 10 epochs)
    python step1_combined_training.py --test_clips 3000 --epochs 10

    # Full training
    python step1_combined_training.py --epochs 30 --patience 7
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
parser = argparse.ArgumentParser(description="Combined branch: dual-encoder fusion training")
parser.add_argument("--verify", action="store_true",
                    help="Quick verify (skips checkpoint saving)")
parser.add_argument("--test_clips", type=int, default=None,
                    help="Limit training clips")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (higher OK — only ~1.25M trainable params)")
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--patience", type=int, default=7,
                    help="Early stopping patience on dev WER (0=disabled)")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--acoustic_ckpt", type=str,
                    default="checkpoints/joint/joint_best_wer.pt",
                    help="Acoustic encoder checkpoint (joint training result)")
parser.add_argument("--semantic_ckpt", type=str,
                    default="checkpoints/semantic_mse/best_dev.pt",
                    help="Semantic encoder checkpoint (MSE distillation result)")
parser.add_argument("--ctc_ckpt", type=str, default=None,
                    help="CTC head warm-start (default: from acoustic checkpoint)")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from combined training checkpoint")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_every", type=int, default=10)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
VOCAB_PATH   = ASER_ROOT / "vocab.json"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "combined"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

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
    """Greedy CTC decode from logits (T, vocab_size)."""
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
# STEP 2: Load training data (all 3 languages)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading training data (all languages)")
print("=" * 70)


def load_clips_from_csv(csv_path, max_clips=None):
    """Load all clips from CSV (all languages, no teacher logits needed)."""
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

            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename
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

# Dev set
dev_clips, _ = load_clips_from_csv(DEV_CSV)
print(f"  Dev set: {len(dev_clips)} clips")

# Clip limiting
N = args.test_clips
if N is None and args.verify:
    N = 100

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
    label = "VERIFY MODE" if args.verify else "LIMITED"
    print(f"\n  {label}: {len(train_clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

    if args.verify and dev_clips:
        dev_n = min(100, len(dev_clips))
        random.seed(args.seed + 1)
        dev_clips = random.sample(dev_clips, dev_n)
        print(f"    Dev subset: {len(dev_clips)} clips")

print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load models (two frozen encoders + trainable fusion)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Loading models")
print("=" * 70)

import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor

WHISPER_ID = "openai/whisper-small"

# ── Feature extractor ────────────────────────────────────────────────────────
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)


def load_frozen_encoder(checkpoint_path, label):
    """Load a Whisper Small encoder and freeze it."""
    print(f"\n  [{label}] Loading Whisper Small encoder...")
    whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
    encoder = whisper_model.encoder.to(DEVICE)

    # Load trained weights
    ckpt_path = PROJECT_ROOT / checkpoint_path
    if not ckpt_path.exists():
        # Try alternate path
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

    # Handle different checkpoint key formats
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    elif "model_state_dict" in ckpt:
        # Acoustic MSE checkpoint uses model_state_dict with prefixed keys
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

    # Freeze completely
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    print(f"    FROZEN ({sum(p.numel() for p in encoder.parameters()):,} params, requires_grad=False)")

    # Clean up
    del whisper_model, ckpt
    gc.collect()

    return encoder


# ── Load both encoders ───────────────────────────────────────────────────────
acoustic_encoder = load_frozen_encoder(args.acoustic_ckpt, "ACOUSTIC ENCODER")
semantic_encoder = load_frozen_encoder(args.semantic_ckpt, "SEMANTIC ENCODER")


# ── Gated Fusion module ─────────────────────────────────────────────────────
class GatedFusion(nn.Module):
    """
    Learns per-frame, per-dimension weighting between acoustic and semantic features.
    gate = σ(W_g @ [feat_a; feat_s] + b_g)
    fused = gate * feat_a + (1 - gate) * feat_s
    """
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic_feat, semantic_feat):
        # acoustic_feat, semantic_feat: (B, T, 768)
        concat = torch.cat([acoustic_feat, semantic_feat], dim=-1)  # (B, T, 1536)
        gate = torch.sigmoid(self.gate_linear(concat))  # (B, T, 768)
        fused = gate * acoustic_feat + (1 - gate) * semantic_feat  # (B, T, 768)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused


print(f"\n  [GATED FUSION] Creating fusion module...")
fusion = GatedFusion(dim=768, dropout=args.dropout).to(DEVICE)
fusion_params = sum(p.numel() for p in fusion.parameters())
print(f"    Params: {fusion_params:,}")

# ── CTC Head ─────────────────────────────────────────────────────────────────
print(f"\n  [CTC HEAD] Linear(768→{vocab_size})...")
ctc_head = nn.Linear(768, vocab_size).to(DEVICE)

# ── Warm-start CTC head ──────────────────────────────────────────────────────
start_epoch = 0
best_dev_wer = float("inf")

if args.resume:
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading combined checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))
    print(f"    Resuming from epoch {start_epoch}, best dev WER: {best_dev_wer:.2f}%")
    del ckpt
else:
    # Warm-start CTC head from acoustic branch
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

# ── Param summary ────────────────────────────────────────────────────────────
acoustic_params = sum(p.numel() for p in acoustic_encoder.parameters())
semantic_params = sum(p.numel() for p in semantic_encoder.parameters())
ctc_params = sum(p.numel() for p in ctc_head.parameters())
total_trainable = fusion_params + ctc_params

print(f"\n  ┌─ Parameter Summary ──────────────────────────────────────────")
print(f"  │ Acoustic encoder: {acoustic_params:,} (FROZEN)")
print(f"  │ Semantic encoder: {semantic_params:,} (FROZEN)")
print(f"  │ Gated fusion:     {fusion_params:,} (trainable)")
print(f"  │ CTC head:         {ctc_params:,} (trainable)")
print(f"  │ Total trainable:  {total_trainable:,}")
print(f"  └──────────────────────────────────────────────────────────────")

if DEVICE == "cuda":
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
    wav = wav.squeeze()
    if max_sec and len(wav) > int(max_sec * SAMPLE_RATE):
        wav = wav[:int(max_sec * SAMPLE_RATE)]
    return wav


def compute_real_frames(num_samples):
    """Whisper: mel stride=160, conv stride=2 → frames = samples // 320."""
    return min(num_samples // 160 // 2, 1500)


def apply_spec_augment(mel_features):
    """SpecAugment: 2 freq masks (max 15), 2 time masks (max 50). Training only."""
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

# Both encoders forward (no grad)
with torch.no_grad():
    acoustic_out = acoustic_encoder(mel_aug.to(DEVICE)).last_hidden_state
    semantic_out = semantic_encoder(mel_aug.to(DEVICE)).last_hidden_state

real_frames = compute_real_frames(len(test_wav))
acoustic_feat = acoustic_out[:, :real_frames, :]
semantic_feat = semantic_out[:, :real_frames, :]
print(f"  Acoustic encoder: {tuple(acoustic_feat.shape)}")
print(f"  Semantic encoder: {tuple(semantic_feat.shape)}")

# Fusion forward (with grad)
fused = fusion(acoustic_feat, semantic_feat)
print(f"  Gated fusion: {tuple(fused.shape)}")

# CTC Head
char_logits = ctc_head(fused)
print(f"  CTC head: {tuple(char_logits.shape)}")

# CTC loss
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

    # Verify gradients
    ctc_loss.backward()
    fusion_grads = sum(1 for p in fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    ctc_grads = sum(1 for p in ctc_head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    acoustic_grads = sum(1 for p in acoustic_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    semantic_grads = sum(1 for p in semantic_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Gradients: fusion={fusion_grads}>0 | ctc_head={ctc_grads}>0 | "
          f"acoustic_enc={acoustic_grads}(should=0) | semantic_enc={semantic_grads}(should=0)")

    # Greedy decode check
    pred = ctc_greedy_decode(char_logits.squeeze(0).detach(), idx_to_char)
    print(f"  Decode test: '{pred[:80]}...'")

fusion.zero_grad()
ctc_head.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ALL VERIFIED\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Training setup")
print("=" * 70)

# Trainable params: fusion + CTC head only
trainable_params = list(fusion.parameters()) + list(ctc_head.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

num_epochs = args.epochs
total_steps_est = len(train_clips) * num_epochs  # no grad_accum
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
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
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)}")
print(f"  Trainable: {total_trainable:,} params (fusion + CTC head)")
print(f"  Dropout: {args.dropout}")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience} (on dev WER)")
print()


# ── Dev evaluation function ──────────────────────────────────────────────────

def evaluate_dev(dev_clips_list):
    """
    Evaluate on dev set — WER from fused features + CTC head.
    Returns: (dev_wer, dev_loss, all_preds, lang_wers)
    """
    fusion.eval()
    ctc_head.eval()

    all_refs = []
    all_hyps = []
    losses = []
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
                mel = feat_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")

                # Both encoders
                a_out = acoustic_encoder(mel.input_features.to(DEVICE)).last_hidden_state
                s_out = semantic_encoder(mel.input_features.to(DEVICE)).last_hidden_state
                real_frames = compute_real_frames(len(wav))
                a_feat = a_out[:, :real_frames, :]
                s_feat = s_out[:, :real_frames, :]

                # Fusion + CTC head
                fused = fusion(a_feat, s_feat)
                logits = ctc_head(fused)  # (1, T, 85)

                # CTC loss
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

                # Decode
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

    fusion.train()
    ctc_head.train()

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

random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

epoch_stats = []
global_step = 0
patience_counter = 0
training_start = time.time()

fusion.train()
ctc_head.train()

for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
    random.shuffle(train_clips)

    ep_losses = []
    ep_start = time.time()
    skipped = 0

    pbar = tqdm(train_clips, desc=f"Epoch {epoch}",
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
            mel_gpu = mel_features.to(DEVICE)

            # ── Both encoders forward (no grad) ──
            with torch.no_grad():
                a_out = acoustic_encoder(mel_gpu).last_hidden_state
                s_out = semantic_encoder(mel_gpu).last_hidden_state

            real_frames = compute_real_frames(num_samples)
            a_feat = a_out[:, :real_frames, :]  # (1, T, 768)
            s_feat = s_out[:, :real_frames, :]  # (1, T, 768)

            # ── Fusion + CTC head (with grad) ──
            fused = fusion(a_feat, s_feat)  # (1, T, 768)
            logits = ctc_head(fused)  # (1, T, 85)

            # ── CTC loss ──
            gt = normalize_text(clip["ground_truth"])
            target_indices = text_to_indices(gt, char_to_idx)

            if not target_indices or real_frames <= len(target_indices):
                skipped += 1
                continue

            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, 1, 85)
            targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            # ── Backward + step ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            ep_losses.append(loss.item())

            # ── Cleanup ──
            if DEVICE == "cuda":
                del a_out, s_out, a_feat, s_feat, fused, logits, mel_gpu
                torch.cuda.empty_cache()

            # Progress bar
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
    print(f"  │ LR: {scheduler.get_last_lr()[0]:.2e}")
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

        if dev_preds:
            n_show = min(3, len(dev_preds))
            print(f"\n  Sample predictions:")
            for ref, hyp in dev_preds[:n_show]:
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
                print()

    # Save stats
    epoch_stats.append({
        "epoch": epoch,
        "train_loss": avg_loss,
        "dev_wer": dev_wer, "dev_loss": dev_loss,
        "lang_wers": lang_wers,
    })

    # ── Checkpoint ───────────────────────────────────────────────────────
    if not args.verify:
        ckpt_data = {
            "epoch": epoch,
            "fusion_state_dict": fusion.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
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
# STEP 8: Final results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 8: Results")
print("=" * 70)

if epoch_stats:
    print(f"\n  {'Epoch':<8} {'Loss':>8} {'DevWER':>10} {'DevLoss':>8} {'Hindi':>8} {'Marathi':>8} {'English':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for s in epoch_stats:
        wer_str = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        hi_str = f"{s['lang_wers'].get('hi', 0):.2f}%" if s['lang_wers'].get('hi') is not None else "N/A"
        mr_str = f"{s['lang_wers'].get('mr', 0):.2f}%" if s['lang_wers'].get('mr') is not None else "N/A"
        en_str = f"{s['lang_wers'].get('en', 0):.2f}%" if s['lang_wers'].get('en') is not None else "N/A"
        print(f"  {s['epoch']:<8} {s['train_loss']:>8.4f} {wer_str:>10} {s['dev_loss']:>8.4f} "
              f"{hi_str:>8} {mr_str:>8} {en_str:>8}")

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
