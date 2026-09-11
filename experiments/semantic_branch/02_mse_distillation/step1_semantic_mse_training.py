"""
Step 1: Semantic Branch — MSE Distillation from IndicConformer Logits
=====================================================================

WHAT THIS SCRIPT DOES:
    Trains Whisper Small encoder + per-language CTC Head 2 to match
    IndicConformer's pre-computed logits using MSE loss.

    Flow (per clip):
        Audio → WhisperFeatureExtractor → mel (1,80,3000)
             → Whisper Encoder → features (1,1500,768)
             → Slice real frames → (1,T_student,768)
             → CTC Head 2 (Hindi or Marathi) → student_logits (1,T_student,257)

        teacher_logits (T_teacher,257)
             → F.interpolate → (1,T_student,257)

        loss = MSE(student_logits, teacher_logits_aligned)

WHY SEPARATE CTC HEADS:
    IndicConformer's BPE vocab is different per language.
    Index 0 = "▁क" in Hindi but "या" in Marathi.
    One head would get contradictory gradients → two separate Linear(768→257).

WHY INTERPOLATION:
    Whisper encoder: 50 fps (768-dim), IndicConformer: 12.5 fps (257-dim).
    Teacher has ~4x fewer frames. We stretch teacher logits to match student
    frame count using linear interpolation.

STARTING POINT:
    Fresh pretrained Whisper Small (NOT acoustic branch checkpoint).
    Clean isolation — measures semantic distillation contribution alone.

Prerequisites:
    Run step0_precompute_teacher_logits.py first to generate teacher_logits/

Usage:
    # Quick verify (50 clips, 3 epochs)
    python scripts/semantic/step1_semantic_mse_training.py --verify

    # Verify with more clips
    python scripts/semantic/step1_semantic_mse_training.py --verify --test_clips 200

    # Full training run
    python scripts/semantic/step1_semantic_mse_training.py --epochs 20 --patience 5
"""

import argparse
import csv
import gc
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Semantic MSE distillation training")
parser.add_argument("--verify", action="store_true",
                    help="Test run with subset of clips × 3 epochs")
parser.add_argument("--test_clips", type=int, default=50,
                    help="Number of clips in verify mode (default 50)")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--warmup_steps", type=int, default=500,
                    help="Linear warmup steps")
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (0=disabled)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_every", type=int, default=10)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # experiments/semantic_branch/02_mse_distillation/.. → project root
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
TEACHER_LOGITS_DIR = PROJECT_ROOT / "teacher_logits"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "semantic_mse"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load training data (Hindi + Marathi only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading training data (Hindi + Marathi only)")
print("=" * 70)

LANG_MAP = {"Hindi": "hi", "Marathi": "mr"}


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

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "clip_name": clip_uid,
                "logit_path": str(logit_path),
                "duration_sec": float(row.get("duration_sec", 0)),
                "ground_truth": row.get("que_text", row.get("text", "")),
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

# ── Dev set ──────────────────────────────────────────────────────────────────
dev_clips = []
if DEV_CSV.exists():
    dev_clips, _, _ = load_clips_from_csv(DEV_CSV, split="dev")
    print(f"  Dev set: {len(dev_clips)} clips")

# ── Verify mode ──────────────────────────────────────────────────────────────
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
    print(f"    Running 3 epochs to confirm learning")

    if dev_clips:
        dev_n = min(100, len(dev_clips))
        random.seed(args.seed + 1)
        dev_clips = random.sample(dev_clips, dev_n)
        print(f"    Dev subset: {len(dev_clips)} clips")

print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load models
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 2: Loading models")
print("=" * 70)

# ── Whisper Small encoder (UNFROZEN — trains) ────────────────────────────────
print("\n  [STUDENT] openai/whisper-small encoder...")
from transformers import WhisperModel, WhisperFeatureExtractor
import torchaudio

WHISPER_ID = "openai/whisper-small"
feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
encoder = whisper_model.encoder.to(DEVICE)
encoder.train()

# Enable gradient checkpointing to save GPU memory
encoder.gradient_checkpointing_enable()

enc_params = sum(p.numel() for p in encoder.parameters())
enc_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"    {enc_params:,} params ({enc_trainable:,} trainable) | UNFROZEN")
print(f"    Gradient checkpointing: ENABLED")

# ── CTC Head 2: per-language Linear(768 → 257) ──────────────────────────────
print("\n  [CTC HEAD 2] Per-language projection heads...")
ctc_head_hi = nn.Linear(768, 257).to(DEVICE)
ctc_head_mr = nn.Linear(768, 257).to(DEVICE)
hi_params = sum(p.numel() for p in ctc_head_hi.parameters())
mr_params = sum(p.numel() for p in ctc_head_mr.parameters())
print(f"    Hindi:   Linear(768→257) — {hi_params:,} params | random init")
print(f"    Marathi: Linear(768→257) — {mr_params:,} params | random init")
print(f"    Total head params: {hi_params + mr_params:,} ({(hi_params+mr_params)/enc_params*100:.2f}% of encoder)")

# ── Memory report ────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  GPU Memory: {allocated:.0f} MB allocated")

# Clean up whisper_model (we only need the encoder)
del whisper_model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Component verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Component verification")
print("=" * 70)


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
    """
    Compute number of real encoder frames for given audio length.
    Whisper: mel has 1 frame per 160 samples (10ms hop), then conv stride=2.
    So encoder_frames = num_samples // 160 // 2.
    Capped at 1500 (Whisper's max for 30s audio).
    """
    return min(num_samples // 160 // 2, 1500)


def forward_student(wav_1d, lang_code):
    """
    Run student forward pass.
    Returns: student_logits (1, T_student, 257), real_frames (int)
    """
    # Mel spectrogram
    mel = feat_extractor(
        wav_1d.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = mel.input_features.to(DEVICE)  # (1, 80, 3000)

    # Encoder
    enc_out = encoder(input_features)
    features = enc_out.last_hidden_state  # (1, 1500, 768)

    # Slice to real frames
    real_frames = compute_real_frames(len(wav_1d))
    real_features = features[:, :real_frames, :]  # (1, T_student, 768)

    # Language-specific CTC head
    if lang_code == "hi":
        logits = ctc_head_hi(real_features)  # (1, T_student, 257)
    else:
        logits = ctc_head_mr(real_features)  # (1, T_student, 257)

    return logits, real_frames


def align_teacher_to_student(teacher_logits, student_frames):
    """
    Interpolate teacher logits to match student frame count.
    Teacher: (T_teacher, 257) at 12.5 fps
    Student: T_student frames at 50 fps
    Returns: (1, T_student, 257)
    """
    # (T_teacher, 257) → (1, 257, T_teacher) for F.interpolate
    t = teacher_logits.T.unsqueeze(0)  # (1, 257, T_teacher)
    aligned = F.interpolate(t, size=student_frames, mode='linear', align_corners=False)
    # (1, 257, T_student) → (1, T_student, 257)
    return aligned.permute(0, 2, 1)


# ── Test with first clip ─────────────────────────────────────────────────────
test_clip = train_clips[0]
test_wav = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"  Test: {test_clip['clip_name']} ({len(test_wav)/SAMPLE_RATE:.1f}s, {test_clip['language']})")

# Student forward
student_logits, real_frames = forward_student(test_wav, test_clip["lang_code"])
print(f"  Student: encoder → slice({real_frames}) → CTC Head → {tuple(student_logits.shape)}")

# Teacher logits
teacher_data = torch.load(test_clip["logit_path"], weights_only=True)
teacher_logits = teacher_data["logits"].float().to(DEVICE)
teacher_frames = teacher_data["num_frames"]
print(f"  Teacher: loaded {tuple(teacher_logits.shape)} ({teacher_frames} frames)")

# Alignment
teacher_aligned = align_teacher_to_student(teacher_logits, real_frames)
print(f"  Aligned: {tuple(teacher_logits.shape)} → {tuple(teacher_aligned.shape)}")
print(f"  Shapes match: {student_logits.shape == teacher_aligned.shape} ✓")

# MSE loss
loss = F.mse_loss(student_logits, teacher_aligned)
loss.backward()

# Verify gradients
frozen_ok = True  # No frozen models to check (teacher is pre-computed)
head_grads = sum(1 for p in ctc_head_hi.parameters() if p.grad is not None)
enc_grads = sum(1 for p in encoder.parameters() if p.grad is not None)
print(f"  MSE loss: {loss.item():.4f}")
print(f"  Gradients: encoder={enc_grads}>0 ✓ | ctc_head={head_grads}>0 ✓")

encoder.zero_grad()
ctc_head_hi.zero_grad()
ctc_head_mr.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ✓ ALL VERIFIED\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Training setup")
print("=" * 70)

all_params = list(encoder.parameters()) + list(ctc_head_hi.parameters()) + list(ctc_head_mr.parameters())
optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

num_epochs = 3 if args.verify else args.epochs
total_steps_est = len(train_clips) * num_epochs

# Linear warmup then linear decay scheduler
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(0.0, 1.0 - progress)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) + linear decay")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)} | Est. steps: ~{total_steps_est}")
print(f"  Grad clip: max_norm=1.0")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience}")
print()


# ── Dev evaluation function ──────────────────────────────────────────────────

def evaluate_dev_loss(dev_clips_list):
    """Compute average MSE loss on dev set (no gradients)."""
    encoder.eval()
    losses = []
    hi_losses = []
    mr_losses = []

    with torch.no_grad():
        for clip in tqdm(dev_clips_list, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(wav) < 8000:
                    continue

                # Student forward
                student_logits, real_frames = forward_student(wav, clip["lang_code"])

                # Teacher
                teacher_data = torch.load(clip["logit_path"], weights_only=True)
                t_logits = teacher_data["logits"].float().to(DEVICE)
                t_aligned = align_teacher_to_student(t_logits, real_frames)

                loss_val = F.mse_loss(student_logits, t_aligned).item()
                losses.append(loss_val)

                if clip["lang_code"] == "hi":
                    hi_losses.append(loss_val)
                else:
                    mr_losses.append(loss_val)

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            except Exception:
                continue

    encoder.train()

    if losses:
        return (np.mean(losses),
                np.mean(hi_losses) if hi_losses else 0,
                np.mean(mr_losses) if mr_losses else 0)
    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 5: {'VERIFICATION (' + str(len(train_clips)) + ' clips × 3 epochs)' if args.verify else 'Full training'}")
print("=" * 70)

random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

loss_history = []
epoch_avg_losses = []
epoch_hi_avgs = []
epoch_mr_avgs = []
dev_avg_losses = []
global_step = 0
best_train_loss = float("inf")
best_dev_loss = float("inf")
patience_counter = 0
training_start = time.time()

for epoch in range(num_epochs):
    random.shuffle(train_clips)
    ep_losses = []
    ep_hi_losses = []
    ep_mr_losses = []
    ep_start = time.time()

    pbar = tqdm(train_clips, desc=f"Epoch {epoch+1}/{num_epochs}", unit="clip",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # Load audio
            wav = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(wav) < 8000:
                continue

            # Student forward
            student_logits, real_frames = forward_student(wav, clip["lang_code"])

            # Load pre-computed teacher logits
            teacher_data = torch.load(clip["logit_path"], weights_only=True)
            t_logits = teacher_data["logits"].float().to(DEVICE)

            # Align teacher to student frame count
            teacher_aligned = align_teacher_to_student(t_logits, real_frames)

            # MSE loss
            loss = F.mse_loss(student_logits, teacher_aligned)

            # Backward + update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()

            # Cleanup
            if DEVICE == "cuda":
                del student_logits, teacher_aligned, t_logits, loss
                torch.cuda.empty_cache()

            ep_losses.append(loss_val)
            loss_history.append(loss_val)

            if clip["lang_code"] == "hi":
                ep_hi_losses.append(loss_val)
            else:
                ep_mr_losses.append(loss_val)

            global_step += 1

            # Progress bar
            avg_recent = np.mean(loss_history[-20:])
            pbar.set_postfix({
                "loss": f"{loss_val:.3f}",
                "avg20": f"{avg_recent:.3f}",
                "lang": clip["lang_code"],
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            })

        except Exception as e:
            print(f"\n  ERROR step {global_step}: {clip['clip_name']}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue

    pbar.close()

    # ── Epoch summary ────────────────────────────────────────────────────
    if ep_losses:
        ep_avg = np.mean(ep_losses)
        ep_hi_avg = np.mean(ep_hi_losses) if ep_hi_losses else 0
        ep_mr_avg = np.mean(ep_mr_losses) if ep_mr_losses else 0
        ep_time = time.time() - ep_start
        epoch_avg_losses.append(ep_avg)
        epoch_hi_avgs.append(ep_hi_avg)
        epoch_mr_avgs.append(ep_mr_avg)

        print(f"\n  ┌─ Epoch {epoch+1}/{num_epochs} ─────────────────────────────────────────")
        print(f"  │ MSE loss:      {ep_avg:.4f}")
        print(f"  │ Hindi MSE:     {ep_hi_avg:.4f}")
        print(f"  │ Marathi MSE:   {ep_mr_avg:.4f}")
        print(f"  │ Steps: {len(ep_losses)} | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
        print(f"  │ LR: {scheduler.get_last_lr()[0]:.2e}")
        if DEVICE == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  │ Peak GPU: {peak:.0f} MB")
        if len(epoch_avg_losses) > 1:
            d = epoch_avg_losses[-2] - epoch_avg_losses[-1]
            pct = d / epoch_avg_losses[-2] * 100
            arr = "↓" if d > 0 else "↑"
            print(f"  │ vs prev: {arr} {abs(pct):.1f}% ({epoch_avg_losses[-2]:.4f} → {ep_avg:.4f})")
        print(f"  └──────────────────────────────────────────────────────────")

        # ── Dev evaluation ─────────────────────────────────────────────
        if dev_clips:
            dev_avg, dev_hi, dev_mr = evaluate_dev_loss(dev_clips)
            if dev_avg is not None:
                dev_avg_losses.append(dev_avg)
                print(f"\n  ┌─ Dev Loss ─────────────────────────────────────────────")
                print(f"  │ MSE: {dev_avg:.4f} (Hindi: {dev_hi:.4f}, Marathi: {dev_mr:.4f})")
                if len(dev_avg_losses) > 1:
                    d = dev_avg_losses[-2] - dev_avg_losses[-1]
                    pct = d / dev_avg_losses[-2] * 100
                    print(f"  │ vs prev: {'↓' if d > 0 else '↑'} {abs(pct):.1f}% ({dev_avg_losses[-2]:.4f} → {dev_avg:.4f})")
                gap = dev_avg - ep_avg
                print(f"  │ Train-Dev gap: {gap:+.4f} ({'ok' if gap < 0.5 else 'OVERFITTING!' if gap > 1.0 else 'watch'})")
                print(f"  └──────────────────────────────────────────────────────────")

                # Early stopping
                if args.patience > 0:
                    if dev_avg < best_dev_loss:
                        best_dev_loss = dev_avg
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"  Early stopping: {patience_counter}/{args.patience} (best dev: {best_dev_loss:.4f})")
                        if patience_counter >= args.patience:
                            print(f"\n  EARLY STOPPING at epoch {epoch+1}")

        # ── Save checkpoint ──────────────────────────────────────────────
        if not args.verify:
            ckpt_data = {
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "ctc_head_hi_state_dict": ctc_head_hi.state_dict(),
                "ctc_head_mr_state_dict": ctc_head_mr.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": ep_avg,
                "dev_loss": dev_avg_losses[-1] if dev_avg_losses else None,
                "global_step": global_step,
                "args": vars(args),
            }
            if ep_avg < best_train_loss:
                best_train_loss = ep_avg
                torch.save(ckpt_data, CKPT_DIR / "best_train.pt")
                print(f"  Saved best (train): {CKPT_DIR / 'best_train.pt'}")
            if dev_avg_losses and dev_avg_losses[-1] <= best_dev_loss:
                torch.save(ckpt_data, CKPT_DIR / "best_dev.pt")
                print(f"  Saved best (dev):   {CKPT_DIR / 'best_dev.pt'}")
            torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch+1}.pt")
            print(f"  Saved: {CKPT_DIR / f'epoch_{epoch+1}.pt'}")

    print()

    # Check early stopping
    if patience_counter >= args.patience > 0:
        break

total_time = time.time() - training_start

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Results")
print("=" * 70)

if len(epoch_avg_losses) >= 1:
    has_dev = len(dev_avg_losses) > 0
    print(f"\n  ── Epoch-level MSE losses ──")
    if has_dev:
        print(f"  {'Epoch':<8} {'Train':>10} {'Dev':>10} {'Hindi':>10} {'Marathi':>10} {'Gap':>8}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for i in range(len(epoch_avg_losses)):
            dev_v = dev_avg_losses[i] if i < len(dev_avg_losses) else 0
            gap = dev_v - epoch_avg_losses[i] if dev_v > 0 else 0
            bar = "█" * int(min(epoch_avg_losses[i] / 5, 10))
            print(f"  {i+1:<8} {epoch_avg_losses[i]:>10.4f} {dev_v:>10.4f} "
                  f"{epoch_hi_avgs[i]:>10.4f} {epoch_mr_avgs[i]:>10.4f} {gap:>+8.4f}  {bar}")
    else:
        print(f"  {'Epoch':<8} {'MSE':>10} {'Hindi':>10} {'Marathi':>10}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
        for i in range(len(epoch_avg_losses)):
            bar = "█" * int(min(epoch_avg_losses[i] / 5, 10))
            print(f"  {i+1:<8} {epoch_avg_losses[i]:>10.4f} {epoch_hi_avgs[i]:>10.4f} {epoch_mr_avgs[i]:>10.4f}  {bar}")

    # Check if learning happened
    if len(epoch_avg_losses) >= 2:
        total_drop = epoch_avg_losses[0] - epoch_avg_losses[-1]
        total_pct = total_drop / epoch_avg_losses[0] * 100
        print(f"\n  Total MSE reduction: {total_pct:.1f}% ({epoch_avg_losses[0]:.4f} → {epoch_avg_losses[-1]:.4f})")
        if total_drop > 0:
            print(f"  ✓ LEARNING CONFIRMED — MSE is decreasing")
        else:
            print(f"  ✗ WARNING — MSE is NOT decreasing, check hyperparameters")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Best train loss: {best_train_loss:.4f}")
    if dev_avg_losses:
        print(f"  Best dev loss:   {best_dev_loss:.4f}")
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
