"""
Phase 1 Training: Path B (Kid-Whisper Teacher) + Path C (MMS-300M Student)

What this script does:
  - Trains MMS-300M to mimic Kid-Whisper encoder's acoustic representations
  - Kid-Whisper is FROZEN (teacher) — never changes
  - MMS-300M is TRAINABLE (student) — learns children's acoustic patterns
  - Loss: MSE between student output and teacher output (frame-level)
  - A Linear(1024→768) projection matches dimensions

Architecture:
  audio → Kid-Whisper encoder (frozen) → Y_kid (T × 768)    [teacher target]
  audio → MMS-300M (trains) → Linear(1024→768) → Y_hat (T × 768) [student attempt]
  Loss = MSE(Y_hat, Y_kid)

Usage:
  # Verification (20 clips, 50 steps — proves learning works):
  python scripts/phase1_pathB_pathC.py --verify

  # Full training:
  python scripts/phase1_pathB_pathC.py --epochs 10 --batch_size 4
"""

import argparse
import csv
import time
import random
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from transformers import WhisperModel, WhisperProcessor
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--verify", action="store_true", help="Quick 50-step test to prove learning works")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--batch_size", type=int, default=1, help="Clips per step (1 for CPU)")
parser.add_argument("--max_audio_sec", type=float, default=15.0, help="Skip clips longer than this (memory)")
parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
parser.add_argument("--log_every", type=int, default=10, help="Print loss every N steps")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "train.csv"   # ALL training data (not just ASR)
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "phase1_pathBC"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load training clips
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading training data...")
print("=" * 60)

with open(TRAIN_CSV, encoding="utf-8") as f:
    all_clips = list(csv.DictReader(f))

# Filter to clips within max_audio_sec (memory constraint)
clips = [c for c in all_clips if float(c["duration_sec"]) <= args.max_audio_sec]
total_hrs = sum(float(c["duration_sec"]) for c in clips) / 3600

print(f"Total training clips: {len(all_clips)}")
print(f"After filtering ≤{args.max_audio_sec}s: {len(clips)} clips ({total_hrs:.1f}h)")

if args.verify:
    random.seed(42)
    clips = random.sample(clips, min(30, len(clips)))
    print(f"VERIFY MODE: using {len(clips)} clips only")

print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Load models
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: Loading models...")
print("=" * 60)

# ── PATH B: Kid-Whisper encoder (FROZEN teacher) ─────────────────────────────
print("Loading Kid-Whisper encoder (frozen teacher)...")
kw_processor = WhisperProcessor.from_pretrained("aadel4/kid-whisper-small-en-myst")
kw_encoder = WhisperModel.from_pretrained("aadel4/kid-whisper-small-en-myst").encoder.to(DEVICE)
kw_encoder.eval()
for p in kw_encoder.parameters():
    p.requires_grad = False
kw_params = sum(p.numel() for p in kw_encoder.parameters())
print(f"  Kid-Whisper encoder: {kw_params:,} params (FROZEN)")

# ── PATH C: MMS-300M (TRAINABLE student) ─────────────────────────────────────
print("Loading MMS-300M (trainable student)...")
mms_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-300m")
mms_model = Wav2Vec2Model.from_pretrained("facebook/mms-300m").to(DEVICE)
mms_model.train()
mms_params = sum(p.numel() for p in mms_model.parameters())
mms_trainable = sum(p.numel() for p in mms_model.parameters() if p.requires_grad)
print(f"  MMS-300M: {mms_params:,} params ({mms_trainable:,} trainable)")

# ── Projection layer: 1024 → 768 ─────────────────────────────────────────────
proj_layer = nn.Linear(1024, 768).to(DEVICE)
proj_params = sum(p.numel() for p in proj_layer.parameters())
print(f"  Projection (1024→768): {proj_params:,} params")

total_trainable = mms_trainable + proj_params
print(f"\n  Total trainable params: {total_trainable:,}")
print(f"  Total frozen params:    {kw_params:,}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Setup training
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: Setting up training...")
print("=" * 60)

# Combine trainable parameters: MMS-300M + projection layer
optimizer = torch.optim.AdamW(
    list(mms_model.parameters()) + list(proj_layer.parameters()),
    lr=args.lr,
    weight_decay=0.01,
)
loss_fn = nn.MSELoss()

print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=0.01)")
print(f"  Loss: MSE (frame-level)")
print(f"  Epochs: {args.epochs if not args.verify else '1 (verify mode)'}")
print()


def load_audio(path, max_sec=None):
    """Load audio, resample to 16kHz mono, optionally truncate."""
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()
    if max_sec and len(wav) > int(max_sec * 16000):
        wav = wav[:int(max_sec * 16000)]
    return wav.numpy()


def get_teacher_target(audio_np):
    """Get Kid-Whisper encoder output (frozen, no gradients)."""
    inputs = kw_processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)
    with torch.no_grad():
        encoder_out = kw_encoder(input_features)
    Y_kid_full = encoder_out.last_hidden_state  # (1, 1500, 768)

    # Mask padding: only keep frames from actual audio
    audio_samples = len(audio_np)
    valid_frames = audio_samples // 160 // 2
    Y_kid = Y_kid_full[:, :valid_frames, :]  # (1, valid_T, 768)
    return Y_kid


def get_student_output(audio_np):
    """Get MMS-300M output + projection (trainable, with gradients)."""
    inputs = mms_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)
    mms_out = mms_model(input_values)
    student_frames = mms_out.last_hidden_state  # (1, T, 1024)
    projected = proj_layer(student_frames)       # (1, T, 768)
    return projected


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
if args.verify:
    print("STEP 4: VERIFICATION RUN (proving learning works)...")
else:
    print("STEP 4: Training...")
print("=" * 60)

num_epochs = 1 if args.verify else args.epochs
max_steps = 50 if args.verify else None
loss_history = []
global_step = 0
best_loss = float("inf")

for epoch in range(num_epochs):
    random.shuffle(clips)
    epoch_losses = []
    epoch_start = time.time()

    for i, clip in enumerate(clips):
        if max_steps and global_step >= max_steps:
            break

        try:
            # Load audio
            audio = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(audio) < 8000:  # skip clips < 0.5s
                continue

            # ── Forward pass ──────────────────────────────────────
            # Teacher: get target (no gradients)
            Y_kid = get_teacher_target(audio)

            # Student: get prediction (with gradients)
            Y_hat = get_student_output(audio)

            # Align frame counts (off by ≤1)
            T = min(Y_kid.shape[1], Y_hat.shape[1])
            Y_kid_aligned = Y_kid[:, :T, :]
            Y_hat_aligned = Y_hat[:, :T, :]

            # ── Loss + backward ───────────────────────────────────
            loss = loss_fn(Y_hat_aligned, Y_kid_aligned)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                list(mms_model.parameters()) + list(proj_layer.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            loss_history.append(loss_val)
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                avg_recent = np.mean(loss_history[-args.log_every:])
                elapsed = time.time() - epoch_start
                print(f"  Step {global_step:>5} | Loss: {loss_val:.4f} | "
                      f"Avg(last {args.log_every}): {avg_recent:.4f} | "
                      f"T={T} frames | {elapsed:.0f}s")

        except Exception as e:
            print(f"  Step {global_step}: ERROR on {clip['audio_path']}: {e}")
            continue

    # Epoch summary
    if epoch_losses:
        epoch_avg = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1}/{num_epochs} done | "
              f"Avg loss: {epoch_avg:.4f} | "
              f"Steps: {len(epoch_losses)} | "
              f"Time: {epoch_time:.0f}s")

        # Save checkpoint
        if not args.verify and epoch_avg < best_loss:
            best_loss = epoch_avg
            ckpt_path = CKPT_DIR / f"best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "mms_state_dict": mms_model.state_dict(),
                "proj_state_dict": proj_layer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_avg,
                "global_step": global_step,
            }, ckpt_path)
            print(f"  Saved best checkpoint: {ckpt_path}")

    print()

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: Results")
print("=" * 60)

if len(loss_history) >= 5:
    first_5 = np.mean(loss_history[:5])
    last_5  = np.mean(loss_history[-5:])
    reduction = (first_5 - last_5) / first_5 * 100

    print(f"\n  Loss curve:")
    print(f"    First 5 steps avg: {first_5:.4f}")
    print(f"    Last 5 steps avg:  {last_5:.4f}")
    print(f"    Reduction:         {reduction:.1f}%")

    if reduction > 5:
        print(f"\n  ✓ LEARNING CONFIRMED — loss decreased by {reduction:.1f}%")
        print(f"    The student (MMS-300M) IS learning from the teacher (Kid-Whisper).")
        print(f"    Architecture works correctly.")
    elif reduction > 0:
        print(f"\n  ~ MARGINAL — loss decreased by {reduction:.1f}% (may need more steps)")
    else:
        print(f"\n  ✗ NO IMPROVEMENT — loss did not decrease. Something is wrong.")

    # Print full loss curve in compact form
    print(f"\n  Step-by-step losses ({len(loss_history)} steps):")
    for i, l in enumerate(loss_history):
        marker = "█" * int(min(l, 5) * 10)  # simple bar
        print(f"    Step {i+1:>3}: {l:.4f} {marker}")
else:
    print("  Not enough steps to evaluate.")

print(f"\n  Total steps: {global_step}")
print(f"  Checkpoint dir: {CKPT_DIR}")
print()
print("=" * 60)
if args.verify:
    print("  VERIFICATION COMPLETE")
    print("  If loss decreased → architecture works → ready for full training")
else:
    print("  TRAINING COMPLETE")
print("=" * 60)
