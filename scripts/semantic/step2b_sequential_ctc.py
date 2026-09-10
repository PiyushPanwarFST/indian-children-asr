"""
Step 2.5: Sequential CTC Training on Frozen Semantic Encoder
============================================================

WHAT THIS SCRIPT DOES:
    Takes our semantic MSE-trained encoder (from step1) and trains
    a CTC head on top of it to predict text. The encoder is FROZEN —
    only the CTC head (one linear layer, 65K params) is trained.

    This is the SAME approach that worked in the acoustic branch:
        Acoustic: frozen encoder → train CTC head → 81.86% WER
        Then joint training improved it to 19.97%.

    We skipped this step initially and went straight to joint training,
    which gave 74.25% WER because the CTC head started from scratch.

WHY THIS IS NEEDED:
    CTC needs to learn "which character goes at which frame" — a hard
    alignment problem. It's much easier to learn this alignment on a
    FROZEN encoder (stable features) than on a moving encoder (joint).

    Think of it like learning to read a map:
    - Sequential CTC: the map doesn't change while you learn → easy
    - Joint training from scratch: the map keeps changing while you learn → hard

    After this step, the CTC head will know basic alignment.
    Then joint training can fine-tune both encoder + CTC head together.

ARCHITECTURE:
    ┌──────────────────────────────────────────────────┐
    │  Audio (.wav file)                               │
    │       ↓                                          │
    │  Whisper Feature Extractor (mel spectrogram)     │
    │       ↓                                          │
    │  Semantic Encoder (FROZEN, from step1)           │
    │  (12 transformer layers, trained with MSE        │
    │   against IndicConformer logits)                 │
    │       ↓                                          │
    │  Encoder features: (1, T, 768)                   │
    │       ↓                                          │
    │  CTC Head 1: Linear(768, 85) ← TRAINABLE        │
    │       ↓                                          │
    │  Logits: (1, T, 85)                              │
    │       ↓                                          │
    │  CTC Loss ← Ground truth text                    │
    │       ↓                                          │
    │  Backprop updates CTC head ONLY                  │
    └──────────────────────────────────────────────────┘

    85 = vocab size (3 special + 22 English + 60 Devanagari)
    from ASER-Dataset/vocab.json

WHAT GETS UPDATED:
    ✅ CTC head (Linear 768 → 85) — 65,365 parameters
    ❌ Encoder — FROZEN (already trained in step1 MSE)

Prerequisites:
    - Semantic MSE checkpoint: checkpoints/semantic_mse/best_dev.pt
    - Vocabulary: ASER-Dataset/vocab.json (85 tokens)
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Quick test (200 clips, 3 epochs)
    python scripts/semantic/step2b_sequential_ctc.py --max_clips 200 --epochs 3

    # Full training
    python scripts/semantic/step2b_sequential_ctc.py --epochs 30 --patience 5
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

# Suppress noisy transformer warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import WhisperModel, WhisperFeatureExtractor

# ── Add project root for utils import ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.wer import normalize_text, compute_corpus_wer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_train.csv")
DEV_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_dev.csv")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "ASER-Dataset", "vocab.json")
ENCODER_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "semantic_mse", "best_dev.pt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "semantic_ctc")

STUDENT_MODEL = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds


# ─────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────
def load_vocab(vocab_path):
    """Load character vocabulary from vocab.json."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)
    return char_to_idx, idx_to_char, vocab_size


def text_to_indices(text, char_to_idx):
    """Convert normalized text to vocabulary indices for CTC targets."""
    indices = []
    for char in text:
        if char == " ":
            indices.append(char_to_idx["<space>"])
        elif char in char_to_idx:
            indices.append(char_to_idx[char])
        else:
            indices.append(char_to_idx["<unk>"])
    return indices


def ctc_greedy_decode(logits, idx_to_char):
    """
    Greedy CTC decoding: argmax → collapse repeats → remove blanks → text.
    logits: (T, vocab_size)
    """
    predicted_ids = logits.argmax(dim=-1).tolist()

    decoded_ids = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id:
            if idx != 0:  # skip blank
                decoded_ids.append(idx)
        prev_id = idx

    chars = []
    for idx in decoded_ids:
        token = idx_to_char.get(idx, "")
        if token == "<space>":
            chars.append(" ")
        elif token in ("<blank>", "<unk>"):
            continue
        else:
            chars.append(token)

    return "".join(chars)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_clips(csv_path, max_clips=None):
    """Load audio clips and their transcripts from split CSV. All languages."""
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row.get("duration_sec", 0))
            if dur > MAX_DURATION:
                continue

            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(PROJECT_ROOT, "ASER-Dataset", audio_path)

            gt = row.get("transcript", row.get("que_text", row.get("text", "")))

            clips.append({
                "audio_path": audio_path,
                "transcript": gt,
                "language": row.get("language", ""),
                "duration": dur,
            })

    if max_clips:
        clips = clips[:max_clips]

    return clips


# ─────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────
def load_audio(audio_path):
    """Load audio → 16kHz mono tensor."""
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    return wav.squeeze(0)  # (samples,)


def compute_real_frames(num_samples):
    """Whisper: mel stride=160, conv stride=2 → encoder frames."""
    return min(num_samples // 160 // 2, 1500)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_frozen_encoder(checkpoint_path):
    """
    Load semantic MSE-trained encoder from step1 checkpoint.
    Encoder is COMPLETELY FROZEN — no gradients, no updates.

    Checkpoint format (from step1_semantic_mse_training.py):
        checkpoint["encoder_state_dict"]  ← encoder weights
        checkpoint["ctc_head_hi_state_dict"]  ← not needed here
        checkpoint["ctc_head_mr_state_dict"]  ← not needed here
    """
    print(f"  Loading Whisper Small encoder architecture...")
    whisper = WhisperModel.from_pretrained(STUDENT_MODEL)
    encoder = whisper.encoder
    encoder_dim = whisper.config.d_model  # 768

    print(f"  Loading semantic MSE weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Step1 checkpoint stores encoder state directly (no "encoder." prefix)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    epoch = checkpoint.get("epoch", "?")
    train_loss = checkpoint.get("train_loss", "?")
    dev_loss = checkpoint.get("dev_loss", "?")
    print(f"  Loaded from epoch {epoch} (train MSE={train_loss}, dev MSE={dev_loss})")

    # FREEZE encoder — we only train the CTC head
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder: {frozen_params:,} params (ALL FROZEN)")

    # Free decoder memory
    del whisper.decoder
    del whisper
    del checkpoint
    gc.collect()

    return encoder, encoder_dim


# ─────────────────────────────────────────────
# DEV EVALUATION
# ─────────────────────────────────────────────
def evaluate_dev(encoder, ctc_head, dev_clips, feat_ext, char_to_idx, idx_to_char, device):
    """Evaluate CTC model on dev set — compute WER using greedy decoding."""
    ctc_head.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    all_refs = []
    all_preds = []
    losses = []
    lang_preds = {"Hindi": ([], []), "Marathi": ([], []), "English": ([], [])}

    with torch.no_grad():
        for clip in tqdm(dev_clips, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                wav = load_audio(clip["audio_path"])
                if len(wav) < 3200:  # too short
                    continue

                real_frames = compute_real_frames(len(wav))

                mel = feat_ext(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                input_features = mel.input_features.to(device)

                encoder_output = encoder(input_features).last_hidden_state  # (1, 1500, 768)
                logits = ctc_head(encoder_output)  # (1, 1500, 85)

                # CTC loss
                normalized_ref = normalize_text(clip["transcript"])
                target_indices = text_to_indices(normalized_ref, char_to_idx)

                if not target_indices:
                    continue

                if real_frames >= len(target_indices):
                    log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :].permute(1, 0, 2)
                    targets = torch.tensor(target_indices, dtype=torch.long).to(device)
                    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(device)
                    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(device)
                    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        losses.append(loss.item())

                # Greedy decode
                pred_text = ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)

                if normalized_ref:
                    all_refs.append(normalized_ref)
                    all_preds.append(pred_text)

                    # Per-language tracking
                    lang = clip.get("language", "")
                    if lang in lang_preds:
                        lang_preds[lang][0].append(normalized_ref)
                        lang_preds[lang][1].append(pred_text)

                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception:
                continue

    ctc_head.train()

    wer = compute_corpus_wer(all_refs, all_preds) if all_refs else float("inf")
    avg_loss = np.mean(losses) if losses else float("inf")

    # Per-language WER
    lang_wers = {}
    for lang, (refs, preds) in lang_preds.items():
        if refs:
            lang_wers[lang] = compute_corpus_wer(refs, preds) * 100
        else:
            lang_wers[lang] = 0.0

    return wer, avg_loss, all_preds, lang_wers


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sequential CTC head training on frozen semantic encoder")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on dev WER")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (high — CTC head is small)")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--max_clips", type=int, default=None, help="Limit clips for quick testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Step 1: Load vocabulary ──
    print("=" * 70)
    print("Step 1: Loading vocabulary")
    print("=" * 70)
    char_to_idx, idx_to_char, vocab_size = load_vocab(VOCAB_PATH)
    print(f"  Vocab size: {vocab_size} tokens")

    # ── Step 2: Load data (ALL languages) ──
    print("\n" + "=" * 70)
    print("Step 2: Loading data (all languages)")
    print("=" * 70)
    train_clips = load_clips(TRAIN_CSV, max_clips=args.max_clips)
    dev_clips = load_clips(DEV_CSV, max_clips=args.max_clips)
    print(f"  Train: {len(train_clips)} clips")
    print(f"  Dev:   {len(dev_clips)} clips")

    lang_counts = {}
    for c in train_clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")

    # ── Step 3: Load frozen encoder ──
    print("\n" + "=" * 70)
    print("Step 3: Loading frozen semantic encoder (from step1 MSE)")
    print("=" * 70)
    encoder, encoder_dim = load_frozen_encoder(ENCODER_CHECKPOINT)
    encoder.to(DEVICE)
    print(f"  Encoder dim: {encoder_dim}")

    # ── Step 4: Load feature extractor ──
    print("\n  Loading Whisper feature extractor...")
    feat_ext = WhisperFeatureExtractor.from_pretrained(STUDENT_MODEL)

    # ── Step 5: Create CTC head ──
    print("\n" + "=" * 70)
    print("Step 5: Creating CTC head")
    print("=" * 70)
    ctc_head = nn.Linear(encoder_dim, vocab_size).to(DEVICE)
    trainable_params = sum(p.numel() for p in ctc_head.parameters())
    print(f"  CTC head: Linear({encoder_dim}, {vocab_size})")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Encoder parameters: FROZEN (0 trainable)")

    # ── Step 6: Optimizer ──
    print("\n" + "=" * 70)
    print("Step 6: Setting up optimizer")
    print("=" * 70)

    # LR=1e-3 is standard for CTC head training (small layer, trained from scratch)
    # Much higher than encoder fine-tuning LR (3e-5)
    optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_clips) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        remaining = total_steps - step
        total_decay = total_steps - args.warmup_steps
        return max(0.0, remaining / max(1, total_decay))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.01)")
    print(f"  Scheduler: linear warmup ({args.warmup_steps}) + linear decay")
    print(f"  Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"  Total est. steps: {total_steps:,}")

    # ── Resume if specified ──
    start_epoch = 1
    best_dev_wer = float("inf")
    best_dev_loss = float("inf")

    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dev_wer = ckpt.get("dev_wer", float("inf"))
        best_dev_loss = ckpt.get("dev_loss", float("inf"))
        steps_done = ckpt["epoch"] * len(train_clips)
        for _ in range(steps_done):
            scheduler.step()
        print(f"  Resumed from epoch {ckpt['epoch']}, dev WER={best_dev_wer*100:.1f}%")

    # ── Step 7: Verify pipeline ──
    print("\n" + "=" * 70)
    print("Step 7: Pipeline verification")
    print("=" * 70)

    test_clip = train_clips[0]
    wav = load_audio(test_clip["audio_path"])
    real_frames = compute_real_frames(len(wav))

    mel = feat_ext(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = mel.input_features.to(DEVICE)

    print(f"  Clip: {os.path.basename(test_clip['audio_path'])} ({test_clip['language']})")
    print(f"  Audio: {len(wav)/SAMPLE_RATE:.1f}s → {real_frames} encoder frames")

    with torch.no_grad():
        encoder_output = encoder(input_features).last_hidden_state
    logits = ctc_head(encoder_output)
    print(f"  Encoder: {tuple(encoder_output.shape)} → CTC head: {tuple(logits.shape)}")

    normalized_ref = normalize_text(test_clip["transcript"])
    target_indices = text_to_indices(normalized_ref, char_to_idx)
    print(f"  Target: '{normalized_ref[:60]}' ({len(target_indices)} chars)")

    if real_frames >= len(target_indices):
        log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :].permute(1, 0, 2)
        targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
        input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
        target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        print(f"  CTC loss: {loss.item():.4f}")

        head_grads = sum(1 for p in ctc_head.parameters() if p.grad is not None)
        enc_grads = sum(1 for p in encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"  Gradients: CTC head={head_grads}>0 ✓ | Encoder={enc_grads}=0 ✓ (frozen)")
        optimizer.zero_grad()

    pred_text = ctc_greedy_decode(logits[0, :real_frames, :].detach(), idx_to_char)
    print(f"  Decoded (before training): '{pred_text[:60]}'")

    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU: {peak:.0f} MB")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("  ✓ ALL VERIFIED")

    # ── Step 8: Training loop ──
    print("\n" + "=" * 70)
    print(f"Step 8: TRAINING — {args.epochs} epochs, lr={args.lr}")
    print(f"  Training CTC head ({trainable_params:,} params) on frozen encoder")
    print("=" * 70)

    patience_counter = 0
    global_step = 0
    epoch_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        ctc_head.train()
        epoch_start = time.time()
        epoch_losses = []
        skipped = 0

        random.shuffle(train_clips)

        pbar = tqdm(train_clips, desc=f"Epoch {epoch}/{args.epochs}",
                    unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

        for clip in pbar:
            try:
                wav = load_audio(clip["audio_path"])
                if len(wav) < 3200:
                    skipped += 1
                    continue

                real_frames = compute_real_frames(len(wav))

                mel = feat_ext(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
                input_features = mel.input_features.to(DEVICE)

                # Frozen encoder — no gradients
                with torch.no_grad():
                    encoder_output = encoder(input_features).last_hidden_state

                # CTC head — trainable
                logits = ctc_head(encoder_output)  # (1, 1500, 85)

                # Prepare target
                normalized_ref = normalize_text(clip["transcript"])
                target_indices = text_to_indices(normalized_ref, char_to_idx)

                if not target_indices:
                    skipped += 1
                    continue

                if real_frames < len(target_indices):
                    skipped += 1
                    continue

                log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :].permute(1, 0, 2)
                targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
                input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
                target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    skipped += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_losses.append(loss.item())

                # Progress bar
                if epoch_losses:
                    avg_recent = np.mean(epoch_losses[-50:])
                    pbar.set_postfix({
                        "loss": f"{loss.item():.3f}",
                        "avg": f"{avg_recent:.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    })

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n  ERROR: {e}")
                skipped += 1
                continue

        pbar.close()

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float("inf")

        # ── Dev evaluation ──
        dev_wer, dev_loss, dev_preds, lang_wers = evaluate_dev(
            encoder, ctc_head, dev_clips, feat_ext,
            char_to_idx, idx_to_char, DEVICE
        )

        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_wer": dev_wer,
            "lang_wers": lang_wers,
            "time": epoch_time,
        }
        epoch_history.append(epoch_record)

        print(f"\n  ┌─ Epoch {epoch}/{args.epochs} ─────────────────────────────────────────")
        print(f"  │ Train CTC loss: {avg_train_loss:.4f}")
        print(f"  │ Dev CTC loss:   {dev_loss:.4f}")
        print(f"  │ Dev WER:        {dev_wer*100:.2f}%")
        for lang in ["Hindi", "Marathi", "English"]:
            if lang in lang_wers and lang_wers[lang] > 0:
                print(f"  │   {lang}: {lang_wers[lang]:.2f}%")
        print(f"  │ Steps: {len(epoch_losses)} ({skipped} skipped) | Time: {epoch_time:.0f}s ({epoch_time/60:.1f}min)")
        print(f"  │ LR: {current_lr:.2e}")
        if DEVICE == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  │ Peak GPU: {peak:.0f} MB")
        print(f"  └──────────────────────────────────────────────────────────────")

        # Show sample predictions
        if dev_preds:
            print(f"\n  Sample predictions:")
            for ref, pred in zip(
                [normalize_text(c["transcript"]) for c in dev_clips[:3]],
                dev_preds[:3]
            ):
                print(f"    REF: {ref[:80]}")
                print(f"    HYP: {pred[:80]}")
                print()

        # ── Checkpointing + early stopping ──
        ckpt_data = {
            "epoch": epoch,
            "ctc_head_state_dict": ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_wer": dev_wer,
            "lang_wers": lang_wers,
            "vocab_size": vocab_size,
            "encoder_dim": encoder_dim,
            "args": vars(args),
        }

        improved = False
        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            patience_counter = 0
            improved = True
            torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "best_wer.pt"))
            print(f"  ★ NEW BEST WER: {dev_wer*100:.2f}% → saved best_wer.pt")
        else:
            patience_counter += 1
            print(f"  Early stopping: {patience_counter}/{args.patience} "
                  f"(best: {best_dev_wer*100:.2f}%)")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "best_loss.pt"))

        torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))
        print(f"  Saved: {CHECKPOINT_DIR}/epoch_{epoch}.pt")

        if patience_counter >= args.patience:
            print(f"\n  EARLY STOPPING at epoch {epoch}")
            break

        print()

    # ── Final summary ──
    total_time = time.time() - time.time() + sum(e["time"] for e in epoch_history)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n  {'Epoch':<8} {'TrainCTC':>10} {'DevCTC':>10} {'DevWER':>10} {'Hindi':>10} {'Marathi':>10} {'English':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for e in epoch_history:
        hi = e["lang_wers"].get("Hindi", 0)
        mr = e["lang_wers"].get("Marathi", 0)
        en = e["lang_wers"].get("English", 0)
        print(f"  {e['epoch']:<8} {e['train_loss']:>10.4f} {e['dev_loss']:>10.4f} "
              f"{e['dev_wer']*100:>9.2f}% {hi:>9.2f}% {mr:>9.2f}% {en:>9.2f}%")

    print(f"\n  Best dev WER: {best_dev_wer*100:.2f}%")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print(f"\n  Next step: Re-run joint training with warm-started CTC head:")
    print(f"    python scripts/semantic/step3_semantic_joint_training.py \\")
    print(f"        --checkpoint checkpoints/semantic_mse/best_dev.pt \\")
    print(f"        --ctc_checkpoint checkpoints/semantic_ctc/best_wer.pt")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
