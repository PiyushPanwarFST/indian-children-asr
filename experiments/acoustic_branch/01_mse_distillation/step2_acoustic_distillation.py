"""
Step 2: Acoustic Branch — Feature-level Knowledge Distillation
==============================================================

WHAT THIS SCRIPT DOES (high-level):
    Teacher model (Kid-Whisper Medium MyST) has been fine-tuned on children's
    speech — it already knows HOW children sound (acoustic patterns like pitch,
    speed, pronunciation errors, etc.). We want our smaller student model
    (Whisper Small) to learn the same acoustic understanding.

    HOW? By making the student's encoder produce the SAME hidden features
    as the teacher's encoder for the same audio input. We compare their
    encoder outputs frame-by-frame using MSE loss.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────┐
    │  Audio (mel spectrogram)                            │
    │       ↓                      ↓                     │
    │  Student Encoder         Teacher Encoder (FROZEN)   │
    │  (Whisper Small)         (Kid-Whisper Medium)       │
    │       ↓                      ↓                     │
    │  (B, 1500, 768)         (B, 1500, 1024)            │
    │       ↓                                            │
    │  Projection Layer                                  │
    │  Linear(768 → 1024)                                │
    │       ↓                      ↓                     │
    │  (B, 1500, 1024)        (B, 1500, 1024)            │
    │       └──────── MSE Loss ────────┘                 │
    │       (only on REAL frames, not padding)           │
    └─────────────────────────────────────────────────────┘

WHY THIS IS CORRECT (vs semantic branch problems):
    ✅ Both models receive the SAME mel spectrogram as input
    ✅ Both models produce encoder features with SAME frame count (1500)
    ✅ Projection layer handles dimension mismatch (768 → 1024)
    ✅ MSE is computed on REAL frames only (padding excluded)
    ✅ Teacher runs ONLINE (not pre-computed) — no stale data
    ✅ No vocabulary mismatch problem — we compare features, not text/logits
    ✅ Uses ALL languages (Hindi + Marathi + English) — acoustic patterns
       are language-independent (children sound like children regardless)

    This is the standard approach used in Distil-Whisper (Gandhe et al. 2024),
    CARE (Rumberg et al. 2022), and FitNets (Romero et al. 2015).

WHAT IS BEING LEARNED:
    - Children speak faster/slower than adults → encoder learns child timing
    - Children's pitch is higher → encoder learns child frequency patterns
    - Children mispronounce words → encoder learns to handle these variations
    - Children's speech has more pauses/hesitations → encoder learns this

Prerequisites:
    - ASER dataset with splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Quick test (20 clips, 2 epochs)
    python scripts/acoustic/step2_acoustic_distillation.py --max_clips 20 --epochs 2

    # Full training on HPC
    python scripts/acoustic/step2_acoustic_distillation.py --epochs 20 --lr 3e-5 --patience 5
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_CSV = "ASER-Dataset/splits/asr_train.csv"
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"
CHECKPOINT_DIR = "checkpoints/acoustic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds

# Model names
STUDENT_MODEL = "openai/whisper-small"
TEACHER_MODEL = "aadel4/kid-whisper-medium-en-myst"


# ─────────────────────────────────────────────
# STUDENT MODEL: Whisper Encoder + Projection
# ─────────────────────────────────────────────
class WhisperAcoustic(nn.Module):
    """
    Student model: Whisper Small encoder + projection layer.

    PURPOSE:
        Whisper Small's encoder outputs 768-dim features per frame.
        Kid-Whisper Medium's encoder outputs 1024-dim features per frame.
        We can't directly compare 768-dim vs 1024-dim with MSE.
        So we add a projection layer (Linear 768→1024) to map student
        features into the teacher's 1024-dim space.

    WHAT IS TRAINABLE:
        ✅ Student encoder (Whisper Small) — all 12 transformer layers
        ✅ Projection layer (768→1024) — 1 linear layer
        ❌ Teacher encoder (Kid-Whisper Medium) — completely FROZEN, not in this class

    ARCHITECTURE:
        mel spectrogram (80, 3000) → Whisper Small Encoder → (1500, 768)
                                                                 ↓
                                                        Dropout(0.1)
                                                                 ↓
                                                      Linear(768, 1024)
                                                                 ↓
                                                          (1500, 1024)  ← compare with teacher

    Args:
        student_model_name (str): HuggingFace model ID for student.
            Example: "openai/whisper-small"
        teacher_dim (int): Dimension of teacher's encoder output.
            For Kid-Whisper Medium: 1024
        dropout (float): Dropout rate before projection. Default: 0.1
            Helps prevent overfitting during fine-tuning.
    """

    def __init__(self, student_model_name, teacher_dim, dropout=0.1):
        super().__init__()

        # Load full Whisper model, but we only keep the encoder
        whisper = WhisperModel.from_pretrained(student_model_name)
        self.encoder = whisper.encoder
        self.student_dim = self.encoder.config.d_model  # 768 for whisper-small

        # Projection layer: maps student features (768) to teacher space (1024)
        # This is necessary because student and teacher have different hidden dimensions.
        # Without this, we can't compute MSE between their outputs.
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.student_dim, teacher_dim)

        # Delete decoder — we only need the encoder for feature extraction.
        # This saves ~50% GPU memory.
        del whisper.decoder

    def forward(self, input_features):
        """
        Forward pass: encode audio → project to teacher dimension.

        Args:
            input_features (Tensor): Mel spectrogram from WhisperProcessor.
                Shape: (batch_size, 80, 3000)
                - 80 = mel frequency bins
                - 3000 = time frames (30 seconds × 100 frames/sec)
                - Shorter audios are zero-padded to 3000 frames

        Returns:
            student_features (Tensor): Raw encoder output before projection.
                Shape: (batch_size, 1500, 768)
                - 1500 = encoder frames (3000 mel frames ÷ 2 from conv stride)
                - 768 = whisper-small hidden dimension
                - This is returned for potential future use, but NOT used in loss

            projected (Tensor): Student features projected to teacher dimension.
                Shape: (batch_size, 1500, 1024)
                - 1024 = kid-whisper-medium hidden dimension
                - THIS is what gets compared with teacher via MSE loss
        """
        # STEP 1: Pass mel spectrogram through Whisper Small's encoder
        # The encoder has 2 conv layers (stride 2) + 12 transformer layers
        # Input: (B, 80, 3000) → Output: (B, 1500, 768)
        student_features = self.encoder(input_features).last_hidden_state

        # STEP 2: Apply dropout + project to teacher's dimension
        # Dropout prevents co-adaptation of features during fine-tuning
        # Linear(768, 1024) maps each frame from 768-dim to 1024-dim
        projected = self.projection(self.dropout(student_features))

        return student_features, projected


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_clips(csv_path, max_clips=None):
    """
    Load audio clips from a split CSV file.

    KEY DIFFERENCE from semantic branch:
        Semantic branch loads only Hindi/Marathi (needs text for CTC).
        Acoustic branch loads ALL languages (Hindi + Marathi + English)
        because we're comparing encoder FEATURES, not text. A child's
        voice sounds like a child's voice regardless of language.

    Args:
        csv_path (str): Path to split CSV file.
            Expected columns: audio_path, language, duration_sec
            Example: "ASER-Dataset/splits/asr_train.csv"
        max_clips (int, optional): Limit number of clips loaded.
            Used for quick testing. None = load all clips.

    Returns:
        list[dict]: List of clip dictionaries, each with:
            - "audio_path" (str): Path to .wav file
            - "language" (str): "Hindi", "Marathi", or "English"
            - "duration" (float): Duration in seconds

    Note:
        Clips longer than MAX_DURATION (30s) are skipped because
        Whisper's mel spectrogram is fixed at 30 seconds.
    """
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append({
                "audio_path": row["audio_path"],
                "language": row["language"],
                "duration": float(row["duration_sec"]),
            })

    # Skip clips longer than 30 seconds — Whisper's mel spectrogram
    # is fixed at 30s (3000 mel frames). Longer clips would be truncated.
    clips = [c for c in clips if c["duration"] <= MAX_DURATION]

    if max_clips:
        clips = clips[:max_clips]

    return clips


# ─────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────
def prepare_audio(audio_path, processor):
    """
    Load an audio file and convert it to Whisper's mel spectrogram format.

    This function handles:
        1. Loading the .wav file
        2. Resampling to 16kHz if needed
        3. Converting stereo to mono if needed
        4. Truncating to 30 seconds max
        5. Creating mel spectrogram via WhisperProcessor
        6. Computing how many encoder frames are REAL (not padding)

    WHY WE NEED real_encoder_frames:
        Whisper ALWAYS produces (1, 80, 3000) mel spectrogram regardless
        of audio length. Short audio gets ZERO-PADDED to 3000 frames.
        After encoder's conv layers (stride 2): 3000 → 1500 frames.

        Example: A 5-second clip has ~500 mel frames, rest is padding.
        Encoder output has ~250 real frames + 1250 padding frames.

        We compute MSE loss ONLY on real frames (not padding) because:
        - Padding frames contain no meaningful audio information
        - Including them would dilute the loss with garbage comparisons
        - Both teacher and student would output near-zero for padding,
          making the loss artificially low

    Args:
        audio_path (str): Path to .wav file
        processor (WhisperProcessor): Whisper's feature extractor

    Returns:
        input_features (Tensor): Mel spectrogram, shape (1, 80, 3000)
            - Same tensor is fed to BOTH student and teacher
        real_encoder_frames (int): Number of non-padding encoder frames
            - Used to slice loss: loss = MSE(student[:real_frames], teacher[:real_frames])
            - Range: 1 to 1500
    """
    # STEP 1: Load audio waveform
    wav, sr = torchaudio.load(audio_path)

    # STEP 2: Resample to 16kHz if needed (Whisper expects 16kHz)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # STEP 3: Convert stereo to mono by averaging channels
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # STEP 4: Truncate to 30 seconds max (16000 samples/sec × 30 sec = 480000 samples)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    audio_np = wav.squeeze(0).numpy()

    # STEP 5: Create mel spectrogram using Whisper's feature extractor
    # return_attention_mask=True gives us a mask showing which mel frames are real
    inputs = processor.feature_extractor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features  # (1, 80, 3000) — always this shape

    # STEP 6: Calculate real encoder frames
    # attention_mask has 1 for real mel frames, 0 for padding
    # Encoder's conv layers have stride 2, so mel frames ÷ 2 = encoder frames
    real_mel_frames = inputs.attention_mask.sum().item()
    real_encoder_frames = int(real_mel_frames) // 2

    return input_features, real_encoder_frames


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_dev(student_model, teacher_encoder, dev_clips, processor, device):
    """
    Evaluate student model on dev set — compute average MSE loss.

    PURPOSE:
        Monitor generalization during training. If dev loss goes up while
        train loss goes down → overfitting. If both go down → learning
        is working. In our 20-epoch test run:
            Epoch 1:  train=2.30, dev=1.69
            Epoch 20: train=0.72, dev=0.81  (gap only 0.088 — very healthy!)

    HOW IT WORKS:
        For each dev clip:
        1. Same mel spectrogram → both student and teacher
        2. Student produces projected features (1500, 1024)
        3. Teacher produces features (1500, 1024) — FROZEN, no gradients
        4. MSE computed on real frames only
        5. Average all per-clip losses → epoch dev loss

    Args:
        student_model (WhisperAcoustic): Student model (encoder + projection)
        teacher_encoder: Kid-Whisper Medium's encoder (frozen)
        dev_clips (list[dict]): Dev set clips from load_clips()
        processor (WhisperProcessor): For creating mel spectrograms
        device (str): "cuda" or "cpu"

    Returns:
        float: Average MSE loss on dev set.
            Lower = student features are closer to teacher features.
            inf if no valid clips (shouldn't happen normally).
    """
    student_model.eval()  # Disable dropout for evaluation
    losses = []

    with torch.no_grad():  # No gradients needed — just measuring performance
        for clip in dev_clips:
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)
            input_features = input_features.to(device)

            if real_frames == 0:
                continue

            # Student: encode + project to teacher dim
            _, projected = student_model(input_features)  # (1, 1500, 1024)

            # Teacher: encode (frozen, no updates ever)
            teacher_features = teacher_encoder(input_features).last_hidden_state  # (1, 1500, 1024)

            # MSE only on real frames — exclude zero-padded frames
            loss = nn.functional.mse_loss(
                projected[:, :real_frames, :],
                teacher_features[:, :real_frames, :],
            )

            if not torch.isnan(loss) and not torch.isinf(loss):
                losses.append(loss.item())

    student_model.train()  # Re-enable dropout for next training epoch

    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    """
    Main training function for acoustic knowledge distillation.

    TRAINING FLOW:
        1. Load train/dev clips from split CSVs (ALL languages)
        2. Load teacher (Kid-Whisper Medium) — freeze completely
        3. Load student (Whisper Small encoder + projection)
        4. For each epoch:
           a. Shuffle training clips
           b. For each clip:
              - Create mel spectrogram from audio
              - Student forward: mel → encoder → projection → (1500, 1024)
              - Teacher forward: mel → encoder → (1500, 1024) [no grad]
              - MSE loss on real frames only
              - Backprop through student only (teacher frozen)
           c. Evaluate on dev set
           d. Save checkpoints, check early stopping
        5. Save final summary

    WHAT GETS UPDATED:
        ✅ Student encoder weights (all 12 transformer layers)
        ✅ Projection layer weights (Linear 768→1024)
        ❌ Teacher encoder weights (NEVER — completely frozen)

    LOSS INTERPRETATION:
        MSE loss = average squared difference per feature dimension per frame
        - Epoch 1: ~2.0-3.0 (student and teacher outputs are very different)
        - Epoch 20: ~0.5-1.0 (student has learned to mimic teacher)
        - Train-Dev gap < 0.2 = healthy, no overfitting
    """
    parser = argparse.ArgumentParser(description="Acoustic branch: Feature-level KD")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (stop if dev loss doesn't improve)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (3e-5 is standard for fine-tuning)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps before decay")
    parser.add_argument("--max_clips", type=int, default=None, help="Limit clips (for quick testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path (e.g., checkpoints/acoustic/epoch_10.pt)")
    args = parser.parse_args()

    # ── Reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load Data ──
    print("Loading data...")
    train_clips = load_clips(TRAIN_CSV, max_clips=args.max_clips)
    dev_clips = load_clips(DEV_CSV, max_clips=args.max_clips)

    print(f"  Training: {len(train_clips)} clips")
    print(f"  Dev:      {len(dev_clips)} clips")

    lang_counts = {}
    for c in train_clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count} clips")

    # ── Load Processor ──
    print("\nLoading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    # ── Load Teacher (Kid-Whisper Medium MyST — FROZEN) ──
    # Kid-Whisper was fine-tuned on MyST (My Science Tutor) dataset of children's speech.
    # It already understands children's acoustic patterns. We want to transfer this.
    # We load the FULL model (encoder + decoder), extract only the encoder, delete the rest.
    print("Loading teacher model (Kid-Whisper Medium MyST)...")
    teacher_full = WhisperForConditionalGeneration.from_pretrained(TEACHER_MODEL)
    teacher_encoder = teacher_full.model.encoder.to(DEVICE)
    teacher_dim = teacher_full.config.d_model  # 1024 (medium model = 1024 hidden dim)
    del teacher_full  # Free decoder memory (~500MB) — we only need the encoder
    torch.cuda.empty_cache()

    # FREEZE teacher completely — we NEVER update its weights.
    # .eval() disables dropout/batchnorm. requires_grad=False prevents gradient computation.
    # The teacher is a fixed reference — student learns to match its outputs.
    teacher_encoder.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    print(f"  Teacher encoder dim: {teacher_dim}")
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  GPU memory after teacher: {mem:.0f} MB")

    # ── Load Student ──
    print("Loading student model (Whisper Small encoder + projection)...")
    student_model = WhisperAcoustic(
        student_model_name=STUDENT_MODEL,
        teacher_dim=teacher_dim,
        dropout=0.1,
    )
    student_model.to(DEVICE)

    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  GPU memory after both models: {mem:.0f} MB")

    # ── Setup Optimizer and LR Scheduler ──
    # AdamW = Adam with weight decay (L2 regularization)
    # Only student parameters are optimized — teacher has requires_grad=False
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)

    total_steps = len(train_clips) * args.epochs

    # Learning rate schedule: linear warmup → linear decay
    # Warmup: LR goes 0 → args.lr over first 500 steps (prevents early instability)
    # Decay: LR goes args.lr → 0 over remaining steps (fine-grained updates at end)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)  # Linear warmup
        else:
            remaining = total_steps - step
            total_decay = total_steps - args.warmup_steps
            return max(0.0, remaining / max(1, total_decay))  # Linear decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume from checkpoint if specified ──
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        student_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Fast-forward scheduler to correct step
        steps_done = ckpt["epoch"] * len(train_clips)
        for _ in range(steps_done):
            scheduler.step()
        print(f"  Resumed from epoch {ckpt['epoch']}, starting at epoch {start_epoch}")
        print(f"  Previous train loss: {ckpt['train_loss']:.6f}")
        print(f"  Previous dev loss: {ckpt['dev_loss']:.6f}")

    # ── Verify Pipeline (1 clip) ──
    print("\nVerifying pipeline with 1 clip...")
    test_clip = train_clips[0]
    input_features, real_frames = prepare_audio(test_clip["audio_path"], processor)
    input_features = input_features.to(DEVICE)

    print(f"  Audio: {test_clip['audio_path'].split('/')[-1]}")
    print(f"  Language: {test_clip['language']}")
    print(f"  Input features: {input_features.shape}")
    print(f"  Real encoder frames: {real_frames} / 1500")

    # Student forward
    student_features, projected = student_model(input_features)
    print(f"  Student output: {student_features.shape}")
    print(f"  Projected output: {projected.shape}")

    # Teacher forward
    with torch.no_grad():
        teacher_features = teacher_encoder(input_features).last_hidden_state
    print(f"  Teacher output: {teacher_features.shape}")

    # MSE loss on real frames only
    loss = nn.functional.mse_loss(
        projected[:, :real_frames, :],
        teacher_features[:, :real_frames, :],
    )
    print(f"  MSE loss: {loss.item():.4f}")
    print("  Pipeline verified ✓")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  GPU peak memory: {peak_mem:.0f} MB")

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"  TRAINING: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
    print(f"{'='*60}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_train_loss = float("inf")
    best_dev_loss = float("inf")
    patience_counter = 0
    global_step = 0
    epoch_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        student_model.train()
        teacher_encoder.eval()  # teacher always in eval mode
        epoch_start = time.time()
        epoch_losses = []
        skipped = 0

        random.shuffle(train_clips)

        for i, clip in enumerate(train_clips):
            step_start = time.time()

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 1: Prepare audio → mel spectrogram
            # Same mel goes to BOTH student and teacher
            # ══════════════════════════════════════════════════════
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)

            if real_frames == 0:
                skipped += 1
                continue

            input_features = input_features.to(DEVICE)  # (1, 80, 3000)

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 2: Student forward pass
            # mel (1, 80, 3000) → encoder → (1, 1500, 768) → projection → (1, 1500, 1024)
            # Gradients ARE tracked here — student will be updated
            # ══════════════════════════════════════════════════════
            _, projected = student_model(input_features)  # (1, 1500, 1024)

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 3: Teacher forward pass (NO gradients)
            # Same mel → teacher encoder → (1, 1500, 1024)
            # torch.no_grad() = teacher is a fixed reference, never updated
            # ══════════════════════════════════════════════════════
            with torch.no_grad():
                teacher_features = teacher_encoder(input_features).last_hidden_state  # (1, 1500, 1024)

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 4: MSE Loss (ONLY on real frames)
            # projected[:, :real_frames, :] = student's first `real_frames` frames
            # teacher_features[:, :real_frames, :] = teacher's first `real_frames` frames
            #
            # MSE = mean((student_frame - teacher_frame)²) averaged over all
            #        real frames and all 1024 dimensions
            #
            # ✅ This is CORRECT because:
            # - Same input → both models, so frame i in student = frame i in teacher
            # - Both output 1024 dims (after projection), so element-wise comparison works
            # - Padding frames excluded → loss reflects actual audio content
            # ══════════════════════════════════════════════════════
            loss = nn.functional.mse_loss(
                projected[:, :real_frames, :],
                teacher_features[:, :real_frames, :],
            )

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 5: Backward pass + parameter update
            # Gradients flow ONLY through student (teacher frozen)
            # Gradient clipping prevents exploding gradients
            # ══════════════════════════════════════════════════════
            optimizer.zero_grad()        # Clear old gradients
            loss.backward()              # Compute gradients for student params
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)  # Prevent explosion
            optimizer.step()             # Update student encoder + projection weights
            scheduler.step()             # Update learning rate
            global_step += 1

            epoch_losses.append(loss.item())

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Progress every 10 clips
            if (i + 1) % 10 == 0 or (i + 1) == len(train_clips):
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                elapsed = time.time() - step_start
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch} | {i+1}/{len(train_clips)} | "
                    f"loss: {avg_loss:.6f} | lr: {current_lr:.2e} | "
                    f"clip: {elapsed:.2f}s",
                    flush=True,
                )

        # ── Epoch Summary ──
        epoch_time = time.time() - epoch_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")

        # ── Dev Evaluation ──
        avg_dev_loss = evaluate_dev(student_model, teacher_encoder, dev_clips, processor, DEVICE)

        gap = avg_dev_loss - avg_train_loss
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": avg_dev_loss,
            "gap": gap,
            "time": epoch_time,
            "skipped": skipped,
        }
        epoch_history.append(epoch_record)

        print(f"\n  {'─'*55}")
        print(f"  Epoch {epoch}/{args.epochs} | time: {epoch_time:.0f}s ({epoch_time/60:.1f} min)")
        print(f"  Train loss: {avg_train_loss:.6f} | Dev loss: {avg_dev_loss:.6f} | Gap: {gap:.6f}")
        print(f"  Skipped: {skipped} clips | GPU peak: {gpu_mem:.0f} MB")
        print(f"  {'─'*55}\n")

        # ── Save Checkpoints ──
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "dev_loss": avg_dev_loss,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_train_model.pt"))

        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            patience_counter = 0
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_dev_model.pt"))
            print(f"  ✓ New best dev loss: {best_dev_loss:.6f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Training Complete ──
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Epoch-by-epoch summary:")
    print(f"  {'Epoch':>5} {'Train':>12} {'Dev':>12} {'Gap':>12} {'Time':>8}")
    print(f"  {'─'*50}")
    for r in epoch_history:
        print(
            f"  {r['epoch']:>5} {r['train_loss']:>12.6f} {r['dev_loss']:>12.6f} "
            f"{r['gap']:>12.6f} {r['time']:>7.0f}s"
        )

    print(f"\n  Best train loss: {best_train_loss:.6f}")
    print(f"  Best dev loss:   {best_dev_loss:.6f}")
    print(f"  Checkpoints in:  {CHECKPOINT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
