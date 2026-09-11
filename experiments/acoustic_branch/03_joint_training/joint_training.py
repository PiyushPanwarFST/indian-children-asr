"""
Joint Training: MSE + CTC Simultaneous Optimization
====================================================

WHAT THIS SCRIPT DOES:
    Trains the student encoder, projection layer, AND CTC head all together
    with a combined loss:  total_loss = alpha * MSE + (1-alpha) * CTC

    This is DIFFERENT from our sequential approach where:
        Step 1: Trained encoder with MSE only (acoustic distillation)
        Step 2: Froze encoder, trained CTC head only (semantic branch)

    In joint training, the encoder receives gradients from BOTH losses
    simultaneously. This means the encoder learns features that are:
        1. Similar to the teacher's acoustic representations (MSE branch)
        2. Directly useful for text prediction (CTC branch)

    Sequential training had a mismatch: the encoder was optimized only for
    MSE, so its features were great for matching the teacher but not
    necessarily optimal for CTC decoding. Joint training eliminates this.

WHY THIS SHOULD WORK BETTER:
    In sequential training (81.9% WER), the encoder was frozen when training
    CTC. The CTC head could only work with whatever features the encoder
    already produced — it couldn't ask the encoder to change.

    In joint training, if the CTC head needs different features to predict
    text better, it sends gradients back through the encoder to reshape
    the features. The encoder balances both signals.

    Reference: CARE paper (Rumberg et al., 2022) uses this exact dual-loss
    approach for children's speech recognition.

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────┐
    │  Audio (.wav)                                            │
    │       ↓                                                  │
    │  Whisper Processor → Mel Spectrogram (1, 80, 3000)       │
    │       ↓                            ↓                     │
    │  Student Encoder (TRAINABLE)    Teacher Encoder (FROZEN)  │
    │  (Whisper Small, 12 layers)     (Kid-Whisper Med, 24 layers)
    │       ↓                            ↓                     │
    │  (1, 1500, 768)                (1, 1500, 1024)           │
    │       ↓                                                  │
    │  ┌────┴─────────────────────┐                            │
    │  │                          │                            │
    │  ↓                          ↓                            │
    │  Projection Layer        CTC Head (TRAINABLE)            │
    │  Linear(768→1024)        Linear(768→85)                  │
    │  (TRAINABLE)                ↓                            │
    │       ↓                 Logits (1500, 85)                │
    │  (1, 1500, 1024)            ↓                            │
    │       ↓                 CTC Loss ← Ground Truth Text     │
    │  MSE Loss ← Teacher features                            │
    │                                                          │
    │  Total Loss = α × MSE + (1-α) × CTC                     │
    │       ↓                                                  │
    │  Backprop → updates Encoder + Projection + CTC Head      │
    └──────────────────────────────────────────────────────────┘

WHAT GETS UPDATED:
    ✅ Student encoder     — ~88M params (12 transformer layers, UNFROZEN)
    ✅ Projection layer    — ~787K params (Linear 768→1024 + bias)
    ✅ CTC head            — ~65K params (Linear 768→85 + bias)
    Total trainable: ~89M params

    ❌ Teacher encoder     — ~300M params (FROZEN, reference only)

WARM START:
    - Encoder + Projection: from checkpoints/acoustic/hpc_best_dev_model.pt
      (our HPC-trained acoustic distillation checkpoint)
    - CTC head: from checkpoints/semantic/hpc_ctc_best_wer.pt
      (our best sequential CTC head, already knows char-to-frame mapping)

    Starting from trained weights means the model doesn't have to learn
    everything from scratch — it just needs to fine-tune the balance
    between the two objectives.

LOSS COMBINATION:
    total_loss = alpha * MSE_loss + (1 - alpha) * CTC_loss

    alpha = 0.5 means equal weight to both branches.

    References for this approach:
    - CARE paper (Rumberg et al., 2022): dual-branch with CTC for children's ASR
    - FitNets (Romero et al., 2015): alpha=0.5 for knowledge distillation
    - DistilBERT (Sanh et al., 2019): combined distillation + task loss

Prerequisites:
    - Acoustic checkpoint: checkpoints/acoustic/hpc_best_dev_model.pt
    - CTC checkpoint: checkpoints/semantic/hpc_ctc_best_wer.pt
    - Teacher model: aadel4/kid-whisper-medium-en-myst (cached on HPC)
    - Vocabulary: ASER-Dataset/vocab.json (85 tokens)
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Quick test (2000 clips, 5 epochs)
    python scripts/joint/joint_training.py --max_clips 2000 --epochs 5

    # Full training (all clips, 30 epochs)
    python scripts/joint/joint_training.py --epochs 30 --patience 5

    # Resume from checkpoint
    python scripts/joint/joint_training.py --epochs 30 --resume checkpoints/joint/joint_epoch_5.pt

References:
    - CTC Loss: Graves et al., "Connectionist Temporal Classification", ICML 2006
    - CARE: Rumberg et al., "Children's ASR Revisited", 2022
    - FitNets: Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015
    - DistilBERT: Sanh et al., "DistilBERT", 2019
    - AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
    - Born-Again Networks: Furlanello et al., ICML 2018
"""

import argparse
import csv
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

from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration

# ─── Add project root to path so we can import utils ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.wer import normalize_text, compute_corpus_wer


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_train.csv")
DEV_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_dev.csv")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "ASER-Dataset", "vocab.json")

# Warm-start checkpoints from previous training stages
ACOUSTIC_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "acoustic", "hpc_best_dev_model.pt")
CTC_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "semantic", "hpc_ctc_best_wer.pt")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "joint")

# Model identifiers (HuggingFace)
STUDENT_MODEL = "openai/whisper-small"
TEACHER_MODEL = "aadel4/kid-whisper-medium-en-myst"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds (Whisper's maximum input length)


# ─────────────────────────────────────────────
# VOCABULARY (same as ctc_training.py)
# ─────────────────────────────────────────────
def load_vocab(vocab_path):
    """
    Load character vocabulary from vocab.json.

    Args:
        vocab_path (str): Path to vocab.json file.
            Created by build_vocab.py.

    Returns:
        char_to_idx (dict): Maps character -> index.
            Example: {"<blank>": 0, "<space>": 1, "a": 3, ...}
        idx_to_char (dict): Maps index -> character (reverse lookup for decoding).
        vocab_size (int): Total number of tokens (85 for our dataset).
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    idx_to_char = {v: k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    return char_to_idx, idx_to_char, vocab_size


def text_to_indices(text, char_to_idx):
    """
    Convert normalized text to a list of vocabulary indices.

    This is how we prepare the TARGET (ground truth) for CTC loss.
    Each character in the text is mapped to its vocab index.
    Spaces become the <space> token (index 1).
    Unknown characters become <unk> (index 2).

    Args:
        text (str): Normalized text (lowercase, no punctuation).
        char_to_idx (dict): Vocabulary mapping from load_vocab().

    Returns:
        list[int]: List of character indices.

    Note:
        <blank> (index 0) is NEVER in the target. It is only used
        by CTC internally to represent "no output at this frame."
    """
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
    Greedy CTC decoding — standard method used in Wav2Vec 2.0 and DeepSpeech.

    Steps:
        1. argmax per frame -> predicted token index
        2. Collapse consecutive duplicates
        3. Remove blank tokens (index 0)
        4. Map indices to characters
        5. Join into text string

    Args:
        logits (Tensor): CTC output, shape (T, vocab_size).
        idx_to_char (dict): Reverse vocab mapping {index: character}.

    Returns:
        str: Decoded text string.
    """
    predicted_ids = logits.argmax(dim=-1).tolist()

    decoded_ids = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id:
            if idx != 0:  # Skip blanks
                decoded_ids.append(idx)
        prev_id = idx

    chars = []
    for idx in decoded_ids:
        token = idx_to_char.get(idx, "")
        if token == "<space>":
            chars.append(" ")
        elif token == "<blank>" or token == "<unk>":
            continue
        else:
            chars.append(token)

    return "".join(chars)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_clips(csv_path, max_clips=None):
    """
    Load audio clips and their transcripts from a split CSV file.

    Joint training needs both audio AND transcript because:
        - MSE branch: audio -> encoder features -> compare with teacher
        - CTC branch: audio -> encoder features -> predict text -> compare with ground truth

    Args:
        csv_path (str): Path to split CSV (e.g., asr_train.csv).
        max_clips (int, optional): Limit clips for quick testing.

    Returns:
        list[dict]: List of clips, each with:
            - "audio_path" (str): Path to .wav file
            - "transcript" (str): Ground truth text
            - "language" (str): "Hindi", "Marathi", or "English"
            - "duration" (float): Duration in seconds
    """
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration_sec"])
            if dur > MAX_DURATION:
                continue
            clips.append({
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],
                "language": row["language"],
                "duration": dur,
            })

    if max_clips:
        clips = clips[:max_clips]

    return clips


# ─────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────
def prepare_audio(audio_path, processor):
    """
    Load audio file and convert to Whisper mel spectrogram.

    Same function as acoustic branch and CTC training — consistent
    audio processing across all experiments.

    Args:
        audio_path (str): Path to .wav file.
        processor (WhisperProcessor): Whisper's feature extractor.

    Returns:
        input_features (Tensor): Mel spectrogram, shape (1, 80, 3000).
        real_encoder_frames (int): Number of non-padding encoder frames (1 to 1500).
            Whisper's conv layers have stride 2, so 3000 mel frames -> 1500 encoder frames.
            For shorter audio, only the first `real_encoder_frames` are meaningful.
    """
    wav, sr = torchaudio.load(audio_path)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    audio_np = wav.squeeze(0).numpy()

    inputs = processor.feature_extractor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features  # (1, 80, 3000)
    real_mel_frames = inputs.attention_mask.sum().item()
    real_encoder_frames = int(real_mel_frames) // 2

    return input_features, real_encoder_frames


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_student_encoder(checkpoint_path):
    """
    Load student encoder with trained weights — UNFROZEN for joint training.

    KEY DIFFERENCE from ctc_training.py:
        In CTC training, the encoder was FROZEN (no gradients).
        In joint training, the encoder is UNFROZEN (receives gradients from
        both MSE and CTC losses). This is the whole point of joint training.

    Steps:
        1. Load fresh Whisper Small model (to get encoder architecture)
        2. Load HPC acoustic checkpoint (187 encoder keys + projection weights)
        3. Replace encoder weights with our trained weights
        4. Keep encoder UNFROZEN (requires_grad = True, which is the default)

    Args:
        checkpoint_path (str): Path to acoustic distillation checkpoint.
            e.g., "checkpoints/acoustic/hpc_best_dev_model.pt"

    Returns:
        encoder (WhisperEncoder): Our trained encoder, UNFROZEN.
        encoder_dim (int): Hidden dimension (768 for Whisper Small).
        checkpoint (dict): Full checkpoint dict (needed to load projection weights).
    """
    # Step 1: Load fresh Whisper Small to get encoder architecture
    print(f"  Loading Whisper Small encoder architecture...")
    whisper = WhisperModel.from_pretrained(STUDENT_MODEL)
    encoder = whisper.encoder
    encoder_dim = whisper.config.d_model  # 768

    # Step 2: Load our trained checkpoint
    print(f"  Loading trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained_state = checkpoint["model_state_dict"]

    epoch = checkpoint.get("epoch", "?")
    train_loss = checkpoint.get("train_loss", 0)
    dev_loss = checkpoint.get("dev_loss", 0)

    # Step 3: Replace encoder weights
    # Checkpoint keys have "encoder." prefix: "encoder.conv1.weight"
    # But encoder.state_dict() keys don't: "conv1.weight"
    # So we strip the "encoder." prefix.
    encoder_state = encoder.state_dict()
    updated = 0
    for key, value in trained_state.items():
        # Skip projection and dropout — handled separately
        if key.startswith("projection") or key.startswith("dropout"):
            continue
        # Strip "encoder." prefix
        if key.startswith("encoder."):
            encoder_key = key[len("encoder."):]
        else:
            encoder_key = key
        if encoder_key in encoder_state and encoder_state[encoder_key].shape == value.shape:
            encoder_state[encoder_key] = value
            updated += 1

    encoder.load_state_dict(encoder_state)
    print(f"  Loaded {updated}/187 encoder weights from epoch {epoch}")
    print(f"  Acoustic distillation: train_MSE={train_loss:.4f}, dev_MSE={dev_loss:.4f}")

    # Step 4: Encoder stays UNFROZEN — this is joint training
    # All 88M encoder params will receive gradients from both MSE and CTC
    encoder.train()
    frozen_count = sum(1 for p in encoder.parameters() if not p.requires_grad)
    print(f"  Encoder mode: TRAINABLE (frozen params: {frozen_count})")

    # Free decoder memory
    del whisper.decoder
    del whisper

    return encoder, encoder_dim, checkpoint


def load_projection_layer(encoder_dim, teacher_dim, acoustic_checkpoint):
    """
    Create projection layer and load trained weights from acoustic checkpoint.

    The projection layer maps student features (768-dim) to teacher space
    (1024-dim) so we can compute MSE loss between them.

    This layer was already trained during acoustic distillation. We load
    those weights as a warm start.

    Args:
        encoder_dim (int): Student encoder hidden dim (768).
        teacher_dim (int): Teacher encoder hidden dim (1024).
        acoustic_checkpoint (dict): The loaded acoustic checkpoint dict.

    Returns:
        projection (nn.Linear): Projection layer with trained weights.
        dropout (nn.Dropout): Dropout layer (same as acoustic training).
    """
    projection = nn.Linear(encoder_dim, teacher_dim)
    dropout = nn.Dropout(0.1)

    # Load projection weights from acoustic checkpoint
    trained_state = acoustic_checkpoint["model_state_dict"]
    loaded_proj = False
    for key, value in trained_state.items():
        if key == "projection.weight":
            projection.weight.data = value
            loaded_proj = True
        elif key == "projection.bias":
            projection.bias.data = value

    if loaded_proj:
        print(f"  Loaded projection weights from acoustic checkpoint")
    else:
        print(f"  WARNING: No projection weights found — initializing randomly")

    proj_params = sum(p.numel() for p in projection.parameters())
    print(f"  Projection: Linear({encoder_dim}, {teacher_dim}) — {proj_params:,} params")

    return projection, dropout


def load_ctc_head(encoder_dim, vocab_size, ctc_checkpoint_path):
    """
    Create CTC head and load trained weights from sequential CTC training.

    The CTC head maps encoder features (768-dim) to character probabilities
    (85-dim). It was already trained during sequential CTC training on the
    frozen encoder. We load those weights as a warm start.

    In joint training, the CTC head will continue to be updated, AND the
    encoder beneath it will also be updated — unlike sequential training
    where the encoder was frozen.

    Args:
        encoder_dim (int): Student encoder hidden dim (768).
        vocab_size (int): Number of vocabulary tokens (85).
        ctc_checkpoint_path (str): Path to CTC checkpoint.
            e.g., "checkpoints/semantic/hpc_ctc_best_wer.pt"

    Returns:
        ctc_head (nn.Linear): CTC prediction layer with trained weights.
    """
    ctc_head = nn.Linear(encoder_dim, vocab_size)

    if os.path.exists(ctc_checkpoint_path):
        print(f"  Loading CTC head from: {ctc_checkpoint_path}")
        ckpt = torch.load(ctc_checkpoint_path, map_location="cpu", weights_only=False)
        ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
        ctc_epoch = ckpt.get("epoch", "?")
        ctc_wer = ckpt.get("dev_wer", 0)
        print(f"  CTC head from epoch {ctc_epoch}, dev_wer={ctc_wer*100:.1f}%")
    else:
        print(f"  WARNING: CTC checkpoint not found at {ctc_checkpoint_path}")
        print(f"  Initializing CTC head randomly (not recommended)")

    ctc_params = sum(p.numel() for p in ctc_head.parameters())
    print(f"  CTC head: Linear({encoder_dim}, {vocab_size}) — {ctc_params:,} params")

    return ctc_head


def load_teacher_encoder():
    """
    Load Kid-Whisper Medium encoder as teacher — COMPLETELY FROZEN.

    The teacher provides the reference features for MSE loss.
    We never update the teacher — it's just a target to match.

    We only load the encoder (not the full model with decoder) to save
    GPU memory. The decoder is ~300M params we don't need.

    Returns:
        teacher_encoder (WhisperEncoder): Teacher encoder, frozen.
        teacher_dim (int): Hidden dimension (1024 for Whisper Medium).
    """
    print(f"  Loading teacher: {TEACHER_MODEL}")
    # Load full model to get config, then extract encoder
    teacher_model = WhisperForConditionalGeneration.from_pretrained(TEACHER_MODEL)
    teacher_encoder = teacher_model.model.encoder
    teacher_dim = teacher_model.config.d_model  # 1024

    # FREEZE teacher — never updated
    teacher_encoder.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher_encoder.parameters())
    print(f"  Teacher dim: {teacher_dim}")
    print(f"  Teacher params: {teacher_params:,} (all FROZEN)")

    # Free decoder and projection memory — we don't need them
    del teacher_model.proj_out
    del teacher_model.model.decoder
    del teacher_model

    return teacher_encoder, teacher_dim


# ─────────────────────────────────────────────
# DEV EVALUATION
# ─────────────────────────────────────────────
def evaluate_dev(encoder, projection, dropout_layer, ctc_head,
                 teacher_encoder, dev_clips, processor,
                 char_to_idx, idx_to_char, alpha, device):
    """
    Evaluate joint model on dev set — compute WER, MSE, CTC, and combined loss.

    Same as CTC evaluation but also computes MSE loss against teacher.
    Reports all three losses (MSE, CTC, combined) plus WER.

    Args:
        encoder: Student encoder (set to eval mode during evaluation).
        projection (nn.Linear): Projection layer (768 -> 1024).
        dropout_layer (nn.Dropout): Dropout (disabled in eval mode).
        ctc_head (nn.Linear): CTC prediction layer.
        teacher_encoder: Frozen teacher encoder.
        dev_clips (list[dict]): Dev clips with audio_path and transcript.
        processor: WhisperProcessor for mel spectrograms.
        char_to_idx (dict): Vocab mapping char -> index.
        idx_to_char (dict): Reverse vocab mapping index -> char.
        alpha (float): Loss weighting (alpha * MSE + (1-alpha) * CTC).
        device (str): "cuda" or "cpu".

    Returns:
        dev_wer (float): Corpus-level WER on dev set.
        avg_mse (float): Average MSE loss on dev set.
        avg_ctc (float): Average CTC loss on dev set.
        avg_combined (float): Average combined loss on dev set.
        predictions (list[str]): All predicted texts.
    """
    encoder.eval()
    ctc_head.eval()
    projection.eval()

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    all_refs = []
    all_preds = []
    mse_losses = []
    ctc_losses = []
    combined_losses = []

    with torch.no_grad():
        for clip in dev_clips:
            # Audio -> mel -> features
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)
            if real_frames == 0:
                continue
            input_features = input_features.to(device)

            # Student encoder forward
            student_features = encoder(input_features).last_hidden_state  # (1, 1500, 768)

            # Teacher encoder forward
            teacher_features = teacher_encoder(input_features).last_hidden_state  # (1, 1500, 1024)

            # ── MSE branch ──
            # Project student features to teacher dimension
            projected = projection(dropout_layer(student_features))  # (1, 1500, 1024)
            # MSE only on real frames (not padding)
            mse = torch.nn.functional.mse_loss(
                projected[:, :real_frames, :],
                teacher_features[:, :real_frames, :]
            )

            # ── CTC branch ──
            normalized_ref = normalize_text(clip["transcript"])
            target_indices = text_to_indices(normalized_ref, char_to_idx)

            if len(target_indices) == 0:
                continue
            if real_frames < len(target_indices):
                continue

            logits = ctc_head(student_features)  # (1, 1500, 85)
            log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :]
            log_probs = log_probs.permute(1, 0, 2)  # (T, 1, 85)

            targets = torch.tensor(target_indices, dtype=torch.long).to(device)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(device)

            ctc = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(ctc) or torch.isinf(ctc):
                continue

            # Combined loss
            combined = alpha * mse + (1 - alpha) * ctc

            mse_losses.append(mse.item())
            ctc_losses.append(ctc.item())
            combined_losses.append(combined.item())

            # Greedy decode for WER
            pred_text = ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)
            all_refs.append(normalized_ref)
            all_preds.append(pred_text)

    # Restore training mode
    encoder.train()
    ctc_head.train()
    projection.train()

    dev_wer = compute_corpus_wer(all_refs, all_preds) if all_refs else float("inf")
    avg_mse = sum(mse_losses) / len(mse_losses) if mse_losses else float("inf")
    avg_ctc = sum(ctc_losses) / len(ctc_losses) if ctc_losses else float("inf")
    avg_combined = sum(combined_losses) / len(combined_losses) if combined_losses else float("inf")

    return dev_wer, avg_mse, avg_ctc, avg_combined, all_preds


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    """
    Main joint training function.

    TRAINING FLOW:
        1. Load vocab (85 tokens)
        2. Load train/dev clips
        3. Load student encoder (UNFROZEN, warm start from acoustic checkpoint)
        4. Load projection layer (warm start from acoustic checkpoint)
        5. Load CTC head (warm start from sequential CTC checkpoint)
        6. Load teacher encoder (FROZEN)
        7. Create optimizer (AdamW over encoder + projection + CTC head)
        8. For each epoch:
           a. Shuffle training clips
           b. For each clip:
              - Audio -> mel -> student encoder (WITH gradients)
              - Audio -> mel -> teacher encoder (NO gradients)
              - MSE loss: projected student features vs teacher features
              - CTC loss: CTC head logits vs ground truth text
              - Combined: alpha * MSE + (1-alpha) * CTC
              - Backprop through encoder + projection + CTC head
           c. Evaluate on dev set (WER + all losses)
           d. Checkpoints + early stopping on dev WER
        9. Save final summary

    WHAT GETS UPDATED:
        ✅ Student encoder: ~88M params (12 transformer layers)
        ✅ Projection layer: ~787K params (Linear 768->1024)
        ✅ CTC head: ~65K params (Linear 768->85)
        Total: ~89M trainable parameters

        ❌ Teacher encoder: ~300M params (FROZEN, never updated)
    """
    parser = argparse.ArgumentParser(description="Joint training: MSE + CTC simultaneous optimization")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on dev WER")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (3e-5 standard for encoder fine-tuning)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps before decay")
    parser.add_argument("--alpha", type=float, default=0.5, help="Loss weight: alpha*MSE + (1-alpha)*CTC. 0.5 = equal weight")
    parser.add_argument("--max_clips", type=int, default=None, help="Limit training clips (for quick testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None, help="Resume from joint checkpoint (e.g., checkpoints/joint/joint_epoch_5.pt)")
    args = parser.parse_args()

    # ── Reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"{'='*60}")
    print(f"  JOINT TRAINING: MSE + CTC")
    print(f"  alpha={args.alpha} (MSE weight={args.alpha}, CTC weight={1-args.alpha})")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── Step 1: Load vocabulary ──
    print("Step 1: Loading vocabulary...")
    char_to_idx, idx_to_char, vocab_size = load_vocab(VOCAB_PATH)
    print(f"  Vocab size: {vocab_size} tokens\n")

    # ── Step 2: Load data ──
    print("Step 2: Loading data...")
    train_clips = load_clips(TRAIN_CSV, max_clips=args.max_clips)
    dev_clips = load_clips(DEV_CSV)  # Always use full dev set for evaluation
    print(f"  Train: {len(train_clips)} clips")
    print(f"  Dev:   {len(dev_clips)} clips")

    lang_counts = {}
    for c in train_clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    print()

    # ── Step 3: Load student encoder (UNFROZEN) ──
    print("Step 3: Loading student encoder (TRAINABLE)...")
    encoder, encoder_dim, acoustic_ckpt = load_student_encoder(ACOUSTIC_CHECKPOINT)
    encoder.to(DEVICE)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder params: {encoder_params:,} (all TRAINABLE)\n")

    # ── Step 4: Load projection layer ──
    print("Step 4: Loading projection layer...")
    # Teacher dim determined from teacher model config
    teacher_dim = 1024  # Kid-Whisper Medium hidden dim
    projection, dropout_layer = load_projection_layer(encoder_dim, teacher_dim, acoustic_ckpt)
    projection.to(DEVICE)
    dropout_layer.to(DEVICE)
    print()

    # ── Step 5: Load CTC head ──
    print("Step 5: Loading CTC head...")
    ctc_head = load_ctc_head(encoder_dim, vocab_size, CTC_CHECKPOINT)
    ctc_head.to(DEVICE)
    print()

    # ── Step 6: Load teacher encoder (FROZEN) ──
    print("Step 6: Loading teacher encoder (FROZEN)...")
    teacher_encoder, teacher_dim_actual = load_teacher_encoder()
    teacher_encoder.to(DEVICE)
    assert teacher_dim == teacher_dim_actual, \
        f"Teacher dim mismatch: expected {teacher_dim}, got {teacher_dim_actual}"
    print()

    # Free the acoustic checkpoint dict from memory
    del acoustic_ckpt

    # ── Step 7: Load processor ──
    print("Step 7: Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)
    print()

    # ── Step 8: Setup optimizer and scheduler ──
    # All trainable parameters: encoder + projection + CTC head
    # We use a SINGLE optimizer for all of them so gradients from both
    # MSE and CTC flow correctly through the shared encoder.
    #
    # LR = 3e-5 (same as acoustic branch, because we're fine-tuning the encoder)
    # This is much lower than CTC-only training (1e-3) because the encoder
    # has 88M params — large models need smaller learning rates.
    print("Step 8: Setting up optimizer...")
    all_trainable_params = (
        list(encoder.parameters()) +
        list(projection.parameters()) +
        list(dropout_layer.parameters()) +
        list(ctc_head.parameters())
    )
    total_trainable = sum(p.numel() for p in all_trainable_params)
    print(f"  Total trainable parameters: {total_trainable:,}")

    optimizer = torch.optim.AdamW(all_trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_clips) * args.epochs

    def lr_lambda(step):
        """Linear warmup then linear decay — standard transformer schedule."""
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        else:
            remaining = total_steps - step
            total_decay = total_steps - args.warmup_steps
            return max(0.0, remaining / max(1, total_decay))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions (both PyTorch built-in)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    # MSE loss: torch.nn.functional.mse_loss (used inline, no need to instantiate)

    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=0.01)")
    print(f"  Scheduler: linear warmup ({args.warmup_steps} steps) -> linear decay")
    print(f"  Total training steps: {total_steps:,}")
    print()

    # ── Resume from checkpoint if specified ──
    start_epoch = 1
    best_dev_wer = float("inf")
    best_dev_loss = float("inf")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)

        # Load all model states
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        projection.load_state_dict(ckpt["projection_state_dict"])
        ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_dev_wer = ckpt.get("dev_wer", float("inf"))
        best_dev_loss = ckpt.get("dev_combined_loss", float("inf"))

        # Fast-forward scheduler to correct step
        steps_done = ckpt["epoch"] * len(train_clips)
        for _ in range(steps_done):
            scheduler.step()

        print(f"  Resumed from epoch {ckpt['epoch']}, starting at epoch {start_epoch}")
        print(f"  Previous dev WER: {ckpt.get('dev_wer', 0)*100:.1f}%")
        print(f"  Previous dev combined loss: {ckpt.get('dev_combined_loss', 0):.4f}")
        print()

    # ── Step 9: Verify pipeline with 1 clip ──
    print("Step 9: Verifying pipeline with 1 clip...")
    test_clip = train_clips[0]
    input_features, real_frames = prepare_audio(test_clip["audio_path"], processor)
    input_features = input_features.to(DEVICE)

    print(f"  Audio: {test_clip['audio_path'].split('/')[-1]}")
    print(f"  Language: {test_clip['language']}")
    print(f"  Transcript: '{test_clip['transcript'][:50]}...'")

    # Student forward (with gradients for verification)
    student_features = encoder(input_features).last_hidden_state  # (1, 1500, 768)
    print(f"  Student features: {student_features.shape}")

    # Teacher forward (no gradients)
    with torch.no_grad():
        teacher_features = teacher_encoder(input_features).last_hidden_state  # (1, 1500, 1024)
    print(f"  Teacher features: {teacher_features.shape}")

    # MSE branch
    projected = projection(dropout_layer(student_features))  # (1, 1500, 1024)
    mse = torch.nn.functional.mse_loss(
        projected[:, :real_frames, :],
        teacher_features[:, :real_frames, :]
    )
    print(f"  MSE loss: {mse.item():.4f}")

    # CTC branch
    logits = ctc_head(student_features)  # (1, 1500, 85)
    normalized_ref = normalize_text(test_clip["transcript"])
    target_indices = text_to_indices(normalized_ref, char_to_idx)

    log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :].permute(1, 0, 2)
    targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
    input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)
    ctc = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    print(f"  CTC loss: {ctc.item():.4f}")

    # Combined
    combined = args.alpha * mse + (1 - args.alpha) * ctc
    print(f"  Combined loss: {combined.item():.4f} (alpha={args.alpha})")

    # Greedy decode
    pred_text = ctc_greedy_decode(logits[0, :real_frames, :].detach(), idx_to_char)
    print(f"  Decoded: '{pred_text[:50]}...'")

    # Verify gradients flow to encoder
    combined.backward()
    encoder_grad = next(encoder.parameters()).grad
    if encoder_grad is not None:
        print(f"  Encoder gradient norm: {encoder_grad.norm().item():.6f} (gradients flowing!)")
    else:
        print(f"  WARNING: No gradients flowing to encoder!")
    optimizer.zero_grad()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  GPU peak memory: {peak_mem:.0f} MB")

    print("  Pipeline verified\n")

    # ── Step 10: Training loop ──
    print(f"{'='*60}")
    print(f"  TRAINING: {args.epochs} epochs, lr={args.lr}")
    print(f"  alpha={args.alpha}: loss = {args.alpha}*MSE + {1-args.alpha}*CTC")
    print(f"  Trainable: encoder ({encoder_params:,}) + projection + CTC head = {total_trainable:,} params")
    print(f"  Patience: {args.patience} epochs on dev WER")
    print(f"{'='*60}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    patience_counter = 0
    global_step = 0
    epoch_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        projection.train()
        ctc_head.train()
        teacher_encoder.eval()  # Teacher always in eval mode

        epoch_start = time.time()
        epoch_mse = []
        epoch_ctc = []
        epoch_combined = []
        skipped = 0

        random.shuffle(train_clips)

        for clip in tqdm(train_clips, desc=f"Epoch {epoch}/{args.epochs}", unit="clip"):

            # ══════════════════════════════════════════════════════
            # STEP 1: Prepare input — audio to mel spectrogram
            # ══════════════════════════════════════════════════════
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)
            if real_frames == 0:
                skipped += 1
                continue
            input_features = input_features.to(DEVICE)

            # Prepare CTC target text
            normalized_ref = normalize_text(clip["transcript"])
            target_indices = text_to_indices(normalized_ref, char_to_idx)

            if len(target_indices) == 0:
                skipped += 1
                continue

            # CTC requires: input_length >= target_length
            if real_frames < len(target_indices):
                skipped += 1
                continue

            # ══════════════════════════════════════════════════════
            # STEP 2: Student encoder forward — WITH gradients
            # Unlike CTC-only training where encoder was frozen,
            # here the encoder receives gradients from both losses.
            # ══════════════════════════════════════════════════════
            student_features = encoder(input_features).last_hidden_state  # (1, 1500, 768)

            # ══════════════════════════════════════════════════════
            # STEP 3: Teacher encoder forward — NO gradients
            # Teacher is frozen, we only use it as a reference.
            # torch.no_grad() saves memory by not building grad graph.
            # ══════════════════════════════════════════════════════
            with torch.no_grad():
                teacher_features = teacher_encoder(input_features).last_hidden_state  # (1, 1500, 1024)

            # ══════════════════════════════════════════════════════
            # STEP 4: MSE loss (acoustic branch)
            #
            # Project student features (768) to teacher space (1024),
            # then compute MSE only on REAL frames (not padding).
            #
            # MSE = mean((projected - teacher)^2) across all real frames.
            # Gradients flow: MSE -> projection -> dropout -> encoder
            # ══════════════════════════════════════════════════════
            projected = projection(dropout_layer(student_features))  # (1, 1500, 1024)
            mse_loss = torch.nn.functional.mse_loss(
                projected[:, :real_frames, :],
                teacher_features[:, :real_frames, :]
            )

            # ══════════════════════════════════════════════════════
            # STEP 5: CTC loss (semantic branch)
            #
            # CTC head predicts character probabilities per frame.
            # CTC loss aligns frames to characters automatically.
            #
            # Gradients flow: CTC -> ctc_head -> encoder
            # (encoder gets gradients from BOTH MSE and CTC!)
            # ══════════════════════════════════════════════════════
            logits = ctc_head(student_features)  # (1, 1500, 85)
            log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :]
            log_probs = log_probs.permute(1, 0, 2)  # (T, 1, 85) — CTC expects time-first

            targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

            ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                skipped += 1
                continue

            # ══════════════════════════════════════════════════════
            # STEP 6: Combine losses
            #
            # total = alpha * MSE + (1-alpha) * CTC
            #
            # With alpha=0.5:
            #   - 50% weight on matching teacher features (acoustic)
            #   - 50% weight on predicting text (semantic)
            #
            # The encoder receives gradients from BOTH losses.
            # This is the key advantage over sequential training.
            # ══════════════════════════════════════════════════════
            total_loss = args.alpha * mse_loss + (1 - args.alpha) * ctc_loss

            # ══════════════════════════════════════════════════════
            # STEP 7: Backward pass + update ALL trainable params
            #
            # Gradients flow to:
            #   - CTC head (from CTC loss)
            #   - Projection layer (from MSE loss)
            #   - Encoder (from BOTH MSE and CTC losses!)
            #
            # Grad clipping prevents exploding gradients — important
            # when training large models like the encoder.
            # max_norm=1.0 is standard (Wav2Vec 2.0, Whisper fine-tuning).
            # ══════════════════════════════════════════════════════
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_mse.append(mse_loss.item())
            epoch_ctc.append(ctc_loss.item())
            epoch_combined.append(total_loss.item())

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        avg_mse = sum(epoch_mse) / len(epoch_mse) if epoch_mse else float("inf")
        avg_ctc = sum(epoch_ctc) / len(epoch_ctc) if epoch_ctc else float("inf")
        avg_combined = sum(epoch_combined) / len(epoch_combined) if epoch_combined else float("inf")

        # ── Dev evaluation ──
        print(f"\n  Evaluating on dev set ({len(dev_clips)} clips)...")
        dev_wer, dev_mse, dev_ctc, dev_combined, dev_preds = evaluate_dev(
            encoder, projection, dropout_layer, ctc_head,
            teacher_encoder, dev_clips, processor,
            char_to_idx, idx_to_char, args.alpha, DEVICE
        )

        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "train_mse": avg_mse,
            "train_ctc": avg_ctc,
            "train_combined": avg_combined,
            "dev_mse": dev_mse,
            "dev_ctc": dev_ctc,
            "dev_combined": dev_combined,
            "dev_wer": dev_wer,
            "time": epoch_time,
            "skipped": skipped,
            "lr": current_lr,
        }
        epoch_history.append(epoch_record)

        print(f"\n  {'─'*65}")
        print(f"  Epoch {epoch}/{args.epochs} | time: {epoch_time:.0f}s ({epoch_time/60:.1f} min)")
        print(f"  Train — MSE: {avg_mse:.4f} | CTC: {avg_ctc:.4f} | Combined: {avg_combined:.4f}")
        print(f"  Dev   — MSE: {dev_mse:.4f} | CTC: {dev_ctc:.4f} | Combined: {dev_combined:.4f}")
        print(f"  Dev WER: {dev_wer*100:.1f}%")
        print(f"  LR: {current_lr:.2e} | Skipped: {skipped} clips")

        # Show 3 sample predictions
        sample_count = min(3, len(dev_preds), len(dev_clips))
        if sample_count > 0:
            print(f"\n  Sample predictions:")
            for i in range(sample_count):
                ref = normalize_text(dev_clips[i]["transcript"])
                pred = dev_preds[i] if i < len(dev_preds) else ""
                print(f"    GT:   '{ref[:60]}{'...' if len(ref) > 60 else ''}'")
                print(f"    Pred: '{pred[:60]}{'...' if len(pred) > 60 else ''}'")
                print()

        print(f"  {'─'*65}\n")

        # ── Save checkpoints ──
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_mse": avg_mse,
            "train_ctc": avg_ctc,
            "train_combined": avg_combined,
            "dev_mse": dev_mse,
            "dev_ctc": dev_ctc,
            "dev_combined_loss": dev_combined,
            "dev_wer": dev_wer,
            "alpha": args.alpha,
            "vocab_size": vocab_size,
            "encoder_dim": encoder_dim,
            "teacher_dim": teacher_dim,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"joint_epoch_{epoch}.pt"))

        # Track best dev WER (primary metric)
        improved = False
        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "joint_best_wer.pt"))
            print(f"  >>> New best dev WER: {best_dev_wer*100:.1f}%")
            improved = True

        if dev_combined < best_dev_loss:
            best_dev_loss = dev_combined
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "joint_best_loss.pt"))
            if not improved:
                print(f"  >>> New best dev combined loss: {best_dev_loss:.4f}")
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Training complete ──
    print(f"\n{'='*65}")
    print(f"  JOINT TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"\n  Epoch-by-epoch summary:")
    print(f"  {'Ep':>3} {'TrMSE':>8} {'TrCTC':>8} {'TrComb':>8} {'DvMSE':>8} {'DvCTC':>8} {'DvComb':>8} {'DvWER':>8} {'LR':>10} {'Time':>6}")
    print(f"  {'─'*90}")
    for r in epoch_history:
        print(
            f"  {r['epoch']:>3} "
            f"{r['train_mse']:>8.4f} {r['train_ctc']:>8.4f} {r['train_combined']:>8.4f} "
            f"{r['dev_mse']:>8.4f} {r['dev_ctc']:>8.4f} {r['dev_combined']:>8.4f} "
            f"{r['dev_wer']*100:>7.1f}% {r['lr']:>10.2e} {r['time']:>5.0f}s"
        )

    print(f"\n  Best dev WER:           {best_dev_wer*100:.1f}%")
    print(f"  Best dev combined loss: {best_dev_loss:.4f}")
    print(f"  Checkpoints:            {CHECKPOINT_DIR}/")
    print(f"\n  Compare with sequential approach:")
    print(f"    Sequential CTC (frozen encoder): 81.9% WER")
    print(f"    Joint training (unfrozen encoder): {best_dev_wer*100:.1f}% WER")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
