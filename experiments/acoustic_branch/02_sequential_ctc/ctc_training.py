"""
Step 2: CTC Head Training (Sequential Approach)
================================================

WHAT THIS SCRIPT DOES:
    Takes our HPC-trained encoder (from acoustic distillation) and trains
    a CTC head on top of it to predict text. The encoder is FROZEN — only
    the CTC head (one linear layer) is trained.

WHY THIS WORKS (unlike Whisper decoder):
    The Whisper decoder failed because it was trained on ORIGINAL encoder
    features, but our encoder now produces DIFFERENT features after acoustic
    distillation. The decoder never learned the new feature space → gibberish.

    The CTC head is trained FROM SCRATCH on the new encoder features.
    It learns to map these new features → text directly. No mismatch.

ARCHITECTURE:
    ┌──────────────────────────────────────────────────┐
    │  Audio (.wav file)                               │
    │       ↓                                          │
    │  Whisper Processor (mel spectrogram)             │
    │       ↓                                          │
    │  Our Trained Encoder (FROZEN, 12 transformer     │
    │  layers from acoustic distillation, epoch 16)    │
    │       ↓                                          │
    │  Encoder features: (batch, 1500, 768)            │
    │       ↓                                          │
    │  CTC Head: Linear(768, 85) ← TRAINABLE          │
    │       ↓                                          │
    │  Logits: (batch, 1500, 85) — one char prob       │
    │       per frame per vocab token                  │
    │       ↓                                          │
    │  torch.nn.CTCLoss (PyTorch built-in)             │
    │       ↓                                          │
    │  Backprop updates CTC head ONLY                  │
    └──────────────────────────────────────────────────┘

    85 = vocab size (3 special + 22 English + 60 Devanagari)
    from ASER-Dataset/vocab.json (built by build_vocab.py)

CTC LOSS — how it works:
    Standard loss for speech recognition. Published in:
    "Connectionist Temporal Classification" (Graves et al., ICML 2006)

    Problem: We have 1500 encoder frames but the target text might be
    20 characters long. We DON'T know which frame maps to which character.

    CTC solves this by considering ALL possible alignments:
        Frames:  [f1][f2][f3][f4][f5][f6][f7][f8]...
        Align 1:  ह   ह   -   ल   ल   ो   -   -    → "हलो"
        Align 2:  -   ह   ह   -   ल   ो   ो   -    → "हलो"
        Align 3:  ह   -   -   ल   -   ो   -   -    → "हलो"
        ...
    CTC sums probabilities of ALL valid alignments → maximizes total.
    The blank token (-) means "no output at this frame."

    PyTorch: torch.nn.CTCLoss(blank=0, zero_infinity=True)
    - blank=0: index 0 in vocab is the blank token (matches our vocab.json)
    - zero_infinity=True: if loss becomes inf (very long target), set to 0
      This prevents NaN gradients from crashing training.

    Used by: Wav2Vec 2.0, DeepSpeech, CARE paper, all CTC-based ASR.

WHAT GETS UPDATED:
    ✅ CTC head (Linear 768 → 85) — weight (768×85) + bias (85)
    ❌ Encoder — FROZEN (already trained in acoustic distillation)
    ❌ Whisper decoder — not used at all in this pipeline

EVALUATION:
    After each epoch, we run greedy CTC decoding on dev set:
    1. Get logits from CTC head → (1500, 85)
    2. argmax per frame → [0, 0, 5, 5, 0, 12, 12, 0, ...] (character indices)
    3. Collapse consecutive duplicates → [0, 5, 0, 12, 0, ...]
    4. Remove blanks (index 0) → [5, 12, ...]
    5. Map indices to characters → "ek"
    6. Compute WER using our standard utils/wer.py (corpus-level)

Prerequisites:
    - HPC checkpoint: checkpoints/acoustic/hpc_best_dev_model.pt
    - Vocabulary: ASER-Dataset/vocab.json (from build_vocab.py)
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv

Usage:
    # Quick test (50 clips, 2 epochs)
    python scripts/semantic/ctc_training.py --max_clips 50 --epochs 2

    # Full training
    python scripts/semantic/ctc_training.py --epochs 30 --lr 1e-3 --patience 5

References:
    - CTC Loss: Graves et al., "Connectionist Temporal Classification", ICML 2006
    - Wav2Vec 2.0 CTC: Baevski et al., NeurIPS 2020
    - CARE paper: Rumberg et al., 2022 (dual-branch with CTC)
    - PyTorch CTCLoss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
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

from transformers import WhisperModel, WhisperProcessor

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
ENCODER_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "acoustic", "hpc_best_dev_model.pt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "semantic")

STUDENT_MODEL = "openai/whisper-small"  # Base model to load encoder architecture
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds


# ─────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────
def load_vocab(vocab_path):
    """
    Load character vocabulary from vocab.json.

    Args:
        vocab_path (str): Path to vocab.json file.
            Created by build_vocab.py.

    Returns:
        char_to_idx (dict): Maps character → index.
            Example: {"<blank>": 0, "<space>": 1, "a": 3, "अ": 27, ...}
        idx_to_char (dict): Maps index → character (reverse lookup for decoding).
            Example: {0: "<blank>", 1: "<space>", 3: "a", 27: "अ", ...}
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
            Example: "राधा के पास"
        char_to_idx (dict): Vocabulary mapping from load_vocab().

    Returns:
        list[int]: List of character indices.
            Example: [60, 69, 52, 69, 1, 36, 74, 1, 54, 69, 67]
            Where 1 = <space> (word boundary between words)

    Note:
        <blank> (index 0) is NEVER in the target.
        <blank> is only used by CTC internally during training to
        represent "no output at this frame." The target text only
        contains actual characters and spaces.
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

    HOW IT WORKS:
        1. For each frame, pick the token with highest probability (argmax)
        2. Collapse consecutive identical tokens into one
        3. Remove all blank tokens (index 0)
        4. Map remaining indices back to characters
        5. Join characters to form text

    Example:
        Logits: (1500, 85) — 1500 frames, 85 possible characters
        Step 1 (argmax):    [0, 0, 60, 60, 60, 0, 69, 69, 0, 0, 52, 52, ...]
        Step 2 (collapse):  [0, 60, 0, 69, 0, 52, ...]
        Step 3 (rm blank):  [60, 69, 52, ...]
        Step 4 (to chars):  ['र', 'ा', 'ध', ...]
        Step 5 (join):      "राध..."

    Args:
        logits (Tensor): CTC output, shape (T, vocab_size).
            T = number of encoder frames (1500 for full 30s audio).
            vocab_size = 85 for our dataset.
            Each row is a probability distribution over characters.
        idx_to_char (dict): Reverse vocab mapping {index: character}.

    Returns:
        str: Decoded text string.

    Reference:
        This is the standard greedy decoding. For better results, beam
        search with a language model can be used (not implemented here,
        we start simple and add complexity only if needed).
    """
    # Step 1: argmax per frame → predicted token index
    predicted_ids = logits.argmax(dim=-1).tolist()  # list of ints, length T

    # Step 2 + 3: Collapse consecutive duplicates and remove blanks
    decoded_ids = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id:  # Only keep if different from previous (collapse)
            if idx != 0:    # Skip blanks (index 0)
                decoded_ids.append(idx)
        prev_id = idx

    # Step 4 + 5: Map indices to characters and join
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

    Unlike acoustic branch (which only needed audio paths),
    CTC training needs BOTH audio AND transcript because we're
    training the model to predict text.

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
                continue  # Skip clips >30s (Whisper mel limit)
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
    Same function as acoustic branch — see step2_acoustic_distillation.py for details.

    Args:
        audio_path (str): Path to .wav file.
        processor (WhisperProcessor): Whisper's feature extractor.

    Returns:
        input_features (Tensor): Mel spectrogram, shape (1, 80, 3000).
        real_encoder_frames (int): Number of non-padding encoder frames (1 to 1500).
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
def load_frozen_encoder(checkpoint_path):
    """
    Load our HPC-trained encoder from acoustic distillation checkpoint.

    IMPORTANT: The encoder is COMPLETELY FROZEN — no gradients, no updates.
    It was already trained in the acoustic branch (MSE loss against Kid-Whisper
    teacher). Now we only train the CTC head on top of its frozen output.

    HOW IT WORKS:
        1. Load fresh Whisper Small encoder (to get the architecture)
        2. Load our HPC checkpoint (187 encoder keys)
        3. Replace encoder weights with trained weights
        4. Freeze all encoder parameters

    Args:
        checkpoint_path (str): Path to HPC checkpoint.
            e.g., "checkpoints/acoustic/hpc_best_dev_model.pt"

    Returns:
        encoder (WhisperEncoder): Our trained encoder, frozen.
        encoder_dim (int): Hidden dimension (768 for Whisper Small).
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
    # Our checkpoint keys: "encoder.conv1.weight", "encoder.layers.0.self_attn.k_proj.weight", etc.
    # But encoder.state_dict() keys: "conv1.weight", "layers.0.self_attn.k_proj.weight", etc.
    # So we need to strip the "encoder." prefix from checkpoint keys.
    encoder_state = encoder.state_dict()
    updated = 0
    for key, value in trained_state.items():
        # Skip projection layer (768→1024, not needed for CTC — was only for MSE with teacher)
        # Skip dropout (not a weight)
        if key.startswith("projection") or key.startswith("dropout"):
            continue
        # Strip "encoder." prefix: "encoder.conv1.weight" → "conv1.weight"
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

    # Step 4: FREEZE encoder — we never update it in CTC training
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Free decoder memory — we don't need it
    del whisper.decoder
    del whisper

    return encoder, encoder_dim


# ─────────────────────────────────────────────
# DEV EVALUATION
# ─────────────────────────────────────────────
def evaluate_dev(encoder, ctc_head, dev_clips, processor, char_to_idx, idx_to_char, device):
    """
    Evaluate CTC model on dev set — compute WER using greedy decoding.

    STEPS FOR EACH CLIP:
        1. Audio → mel spectrogram → encoder features (1500, 768)
        2. CTC head → logits (1500, 85)
        3. Greedy decode → predicted text
        4. Collect all predictions and references
    After all clips:
        5. Compute corpus-level WER using our standard utils/wer.py

    Args:
        encoder: Frozen trained encoder.
        ctc_head (nn.Linear): CTC prediction layer.
        dev_clips (list[dict]): Dev clips with audio_path and transcript.
        processor: WhisperProcessor for mel spectrograms.
        char_to_idx (dict): Vocab mapping char → index.
        idx_to_char (dict): Reverse vocab mapping index → char.
        device (str): "cuda" or "cpu".

    Returns:
        wer (float): Corpus-level WER on dev set.
        avg_ctc_loss (float): Average CTC loss on dev set.
        predictions (list[str]): All predicted texts (for inspection).
    """
    ctc_head.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    all_refs = []
    all_preds = []
    losses = []

    with torch.no_grad():
        for clip in dev_clips:
            # Step 1: Audio → mel → encoder features
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)
            if real_frames == 0:
                continue
            input_features = input_features.to(device)

            # Step 2: Encoder (frozen) → features → CTC head → logits
            encoder_output = encoder(input_features).last_hidden_state  # (1, 1500, 768)
            logits = ctc_head(encoder_output)  # (1, 1500, vocab_size)

            # Step 3: Compute CTC loss for monitoring
            # Normalize transcript same way as vocab was built
            normalized_ref = normalize_text(clip["transcript"])
            target_indices = text_to_indices(normalized_ref, char_to_idx)

            if len(target_indices) == 0:
                continue

            # CTC loss expects: log_probs (T, B, C), targets (sum_of_target_lengths,)
            # T = time steps, B = batch, C = vocab size
            log_probs = logits.log_softmax(dim=-1)          # (1, 1500, 85)
            log_probs = log_probs[:, :real_frames, :]       # Only real frames
            log_probs = log_probs.permute(1, 0, 2)          # (T, 1, 85) — CTC expects time-first

            targets = torch.tensor(target_indices, dtype=torch.long)  # (target_len,)
            input_lengths = torch.tensor([real_frames], dtype=torch.long)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long)

            # CTC requires: input_length >= target_length (need enough frames for all chars)
            if real_frames < len(target_indices):
                continue

            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            if not torch.isnan(loss) and not torch.isinf(loss):
                losses.append(loss.item())

            # Step 4: Greedy decode for WER
            # Use only real frames for decoding (padding frames would add garbage)
            pred_text = ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)

            all_refs.append(normalized_ref)
            all_preds.append(pred_text)

    ctc_head.train()

    # Step 5: Corpus-level WER
    wer = compute_corpus_wer(all_refs, all_preds) if all_refs else float("inf")
    avg_loss = sum(losses) / len(losses) if losses else float("inf")

    return wer, avg_loss, all_preds


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    """
    Main training function for CTC head.

    TRAINING FLOW:
        1. Load vocab (85 tokens from vocab.json)
        2. Load train/dev clips (audio + transcripts)
        3. Load frozen encoder (from HPC acoustic distillation)
        4. Create CTC head: nn.Linear(768, 85) — single linear layer
        5. For each epoch:
           a. Shuffle training clips
           b. For each clip:
              - Audio → mel → frozen encoder → features (1500, 768)
              - CTC head → logits (1500, 85)
              - CTC loss (logits vs target text)
              - Backprop through CTC head ONLY (encoder frozen)
           c. Evaluate WER on dev set (greedy decode)
           d. Save checkpoints, check early stopping
        6. Save final summary

    WHAT GETS UPDATED:
        ✅ CTC head: Linear(768, 85) — weight (768×85=65,280) + bias (85) = 65,365 parameters
        ❌ Encoder (frozen, ~242M params) — NOT updated
    """
    parser = argparse.ArgumentParser(description="CTC head training on frozen encoder")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (stop if dev WER doesn't improve)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (higher than acoustic branch because CTC head is small)")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--max_clips", type=int, default=None, help="Limit clips for quick testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from CTC checkpoint (e.g., checkpoints/semantic/ctc_epoch_17.pt)")
    args = parser.parse_args()

    # ── Reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Step 1: Load vocabulary ──
    print("Loading vocabulary...")
    char_to_idx, idx_to_char, vocab_size = load_vocab(VOCAB_PATH)
    print(f"  Vocab size: {vocab_size} tokens")

    # ── Step 2: Load data ──
    print("\nLoading data...")
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
    print("\nLoading frozen encoder (from acoustic distillation)...")
    encoder, encoder_dim = load_frozen_encoder(ENCODER_CHECKPOINT)
    encoder.to(DEVICE)
    print(f"  Encoder dim: {encoder_dim}")
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,} (all frozen)")

    # ── Step 4: Load processor ──
    print("\nLoading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    # ── Step 5: Create CTC head ──
    # Just one linear layer: maps encoder features (768) to character probabilities (85)
    # This is all we're training — 65,365 parameters total
    print("\nCreating CTC head...")
    ctc_head = nn.Linear(encoder_dim, vocab_size).to(DEVICE)
    trainable_params = sum(p.numel() for p in ctc_head.parameters())
    print(f"  CTC head: Linear({encoder_dim}, {vocab_size})")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Step 6: Setup optimizer and scheduler ──
    # AdamW with higher LR than acoustic branch:
    #   Acoustic branch: LR=3e-5 (fine-tuning 242M encoder params — need small steps)
    #   CTC head: LR=1e-3 (training 65K params from scratch — can use bigger steps)
    # This LR is standard for CTC head training in Wav2Vec 2.0 fine-tuning.
    optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_clips) * args.epochs

    def lr_lambda(step):
        """Linear warmup then linear decay — same schedule as acoustic branch."""
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        else:
            remaining = total_steps - step
            total_decay = total_steps - args.warmup_steps
            return max(0.0, remaining / max(1, total_decay))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # CTC loss function
    # blank=0: vocab index 0 is the blank token (from vocab.json)
    # zero_infinity=True: prevents NaN when loss is inf (very long transcripts)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # ── Resume from checkpoint if specified ──
    start_epoch = 1
    best_dev_wer = float("inf")
    best_dev_loss = float("inf")

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dev_wer = ckpt.get("dev_wer", float("inf"))
        best_dev_loss = ckpt.get("dev_loss", float("inf"))
        # Fast-forward scheduler to correct step
        steps_done = ckpt["epoch"] * len(train_clips)
        for _ in range(steps_done):
            scheduler.step()
        print(f"  Resumed from epoch {ckpt['epoch']}, starting at epoch {start_epoch}")
        print(f"  Previous train loss: {ckpt['train_loss']:.4f}")
        print(f"  Previous dev loss: {ckpt['dev_loss']:.4f}")
        print(f"  Previous dev WER: {ckpt['dev_wer']*100:.1f}%")

    # ── Step 7: Verify pipeline with 1 clip ──
    print("\nVerifying pipeline with 1 clip...")
    test_clip = train_clips[0]
    input_features, real_frames = prepare_audio(test_clip["audio_path"], processor)
    input_features = input_features.to(DEVICE)

    print(f"  Audio: {test_clip['audio_path'].split('/')[-1]}")
    print(f"  Language: {test_clip['language']}")
    print(f"  Transcript: '{test_clip['transcript'][:50]}...'")

    # Forward pass through frozen encoder
    with torch.no_grad():
        encoder_output = encoder(input_features).last_hidden_state  # (1, 1500, 768)
    print(f"  Encoder output: {encoder_output.shape}")

    # CTC head
    logits = ctc_head(encoder_output)  # (1, 1500, 85)
    print(f"  CTC logits: {logits.shape}")

    # Prepare target
    normalized_ref = normalize_text(test_clip["transcript"])
    target_indices = text_to_indices(normalized_ref, char_to_idx)
    print(f"  Target text: '{normalized_ref[:50]}...'")
    print(f"  Target indices: {target_indices[:15]}... (length {len(target_indices)})")

    # CTC loss
    log_probs = logits.log_softmax(dim=-1)[:, :real_frames, :].permute(1, 0, 2)
    targets = torch.tensor(target_indices, dtype=torch.long)
    input_lengths = torch.tensor([real_frames], dtype=torch.long)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long)

    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    print(f"  CTC loss: {loss.item():.4f}")

    # Greedy decode (will be garbage before training — just verifying it runs)
    pred_text = ctc_greedy_decode(logits[0, :real_frames, :].detach(), idx_to_char)
    print(f"  Decoded (before training): '{pred_text[:50]}...'")
    print("  Pipeline verified ✓")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  GPU peak memory: {peak_mem:.0f} MB")

    # ── Step 8: Training loop ──
    print(f"\n{'='*60}")
    print(f"  TRAINING: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
    print(f"  Training CTC head ({trainable_params:,} params) on frozen encoder")
    print(f"{'='*60}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not args.resume:
        best_dev_wer = float("inf")
        best_dev_loss = float("inf")
    patience_counter = 0
    global_step = 0
    epoch_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        ctc_head.train()
        epoch_start = time.time()
        epoch_losses = []
        skipped = 0

        random.shuffle(train_clips)

        for clip in tqdm(train_clips, desc=f"Epoch {epoch}/{args.epochs}", unit="clip"):

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 1: Audio → mel spectrogram
            # ══════════════════════════════════════════════════════
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)
            if real_frames == 0:
                skipped += 1
                continue
            input_features = input_features.to(DEVICE)

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 2: Frozen encoder → features
            # No gradients for encoder — it's frozen, we never update it
            # ══════════════════════════════════════════════════════
            with torch.no_grad():
                encoder_output = encoder(input_features).last_hidden_state  # (1, 1500, 768)

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 3: CTC head → logits
            # This IS tracked for gradients — CTC head will be updated
            # ══════════════════════════════════════════════════════
            logits = ctc_head(encoder_output)  # (1, 1500, vocab_size)

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 4: Prepare target + compute CTC loss
            #
            # CTC loss needs:
            #   - log_probs: shape (T, B, C) — T=time, B=batch, C=vocab
            #   - targets: 1D tensor of target character indices
            #   - input_lengths: how many encoder frames (real, not padding)
            #   - target_lengths: how many characters in target text
            #
            # CTC internally considers ALL possible alignments of
            # T frames to target_length characters, sums their
            # probabilities, and computes -log(total_probability).
            # ══════════════════════════════════════════════════════
            normalized_ref = normalize_text(clip["transcript"])
            target_indices = text_to_indices(normalized_ref, char_to_idx)

            if len(target_indices) == 0:
                skipped += 1
                continue

            # CTC requires: input_length >= target_length
            # (need at least as many frames as characters to align)
            if real_frames < len(target_indices):
                skipped += 1
                continue

            log_probs = logits.log_softmax(dim=-1)          # (1, 1500, 85)
            log_probs = log_probs[:, :real_frames, :]       # Only real frames
            log_probs = log_probs.permute(1, 0, 2)          # (T, 1, 85)

            targets = torch.tensor(target_indices, dtype=torch.long).to(DEVICE)
            input_lengths = torch.tensor([real_frames], dtype=torch.long).to(DEVICE)
            target_lengths = torch.tensor([len(target_indices)], dtype=torch.long).to(DEVICE)

            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            # ══════════════════════════════════════════════════════
            # TRAIN STEP 5: Backward pass + update CTC head
            # Gradients flow through CTC head ONLY (encoder frozen)
            # ══════════════════════════════════════════════════════
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss.item())

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")

        # ── Dev evaluation (WER + loss) ──
        print(f"\n  Evaluating on dev set...")
        dev_wer, dev_loss, dev_preds = evaluate_dev(
            encoder, ctc_head, dev_clips, processor,
            char_to_idx, idx_to_char, DEVICE
        )

        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_wer": dev_wer,
            "time": epoch_time,
            "skipped": skipped,
            "lr": current_lr,
        }
        epoch_history.append(epoch_record)

        print(f"\n  {'─'*60}")
        print(f"  Epoch {epoch}/{args.epochs} | time: {epoch_time:.0f}s ({epoch_time/60:.1f} min)")
        print(f"  Train CTC loss: {avg_train_loss:.4f}")
        print(f"  Dev CTC loss:   {dev_loss:.4f} | Dev WER: {dev_wer*100:.1f}%")
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

        print(f"  {'─'*60}\n")

        # ── Save checkpoints ──
        checkpoint = {
            "epoch": epoch,
            "ctc_head_state_dict": ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_wer": dev_wer,
            "vocab_size": vocab_size,
            "encoder_dim": encoder_dim,
            "encoder_checkpoint": ENCODER_CHECKPOINT,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"ctc_epoch_{epoch}.pt"))

        # Track best dev WER (primary metric) and best dev loss
        improved = False
        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "ctc_best_wer.pt"))
            print(f"  ✓ New best dev WER: {best_dev_wer*100:.1f}%")
            improved = True

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "ctc_best_loss.pt"))
            if not improved:
                print(f"  ✓ New best dev loss: {best_dev_loss:.4f}")
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
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Epoch-by-epoch summary:")
    print(f"  {'Epoch':>5} {'TrainLoss':>10} {'DevLoss':>10} {'DevWER':>10} {'LR':>10} {'Time':>8}")
    print(f"  {'─'*58}")
    for r in epoch_history:
        print(
            f"  {r['epoch']:>5} {r['train_loss']:>10.4f} {r['dev_loss']:>10.4f} "
            f"{r['dev_wer']*100:>9.1f}% {r['lr']:>10.2e} {r['time']:>7.0f}s"
        )

    print(f"\n  Best dev WER:  {best_dev_wer*100:.1f}%")
    print(f"  Best dev loss: {best_dev_loss:.4f}")
    print(f"  Checkpoints:   {CHECKPOINT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
