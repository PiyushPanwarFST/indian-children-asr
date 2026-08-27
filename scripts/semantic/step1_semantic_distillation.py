"""
Step 1: Semantic Branch — CTC-based Knowledge Distillation (v2 with fixes)
==========================================================================

WHAT THIS SCRIPT DOES:
    Trains Whisper Small to transcribe Hindi/Marathi children's speech
    using pseudo-labels from IndicConformer (teacher).

    Flow:
        Audio → Whisper Encoder → CTC Head → Logits → CTC Loss vs pseudo-labels

FIXES IN THIS VERSION (3 fixes for overfitting):

    FIX 1: Teacher confidence filtering
        SOURCE: uDistil-Whisper (Waheed et al., NAACL 2025)
        WHAT: Filter out clips where teacher was not confident in its prediction.
        HOW: Step 0 now saves a confidence score per clip. We discard clips
             with confidence < 0.5 (teacher wasn't sure → bad pseudo-label).
        WHY: 18% of IndicConformer's predictions on children's speech are garbage.
             Training on garbage causes the student to memorize noise → overfit.

    FIX 2: SpecAugment (data augmentation on mel spectrogram)
        SOURCE: SpecAugment (Park et al., Interspeech 2019)
        Also used in: Distil-Whisper, Conformer, Wav2Vec2
        WHAT: Randomly mask blocks of time and frequency in the mel spectrogram.
        HOW: During training, randomly zero out:
             - 2 frequency bands of width 15 (out of 80 mel bins)
             - 2 time blocks of width 50 (out of 3000 mel frames)
        WHY: Forces the student to be robust — it can't memorize exact audio patterns.
             Teacher generated labels on CLEAN audio, student trains on AUGMENTED audio.
             This is the single most impactful regularization for ASR.

    FIX 3: Label smoothing in CTC loss
        SOURCE: Conformer (Gulati et al., Interspeech 2020), used value 0.1
        Also standard in: ESPnet, WeNet, NeMo ASR recipes
        WHAT: Instead of 100% probability on the correct token, spread 10% across all tokens.
        HOW: Blend CTC loss with a uniform distribution penalty:
             smoothed_loss = (1 - 0.1) * CTC_loss + 0.1 * uniform_penalty
        WHY: Prevents the model from becoming overconfident in its predictions.
             Overconfident models overfit — they memorize exact training targets.

Architecture: Whisper Encoder → Dropout → Linear(768, vocab) → CTC Loss

Prerequisites:
    - Run step0_generate_teacher_transcripts.py first (generates confidence scores)

Usage:
    # Quick verify (20 clips, 2 epochs)
    python scripts/semantic/step1_semantic_distillation.py --max_clips 20 --epochs 2

    # Full training
    python scripts/semantic/step1_semantic_distillation.py --epochs 20 --patience 5
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
from transformers import WhisperModel, WhisperProcessor


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEACHER_CSV = "benchmarks/teacher_transcripts.csv"
TEACHER_DEV_CSV = "benchmarks/teacher_transcripts_dev.csv"
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"
CHECKPOINT_DIR = "checkpoints/semantic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds

# FIX 1 config: Confidence threshold for filtering
# Clips with teacher confidence below this are discarded
# SOURCE: uDistil-Whisper uses various filtering; we use confidence >= 0.5
CONFIDENCE_THRESHOLD = 0.5

# FIX 2 config: SpecAugment parameters
# SOURCE: SpecAugment (Park et al., Interspeech 2019), Table 1 "SM" policy
SPEC_AUG_FREQ_MASKS = 2       # Number of frequency masks to apply
SPEC_AUG_FREQ_WIDTH = 15      # Max width of each frequency mask (out of 80 mel bins)
SPEC_AUG_TIME_MASKS = 2       # Number of time masks to apply
SPEC_AUG_TIME_WIDTH = 50      # Max width of each time mask (out of 3000 mel frames)

# FIX 3 config: Label smoothing
# SOURCE: Conformer (Gulati et al., 2020) uses 0.1
LABEL_SMOOTHING = 0.1


# ─────────────────────────────────────────────
# FIX 2: SpecAugment — Data Augmentation
# ─────────────────────────────────────────────
def apply_spec_augment(input_features,
                       num_freq_masks=SPEC_AUG_FREQ_MASKS,
                       freq_mask_width=SPEC_AUG_FREQ_WIDTH,
                       num_time_masks=SPEC_AUG_TIME_MASKS,
                       time_mask_width=SPEC_AUG_TIME_WIDTH):
    """
    Apply SpecAugment to mel spectrogram — randomly mask frequency and time bands.

    SOURCE: "SpecAugment: A Simple Data Augmentation Method for ASR"
            Park et al., Interspeech 2019
            https://arxiv.org/abs/1904.08779

    Also used in: Distil-Whisper, Conformer, Wav2Vec2, HuggingFace ASR recipes

    WHAT IT DOES:
        Takes a mel spectrogram and zeros out random rectangles:

        Before SpecAugment:          After SpecAugment:
        ████████████████████         ████████████████████
        ████████████████████         ████████████████████
        ████████████████████         ░░░░░░░░░░░░░░░░░░░░  ← frequency mask (whole band zeroed)
        ████████████████████         ████████████████████
        ████████████████████         ██████░░░░██████████  ← time mask (vertical strip zeroed)
        ████████████████████         ████████████████████

    WHY IT HELPS:
        - Teacher generated pseudo-labels on CLEAN audio
        - Student trains on AUGMENTED (masked) audio
        - Student can't memorize exact spectral patterns → must learn robust features
        - Like training with partial information → better generalization

    Args:
        input_features (Tensor): Mel spectrogram, shape (1, 80, 3000)
            - 80 = frequency bins (mel scale)
            - 3000 = time frames (100 frames/sec × 30 sec)
        num_freq_masks (int): How many horizontal bands to mask. Default: 2
        freq_mask_width (int): Max width of each freq mask. Default: 15 bins
        num_time_masks (int): How many vertical strips to mask. Default: 2
        time_mask_width (int): Max width of each time mask. Default: 50 frames

    Returns:
        Tensor: Augmented mel spectrogram, same shape (1, 80, 3000)
            Some regions will be zeroed out.
    """
    # Clone so we don't modify the original tensor
    augmented = input_features.clone()

    _, num_freq_bins, num_time_steps = augmented.shape  # (1, 80, 3000)

    # ── Frequency masking ──
    # Zero out entire horizontal bands (e.g., bins 20-35 all set to 0)
    # This removes some frequency information → student must infer from context
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_width)  # mask width (0 to 15)
        f0 = random.randint(0, max(0, num_freq_bins - f))  # start position
        augmented[:, f0:f0 + f, :] = 0  # zero out frequency band

    # ── Time masking ──
    # Zero out entire vertical strips (e.g., frames 100-150 all set to 0)
    # This removes some temporal information → student must handle gaps
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_width)  # mask width (0 to 50)
        t0 = random.randint(0, max(0, num_time_steps - t))  # start position
        augmented[:, :, t0:t0 + t] = 0  # zero out time strip

    return augmented


# ─────────────────────────────────────────────
# STUDENT MODEL: Whisper Encoder + CTC Head
# ─────────────────────────────────────────────
class WhisperCTC(nn.Module):
    """
    Whisper encoder with a CTC head on top.

    Architecture:
        Audio (80×3000) → Encoder → Features (1500×768) → Dropout → CTC Head → Logits (1500×51866)

    Args:
        whisper_model_name (str): HuggingFace model ID (e.g., "openai/whisper-small")
        vocab_size (int): Number of tokens in vocabulary (51866 for Whisper)
        dropout (float): Dropout rate (default 0.1)

    Input:  input_features, shape (batch, 80, 3000)
    Output: logits, shape (batch, 1500, vocab_size)
    """

    def __init__(self, whisper_model_name, vocab_size, dropout=0.1):
        super().__init__()
        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper.encoder
        self.dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(self.encoder.config.d_model, vocab_size)
        del whisper.decoder

    def forward(self, input_features):
        encoder_output = self.encoder(input_features).last_hidden_state  # (B, 1500, 768)
        encoder_output = self.dropout(encoder_output)  # (B, 1500, 768)
        logits = self.ctc_head(encoder_output)  # (B, 1500, vocab_size)
        return logits


# ─────────────────────────────────────────────
# FIX 3: CTC Loss with Label Smoothing
# ─────────────────────────────────────────────
def ctc_loss_with_label_smoothing(log_probs, targets, input_lengths, target_lengths,
                                  blank=0, label_smoothing=LABEL_SMOOTHING):
    """
    CTC loss with label smoothing — prevents overconfident predictions.

    SOURCE: Conformer (Gulati et al., Interspeech 2020), Section 3.5
            Also standard in ESPnet and WeNet ASR toolkits.
            https://arxiv.org/abs/2005.08100

    WHAT IT DOES:
        Standard CTC loss pushes the model to put 100% probability on the correct
        token at each frame. This makes the model overconfident → overfitting.

        Label smoothing blends two objectives:
            smoothed_loss = (1 - ε) × CTC_loss + ε × uniform_penalty

        where ε = 0.1 (label_smoothing) and uniform_penalty encourages the model
        to spread some probability across all tokens (not just the correct one).

    WHY IT HELPS:
        - Without smoothing: model puts 99.9% on correct token → memorizes training data
        - With smoothing: model puts ~90% on correct token → more generalizable
        - Especially important when pseudo-labels have errors (noisy teacher)

    MATH:
        uniform_penalty = -mean(log_probs)
        This is equivalent to KL divergence from a uniform distribution.
        It penalizes the model for being too "peaky" in its predictions.

    Args:
        log_probs (Tensor): Student's log probabilities, shape (T, batch, vocab_size)
        targets (Tensor): Target token IDs, shape (sum_of_target_lengths,)
        input_lengths (Tensor): Number of real frames per sample, shape (batch,)
        target_lengths (Tensor): Number of target tokens per sample, shape (batch,)
        blank (int): Blank token ID for CTC (default 0)
        label_smoothing (float): Smoothing factor ε (default 0.1)
            - 0.0 = no smoothing (standard CTC)
            - 0.1 = recommended (Conformer paper)
            - 1.0 = completely uniform (useless)

    Returns:
        Tensor: Scalar loss value (smoothed CTC loss)
    """
    # ── Standard CTC loss ──
    # This is the normal CTC loss that aligns logits to target tokens
    ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
    with torch.backends.cudnn.flags(enabled=False):
        ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    # ── Uniform distribution penalty ──
    # -mean(log_probs) = KL divergence from uniform distribution
    # This encourages the model to not be overconfident at any frame
    # If model puts 100% on one token, -log(1.0) = 0 for that token but
    # -log(0) = inf for others → high penalty. Smoothing prevents this.
    uniform_penalty = -log_probs.mean()

    # ── Blend: (1 - ε) × CTC + ε × uniform ──
    smoothed_loss = (1 - label_smoothing) * ctc_loss + label_smoothing * uniform_penalty

    return smoothed_loss


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_teacher_transcripts(csv_path, max_clips=None, min_confidence=CONFIDENCE_THRESHOLD):
    """
    Load teacher's pseudo-labels from CSV, filtering by confidence.

    FIX 1: Confidence-based filtering
        SOURCE: uDistil-Whisper (Waheed et al., NAACL 2025)
        They filter pseudo-labels using confidence metrics.
        We use the geometric mean of teacher's frame-level max probabilities
        (computed in Step 0).

    Args:
        csv_path (str): Path to teacher_transcripts.csv (with confidence column)
        max_clips (int, optional): Limit clips for testing
        min_confidence (float): Minimum teacher confidence to keep a clip.
            Default: 0.5 (keep clips where teacher was ≥50% confident)

    Returns:
        list[dict]: Filtered clips with audio_path, teacher_transcript, confidence, etc.
    """
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append({
                "audio_path": row["audio_path"],
                "teacher_transcript": row["teacher_transcript"],
                "ground_truth": row["ground_truth"],
                "language": row["language"],
                "duration": float(row["duration"]),
                "confidence": float(row.get("confidence", 1.0)),  # backward compat
            })

    # Skip clips longer than 30 seconds
    clips = [c for c in clips if c["duration"] <= MAX_DURATION]

    # ── FIX 1: Filter by teacher confidence ──
    # Remove clips where teacher wasn't confident → likely garbage predictions
    before_conf = len(clips)
    clips = [c for c in clips if c["confidence"] >= min_confidence]
    filtered_conf = before_conf - len(clips)
    if filtered_conf > 0:
        print(f"  [FIX 1] Filtered {filtered_conf} clips with confidence < {min_confidence} "
              f"({filtered_conf/before_conf*100:.1f}% removed)")

    # Also filter by minimum text length (original filter)
    before_len = len(clips)
    clips = [c for c in clips if len(c["teacher_transcript"].strip()) >= 10]
    filtered_len = before_len - len(clips)
    if filtered_len > 0:
        print(f"  Filtered {filtered_len} clips with transcript < 10 chars")

    if max_clips:
        clips = clips[:max_clips]

    return clips


def load_dev_clips(csv_path, max_clips=None, min_confidence=CONFIDENCE_THRESHOLD):
    """
    Load dev clips with teacher pseudo-labels for validation.

    We now use teacher pseudo-labels for dev evaluation too (not ground truth),
    so that train and dev losses are computed the same way and are comparable.

    Args:
        csv_path (str): Path to teacher_transcripts_dev.csv
        max_clips (int, optional): Limit clips for testing
        min_confidence (float): Minimum confidence threshold

    Returns:
        list[dict]: Dev clips with teacher_transcript and confidence
    """
    # If dev teacher transcripts exist, use them
    if os.path.exists(csv_path):
        clips = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clips.append({
                    "audio_path": row["audio_path"],
                    "teacher_transcript": row["teacher_transcript"],
                    "transcript": row["ground_truth"],  # keep for WER
                    "language": row["language"],
                    "duration": float(row["duration"]),
                    "confidence": float(row.get("confidence", 1.0)),
                })
        clips = [c for c in clips if c["duration"] <= MAX_DURATION]
        clips = [c for c in clips if c["confidence"] >= min_confidence]
        clips = [c for c in clips if len(c["teacher_transcript"].strip()) >= 10]
        if max_clips:
            clips = clips[:max_clips]
        return clips

    # Fallback: load from regular dev CSV with ground truth
    print(f"  WARNING: {csv_path} not found, using ground truth for dev")
    clips = []
    fallback_csv = DEV_CSV
    with open(fallback_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"] not in ("Hindi", "Marathi"):
                continue
            clips.append({
                "audio_path": row["audio_path"],
                "teacher_transcript": row["transcript"],  # use ground truth as fallback
                "transcript": row["transcript"],
                "language": row["language"],
                "duration": float(row["duration_sec"]),
                "confidence": 1.0,
            })
    clips = [c for c in clips if c["duration"] <= MAX_DURATION]
    if max_clips:
        clips = clips[:max_clips]
    return clips


# ─────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────
def prepare_audio(audio_path, processor):
    """
    Load audio and create mel spectrogram + count real frames.

    Args:
        audio_path (str): Path to .wav file
        processor (WhisperProcessor): Feature extractor

    Returns:
        input_features (Tensor): shape (1, 80, 3000)
        real_encoder_frames (int): non-padding frame count (1 to 1500)
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


def prepare_targets(text, tokenizer):
    """
    Convert text to token IDs for CTC loss.

    Args:
        text (str): Teacher's pseudo-label transcript
        tokenizer: Whisper tokenizer

    Returns:
        target_ids (Tensor): shape (num_tokens,)
        target_length (int): number of tokens
    """
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    target_ids = torch.tensor(tokens, dtype=torch.long)
    return target_ids, len(tokens)


def ctc_decode_greedy(logits, tokenizer):
    """
    CTC greedy decoding: logits → text.

    Args:
        logits (Tensor): shape (1, 1500, vocab_size)
        tokenizer: Whisper tokenizer

    Returns:
        str: Decoded text
    """
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (1500,)
    collapsed = torch.unique_consecutive(predicted_ids)
    collapsed = collapsed[collapsed != 0]  # 0 = blank
    if len(collapsed) == 0:
        return ""
    return tokenizer.decode(collapsed.tolist(), skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def normalize_text(text):
    """Normalize text for WER computation."""
    text = text.strip().lower()
    for ch in ".,!?;:\"'()[]{}—–-।":
        text = text.replace(ch, "")
    return " ".join(text.split())


def compute_wer(predicted, reference):
    """Compute Word Error Rate."""
    from jiwer import wer as jiwer_wer
    pred = normalize_text(predicted)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
    return jiwer_wer(ref, pred)


def evaluate_dev(model, dev_clips, processor, device):
    """
    Evaluate on dev set using teacher pseudo-labels (same metric as training).

    Uses the same CTC loss with label smoothing as training,
    so train and dev losses are directly comparable.

    Args:
        model (WhisperCTC): Student model
        dev_clips (list): Dev clips with teacher_transcript
        processor (WhisperProcessor): Feature extractor + tokenizer
        device (str): "cuda" or "cpu"

    Returns:
        float: Average CTC loss on dev set
    """
    model.eval()
    tokenizer = processor.tokenizer
    losses = []

    with torch.no_grad():
        for clip in dev_clips:
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)
            input_features = input_features.to(device)

            # NOTE: No SpecAugment during evaluation — only during training
            # This is standard practice (augment train, evaluate clean)

            target_ids, target_len = prepare_targets(clip["teacher_transcript"], tokenizer)

            if target_len >= real_frames or target_len == 0:
                continue

            logits = model(input_features)  # (1, 1500, vocab_size)

            log_probs = nn.functional.log_softmax(logits.float(), dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (1500, 1, vocab_size)

            input_lengths = torch.tensor([real_frames], dtype=torch.long)
            target_lengths = torch.tensor([target_len], dtype=torch.long)

            # Use label smoothing in eval too for consistent comparison
            loss = ctc_loss_with_label_smoothing(
                log_probs, target_ids.to(device), input_lengths, target_lengths
            )

            if not torch.isinf(loss) and not torch.isnan(loss):
                losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Semantic branch: CTC-based KD (v2 with fixes)")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--max_clips", type=int, default=None, help="Limit clips (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--confidence_threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Min teacher confidence to keep clip (FIX 1)")
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING,
                        help="Label smoothing factor (FIX 3)")
    parser.add_argument("--no_spec_augment", action="store_true",
                        help="Disable SpecAugment (FIX 2) for ablation")
    args = parser.parse_args()

    # ── Reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_spec_augment = not args.no_spec_augment

    # ── Print fix configuration ──
    print(f"\n{'='*60}")
    print(f"  SEMANTIC BRANCH v2 — With 3 Fixes")
    print(f"{'='*60}")
    print(f"  FIX 1: Confidence filtering    threshold={args.confidence_threshold}")
    print(f"         Source: uDistil-Whisper (Waheed et al., NAACL 2025)")
    print(f"  FIX 2: SpecAugment             enabled={use_spec_augment}")
    print(f"         Source: Park et al., Interspeech 2019")
    print(f"  FIX 3: Label smoothing         epsilon={args.label_smoothing}")
    print(f"         Source: Conformer (Gulati et al., Interspeech 2020)")
    print(f"{'='*60}\n")

    # ── Load Data ──
    print("Loading data...")
    if not os.path.exists(TEACHER_CSV):
        print(f"ERROR: {TEACHER_CSV} not found.")
        print("Run step0_generate_teacher_transcripts.py first.")
        return

    train_clips = load_teacher_transcripts(
        TEACHER_CSV, max_clips=args.max_clips,
        min_confidence=args.confidence_threshold,
    )
    dev_clips = load_dev_clips(
        TEACHER_DEV_CSV, max_clips=args.max_clips,
        min_confidence=args.confidence_threshold,
    )

    print(f"  Training: {len(train_clips)} clips (after filtering)")
    print(f"  Dev:      {len(dev_clips)} clips")

    lang_counts = {}
    for c in train_clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count} clips")

    # ── Load Processor ──
    print("\nLoading Whisper processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    vocab_size = processor.tokenizer.vocab_size + len(processor.tokenizer.get_added_vocab())
    print(f"  Vocabulary size: {vocab_size}")

    # ── Build Student Model ──
    print("Loading student model (Whisper encoder + CTC head)...")
    model = WhisperCTC(
        whisper_model_name="openai/whisper-small",
        vocab_size=vocab_size,
        dropout=0.1,
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_clips) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        else:
            remaining = total_steps - step
            total_decay = total_steps - args.warmup_steps
            return max(0.0, remaining / max(1, total_decay))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Verify Pipeline (1 clip) ──
    print("\nVerifying pipeline with 1 clip...")
    test_clip = train_clips[0]
    input_features, real_frames = prepare_audio(test_clip["audio_path"], processor)
    target_ids, target_len = prepare_targets(
        test_clip["teacher_transcript"], processor.tokenizer
    )
    print(f"  Audio: {test_clip['audio_path'].split('/')[-1]}")
    print(f"  Teacher transcript: {test_clip['teacher_transcript'][:50]}...")
    print(f"  Teacher confidence: {test_clip['confidence']:.3f}")
    print(f"  Input features: {input_features.shape}")
    print(f"  Real encoder frames: {real_frames} / 1500")
    print(f"  Target tokens: {target_len}")

    # Test SpecAugment
    if use_spec_augment:
        augmented = apply_spec_augment(input_features.clone())
        zeros_before = (input_features == 0).sum().item()
        zeros_after = (augmented == 0).sum().item()
        print(f"  [FIX 2] SpecAugment: {zeros_after - zeros_before} values zeroed out")

    # Test forward + loss
    input_features_device = input_features.to(DEVICE)
    logits = model(input_features_device)
    log_probs = nn.functional.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    input_lengths = torch.tensor([real_frames], dtype=torch.long)
    target_lengths = torch.tensor([target_len], dtype=torch.long)

    loss = ctc_loss_with_label_smoothing(
        log_probs, target_ids.to(DEVICE), input_lengths, target_lengths,
        label_smoothing=args.label_smoothing,
    )
    print(f"  [FIX 3] CTC loss (smoothed, ε={args.label_smoothing}): {loss.item():.4f}")
    print("  Pipeline verified ✓")

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"  TRAINING: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
    print(f"  Fixes: confidence>={args.confidence_threshold}, SpecAug={use_spec_augment}, "
          f"smooth={args.label_smoothing}")
    print(f"{'='*60}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_train_loss = float("inf")
    best_dev_loss = float("inf")
    patience_counter = 0
    global_step = 0
    epoch_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        epoch_losses = []
        skipped = 0

        random.shuffle(train_clips)

        for i, clip in enumerate(train_clips):
            step_start = time.time()

            # ── STEP 1: Prepare audio → mel spectrogram ──
            input_features, real_frames = prepare_audio(clip["audio_path"], processor)

            # ── STEP 2: Get teacher's pseudo-label → token IDs ──
            target_ids, target_len = prepare_targets(
                clip["teacher_transcript"], processor.tokenizer
            )

            if target_len >= real_frames or target_len == 0 or real_frames == 0:
                skipped += 1
                continue

            # ── FIX 2: Apply SpecAugment to mel spectrogram ──
            # Only during training, not during evaluation
            # Teacher generated labels on CLEAN audio, student trains on AUGMENTED audio
            if use_spec_augment:
                input_features = apply_spec_augment(input_features)

            input_features = input_features.to(DEVICE)

            # ── STEP 3: Student forward → logits ──
            logits = model(input_features)  # (1, 1500, vocab_size)

            # Convert to log probabilities for CTC
            log_probs = nn.functional.log_softmax(logits.float(), dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (1500, 1, vocab_size)

            input_lengths = torch.tensor([real_frames], dtype=torch.long)
            target_lengths = torch.tensor([target_len], dtype=torch.long)

            # ── FIX 3: CTC Loss with label smoothing ──
            loss = ctc_loss_with_label_smoothing(
                log_probs, target_ids.to(DEVICE), input_lengths, target_lengths,
                label_smoothing=args.label_smoothing,
            )

            if torch.isinf(loss) or torch.isnan(loss):
                skipped += 1
                continue

            # ── STEP 4: Backward + update ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss.item())

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_clips):
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                elapsed = time.time() - step_start
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch} | {i+1}/{len(train_clips)} | "
                    f"loss: {avg_loss:.4f} | lr: {current_lr:.2e} | "
                    f"clip: {elapsed:.2f}s",
                    flush=True,
                )

        # ── Epoch Summary ──
        epoch_time = time.time() - epoch_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        avg_dev_loss = evaluate_dev(model, dev_clips, processor, DEVICE)

        gap = avg_dev_loss - avg_train_loss
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

        epoch_record = {
            "epoch": epoch, "train_loss": avg_train_loss,
            "dev_loss": avg_dev_loss, "gap": gap,
            "time": epoch_time, "skipped": skipped,
        }
        epoch_history.append(epoch_record)

        print(f"\n  {'─'*55}")
        print(f"  Epoch {epoch}/{args.epochs} | time: {epoch_time:.0f}s ({epoch_time/60:.1f} min)")
        print(f"  Train loss: {avg_train_loss:.4f} | Dev loss: {avg_dev_loss:.4f} | Gap: {gap:.4f}")
        print(f"  Skipped: {skipped} clips | GPU peak: {gpu_mem:.0f} MB")
        print(f"  {'─'*55}\n")

        # ── Save Checkpoints ──
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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
            print(f"  ✓ New best dev loss: {best_dev_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Print Summary ──
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Fixes applied:")
    print(f"    FIX 1: Confidence filtering (threshold={args.confidence_threshold})")
    print(f"    FIX 2: SpecAugment (enabled={use_spec_augment})")
    print(f"    FIX 3: Label smoothing (epsilon={args.label_smoothing})")
    print(f"\n  Epoch-by-epoch summary:")
    print(f"  {'Epoch':>5} {'Train':>10} {'Dev':>10} {'Gap':>10} {'Time':>8}")
    print(f"  {'─'*45}")
    for r in epoch_history:
        print(
            f"  {r['epoch']:>5} {r['train_loss']:>10.4f} {r['dev_loss']:>10.4f} "
            f"{r['gap']:>10.4f} {r['time']:>7.0f}s"
        )

    print(f"\n  Best train loss: {best_train_loss:.4f}")
    print(f"  Best dev loss:   {best_dev_loss:.4f}")
    print(f"  Checkpoints in:  {CHECKPOINT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
