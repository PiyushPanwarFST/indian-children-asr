"""
Step 1: Pre-Compute CTC-Based Word Alignments
==============================================

WHAT:
    Uses our trained acoustic CTC model (Whisper encoder + CTC head) to
    compute which audio frames correspond to which words in the transcript.
    Saves per-clip alignment files mapping each ground-truth word to a
    (start_frame, end_frame) range in the encoder output.

WHY:
    For cross-modal knowledge distillation with MuRIL, we need to compute
    MSE loss between word-level speech embeddings and MuRIL text embeddings.
    To extract word-level speech embeddings, we must know which encoder
    frames belong to which word. CTC alignment gives us this mapping.

    The alternative (attention-based alignment) requires training an
    attention model. CTC alignment is free — we already have a trained
    CTC model from the joint training step.

    Pre-computing alignments (instead of computing on-the-fly) saves
    ~2-3 seconds per clip per epoch. Over 9K clips x 20 epochs = ~100 hours
    saved. One-time cost: ~30 minutes.

OUTPUT:
    Per clip: ctc_alignments/{split}/{uid}.pt containing:
        - word_boundaries: list of dicts with word, start_frame, end_frame,
          chars_matched, chars_total
        - num_frames: total encoder frames for this clip
        - num_words_aligned: how many words got valid alignments
        - num_words_total: total words in ground truth
        - alignment_quality: fraction of words successfully aligned

    A word is "successfully aligned" if >= 50% of its characters were
    matched in the CTC output. Words with poor alignment are skipped
    entirely (not included in word_boundaries).

HOW IT WORKS:
    1. Run audio through Whisper encoder -> (1, T, 768) features
    2. Run features through CTC head -> (1, T, 85) logits
    3. Argmax over vocab dim -> frame-level character predictions
    4. CTC collapse: group consecutive identical non-blank predictions
       into character segments with (char_index, start_frame, end_frame)
    5. For each ground-truth word, greedily match its characters (converted
       to vocab indices) against the CTC segments left-to-right
    6. Record the frame span from first matched char to last matched char

Usage:
    # Quick verify with 5 clips
    python step1_precompute_ctc_alignments.py --verify --max_clips 5

    # Process training split
    python step1_precompute_ctc_alignments.py --split train

    # Process all splits
    python step1_precompute_ctc_alignments.py --split all

    # Custom checkpoint
    python step1_precompute_ctc_alignments.py --checkpoint path/to/ckpt.pt
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# Suppress noisy transformer warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import WhisperModel, WhisperFeatureExtractor

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT = PROJECT_ROOT / "ASER-Dataset"

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text

# ── Config ───────────────────────────────────────────────────────────────────
WHISPER_ID = "openai/whisper-small"
SAMPLE_RATE = 16000
MAX_DURATION = 30.0  # seconds

TRAIN_CSV = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV = ASER_ROOT / "splits" / "asr_dev.csv"
TEST_CSV = ASER_ROOT / "splits" / "asr_test.csv"

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "joint" / "joint_best_wer.pt"
VOCAB_PATH = ASER_ROOT / "vocab.json"

BLANK_IDX = 0
SPACE_IDX = 1
UNK_IDX = 2

# Minimum fraction of word characters that must match CTC segments
# for the word alignment to be considered valid.
MIN_CHAR_MATCH_RATIO = 0.5


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio(path, max_sec=30.0):
    """
    Load and preprocess audio file to 16kHz mono waveform.

    Args:
        path (str): Path to audio file.
        max_sec (float): Maximum duration in seconds. Audio beyond this is
            truncated. None for no limit.

    Returns:
        torch.Tensor: 1-D waveform tensor at 16kHz.
    """
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()
    if max_sec and len(wav) > int(max_sec * 16000):
        wav = wav[:int(max_sec * 16000)]
    return wav


def compute_real_frames(num_samples, sample_rate=16000):
    """
    Compute number of real encoder output frames for a given audio length.

    Whisper's mel spectrogram uses 10ms hop (160 samples per frame), then
    the convolutional frontend has stride 2, so encoder output has
    num_samples // 160 // 2 frames, capped at 1500 (Whisper's max for 30s).

    Args:
        num_samples (int): Number of audio samples at sample_rate.
        sample_rate (int): Sample rate (default 16000).

    Returns:
        int: Number of encoder frames.
    """
    frames = num_samples // 320  # 160 * 2 = 320
    return min(frames, 1500)


# ── Vocabulary ───────────────────────────────────────────────────────────────

def load_vocab(vocab_path):
    """
    Load character vocabulary from vocab.json.

    Args:
        vocab_path (str or Path): Path to vocab.json.

    Returns:
        char_to_idx (dict): Maps character string to index.
        idx_to_char (dict): Maps index to character string.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char


def text_to_indices(text, char_to_idx):
    """
    Convert normalized text to a list of vocabulary indices.

    Each character is mapped to its vocab index. Spaces become <space> (1).
    Unknown characters become <unk> (2). <blank> (0) is never in targets.

    Hindi/Marathi characters may be multi-codepoint (e.g., "ख़" = ख + ़).
    The vocab contains both single-codepoint and multi-codepoint entries.
    We try longest-match-first: check 2-char substring before falling back
    to single char.

    Args:
        text (str): Normalized text (lowercase, no punctuation).
        char_to_idx (dict): Vocabulary mapping.

    Returns:
        list[int]: Character indices.
    """
    indices = []
    i = 0
    while i < len(text):
        if text[i] == " ":
            indices.append(char_to_idx["<space>"])
            i += 1
        # Try 2-char match first (for nukta combinations like ख़, ग़, etc.)
        elif i + 1 < len(text) and text[i:i+2] in char_to_idx:
            indices.append(char_to_idx[text[i:i+2]])
            i += 2
        elif text[i] in char_to_idx:
            indices.append(char_to_idx[text[i]])
            i += 1
        else:
            indices.append(char_to_idx.get("<unk>", UNK_IDX))
            i += 1
    return indices


# ── Clip loading ─────────────────────────────────────────────────────────────

def load_clips(csv_path, max_clips=None):
    """
    Load clips from ASER split CSV file.

    Reads Hindi, Marathi, and English clips. Constructs a unique ID per clip
    using child_id + filename basename.

    Args:
        csv_path (str or Path): Path to CSV file.
        max_clips (int or None): Maximum clips to load. None for all.

    Returns:
        list[dict]: Each dict has keys: audio_path, language, ground_truth,
            filename, clip_uid.
    """
    clips = []
    lang_map = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row.get("language", "")
            lang_code = lang_map.get(lang, "")

            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = str(ASER_ROOT / audio_path)

            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "ground_truth": row.get("transcript", row.get("que_text", row.get("text", ""))),
                "filename": os.path.basename(audio_path),
                "clip_uid": clip_uid,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips


# ── CTC alignment logic ─────────────────────────────────────────────────────

def ctc_collapse(frame_predictions):
    """
    CTC collapse: convert frame-level predictions to character segments.

    Groups consecutive identical non-blank predictions into segments.
    Each segment records the character index and the frame range it spans.

    Example:
        frame_predictions = [0, 0, 5, 5, 5, 0, 0, 12, 12, 0]
        (where 0 = blank)
        Result: [(5, 2, 4), (12, 7, 8)]
        Meaning: char_idx=5 spans frames 2-4, char_idx=12 spans frames 7-8.

    Args:
        frame_predictions (list[int]): Per-frame predicted character indices.
            Index 0 is blank.

    Returns:
        list[tuple]: Each tuple is (char_idx, start_frame, end_frame).
            Frame indices are inclusive on both ends.
    """
    segments = []
    prev_idx = None
    seg_start = None

    for frame_i, idx in enumerate(frame_predictions):
        if idx != prev_idx:
            # Previous segment ended — save it if it was non-blank
            if prev_idx is not None and prev_idx != BLANK_IDX:
                segments.append((prev_idx, seg_start, frame_i - 1))
            seg_start = frame_i
            prev_idx = idx

    # Handle last segment
    if prev_idx is not None and prev_idx != BLANK_IDX:
        segments.append((prev_idx, seg_start, len(frame_predictions) - 1))

    return segments


def match_word_to_segments(word_indices, segments, seg_cursor):
    """
    Greedily match a word's character indices against CTC segments.

    Starting from seg_cursor, for each character index in word_indices,
    scan forward through segments to find a matching one. Record the
    start_frame of the first match and end_frame of the last match.

    Args:
        word_indices (list[int]): Vocabulary indices for this word's characters.
            Does NOT include <space> or <blank>.
        segments (list[tuple]): CTC segments from ctc_collapse().
            Each is (char_idx, start_frame, end_frame).
        seg_cursor (int): Index into segments to start searching from.

    Returns:
        dict or None: If >= MIN_CHAR_MATCH_RATIO of characters matched:
            {
                "start_frame": int,
                "end_frame": int,
                "chars_matched": int,
                "next_cursor": int  (segment index after last match)
            }
            None if too few characters matched.
    """
    if not word_indices:
        return None

    matched = 0
    first_frame = None
    last_frame = None
    cursor = seg_cursor

    for char_idx in word_indices:
        # Scan forward from cursor to find this character
        found = False
        for s in range(cursor, len(segments)):
            if segments[s][0] == char_idx:
                # Match found
                if first_frame is None:
                    first_frame = segments[s][1]
                last_frame = segments[s][2]
                cursor = s + 1  # Move past this segment
                matched += 1
                found = True
                break

        # If not found, skip this character (CTC may have missed it)

    chars_total = len(word_indices)
    match_ratio = matched / chars_total if chars_total > 0 else 0.0

    if match_ratio < MIN_CHAR_MATCH_RATIO or first_frame is None:
        # Bad alignment — return None but still advance cursor
        return None

    return {
        "start_frame": first_frame,
        "end_frame": last_frame,
        "chars_matched": matched,
        "next_cursor": cursor,
    }


def compute_alignment(frame_predictions, ground_truth, char_to_idx, num_frames):
    """
    Compute word-level CTC alignment for one clip.

    Pipeline:
        1. CTC collapse frame predictions into character segments
        2. Normalize and split ground truth into words
        3. For each word, convert to vocab indices and match against segments
        4. Record valid word boundaries

    Args:
        frame_predictions (list[int]): Per-frame CTC predictions (after argmax).
        ground_truth (str): Raw ground truth transcript.
        char_to_idx (dict): Vocabulary mapping.
        num_frames (int): Total encoder frames.

    Returns:
        dict: Alignment result with keys:
            - word_boundaries (list[dict]): Each has word, start_frame,
              end_frame, chars_matched, chars_total.
            - num_frames (int): Total encoder frames.
            - num_words_aligned (int): Words with valid alignment.
            - num_words_total (int): Total words in ground truth.
            - alignment_quality (float): Fraction of words aligned.
    """
    # Step 1: CTC collapse
    segments = ctc_collapse(frame_predictions)

    # Step 2: Normalize and split ground truth
    gt_normalized = normalize_text(ground_truth)
    if not gt_normalized.strip():
        return {
            "word_boundaries": [],
            "num_frames": num_frames,
            "num_words_aligned": 0,
            "num_words_total": 0,
            "alignment_quality": 0.0,
        }

    gt_words = gt_normalized.split()

    # Step 3: Match each word
    word_boundaries = []
    seg_cursor = 0

    for word in gt_words:
        # Convert word characters to vocab indices (no spaces)
        word_indices = text_to_indices(word, char_to_idx)
        # Remove any <space> tokens that might sneak in (shouldn't happen
        # since we split on spaces, but be safe)
        word_indices = [idx for idx in word_indices if idx != SPACE_IDX]

        if not word_indices:
            continue

        result = match_word_to_segments(word_indices, segments, seg_cursor)
        if result is not None:
            word_boundaries.append({
                "word": word,
                "start_frame": result["start_frame"],
                "end_frame": result["end_frame"],
                "chars_matched": result["chars_matched"],
                "chars_total": len(word_indices),
            })
            seg_cursor = result["next_cursor"]

    num_words_total = len(gt_words)
    num_words_aligned = len(word_boundaries)
    alignment_quality = num_words_aligned / num_words_total if num_words_total > 0 else 0.0

    return {
        "word_boundaries": word_boundaries,
        "num_frames": num_frames,
        "num_words_aligned": num_words_aligned,
        "num_words_total": num_words_total,
        "alignment_quality": alignment_quality,
    }


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """
    Load Whisper encoder and CTC head from joint training checkpoint.

    The checkpoint contains:
        - encoder_state_dict: Whisper encoder weights (may have 'encoder.' prefix)
        - ctc_head_state_dict: Linear(768, 85) weights

    Args:
        checkpoint_path (str or Path): Path to joint_best_wer.pt.
        device (str): Device to load model on.

    Returns:
        encoder (nn.Module): Whisper encoder, eval mode.
        ctc_head (nn.Module): Linear CTC head, eval mode.
        feat_extractor: WhisperFeatureExtractor instance.
    """
    print(f"  Loading Whisper encoder from {WHISPER_ID}...")
    whisper_model = WhisperModel.from_pretrained(WHISPER_ID)
    encoder = whisper_model.encoder.to(device)

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # Load encoder weights (handle 'encoder.' prefix)
    state_dict = ckpt["encoder_state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("encoder.", "") if k.startswith("encoder.") else k
        cleaned[key] = v
    encoder.load_state_dict(cleaned)
    encoder.eval()

    # Load CTC head
    ctc_head_sd = ckpt["ctc_head_state_dict"]
    # Infer vocab size from weight shape
    vocab_size = ctc_head_sd["weight"].shape[0] if "weight" in ctc_head_sd else 85
    hidden_size = ctc_head_sd["weight"].shape[1] if "weight" in ctc_head_sd else 768
    ctc_head = nn.Linear(hidden_size, vocab_size).to(device)

    # Handle possible 'ctc_head.' prefix
    cleaned_ctc = {}
    for k, v in ctc_head_sd.items():
        key = k.replace("ctc_head.", "") if k.startswith("ctc_head.") else k
        cleaned_ctc[key] = v
    ctc_head.load_state_dict(cleaned_ctc)
    ctc_head.eval()

    feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  CTC head: Linear({hidden_size}, {vocab_size})")

    return encoder, ctc_head, feat_extractor


# ── Main processing ─────────────────────────────────────────────────────────

def process_clip(clip, encoder, ctc_head, feat_extractor, char_to_idx, device):
    """
    Process a single clip: load audio, run CTC, compute alignment.

    Args:
        clip (dict): Clip info with audio_path, ground_truth, clip_uid.
        encoder (nn.Module): Whisper encoder.
        ctc_head (nn.Module): CTC head.
        feat_extractor: WhisperFeatureExtractor.
        char_to_idx (dict): Vocabulary mapping.
        device (str): Device.

    Returns:
        dict: Alignment result (same format as compute_alignment output).
    """
    # Load audio
    wav = load_audio(clip["audio_path"], max_sec=MAX_DURATION)
    num_samples = len(wav)
    num_frames = compute_real_frames(num_samples)

    # Compute mel spectrogram
    mel = feat_extractor(
        wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = mel.input_features.to(device)  # (1, 80, 3000)

    # Forward through encoder
    with torch.no_grad():
        encoder_out = encoder(input_features).last_hidden_state  # (1, T, 768)
        # Trim to real frames
        encoder_out = encoder_out[:, :num_frames, :]
        # CTC head
        logits = ctc_head(encoder_out)  # (1, T, vocab_size)

    # Argmax -> frame predictions
    frame_preds = logits[0].argmax(dim=-1).cpu().tolist()  # list of length T

    # Compute alignment
    alignment = compute_alignment(
        frame_preds, clip["ground_truth"], char_to_idx, num_frames
    )

    return alignment


def verify_alignments(clips, encoder, ctc_head, feat_extractor, char_to_idx,
                      idx_to_char, device, num_verify=5):
    """
    Run alignment on a few clips and print detailed output for verification.

    Shows CTC predictions, character segments, word matches, and final
    boundaries so the user can visually confirm correctness.

    Args:
        clips (list[dict]): Clips to verify.
        encoder, ctc_head, feat_extractor: Model components.
        char_to_idx (dict): Vocabulary mapping.
        idx_to_char (dict): Reverse vocabulary mapping.
        device (str): Device.
        num_verify (int): Number of clips to verify.
    """
    print(f"\n{'='*70}")
    print(f"  VERIFICATION MODE — {num_verify} clips with detailed output")
    print(f"{'='*70}\n")

    verify_clips = clips[:num_verify]

    for i, clip in enumerate(verify_clips):
        print(f"--- Clip {i+1}/{num_verify}: {clip['clip_uid']} ---")
        print(f"  Ground truth: {clip['ground_truth'][:100]}")

        # Load and process
        wav = load_audio(clip["audio_path"], max_sec=MAX_DURATION)
        num_samples = len(wav)
        num_frames = compute_real_frames(num_samples)

        mel = feat_extractor(
            wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = mel.input_features.to(device)

        with torch.no_grad():
            encoder_out = encoder(input_features).last_hidden_state
            encoder_out = encoder_out[:, :num_frames, :]
            logits = ctc_head(encoder_out)

        frame_preds = logits[0].argmax(dim=-1).cpu().tolist()

        # Show CTC collapse
        segments = ctc_collapse(frame_preds)
        print(f"  Num frames: {num_frames}")
        print(f"  CTC segments ({len(segments)}):")
        seg_strs = []
        for char_idx, sf, ef in segments[:30]:  # Show first 30
            char_name = idx_to_char.get(char_idx, f"?{char_idx}")
            seg_strs.append(f"    '{char_name}'(idx={char_idx}) frames {sf}-{ef}")
        print("\n".join(seg_strs))
        if len(segments) > 30:
            print(f"    ... and {len(segments) - 30} more segments")

        # Show decoded text
        decoded_chars = []
        for char_idx, _, _ in segments:
            char_name = idx_to_char.get(char_idx, "")
            if char_name == "<space>":
                decoded_chars.append(" ")
            elif char_name not in ("<blank>", "<unk>"):
                decoded_chars.append(char_name)
        decoded_text = "".join(decoded_chars)
        print(f"  CTC decoded: {decoded_text[:100]}")

        # Show alignment
        alignment = compute_alignment(
            frame_preds, clip["ground_truth"], char_to_idx, num_frames
        )

        print(f"  Alignment quality: {alignment['alignment_quality']:.2f} "
              f"({alignment['num_words_aligned']}/{alignment['num_words_total']} words)")
        for wb in alignment["word_boundaries"][:10]:
            print(f"    '{wb['word']}': frames {wb['start_frame']}-{wb['end_frame']} "
                  f"({wb['chars_matched']}/{wb['chars_total']} chars)")
        if len(alignment["word_boundaries"]) > 10:
            print(f"    ... and {len(alignment['word_boundaries']) - 10} more words")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CTC-based word alignments for cross-modal KD"
    )
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev", "test", "all"],
                        help="Which split(s) to process. 'all' does train+dev+test.")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Max clips per split (for testing). Default: all.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if not set.")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification mode: show 5 clips with detailed output.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override default checkpoint path.")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT

    # Output root
    output_root = PROJECT_ROOT / "ctc_alignments"

    # Determine splits to process
    split_map = {
        "train": [("train", TRAIN_CSV)],
        "dev": [("dev", DEV_CSV)],
        "test": [("test", TEST_CSV)],
        "all": [("train", TRAIN_CSV), ("dev", DEV_CSV), ("test", TEST_CSV)],
    }
    splits = split_map[args.split]

    print(f"{'='*70}")
    print(f"  PRE-COMPUTE CTC WORD ALIGNMENTS")
    print(f"  Splits: {[s[0] for s in splits]}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {device}")
    print(f"  Output: {output_root}/")
    print(f"{'='*70}\n")

    # Load vocab
    char_to_idx, idx_to_char = load_vocab(VOCAB_PATH)
    print(f"  Vocab: {len(char_to_idx)} tokens from {VOCAB_PATH}")

    # Load model
    encoder, ctc_head, feat_extractor = load_model(ckpt_path, device)
    print()

    # Process each split
    for split_name, csv_path in splits:
        print(f"\n{'='*70}")
        print(f"  Processing split: {split_name}")
        print(f"  CSV: {csv_path}")
        print(f"{'='*70}\n")

        # Load clips
        clips = load_clips(csv_path, max_clips=args.max_clips)
        print(f"  Loaded {len(clips)} clips")

        # Verify mode
        if args.verify:
            verify_alignments(
                clips, encoder, ctc_head, feat_extractor,
                char_to_idx, idx_to_char, device, num_verify=5
            )
            continue

        # Create output directory
        out_dir = output_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Process clips
        success = 0
        errors = 0
        skipped = 0
        total_quality = 0.0
        total_words = 0
        total_aligned = 0
        start_time = time.time()

        for i, clip in enumerate(clips):
            uid = clip["clip_uid"]
            save_path = out_dir / f"{uid}.pt"

            # Skip if already computed
            if save_path.exists():
                if i < 3:
                    print(f"  [{i+1}/{len(clips)}] SKIP (exists): {uid}")
                skipped += 1
                continue

            try:
                alignment = process_clip(
                    clip, encoder, ctc_head, feat_extractor, char_to_idx, device
                )

                # Save
                torch.save(alignment, str(save_path))

                # Stats
                success += 1
                total_quality += alignment["alignment_quality"]
                total_words += alignment["num_words_total"]
                total_aligned += alignment["num_words_aligned"]

                # Progress
                elapsed = time.time() - start_time
                processed = success + errors
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(clips) - skipped - processed) / rate if rate > 0 else 0

                if i < 10 or (i + 1) % 200 == 0 or i == len(clips) - 1:
                    print(f"  [{i+1:5d}/{len(clips)}] {uid:30s} "
                          f"frames={alignment['num_frames']:4d} "
                          f"words={alignment['num_words_aligned']}/{alignment['num_words_total']} "
                          f"quality={alignment['alignment_quality']:.2f} "
                          f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

            except Exception as e:
                print(f"  [{i+1}/{len(clips)}] ERROR {uid}: {e}")
                errors += 1

        # Summary
        elapsed = time.time() - start_time
        print(f"\n  --- {split_name} Summary ---")
        print(f"  Processed: {success} clips in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Skipped (existing): {skipped}")
        print(f"  Errors: {errors}")
        if success > 0:
            avg_quality = total_quality / success
            word_rate = total_aligned / total_words if total_words > 0 else 0
            print(f"  Avg alignment quality: {avg_quality:.3f}")
            print(f"  Total words aligned: {total_aligned}/{total_words} ({word_rate:.1%})")

        # Disk space
        if success > 0:
            dir_size = sum(
                f.stat().st_size for f in out_dir.iterdir() if f.suffix == '.pt'
            )
            print(f"  Disk usage: {dir_size / 1024 / 1024:.1f} MB "
                  f"(~{dir_size / (success + skipped) / 1024:.1f} KB per clip)")

        print(f"  Output: {out_dir}/")

    print(f"\n{'='*70}")
    print(f"  ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
