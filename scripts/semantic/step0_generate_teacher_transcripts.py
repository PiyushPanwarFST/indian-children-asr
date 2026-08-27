"""
Step 0: Generate Teacher Transcripts + Confidence Scores (One-Time Offline)
==========================================================================
Runs IndicConformer on all Hindi/Marathi training clips to produce
"pseudo-labels" — the teacher's transcripts that the student will
learn from in Step 1.

NEW in this version (Fix 1 of 3):
    We now also save a CONFIDENCE SCORE for each transcript.
    This score measures how confident the teacher was in its prediction.
    In Step 1, clips with low confidence will be filtered out.

    SOURCE: uDistil-Whisper (Waheed et al., NAACL 2025)
    They use teacher's token-level confidence to filter bad pseudo-labels.
    We adapt this idea: instead of token-level, we use frame-level
    geometric mean of max softmax probabilities.

    FORMULA:
        For each frame t, teacher outputs logprobs over vocab.
        confidence_t = max(softmax(logprobs_t))  = probability of best token
        clip_confidence = geometric_mean(confidence_t for all non-blank frames)

    INTUITION:
        - High confidence (>0.8): teacher is sure → good pseudo-label
        - Low confidence (<0.3): teacher is uncertain → likely garbage
        - We only keep clips where teacher was confident

Why offline? IndicConformer takes ~4s per clip. Running it during
training would add hours per epoch. We pre-compute once and reuse.

Input:  ASER-Dataset/splits/asr_train.csv
Output: benchmarks/teacher_transcripts.csv (now with confidence column)

Usage:
    # Quick test (20 clips)
    python scripts/semantic/step0_generate_teacher_transcripts.py --max_clips 20

    # Full run (~9169 Hindi+Marathi clips, ~10 hours)
    python scripts/semantic/step0_generate_teacher_transcripts.py
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torchaudio


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_CSV = "ASER-Dataset/splits/asr_train.csv"
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"
OUTPUT_TRAIN_CSV = "benchmarks/teacher_transcripts.csv"
OUTPUT_DEV_CSV = "benchmarks/teacher_transcripts_dev.csv"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds


def load_clips(csv_path, max_clips=None):
    """
    Load Hindi and Marathi clips from a CSV file.
    We skip English because IndicConformer doesn't support it.

    Args:
        csv_path (str): Path to CSV file with columns: audio_path, transcript, language, duration_sec
        max_clips (int, optional): Limit number of clips for testing

    Returns:
        list[dict]: Each dict has audio_path, transcript, language, duration
    """
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only Hindi and Marathi — IndicConformer can't do English
            if row["language"] not in ("Hindi", "Marathi"):
                continue
            clips.append({
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],  # ground truth (for comparison)
                "language": row["language"],
                "duration": float(row["duration_sec"]),
            })

    # Skip clips longer than 30 seconds
    clips = [c for c in clips if c["duration"] <= MAX_DURATION]

    if max_clips:
        clips = clips[:max_clips]

    return clips


def load_indicconformer():
    """
    Load the IndicConformer 600M model.
    This model uses ONNX internally (AI4Bharat's custom packaging).
    trust_remote_code=True is needed because it's a custom HuggingFace model.

    Returns:
        model: IndicConformer model with .forward(), .encode(), ._ctc_decode() methods
    """
    from transformers import AutoModel

    print("Loading IndicConformer 600M...")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    )
    print("Model loaded.")
    return model


def transcribe_clip_with_confidence(model, audio_path, language):
    """
    Run IndicConformer on one audio clip → return predicted text AND confidence score.

    NEW: Instead of just returning text, we now also compute a confidence score
    by accessing the teacher's internal CTC logprobs.

    How confidence is computed:
        1. model.encode(wav) → encoder_outputs
        2. CTC decoder → logprobs, shape (1, T, vocab_size)
        3. For each frame: max_prob = max(exp(logprobs[frame]))
        4. Find non-blank frames (where argmax != BLANK_ID)
        5. confidence = geometric_mean(max_probs of non-blank frames)

    SOURCE: Adapted from uDistil-Whisper (Waheed et al., NAACL 2025)
    They use geometric mean of token probabilities as a confidence metric.

    Args:
        model: IndicConformer model
        audio_path (str): Path to .wav file
        language (str): "Hindi" or "Marathi"

    Returns:
        predicted_text (str): Decoded transcript
        confidence (float): 0.0 to 1.0, higher = more confident
            - > 0.8: very confident, good pseudo-label
            - 0.5-0.8: moderate confidence
            - < 0.3: low confidence, likely garbage
    """
    # Load audio — keep as 2D: shape (1, num_samples)
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo → mono

    # Trim to max duration
    max_samples = MAX_DURATION * SAMPLE_RATE
    wav = wav[:, :max_samples]

    # IndicConformer language codes
    lang_map = {"Hindi": "hi", "Marathi": "mr"}
    ic_lang = lang_map[language]

    try:
        # ── STEP 1: Encode audio → encoder outputs ──
        # model.encode() runs: preprocessor → ONNX encoder
        # Returns encoder_outputs shape: (1, T, 1024) where T varies with audio length
        encoder_outputs, encoded_lengths = model.encode(wav)

        # ── STEP 2: Get CTC logprobs from the CTC decoder ──
        # This is the SAME step that _ctc_decode() does internally.
        # logprobs shape: (1, T, 5633) for all languages
        logprobs_raw = model.models['ctc_decoder'].run(
            ['logprobs'], {'encoder_output': encoder_outputs}
        )[0]

        # Apply language mask (select only this language's tokens)
        # Then log_softmax to get proper log probabilities
        # After masking: shape (1, T, vocab_size_for_this_language)
        # For Hindi: vocab_size = 257 characters
        logprobs = torch.from_numpy(
            logprobs_raw[:, :, model.language_masks[ic_lang]]
        ).log_softmax(dim=-1)

        # ── STEP 3: Greedy decode (same as _ctc_decode) ──
        indices = torch.argmax(logprobs[0], dim=-1)  # (T,)
        collapsed_indices = torch.unique_consecutive(indices, dim=-1)
        predicted_text = ''.join(
            [model.vocab[ic_lang][x] for x in collapsed_indices
             if x != model.config.BLANK_ID]
        ).replace('▁', ' ').strip()

        # ── STEP 4: Compute confidence score ──
        # Convert log probabilities → probabilities
        probs = logprobs[0].exp()  # (T, vocab_size)

        # Get max probability at each frame (how sure is teacher about best token)
        max_probs = probs.max(dim=-1).values  # (T,)

        # Find non-blank frames (frames where model predicts actual characters)
        non_blank_mask = indices != model.config.BLANK_ID
        non_blank_probs = max_probs[non_blank_mask]

        if len(non_blank_probs) > 0:
            # Geometric mean = exp(mean(log(probs)))
            # More robust than arithmetic mean — single low-confidence frame
            # pulls the score down significantly
            log_probs_mean = non_blank_probs.log().mean().item()
            confidence = np.exp(log_probs_mean)
        else:
            # No non-blank frames → teacher predicted all blanks → zero confidence
            confidence = 0.0

    except Exception as e:
        print(f"    Error: {e}")
        predicted_text = ""
        confidence = 0.0

    return predicted_text, confidence


def process_clips(clips, model, label):
    """
    Transcribe a list of clips with confidence scores.

    Args:
        clips (list): Clips to process
        model: IndicConformer model
        label (str): "train" or "dev" for progress display

    Returns:
        list[dict]: Results with audio_path, ground_truth, teacher_transcript,
                    language, duration, confidence
    """
    results = []
    start_total = time.time()

    for i, clip in enumerate(clips):
        start = time.time()

        teacher_transcript, confidence = transcribe_clip_with_confidence(
            model, clip["audio_path"], clip["language"]
        )
        elapsed = time.time() - start

        results.append({
            "audio_path": clip["audio_path"],
            "ground_truth": clip["transcript"],
            "teacher_transcript": teacher_transcript,
            "language": clip["language"],
            "duration": clip["duration"],
            "confidence": round(confidence, 4),  # NEW: save confidence score
        })

        # Progress every 10 clips
        if (i + 1) % 10 == 0 or (i + 1) == len(clips):
            eta = (elapsed * (len(clips) - i - 1)) / 60
            print(
                f"    [{label}] {i+1}/{len(clips)} clips | "
                f"last: {elapsed:.1f}s | conf: {confidence:.3f} | "
                f"ETA: {eta:.0f} min",
                flush=True,
            )

    total_time = time.time() - start_total
    return results, total_time


def save_results(results, output_csv):
    """Save transcription results to CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path", "ground_truth", "teacher_transcript",
                "language", "duration", "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def print_confidence_stats(results, label):
    """Print confidence distribution statistics."""
    confs = [r["confidence"] for r in results]
    confs_arr = np.array(confs)

    print(f"\n  [{label}] Confidence statistics:")
    print(f"    Mean:   {confs_arr.mean():.3f}")
    print(f"    Median: {np.median(confs_arr):.3f}")
    print(f"    Min:    {confs_arr.min():.3f}")
    print(f"    Max:    {confs_arr.max():.3f}")

    # Show distribution in buckets
    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9]:
        count = (confs_arr >= threshold).sum()
        pct = count / len(confs_arr) * 100
        print(f"    >= {threshold}: {count}/{len(confs_arr)} ({pct:.1f}%)")

    # How many would be filtered at different thresholds
    print(f"\n  [{label}] Filtering impact:")
    for threshold in [0.3, 0.5, 0.7]:
        kept = (confs_arr >= threshold).sum()
        removed = len(confs_arr) - kept
        print(f"    threshold={threshold}: keep {kept}, remove {removed} ({removed/len(confs_arr)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher transcripts with confidence scores"
    )
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Limit number of clips (for quick testing)",
    )
    args = parser.parse_args()

    # Load training clips (Hindi + Marathi only)
    train_clips = load_clips(TRAIN_CSV, max_clips=args.max_clips)
    dev_clips = load_clips(DEV_CSV, max_clips=args.max_clips)
    print(f"Loaded {len(train_clips)} train + {len(dev_clips)} dev Hindi/Marathi clips")

    lang_counts = {}
    for c in train_clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} train clips")

    # Load teacher model
    model = load_indicconformer()

    # Transcribe train clips
    print(f"\n{'='*60}")
    print(f"  Transcribing {len(train_clips)} TRAIN clips...")
    print(f"{'='*60}")
    train_results, train_time = process_clips(train_clips, model, "train")

    # Transcribe dev clips (we need teacher transcripts for dev too,
    # so we can compute dev loss with teacher pseudo-labels)
    print(f"\n{'='*60}")
    print(f"  Transcribing {len(dev_clips)} DEV clips...")
    print(f"{'='*60}")
    dev_results, dev_time = process_clips(dev_clips, model, "dev")

    # Save results
    save_results(train_results, OUTPUT_TRAIN_CSV)
    save_results(dev_results, OUTPUT_DEV_CSV)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DONE: Teacher transcripts + confidence generated")
    print(f"{'='*60}")
    print(f"  Train clips:  {len(train_results)} ({train_time:.0f}s)")
    print(f"  Dev clips:    {len(dev_results)} ({dev_time:.0f}s)")
    print(f"  Train output: {OUTPUT_TRAIN_CSV}")
    print(f"  Dev output:   {OUTPUT_DEV_CSV}")

    # Confidence statistics
    print_confidence_stats(train_results, "train")
    print_confidence_stats(dev_results, "dev")

    # Show a few examples
    print(f"\n  Sample predictions:")
    for r in train_results[:3]:
        print(f"    [{r['language']}] confidence={r['confidence']:.3f}")
        print(f"    Ground truth: {r['ground_truth'][:60]}...")
        print(f"    Teacher pred: {r['teacher_transcript'][:60]}...")
        print()

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
