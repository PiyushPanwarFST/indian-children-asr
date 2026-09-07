"""
Step 3: Evaluate WER — Does Acoustic Training Actually Improve Transcription?
==============================================================================

PURPOSE:
    Training loss decreasing does NOT prove the model is better at ASR.
    Loss only shows: "student encoder features are closer to teacher features."
    WER shows: "can the student actually transcribe children's speech better?"

    Professor wants proof: compare WER BEFORE and AFTER acoustic training.

HOW IT WORKS:
    1. Load ORIGINAL Whisper Small (untrained) → run on test clips → compute WER
    2. Load our TRAINED encoder → plug into Whisper Small → run on same clips → compute WER
    3. Compare: if WER goes DOWN, acoustic training helped.

    For step 2, we:
        - Load fresh full Whisper model (encoder + decoder)
        - Replace encoder weights with our trained encoder
        - Decoder is ORIGINAL pretrained — never modified
        - This is standard practice (Distil-Whisper does the same)

Usage:
    python scripts/acoustic/step3_evaluate_wer.py
    python scripts/acoustic/step3_evaluate_wer.py --max_clips 100
    python scripts/acoustic/step3_evaluate_wer.py --checkpoint checkpoints/acoustic/best_dev_model.pt
"""

import argparse
import csv
import os
import time

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as jiwer_wer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEST_CSV = "ASER-Dataset/splits/asr_test.csv"
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"
CHECKPOINT_DIR = "checkpoints/acoustic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30
STUDENT_MODEL = "openai/whisper-small"


def load_test_clips(csv_path, max_clips=None, languages=None):
    """
    Load test clips for WER evaluation.

    Args:
        csv_path: Path to test CSV
        max_clips: Limit clips for quick testing
        languages: List of languages to include. None = all.
    """
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if languages and row["language"] not in languages:
                continue
            clips.append({
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],
                "language": row["language"],
                "duration": float(row["duration_sec"]),
            })

    clips = [c for c in clips if c["duration"] <= MAX_DURATION]

    if max_clips:
        clips = clips[:max_clips]

    return clips


def normalize_text(text):
    """Normalize text for WER: lowercase, remove punctuation."""
    text = text.strip().lower()
    for ch in ".,!?;:\"'()[]{}—–-।॥":
        text = text.replace(ch, "")
    return " ".join(text.split())


def transcribe_clip(model, processor, audio_path, language):
    """
    Transcribe one audio clip using Whisper's full pipeline (encoder + decoder).

    Args:
        model: WhisperForConditionalGeneration (full model with decoder)
        processor: WhisperProcessor
        audio_path: Path to .wav file
        language: "Hindi", "Marathi", or "English"

    Returns:
        str: Predicted transcript
    """
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    audio_np = wav.squeeze(0).numpy()

    # Create mel spectrogram
    inputs = processor.feature_extractor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs.input_features.to(DEVICE)

    # Language mapping for Whisper's forced decoder
    lang_map = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
    lang_code = lang_map.get(language, "hi")

    # Generate with Whisper's decoder (autoregressive)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang_code, task="transcribe"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    # Decode token IDs → text
    transcript = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return transcript


def load_trained_model(checkpoint_path, processor):
    """
    Load full Whisper model and replace encoder with our trained encoder.

    Steps:
        1. Load fresh Whisper Small (encoder + decoder)
        2. Load our checkpoint (trained encoder + projection layer)
        3. Copy trained encoder weights into Whisper
        4. Decoder stays original/untrained

    Args:
        checkpoint_path: Path to .pt checkpoint from acoustic training
        processor: WhisperProcessor

    Returns:
        WhisperForConditionalGeneration with trained encoder
    """
    # Load fresh full Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    model.to(DEVICE)

    # Load our trained checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    trained_state = checkpoint["model_state_dict"]

    # Our checkpoint has keys like:
    #   "encoder.layers.0.self_attn.k_proj.weight"
    #   "projection.weight"  ← we don't need this for inference
    #   "dropout...."        ← we don't need this either
    #
    # Whisper model has keys like:
    #   "model.encoder.layers.0.self_attn.k_proj.weight"
    #
    # So we need to map: "encoder.X" → "model.encoder.X"

    whisper_state = model.state_dict()
    updated = 0

    for key, value in trained_state.items():
        # Skip projection layer and dropout (not part of Whisper)
        if key.startswith("projection") or key.startswith("dropout"):
            continue

        # Map our key to Whisper's key format
        whisper_key = f"model.{key}"

        if whisper_key in whisper_state:
            if whisper_state[whisper_key].shape == value.shape:
                whisper_state[whisper_key] = value
                updated += 1

    model.load_state_dict(whisper_state)
    print(f"  Loaded {updated} encoder weight tensors from checkpoint")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Checkpoint train loss: {checkpoint.get('train_loss', '?'):.6f}")
    print(f"  Checkpoint dev loss: {checkpoint.get('dev_loss', '?'):.6f}")

    return model


def evaluate_wer(model, processor, clips, label):
    """
    Run model on clips and compute WER.

    Args:
        model: WhisperForConditionalGeneration
        processor: WhisperProcessor
        clips: List of test clips
        label: "ORIGINAL" or "TRAINED" for display

    Returns:
        dict: Results with overall WER and per-language WER
    """
    model.eval()

    all_predictions = []
    all_references = []
    per_language = {}
    errors = 0

    print(f"\n  [{label}] Evaluating {len(clips)} clips...")
    start_time = time.time()

    for i, clip in enumerate(clips):
        try:
            predicted = transcribe_clip(model, processor, clip["audio_path"], clip["language"])
        except Exception as e:
            print(f"    Error on {clip['audio_path']}: {e}")
            errors += 1
            continue

        pred_norm = normalize_text(predicted)
        ref_norm = normalize_text(clip["transcript"])

        if not ref_norm:
            continue

        all_predictions.append(pred_norm)
        all_references.append(ref_norm)

        # Per-language tracking
        lang = clip["language"]
        if lang not in per_language:
            per_language[lang] = {"predictions": [], "references": []}
        per_language[lang]["predictions"].append(pred_norm)
        per_language[lang]["references"].append(ref_norm)

        # Progress
        if (i + 1) % 20 == 0 or (i + 1) == len(clips):
            elapsed = time.time() - start_time
            clip_wer = jiwer_wer(ref_norm, pred_norm) if ref_norm else 0
            print(
                f"    [{label}] {i+1}/{len(clips)} | "
                f"clip WER: {clip_wer:.2%} | "
                f"time: {elapsed:.0f}s",
                flush=True,
            )

    total_time = time.time() - start_time

    # Compute overall WER
    if all_references:
        overall_wer = jiwer_wer(all_references, all_predictions)
    else:
        overall_wer = float("inf")

    # Compute per-language WER
    lang_wers = {}
    for lang, data in per_language.items():
        if data["references"]:
            lang_wers[lang] = {
                "wer": jiwer_wer(data["references"], data["predictions"]),
                "clips": len(data["references"]),
            }

    results = {
        "label": label,
        "overall_wer": overall_wer,
        "total_clips": len(all_predictions),
        "errors": errors,
        "time": total_time,
        "per_language": lang_wers,
    }

    return results


def print_results(original_results, trained_results):
    """Print comparison table."""
    print(f"\n{'='*65}")
    print(f"  WER COMPARISON: Original vs Acoustic-Trained Whisper Small")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<25} {'Original':>15} {'Trained':>15} {'Change':>10}")
    print(f"  {'─'*65}")

    o_wer = original_results["overall_wer"]
    t_wer = trained_results["overall_wer"]
    change = t_wer - o_wer
    direction = "↓ better" if change < 0 else "↑ worse" if change > 0 else "= same"

    print(f"  {'Overall WER':<25} {o_wer:>14.2%} {t_wer:>14.2%} {change:>+9.2%} {direction}")

    # Per-language
    all_langs = set(list(original_results["per_language"].keys()) +
                    list(trained_results["per_language"].keys()))

    for lang in sorted(all_langs):
        o_lang = original_results["per_language"].get(lang, {})
        t_lang = trained_results["per_language"].get(lang, {})
        o_w = o_lang.get("wer", float("nan"))
        t_w = t_lang.get("wer", float("nan"))
        o_c = o_lang.get("clips", 0)
        change = t_w - o_w
        direction = "↓" if change < 0 else "↑" if change > 0 else "="

        print(f"  {f'{lang} ({o_c} clips)':<25} {o_w:>14.2%} {t_w:>14.2%} {change:>+9.2%} {direction}")

    print(f"\n  {'Time':<25} {original_results['time']:>13.0f}s {trained_results['time']:>13.0f}s")
    print(f"  {'Errors':<25} {original_results['errors']:>15} {trained_results['errors']:>15}")
    print(f"{'='*65}")

    if t_wer < o_wer:
        print(f"\n  ✓ Acoustic training IMPROVED WER by {abs(change):.2%}")
    elif t_wer > o_wer:
        print(f"\n  ✗ Acoustic training WORSENED WER by {abs(change):.2%}")
    else:
        print(f"\n  = No change in WER")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER: Original vs Trained Whisper")
    parser.add_argument("--max_clips", type=int, default=None, help="Limit test clips")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (default: best_dev_model.pt)")
    parser.add_argument("--eval_set", type=str, default="dev", choices=["dev", "test"],
                        help="Which set to evaluate on")
    parser.add_argument("--languages", type=str, nargs="+", default=None,
                        help="Languages to evaluate (default: all)")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_dev_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run step2_acoustic_distillation.py first.")
        return

    # Load test clips
    eval_csv = DEV_CSV if args.eval_set == "dev" else TEST_CSV
    clips = load_test_clips(eval_csv, max_clips=args.max_clips, languages=args.languages)
    print(f"Loaded {len(clips)} {args.eval_set} clips for evaluation")

    lang_counts = {}
    for c in clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} clips")

    # Load processor
    print("\nLoading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    # ── Evaluate ORIGINAL Whisper Small ──
    print("\n" + "="*50)
    print("  STEP 1: Evaluate ORIGINAL Whisper Small")
    print("="*50)
    original_model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    original_model.to(DEVICE)
    original_model.eval()

    original_results = evaluate_wer(original_model, processor, clips, "ORIGINAL")

    # Free memory
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Evaluate TRAINED Whisper Small ──
    print("\n" + "="*50)
    print(f"  STEP 2: Evaluate TRAINED Whisper Small")
    print(f"  Checkpoint: {checkpoint_path}")
    print("="*50)
    trained_model = load_trained_model(checkpoint_path, processor)
    trained_model.eval()

    trained_results = evaluate_wer(trained_model, processor, clips, "TRAINED")

    # ── Print Comparison ──
    print_results(original_results, trained_results)


if __name__ == "__main__":
    main()
