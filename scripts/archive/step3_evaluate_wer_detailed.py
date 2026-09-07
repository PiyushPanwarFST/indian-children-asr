"""
Step 3 (Detailed): WER Evaluation with Side-by-Side Output Comparison
=====================================================================

Shows 3 outputs for each clip:
  1. Ground Truth  (from ASER dataset)
  2. Original Whisper Small output (untrained baseline)
  3. Trained Whisper Small output  (after acoustic distillation)

Saves results to a CSV file for professor review.

Usage:
    python scripts/acoustic/step3_evaluate_wer_detailed.py --max_clips 20
    python scripts/acoustic/step3_evaluate_wer_detailed.py --max_clips 50
    python scripts/acoustic/step3_evaluate_wer_detailed.py --checkpoint checkpoints/acoustic/epoch_20.pt
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
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"
CHECKPOINT_DIR = "checkpoints/acoustic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30
STUDENT_MODEL = "openai/whisper-small"
OUTPUT_DIR = "results/acoustic"


def normalize_text(text):
    """Normalize text for WER: lowercase, remove punctuation."""
    text = text.strip().lower()
    for ch in ".,!?;:\"'()[]{}—–-।॥":
        text = text.replace(ch, "")
    return " ".join(text.split())


def load_audio(audio_path):
    """Load and preprocess audio file."""
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    return wav.squeeze(0).numpy()


def transcribe(model, processor, audio_np, language):
    """Transcribe audio using Whisper's full pipeline."""
    inputs = processor.feature_extractor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs.input_features.to(DEVICE)

    lang_map = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
    lang_code = lang_map.get(language, "hi")

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang_code, task="transcribe"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def load_trained_model(checkpoint_path):
    """Load Whisper model with trained encoder weights."""
    model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    model.to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    trained_state = checkpoint["model_state_dict"]

    whisper_state = model.state_dict()
    updated = 0

    for key, value in trained_state.items():
        if key.startswith("projection") or key.startswith("dropout"):
            continue
        whisper_key = f"model.{key}"
        if whisper_key in whisper_state:
            if whisper_state[whisper_key].shape == value.shape:
                whisper_state[whisper_key] = value
                updated += 1

    model.load_state_dict(whisper_state)
    epoch = checkpoint.get("epoch", "?")
    train_loss = checkpoint.get("train_loss", 0)
    dev_loss = checkpoint.get("dev_loss", 0)
    print(f"  Loaded {updated}/187 encoder weights from epoch {epoch}")
    print(f"  Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
    return model, epoch, train_loss, dev_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_clips", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--languages", type=str, nargs="+", default=None)
    args = parser.parse_args()

    # Find checkpoint
    checkpoint_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "best_dev_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Load clips
    clips = []
    with open(DEV_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.languages and row["language"] not in args.languages:
                continue
            if float(row["duration_sec"]) <= MAX_DURATION:
                clips.append(row)
    if args.max_clips:
        clips = clips[:args.max_clips]

    print(f"Evaluating {len(clips)} dev clips")
    lang_counts = {}
    for c in clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")

    # Load processor
    processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    # Load ORIGINAL model
    print("\nLoading Original Whisper Small...")
    original_model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    original_model.to(DEVICE).eval()

    # Load TRAINED model
    print(f"\nLoading Trained model from: {checkpoint_path}")
    trained_model, ckpt_epoch, ckpt_train_loss, ckpt_dev_loss = load_trained_model(checkpoint_path)
    trained_model.eval()

    # ── Run evaluation ──
    results = []
    original_refs, original_preds = [], []
    trained_refs, trained_preds = [], []
    per_lang_original = {}
    per_lang_trained = {}

    print(f"\n{'='*80}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}\n")

    for i, clip in enumerate(clips):
        audio_path = clip["audio_path"]
        ground_truth = clip["transcript"]
        language = clip["language"]
        filename = os.path.basename(audio_path)

        try:
            audio_np = load_audio(audio_path)
            original_output = transcribe(original_model, processor, audio_np, language)
            trained_output = transcribe(trained_model, processor, audio_np, language)
        except Exception as e:
            print(f"  ERROR on {filename}: {e}")
            continue

        # Normalize for WER
        gt_norm = normalize_text(ground_truth)
        orig_norm = normalize_text(original_output)
        train_norm = normalize_text(trained_output)

        if not gt_norm:
            continue

        # Compute per-clip WER
        orig_wer = jiwer_wer(gt_norm, orig_norm)
        train_wer = jiwer_wer(gt_norm, train_norm)

        # Store for overall WER
        original_refs.append(gt_norm)
        original_preds.append(orig_norm)
        trained_refs.append(gt_norm)
        trained_preds.append(train_norm)

        # Per-language tracking
        for store, pred in [(per_lang_original, orig_norm), (per_lang_trained, train_norm)]:
            if language not in store:
                store[language] = {"refs": [], "preds": []}
            store[language]["refs"].append(gt_norm)
            store[language]["preds"].append(pred)

        # Store result
        results.append({
            "clip": filename,
            "language": language,
            "ground_truth": ground_truth,
            "original_whisper": original_output,
            "trained_whisper": trained_output,
            "original_wer": f"{orig_wer:.2%}",
            "trained_wer": f"{train_wer:.2%}",
        })

        # Print side-by-side
        change = "BETTER" if train_wer < orig_wer else "WORSE" if train_wer > orig_wer else "SAME"
        print(f"  Clip {i+1}/{len(clips)}: {filename} [{language}]")
        print(f"    Ground Truth:      {ground_truth[:100]}")
        print(f"    Original Whisper:  {original_output[:100]}")
        print(f"    Trained Whisper:   {trained_output[:100]}")
        print(f"    WER: Original={orig_wer:.0%} | Trained={train_wer:.0%} | {change}")
        print()

    # ── Overall WER ──
    overall_orig_wer = jiwer_wer(original_refs, original_preds) if original_refs else float("inf")
    overall_train_wer = jiwer_wer(trained_refs, trained_preds) if trained_refs else float("inf")
    change = overall_train_wer - overall_orig_wer

    print(f"{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Checkpoint: epoch {ckpt_epoch}, train_loss={ckpt_train_loss:.4f}, dev_loss={ckpt_dev_loss:.4f}")
    print(f"  Clips evaluated: {len(results)}")
    print(f"")
    print(f"  {'Model':<30} {'WER':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Original Whisper Small':<30} {overall_orig_wer:>9.2%}")
    print(f"  {'Trained Whisper Small':<30} {overall_train_wer:>9.2%}")
    print(f"  {'Change':<30} {change:>+9.2%} {'(WORSE)' if change > 0 else '(BETTER)' if change < 0 else ''}")

    # Per-language
    print(f"\n  Per-Language WER:")
    all_langs = sorted(set(list(per_lang_original.keys()) + list(per_lang_trained.keys())))
    for lang in all_langs:
        o_data = per_lang_original.get(lang, {"refs": [], "preds": []})
        t_data = per_lang_trained.get(lang, {"refs": [], "preds": []})
        o_wer = jiwer_wer(o_data["refs"], o_data["preds"]) if o_data["refs"] else 0
        t_wer = jiwer_wer(t_data["refs"], t_data["preds"]) if t_data["refs"] else 0
        n = len(o_data["refs"])
        print(f"    {lang} ({n} clips): Original={o_wer:.2%} | Trained={t_wer:.2%} | Change={t_wer-o_wer:+.2%}")

    # ── Save to CSV ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, f"wer_comparison_epoch{ckpt_epoch}.csv")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "language", "ground_truth", "original_whisper",
            "trained_whisper", "original_wer", "trained_wer"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Detailed results saved to: {output_csv}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
