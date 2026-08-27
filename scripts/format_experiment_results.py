"""
Format Experiment Results
=========================
Reads the raw prediction CSVs from benchmarks/ and generates
clean, readable experiment files in experiments/ folder.

For each model, creates:
  1. predictions_all.txt         — every clip: GT vs Prediction
  2. predictions_mismatches.txt  — only clips where GT ≠ Prediction (with diff)
  3. results_summary.txt         — WER per language + overall stats

Matches the format used in Kid_Whisper_ASR/experiments/.

Usage:
    python scripts/format_experiment_results.py
"""

import csv
import json
import os
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG — which models to process
# ─────────────────────────────────────────────
MODELS = [
    {
        "name": "whisper_small",
        "display_name": "Whisper Small (244M)",
        "predictions_csv": "benchmarks/whisper-small_predictions.csv",
        "results_json": "benchmarks/whisper-small_results.json",
    },
    {
        "name": "indicconformer_600m",
        "display_name": "IndicConformer 600M",
        "predictions_csv": "benchmarks/indicconformer_predictions.csv",
        "results_json": "benchmarks/indicconformer_results.json",
    },
    {
        "name": "kid_whisper_medium_myst",
        "display_name": "Kid-Whisper Medium MyST (769M)",
        "predictions_csv": "benchmarks/kid-whisper-myst_predictions.csv",
        "results_json": "benchmarks/kid-whisper-myst_results.json",
    },
    {
        "name": "kid_whisper_medium_myst_cslu",
        "display_name": "Kid-Whisper Medium MyST+CSLU (769M)",
        "predictions_csv": "benchmarks/kid-whisper-myst-cslu_predictions.csv",
        "results_json": "benchmarks/kid-whisper-myst-cslu_results.json",
    },
]


def normalize_text(text):
    """Same normalization used during WER computation."""
    text = text.strip().lower()
    for ch in ".,!?;:\"'()[]{}—–-।":
        text = text.replace(ch, "")
    text = " ".join(text.split())
    return text


def compute_diff(gt_words, pred_words):
    """
    Find which words were added/removed between ground truth and prediction.
    Uses simple set difference — not perfect but gives a quick overview.
    """
    gt_set = {}
    pred_set = {}

    # Count word frequencies
    for w in gt_words:
        gt_set[w] = gt_set.get(w, 0) + 1
    for w in pred_words:
        pred_set[w] = pred_set.get(w, 0) + 1

    # Words in prediction but not in GT (added)
    added = []
    for w, count in pred_set.items():
        gt_count = gt_set.get(w, 0)
        if count > gt_count:
            added.extend([w] * (count - gt_count))

    # Words in GT but not in prediction (removed)
    removed = []
    for w, count in gt_set.items():
        pred_count = pred_set.get(w, 0)
        if count > pred_count:
            removed.extend([w] * (count - pred_count))

    return added, removed


def format_model(model_config):
    """Generate all output files for one model."""
    name = model_config["name"]
    display = model_config["display_name"]
    csv_path = model_config["predictions_csv"]
    json_path = model_config["results_json"]

    output_dir = f"experiments/{name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load results summary
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Load per-clip predictions
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(row)

    overall_wer = results["Overall"]["wer"] * 100

    # ─── File 1: All predictions ───
    all_path = os.path.join(output_dir, "predictions_all.txt")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write(f"# {display} — All Predictions\n")
        f.write(f"# Overall WER: {overall_wer:.2f}%\n")
        f.write(f"# Test set: {len(clips)} clips\n")
        f.write(f"# Date: 2026-08-25\n\n")

        for lang in ["Hindi", "Marathi", "English"]:
            lang_clips = [c for c in clips if c["language"] == lang]
            if not lang_clips:
                continue

            lang_wer = results.get(lang, {}).get("wer", 0) * 100
            f.write(f"{'='*80}\n")
            f.write(f"  {lang} — {len(lang_clips)} clips — WER: {lang_wer:.2f}%\n")
            f.write(f"{'='*80}\n\n")

            for clip in lang_clips:
                # Extract just the filename from full path
                filename = Path(clip["audio_path"]).name
                clip_wer = float(clip["wer"]) * 100

                f.write(f"File: {filename}\n")
                f.write(f"WER:  {clip_wer:.1f}%\n")
                f.write(f"GT:   {clip['ground_truth']}\n")
                f.write(f"PRED: {clip['prediction']}\n")
                f.write(f"{'-'*80}\n")

    # ─── File 2: Mismatches only ───
    mismatch_path = os.path.join(output_dir, "predictions_mismatches.txt")
    with open(mismatch_path, "w", encoding="utf-8") as f:
        f.write(f"# {display} — Mismatches Only\n")
        f.write(f"# Overall WER: {overall_wer:.2f}%\n")
        f.write(f"# Only showing clips where prediction ≠ ground truth\n\n")

        total_mismatches = 0

        for lang in ["Hindi", "Marathi", "English"]:
            lang_clips = [c for c in clips if c["language"] == lang]
            if not lang_clips:
                continue

            # Filter to mismatches only
            mismatches = []
            for clip in lang_clips:
                gt_norm = normalize_text(clip["ground_truth"])
                pred_norm = normalize_text(clip["prediction"])
                if gt_norm != pred_norm:
                    mismatches.append(clip)

            total_mismatches += len(mismatches)
            lang_wer = results.get(lang, {}).get("wer", 0) * 100

            f.write(f"{'='*80}\n")
            f.write(f"  {lang} — {len(mismatches)}/{len(lang_clips)} mismatches — WER: {lang_wer:.2f}%\n")
            f.write(f"{'='*80}\n\n")

            for clip in mismatches:
                filename = Path(clip["audio_path"]).name
                clip_wer = float(clip["wer"]) * 100

                # Compute word-level diff
                gt_words = normalize_text(clip["ground_truth"]).split()
                pred_words = normalize_text(clip["prediction"]).split()
                added, removed = compute_diff(gt_words, pred_words)

                f.write(f"File: {filename}\n")
                f.write(f"WER:  {clip_wer:.1f}%\n")
                f.write(f"GT:   {clip['ground_truth']}\n")
                f.write(f"PRED: {clip['prediction']}\n")
                if added:
                    f.write(f"+ Added:   {added}\n")
                if removed:
                    f.write(f"- Removed: {removed}\n")
                f.write(f"{'-'*80}\n")

        f.write(f"\nTotal mismatches: {total_mismatches}/{len(clips)}\n")

    # ─── File 3: Results summary ───
    summary_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# {display}\n")
        f.write(f"# Model ID: {results.get('model_id', 'N/A')}\n")
        f.write(f"# Date: 2026-08-25\n")
        f.write(f"# Test set: ASER-Dataset/splits/asr_test.csv\n\n")

        f.write(f"{'='*50}\n")
        f.write(f"  {'Language':<12} {'Clips':>6} {'WER':>10}\n")
        f.write(f"  {'-'*35}\n")

        for lang in ["Hindi", "Marathi", "English"]:
            if lang in results:
                r = results[lang]
                wer_pct = r["wer"] * 100
                f.write(f"  {lang:<12} {r['num_clips']:>6} {wer_pct:>9.2f}%\n")

        o = results["Overall"]
        f.write(f"  {'-'*35}\n")
        f.write(f"  {'Overall':<12} {o['num_clips']:>6} {o['wer']*100:>9.2f}%\n")
        f.write(f"{'='*50}\n\n")

        f.write(f"  Total inference time: {results['total_time']:.0f}s ({results['total_time']/60:.1f} min)\n")
        f.write(f"  Avg time per clip:    {o['avg_time_per_clip']:.2f}s\n")
        f.write(f"  GPU memory peak:      {results.get('gpu_memory_mb', 'N/A')} MB\n")

    print(f"  {name}/")
    print(f"    predictions_all.txt         ({len(clips)} clips)")
    print(f"    predictions_mismatches.txt")
    print(f"    results_summary.txt")


def write_comparison_table():
    """Write a comparison table across all models."""
    output_path = "experiments/COMPARISON.txt"

    # Load all results
    all_results = {}
    for m in MODELS:
        with open(m["results_json"], "r", encoding="utf-8") as f:
            all_results[m["display_name"]] = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ASR Model Comparison on ASER Test Set\n")
        f.write("# Date: 2026-08-25\n")
        f.write("# Test set: 1695 clips (782 Hindi, 366 Marathi, 547 English)\n\n")

        f.write(f"{'='*85}\n")
        f.write(f"  {'Model':<40} {'Hindi':>10} {'Marathi':>10} {'English':>10} {'Overall':>10}\n")
        f.write(f"  {'-'*75}\n")

        for name, res in all_results.items():
            h = res.get("Hindi", {}).get("wer", float("nan")) * 100
            m = res.get("Marathi", {}).get("wer", float("nan")) * 100
            e = res.get("English", {}).get("wer", float("nan")) * 100
            o = res.get("Overall", {}).get("wer", float("nan")) * 100

            # Mark N/A for IndicConformer English
            e_str = "N/A" if "IndicConformer" in name else f"{e:.2f}%"

            f.write(f"  {name:<40} {h:>9.2f}% {m:>9.2f}% {e_str:>10} {o:>9.2f}%\n")

        f.write(f"{'='*85}\n\n")

        f.write("# Role Assignment (decided by benchmark data)\n\n")
        f.write("  Semantic Teacher : IndicConformer 600M\n")
        f.write("                     Best Hindi (39.01%) & Marathi (44.61%)\n")
        f.write("                     Trained on Indian languages (22 languages)\n\n")
        f.write("  Acoustic Teacher : Kid-Whisper Medium MyST\n")
        f.write("                     Best English children's speech (37.35%)\n")
        f.write("                     Fine-tuned on MyST children's speech dataset\n\n")
        f.write("  Student          : Whisper Small\n")
        f.write("                     General-purpose baseline, room to improve on all languages\n")
        f.write("                     Hindi: 161.81% → target closer to 39.01%\n")
        f.write("                     Marathi: 170.78% → target closer to 44.61%\n")
        f.write("                     English: 47.42% → target closer to 37.35%\n")

    print(f"\n  COMPARISON.txt")


def main():
    print("Generating formatted experiment results...\n")

    for model_config in MODELS:
        format_model(model_config)
        print()

    write_comparison_table()

    print(f"\nDone! Results in experiments/")


if __name__ == "__main__":
    main()
