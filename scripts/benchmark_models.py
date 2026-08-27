"""
Benchmark ASR Models on ASER Test Set
=====================================
Tests how well each model transcribes Indian children's speech.
Computes WER (Word Error Rate) per language and overall.

Models tested:
  1. whisper-small         → openai/whisper-small (244M) — proposed student
  2. kid-whisper-myst      → aadel4/kid-whisper-medium-en-myst (769M) — proposed acoustic teacher
  3. kid-whisper-myst-cslu → aadel4/kid-whisper-medium-en-myst_cslu (769M) — alternative acoustic teacher
  4. indicconformer        → ai4bharat/indic-conformer-600m-multilingual (600M) — proposed semantic teacher

Usage:
    python scripts/benchmark_models.py --model whisper-small
    python scripts/benchmark_models.py --model kid-whisper-myst
    python scripts/benchmark_models.py --model kid-whisper-myst-cslu
    python scripts/benchmark_models.py --model indicconformer
    python scripts/benchmark_models.py --model all
    python scripts/benchmark_models.py --model whisper-small --max_clips 10   # quick test
"""

import argparse
import csv
import json
import os
import time

import torch
import torchaudio
from jiwer import wer as compute_wer_score


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEST_CSV = "ASER-Dataset/splits/asr_test.csv"
OUTPUT_DIR = "benchmarks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def load_audio(path, max_sec=MAX_DURATION):
    """Load audio file → numpy array (16kHz mono)."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_samples = max_sec * SAMPLE_RATE
    wav = wav[:, :max_samples]
    return wav.squeeze(0).numpy()


def load_test_data(csv_path, max_clips=None):
    """Load test clips from CSV."""
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    """Basic text normalization for WER computation."""
    text = text.strip().lower()
    for ch in ".,!?;:\"'()[]{}—–-।":
        text = text.replace(ch, "")
    text = " ".join(text.split())
    return text


def compute_wer(predicted, reference):
    """Compute Word Error Rate between two strings."""
    pred = normalize_text(predicted)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
    return compute_wer_score(ref, pred)


def save_results(model_name, results, clips):
    """Save benchmark results to JSON and per-clip CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save summary
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Save per-clip predictions
    detail_path = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv")
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "language", "ground_truth", "prediction", "wer"])
        for clip in clips:
            if "prediction" in clip:
                writer.writerow([
                    clip["audio_path"],
                    clip["language"],
                    clip["transcript"],
                    clip["prediction"],
                    f"{clip['clip_wer']:.4f}",
                ])
    print(f"Per-clip predictions saved to: {detail_path}")


def print_results(model_name, results):
    """Print results in a nice table."""
    print(f"\n{'='*60}")
    print(f"  RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Language':<12} {'Clips':>6} {'WER':>8} {'Avg Time':>10}")
    print(f"  {'-'*40}")
    for lang in ["Hindi", "Marathi", "English", "Overall"]:
        if lang in results:
            r = results[lang]
            print(f"  {lang:<12} {r['num_clips']:>6} {r['wer']:.2%} {r['avg_time_per_clip']:.2f}s")
    print(f"{'='*60}")
    print(f"  Total time: {results['total_time']:.1f}s ({results['total_time']/60:.1f} min)")
    print(f"  GPU memory peak: {results.get('gpu_memory_mb', 'N/A')} MB")
    print(f"{'='*60}")


def compute_summary(lang_results):
    """Compute per-language and overall averages from lang_results dict."""
    results = {}
    all_wers = []
    all_times = []
    for lang, items in lang_results.items():
        wers = [x["wer"] for x in items]
        times = [x["time"] for x in items]
        results[lang] = {
            "num_clips": len(items),
            "wer": sum(wers) / len(wers),
            "avg_time_per_clip": sum(times) / len(times),
        }
        all_wers.extend(wers)
        all_times.extend(times)

    results["Overall"] = {
        "num_clips": len(all_wers),
        "wer": sum(all_wers) / len(all_wers),
        "avg_time_per_clip": sum(all_times) / len(all_times),
    }
    return results


# ─────────────────────────────────────────────
# WHISPER-BASED BENCHMARK (shared logic)
# ─────────────────────────────────────────────
def benchmark_whisper_model(clips, model_id, model_name, processor_id=None,
                            is_english_only=False):
    """
    Benchmark any Whisper-based model (Whisper Small, Kid-Whisper Medium, etc.)
    All Whisper models share the same inference pipeline:
      audio → mel spectrogram → encoder → decoder → text

    is_english_only: If True, skip language/task params (English-only models
                     like Kid-Whisper EN don't accept these).
    """
    print(f"\n[1/3] Loading {model_name}...")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    # Some fine-tuned models use the base processor
    if processor_id is None:
        processor_id = model_id

    processor = WhisperProcessor.from_pretrained(processor_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()

    mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"    Model loaded. GPU memory: {mem:.0f} MB")
    if is_english_only:
        print(f"    Note: English-only model — will transcribe ALL clips as English")
    print(f"[2/3] Transcribing {len(clips)} clips...")

    lang_results = {}
    start_total = time.time()

    for i, clip in enumerate(clips):
        start = time.time()
        lang = clip["language"]

        audio = load_audio(clip["audio_path"])

        # Prepare mel spectrogram
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)

        # Generate transcript
        with torch.no_grad():
            if is_english_only:
                # English-only models don't accept language or task params
                generated_ids = model.generate(input_features)
            else:
                # Multilingual models: set language hint
                if lang == "Hindi":
                    forced_lang = "hi"
                elif lang == "Marathi":
                    forced_lang = "mr"
                else:
                    forced_lang = "en"
                generated_ids = model.generate(
                    input_features,
                    language=forced_lang,
                    task="transcribe",
                )

        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = time.time() - start

        # Compute WER
        clip_wer = compute_wer(predicted, clip["transcript"])
        clip["prediction"] = predicted
        clip["clip_wer"] = clip_wer

        if lang not in lang_results:
            lang_results[lang] = []
        lang_results[lang].append({"wer": clip_wer, "time": elapsed})

        if (i + 1) % 10 == 0 or (i + 1) == len(clips):
            avg_wer = sum(x["wer"] for v in lang_results.values() for x in v) / (i + 1)
            print(f"    {i+1}/{len(clips)} clips | avg WER: {avg_wer:.1%} | last clip: {elapsed:.2f}s", flush=True)

    total_time = time.time() - start_total
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    results = compute_summary(lang_results)
    results["total_time"] = total_time
    results["gpu_memory_mb"] = int(peak_mem)
    results["model_name"] = model_name
    results["model_id"] = model_id

    del model, processor
    torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────
# MODEL 1: WHISPER SMALL (proposed student)
# ─────────────────────────────────────────────
def benchmark_whisper_small(clips):
    """
    openai/whisper-small — 244M params
    General-purpose ASR, 99 languages including Hindi, Marathi, English.
    This is our proposed STUDENT model.
    """
    return benchmark_whisper_model(
        clips,
        model_id="openai/whisper-small",
        model_name="whisper-small",
    )


# ─────────────────────────────────────────────
# MODEL 2: KID-WHISPER MEDIUM (MyST only)
# ─────────────────────────────────────────────
def benchmark_kid_whisper_myst(clips):
    """
    aadel4/kid-whisper-medium-en-myst — 769M params
    Whisper Medium fine-tuned on MyST children's speech dataset.
    This is our proposed ACOUSTIC TEACHER.
    English-only model — cannot set language/task params.
    Uses base whisper-medium processor (same tokenizer/feature extractor).
    """
    return benchmark_whisper_model(
        clips,
        model_id="aadel4/kid-whisper-medium-en-myst",
        model_name="kid-whisper-medium-myst",
        processor_id="openai/whisper-medium.en",
        is_english_only=True,
    )


# ─────────────────────────────────────────────
# MODEL 3: KID-WHISPER MEDIUM (MyST + CSLU)
# ─────────────────────────────────────────────
def benchmark_kid_whisper_myst_cslu(clips):
    """
    aadel4/kid-whisper-medium-en-myst_cslu — 769M params
    Whisper Medium fine-tuned on MyST + CSLU children's speech datasets.
    More children's data = broader generalization.
    English-only model — cannot set language/task params.
    Uses base whisper-medium.en processor (English-only tokenizer).
    """
    return benchmark_whisper_model(
        clips,
        model_id="aadel4/kid-whisper-medium-en-myst_cslu",
        model_name="kid-whisper-medium-myst-cslu",
        processor_id="openai/whisper-medium.en",
        is_english_only=True,
    )


# ─────────────────────────────────────────────
# MODEL 4: INDICCONFORMER 600M (proposed semantic teacher)
# ─────────────────────────────────────────────
def benchmark_indicconformer(clips):
    """
    ai4bharat/indic-conformer-600m-multilingual — 600M params
    Conformer-based CTC+RNNT ASR for 22 Indian languages.
    This is our proposed SEMANTIC TEACHER.
    Uses HuggingFace with trust_remote_code=True (ONNX-based inference).
    Requires: pip install onnxruntime
    """
    print("\n[1/3] Loading IndicConformer 600M...")

    try:
        import onnxruntime
    except ImportError:
        print("    ERROR: onnxruntime is not installed.")
        print("    Install with: pip install onnxruntime")
        return None

    from transformers import AutoModel

    model_id = "ai4bharat/indic-conformer-600m-multilingual"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    print(f"    Model loaded (ONNX-based inference).")
    print(f"[2/3] Transcribing {len(clips)} clips...")

    # Language mapping: our names → IndicConformer codes
    # Only Hindi and Marathi — IndicConformer does NOT support English
    lang_map = {"Hindi": "hi", "Marathi": "mr"}

    lang_results = {}
    start_total = time.time()

    for i, clip in enumerate(clips):
        start = time.time()
        lang = clip["language"]

        # Load audio as 2D tensor (batch, samples) — IndicConformer expects this
        wav, sr = torchaudio.load(clip["audio_path"])
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        max_samples = MAX_DURATION * SAMPLE_RATE
        wav = wav[:, :max_samples]
        # Keep as 2D: shape (1, num_samples) — do NOT squeeze

        # IndicConformer language code — only supports 22 Indian languages (no English)
        ic_lang = lang_map.get(lang)
        if ic_lang is None:
            # Skip unsupported languages (e.g., English)
            elapsed = time.time() - start
            clip["prediction"] = "[UNSUPPORTED_LANGUAGE]"
            clip["clip_wer"] = 1.0
            if lang not in lang_results:
                lang_results[lang] = []
            lang_results[lang].append({"wer": 1.0, "time": elapsed})
            continue

        # Transcribe
        try:
            predicted = model.forward(wav, lang=ic_lang)
            if not isinstance(predicted, str):
                predicted = str(predicted)
        except Exception as e:
            print(f"    Error on clip {i}: {e}")
            predicted = ""

        elapsed = time.time() - start

        clip_wer = compute_wer(predicted, clip["transcript"])
        clip["prediction"] = predicted
        clip["clip_wer"] = clip_wer

        if lang not in lang_results:
            lang_results[lang] = []
        lang_results[lang].append({"wer": clip_wer, "time": elapsed})

        if (i + 1) % 10 == 0 or (i + 1) == len(clips):
            avg_wer = sum(x["wer"] for v in lang_results.values() for x in v) / (i + 1)
            print(f"    {i+1}/{len(clips)} clips | avg WER: {avg_wer:.1%} | last clip: {elapsed:.2f}s", flush=True)

    total_time = time.time() - start_total

    results = compute_summary(lang_results)
    results["total_time"] = total_time
    results["gpu_memory_mb"] = "N/A (ONNX)"
    results["model_name"] = "indicconformer-600m"
    results["model_id"] = model_id

    del model

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
AVAILABLE_MODELS = {
    "whisper-small": benchmark_whisper_small,
    "kid-whisper-myst": benchmark_kid_whisper_myst,
    "kid-whisper-myst-cslu": benchmark_kid_whisper_myst_cslu,
    "indicconformer": benchmark_indicconformer,
}

def main():
    parser = argparse.ArgumentParser(description="Benchmark ASR models on ASER test set")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(AVAILABLE_MODELS.keys()) + ["all"],
                        help="Which model to benchmark")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Limit number of clips (for quick testing)")
    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {TEST_CSV}...")
    clips = load_test_data(TEST_CSV, max_clips=args.max_clips)
    print(f"Loaded {len(clips)} clips")

    lang_counts = {}
    for c in clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} clips")

    # Run benchmarks
    if args.model == "all":
        models_to_run = list(AVAILABLE_MODELS.keys())
    else:
        models_to_run = [args.model]

    all_results = {}
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  BENCHMARKING: {model_name}")
        print(f"{'='*60}")

        model_clips = [dict(c) for c in clips]
        benchmark_fn = AVAILABLE_MODELS[model_name]
        results = benchmark_fn(model_clips)

        if results is not None:
            print_results(model_name, results)
            save_results(model_name, results, model_clips)
            all_results[model_name] = results

    # Comparison table if multiple models ran
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"  {'Model':<28} {'Hindi':>8} {'Marathi':>8} {'English':>8} {'Overall':>8}")
        print(f"  {'-'*60}")
        for name, res in all_results.items():
            h = res.get("Hindi", {}).get("wer", float("nan"))
            m = res.get("Marathi", {}).get("wer", float("nan"))
            e = res.get("English", {}).get("wer", float("nan"))
            o = res.get("Overall", {}).get("wer", float("nan"))
            print(f"  {name:<28} {h:>7.1%} {m:>7.1%} {e:>7.1%} {o:>7.1%}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
