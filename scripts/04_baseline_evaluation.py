"""
Script 4: Baseline Evaluation — Zero-Shot ASR on ASER Test Set

What this script does:
  Runs 2 pretrained ASR models DIRECTLY on our asr_test.csv (no training).
  Computes Word Error Rate (WER) for each model on each language.
  This gives us the "before" numbers — how well existing models perform
  on Indian children's speech WITHOUT any of our Phase 1/2 training.

Why we need baselines:
  Our paper's contribution = (Phase 1+2 WER) vs (Baseline WER).
  If baselines already work well → our contribution is small.
  If baselines fail badly → our method provides clear improvement.

The 2 baseline models:
  1. Whisper Small (OpenAI) — general multilingual ASR, 99 languages
     Tests: "Does a general adult ASR work on Indian children?"

  2. Kid-Whisper Small-EN (aadel4/kid-whisper-small-en-myst)
     English-only Whisper Small fine-tuned on MyST children's speech (125h)
     Best Kid-Whisper variant: 9.11% WER on MyST test (vs 11.80% multilingual)
     Tests: "Does a children's model work on Indian languages?"

Expected results: BOTH should have HIGH WER because:
  - Whisper: trained on adults, not children
  - Kid-Whisper: trained on English children, not Indian languages

Input:  ASER-Dataset/splits/asr_test.csv
Output: ASER-Dataset/baseline_results.csv + printed WER tables
"""

import csv
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from collections import defaultdict
from jiwer import wer as compute_wer

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TEST_CSV     = ASER_ROOT / "splits" / "asr_test.csv"
RESULTS_CSV  = ASER_ROOT / "baseline_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load test set
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading test set...")
print("=" * 70)

with open(TEST_CSV, encoding="utf-8") as f:
    test_clips = list(csv.DictReader(f))

# Separate by language
hindi_clips   = [r for r in test_clips if r["language"] == "Hindi"]
marathi_clips = [r for r in test_clips if r["language"] == "Marathi"]
english_clips = [r for r in test_clips if r["language"] == "English"]

total_hrs = sum(float(r["duration_sec"]) for r in test_clips) / 3600

print(f"Total test clips: {len(test_clips)} ({total_hrs:.2f}h)")
print(f"  Hindi:   {len(hindi_clips)} clips")
print(f"  Marathi: {len(marathi_clips)} clips")
print(f"  English: {len(english_clips)} clips")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_16k(path):
    """Load audio file and resample to 16kHz mono."""
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze().numpy()


def calculate_wer_safe(reference, hypothesis):
    """
    Calculate WER between reference and hypothesis.
    Handles edge cases: empty strings, None values.
    Returns WER as a float (0.0 to 1.0+).
    """
    ref = reference.strip() if reference else ""
    hyp = hypothesis.strip() if hypothesis else ""

    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0  # model produced nothing → 100% error

    try:
        return compute_wer(ref, hyp)
    except Exception:
        return 1.0


def evaluate_model(clips, predict_fn, model_name, lang_name):
    """
    Run a prediction function on a list of clips.
    Returns list of result dicts with reference, hypothesis, wer.
    """
    results = []
    errors = 0

    for i, clip in enumerate(clips):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"    [{i+1}/{len(clips)}] {model_name} on {lang_name}...")

        try:
            audio = load_audio_16k(clip["audio_path"])
            hypothesis = predict_fn(audio)
        except Exception as e:
            hypothesis = ""
            errors += 1

        reference = clip["transcript"]
        clip_wer = calculate_wer_safe(reference, hypothesis)

        results.append({
            "audio_path": clip["audio_path"],
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": clip_wer,
            "language": clip["language"],
            "model": model_name,
            "duration_sec": clip["duration_sec"],
        })

    if errors:
        print(f"    Errors: {errors}/{len(clips)}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Baseline 1 — Whisper Small (multilingual)
# ══════════════════════════════════════════════════════════════════════════════
#
# OpenAI Whisper (Radford et al., 2023)
# - Encoder-Decoder transformer trained on 680,000 hours of web audio
# - Supports 99 languages including Hindi, Marathi, English
# - 244M parameters (Small variant)
# - Takes raw audio → outputs text directly
# - Trained on ADULT speech from the internet
#
# Why it should struggle on our data:
# - Never saw children's speech during training
# - Children have higher pitch, different rhythm, mispronunciations
# - Indian children reading aloud ≠ internet podcast audio
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Baseline 1 — Whisper Small (OpenAI)")
print("=" * 70)

import whisper

whisper_model = whisper.load_model("small", device=DEVICE)
print(f"Whisper Small loaded. Params: {sum(p.numel() for p in whisper_model.parameters()):,}")

WHISPER_LANG = {"Hindi": "hi", "Marathi": "mr", "English": "en"}


def predict_whisper(audio):
    """Run Whisper inference on a numpy audio array."""
    audio_padded = whisper.pad_or_trim(audio.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio_padded).to(DEVICE)
    options = whisper.DecodingOptions(language=predict_whisper._lang, without_timestamps=True)
    result = whisper.decode(whisper_model, mel, options)
    return result.text


all_whisper_results = []

for lang, clips in [("Hindi", hindi_clips), ("Marathi", marathi_clips), ("English", english_clips)]:
    print(f"\n  Running Whisper on {lang} ({len(clips)} clips)...")
    predict_whisper._lang = WHISPER_LANG[lang]
    t0 = time.time()
    results = evaluate_model(clips, predict_whisper, "Whisper-Small", lang)
    elapsed = time.time() - t0
    all_whisper_results.extend(results)
    avg_wer = np.mean([r["wer"] for r in results])
    print(f"  {lang} done in {elapsed:.0f}s | Avg WER: {avg_wer:.2%}")

del whisper_model
torch.cuda.empty_cache() if DEVICE == "cuda" else None
print()


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Baseline 2 — Kid-Whisper Small-EN
# ══════════════════════════════════════════════════════════════════════════════
#
# Kid-Whisper (Attia et al., 2024)
# - Base: openai/whisper-small.en (English-only Whisper, NOT multilingual)
# - Fine-tuned on MyST dataset (125h of American English children, ages 8-11)
# - 244M parameters
# - MyST WER: 9.11% (best among all Kid-Whisper variants)
#
# Why English-only base is better than multilingual base:
# - Both are fine-tuned on English-only MyST data
# - English-only base concentrates all 244M params on English
# - Multilingual base splits capacity across 99 languages → worse (11.80%)
# - After MyST fine-tuning, multilingual's non-English knowledge is gone anyway
#
# Why it should struggle on our data:
# - MyST = English-only → cannot transcribe Hindi/Marathi at all
# - Even for English: MyST = American children, ours = Indian-accented English
# - This proves: children's model alone is not enough for Indian languages
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Baseline 2 — Kid-Whisper Small-EN (MyST)")
print("=" * 70)

from transformers import WhisperForConditionalGeneration, WhisperProcessor

KW_MODEL_ID = "aadel4/kid-whisper-small-en-myst"
# IMPORTANT: Kid-Whisper's own tokenizer has vocab_size=0 (broken upload).
# Use the base model's processor (openai/whisper-small.en) for correct decoding.
# The model weights are fine — only the tokenizer is broken in the aadel4 repo.
kw_processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
kw_model = WhisperForConditionalGeneration.from_pretrained(KW_MODEL_ID).to(DEVICE)
kw_model.eval()
print(f"Kid-Whisper loaded ({KW_MODEL_ID}). Params: {sum(p.numel() for p in kw_model.parameters()):,}")
print(f"  Tokenizer: openai/whisper-small.en (base model — aadel4 tokenizer has vocab_size=0)")


def predict_kid_whisper(audio):
    """Run Kid-Whisper inference on a numpy audio array."""
    input_features = kw_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = kw_model.generate(input_features, max_new_tokens=225)

    transcription = kw_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


all_kw_results = []

for lang, clips in [("Hindi", hindi_clips), ("Marathi", marathi_clips), ("English", english_clips)]:
    print(f"\n  Running Kid-Whisper on {lang} ({len(clips)} clips)...")
    t0 = time.time()
    results = evaluate_model(clips, predict_kid_whisper, "Kid-Whisper-EN", lang)
    all_kw_results.extend(results)
    elapsed = time.time() - t0
    avg_wer = np.mean([r["wer"] for r in results])
    print(f"  {lang} done in {elapsed:.0f}s | Avg WER: {avg_wer:.2%}")

del kw_model
torch.cuda.empty_cache() if DEVICE == "cuda" else None
print()


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Compile results and save
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Saving results...")
print("=" * 70)

all_results = all_whisper_results + all_kw_results

result_fields = ["model", "language", "audio_path", "reference", "hypothesis",
                 "wer", "duration_sec"]

with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=result_fields)
    writer.writeheader()
    writer.writerows(all_results)

print(f"Saved: {RESULTS_CSV} ({len(all_results)} rows)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Print WER tables
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  BASELINE RESULTS — Word Error Rate (WER)")
print("  Lower is better. 0% = perfect, 100% = all words wrong")
print("=" * 70)

models = ["Whisper-Small", "Kid-Whisper-EN"]
languages = ["Hindi", "Marathi", "English"]

# ── Per-clip average WER ──────────────────────────────────────────────────────
wer_lookup = defaultdict(list)
for r in all_results:
    wer_lookup[(r["model"], r["language"])].append(r["wer"])

print(f"\n  Average per-clip WER:")
print(f"  {'Model':<20}", end="")
for lang in languages:
    print(f" {lang:>10}", end="")
print(f" {'Overall':>10}")
print(f"  {'-'*20}", end="")
for _ in languages:
    print(f" {'-'*10}", end="")
print(f" {'-'*10}")

for model in models:
    print(f"  {model:<20}", end="")
    all_model_wers = []
    for lang in languages:
        wers = wer_lookup.get((model, lang), [])
        if wers:
            avg = np.mean(wers)
            print(f" {avg:>9.1%}", end="")
            all_model_wers.extend(wers)
        else:
            print(f" {'N/A':>10}", end="")
    if all_model_wers:
        print(f" {np.mean(all_model_wers):>9.1%}")
    else:
        print(f" {'N/A':>10}")

# ── Corpus-level WER (standard for ASR papers) ───────────────────────────────
print(f"\n  Corpus-level WER (all text concatenated — standard metric):")
print(f"  {'Model':<20}", end="")
for lang in languages:
    print(f" {lang:>10}", end="")
print(f" {'Overall':>10}")
print(f"  {'-'*20}", end="")
for _ in languages:
    print(f" {'-'*10}", end="")
print(f" {'-'*10}")

for model in models:
    print(f"  {model:<20}", end="")
    all_refs_all = []
    all_hyps_all = []
    for lang in languages:
        lang_results = [r for r in all_results if r["model"] == model and r["language"] == lang]
        if lang_results:
            all_refs = " ".join(r["reference"] for r in lang_results)
            all_hyps = " ".join(r["hypothesis"] for r in lang_results)
            all_refs_all.append(all_refs)
            all_hyps_all.append(all_hyps)
            if all_refs.strip() and all_hyps.strip():
                corpus_wer = compute_wer(all_refs, all_hyps)
                print(f" {corpus_wer:>9.1%}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    # Overall corpus WER
    full_ref = " ".join(all_refs_all)
    full_hyp = " ".join(all_hyps_all)
    if full_ref.strip() and full_hyp.strip():
        overall_corpus = compute_wer(full_ref, full_hyp)
        print(f" {overall_corpus:>9.1%}")
    else:
        print(f" {'N/A':>10}")

# ── Sample predictions ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  SAMPLE PREDICTIONS (first 3 per model per language)")
print("=" * 70)

for model in models:
    model_results = [r for r in all_results if r["model"] == model]
    for lang in languages:
        lang_results = [r for r in model_results if r["language"] == lang]
        if lang_results:
            print(f"\n  {model} | {lang}:")
            for r in lang_results[:3]:
                print(f"    REF: {r['reference'][:80]}")
                print(f"    HYP: {r['hypothesis'][:80]}")
                print(f"    WER: {r['wer']:.0%}")
                print()

print("=" * 70)
print("  BASELINE EVALUATION COMPLETE")
print("  Results saved to:", RESULTS_CSV)
print("=" * 70)
