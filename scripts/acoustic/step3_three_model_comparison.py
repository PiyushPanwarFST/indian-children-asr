"""
Step 3: Three-Model WER Comparison
====================================
PURPOSE: Compare Word Error Rate (WER) of 3 models on the SAME audio clips
         to show how acoustic distillation affected the student encoder.

Models compared:
  1. Original Whisper Small  (openai/whisper-small) — pretrained, no fine-tuning
  2. Teacher Kid-Whisper Med (aadel4/kid-whisper-medium-en-myst) — English-only children's ASR
  3. Student Trained Small   (our whisper-small with HPC-trained encoder from acoustic distillation)

HOW IT WORKS:
  - For each audio clip, all 3 models transcribe the same audio
  - We compare each transcription against the ground truth using WER
  - Results saved as .csv (raw data) and .txt (professor-friendly report)

USAGE:
  # Test on 50 clips per language from test set, using HPC checkpoint:
  python scripts/acoustic/step3_three_model_comparison.py \\
      --per_lang 50 --eval_set test --checkpoint checkpoints/acoustic/hpc_best_dev_model.pt

  # Full test set (all clips):
  python scripts/acoustic/step3_three_model_comparison.py \\
      --per_lang 9999 --eval_set test --checkpoint checkpoints/acoustic/hpc_best_dev_model.pt
"""

import argparse
import csv
import os
import sys
import time
import warnings
import logging

# Suppress noisy warnings from transformers
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# Use the SAME WER module as all other evaluation scripts (baselines, etc.)
# This ensures consistent normalization + corpus-level WER across all experiments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.utils.wer import normalize_text, compute_corpus_wer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"       # development set (seen during training for validation)
TEST_CSV = "ASER-Dataset/splits/asr_test.csv"      # test set (never seen during training — use this for final eval)
CHECKPOINT_DIR = "checkpoints/acoustic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # auto-detect GPU
SAMPLE_RATE = 16000                                # Whisper expects 16kHz audio
MAX_DURATION = 30                                  # skip clips longer than 30 seconds
STUDENT_MODEL = "openai/whisper-small"             # base model for original + student
TEACHER_MODEL = "aadel4/kid-whisper-medium-en-myst"  # teacher model (fine-tuned on English children)
OUTPUT_DIR = "results/acoustic"



def load_audio(audio_path):
    """Load audio file, convert to 16kHz mono numpy array for Whisper."""
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)  # resample to 16kHz
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo → mono
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]  # trim to max 30 seconds
    return wav.squeeze(0).numpy()  # return as 1D numpy array


def transcribe_multilingual(model, processor, audio_np, language):
    """
    Transcribe using a MULTILINGUAL Whisper model (original whisper-small or student).

    Used for: Original Whisper Small and our Student (trained encoder + original decoder)
    Both are multilingual models that understand language tokens.

    Input:  audio numpy array (1D, 16kHz) + language name ("Hindi"/"Marathi"/"English")
    Output: transcribed text string

    Steps:
      1. Audio numpy → mel spectrogram tensor (80 frequency bands × time frames)
      2. Set forced_decoder_ids = language + task tokens (e.g., <|hi|> <|transcribe|>)
         This forces decoder to output Hindi text instead of guessing the language
      3. Forward pass: mel → encoder → decoder → token IDs
      4. Token IDs → text string
    """
    # Audio → mel spectrogram
    inputs = processor.feature_extractor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs.input_features.to(DEVICE)  # shape: (1, 80, 3000)

    # Set language hint — tells decoder which language to generate
    lang_map = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
    lang_code = lang_map.get(language, "hi")
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=lang_code, task="transcribe"
    )
    # forced_decoder_ids = [(1, <|hi|> token_id), (2, <|transcribe|> token_id)]

    # Forward pass: encoder processes mel → decoder generates text tokens
    with torch.no_grad():  # no gradients needed (inference only, not training)
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    # Token IDs → text
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_english_only(model, processor, audio_np):
    """
    Transcribe using an ENGLISH-ONLY model (Kid-Whisper Medium).

    Used for: Teacher model (kid-whisper-medium-en-myst)

    CRITICAL DIFFERENCES from multilingual:
      - NO forced_decoder_ids — English-only models don't have <|hi|> or <|mr|> tokens
      - Uses English-only processor (whisper-medium.en) — different tokenizer
      - If we force Hindi tokens into this model, it outputs garbage
      - For Hindi/Marathi audio, it will output English nonsense (expected behavior)

    Input:  audio numpy array (1D, 16kHz)
    Output: transcribed text string (always in English)
    """
    # Audio → mel spectrogram (same as multilingual)
    inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    # Forward pass — NO language hint, decoder generates in English by default
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            max_new_tokens=225,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def load_trained_model(checkpoint_path):
    """
    Load the STUDENT model = fresh Whisper Small + our HPC-trained encoder weights.

    KEY CONCEPT:
      Our acoustic distillation only trained the ENCODER (187 weights).
      The DECODER was never trained — it stays as original Whisper Small's decoder.
      So here we:
        1. Load a fresh Whisper Small (encoder + decoder)
        2. REPLACE only the encoder weights with our HPC-trained weights
        3. The decoder remains untrained (this is WHY it produces garbage — decoder
           can't understand the new encoder's output distribution)

    The checkpoint file (e.g., hpc_best_dev_model.pt) contains:
      - model_state_dict: 189 keys (187 encoder + 2 projection layer)
      - epoch, train_loss, dev_loss: training metadata
    """
    # Step 1: Load fresh, untrained Whisper Small from HuggingFace
    model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    model.to(DEVICE)

    # Step 2: Load our trained checkpoint (from HPC or local training)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    trained_state = checkpoint["model_state_dict"]
    # Keys look like: "encoder.conv1.weight", "encoder.layers.0.self_attn.k_proj.weight", etc.

    # Step 3: Replace encoder weights in the fresh model
    whisper_state = model.state_dict()
    updated = 0
    for key, value in trained_state.items():
        # Skip projection layer (768→1024 dim, used during training to match teacher size)
        # Skip dropout (not a real weight)
        if key.startswith("projection") or key.startswith("dropout"):
            continue
        # HuggingFace Whisper keys have "model." prefix, our checkpoint doesn't
        # e.g., our "encoder.conv1.weight" → HF's "model.encoder.conv1.weight"
        whisper_key = f"model.{key}"
        if whisper_key in whisper_state and whisper_state[whisper_key].shape == value.shape:
            whisper_state[whisper_key] = value  # REPLACE original weight with trained weight
            updated += 1

    # Step 4: Load the hybrid model (trained encoder + original decoder)
    model.load_state_dict(whisper_state)
    epoch = checkpoint.get("epoch", "?")
    train_loss = checkpoint.get("train_loss", 0)
    dev_loss = checkpoint.get("dev_loss", 0)
    return model, epoch, train_loss, dev_loss, updated


def load_balanced_clips(csv_path, per_lang=10):
    """Load equal number of clips per language so results are not biased."""
    by_lang = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"]
            if float(row["duration_sec"]) > MAX_DURATION:
                continue  # skip clips > 30s (Whisper can't handle them)
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(row)

    # Take equal clips from each language (e.g., 50 Hindi, 50 English, 50 Marathi)
    clips = []
    for lang in sorted(by_lang.keys()):
        selected = by_lang[lang][:per_lang]
        clips.extend(selected)

    return clips, by_lang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_lang", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_set", type=str, default="test", choices=["dev", "test"],
                        help="Which set to evaluate on (default: test)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "best_dev_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Load clips
    eval_csv = TEST_CSV if args.eval_set == "test" else DEV_CSV
    clips, all_clips = load_balanced_clips(eval_csv, per_lang=args.per_lang)

    # ── Print header ──
    print("=" * 80)
    print("  THREE-MODEL WER COMPARISON")
    print("=" * 80)
    print()
    print("  CHECKPOINT INFO:")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_epoch = checkpoint.get("epoch", "?")
    ckpt_train_loss = checkpoint.get("train_loss", 0)
    ckpt_dev_loss = checkpoint.get("dev_loss", 0)
    print(f"    Path:       {checkpoint_path}")
    print(f"    Epoch:      {ckpt_epoch}")
    print(f"    Train MSE:  {ckpt_train_loss:.4f}")
    print(f"    Dev MSE:    {ckpt_dev_loss:.4f}")
    is_hpc = "hpc" in checkpoint_path.lower() or ckpt_train_loss < 0.40
    if is_hpc:
        training_info = "13,765 clips (HPC Agastya V100)"
    else:
        training_info = "2,000 clips (LOCAL machine)"
    print(f"    Training:   {training_info}")
    print()
    print("  EVALUATION SET:")
    print(f"    Source: asr_{args.eval_set}.csv")
    for lang in sorted(all_clips.keys()):
        n_selected = min(args.per_lang, len(all_clips[lang]))
        print(f"    {lang}: {n_selected} clips evaluated (out of {len(all_clips[lang])} available)")
    print(f"    Total: {len(clips)} clips")
    print()
    print("  MODELS:")
    print(f"    1. Original Whisper Small  (openai/whisper-small, untrained)")
    print(f"    2. Teacher Kid-Whisper Med (aadel4/kid-whisper-medium-en-myst)")
    print(f"    3. Student Trained Small   (our acoustic-distilled encoder)")
    print()
    del checkpoint  # free memory

    # ── Load all 3 models into GPU/CPU memory ──
    print("  Loading models...")

    # MODEL 1: Original Whisper Small — baseline, no fine-tuning at all
    original_model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    original_model.to(DEVICE).eval()  # .eval() = inference mode (no dropout)
    original_processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)
    print("    [1/3] Original Whisper Small loaded")

    # MODEL 2: Teacher — Kid-Whisper Medium (English-only children's ASR)
    # IMPORTANT: Must use English-only processor (whisper-medium.en) and NO language hint
    # Using multilingual processor produces garbage — verified by testing
    teacher_model = WhisperForConditionalGeneration.from_pretrained(TEACHER_MODEL)
    teacher_model.to(DEVICE).eval()
    teacher_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    print("    [2/3] Teacher Kid-Whisper Medium loaded")

    # MODEL 3: Student — Whisper Small with OUR trained encoder from HPC
    # This loads fresh whisper-small, then replaces 187 encoder weights with our checkpoint
    # The decoder stays original (untrained) — this is the mismatch problem
    student_model, _, _, _, n_weights = load_trained_model(checkpoint_path)
    student_model.eval()
    student_processor = original_processor  # same tokenizer as original whisper-small
    print(f"    [3/3] Student Trained Small loaded ({n_weights}/187 encoder weights)")
    print()

    # ── Evaluate all clips ──
    # For each clip, we run all 3 models on the SAME audio and compare outputs
    # This lets us see exactly how the student's trained encoder changed things
    results = []
    per_lang_data = {}  # stores refs/preds per language for corpus-level WER at the end

    for clip in tqdm(clips, desc="  Evaluating", unit="clip"):
        language = clip["language"]
        filename = os.path.basename(clip["audio_path"])
        ground_truth = clip["transcript"]

        try:
            # STEP 1: Load audio (same audio for all 3 models — fair comparison)
            audio_np = load_audio(clip["audio_path"])

            # STEP 2: Get transcription from each model
            # Original: fresh whisper-small, multilingual, with language hint
            orig_out = transcribe_multilingual(original_model, original_processor, audio_np, language)
            # Teacher: kid-whisper-medium, English-only, NO language hint
            teacher_out = transcribe_english_only(teacher_model, teacher_processor, audio_np)
            # Student: whisper-small with HPC-trained encoder, multilingual, with language hint
            student_out = transcribe_multilingual(student_model, student_processor, audio_np, language)
        except Exception as e:
            continue

        # Skip clips with empty ground truth (can't compute WER)
        gt_norm = normalize_text(ground_truth)
        if not gt_norm:
            continue

        # STEP 3: Compute per-clip WER (used in CSV for individual clip inspection)
        # Note: final summary uses corpus-level WER across ALL clips (more accurate)
        orig_wer = compute_corpus_wer([ground_truth], [orig_out])
        teacher_wer = compute_corpus_wer([ground_truth], [teacher_out])
        student_wer = compute_corpus_wer([ground_truth], [student_out])

        # STEP 4: Collect raw predictions per language
        # We store RAW text here (not normalized) because compute_corpus_wer
        # handles normalization internally — keeping things consistent
        if language not in per_lang_data:
            per_lang_data[language] = {
                "original": {"refs": [], "preds": []},
                "teacher": {"refs": [], "preds": []},
                "student": {"refs": [], "preds": []},
            }
        per_lang_data[language]["original"]["refs"].append(ground_truth)
        per_lang_data[language]["original"]["preds"].append(orig_out)
        per_lang_data[language]["teacher"]["refs"].append(ground_truth)
        per_lang_data[language]["teacher"]["preds"].append(teacher_out)
        per_lang_data[language]["student"]["refs"].append(ground_truth)
        per_lang_data[language]["student"]["preds"].append(student_out)

        # STEP 5: Save per-clip result for CSV output
        results.append({
            "clip": filename,
            "language": language,
            "ground_truth": ground_truth,
            "original_whisper_small": orig_out,
            "teacher_kid_whisper_medium": teacher_out,
            "student_trained_whisper_small": student_out,
            "wer_original": f"{orig_wer:.1%}",
            "wer_teacher": f"{teacher_wer:.1%}",
            "wer_student": f"{student_wer:.1%}",
        })

    # ── Summary Table ──
    # Now compute CORPUS-LEVEL WER — this is the standard metric for ASR papers
    # It treats all clips as one big text and computes WER across everything
    # This gives more weight to longer clips (more words = more important)
    print()
    print("=" * 80)
    print("  FINAL WER SUMMARY (corpus-level, same method as baselines)")
    print("=" * 80)
    print()

    # Merge all languages together for overall WER
    all_orig = {"refs": [], "preds": []}
    all_teacher = {"refs": [], "preds": []}
    all_student = {"refs": [], "preds": []}
    for lang_data in per_lang_data.values():
        all_orig["refs"].extend(lang_data["original"]["refs"])
        all_orig["preds"].extend(lang_data["original"]["preds"])
        all_teacher["refs"].extend(lang_data["teacher"]["refs"])
        all_teacher["preds"].extend(lang_data["teacher"]["preds"])
        all_student["refs"].extend(lang_data["student"]["refs"])
        all_student["preds"].extend(lang_data["student"]["preds"])

    overall_orig = compute_corpus_wer(all_orig["refs"], all_orig["preds"])
    overall_teacher = compute_corpus_wer(all_teacher["refs"], all_teacher["preds"])
    overall_student = compute_corpus_wer(all_student["refs"], all_student["preds"])

    print(f"  {'':20} {'Original':>12} {'Teacher':>12} {'Student':>12}")
    print(f"  {'':20} {'Whisper Sm':>12} {'Kid-Wh Med':>12} {'Trained Sm':>12}")
    print(f"  {'─'*60}")

    for lang in sorted(per_lang_data.keys()):
        data = per_lang_data[lang]
        n = len(data["original"]["refs"])
        o = compute_corpus_wer(data["original"]["refs"], data["original"]["preds"])
        t = compute_corpus_wer(data["teacher"]["refs"], data["teacher"]["preds"])
        s = compute_corpus_wer(data["student"]["refs"], data["student"]["preds"])
        print(f"  {f'{lang} ({n} clips)':<20} {o:>11.1%} {t:>11.1%} {s:>11.1%}")

    print(f"  {'─'*60}")
    n_total = len(all_orig["refs"])
    print(f"  {f'Overall ({n_total} clips)':<20} {overall_orig:>11.1%} {overall_teacher:>11.1%} {overall_student:>11.1%}")
    print()

    # Best model per language
    print("  BEST MODEL PER LANGUAGE:")
    for lang in sorted(per_lang_data.keys()):
        data = per_lang_data[lang]
        o = compute_corpus_wer(data["original"]["refs"], data["original"]["preds"])
        t = compute_corpus_wer(data["teacher"]["refs"], data["teacher"]["preds"])
        s = compute_corpus_wer(data["student"]["refs"], data["student"]["preds"])
        best = min(o, t, s)
        if best == o:
            print(f"    {lang}: Original Whisper Small ({o:.1%})")
        elif best == t:
            print(f"    {lang}: Teacher Kid-Whisper Medium ({t:.1%})")
        else:
            print(f"    {lang}: Student Trained Small ({s:.1%})")

    print()
    print("=" * 80)

    # ── Save to CSV ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_total = len(results)
    source_tag = "hpc" if is_hpc else "local"
    output_csv = os.path.join(OUTPUT_DIR, f"three_model_{source_tag}_epoch{ckpt_epoch}_{args.eval_set}_{n_total}clips.csv")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "language", "ground_truth",
            "original_whisper_small", "teacher_kid_whisper_medium",
            "student_trained_whisper_small",
            "wer_original", "wer_teacher", "wer_student"
        ])
        writer.writeheader()
        writer.writerows(results)

    # ── Save clean text report ──
    report_path = os.path.join(OUTPUT_DIR, f"three_model_{source_tag}_epoch{ckpt_epoch}_{args.eval_set}_{n_total}clips.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("THREE-MODEL WER COMPARISON REPORT\n")
        f.write(f"{'='*80}\n\n")
        f.write("MODELS:\n")
        f.write(f"  1. Original  — openai/whisper-small (pretrained, no fine-tuning)\n")
        f.write(f"  2. Teacher   — aadel4/kid-whisper-medium-en-myst (Kid-Whisper Medium, English-only)\n")
        f.write(f"  3. Student   — openai/whisper-small encoder fine-tuned via acoustic distillation\n")
        f.write(f"               Checkpoint: epoch {ckpt_epoch}, train_MSE={ckpt_train_loss:.4f}, dev_MSE={ckpt_dev_loss:.4f}\n")
        f.write(f"               Training data: {training_info}\n\n")
        f.write(f"Evaluated on: asr_{args.eval_set}.csv, {len(clips)} clips ({args.per_lang} per language)\n")
        f.write(f"Device: {DEVICE.upper()}\n\n")

        for lang in sorted(per_lang_data.keys()):
            f.write(f"{'='*80}\n")
            f.write(f"LANGUAGE: {lang}\n")
            f.write(f"{'='*80}\n\n")

            lang_results = [r for r in results if r["language"] == lang]
            for r in lang_results:
                f.write(f"File: {r['clip']}\n")
                f.write(f"GT:       {r['ground_truth']}\n")
                f.write(f"Original (whisper-small):       {r['original_whisper_small']}  (WER: {r['wer_original']})\n")
                f.write(f"Teacher  (kid-whisper-medium):   {r['teacher_kid_whisper_medium']}  (WER: {r['wer_teacher']})\n")
                f.write(f"Student  (distilled whisper-sm): {r['student_trained_whisper_small']}  (WER: {r['wer_student']})\n")
                f.write(f"{'-'*80}\n")
            f.write("\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"SUMMARY TABLE\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'':20} {'Original':>16} {'Teacher':>16} {'Student':>16}\n")
        f.write(f"{'':20} {'whisper-small':>16} {'kid-wh-medium':>16} {'distilled-sm':>16}\n")
        f.write(f"{'─'*70}\n")
        for lang in sorted(per_lang_data.keys()):
            data = per_lang_data[lang]
            n = len(data["original"]["refs"])
            o = compute_corpus_wer(data["original"]["refs"], data["original"]["preds"])
            t = compute_corpus_wer(data["teacher"]["refs"], data["teacher"]["preds"])
            s = compute_corpus_wer(data["student"]["refs"], data["student"]["preds"])
            f.write(f"{f'  {lang} ({n} clips)':<20} {o:>15.1%} {t:>15.1%} {s:>15.1%}\n")
        f.write(f"{'─'*70}\n")
        f.write(f"{f'  Overall ({n_total})':<20} {overall_orig:>15.1%} {overall_teacher:>15.1%} {overall_student:>15.1%}\n")

    print(f"  CSV saved:    {output_csv}")
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
