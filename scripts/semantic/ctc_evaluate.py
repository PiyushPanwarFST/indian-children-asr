"""
CTC Head Evaluation — Compare GT vs Teacher vs Student (CTC)
=============================================================

PURPOSE:
    Evaluate the CTC head on test/dev set and produce a side-by-side
    comparison report showing:
      - Ground Truth transcript
      - Teacher output (Kid-Whisper Medium decoder)
      - Student output (our trained encoder + CTC head)
    Plus corpus-level WER per language and overall.

    This is the equivalent of step3_three_model_comparison.py but for
    the CTC-based student instead of the Whisper-decoder-based student.

MODELS:
    1. Teacher: Kid-Whisper Medium (aadel4/kid-whisper-medium-en-myst)
       - English-only processor, NO forced_decoder_ids
       - Good on English children's speech, poor on Hindi/Marathi
    2. Student: Our trained encoder (frozen, from acoustic distillation)
       + CTC head (trained in ctc_training.py)
       - CTC greedy decoding → character-level output → text

WER:
    Uses corpus-level WER from scripts/utils/wer.py (same as all experiments)

Usage:
    # Quick test (10 clips per language):
    python scripts/semantic/ctc_evaluate.py --per_lang 10

    # Full test set:
    python scripts/semantic/ctc_evaluate.py --per_lang 9999 --eval_set test

    # Specify CTC checkpoint:
    python scripts/semantic/ctc_evaluate.py --per_lang 9999 --ctc_checkpoint checkpoints/semantic/ctc_best_wer.pt
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
import logging

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# ─── Project imports ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.wer import normalize_text, compute_corpus_wer


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEV_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_dev.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_test.csv")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "ASER-Dataset", "vocab.json")
ENCODER_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "acoustic", "hpc_best_dev_model.pt")
CTC_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "semantic")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "semantic")

STUDENT_MODEL = "openai/whisper-small"
TEACHER_MODEL = "aadel4/kid-whisper-medium-en-myst"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30


# ─────────────────────────────────────────────
# VOCABULARY (same functions as ctc_training.py)
# ─────────────────────────────────────────────
def load_vocab(vocab_path):
    """Load character vocabulary from vocab.json."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)


def ctc_greedy_decode(logits, idx_to_char):
    """
    Standard CTC greedy decoding:
      1. argmax per frame → token index
      2. Collapse consecutive duplicates
      3. Remove blanks (index 0)
      4. Map to characters
    """
    predicted_ids = logits.argmax(dim=-1).tolist()

    decoded_ids = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id:
            if idx != 0:  # skip blanks
                decoded_ids.append(idx)
        prev_id = idx

    chars = []
    for idx in decoded_ids:
        token = idx_to_char.get(idx, "")
        if token == "<space>":
            chars.append(" ")
        elif token in ("<blank>", "<unk>"):
            continue
        else:
            chars.append(token)

    return "".join(chars)


# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────
def load_audio(audio_path):
    """Load audio → 16kHz mono numpy array."""
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    return wav.squeeze(0).numpy()


def prepare_mel(audio_np, processor):
    """
    Audio numpy → mel spectrogram + real frame count.
    Returns:
        input_features: (1, 80, 3000)
        real_frames: int (number of non-padding encoder frames)
    """
    inputs = processor.feature_extractor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features  # (1, 80, 3000)
    real_mel = inputs.attention_mask.sum().item()
    real_frames = int(real_mel) // 2
    return input_features, real_frames


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_frozen_encoder(checkpoint_path):
    """
    Load our HPC-trained encoder (frozen).
    Same as ctc_training.py — strips 'encoder.' prefix from checkpoint keys.
    """
    whisper = WhisperModel.from_pretrained(STUDENT_MODEL)
    encoder = whisper.encoder
    encoder_dim = whisper.config.d_model  # 768

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained_state = checkpoint["model_state_dict"]

    encoder_state = encoder.state_dict()
    updated = 0
    for key, value in trained_state.items():
        if key.startswith("projection") or key.startswith("dropout"):
            continue
        encoder_key = key[len("encoder."):] if key.startswith("encoder.") else key
        if encoder_key in encoder_state and encoder_state[encoder_key].shape == value.shape:
            encoder_state[encoder_key] = value
            updated += 1

    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    del whisper.decoder
    del whisper

    return encoder, encoder_dim, updated


def load_ctc_head(ctc_checkpoint_path, encoder_dim, vocab_size):
    """
    Load trained CTC head from checkpoint.

    Args:
        ctc_checkpoint_path: Path to CTC checkpoint (e.g., ctc_best_wer.pt)
        encoder_dim: 768 (must match what CTC head was trained with)
        vocab_size: 85 (must match vocab.json)

    Returns:
        ctc_head: nn.Linear with trained weights
        epoch: which epoch this checkpoint is from
        dev_wer: dev WER at checkpoint time
    """
    ctc_head = nn.Linear(encoder_dim, vocab_size)

    checkpoint = torch.load(ctc_checkpoint_path, map_location="cpu", weights_only=False)
    ctc_head.load_state_dict(checkpoint["ctc_head_state_dict"])
    ctc_head.eval()

    epoch = checkpoint.get("epoch", "?")
    dev_wer = checkpoint.get("dev_wer", 0)
    dev_loss = checkpoint.get("dev_loss", 0)
    train_loss = checkpoint.get("train_loss", 0)

    return ctc_head, epoch, dev_wer, dev_loss, train_loss


def transcribe_teacher(model, processor, audio_np):
    """
    Transcribe using Teacher (Kid-Whisper Medium — English-only).
    NO forced_decoder_ids — English-only model.
    """
    inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(input_features, max_new_tokens=225)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_ctc(encoder, ctc_head, processor, audio_np, idx_to_char):
    """
    Transcribe using our Student (trained encoder + CTC head).

    Steps:
        1. Audio → mel spectrogram
        2. Mel → frozen encoder → features (1500, 768)
        3. Features → CTC head → logits (1500, 85)
        4. Greedy decode → text
    """
    input_features, real_frames = prepare_mel(audio_np, processor)
    input_features = input_features.to(DEVICE)

    if real_frames == 0:
        return ""

    with torch.no_grad():
        encoder_output = encoder(input_features).last_hidden_state  # (1, 1500, 768)
        logits = ctc_head(encoder_output)  # (1, 1500, 85)

    # Decode only real frames (skip padding)
    return ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_balanced_clips(csv_path, per_lang=10):
    """Load equal number of clips per language."""
    by_lang = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"]
            if float(row["duration_sec"]) > MAX_DURATION:
                continue
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(row)

    clips = []
    for lang in sorted(by_lang.keys()):
        selected = by_lang[lang][:per_lang]
        clips.extend(selected)

    return clips, by_lang


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CTC Head Evaluation — GT vs Teacher vs Student(CTC)")
    parser.add_argument("--per_lang", type=int, default=10, help="Clips per language")
    parser.add_argument("--eval_set", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--ctc_checkpoint", type=str, default=None,
                        help="Path to CTC checkpoint (default: checkpoints/semantic/ctc_best_wer.pt)")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Path to encoder checkpoint (default: checkpoints/acoustic/hpc_best_dev_model.pt)")
    args = parser.parse_args()

    ctc_ckpt_path = args.ctc_checkpoint or os.path.join(CTC_CHECKPOINT_DIR, "ctc_best_wer.pt")
    enc_ckpt_path = args.encoder_checkpoint or ENCODER_CHECKPOINT

    if not os.path.exists(ctc_ckpt_path):
        print(f"ERROR: CTC checkpoint not found: {ctc_ckpt_path}")
        return
    if not os.path.exists(enc_ckpt_path):
        print(f"ERROR: Encoder checkpoint not found: {enc_ckpt_path}")
        return

    # ── Load vocab ──
    print("Loading vocabulary...")
    char_to_idx, idx_to_char, vocab_size = load_vocab(VOCAB_PATH)
    print(f"  Vocab size: {vocab_size}")

    # ── Load clips ──
    eval_csv = TEST_CSV if args.eval_set == "test" else DEV_CSV
    clips, all_clips = load_balanced_clips(eval_csv, per_lang=args.per_lang)

    # ── Print header ──
    print(f"\n{'='*80}")
    print(f"  CTC HEAD EVALUATION — GT vs Teacher vs Student(CTC)")
    print(f"{'='*80}")
    print(f"\n  Eval set: asr_{args.eval_set}.csv")
    for lang in sorted(all_clips.keys()):
        n = min(args.per_lang, len(all_clips[lang]))
        print(f"    {lang}: {n} clips (of {len(all_clips[lang])} available)")
    print(f"    Total: {len(clips)} clips")
    print(f"  Device: {DEVICE}")

    # ── Load encoder ──
    print(f"\n  Loading frozen encoder...")
    encoder, encoder_dim, n_enc = load_frozen_encoder(enc_ckpt_path)
    encoder.to(DEVICE)
    print(f"    Loaded {n_enc}/187 encoder weights")

    # ── Load CTC head ──
    print(f"  Loading CTC head...")
    ctc_head, ctc_epoch, ctc_dev_wer, ctc_dev_loss, ctc_train_loss = load_ctc_head(
        ctc_ckpt_path, encoder_dim, vocab_size
    )
    ctc_head.to(DEVICE)
    print(f"    Epoch: {ctc_epoch}, Train loss: {ctc_train_loss:.4f}, Dev loss: {ctc_dev_loss:.4f}, Dev WER: {ctc_dev_wer*100:.1f}%")

    # ── Load teacher ──
    print(f"  Loading teacher (Kid-Whisper Medium)...")
    teacher_model = WhisperForConditionalGeneration.from_pretrained(TEACHER_MODEL)
    teacher_model.to(DEVICE).eval()
    teacher_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    print(f"    Teacher loaded")

    # ── Load student processor (for mel spectrogram) ──
    student_processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    print(f"\n{'='*80}")
    print(f"  RUNNING EVALUATION...")
    print(f"{'='*80}\n")

    # ── Evaluate ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    per_lang_data = {}  # {lang: {"refs": [], "teacher_preds": [], "student_preds": []}}

    for clip in tqdm(clips, desc="  Evaluating", unit="clip"):
        language = clip["language"]
        filename = os.path.basename(clip["audio_path"])
        ground_truth = clip["transcript"]

        try:
            audio_np = load_audio(clip["audio_path"])

            # Teacher transcription (Kid-Whisper decoder)
            teacher_out = transcribe_teacher(teacher_model, teacher_processor, audio_np)

            # Student transcription (CTC head)
            student_out = transcribe_ctc(encoder, ctc_head, student_processor, audio_np, idx_to_char)

            # Store results
            result = {
                "file": filename,
                "language": language,
                "ground_truth": ground_truth,
                "teacher": teacher_out,
                "student_ctc": student_out,
            }
            results.append(result)

            # Collect per-language data for WER
            if language not in per_lang_data:
                per_lang_data[language] = {"refs": [], "teacher_preds": [], "student_preds": []}
            per_lang_data[language]["refs"].append(ground_truth)
            per_lang_data[language]["teacher_preds"].append(teacher_out)
            per_lang_data[language]["student_preds"].append(student_out)

        except Exception as e:
            print(f"  Error on {filename}: {e}")
            continue

    # ── Compute WER ──
    print(f"\n{'='*80}")
    print(f"  RESULTS — Corpus-Level WER")
    print(f"{'='*80}\n")

    all_refs = []
    all_teacher_preds = []
    all_student_preds = []

    header = f"  {'Language':<20} {'Teacher (Kid-Wh Med)':>20} {'Student (CTC Head)':>20}"
    print(header)
    print(f"  {'─'*60}")

    for lang in sorted(per_lang_data.keys()):
        data = per_lang_data[lang]
        n = len(data["refs"])

        teacher_wer = compute_corpus_wer(data["refs"], data["teacher_preds"])
        student_wer = compute_corpus_wer(data["refs"], data["student_preds"])

        print(f"  {lang + f' ({n} clips)':<20} {teacher_wer*100:>19.1f}% {student_wer*100:>19.1f}%")

        all_refs.extend(data["refs"])
        all_teacher_preds.extend(data["teacher_preds"])
        all_student_preds.extend(data["student_preds"])

    print(f"  {'─'*60}")
    overall_teacher = compute_corpus_wer(all_refs, all_teacher_preds)
    overall_student = compute_corpus_wer(all_refs, all_student_preds)
    print(f"  {'Overall (' + str(len(all_refs)) + ')':<20} {overall_teacher*100:>19.1f}% {overall_student*100:>19.1f}%")

    # ── Save detailed report ──
    n_clips = len(clips)
    report_name = f"ctc_eval_{args.eval_set}_{n_clips}clips.txt"
    report_path = os.path.join(OUTPUT_DIR, report_name)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CTC HEAD EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Eval set: asr_{args.eval_set}.csv, {len(clips)} clips\n")
        f.write(f"Encoder: {enc_ckpt_path}\n")
        f.write(f"CTC head: {ctc_ckpt_path} (epoch {ctc_epoch})\n")
        f.write(f"Device: {DEVICE}\n\n")

        f.write("MODELS:\n")
        f.write(f"  Teacher: {TEACHER_MODEL} (English-only decoder)\n")
        f.write(f"  Student: Trained encoder + CTC head (vocab={vocab_size}, greedy decode)\n\n")

        # Write per-language results
        for lang in sorted(per_lang_data.keys()):
            f.write(f"{'='*80}\n")
            f.write(f"LANGUAGE: {lang}\n")
            f.write(f"{'='*80}\n\n")

            lang_results = [r for r in results if r["language"] == lang]
            for r in lang_results:
                f.write(f"File: {r['file']}\n")
                f.write(f"GT:      {r['ground_truth']}\n")
                f.write(f"Teacher: {r['teacher']}\n")
                f.write(f"Student: {r['student_ctc']}\n")
                f.write("-" * 80 + "\n")
            f.write("\n")

        # Write summary table
        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY TABLE — Corpus-Level WER\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Language':<25} {'Teacher (Kid-Wh Med)':>20} {'Student (CTC Head)':>20}\n")
        f.write(f"{'─'*65}\n")
        for lang in sorted(per_lang_data.keys()):
            data = per_lang_data[lang]
            n = len(data["refs"])
            tw = compute_corpus_wer(data["refs"], data["teacher_preds"])
            sw = compute_corpus_wer(data["refs"], data["student_preds"])
            f.write(f"{lang + f' ({n} clips)':<25} {tw*100:>19.1f}% {sw*100:>19.1f}%\n")
        f.write(f"{'─'*65}\n")
        f.write(f"{'Overall (' + str(len(all_refs)) + ')':<25} {overall_teacher*100:>19.1f}% {overall_student*100:>19.1f}%\n")

    print(f"\n  Report saved: {report_path}")

    # ── Save CSV ──
    csv_name = f"ctc_eval_{args.eval_set}_{n_clips}clips.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "language", "ground_truth", "teacher", "student_ctc"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  CSV saved:    {csv_path}")
    print(f"\n{'='*80}")
    print(f"  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
