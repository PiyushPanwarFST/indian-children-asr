"""
Joint Training Evaluation — Full Comparison on Test Set
========================================================

PURPOSE:
    Evaluate the joint-trained model on the test set and produce a
    side-by-side comparison report showing:
      - Ground Truth transcript
      - Baseline Whisper Small output (original, no fine-tuning)
      - Teacher output (Kid-Whisper Medium decoder)
      - Student output (joint-trained encoder + CTC head)
    Plus corpus-level WER per language and overall.

    This gives us the final numbers for the paper, comparing all models.

MODELS:
    1. Baseline: Whisper Small (openai/whisper-small)
       - Multilingual decoder with language hints
       - 244M params, no fine-tuning
    2. Teacher: Kid-Whisper Medium (aadel4/kid-whisper-medium-en-myst)
       - English-only processor, NO forced_decoder_ids
       - 769M params, fine-tuned on English children's speech
    3. Student (Joint): Our jointly-trained encoder + CTC head
       - Encoder UNFROZEN during training (updated by both MSE + CTC)
       - CTC greedy decoding, character-level, vocab=85
       - Loaded from checkpoints/joint/joint_best_wer.pt

KEY DIFFERENCE from ctc_evaluate.py:
    - Loads encoder from JOINT checkpoint (not acoustic checkpoint)
    - Joint checkpoint has encoder_state_dict directly (not wrapped in model_state_dict)
    - Also compares against baseline Whisper Small (not just teacher)

WER:
    Uses corpus-level WER from scripts/utils/wer.py (same as all experiments)

Usage:
    # Quick test (10 clips per language):
    python scripts/joint/joint_evaluate.py --per_lang 10

    # Full test set:
    python scripts/joint/joint_evaluate.py --per_lang 9999 --eval_set test

    # Specify joint checkpoint:
    python scripts/joint/joint_evaluate.py --per_lang 9999 --checkpoint checkpoints/joint/joint_best_wer.pt
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
JOINT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "joint", "joint_best_wer.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "joint")

STUDENT_MODEL = "openai/whisper-small"
TEACHER_MODEL = "aadel4/kid-whisper-medium-en-myst"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30


# ─────────────────────────────────────────────
# VOCABULARY
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
      1. argmax per frame -> token index
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
    """Load audio -> 16kHz mono numpy array."""
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    return wav.squeeze(0).numpy()


def prepare_mel(audio_np, processor):
    """
    Audio numpy -> mel spectrogram + real frame count.
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
def load_joint_model(checkpoint_path, vocab_size):
    """
    Load jointly-trained encoder + CTC head from joint checkpoint.

    KEY DIFFERENCE from ctc_evaluate.py:
        - Joint checkpoint stores encoder_state_dict directly
          (not wrapped inside model_state_dict with 'encoder.' prefix)
        - CTC head is also in the same checkpoint

    Args:
        checkpoint_path (str): Path to joint checkpoint
            e.g., "checkpoints/joint/joint_best_wer.pt"
        vocab_size (int): Number of vocab tokens (85)

    Returns:
        encoder: Trained encoder (eval mode)
        ctc_head: Trained CTC head (eval mode)
        checkpoint: Full checkpoint dict (for metadata)
    """
    print(f"  Loading joint checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load encoder architecture from Whisper Small
    whisper = WhisperModel.from_pretrained(STUDENT_MODEL)
    encoder = whisper.encoder
    encoder_dim = whisper.config.d_model  # 768

    # Load trained encoder weights from joint checkpoint
    # Joint checkpoint saves encoder.state_dict() directly — no prefix stripping needed
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Load CTC head
    ctc_head = nn.Linear(encoder_dim, vocab_size)
    ctc_head.load_state_dict(checkpoint["ctc_head_state_dict"])
    ctc_head.eval()

    # Free decoder memory
    del whisper.decoder
    del whisper

    epoch = checkpoint.get("epoch", "?")
    dev_wer = checkpoint.get("dev_wer", 0)
    alpha = checkpoint.get("alpha", 0.5)
    print(f"  Epoch: {epoch}, Dev WER: {dev_wer*100:.1f}%, Alpha: {alpha}")

    return encoder, ctc_head, encoder_dim, checkpoint


# ─────────────────────────────────────────────
# TRANSCRIPTION FUNCTIONS
# ─────────────────────────────────────────────
def transcribe_baseline(model, processor, audio_np, language):
    """
    Transcribe using Baseline Whisper Small (multilingual, with language hint).
    This is the unmodified, pretrained model — our starting point.
    """
    lang_map = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
    lang_code = lang_map.get(language, "en")

    inputs = processor.feature_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_features.to(DEVICE),
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_teacher(model, processor, audio_np):
    """
    Transcribe using Teacher (Kid-Whisper Medium — English-only).
    NO forced_decoder_ids — English-only model doesn't use language hints.
    """
    inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features.to(DEVICE), max_new_tokens=225)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_joint(encoder, ctc_head, processor, audio_np, idx_to_char):
    """
    Transcribe using our joint-trained Student (encoder + CTC head).

    Steps:
        1. Audio -> mel spectrogram
        2. Mel -> jointly-trained encoder -> features (1500, 768)
        3. Features -> CTC head -> logits (1500, 85)
        4. Greedy decode -> text
    """
    input_features, real_frames = prepare_mel(audio_np, processor)
    input_features = input_features.to(DEVICE)

    if real_frames == 0:
        return ""

    with torch.no_grad():
        encoder_output = encoder(input_features).last_hidden_state  # (1, 1500, 768)
        logits = ctc_head(encoder_output)  # (1, 1500, 85)

    return ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_clips(csv_path, per_lang=10):
    """Load clips, optionally limiting per language."""
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
    parser = argparse.ArgumentParser(description="Joint Training Evaluation — Full Comparison")
    parser.add_argument("--per_lang", type=int, default=10, help="Clips per language (9999 for all)")
    parser.add_argument("--eval_set", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to joint checkpoint (default: checkpoints/joint/joint_best_wer.pt)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint or JOINT_CHECKPOINT

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Joint checkpoint not found: {ckpt_path}")
        print(f"  You need to download it from HPC first:")
        print(f"  scp -i Aamir.pem 2025rcs1026@<hpc>:~/Indain_children_spech/checkpoints/joint/joint_best_wer.pt checkpoints/joint/")
        return

    # ── Load vocab ──
    print("Loading vocabulary...")
    char_to_idx, idx_to_char, vocab_size = load_vocab(VOCAB_PATH)
    print(f"  Vocab size: {vocab_size}")

    # ── Load clips ──
    eval_csv = TEST_CSV if args.eval_set == "test" else DEV_CSV
    clips, all_clips = load_clips(eval_csv, per_lang=args.per_lang)

    # ── Print header ──
    print(f"\n{'='*80}")
    print(f"  JOINT TRAINING EVALUATION — Full Comparison")
    print(f"{'='*80}")
    print(f"\n  Eval set: asr_{args.eval_set}.csv")
    for lang in sorted(all_clips.keys()):
        n = min(args.per_lang, len(all_clips[lang]))
        print(f"    {lang}: {n} clips (of {len(all_clips[lang])} available)")
    print(f"    Total: {len(clips)} clips")
    print(f"  Device: {DEVICE}")

    # ── Load joint model (encoder + CTC head) ──
    print(f"\n  Loading joint-trained model...")
    encoder, ctc_head, encoder_dim, joint_ckpt = load_joint_model(ckpt_path, vocab_size)
    encoder.to(DEVICE)
    ctc_head.to(DEVICE)

    # ── Load baseline Whisper Small ──
    print(f"\n  Loading baseline Whisper Small...")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(STUDENT_MODEL)
    baseline_model.to(DEVICE).eval()
    baseline_processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)
    print(f"    Baseline loaded")

    # ── Load teacher (Kid-Whisper Medium) ──
    print(f"  Loading teacher (Kid-Whisper Medium)...")
    teacher_model = WhisperForConditionalGeneration.from_pretrained(TEACHER_MODEL)
    teacher_model.to(DEVICE).eval()
    teacher_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    print(f"    Teacher loaded")

    # ── Student processor (for mel spectrogram) ──
    student_processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    print(f"\n{'='*80}")
    print(f"  RUNNING EVALUATION ({len(clips)} clips)...")
    print(f"{'='*80}\n")

    # ── Evaluate ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    per_lang_data = {}

    for clip in tqdm(clips, desc="  Evaluating", unit="clip"):
        language = clip["language"]
        filename = os.path.basename(clip["audio_path"])
        ground_truth = clip["transcript"]

        try:
            audio_np = load_audio(clip["audio_path"])

            # Baseline Whisper Small (with language hint)
            baseline_out = transcribe_baseline(baseline_model, baseline_processor, audio_np, language)

            # Teacher (Kid-Whisper Medium, English-only)
            teacher_out = transcribe_teacher(teacher_model, teacher_processor, audio_np)

            # Student (joint-trained encoder + CTC head)
            student_out = transcribe_joint(encoder, ctc_head, student_processor, audio_np, idx_to_char)

            result = {
                "file": filename,
                "language": language,
                "ground_truth": ground_truth,
                "baseline": baseline_out,
                "teacher": teacher_out,
                "student_joint": student_out,
            }
            results.append(result)

            if language not in per_lang_data:
                per_lang_data[language] = {
                    "refs": [], "baseline_preds": [],
                    "teacher_preds": [], "student_preds": [],
                }
            per_lang_data[language]["refs"].append(ground_truth)
            per_lang_data[language]["baseline_preds"].append(baseline_out)
            per_lang_data[language]["teacher_preds"].append(teacher_out)
            per_lang_data[language]["student_preds"].append(student_out)

        except Exception as e:
            print(f"  Error on {filename}: {e}")
            continue

    # ── Free GPU memory for baseline and teacher ──
    del baseline_model, baseline_processor
    del teacher_model, teacher_processor
    torch.cuda.empty_cache()

    # ── Compute WER ──
    print(f"\n{'='*80}")
    print(f"  RESULTS — Corpus-Level WER (Test Set)")
    print(f"{'='*80}\n")

    all_refs = []
    all_baseline = []
    all_teacher = []
    all_student = []

    header = f"  {'Language':<22} {'Baseline (Wh-Sm)':>18} {'Teacher (Kid-Wh)':>18} {'Student (Joint)':>18}"
    print(header)
    print(f"  {'─'*76}")

    for lang in sorted(per_lang_data.keys()):
        data = per_lang_data[lang]
        n = len(data["refs"])

        baseline_wer = compute_corpus_wer(data["refs"], data["baseline_preds"])
        teacher_wer = compute_corpus_wer(data["refs"], data["teacher_preds"])
        student_wer = compute_corpus_wer(data["refs"], data["student_preds"])

        print(f"  {lang + f' ({n})':<22} {baseline_wer*100:>17.1f}% {teacher_wer*100:>17.1f}% {student_wer*100:>17.1f}%")

        all_refs.extend(data["refs"])
        all_baseline.extend(data["baseline_preds"])
        all_teacher.extend(data["teacher_preds"])
        all_student.extend(data["student_preds"])

    print(f"  {'─'*76}")
    overall_baseline = compute_corpus_wer(all_refs, all_baseline)
    overall_teacher = compute_corpus_wer(all_refs, all_teacher)
    overall_student = compute_corpus_wer(all_refs, all_student)
    n_total = len(all_refs)
    print(f"  {'Overall (' + str(n_total) + ')':<22} {overall_baseline*100:>17.1f}% {overall_teacher*100:>17.1f}% {overall_student*100:>17.1f}%")

    # ── Save detailed report (.txt) ──
    n_clips = len(clips)
    joint_epoch = joint_ckpt.get("epoch", "?")
    joint_dev_wer = joint_ckpt.get("dev_wer", 0)
    joint_alpha = joint_ckpt.get("alpha", 0.5)

    report_name = f"joint_eval_{args.eval_set}_{n_clips}clips.txt"
    report_path = os.path.join(OUTPUT_DIR, report_name)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("JOINT TRAINING EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Eval set: asr_{args.eval_set}.csv, {n_clips} clips\n")
        f.write(f"Joint checkpoint: {ckpt_path} (epoch {joint_epoch})\n")
        f.write(f"Dev WER at checkpoint: {joint_dev_wer*100:.1f}%\n")
        f.write(f"Alpha: {joint_alpha}\n")
        f.write(f"Device: {DEVICE}\n\n")

        f.write("MODELS:\n")
        f.write(f"  Baseline: openai/whisper-small (244M, multilingual, language hints)\n")
        f.write(f"  Teacher:  {TEACHER_MODEL} (769M, English-only decoder)\n")
        f.write(f"  Student:  Joint-trained encoder + CTC head (vocab={vocab_size}, greedy decode)\n\n")

        # Per-language detailed results
        for lang in sorted(per_lang_data.keys()):
            f.write(f"{'='*80}\n")
            f.write(f"LANGUAGE: {lang}\n")
            f.write(f"{'='*80}\n\n")

            lang_results = [r for r in results if r["language"] == lang]
            for r in lang_results:
                f.write(f"File:     {r['file']}\n")
                f.write(f"GT:       {r['ground_truth']}\n")
                f.write(f"Baseline: {r['baseline']}\n")
                f.write(f"Teacher:  {r['teacher']}\n")
                f.write(f"Student:  {r['student_joint']}\n")
                f.write("-" * 80 + "\n")
            f.write("\n")

        # Summary table
        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY TABLE — Corpus-Level WER\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Language':<25} {'Baseline (Wh-Sm)':>18} {'Teacher (Kid-Wh)':>18} {'Student (Joint)':>18}\n")
        f.write(f"{'─'*79}\n")
        for lang in sorted(per_lang_data.keys()):
            data = per_lang_data[lang]
            n = len(data["refs"])
            bw = compute_corpus_wer(data["refs"], data["baseline_preds"])
            tw = compute_corpus_wer(data["refs"], data["teacher_preds"])
            sw = compute_corpus_wer(data["refs"], data["student_preds"])
            f.write(f"{lang + f' ({n} clips)':<25} {bw*100:>17.1f}% {tw*100:>17.1f}% {sw*100:>17.1f}%\n")
        f.write(f"{'─'*79}\n")
        f.write(f"{'Overall (' + str(n_total) + ')':<25} {overall_baseline*100:>17.1f}% {overall_teacher*100:>17.1f}% {overall_student*100:>17.1f}%\n")

    print(f"\n  Report saved: {report_path}")

    # ── Save CSV ──
    csv_name = f"joint_eval_{args.eval_set}_{n_clips}clips.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "language", "ground_truth", "baseline", "teacher", "student_joint"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  CSV saved:    {csv_path}")

    # ── Final comparison summary ──
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"\n  Sequential CTC (frozen encoder):    81.9% WER (test)")
    print(f"  Joint training dev WER:              {joint_dev_wer*100:.1f}%")
    print(f"  Joint training test WER:             {overall_student*100:.1f}%")
    print(f"\n{'='*80}")
    print(f"  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
