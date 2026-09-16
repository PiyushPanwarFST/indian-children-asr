"""
Combined Branch Evaluation — Test Set WER with Side-by-Side Comparison
======================================================================

WHAT THIS SCRIPT DOES:
    Evaluates the trained dual-encoder gated fusion model on the test set.
    Compares three systems side-by-side:
      1. Baseline Whisper Small (244M, no fine-tuning) — seq2seq decode
      2. Acoustic-only encoder + CTC head (single distilled encoder)
      3. Combined fusion model (dual-encoder + gated fusion + CTC head)

    Reports per-language (Hindi, Marathi, English) and overall WER.
    Saves detailed results CSV and summary TXT.

IMPORTANT — ENCODER LOADING:
    Since both encoders are UNFROZEN during combined training, the combined
    checkpoint contains CO-ADAPTED encoder weights. This script loads encoder
    weights FROM THE COMBINED CHECKPOINT, not from the original distilled
    checkpoints. The original checkpoints are only used for the acoustic-only
    comparison model.

ARCHITECTURE (same as training, but no SpecAugment/dropout):
    Audio → Mel → Acoustic Encoder → feat_a (T, 768)
                → Semantic Encoder → feat_s (T, 768)
    [feat_a; feat_s] → GatedFusion → fused (T, 768) → CTC Head → logits (T, 85)
    logits → greedy CTC decode → predicted text

Usage:
    # Quick test (10 clips per language)
    python step2_combined_evaluate.py --per_lang 10

    # Full test set evaluation
    python step2_combined_evaluate.py --per_lang 9999 --eval_set test

    # Skip baseline to save GPU memory
    python step2_combined_evaluate.py --per_lang 9999 --skip_baseline
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

# ── Paths and constants ─────────────────────────────────────────────────────
ASER_ROOT = PROJECT_ROOT / "ASER-Dataset"
DEV_CSV = ASER_ROOT / "splits" / "asr_dev.csv"
TEST_CSV = ASER_ROOT / "splits" / "asr_test.csv"
VOCAB_PATH = ASER_ROOT / "vocab.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "combined_branch" / "01_feature_fusion"

WHISPER_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_vocab(vocab_path):
    """
    Load character vocabulary from JSON file.

    Args:
        vocab_path (str): Path to vocab.json

    Returns:
        tuple: (char_to_idx, idx_to_char, vocab_size)
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)


def ctc_greedy_decode(logits, idx_to_char):
    """
    Greedy CTC decode: argmax → collapse repeats → remove blanks → text.

    Args:
        logits (Tensor): Shape (T, vocab_size) — raw logits from CTC head
        idx_to_char (dict): Index-to-character mapping

    Returns:
        str: Decoded text
    """
    predicted_ids = logits.argmax(dim=-1).tolist()
    decoded_ids = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id:
            if idx != 0:  # skip blank token (index 0)
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


# ══════════════════════════════════════════════════════════════════════════════
# Audio loading
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(audio_path):
    """
    Load audio file → 16kHz mono numpy array, capped at MAX_DURATION seconds.

    Args:
        audio_path (str): Path to audio file

    Returns:
        np.ndarray: 1D audio samples at 16kHz
    """
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    return wav.squeeze(0).numpy()


def compute_real_frames(num_samples):
    """
    Compute encoder output frames: samples // 160 // 2, capped at 1500.

    Args:
        num_samples (int): Number of audio samples

    Returns:
        int: Number of real encoder frames
    """
    return min(num_samples // 160 // 2, 1500)


# ══════════════════════════════════════════════════════════════════════════════
# GatedFusion module (must match training definition)
# ══════════════════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    """
    Gated fusion module — identical architecture to training script.
    At eval time, dropout is set to 0.0 (no regularization needed).
    """
    def __init__(self, dim=768, dropout=0.0):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic_feat, semantic_feat):
        concat = torch.cat([acoustic_feat, semantic_feat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        fused = gate * acoustic_feat + (1 - gate) * semantic_feat
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_combined_model(combined_ckpt, vocab_size):
    """
    Load the full combined model from a single combined checkpoint.

    CRITICAL: Since encoders are unfrozen during training, the combined checkpoint
    contains co-adapted encoder weights. We load EVERYTHING from this checkpoint:
    acoustic encoder, semantic encoder, fusion module, and CTC head.

    Args:
        combined_ckpt (str): Path to combined checkpoint (e.g., best_wer.pt)
        vocab_size (int): Number of vocabulary tokens (85)

    Returns:
        tuple: (acoustic_encoder, semantic_encoder, fusion, ctc_head) — all in eval mode
    """
    print(f"\n  Loading combined model from: {combined_ckpt}")
    c_ckpt = torch.load(combined_ckpt, map_location=DEVICE, weights_only=False)

    epoch = c_ckpt.get("epoch", "?")
    dev_wer = c_ckpt.get("dev_wer", 0)
    print(f"    Epoch: {epoch}, Dev WER: {dev_wer:.2f}%")

    # Load acoustic encoder with co-adapted weights from combined checkpoint
    print(f"  [ACOUSTIC] Loading co-adapted encoder from combined checkpoint...")
    whisper_a = WhisperModel.from_pretrained(WHISPER_ID)
    acoustic_encoder = whisper_a.encoder.to(DEVICE)
    if "acoustic_encoder_state_dict" in c_ckpt:
        acoustic_encoder.load_state_dict(c_ckpt["acoustic_encoder_state_dict"])
        print(f"    Loaded co-adapted acoustic encoder weights")
    else:
        print(f"    WARNING: No acoustic encoder in checkpoint — using base Whisper weights")
    acoustic_encoder.eval()
    for p in acoustic_encoder.parameters():
        p.requires_grad = False
    del whisper_a

    # Load semantic encoder with co-adapted weights from combined checkpoint
    print(f"  [SEMANTIC] Loading co-adapted encoder from combined checkpoint...")
    whisper_s = WhisperModel.from_pretrained(WHISPER_ID)
    semantic_encoder = whisper_s.encoder.to(DEVICE)
    if "semantic_encoder_state_dict" in c_ckpt:
        semantic_encoder.load_state_dict(c_ckpt["semantic_encoder_state_dict"])
        print(f"    Loaded co-adapted semantic encoder weights")
    else:
        print(f"    WARNING: No semantic encoder in checkpoint — using base Whisper weights")
    semantic_encoder.eval()
    for p in semantic_encoder.parameters():
        p.requires_grad = False
    del whisper_s

    # Load fusion module
    fusion = GatedFusion(dim=768, dropout=0.0).to(DEVICE)
    fusion.load_state_dict(c_ckpt["fusion_state_dict"])
    fusion.eval()
    print(f"  [FUSION] Loaded")

    # Load CTC head
    ctc_head = nn.Linear(768, vocab_size).to(DEVICE)
    ctc_head.load_state_dict(c_ckpt["ctc_head_state_dict"])
    ctc_head.eval()
    print(f"  [CTC HEAD] Loaded")

    del c_ckpt
    return acoustic_encoder, semantic_encoder, fusion, ctc_head


def load_acoustic_only_model(acoustic_ckpt, vocab_size):
    """
    Load acoustic-only model (single encoder + CTC head) for comparison.

    This uses the ORIGINAL distilled acoustic encoder (before co-adaptation),
    so we can compare: single encoder vs dual-encoder fusion.

    Args:
        acoustic_ckpt (str): Path to acoustic branch checkpoint
        vocab_size (int): Number of vocabulary tokens

    Returns:
        tuple: (encoder, ctc_head) — both in eval mode
    """
    print(f"\n  [ACOUSTIC ONLY] Loading from {acoustic_ckpt}")
    whisper = WhisperModel.from_pretrained(WHISPER_ID)
    encoder = whisper.encoder.to(DEVICE)
    ckpt = torch.load(acoustic_ckpt, map_location=DEVICE, weights_only=False)
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    elif "model_state_dict" in ckpt:
        state = {k[len("encoder."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    ctc_head = nn.Linear(768, vocab_size).to(DEVICE)
    ctc_head.load_state_dict(ckpt["ctc_head_state_dict"])
    ctc_head.eval()
    del whisper, ckpt
    return encoder, ctc_head


# ══════════════════════════════════════════════════════════════════════════════
# Transcription functions
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_baseline(model, processor, audio_np, language):
    """
    Transcribe using vanilla Whisper Small (seq2seq with decoder).

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        audio_np (np.ndarray): Audio samples at 16kHz
        language (str): Language name ("Hindi", "Marathi", "English")

    Returns:
        str: Transcribed text
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


def transcribe_acoustic_only(encoder, ctc_head, feat_extractor, audio_np, idx_to_char):
    """
    Transcribe using single acoustic encoder + CTC head.

    Args:
        encoder: Whisper encoder with distilled weights
        ctc_head: Linear CTC head
        feat_extractor: WhisperFeatureExtractor
        audio_np (np.ndarray): Audio samples at 16kHz
        idx_to_char (dict): Index-to-character mapping

    Returns:
        str: CTC-decoded text
    """
    inputs = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    real_frames = compute_real_frames(len(audio_np))
    with torch.no_grad():
        features = encoder(inputs.input_features.to(DEVICE)).last_hidden_state
        logits = ctc_head(features)
    return ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)


def transcribe_combined(acoustic_enc, semantic_enc, fusion, ctc_head, feat_extractor, audio_np, idx_to_char):
    """
    Transcribe using dual-encoder gated fusion model.

    Both encoders process the same mel spectrogram, fusion combines them,
    CTC head produces character logits, greedy decode gives final text.

    Args:
        acoustic_enc: Co-adapted acoustic encoder
        semantic_enc: Co-adapted semantic encoder
        fusion: GatedFusion module
        ctc_head: Linear CTC head
        feat_extractor: WhisperFeatureExtractor
        audio_np (np.ndarray): Audio samples at 16kHz
        idx_to_char (dict): Index-to-character mapping

    Returns:
        str: CTC-decoded text from fused features
    """
    inputs = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    real_frames = compute_real_frames(len(audio_np))
    mel_gpu = inputs.input_features.to(DEVICE)
    with torch.no_grad():
        a_feat = acoustic_enc(mel_gpu).last_hidden_state[:, :real_frames, :]
        s_feat = semantic_enc(mel_gpu).last_hidden_state[:, :real_frames, :]
        fused = fusion(a_feat, s_feat)
        logits = ctc_head(fused)
    return ctc_greedy_decode(logits[0], idx_to_char)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_clips(csv_path, per_lang=10):
    """
    Load test/dev clips from CSV, limited to per_lang clips per language.

    Args:
        csv_path (str): Path to asr_test.csv or asr_dev.csv
        per_lang (int): Max clips per language (use 9999 for all)

    Returns:
        tuple: (clips, by_lang) where clips is flat list, by_lang is dict by language
    """
    by_lang = {}
    aser_root = str(ASER_ROOT)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row["language"]
            if float(row["duration_sec"]) > MAX_DURATION:
                continue
            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                row["audio_path"] = os.path.join(aser_root, audio_path)
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(row)
    clips = []
    for lang in sorted(by_lang.keys()):
        clips.extend(by_lang[lang][:per_lang])
    return clips, by_lang


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Combined Branch Evaluation")
    parser.add_argument("--per_lang", type=int, default=10,
                        help="Max clips per language (9999 for all)")
    parser.add_argument("--eval_set", type=str, default="test", choices=["dev", "test"],
                        help="Evaluate on dev or test set")
    parser.add_argument("--combined_ckpt", type=str,
                        default="checkpoints/combined_unfrozen/best_wer.pt",
                        help="Path to combined checkpoint (contains co-adapted encoders)")
    parser.add_argument("--acoustic_ckpt", type=str,
                        default="checkpoints/joint/joint_best_wer.pt",
                        help="Path to acoustic-only checkpoint (for comparison)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline Whisper to save GPU memory")
    args = parser.parse_args()

    combined_ckpt = str(PROJECT_ROOT / args.combined_ckpt)
    acoustic_ckpt = str(PROJECT_ROOT / args.acoustic_ckpt)

    if not os.path.exists(combined_ckpt):
        print(f"ERROR: Combined checkpoint not found: {combined_ckpt}")
        return

    # Load vocabulary
    print("Loading vocabulary...")
    char_to_idx, idx_to_char, vocab_size = load_vocab(str(VOCAB_PATH))
    print(f"  Vocab: {vocab_size} tokens")

    # Load test/dev clips
    eval_csv = str(TEST_CSV) if args.eval_set == "test" else str(DEV_CSV)
    clips, all_clips = load_clips(eval_csv, per_lang=args.per_lang)
    print(f"\n  Evaluating on {len(clips)} clips ({args.eval_set} set)")
    for lang in sorted(all_clips.keys()):
        n = min(args.per_lang, len(all_clips[lang]))
        print(f"    {lang}: {n} clips")

    # ── Load models ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading models...")
    print("=" * 70)

    from transformers import WhisperFeatureExtractor
    feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

    # Combined model — loads EVERYTHING from combined checkpoint
    acoustic_enc, semantic_enc, fusion, ctc_head = load_combined_model(
        combined_ckpt, vocab_size)

    # Acoustic-only model (for comparison — uses original distilled weights)
    acoustic_only_enc, acoustic_only_ctc = load_acoustic_only_model(acoustic_ckpt, vocab_size)

    # Baseline Whisper (optional)
    baseline_model = None
    baseline_processor = None
    if not args.skip_baseline:
        print(f"\n  [BASELINE] Loading Whisper Small...")
        baseline_processor = WhisperProcessor.from_pretrained(WHISPER_ID)
        baseline_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID).to(DEVICE)
        baseline_model.eval()

    import gc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluating...")
    print("=" * 70)

    results = []
    lang_results = {}

    for clip in tqdm(clips, desc="Evaluating", bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            audio_np = load_audio(clip["audio_path"])
            lang = clip["language"]
            gt = normalize_text(clip.get("transcript", clip.get("que_text", clip.get("text", ""))))
            if not gt:
                continue

            # Combined fusion model
            combined_pred = transcribe_combined(
                acoustic_enc, semantic_enc, fusion, ctc_head, feat_extractor, audio_np, idx_to_char)

            # Acoustic-only model
            acoustic_pred = transcribe_acoustic_only(
                acoustic_only_enc, acoustic_only_ctc, feat_extractor, audio_np, idx_to_char)

            # Baseline Whisper
            baseline_pred = ""
            if baseline_model is not None:
                baseline_pred = normalize_text(transcribe_baseline(
                    baseline_model, baseline_processor, audio_np, lang))

            results.append({
                "clip": clip.get("child_id", "") + "_" + os.path.basename(clip["audio_path"]),
                "language": lang,
                "ground_truth": gt,
                "baseline": baseline_pred,
                "acoustic": acoustic_pred,
                "combined": combined_pred,
            })

            if lang not in lang_results:
                lang_results[lang] = {"refs": [], "baseline": [], "acoustic": [], "combined": []}
            lang_results[lang]["refs"].append(gt)
            lang_results[lang]["baseline"].append(baseline_pred)
            lang_results[lang]["acoustic"].append(acoustic_pred)
            lang_results[lang]["combined"].append(combined_pred)

        except Exception as e:
            print(f"\n  ERROR: {clip.get('audio_path', '?')}: {e}")
            continue

    # ── Compute and print WERs ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_refs = [r["ground_truth"] for r in results]
    all_combined = [r["combined"] for r in results]
    all_acoustic = [r["acoustic"] for r in results]
    all_baseline = [r["baseline"] for r in results]

    overall_combined = compute_corpus_wer(all_refs, all_combined) * 100
    overall_acoustic = compute_corpus_wer(all_refs, all_acoustic) * 100
    overall_baseline = compute_corpus_wer(all_refs, all_baseline) * 100 if baseline_model else 0

    print(f"\n  {'Model':<25} {'Overall':>10} ", end="")
    for lang in sorted(lang_results.keys()):
        print(f"{lang:>10} ", end="")
    print()
    print(f"  {'─'*25} {'─'*10} " + " ".join(f"{'─'*10}" for _ in lang_results))

    for model_name, preds_key in [("Baseline Whisper", "baseline"), ("Acoustic Only", "acoustic"), ("Combined (Fusion)", "combined")]:
        if model_name == "Baseline Whisper" and baseline_model is None:
            continue
        all_preds = [r[preds_key] for r in results]
        wer = compute_corpus_wer(all_refs, all_preds) * 100
        print(f"  {model_name:<25} {wer:>9.2f}%", end="")
        for lang in sorted(lang_results.keys()):
            lr = lang_results[lang]
            lang_wer = compute_corpus_wer(lr["refs"], lr[preds_key]) * 100
            print(f" {lang_wer:>9.2f}%", end="")
        print()

    # ── Save results ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_clips = len(results)

    # CSV with per-clip predictions
    csv_path = OUTPUT_DIR / f"combined_eval_{args.eval_set}_{n_clips}clips.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "language", "ground_truth", "baseline", "acoustic", "combined"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV: {csv_path}")

    # Summary text file
    txt_path = OUTPUT_DIR / f"combined_eval_{args.eval_set}_{n_clips}clips.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Combined Branch Evaluation — {args.eval_set} set ({n_clips} clips)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Overall WER:\n")
        if baseline_model:
            f.write(f"  Baseline Whisper Small: {overall_baseline:.2f}%\n")
        f.write(f"  Acoustic Only:         {overall_acoustic:.2f}%\n")
        f.write(f"  Combined (Fusion):     {overall_combined:.2f}%\n\n")
        f.write(f"Per-language WER:\n")
        for lang in sorted(lang_results.keys()):
            lr = lang_results[lang]
            a_wer = compute_corpus_wer(lr["refs"], lr["acoustic"]) * 100
            c_wer = compute_corpus_wer(lr["refs"], lr["combined"]) * 100
            f.write(f"  {lang}: Acoustic {a_wer:.2f}% → Combined {c_wer:.2f}%\n")
        f.write(f"\nSample predictions:\n")
        for r in results[:10]:
            f.write(f"\n  [{r['language']}] {r['clip']}\n")
            f.write(f"  GT:       {r['ground_truth']}\n")
            f.write(f"  Acoustic: {r['acoustic']}\n")
            f.write(f"  Combined: {r['combined']}\n")
    print(f"  TXT: {txt_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
