"""
Combined Branch Evaluation — Full Comparison on Test Set
=========================================================

PURPOSE:
    Evaluate the combined (dual-encoder fusion) model on test set with
    side-by-side comparison:
      - Ground Truth
      - Baseline Whisper Small (no fine-tuning)
      - Acoustic branch only (joint-trained encoder + CTC)
      - Combined (acoustic + semantic fusion + CTC)
    Plus corpus-level WER per language and overall.

MODELS:
    1. Baseline: Whisper Small (244M, no fine-tuning)
    2. Acoustic only: Joint-trained encoder + CTC head (19.97% WER)
    3. Combined: Acoustic + Semantic encoders → GatedFusion → CTC head

Usage:
    # Quick test (10 clips per language)
    python step2_combined_evaluate.py --per_lang 10

    # Full test set
    python step2_combined_evaluate.py --per_lang 9999 --eval_set test
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

# ─── Project imports ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

# ─── Config ───
ASER_ROOT = PROJECT_ROOT / "ASER-Dataset"
DEV_CSV = ASER_ROOT / "splits" / "asr_dev.csv"
TEST_CSV = ASER_ROOT / "splits" / "asr_test.csv"
VOCAB_PATH = ASER_ROOT / "vocab.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "combined_branch" / "01_feature_fusion"

WHISPER_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30


# ─── Vocabulary ───
def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)


def ctc_greedy_decode(logits, idx_to_char):
    predicted_ids = logits.argmax(dim=-1).tolist()
    decoded_ids = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id:
            if idx != 0:
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


# ─── Audio ───
def load_audio(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
    return wav.squeeze(0).numpy()


def compute_real_frames(num_samples):
    return min(num_samples // 160 // 2, 1500)


# ─── GatedFusion (must match training) ───
class GatedFusion(nn.Module):
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


# ─── Model Loading ───
def load_combined_model(combined_ckpt, acoustic_ckpt, semantic_ckpt, vocab_size):
    """Load both frozen encoders + trained fusion + CTC head."""
    print(f"\n  Loading combined model...")

    # Acoustic encoder
    print(f"  [ACOUSTIC] Loading from {acoustic_ckpt}")
    whisper_a = WhisperModel.from_pretrained(WHISPER_ID)
    acoustic_encoder = whisper_a.encoder.to(DEVICE)
    a_ckpt = torch.load(acoustic_ckpt, map_location=DEVICE, weights_only=False)
    if "encoder_state_dict" in a_ckpt:
        acoustic_encoder.load_state_dict(a_ckpt["encoder_state_dict"])
    elif "model_state_dict" in a_ckpt:
        state = {k[len("encoder."):]: v for k, v in a_ckpt["model_state_dict"].items() if k.startswith("encoder.")}
        acoustic_encoder.load_state_dict(state)
    acoustic_encoder.eval()
    for p in acoustic_encoder.parameters():
        p.requires_grad = False
    del whisper_a, a_ckpt

    # Semantic encoder
    print(f"  [SEMANTIC] Loading from {semantic_ckpt}")
    whisper_s = WhisperModel.from_pretrained(WHISPER_ID)
    semantic_encoder = whisper_s.encoder.to(DEVICE)
    s_ckpt = torch.load(semantic_ckpt, map_location=DEVICE, weights_only=False)
    semantic_encoder.load_state_dict(s_ckpt["encoder_state_dict"])
    semantic_encoder.eval()
    for p in semantic_encoder.parameters():
        p.requires_grad = False
    del whisper_s, s_ckpt

    # Fusion + CTC head from combined checkpoint
    print(f"  [COMBINED] Loading from {combined_ckpt}")
    c_ckpt = torch.load(combined_ckpt, map_location=DEVICE, weights_only=False)
    fusion = GatedFusion(dim=768, dropout=0.0).to(DEVICE)  # no dropout at eval
    fusion.load_state_dict(c_ckpt["fusion_state_dict"])
    fusion.eval()

    ctc_head = nn.Linear(768, vocab_size).to(DEVICE)
    ctc_head.load_state_dict(c_ckpt["ctc_head_state_dict"])
    ctc_head.eval()

    epoch = c_ckpt.get("epoch", "?")
    dev_wer = c_ckpt.get("dev_wer", 0)
    print(f"    Epoch: {epoch}, Dev WER: {dev_wer:.2f}%")
    del c_ckpt

    return acoustic_encoder, semantic_encoder, fusion, ctc_head


def load_acoustic_only_model(acoustic_ckpt, vocab_size):
    """Load acoustic-only model for comparison."""
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


# ─── Transcription functions ───
def transcribe_baseline(model, processor, audio_np, language):
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
    inputs = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    real_frames = compute_real_frames(len(audio_np) * SAMPLE_RATE // SAMPLE_RATE)  # already in samples
    real_frames = compute_real_frames(len(audio_np))
    with torch.no_grad():
        features = encoder(inputs.input_features.to(DEVICE)).last_hidden_state
        logits = ctc_head(features)
    return ctc_greedy_decode(logits[0, :real_frames, :], idx_to_char)


def transcribe_combined(acoustic_enc, semantic_enc, fusion, ctc_head, feat_extractor, audio_np, idx_to_char):
    inputs = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    real_frames = compute_real_frames(len(audio_np))
    mel_gpu = inputs.input_features.to(DEVICE)
    with torch.no_grad():
        a_feat = acoustic_enc(mel_gpu).last_hidden_state[:, :real_frames, :]
        s_feat = semantic_enc(mel_gpu).last_hidden_state[:, :real_frames, :]
        fused = fusion(a_feat, s_feat)
        logits = ctc_head(fused)
    return ctc_greedy_decode(logits[0], idx_to_char)


# ─── Data loading ───
def load_clips(csv_path, per_lang=10):
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


# ─── Main ───
def main():
    parser = argparse.ArgumentParser(description="Combined Branch Evaluation")
    parser.add_argument("--per_lang", type=int, default=10)
    parser.add_argument("--eval_set", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--combined_ckpt", type=str, default="checkpoints/combined/best_wer.pt")
    parser.add_argument("--acoustic_ckpt", type=str, default="checkpoints/joint/joint_best_wer.pt")
    parser.add_argument("--semantic_ckpt", type=str, default="checkpoints/semantic_mse/best_dev.pt")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline Whisper (saves memory)")
    args = parser.parse_args()

    combined_ckpt = str(PROJECT_ROOT / args.combined_ckpt)
    acoustic_ckpt = str(PROJECT_ROOT / args.acoustic_ckpt)
    semantic_ckpt = str(PROJECT_ROOT / args.semantic_ckpt)

    if not os.path.exists(combined_ckpt):
        print(f"ERROR: Combined checkpoint not found: {combined_ckpt}")
        return

    # Load vocab
    print("Loading vocabulary...")
    char_to_idx, idx_to_char, vocab_size = load_vocab(str(VOCAB_PATH))
    print(f"  Vocab: {vocab_size} tokens")

    # Load clips
    eval_csv = str(TEST_CSV) if args.eval_set == "test" else str(DEV_CSV)
    clips, all_clips = load_clips(eval_csv, per_lang=args.per_lang)
    print(f"\n  Evaluating on {len(clips)} clips ({args.eval_set} set)")
    for lang in sorted(all_clips.keys()):
        n = min(args.per_lang, len(all_clips[lang]))
        print(f"    {lang}: {n} clips")

    # Load models
    print("\n" + "=" * 70)
    print("Loading models...")
    print("=" * 70)

    from transformers import WhisperFeatureExtractor
    feat_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

    # Combined model
    acoustic_enc, semantic_enc, fusion, ctc_head = load_combined_model(
        combined_ckpt, acoustic_ckpt, semantic_ckpt, vocab_size)

    # Acoustic-only model (for comparison)
    acoustic_only_enc, acoustic_only_ctc = load_acoustic_only_model(acoustic_ckpt, vocab_size)

    # Baseline Whisper
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

    # Evaluate
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

            # Combined
            combined_pred = transcribe_combined(
                acoustic_enc, semantic_enc, fusion, ctc_head, feat_extractor, audio_np, idx_to_char)

            # Acoustic only
            acoustic_pred = transcribe_acoustic_only(
                acoustic_only_enc, acoustic_only_ctc, feat_extractor, audio_np, idx_to_char)

            # Baseline
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

    # Compute WERs
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
        all_preds = [r[preds_key.split()[0] if " " in preds_key else preds_key] for r in results]
        wer = compute_corpus_wer(all_refs, all_preds) * 100
        print(f"  {model_name:<25} {wer:>9.2f}%", end="")
        for lang in sorted(lang_results.keys()):
            lr = lang_results[lang]
            lang_wer = compute_corpus_wer(lr["refs"], lr[preds_key.split()[0] if " " in preds_key else preds_key]) * 100
            print(f" {lang_wer:>9.2f}%", end="")
        print()

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_clips = len(results)

    # CSV
    csv_path = OUTPUT_DIR / f"combined_eval_{args.eval_set}_{n_clips}clips.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "language", "ground_truth", "baseline", "acoustic", "combined"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV: {csv_path}")

    # Summary
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
