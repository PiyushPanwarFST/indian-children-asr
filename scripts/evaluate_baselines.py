"""
Baseline Evaluation — All Models, One Standard WER
====================================================
Evaluates ALL baseline models on asr_test.csv using the SAME WER method.
This is the SINGLE SOURCE OF TRUTH for baseline numbers.

Models:
  1. Whisper Small (openai/whisper-small) — multilingual, language hint used
  2. Kid-Whisper Medium MyST (aadel4/kid-whisper-medium-en-myst) — English-only, NO language hint
  3. IndicConformer 600M (ai4bharat/indic-conformer-600m-multilingual) — Hindi/Marathi only

WER method: corpus-level via jiwer (standard for ASR papers)
Normalization: lowercase + remove punctuation (via scripts/utils/wer.py)

Usage:
    python scripts/evaluate_baselines.py                          # all models
    python scripts/evaluate_baselines.py --model whisper-small    # one model
    python scripts/evaluate_baselines.py --max_clips 10           # quick test
"""

import argparse
import csv
import os
import sys
import time
import warnings
import logging

# Suppress noisy warnings from transformers and onnxruntime
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import torchaudio
from tqdm import tqdm

# Add project root to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.wer import normalize_text, compute_corpus_wer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TEST_CSV = "ASER-Dataset/splits/asr_test.csv"
OUTPUT_DIR = "results/baselines"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # skip clips longer than 30s


# ─────────────────────────────────────────────
# AUDIO LOADING (same for all models)
# ─────────────────────────────────────────────
def load_audio(path):
    """
    Load audio file and prepare it for Whisper.

    Input:  file path (any sample rate, mono or stereo)
    Output: 1D numpy array of audio samples at 16kHz mono

    Why 16kHz? Whisper was trained on 16kHz audio — giving it
    48kHz or 44.1kHz audio would produce wrong mel spectrograms.
    """
    wav, sr = torchaudio.load(path)           # wav shape: (channels, samples), sr = sample rate
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)  # convert to 16kHz
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)   # stereo (2 channels) → mono (1 channel)
    wav = wav[:, :MAX_DURATION * SAMPLE_RATE] # trim to max 30 seconds (30 * 16000 = 480000 samples)
    return wav.squeeze(0).numpy()             # shape: (num_samples,) — 1D numpy array


def load_test_clips(csv_path, max_clips=None):
    """Load test clips from CSV, skip clips > 30s."""
    clips = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["duration_sec"]) > MAX_DURATION:
                continue
            clips.append(row)
    if max_clips:
        clips = clips[:max_clips]
    return clips


# ─────────────────────────────────────────────
# MODEL 1: WHISPER SMALL (multilingual)
# ─────────────────────────────────────────────
def evaluate_whisper_small(clips):
    """
    openai/whisper-small (244M params)
    Multilingual model — supports 99 languages including Hindi, Marathi, English.
    This is our STUDENT base model (before any training).
    """
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print("  Loading openai/whisper-small...")

    # Processor has 2 parts:
    #   - feature_extractor: converts raw audio → mel spectrogram (input for encoder)
    #   - tokenizer: converts token IDs → text (output from decoder)
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # The full model has encoder + decoder
    #   - encoder: mel spectrogram → hidden features (768-dim vectors per frame)
    #   - decoder: hidden features → text tokens (autoregressive, one token at a time)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.to(DEVICE).eval()  # .eval() = inference mode, turns off dropout
    print(f"  Loaded on {DEVICE}")

    lang_map = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
    predictions = []

    for clip in tqdm(clips, desc="  Whisper Small", unit="clip"):

        # STEP 1: Load audio file → numpy array (16kHz mono)
        # Input:  audio file path
        # Output: 1D numpy array, e.g. shape (48000,) for 3 seconds of audio
        audio_np = load_audio(clip["audio_path"])

        # STEP 2: Convert audio → mel spectrogram (what encoder expects)
        # Input:  numpy audio array
        # Output: tensor of shape (1, 80, 3000) — 80 mel bands, 3000 time frames
        lang_code = lang_map.get(clip["language"], "en")
        inputs = processor.feature_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")

        # STEP 3: Create forced decoder starting tokens
        # This tells the decoder: "generate text in Hindi" or "generate text in English"
        # Without this, Whisper might guess the wrong language
        # Input:  language code like "hi"
        # Output: list of (position, token_id) pairs, e.g. [(1, 50266), (2, 50359)]
        #         position 1 = <|hi|> token, position 2 = <|transcribe|> token
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")

        # STEP 4: Forward pass — run the full model (encoder + decoder)
        # "Forward pass" = pass input through all layers of the model to get output
        # torch.no_grad() = don't track gradients (saves memory, we're not training)
        #
        # What happens inside model.generate():
        #   1. Encoder reads mel spectrogram → produces hidden features
        #   2. Decoder starts with forced tokens (<|hi|>, <|transcribe|>)
        #   3. Decoder generates next token based on encoder output + previous tokens
        #   4. Repeat step 3 until <|endoftext|> token or max_new_tokens reached
        #
        # Input:  mel spectrogram tensor (1, 80, 3000)
        # Output: token IDs tensor, e.g. (1, 25) — 1 batch, 25 tokens generated
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features.to(DEVICE),
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,  # max 225 tokens output (covers longest clips)
            )

        # STEP 5: Convert token IDs → human readable text
        # Input:  tensor of token IDs, e.g. [50258, 50266, 50359, 2494, 1371, ...]
        # Output: string like "बग़ीचे में पेड़ हैं"
        # skip_special_tokens=True removes <|startoftranscript|>, <|hi|>, etc.
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        predictions.append(text)

    # Free GPU memory after done with this model
    del model, processor
    torch.cuda.empty_cache()
    return predictions


# ─────────────────────────────────────────────
# MODEL 2: KID-WHISPER MEDIUM MyST (English-only)
# ─────────────────────────────────────────────
def evaluate_kid_whisper(clips):
    """
    aadel4/kid-whisper-medium-en-myst (769M params)
    Whisper Medium fine-tuned on MyST English children's speech (125 hours).
    This is our ACOUSTIC TEACHER — best on English children (37% WER in old benchmark).

    IMPORTANT — this is an English-only fine-tuned model:
      - MUST use English-only processor (whisper-medium.en)
      - MUST NOT pass language hints or forced_decoder_ids
      - If you use multilingual processor → garbage output (we verified this!)
      - For Hindi/Marathi audio, it will output English-sounding nonsense (expected)
    """
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print("  Loading aadel4/kid-whisper-medium-en-myst...")

    # English-only processor — different tokenizer than multilingual whisper-medium
    # whisper-medium.en has ~51K English tokens
    # whisper-medium (multilingual) has ~51K multilingual tokens
    # Kid-Whisper was fine-tuned with the English-only one, so we MUST use it
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")

    # Model weights from HuggingFace — same architecture as whisper-medium
    # but encoder+decoder were fine-tuned on MyST children's speech
    model = WhisperForConditionalGeneration.from_pretrained("aadel4/kid-whisper-medium-en-myst")
    model.to(DEVICE).eval()
    print(f"  Loaded on {DEVICE}")

    predictions = []

    for clip in tqdm(clips, desc="  Kid-Whisper Med", unit="clip"):

        # STEP 1: Load audio → numpy array
        audio_np = load_audio(clip["audio_path"])

        # STEP 2: Audio → mel spectrogram
        # Using processor() directly (shortcut) instead of processor.feature_extractor()
        # Both do the same thing: audio → mel spectrogram tensor
        inputs = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")

        # STEP 3: Forward pass — NO forced_decoder_ids!
        # English-only models don't have <|hi|> or <|mr|> tokens in their vocabulary
        # If we force those tokens, decoder gets confused → outputs garbage
        # Without language hint, decoder just generates in English (its only language)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features.to(DEVICE),
                max_new_tokens=225,
            )

        # STEP 4: Token IDs → text
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        predictions.append(text)

    del model, processor
    torch.cuda.empty_cache()
    return predictions


# ─────────────────────────────────────────────
# MODEL 3: INDICCONFORMER 600M (Hindi/Marathi only)
# ─────────────────────────────────────────────
def evaluate_indicconformer(clips):
    """
    ai4bharat/indic-conformer-600m-multilingual (600M params)
    CTC+RNNT model trained on 22 Indian languages by AI4Bharat (IIT Madras).
    This is our SEMANTIC TEACHER — best on Hindi (39%) and Marathi (44%).

    Key differences from Whisper:
      - Architecture: Conformer (conv + attention), NOT Transformer
      - Decoding: CTC/RNNT (frame-level), NOT autoregressive decoder
      - Languages: 22 Indian languages only — NO English support
      - Inference: runs via ONNX runtime, not PyTorch
      - Input: expects 2D tensor (1, samples), NOT numpy array
    """
    from transformers import AutoModel

    print("  Loading ai4bharat/indic-conformer-600m-multilingual...")

    # trust_remote_code=True because IndicConformer uses custom code
    # (not a standard HuggingFace architecture)
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    )
    print("  Loaded (ONNX-based)")

    lang_map = {"Hindi": "hi", "Marathi": "mr"}
    predictions = []

    for clip in tqdm(clips, desc="  IndicConformer", unit="clip"):
        lang = clip["language"]
        ic_lang = lang_map.get(lang)

        # IndicConformer does NOT support English — skip English clips
        if ic_lang is None:
            predictions.append("")
            continue

        # Load audio as 2D TENSOR (not numpy!)
        # IndicConformer expects shape (1, num_samples) — batch dimension required
        # This is different from Whisper which wants 1D numpy array
        wav, sr = torchaudio.load(clip["audio_path"])
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav[:, :MAX_DURATION * SAMPLE_RATE]
        # wav shape: (1, num_samples) — keep 2D, do NOT squeeze

        try:
            # model.forward() = forward pass = pass audio through all model layers
            # This is the same as model(wav, lang="hi") — just explicit syntax
            #
            # Inside forward pass:
            #   1. Audio → mel spectrogram (inside model)
            #   2. Conformer encoder → frame-level features
            #   3. CTC decoder → text (frame-by-frame, not autoregressive)
            #
            # Input:  2D tensor (1, num_samples) + language code
            # Output: string like "बगीचे में पेड़ हैं"
            text = model.forward(wav, lang=ic_lang)
            if not isinstance(text, str):
                text = str(text)
        except Exception as e:
            text = ""

        predictions.append(text.strip())

    del model
    return predictions


# ─────────────────────────────────────────────
# RESULTS COMPUTATION & REPORTING
# ─────────────────────────────────────────────
def compute_results(clips, predictions, model_name, skip_languages=None):
    """
    Compute corpus-level WER per language and overall.
    skip_languages: set of languages to exclude (e.g., {"English"} for IndicConformer)
    """
    skip_languages = skip_languages or set()

    # Group by language
    by_lang = {}
    for clip, pred in zip(clips, predictions):
        lang = clip["language"]
        if lang in skip_languages:
            continue
        if lang not in by_lang:
            by_lang[lang] = {"refs": [], "preds": [], "clips": []}
        by_lang[lang]["refs"].append(clip["transcript"])
        by_lang[lang]["preds"].append(pred)
        by_lang[lang]["clips"].append(clip)

    # Per-language corpus WER
    results = {}
    all_refs = []
    all_preds = []
    for lang in sorted(by_lang.keys()):
        data = by_lang[lang]
        lang_wer = compute_corpus_wer(data["refs"], data["preds"])
        results[lang] = {"wer": lang_wer, "n_clips": len(data["refs"])}
        all_refs.extend(data["refs"])
        all_preds.extend(data["preds"])

    # Overall corpus WER
    overall_wer = compute_corpus_wer(all_refs, all_preds)
    results["Overall"] = {"wer": overall_wer, "n_clips": len(all_refs)}

    return results, list(zip(clips, predictions))


def print_results_table(all_results):
    """Print comparison table."""
    languages = ["Hindi", "Marathi", "English", "Overall"]

    print(f"\n{'='*80}")
    print(f"  BASELINE RESULTS — Corpus-Level WER (standard metric)")
    print(f"  Lower is better. 0% = perfect, 100% = all words wrong")
    print(f"  Evaluated on: asr_test.csv | Device: {DEVICE}")
    print(f"{'='*80}\n")

    # Header
    print(f"  {'Model':<32}", end="")
    for lang in languages:
        print(f" {lang:>10}", end="")
    print()
    print(f"  {'-'*32}", end="")
    for _ in languages:
        print(f" {'-'*10}", end="")
    print()

    # Rows
    for model_name, results in all_results.items():
        print(f"  {model_name:<32}", end="")
        for lang in languages:
            if lang in results:
                print(f" {results[lang]['wer']:>9.2%}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    print(f"  {'-'*32}", end="")
    for _ in languages:
        print(f" {'-'*10}", end="")
    print()


def save_report(all_results, all_predictions, output_dir):
    """Save .txt report and per-model .csv files."""
    os.makedirs(output_dir, exist_ok=True)

    languages = ["Hindi", "Marathi", "English", "Overall"]

    # Save summary report
    report_path = os.path.join(output_dir, "baseline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BASELINE ASR EVALUATION REPORT\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Test set: asr_test.csv\n")
        f.write(f"WER method: corpus-level (jiwer) — standard for ASR papers\n")
        f.write(f"Normalization: lowercase + remove punctuation\n")
        f.write(f"Device: {DEVICE}\n\n")

        f.write("MODELS:\n")
        f.write("  1. Whisper Small       — openai/whisper-small (244M, multilingual)\n")
        f.write("  2. Kid-Whisper Medium  — aadel4/kid-whisper-medium-en-myst (769M, English-only)\n")
        f.write("  3. IndicConformer 600M — ai4bharat/indic-conformer-600m-multilingual (Hindi/Marathi)\n\n")

        f.write(f"{'='*80}\n")
        f.write("RESULTS — Corpus-Level WER\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"{'Model':<32}")
        for lang in languages:
            f.write(f" {lang:>10}")
        f.write("\n")
        f.write(f"{'-'*32}")
        for _ in languages:
            f.write(f" {'-'*10}")
        f.write("\n")

        for model_name, results in all_results.items():
            f.write(f"{model_name:<32}")
            for lang in languages:
                if lang in results:
                    f.write(f" {results[lang]['wer']:>9.2%}")
                else:
                    f.write(f" {'N/A':>10}")
            f.write("\n")

        f.write(f"{'-'*32}")
        for _ in languages:
            f.write(f" {'-'*10}")
        f.write("\n")

    print(f"\n  Report saved: {report_path}")

    # Save per-model CSV with all predictions
    for model_name, clip_preds in all_predictions.items():
        csv_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_predictions.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["audio_path", "language", "ground_truth", "prediction",
                             "gt_normalized", "pred_normalized"])
            for clip, pred in clip_preds:
                writer.writerow([
                    clip["audio_path"],
                    clip["language"],
                    clip["transcript"],
                    pred,
                    normalize_text(clip["transcript"]),
                    normalize_text(pred),
                ])
        print(f"  Predictions saved: {csv_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "whisper-small": {
        "fn": evaluate_whisper_small,
        "display": "Whisper Small (244M)",
        "skip_langs": set(),
    },
    "kid-whisper-medium": {
        "fn": evaluate_kid_whisper,
        "display": "Kid-Whisper Medium (769M)",
        "skip_langs": set(),  # evaluate on all, but expect garbage on Hindi/Marathi
    },
    "indicconformer": {
        "fn": evaluate_indicconformer,
        "display": "IndicConformer 600M",
        "skip_langs": {"English"},  # does not support English
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Limit clips for quick testing")
    args = parser.parse_args()

    # Load test clips
    clips = load_test_clips(TEST_CSV, max_clips=args.max_clips)
    print(f"\nLoaded {len(clips)} test clips")
    lang_counts = {}
    for c in clips:
        lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")
    print()

    # Decide which models to run
    if args.model == "all":
        models_to_run = list(MODEL_REGISTRY.keys())
    else:
        models_to_run = [args.model]

    all_results = {}
    all_predictions = {}

    for model_key in models_to_run:
        info = MODEL_REGISTRY[model_key]
        print(f"{'='*80}")
        print(f"  Evaluating: {info['display']}")
        print(f"{'='*80}")

        t0 = time.time()
        predictions = info["fn"](clips)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        results, clip_preds = compute_results(
            clips, predictions, model_key, skip_languages=info["skip_langs"]
        )
        all_results[info["display"]] = results
        all_predictions[info["display"]] = clip_preds

    # Print and save
    print_results_table(all_results)
    save_report(all_results, all_predictions, OUTPUT_DIR)


if __name__ == "__main__":
    main()
