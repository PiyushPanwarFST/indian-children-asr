"""
Baseline Fine-tuning Evaluation — Test Set WER
================================================

Evaluates a fine-tuned baseline Whisper model (seq2seq) on the ASER test set.

The baseline was trained using WhisperForConditionalGeneration with BPE tokenizer
and autoregressive decoding (NOT CTC). So evaluation uses model.generate() with
Whisper's native BPE tokenizer, matching how the model was trained.

Checkpoint format (from baseline_finetune.py):
    - model_state_dict: full WhisperForConditionalGeneration state dict
    - model_name: "whisper-small" or "kid-whisper"
    - hf_id: HuggingFace model ID
    - epoch, dev_wer, etc.

Usage:
    # Evaluate fine-tuned Whisper Small
    python baseline_finetune_evaluate.py --model whisper-small \
        --checkpoint checkpoints/baselines/whisper-small/best_wer.pt

    # Evaluate fine-tuned Kid-Whisper Medium
    python baseline_finetune_evaluate.py --model kid-whisper \
        --checkpoint checkpoints/baselines/kid-whisper/best_wer.pt
"""

import argparse
import csv
import os
import sys
import time

import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate fine-tuned baseline on test set")
parser.add_argument("--model", type=str, required=True,
                    choices=["whisper-small", "kid-whisper"],
                    help="Which model was fine-tuned")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best_wer.pt checkpoint")
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--save_predictions", action="store_true", default=True,
                    help="Save per-clip predictions to file")
args = parser.parse_args()

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "whisper-small": {
        "hf_id": "openai/whisper-small",
    },
    "kid-whisper": {
        "hf_id": "aadel4/kid-whisper-medium-en-myst",
    },
}

config = MODEL_CONFIGS[args.model]
HF_ID = config["hf_id"]

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TEST_CSV     = ASER_ROOT / "splits" / "asr_test.csv"
RESULTS_DIR  = PROJECT_ROOT / "results" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
WHISPER_LANG_MAP = {"Hindi": "hindi", "Marathi": "marathi", "English": "english"}

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"Device: {DEVICE}")
print(f"Model: {args.model} ({HF_ID})")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading test set")
print("=" * 70)

test_clips = []
with open(TEST_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang = row.get("language", "")
        lang_code = LANG_MAP.get(lang)
        if lang_code is None:
            continue
        audio_path = row["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(ASER_ROOT / audio_path)
        gt = row.get("transcript", row.get("que_text", row.get("text", "")))
        test_clips.append({
            "audio_path": audio_path,
            "language": lang,
            "lang_code": lang_code,
            "ground_truth": gt,
        })

hi_n = sum(1 for c in test_clips if c["lang_code"] == "hi")
mr_n = sum(1 for c in test_clips if c["lang_code"] == "mr")
en_n = sum(1 for c in test_clips if c["lang_code"] == "en")
print(f"  Total: {len(test_clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load model + checkpoint
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading model + checkpoint")
print("=" * 70)

import gc
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load processor (feature extractor + tokenizer)
print(f"  Loading processor: {HF_ID}")
processor = WhisperProcessor.from_pretrained(HF_ID)
tokenizer = processor.tokenizer
feat_extractor = processor.feature_extractor

# Load full model
print(f"  Loading full model: {HF_ID}")
model = WhisperForConditionalGeneration.from_pretrained(HF_ID).to(DEVICE)

# Load checkpoint
ckpt_path = PROJECT_ROOT / args.checkpoint
print(f"  Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
epoch = ckpt.get("epoch", "?")
dev_wer = ckpt.get("dev_wer", "?")
print(f"  Checkpoint from epoch {epoch}, dev WER: {dev_wer}%")
del ckpt
gc.collect()

model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,}")
print(f"  Decoding: model.generate() with Whisper BPE tokenizer")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU Memory: {allocated:.0f} MB")


def load_audio(path, max_sec=None):
    """Load audio -> 16kHz mono numpy array."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()
    if max_sec and len(wav) > int(max_sec * SAMPLE_RATE):
        wav = wav[:int(max_sec * SAMPLE_RATE)]
    return wav.numpy()


def get_forced_decoder_ids(language):
    """Get forced decoder IDs for language-specific generation."""
    whisper_lang = WHISPER_LANG_MAP.get(language, "english")
    forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
    return forced_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Evaluate on test set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Evaluating on test set (seq2seq decoding)")
print("=" * 70)

all_refs = []
all_hyps = []
all_predictions = []  # For saving per-clip predictions
lang_refs = {"hi": [], "mr": [], "en": []}
lang_hyps = {"hi": [], "mr": [], "en": []}
errors = 0

start_time = time.time()

with torch.no_grad():
    for clip in tqdm(test_clips, desc="  Test eval", unit="clip",
                     bar_format="{l_bar}{bar:30}{r_bar}"):
        try:
            audio_np = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(audio_np) < 8000:
                continue

            # Mel features
            mel = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = mel.input_features.to(DEVICE)

            # Generate prediction (autoregressive decoding)
            forced_decoder_ids = get_forced_decoder_ids(clip["language"])
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )
            predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            ref = normalize_text(clip["ground_truth"])
            hyp = normalize_text(predicted)

            if ref:
                all_refs.append(ref)
                all_hyps.append(hyp)
                lc = clip["lang_code"]
                if lc in lang_refs:
                    lang_refs[lc].append(ref)
                    lang_hyps[lc].append(hyp)

                # Store for saving
                all_predictions.append({
                    "lang": lc,
                    "ref": ref,
                    "hyp": hyp,
                })

            if DEVICE == "cuda":
                del input_features, generated_ids
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            continue

eval_time = time.time() - start_time


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 4: Test Results — {args.model} seq2seq fine-tuned")
print("=" * 70)

overall_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0

lang_wers = {}
for lc in ("hi", "mr", "en"):
    if lang_refs[lc]:
        lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
    else:
        lang_wers[lc] = None

print(f"\n  ┌─ {args.model} Fine-tuned — Test Set ───────────────────────────")
print(f"  │ Overall WER: {overall_wer:.2f}%")
if lang_wers.get("hi") is not None:
    print(f"  │   Hindi:   {lang_wers['hi']:.2f}%")
if lang_wers.get("mr") is not None:
    print(f"  │   Marathi: {lang_wers['mr']:.2f}%")
if lang_wers.get("en") is not None:
    print(f"  │   English: {lang_wers['en']:.2f}%")
print(f"  │ Clips evaluated: {len(all_refs)} | Errors: {errors}")
print(f"  │ Time: {eval_time:.0f}s ({eval_time/60:.1f}min)")
print(f"  └──────────────────────────────────────────────────────────────")

# ── Comparison table ─────────────────────────────────────────────────────────
hi_str = f"{lang_wers['hi']:.2f}%" if lang_wers.get('hi') is not None else "—"
mr_str = f"{lang_wers['mr']:.2f}%" if lang_wers.get('mr') is not None else "—"
en_str = f"{lang_wers['en']:.2f}%" if lang_wers.get('en') is not None else "—"

print(f"\n  ┌─ System Comparison (Test Set WER) ─────────────────────────────")
print(f"  │ {'System':<40} {'Overall':>8} {'Hindi':>8} {'Marathi':>8} {'English':>8}")
print(f"  │ {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
print(f"  │ {'Acoustic (Kid-Whisper dist.)':<40} {'19.96%':>8} {'15.88%':>8} {'29.96%':>8} {'21.63%':>8}")
print(f"  │ {'Semantic joint (Head 2, prof.)':<40} {'50.24%':>8} {'41.11%':>8} {'72.96%':>8} {'—':>8}")
print(f"  │ {'Combined (gated fusion, e21)':<40} {'14.72%':>8} {'10.55%':>8} {'25.27%':>8} {'17.04%':>8}")
print(f"  │ {f'{args.model} seq2seq fine-tuned':<40} {f'{overall_wer:.2f}%':>8} {hi_str:>8} {mr_str:>8} {en_str:>8}")
print(f"  └──────────────────────────────────────────────────────────────────")

# ── Save results ─────────────────────────────────────────────────────────────
results_file = RESULTS_DIR / f"{args.model}_test_results.txt"
with open(results_file, "w") as f:
    f.write(f"Baseline Fine-tuning Test Results: {args.model}\n")
    f.write(f"{'=' * 50}\n")
    f.write(f"Model: {HF_ID}\n")
    f.write(f"Checkpoint: {args.checkpoint}\n")
    f.write(f"Epoch: {epoch}\n")
    f.write(f"Dev WER: {dev_wer}%\n\n")
    f.write(f"Overall WER: {overall_wer:.2f}%\n")
    if lang_wers.get("hi") is not None:
        f.write(f"  Hindi:   {lang_wers['hi']:.2f}%\n")
    if lang_wers.get("mr") is not None:
        f.write(f"  Marathi: {lang_wers['mr']:.2f}%\n")
    if lang_wers.get("en") is not None:
        f.write(f"  English: {lang_wers['en']:.2f}%\n")
    f.write(f"\nClips evaluated: {len(all_refs)}\n")
    f.write(f"Errors: {errors}\n")

print(f"\n  Results saved: {results_file}")

# ── Save predictions ────────────────────────────────────────────────────────
if args.save_predictions and all_predictions:
    pred_file = RESULTS_DIR / f"{args.model}_test_predictions.txt"
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write(f"=== {args.model} seq2seq fine-tuned: Test Predictions ===\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Overall WER: {overall_wer:.2f}%\n\n")
        for i, pred in enumerate(all_predictions):
            per_wer = compute_corpus_wer([pred["ref"]], [pred["hyp"]]) * 100
            f.write(f"[{i+1}] LANG={pred['lang']} WER={per_wer:.1f}%\n")
            f.write(f"  GT:   {pred['ref']}\n")
            f.write(f"  PRED: {pred['hyp']}\n\n")
    print(f"  Predictions saved: {pred_file}")

# ── Sample predictions ───────────────────────────────────────────────────────
print(f"\n  Sample predictions:")
n_show = min(5, len(all_refs))
for i in range(n_show):
    print(f"    REF: {all_refs[i]}")
    print(f"    HYP: {all_hyps[i]}")
    print()

print(f"{'=' * 70}")
