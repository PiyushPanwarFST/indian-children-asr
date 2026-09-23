"""
Baseline Fine-tuning — Full Whisper Seq2Seq Fine-tuning on ASER Dataset
========================================================================

WHAT THIS SCRIPT DOES:
    Fine-tunes the COMPLETE Whisper model (encoder + decoder) on ASER dataset
    using standard seq2seq training with cross-entropy loss.

    This is how people ACTUALLY fine-tune Whisper in practice — using the full
    encoder-decoder architecture with Whisper's native BPE tokenizer and
    autoregressive decoding. NOT just the encoder + CTC head.

    This provides a fair baseline: "What if you just fine-tuned Whisper on ASER
    without any knowledge distillation or dual-encoder fusion?"

SUPPORTED MODELS:
    1. whisper-small — OpenAI Whisper Small (242M params)
       Tests: "What if we just fine-tuned a general ASR model on ASER?"

    2. kid-whisper — Kid-Whisper Medium (769M params)
       Tests: "What if we just fine-tuned a children's ASR model on ASER?"

ARCHITECTURE:
    ┌───────────────────────────────────────────────────────────────┐
    │  Audio (.wav) — ALL languages (Hindi + Marathi + English)     │
    │       ↓                                                      │
    │  Mel Spectrogram                                             │
    │       ↓                                                      │
    │  Whisper Encoder (ALL layers UNFROZEN)                       │
    │       ↓                                                      │
    │  Whisper Decoder (ALL layers UNFROZEN)                       │
    │    - autoregressive, teacher forcing during training         │
    │    - uses Whisper's native BPE tokenizer                     │
    │    - language token: <|hi|>, <|mr|>, <|en|>                  │
    │    - task token: <|transcribe|>                              │
    │       ↓                                                      │
    │  Cross-Entropy Loss ← tokenized ground truth                 │
    │       ↓                                                      │
    │  Backprop → entire encoder + decoder                         │
    └───────────────────────────────────────────────────────────────┘

    At inference: model.generate() with beam search / greedy decoding.

KEY DIFFERENCE FROM OUR PROPOSED METHOD:
    - Our method: 2 frozen encoders + lightweight gated fusion + CTC head
      (character-level, 85 tokens, 1.25M trainable)
    - This baseline: 1 full Whisper model, encoder+decoder, BPE tokenizer
      (subword-level, 51K tokens, ALL 242M+ trainable)

Prerequisites:
    - Splits: ASER-Dataset/splits/asr_train.csv, asr_dev.csv
    - HuggingFace model cached (run with internet first, or pre-cache)

Usage:
    # Whisper Small — test run
    python baseline_finetune.py --model whisper-small --test_clips 3000 --epochs 10

    # Whisper Small — full training
    python baseline_finetune.py --model whisper-small --epochs 25 --patience 7

    # Kid-Whisper Medium — full training
    python baseline_finetune.py --model kid-whisper --epochs 25 --patience 7 --grad_checkpoint
"""

import argparse
import csv
import gc
import os
import random
import sys
import time

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Baseline: full Whisper seq2seq fine-tuning")
parser.add_argument("--model", type=str, required=True,
                    choices=["whisper-small", "kid-whisper"],
                    help="Model to fine-tune: whisper-small or kid-whisper")
parser.add_argument("--verify", action="store_true",
                    help="Quick verify mode (100 clips, skips checkpoint saving)")
parser.add_argument("--test_clips", type=int, default=None,
                    help="Limit training clips for test runs")
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate (1e-5 standard for full Whisper fine-tuning)")
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--max_audio_sec", type=float, default=30.0)
parser.add_argument("--patience", type=int, default=7,
                    help="Early stopping patience on dev WER (0=disabled)")
parser.add_argument("--dropout", type=float, default=0.0,
                    help="Override model dropout (0.0=use model default)")
parser.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing for cross-entropy loss")
parser.add_argument("--grad_checkpoint", action="store_true",
                    help="Enable gradient checkpointing (saves memory)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path")
args = parser.parse_args()

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "whisper-small": {
        "hf_id": "openai/whisper-small",
        "description": "Whisper Small (242M params, general multilingual ASR)",
    },
    "kid-whisper": {
        "hf_id": "aadel4/kid-whisper-medium-en-myst",
        "description": "Kid-Whisper Medium (769M params, children's English ASR)",
    },
}

config = MODEL_CONFIGS[args.model]
HF_ID = config["hf_id"]

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "baselines" / args.model
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Language mapping for Whisper decoder prompts
LANG_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}
# Whisper language tokens — Hindi and Marathi use "hi" and "mr"
WHISPER_LANG_MAP = {"Hindi": "hindi", "Marathi": "marathi", "English": "english"}

# ── Add project root for utils import ────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.wer import normalize_text, compute_corpus_wer

print(f"{'=' * 70}")
print(f"  BASELINE: Full Whisper Seq2Seq Fine-tuning")
print(f"  Model: {args.model} — {config['description']}")
print(f"{'=' * 70}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load training data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading training data (all languages)")
print("=" * 70)


def load_clips_from_csv(csv_path, max_clips=None):
    clips = []
    skipped = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row.get("language", "")
            lang_code = LANG_MAP.get(lang)
            if lang_code is None:
                skipped += 1
                continue

            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = str(ASER_ROOT / audio_path)

            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename
            gt = row.get("transcript", row.get("que_text", row.get("text", "")))

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "clip_name": clip_uid,
                "duration_sec": float(row.get("duration_sec", 0)),
                "ground_truth": gt,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips, skipped


train_clips, skip_other = load_clips_from_csv(TRAIN_CSV)
hi_clips = [c for c in train_clips if c["lang_code"] == "hi"]
mr_clips = [c for c in train_clips if c["lang_code"] == "mr"]
en_clips = [c for c in train_clips if c["lang_code"] == "en"]
total_hrs = sum(c["duration_sec"] for c in train_clips) / 3600

print(f"  Total: {len(train_clips)} clips ({total_hrs:.1f}h)")
print(f"    Hindi:   {len(hi_clips)} clips")
print(f"    Marathi: {len(mr_clips)} clips")
print(f"    English: {len(en_clips)} clips")
print(f"  Skipped: {skip_other} unknown language")

if len(train_clips) == 0:
    print("\n  ERROR: No training clips found!")
    exit(1)

# Dev set
dev_clips, _ = load_clips_from_csv(DEV_CSV)
print(f"  Dev set: {len(dev_clips)} clips")

# Clip limiting
N = args.test_clips
if N is None and args.verify:
    N = 100

if N and N < len(train_clips):
    random.seed(args.seed)
    total_all = len(train_clips)
    lang_clips = {"hi": hi_clips, "mr": mr_clips, "en": en_clips}
    sampled = []
    remaining = N
    for lc, lclips in lang_clips.items():
        n_lang = min(int(N * len(lclips) / total_all), len(lclips))
        if lclips:
            sampled.extend(random.sample(lclips, n_lang))
            remaining -= n_lang
    if remaining > 0:
        leftover = [c for c in train_clips if c not in sampled]
        sampled.extend(random.sample(leftover, min(remaining, len(leftover))))
    train_clips = sampled
    random.shuffle(train_clips)
    hi_n = sum(1 for c in train_clips if c["lang_code"] == "hi")
    mr_n = sum(1 for c in train_clips if c["lang_code"] == "mr")
    en_n = sum(1 for c in train_clips if c["lang_code"] == "en")
    label = "VERIFY MODE" if args.verify else "LIMITED"
    print(f"\n  {label}: {len(train_clips)} clips ({hi_n} Hindi + {mr_n} Marathi + {en_n} English)")

    if args.verify and dev_clips:
        dev_n = min(100, len(dev_clips))
        random.seed(args.seed + 1)
        dev_clips = random.sample(dev_clips, dev_n)
        print(f"    Dev subset: {len(dev_clips)} clips")

print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load full Whisper model (encoder + decoder)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 2: Loading FULL Whisper model: {HF_ID}")
print("=" * 70)

import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Load processor (feature extractor + tokenizer) ──────────────────────────
print(f"\n  Loading processor (feature extractor + BPE tokenizer)...")
processor = WhisperProcessor.from_pretrained(HF_ID)
tokenizer = processor.tokenizer
feat_extractor = processor.feature_extractor

print(f"  Tokenizer vocab: {tokenizer.vocab_size} BPE tokens")

# ── Load full model (encoder + decoder) ─────────────────────────────────────
print(f"\n  Loading full model: {HF_ID}")
model = WhisperForConditionalGeneration.from_pretrained(HF_ID).to(DEVICE)

# ALL parameters unfrozen
for param in model.parameters():
    param.requires_grad = True

# Override dropout if specified
if args.dropout > 0:
    model.config.dropout = args.dropout
    model.config.attention_dropout = args.dropout
    model.config.activation_dropout = args.dropout
    # Apply to all dropout layers in the model
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout
    print(f"  Dropout overridden to: {args.dropout}")

model.train()

total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.model.decoder.parameters())

print(f"  Total params:   {total_params:,} (ALL UNFROZEN)")
print(f"    Encoder:      {encoder_params:,}")
print(f"    Decoder:      {decoder_params:,}")

if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
    print(f"  Gradient checkpointing: ENABLED")

# ── Resume from checkpoint ──────────────────────────────────────────────────
start_epoch = 0
best_dev_wer = float("inf")

if args.resume:
    ckpt_path = PROJECT_ROOT / args.resume
    print(f"\n  [RESUME] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_dev_wer = ckpt.get("dev_wer", float("inf"))
    print(f"    Resuming from epoch {start_epoch}, best dev WER: {best_dev_wer:.2f}%")
    del ckpt
    gc.collect()

print(f"\n  ┌─ Parameter Summary ──────────────────────────────────────────")
print(f"  │ Encoder:  {encoder_params:,} (ALL TRAINABLE)")
print(f"  │ Decoder:  {decoder_params:,} (ALL TRAINABLE)")
print(f"  │ Total:    {total_params:,}")
print(f"  │ Decoding: Whisper BPE tokenizer ({tokenizer.vocab_size} tokens)")
print(f"  │ Method:   Seq2Seq with teacher forcing + cross-entropy loss")
print(f"  └──────────────────────────────────────────────────────────────")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  GPU Memory: {allocated:.0f} MB allocated")

print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Helper functions
# ══════════════════════════════════════════════════════════════════════════════

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


def prepare_labels(text, language):
    """
    Tokenize ground truth text using Whisper's BPE tokenizer.
    Sets language and task tokens for proper decoder prompting.
    Returns token IDs with proper special tokens.
    Truncates to 448 tokens (Whisper's max decoder position embeddings).
    """
    whisper_lang = WHISPER_LANG_MAP.get(language, "english")

    # Set language for tokenizer
    tokenizer.set_prefix_tokens(language=whisper_lang, task="transcribe")

    # Tokenize the text
    labels = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

    # Whisper decoder max position embeddings = 448
    if labels.shape[0] > 448:
        labels = labels[:448]

    return labels


def get_forced_decoder_ids(language):
    """Get forced decoder IDs for language-specific generation."""
    whisper_lang = WHISPER_LANG_MAP.get(language, "english")
    forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
    return forced_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Component verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Component verification")
print("=" * 70)

test_clip = train_clips[0]
test_audio = load_audio(test_clip["audio_path"], max_sec=15.0)
print(f"  Test: {test_clip['clip_name']} ({len(test_audio)/SAMPLE_RATE:.1f}s, {test_clip['language']})")

# Mel features
mel = feat_extractor(test_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
print(f"  Mel: {tuple(mel.input_features.shape)}")

# Prepare labels
gt_text = test_clip["ground_truth"]
labels = prepare_labels(gt_text, test_clip["language"])
print(f"  GT: '{gt_text[:60]}...'")
print(f"  Labels: {labels.shape} tokens")

# Forward pass with labels (teacher forcing)
with torch.no_grad():
    outputs = model(
        input_features=mel.input_features.to(DEVICE),
        labels=labels.unsqueeze(0).to(DEVICE),
    )
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")

# Test generation
with torch.no_grad():
    forced_decoder_ids = get_forced_decoder_ids(test_clip["language"])
    generated_ids = model.generate(
        mel.input_features.to(DEVICE),
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=225,
    )
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  Generated: '{decoded[:80]}...'")

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ALL VERIFIED\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 5: Training setup")
print("=" * 70)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

num_epochs = args.epochs
total_steps_est = len(train_clips) * num_epochs
warmup_steps = min(args.warmup_steps, total_steps_est // 4)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Load optimizer state if resuming
if args.resume:
    resume_ckpt = torch.load(PROJECT_ROOT / args.resume, map_location=DEVICE, weights_only=False)
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
    del resume_ckpt
    gc.collect()

print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.01)")
print(f"  Scheduler: linear warmup ({warmup_steps} steps) + cosine decay")
print(f"  Epochs: {num_epochs} | Clips: {len(train_clips)}")
print(f"  Total trainable: {total_params:,} params (FULL encoder + decoder)")
print(f"  Training method: Seq2Seq with teacher forcing")
if args.patience > 0:
    print(f"  Early stopping: patience={args.patience} (on dev WER)")
print()


# ── Dev evaluation function ──────────────────────────────────────────────────

def evaluate_dev(dev_clips_list):
    """
    Evaluate on dev set using model.generate() (autoregressive decoding).
    Returns: (dev_wer, dev_loss, sample_preds, lang_wers)
    """
    model.eval()

    all_refs = []
    all_hyps = []
    losses = []
    lang_refs = {"hi": [], "mr": [], "en": []}
    lang_hyps = {"hi": [], "mr": [], "en": []}

    with torch.no_grad():
        for clip in tqdm(dev_clips_list, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                audio_np = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(audio_np) < 8000:
                    continue

                # Mel features
                mel = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                input_features = mel.input_features.to(DEVICE)

                # Compute loss (teacher forcing)
                labels = prepare_labels(clip["ground_truth"], clip["language"])
                outputs = model(
                    input_features=input_features,
                    labels=labels.unsqueeze(0).to(DEVICE),
                )
                if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                    losses.append(outputs.loss.item())

                # Generate prediction (autoregressive)
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

                if DEVICE == "cuda":
                    del input_features, outputs, generated_ids
                    torch.cuda.empty_cache()

            except Exception as e:
                continue

    model.train()

    dev_wer = compute_corpus_wer(all_refs, all_hyps) * 100 if all_refs else 999.0
    dev_loss = np.mean(losses) if losses else 0.0

    lang_wers = {}
    for lc in ("hi", "mr", "en"):
        if lang_refs[lc]:
            lang_wers[lc] = compute_corpus_wer(lang_refs[lc], lang_hyps[lc]) * 100
        else:
            lang_wers[lc] = None

    return dev_wer, dev_loss, list(zip(all_refs, all_hyps)), lang_wers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
mode_str = f"VERIFICATION ({len(train_clips)} clips)" if args.verify else "Full training"
print(f"STEP 6: {mode_str} — {args.model} (seq2seq)")
print("=" * 70)

random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

epoch_stats = []
global_step = 0
patience_counter = 0
training_start = time.time()

model.train()

for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
    random.shuffle(train_clips)

    ep_losses = []
    ep_start = time.time()
    skipped = 0

    pbar = tqdm(train_clips, desc=f"Epoch {epoch}",
                unit="clip", bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            # ── Load audio ──
            audio_np = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(audio_np) < 8000:
                skipped += 1
                continue

            # ── Mel features ──
            mel = feat_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = mel.input_features.to(DEVICE)

            # ── Prepare labels (BPE tokenized ground truth) ──
            labels = prepare_labels(clip["ground_truth"], clip["language"])
            labels = labels.unsqueeze(0).to(DEVICE)

            # ── Forward pass (teacher forcing, cross-entropy loss) ──
            if args.label_smoothing > 0:
                outputs = model(
                    input_features=input_features,
                    decoder_input_ids=labels[:, :-1],
                )
                logits = outputs.logits
                target = labels[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=args.label_smoothing,
                )
            else:
                outputs = model(
                    input_features=input_features,
                    labels=labels,
                )
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue

            # ── Backward + step ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            ep_losses.append(loss.item())

            # ── Cleanup ──
            if DEVICE == "cuda":
                del input_features, labels, outputs
                torch.cuda.empty_cache()

            # Progress bar
            if ep_losses:
                avg_recent = np.mean(ep_losses[-20:])
                pbar.set_postfix({
                    "loss": f"{ep_losses[-1]:.3f}",
                    "avg": f"{avg_recent:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                })

        except Exception as e:
            print(f"\n  ERROR step {global_step}: {clip['clip_name']}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

    pbar.close()

    # ── Epoch summary ────────────────────────────────────────────────────
    ep_time = time.time() - ep_start
    avg_loss = np.mean(ep_losses) if ep_losses else 0

    print(f"\n  ┌─ Epoch {epoch} ─────────────────────────────────────────────────")
    print(f"  │ CE loss:    {avg_loss:.4f}")
    print(f"  │ Steps: {len(ep_losses)} ({skipped} skipped) | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
    print(f"  │ LR: {scheduler.get_last_lr()[0]:.2e}")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  │ Peak GPU: {peak:.0f} MB")
    print(f"  └──────────────────────────────────────────────────────────────")

    # ── Dev evaluation ───────────────────────────────────────────────────
    dev_wer, dev_loss, dev_preds, lang_wers = 999.0, 0.0, [], {}
    if dev_clips:
        dev_wer, dev_loss, dev_preds, lang_wers = evaluate_dev(dev_clips)
        print(f"\n  ┌─ Dev Results ───────────────────────────────────────────────")
        print(f"  │ WER:  {dev_wer:.2f}%")
        for lc, lname in [("hi", "Hindi"), ("mr", "Marathi"), ("en", "English")]:
            if lang_wers.get(lc) is not None:
                print(f"  │   {lname}: {lang_wers[lc]:.2f}%")
        print(f"  │ Loss: {dev_loss:.4f}")
        print(f"  └──────────────────────────────────────────────────────────────")

        if dev_preds:
            n_show = min(3, len(dev_preds))
            print(f"\n  Sample predictions:")
            for ref, hyp in dev_preds[:n_show]:
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")
                print()

    # Save stats
    epoch_stats.append({
        "epoch": epoch,
        "train_loss": avg_loss,
        "dev_wer": dev_wer, "dev_loss": dev_loss,
        "lang_wers": lang_wers,
    })

    # ── Checkpoint ───────────────────────────────────────────────────────
    if not args.verify:
        ckpt_data = {
            "epoch": epoch,
            "model_name": args.model,
            "hf_id": HF_ID,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_loss,
            "dev_wer": dev_wer,
            "dev_loss": dev_loss,
            "lang_wers": lang_wers,
            "global_step": global_step,
            "args": vars(args),
        }

        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer
            patience_counter = 0
            torch.save(ckpt_data, CKPT_DIR / "best_wer.pt")
            print(f"  NEW BEST DEV WER: {dev_wer:.2f}% -> saved best_wer.pt")
        else:
            patience_counter += 1
            print(f"  Early stopping: {patience_counter}/{args.patience} "
                  f"(best: {best_dev_wer:.2f}%)")
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n  EARLY STOPPING at epoch {epoch}")
                torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch}.pt")
                break

        torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch}.pt")
        print(f"  Saved: {CKPT_DIR / f'epoch_{epoch}.pt'}")

    print()

total_time = time.time() - training_start

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Final results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 7: Results — {args.model} seq2seq fine-tuning")
print("=" * 70)

if epoch_stats:
    print(f"\n  {'Epoch':<8} {'Loss':>8} {'DevWER':>10} {'DevLoss':>8} {'Hindi':>8} {'Marathi':>8} {'English':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for s in epoch_stats:
        wer_str = f"{s['dev_wer']:.2f}%" if s['dev_wer'] < 999 else "N/A"
        hi_str = f"{s['lang_wers'].get('hi', 0):.2f}%" if s['lang_wers'].get('hi') is not None else "N/A"
        mr_str = f"{s['lang_wers'].get('mr', 0):.2f}%" if s['lang_wers'].get('mr') is not None else "N/A"
        en_str = f"{s['lang_wers'].get('en', 0):.2f}%" if s['lang_wers'].get('en') is not None else "N/A"
        print(f"  {s['epoch']:<8} {s['train_loss']:>8.4f} {wer_str:>10} {s['dev_loss']:>8.4f} "
              f"{hi_str:>8} {mr_str:>8} {en_str:>8}")

    if best_dev_wer < 999:
        print(f"\n  Best dev WER: {best_dev_wer:.2f}%")
        print(f"\n  Comparison:")
        print(f"    Whisper Small (zero-shot):              126.83% (no fine-tuning)")
        print(f"    Acoustic branch (Kid-Whisper dist):      19.96% (test)")
        print(f"    Combined (gated fusion):                 18.92% (test)")
        print(f"    {args.model} seq2seq fine-tuned (this):  {best_dev_wer:.2f}% (dev)")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Total steps: {global_step}")
if not args.verify:
    print(f"  Checkpoints: {CKPT_DIR}/")
print(f"\n{'=' * 70}")
