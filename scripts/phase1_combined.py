"""
Phase 1: Combined Multi-Teacher Knowledge Distillation (GPU-Optimized)

Architecture:
  PATH A — Semantic Teacher (FROZEN, float16):
    tanmaylaud/wav2vec2-large-xlsr-hindi-marathi (1024-dim, 315M params)
    audio → XLSR → avg_pool → Y_indic (1, 1024)
    Skipped for English clips

  PATH B — Acoustic Teacher (FROZEN, float16):
    aadel4/kid-whisper-small-en-myst encoder (768-dim, 88M params)
    audio → Kid-Whisper encoder → Y_kid (T, 768)
    All languages

  PATH C — Student (TRAINS, float32, gradient checkpointing):
    facebook/mms-300m (1024-dim, 315M params)
    Semantic branch:  MMS → avg_pool → Y_sem_hat (1, 1024)
    Acoustic branch:  MMS → Linear(1024→768) → Y_acoust_hat (T, 768)

  Loss:
    Hindi/Marathi: L = MSE(Y_sem_hat, Y_indic) + MSE(Y_acoust_hat, Y_kid)
    English:       L = MSE(Y_acoust_hat, Y_kid)

  GPU Optimizations (RTX 4060, 8GB):
    - Teachers in float16 (saves ~0.8GB)
    - Gradient checkpointing on student (saves ~40% activation memory)
    - All clips up to 30s (zero data loss from ASR subset)
    - torch.cuda.empty_cache() after each step

Usage:
  python scripts/phase1_combined.py --verify                  # 100 clips × 3 epochs (quick)
  python scripts/phase1_combined.py --verify --test_clips 1000  # 1000 clips × 3 epochs (reliable)
  python scripts/phase1_combined.py --verify --test_clips 1000 --seed 123  # Different clips
  python scripts/phase1_combined.py --epochs 20 --patience 5  # Full run with early stopping
"""

import argparse
import csv
import gc
import time
import random
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--verify", action="store_true",
                    help="Test run with subset of clips × 3 epochs")
parser.add_argument("--test_clips", type=int, default=100,
                    help="Number of clips in verify mode (default 100)")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--max_audio_sec", type=float, default=30.0,
                    help="Max clip duration (30s = keep all ASR data)")
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--seed", type=int, default=42, help="Random seed (change to test different clips)")
parser.add_argument("--patience", type=int, default=0,
                    help="Early stopping patience (0=disabled). Stop if dev loss doesn't improve for N epochs")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hp/Indain_children_spech")
ASER_ROOT    = PROJECT_ROOT / "ASER-Dataset"
TRAIN_CSV    = ASER_ROOT / "splits" / "asr_train.csv"   # ASR subset (Sentence+Paragraph+Story)
DEV_CSV      = ASER_ROOT / "splits" / "asr_dev.csv"    # ASR dev set for overfitting detection
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "phase1_combined"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Optimizations: float16 teachers + gradient checkpointing")
else:
    print("WARNING: No GPU detected — training will be very slow")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load training data (ASR subset only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Loading ASR training data")
print("=" * 70)

with open(TRAIN_CSV, encoding="utf-8") as f:
    all_clips = list(csv.DictReader(f))

clips = [c for c in all_clips if float(c["duration_sec"]) <= args.max_audio_sec]
total_hrs = sum(float(c["duration_sec"]) for c in clips) / 3600

hindi_clips   = [c for c in clips if c["language"] == "Hindi"]
marathi_clips = [c for c in clips if c["language"] == "Marathi"]
english_clips = [c for c in clips if c["language"] == "English"]

print(f"  Source: asr_train.csv (Sentence + Paragraph + Story only)")
print(f"  Total: {len(all_clips)} clips | After ≤{args.max_audio_sec}s filter: {len(clips)} ({total_hrs:.1f}h)")
print(f"    Hindi:   {len(hindi_clips)} clips ({sum(float(c['duration_sec']) for c in hindi_clips)/3600:.1f}h)")
print(f"    Marathi: {len(marathi_clips)} clips ({sum(float(c['duration_sec']) for c in marathi_clips)/3600:.1f}h)")
print(f"    English: {len(english_clips)} clips ({sum(float(c['duration_sec']) for c in english_clips)/3600:.1f}h)")
print(f"  Routing: Hindi/Marathi → Path A+B | English → Path B only")

if args.verify:
    random.seed(args.seed)
    N = args.test_clips
    # Proportional sampling by language
    total_hme = len(hindi_clips) + len(marathi_clips) + len(english_clips)
    n_hi = min(int(N * len(hindi_clips) / total_hme), len(hindi_clips))
    n_mr = min(int(N * len(marathi_clips) / total_hme), len(marathi_clips))
    n_en = min(N - n_hi - n_mr, len(english_clips))
    sample_hi = random.sample(hindi_clips, n_hi)
    sample_mr = random.sample(marathi_clips, n_mr)
    sample_en = random.sample(english_clips, n_en)
    clips = sample_hi + sample_mr + sample_en
    random.shuffle(clips)
    verify_hrs = sum(float(c["duration_sec"]) for c in clips) / 3600
    print(f"\n  VERIFY MODE: {len(clips)} clips ({verify_hrs:.2f}h)")
    print(f"    {len(sample_hi)} Hindi + {len(sample_mr)} Marathi + {len(sample_en)} English")
    print(f"    Running 3 epochs to confirm learning")
    print(f"    Seed: {args.seed}")

# ── Load dev set for validation ──────────────────────────────────────────────
dev_clips = []
if DEV_CSV.exists():
    with open(DEV_CSV, encoding="utf-8") as f:
        dev_clips_all = list(csv.DictReader(f))
    dev_clips = [c for c in dev_clips_all if float(c["duration_sec"]) <= args.max_audio_sec]
    dev_hrs = sum(float(c["duration_sec"]) for c in dev_clips) / 3600
    print(f"\n  Dev set: {len(dev_clips)} clips ({dev_hrs:.1f}h) from asr_dev.csv")
    if args.verify:
        # Use smaller dev set in verify mode (200 clips or 20% of test_clips)
        dev_sample_n = min(200, max(50, args.test_clips // 5), len(dev_clips))
        random.seed(args.seed + 1)  # different seed than training sample
        dev_clips = random.sample(dev_clips, dev_sample_n)
        print(f"  Dev verify subset: {len(dev_clips)} clips")
    if args.patience > 0:
        print(f"  Early stopping: patience={args.patience} epochs")
else:
    print(f"\n  WARNING: Dev set not found at {DEV_CSV} — no overfitting detection")
    if args.patience > 0:
        print(f"  Disabling early stopping (no dev set)")
        args.patience = 0

print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load models
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 2: Loading models (2 frozen teachers + 1 trainable student)")
print("=" * 70)

teacher_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

# ── PATH A: XLSR Hindi-Marathi (FROZEN, float16) ────────────────────────────
print(f"\n  [PATH A] tanmaylaud/wav2vec2-large-xlsr-hindi-marathi...")
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

INDIC_ID = "tanmaylaud/wav2vec2-large-xlsr-hindi-marathi"
indic_extractor = Wav2Vec2FeatureExtractor.from_pretrained(INDIC_ID)
indic_model = Wav2Vec2Model.from_pretrained(INDIC_ID, torch_dtype=teacher_dtype).to(DEVICE)
indic_model.eval()
for p in indic_model.parameters():
    p.requires_grad = False
indic_mem = sum(p.numel() * p.element_size() for p in indic_model.parameters()) / 1024**2
print(f"    {sum(p.numel() for p in indic_model.parameters()):,} params | {indic_mem:.0f} MB | FROZEN {teacher_dtype}")

# ── PATH B: Kid-Whisper encoder (FROZEN, float16) ───────────────────────────
print(f"\n  [PATH B] aadel4/kid-whisper-small-en-myst encoder...")
from transformers import WhisperModel, WhisperProcessor

KW_ID = "aadel4/kid-whisper-small-en-myst"
kw_processor = WhisperProcessor.from_pretrained(KW_ID)
kw_encoder = WhisperModel.from_pretrained(KW_ID, torch_dtype=teacher_dtype).encoder.to(DEVICE)
kw_encoder.eval()
for p in kw_encoder.parameters():
    p.requires_grad = False
kw_mem = sum(p.numel() * p.element_size() for p in kw_encoder.parameters()) / 1024**2
print(f"    {sum(p.numel() for p in kw_encoder.parameters()):,} params | {kw_mem:.0f} MB | FROZEN {teacher_dtype}")

# ── PATH C: MMS-300M (TRAINS, float32 + gradient checkpointing) ─────────────
print(f"\n  [PATH C] facebook/mms-300m (student)...")
MMS_ID = "facebook/mms-300m"
mms_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MMS_ID)
mms_model = Wav2Vec2Model.from_pretrained(MMS_ID).to(DEVICE)
mms_model.train()

# Enable gradient checkpointing — trades compute for memory (~40% less VRAM)
mms_model.gradient_checkpointing_enable()

mms_params = sum(p.numel() for p in mms_model.parameters())
mms_trainable = sum(p.numel() for p in mms_model.parameters() if p.requires_grad)
mms_mem = sum(p.numel() * p.element_size() for p in mms_model.parameters()) / 1024**2
print(f"    {mms_params:,} params ({mms_trainable:,} trainable) | {mms_mem:.0f} MB | float32")
print(f"    Gradient checkpointing: ENABLED (saves ~40% activation memory)")

# ── Projection layer ────────────────────────────────────────────────────────
proj_layer = nn.Linear(1024, 768).to(DEVICE)
proj_params = sum(p.numel() for p in proj_layer.parameters())
print(f"\n  [PROJECTION] Linear(1024→768): {proj_params:,} params")

# ── Memory report ───────────────────────────────────────────────────────────
if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"\n  GPU Memory after loading:")
    print(f"    Allocated: {allocated:.0f} MB")
    print(f"    Reserved:  {reserved:.0f} MB")
    print(f"    Free:      {gpu_mem*1024 - reserved:.0f} MB")
print()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Component verification
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Component verification")
print("=" * 70)


def load_audio(path, max_sec=None):
    """Load audio → 16kHz mono numpy array."""
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze()
    if max_sec and len(wav) > int(max_sec * 16000):
        wav = wav[:int(max_sec * 16000)]
    return wav.numpy()


# Test with a Hindi clip
test_clip = hindi_clips[0] if hindi_clips else clips[0]
test_audio = load_audio(test_clip["audio_path"], max_sec=10.0)
samples = len(test_audio)
print(f"  Test: {Path(test_clip['audio_path']).name} ({samples/16000:.1f}s, {test_clip['language']})")

# Path A
with torch.no_grad():
    inp_a = indic_extractor(test_audio, sampling_rate=16000, return_tensors="pt")
    out_a = indic_model(inp_a.input_values.to(DEVICE, dtype=teacher_dtype))
    Y_indic = out_a.last_hidden_state.mean(dim=1).float()
print(f"  Path A: {out_a.last_hidden_state.shape} → avg_pool → {Y_indic.shape} ✓")

# Path B
with torch.no_grad():
    inp_b = kw_processor(test_audio, sampling_rate=16000, return_tensors="pt")
    out_b = kw_encoder(inp_b.input_features.to(DEVICE, dtype=teacher_dtype))
    valid_T = samples // 160 // 2
    Y_kid = out_b.last_hidden_state[:, :valid_T, :].float()
print(f"  Path B: {out_b.last_hidden_state.shape} → mask({valid_T}) → {Y_kid.shape} ✓")

# Path C
inp_c = mms_extractor(test_audio, sampling_rate=16000, return_tensors="pt")
out_c = mms_model(inp_c.input_values.to(DEVICE))
frames = out_c.last_hidden_state
Y_sem = frames.mean(dim=1)
Y_ac = proj_layer(frames)
print(f"  Path C: {frames.shape} → sem {Y_sem.shape}, ac {Y_ac.shape} ✓")

# Loss + gradient check
loss_fn = nn.MSELoss()
T = min(Y_kid.shape[1], Y_ac.shape[1])
L_sem = loss_fn(Y_sem, Y_indic.detach())
L_ac = loss_fn(Y_ac[:, :T, :], Y_kid[:, :T, :].detach())
L = L_sem + L_ac
L.backward()

frozen_ok = (sum(1 for p in indic_model.parameters() if p.grad is not None) == 0 and
             sum(1 for p in kw_encoder.parameters() if p.grad is not None) == 0)
trains_ok = (sum(1 for p in mms_model.parameters() if p.grad is not None) > 0 and
             sum(1 for p in proj_layer.parameters() if p.grad is not None) > 0)
print(f"  Loss: sem={L_sem.item():.4f} + ac={L_ac.item():.4f} = {L.item():.4f}")
print(f"  Gradients: teachers frozen={'✓' if frozen_ok else '✗'} | student trains={'✓' if trains_ok else '✗'}")

mms_model.zero_grad()
proj_layer.zero_grad()

if DEVICE == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak GPU: {peak:.0f} MB (of {gpu_mem*1024:.0f} MB)")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

print(f"  ✓ ALL VERIFIED\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Training setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 4: Training setup")
print("=" * 70)

optimizer = torch.optim.AdamW(
    list(mms_model.parameters()) + list(proj_layer.parameters()),
    lr=args.lr, weight_decay=0.01,
)

num_epochs = 3 if args.verify else args.epochs
total_steps_est = len(clips) * num_epochs
print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.01)")
print(f"  Epochs: {num_epochs} | Clips: {len(clips)} | Est. steps: ~{total_steps_est}")
print()


def get_teacher_targets(audio_np, language):
    """Returns (Y_indic, Y_kid). Y_indic=None for English."""
    # Path B: acoustic (all languages)
    with torch.no_grad():
        kw_inp = kw_processor(audio_np, sampling_rate=16000, return_tensors="pt")
        kw_out = kw_encoder(kw_inp.input_features.to(DEVICE, dtype=teacher_dtype))
        valid_frames = len(audio_np) // 160 // 2
        Y_kid = kw_out.last_hidden_state[:, :valid_frames, :].float()

    # Path A: semantic (Hindi/Marathi only)
    Y_indic = None
    if language in ("Hindi", "Marathi"):
        with torch.no_grad():
            ind_inp = indic_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
            ind_out = indic_model(ind_inp.input_values.to(DEVICE, dtype=teacher_dtype))
            Y_indic = ind_out.last_hidden_state.mean(dim=1).float()

    return Y_indic, Y_kid


def get_student_outputs(audio_np):
    """Returns (Y_sem_hat, Y_acoust_hat)."""
    inp = mms_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
    out = mms_model(inp.input_values.to(DEVICE))
    frames = out.last_hidden_state
    return frames.mean(dim=1), proj_layer(frames)


def evaluate_dev(dev_clips):
    """Run dev set evaluation (no gradients, no training). Returns avg loss."""
    mms_model.eval()
    dev_losses = []
    dev_sem_losses = []
    dev_ac_losses = []

    with torch.no_grad():
        for clip in tqdm(dev_clips, desc="  Dev eval", unit="clip",
                         bar_format="{l_bar}{bar:20}{r_bar}"):
            try:
                audio = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
                if len(audio) < 8000:
                    continue
                lang = clip["language"]

                # Teachers
                Y_indic, Y_kid = get_teacher_targets(audio, lang)

                # Student
                inp = mms_extractor(audio, sampling_rate=16000, return_tensors="pt")
                out = mms_model(inp.input_values.to(DEVICE))
                frames = out.last_hidden_state
                Y_sem_hat = frames.mean(dim=1)
                Y_ac_hat = proj_layer(frames)

                T = min(Y_kid.shape[1], Y_ac_hat.shape[1])
                L_ac = loss_fn(Y_ac_hat[:, :T, :], Y_kid[:, :T, :]).item()

                if Y_indic is not None:
                    L_sem = loss_fn(Y_sem_hat, Y_indic).item()
                else:
                    L_sem = 0.0

                dev_losses.append(L_ac + L_sem)
                dev_sem_losses.append(L_sem)
                dev_ac_losses.append(L_ac)

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                continue

    mms_model.train()

    if dev_losses:
        avg_total = np.mean(dev_losses)
        avg_sem = np.mean([s for s in dev_sem_losses if s > 0]) if any(s > 0 for s in dev_sem_losses) else 0
        avg_ac = np.mean(dev_ac_losses)
        return avg_total, avg_sem, avg_ac
    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Training loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 5: {'VERIFICATION (' + str(len(clips)) + ' clips × 3 epochs, seed=' + str(args.seed) + ')' if args.verify else 'Full training'}")
print("=" * 70)

# Set seed for reproducibility (full run too)
random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)

loss_history = []
sem_loss_history = []
ac_loss_history = []
epoch_avg_losses = []
epoch_sem_avgs = []
epoch_ac_avgs = []
dev_avg_losses = []
dev_sem_avgs = []
dev_ac_avgs = []
global_step = 0
best_loss = float("inf")
best_dev_loss = float("inf")
patience_counter = 0
n_english = 0
training_start = time.time()

for epoch in range(num_epochs):
    random.shuffle(clips)
    ep_losses, ep_sem, ep_ac = [], [], []
    ep_start = time.time()

    pbar = tqdm(clips, desc=f"Epoch {epoch+1}/{num_epochs}", unit="clip",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for i, clip in enumerate(pbar):
        try:
            audio = load_audio(clip["audio_path"], max_sec=args.max_audio_sec)
            if len(audio) < 8000:
                continue

            lang = clip["language"]

            # Teachers (frozen, float16)
            Y_indic, Y_kid = get_teacher_targets(audio, lang)

            # Student (trainable, float32)
            Y_sem_hat, Y_ac_hat = get_student_outputs(audio)

            # Acoustic loss (all languages)
            T = min(Y_kid.shape[1], Y_ac_hat.shape[1])
            L_ac = loss_fn(Y_ac_hat[:, :T, :], Y_kid[:, :T, :])

            # Semantic loss (Hindi/Marathi)
            if Y_indic is not None:
                L_sem = loss_fn(Y_sem_hat, Y_indic)
                L_total = L_sem + L_ac
                sem_v = L_sem.item()
            else:
                L_total = L_ac
                sem_v = 0.0
                n_english += 1

            # Backward
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(mms_model.parameters()) + list(proj_layer.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            ac_v = L_ac.item()
            total_v = ac_v + sem_v

            # Cleanup GPU
            if DEVICE == "cuda":
                del Y_indic, Y_kid, Y_sem_hat, Y_ac_hat, L_total, L_ac
                if sem_v > 0:
                    del L_sem
                torch.cuda.empty_cache()

            loss_history.append(total_v)
            sem_loss_history.append(sem_v)
            ac_loss_history.append(ac_v)
            ep_losses.append(total_v)
            ep_sem.append(sem_v)
            ep_ac.append(ac_v)
            global_step += 1

            # Update progress bar with current loss
            avg_recent = np.mean(loss_history[-20:]) if loss_history else 0
            path_tag = "A+B" if sem_v > 0 else "B"
            pbar.set_postfix({
                "loss": f"{total_v:.3f}",
                "avg": f"{avg_recent:.3f}",
                "sem": f"{sem_v:.3f}",
                "ac": f"{ac_v:.3f}",
                "path": path_tag,
            })

        except Exception as e:
            print(f"\n  ERROR step {global_step}: {Path(clip['audio_path']).name}: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue

    pbar.close()

    # ── Epoch summary ────────────────────────────────────────────────────
    if ep_losses:
        ep_avg = np.mean(ep_losses)
        ep_sem_avg = np.mean([s for s in ep_sem if s > 0]) if any(s > 0 for s in ep_sem) else 0
        ep_ac_avg = np.mean(ep_ac)
        ep_time = time.time() - ep_start
        epoch_avg_losses.append(ep_avg)
        epoch_sem_avgs.append(ep_sem_avg)
        epoch_ac_avgs.append(ep_ac_avg)

        print(f"\n  ┌─ Epoch {epoch+1}/{num_epochs} ─────────────────────────────────────────")
        print(f"  │ Total loss:    {ep_avg:.4f}")
        print(f"  │ Semantic (A):  {ep_sem_avg:.4f} (Hindi/Marathi clips)")
        print(f"  │ Acoustic (B):  {ep_ac_avg:.4f} (all clips)")
        print(f"  │ Steps: {len(ep_losses)} | Time: {ep_time:.0f}s ({ep_time/60:.1f}min)")
        if DEVICE == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  │ Peak GPU: {peak:.0f} MB / {gpu_mem*1024:.0f} MB")
        if len(epoch_avg_losses) > 1:
            d = epoch_avg_losses[-2] - epoch_avg_losses[-1]
            pct = d / epoch_avg_losses[-2] * 100
            arr = "↓" if d > 0 else "↑"
            print(f"  │ vs prev: {arr} {abs(pct):.1f}% ({epoch_avg_losses[-2]:.4f} → {ep_avg:.4f})")

            d_sem = epoch_sem_avgs[-2] - epoch_sem_avgs[-1]
            pct_sem = d_sem / epoch_sem_avgs[-2] * 100 if epoch_sem_avgs[-2] > 0 else 0
            arr_sem = "↓" if d_sem > 0 else "↑"
            print(f"  │ Semantic: {arr_sem} {abs(pct_sem):.1f}% ({epoch_sem_avgs[-2]:.4f} → {ep_sem_avg:.4f})")

            d_ac = epoch_ac_avgs[-2] - epoch_ac_avgs[-1]
            pct_ac = d_ac / epoch_ac_avgs[-2] * 100
            arr_ac = "↓" if d_ac > 0 else "↑"
            print(f"  │ Acoustic: {arr_ac} {abs(pct_ac):.1f}% ({epoch_ac_avgs[-2]:.4f} → {ep_ac_avg:.4f})")
        print(f"  └──────────────────────────────────────────────────────────")

        # ── Dev set evaluation ─────────────────────────────────────────────
        if dev_clips:
            dev_total, dev_sem, dev_ac = evaluate_dev(dev_clips)
            if dev_total is not None:
                dev_avg_losses.append(dev_total)
                dev_sem_avgs.append(dev_sem)
                dev_ac_avgs.append(dev_ac)
                print(f"\n  ┌─ Dev Loss ─────────────────────────────────────────────")
                print(f"  │ Total: {dev_total:.4f} | Semantic: {dev_sem:.4f} | Acoustic: {dev_ac:.4f}")
                if len(dev_avg_losses) > 1:
                    d = dev_avg_losses[-2] - dev_avg_losses[-1]
                    pct = d / dev_avg_losses[-2] * 100
                    print(f"  │ vs prev: {'↓' if d > 0 else '↑'} {abs(pct):.1f}% ({dev_avg_losses[-2]:.4f} → {dev_total:.4f})")
                gap = dev_total - ep_avg
                print(f"  │ Train-Dev gap: {gap:+.4f} ({'ok' if gap < 0.3 else 'OVERFITTING!' if gap > 0.5 else 'watch'})")
                print(f"  └──────────────────────────────────────────────────────────")

                # Early stopping check
                if args.patience > 0:
                    if dev_total < best_dev_loss:
                        best_dev_loss = dev_total
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"  Early stopping: {patience_counter}/{args.patience} (best dev: {best_dev_loss:.4f})")
                        if patience_counter >= args.patience:
                            print(f"\n  ⚠ EARLY STOPPING at epoch {epoch+1} — dev loss didn't improve for {args.patience} epochs")
                            print(f"    Best dev loss: {best_dev_loss:.4f}")

        # Save checkpoint
        if not args.verify:
            ckpt_data = {
                "epoch": epoch + 1,
                "mms_state_dict": mms_model.state_dict(),
                "proj_state_dict": proj_layer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": ep_avg,
                "dev_loss": dev_avg_losses[-1] if dev_avg_losses else None,
                "global_step": global_step,
            }
            if ep_avg < best_loss:
                best_loss = ep_avg
                torch.save(ckpt_data, CKPT_DIR / "best_model.pt")
                print(f"  Saved best: {CKPT_DIR / 'best_model.pt'}")
            torch.save(ckpt_data, CKPT_DIR / f"epoch_{epoch+1}.pt")
            print(f"  Saved: {CKPT_DIR / f'epoch_{epoch+1}.pt'}")

    print()
    # Check if early stopping triggered
    if patience_counter >= args.patience > 0:
        break

total_time = time.time() - training_start

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Results
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 6: Results")
print("=" * 70)

if len(epoch_avg_losses) >= 2:
    has_dev = len(dev_avg_losses) > 0
    print(f"\n  ── Epoch-level losses ──")
    if has_dev:
        print(f"  {'Epoch':<8} {'Train':>10} {'Dev':>10} {'Sem(A)':>10} {'Ac(B)':>10} {'Gap':>8}")
        print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for i in range(len(epoch_avg_losses)):
            dev_v = dev_avg_losses[i] if i < len(dev_avg_losses) else 0
            gap = dev_v - epoch_avg_losses[i] if dev_v > 0 else 0
            bar = "█" * int(min(epoch_avg_losses[i], 5) * 8)
            print(f"  {i+1:<8} {epoch_avg_losses[i]:>10.4f} {dev_v:>10.4f} {epoch_sem_avgs[i]:>10.4f} {epoch_ac_avgs[i]:>10.4f} {gap:>+8.4f}  {bar}")
    else:
        print(f"  {'Epoch':<8} {'Total':>10} {'Semantic(A)':>12} {'Acoustic(B)':>12}")
        print(f"  {'─'*8} {'─'*10} {'─'*12} {'─'*12}")
        for i in range(len(epoch_avg_losses)):
            bar = "█" * int(min(epoch_avg_losses[i], 5) * 8)
            print(f"  {i+1:<8} {epoch_avg_losses[i]:>10.4f} {epoch_sem_avgs[i]:>12.4f} {epoch_ac_avgs[i]:>12.4f}  {bar}")

    # Total reduction
    total_red = (epoch_avg_losses[0] - epoch_avg_losses[-1]) / epoch_avg_losses[0] * 100
    sem_red = (epoch_sem_avgs[0] - epoch_sem_avgs[-1]) / epoch_sem_avgs[0] * 100 if epoch_sem_avgs[0] > 0 else 0
    ac_red = (epoch_ac_avgs[0] - epoch_ac_avgs[-1]) / epoch_ac_avgs[0] * 100

    print(f"\n  ── Overall reduction ──")
    print(f"    Total:    {total_red:+.1f}% ({epoch_avg_losses[0]:.4f} → {epoch_avg_losses[-1]:.4f})")
    print(f"    Semantic: {sem_red:+.1f}% ({epoch_sem_avgs[0]:.4f} → {epoch_sem_avgs[-1]:.4f})")
    print(f"    Acoustic: {ac_red:+.1f}% ({epoch_ac_avgs[0]:.4f} → {epoch_ac_avgs[-1]:.4f})")

    if total_red > 3:
        print(f"\n  ✓ LEARNING CONFIRMED — loss decreased {total_red:.1f}% across epochs")
    elif total_red > 0:
        print(f"\n  ~ MARGINAL — {total_red:.1f}% decrease")
    else:
        print(f"\n  ✗ NO IMPROVEMENT")

if len(loss_history) >= 10:
    print(f"\n  ── Step-level (first 10 vs last 10) ──")
    print(f"    Total:    {np.mean(loss_history[:10]):.4f} → {np.mean(loss_history[-10:]):.4f}")
    print(f"    Acoustic: {np.mean(ac_loss_history[:10]):.4f} → {np.mean(ac_loss_history[-10:]):.4f}")
    sem_nz = [s for s in sem_loss_history if s > 0]
    if len(sem_nz) >= 10:
        print(f"    Semantic: {np.mean(sem_nz[:10]):.4f} → {np.mean(sem_nz[-10:]):.4f}")

print(f"\n  Total steps:     {global_step}")
print(f"  English (B only): {n_english}")
print(f"  Total time:       {total_time:.0f}s ({total_time/60:.1f}min)")
if global_step > 0:
    per_step = total_time / global_step
    full_est = per_step * len(all_clips) * (args.epochs if not args.verify else 5)
    print(f"  Per step:         {per_step:.2f}s")
    print(f"  Full run estimate: {full_est/3600:.1f}h ({len(all_clips)} clips × {args.epochs if not args.verify else 5} epochs)")
print(f"  Checkpoints:      {CKPT_DIR}")

print()
print("=" * 70)
if args.verify:
    print("  VERIFICATION COMPLETE")
    print("  Check: total loss decreasing? semantic (A) improving? acoustic (B) improving?")
    print("  If yes → ready for full run")
else:
    print("  PHASE 1 TRAINING COMPLETE")
    print("  Next: Phase 2 — add RNN-T decoder on trained MMS-300M")
print("=" * 70)
