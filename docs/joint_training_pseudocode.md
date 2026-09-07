# Joint Training — Full Technical Pseudocode

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Audio (.wav)                                                │
│       ↓                                                      │
│  Whisper Processor → Mel Spectrogram (1, 80, 3000)           │
│       ↓                            ↓                         │
│  Student Encoder (TRAINABLE)    Teacher Encoder (FROZEN)     │
│  (Whisper Small, 12 layers)     (Kid-Whisper Med, 24 layers) │
│       ↓                            ↓                         │
│  (1, 1500, 768)                (1, 1500, 1024)               │
│       ↓                                                      │
│  ┌────┴─────────────────────┐                                │
│  │                          │                                │
│  ↓                          ↓                                │
│  Projection Layer        CTC Head (TRAINABLE)                │
│  Linear(768→1024)        Linear(768→85)                      │
│  (TRAINABLE)                ↓                                │
│       ↓                 Logits (1500, 85)                    │
│  (1, 1500, 1024)            ↓                                │
│       ↓                 CTC Loss ← Ground Truth Text         │
│  MSE Loss ← Teacher features                                │
│                                                              │
│  Total Loss = α × MSE + (1-α) × CTC                         │
│       ↓                                                      │
│  Backprop → updates Encoder + Projection + CTC Head          │
└──────────────────────────────────────────────────────────────┘
```

## What Gets Updated

```
TRAINABLE (updated every step):
    ✅ Student encoder     — 88,154,112 params (12 transformer layers)
    ✅ Projection layer    — 787,456 params (Linear 768→1024 + bias)
    ✅ CTC head            — 65,365 params (Linear 768→85 + bias)
    Total trainable: ~89M params

FROZEN (never updated):
    ❌ Teacher encoder     — ~300M params (reference only)
```

## Step-by-Step Pseudocode

### STEP 0: SETUP

```
Load vocab.json (85 tokens, built by build_vocab.py)
Load train/dev clips (audio_path + transcript + language)

Load Teacher encoder (Kid-Whisper Medium):
    teacher = WhisperForConditionalGeneration("kid-whisper-medium")
    teacher_encoder = teacher.model.encoder
    FREEZE teacher_encoder (requires_grad = False)
    teacher_dim = 1024
    Delete teacher decoder (save ~500MB memory)

Load Student encoder (Whisper Small):
    Warm start from our HPC acoustic checkpoint (hpc_best_dev_model.pt)
    UNFREEZE student encoder (requires_grad = True)
    ← KEY DIFFERENCE from sequential: encoder is now trainable

Create Projection layer:
    projection = Linear(768, 1024)
    Load weights from hpc_best_dev_model.pt if available

Create CTC head:
    ctc_head = Linear(768, 85)
    Load weights from hpc_ctc_best_wer.pt (warm start from sequential training)

Optimizer:
    AdamW(params=[encoder, projection, ctc_head], lr=3e-5)
    LR schedule: linear warmup (500 steps) → linear decay

Loss functions:
    mse_loss = torch.nn.functional.mse_loss (PyTorch built-in)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True) (PyTorch built-in)

Alpha:
    α = 0.5 (equal weight to both losses)
```

### STEP 1: TRAINING LOOP (for each epoch)

```
Shuffle training clips
student_encoder.train()
teacher_encoder.eval()

FOR EACH CLIP:

    # 1a: Prepare Input
    audio → load wav → resample 16kHz → mono
    mel = WhisperProcessor(audio)           # (1, 80, 3000)
    real_frames = compute_real_frames(mel)

    text = normalize_text(clip["transcript"])
    target_indices = text_to_indices(text, vocab)

    if real_frames < len(target_indices): skip

    # 1b: Student Forward (WITH gradients)
    student_features = student_encoder(mel)  # (1, 1500, 768)

    # 1c: Teacher Forward (NO gradients)
    with torch.no_grad():
        teacher_features = teacher_encoder(mel)  # (1, 1500, 1024)

    # 1d: MSE Loss (acoustic branch)
    projected = projection(dropout(student_features))  # (1, 1500, 1024)
    mse = MSE(projected[:, :real_frames], teacher_features[:, :real_frames])

    # 1e: CTC Loss (semantic branch)
    logits = ctc_head(student_features)  # (1, 1500, 85)
    log_probs = logits.log_softmax()[:, :real_frames].permute(1,0,2)
    ctc = CTC_loss(log_probs, target_indices, [real_frames], [len(target)])

    # 1f: Combine Losses
    total_loss = α * mse + (1-α) * ctc

    # 1g: Backward + Update
    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(all_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

### STEP 2: DEV EVALUATION

```
After each epoch:
    Run all dev clips through encoder + CTC head
    Greedy decode → predicted text
    Compute: dev_mse, dev_ctc, dev_wer (corpus-level)
    Print sample predictions
    Early stopping on dev WER (patience=5)
```

### STEP 3: CHECKPOINTS

```
Save: encoder + projection + ctc_head + optimizer + metadata
Best models: joint_best_wer.pt, joint_best_loss.pt
Resume support: --resume flag for walltime interruptions
```

## Key Decisions & References

| Decision | Value | Reference |
|---|---|---|
| Loss combination | α × MSE + (1-α) × CTC | CARE (Rumberg 2022), DistilBERT |
| α value | 0.5 | CARE paper, FitNets (Romero 2015) |
| Learning rate | 3e-5 | Wav2Vec 2.0, Whisper fine-tuning |
| Warmup steps | 500 | Standard transformer practice |
| Optimizer | AdamW | Loshchilov & Hutter 2019 |
| CTC decoding | Greedy | Wav2Vec 2.0 default |
| Grad clipping | max_norm=1.0 | Standard practice |

## Comparison: Sequential vs Joint

```
SEQUENTIAL (what we did):
    Step 1: encoder trained with MSE only → features optimized for teacher matching
    Step 2: CTC trained on frozen encoder → limited by fixed features
    Result: 81.9% WER overall

JOINT (what we're doing):
    Single step: encoder trained with MSE + CTC simultaneously
    Encoder learns features that are BOTH teacher-like AND text-decodable
    Expected: better WER because no encoder-decoder mismatch
```

## Test Plan

```
Phase 1: Test run (HPC, 2000 clips, 5 epochs, 6hr walltime)
    Verify: dev WER improving, losses decreasing, no errors
    Compare: must show improvement over sequential (81.9%)

Phase 2: Full run (HPC, all clips, 30 epochs, 72hr walltime)
    patience=5, --resume if needed
```
