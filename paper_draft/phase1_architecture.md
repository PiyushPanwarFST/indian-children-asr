# Phase 1: Multi-Teacher Knowledge Distillation — Full Architecture

## Quick Summary
Train MMS-300M (student) to learn from two frozen teachers:
- Teacher A (XLSR): Hindi/Marathi language understanding
- Teacher B (Kid-Whisper): Children's acoustic patterns
No text labels needed — self-supervised representation learning.

---

## Full Architecture Diagram

```
                         RAW AUDIO (10 seconds example)
              [0.002, 0.008, -0.003, 0.012, ...]
                      <- 160,000 numbers ->
                     (10 seconds x 16,000 Hz)
                              |
              |---------------|---------------|
              |               |               |
              v               v               v
          PATH A          PATH C           PATH B
       (Semantic        (STUDENT -       (Acoustic
        Teacher)        OUR MODEL)        Teacher)
       FROZEN f16      TRAINS f32       FROZEN f16
              |               |               |
              v               v               v
     ┌────────────┐  ┌──────────────┐  ┌─────────────┐
     │IndicWav2Vec│  │  MMS-300M    │  │ Kid-Whisper  │
     │ Feature    │  │  Feature     │  │ Processor    │
     │ Extractor  │  │  Extractor   │  │ (mel spec)   │
     │            │  │              │  │              │
     │ normalize  │  │  normalize   │  │ FFT + mel    │
     │ zero-mean  │  │  zero-mean   │  │ + pad to 30s │
     └─────┬──────┘  └──────┬───────┘  └──────┬───────┘
           |                |                  |
    (1, 160000)      (1, 160000)       (1, 80, 3000)
     normalized       normalized        mel spectrogram
           |                |            80 freq bins
           v                v            3000 time steps
     ┌────────────┐  ┌──────────────┐  ┌──────────────┐
     │ XLSR-53    │  │  MMS-300M    │  │ Kid-Whisper  │
     │ finetuned  │  │  wav2vec2    │  │ Whisper      │
     │ Hindi+     │  │              │  │ encoder      │
     │ Marathi    │  │ CNN: every   │  │              │
     │            │  │ 320 samples  │  │ 2x Conv1d    │
     │ CNN: every │  │ → 1 frame    │  │ downsample   │
     │ 320 samples│  │              │  │              │
     │ → 1 frame  │  │ 24 Trans-   │  │ 12 Trans-    │
     │            │  │ former       │  │ former       │
     │ 24 Trans-  │  │ layers       │  │ layers       │
     │ former     │  │              │  │              │
     │ layers     │  │ 315M params  │  │ 88M params   │
     │            │  │ ALL TRAIN    │  │ ALL FROZEN   │
     │ 315M params│  │              │  │              │
     │ ALL FROZEN │  │ gradient     │  └──────┬───────┘
     └─────┬──────┘  │ checkpointing│         |
           |         └──────┬───────┘  (1, 1500, 768)
    (1, 499, 1024)          |          1500 frames total
     499 frames      (1, 499, 1024)    but only 500 are
     1024-dim each    499 frames       real audio!
           |         1024-dim each           |
           |                |                v
           v                |         ┌──────────────┐
     ┌──────────┐           |         │   MASKING    │
     │ AVG POOL │           |         │              │
     │          │           |         │ Keep frames  │
     │ average  │           |         │ 1 to 500     │
     │ all 499  │     ┌─────┴─────┐   │ Throw away   │
     │ frames   │     |           |   │ frames       │
     │ into 1   │     v           v   │ 501 to 1500  │
     └────┬─────┘  SEMANTIC   ACOUSTIC│ (padding     │
          |        BRANCH     BRANCH  │  garbage)    │
          v           |           |   └──────┬───────┘
       Y_indic        v           v          |
      (1, 1024)   ┌────────┐ ┌────────┐  (1, 500, 768)
     "correct     │AVG POOL│ │LINEAR  │   500 real frames
      semantic    │        │ │1024→768│   768-dim each
      answer"     │average │ │        │      |
                  │499→1   │ │per-    │      v
                  └───┬────┘ │frame   │   Y_kid
                      |      │project │  (1, 500, 768)
                      v      │ion     │  "correct
                  Y_sem_hat  └───┬────┘   acoustic
                  (1, 1024)      |         answer"
                  "student's     v
                   semantic  Y_ac_hat
                   guess"   (1, 499, 768)
                             "student's
                              acoustic
                              guess"
                      |           |
       ┌──────────────┘           └────────────────┐
       |                                           |
       v                                           v
  ┌──────────────────┐               ┌────────────────────┐
  │   SEMANTIC LOSS  │               │   ACOUSTIC LOSS    │
  │                  │               │                    │
  │ MSE(Y_sem_hat,   │               │ T = min(500, 499)  │
  │     Y_indic)     │               │ = 499              │
  │                  │               │                    │
  │ Compare 1024     │               │ MSE(Y_ac_hat[:T],  │
  │ values:          │               │     Y_kid[:T])     │
  │ (guess-answer)²  │               │                    │
  │ averaged         │               │ Compare 499 x 768  │
  │                  │               │ = 383,232 values:  │
  │ Result: ~0.05    │               │ (guess-answer)²    │
  │                  │               │ averaged           │
  │ Hindi/Marathi    │               │                    │
  │ clips ONLY       │               │ Result: ~1.3-1.9   │
  │ (English skips   │               │                    │
  │  this loss)      │               │ ALL clips          │
  └────────┬─────────┘               └─────────┬──────────┘
           |                                   |
           └──────────┐    ┌───────────────────┘
                      v    v
               ┌──────────────────┐
               │   TOTAL LOSS     │
               │                  │
               │ Hindi/Marathi:   │
               │ L = L_sem + L_ac │
               │ = ~0.05 + ~1.5   │
               │ = ~1.55          │
               │                  │
               │ English:         │
               │ L = L_ac only    │
               │ = ~1.5           │
               └────────┬─────────┘
                        |
                        v
               ┌──────────────────┐
               │   BACKWARD()     │
               │                  │
               │ Compute gradient │
               │ dL/dw for every  │
               │ trainable weight │
               │                  │
               │ Gradients flow   │
               │ ONLY to:        │
               │ - MMS-300M      │
               │ - Linear(1024   │
               │   →768)         │
               │                  │
               │ Teachers get     │
               │ ZERO gradients   │
               │ (frozen)        │
               └────────┬─────────┘
                        |
                        v
               ┌──────────────────────────────────┐
               │   OPTIMIZER.STEP()                │
               │                                   │
               │   AdamW optimizer                 │
               │   lr = 3e-5 (0.00003)             │
               │   weight_decay = 0.01             │
               │                                   │
               │   For each trainable weight w:    │
               │   w_new = w - lr × gradient       │
               │   (simplified — AdamW also uses   │
               │    momentum and adaptive lr)      │
               │                                   │
               │   BEFORE this step:               │
               │   clip_grad_norm(max=1.0)         │
               │   If gradient too large, scale    │
               │   it down to prevent explosion    │
               │                                   │
               │   This is WHERE LEARNING HAPPENS  │
               │   Student weights get updated     │
               │   Teachers stay exactly the same  │
               └──────────────────────────────────┘
```

---

## Where Each Parameter Is Used In The Workflow

### Training Parameters — Where They Appear

```
STEP 1: Load audio clip
  └─ max_audio_sec = 30.0 seconds (keep all ASR data, skip nothing)

STEP 2: Forward pass through teachers (FROZEN)
  └─ float16 precision (saves GPU memory, teachers don't need gradients)
  └─ torch.no_grad() context (tells PyTorch: don't track operations)

STEP 3: Forward pass through student (TRAINS)
  └─ float32 precision (needs accurate gradients)
  └─ gradient_checkpointing (recompute activations to save ~40% memory)

STEP 4: Compute loss
  └─ MSELoss() — Mean Squared Error: average of (prediction - target)²
  └─ Frame alignment: T = min(teacher_frames, student_frames)

STEP 5: backward() — compute gradients
  └─ clip_grad_norm(max_norm=1.0) — cap gradient magnitude
      Prevents one extreme clip from destabilizing training

STEP 6: optimizer.step() — update weights
  └─ AdamW optimizer
  └─ learning_rate = 3e-5 — how big each weight update is
  └─ weight_decay = 0.01 — L2 regularization (keep weights small)

STEP 7: GPU cleanup
  └─ del tensors + torch.cuda.empty_cache() — free VRAM

After each EPOCH:
  └─ Dev evaluation (same forward pass, no backward, no weight update)
  └─ Early stopping (patience=5) — stop if dev loss stagnates
  └─ Save checkpoint (.pt file with model weights + optimizer state)
```

### The Three Models — Technical Specs

```
┌─────────────────────────────────────────────────────────────────────┐
│ MODEL          │ HuggingFace ID                    │ Role          │
├─────────────────────────────────────────────────────────────────────┤
│ XLSR Hindi-    │ tanmaylaud/wav2vec2-large-xlsr-   │ Semantic      │
│ Marathi        │ hindi-marathi                     │ Teacher       │
│                │                                   │               │
│ Architecture:  wav2vec2 (CNN + 24 Transformer layers)              │
│ Base model:    facebook/wav2vec2-large-xlsr-53 (53 languages)      │
│ Fine-tuned on: Hindi + Marathi adult speech (ASR task)             │
│ Params:        315,438,720 (315M) — ALL FROZEN                    │
│ Output dim:    1024 per frame                                      │
│ Precision:     float16 (saves ~600MB GPU memory)                   │
│ Input:         raw audio waveform (normalized)                     │
│ Output:        (1, T, 1024) → avg_pool → (1, 1024)               │
│                                                                    │
│ Why this model: Knows Hindi/Marathi phonemes and semantics.        │
│ Trained on Indian adult speech → understands Indian languages.     │
│ Student learns "what this audio means in Hindi/Marathi context"    │
├─────────────────────────────────────────────────────────────────────┤
│ Kid-Whisper    │ aadel4/kid-whisper-small-en-myst  │ Acoustic      │
│ Encoder        │ (encoder only)                    │ Teacher       │
│                │                                   │               │
│ Architecture:  Whisper encoder (2 Conv1d + 12 Transformer layers)  │
│ Base model:    openai/whisper-small.en (English-only Whisper)      │
│ Fine-tuned on: MyST dataset (125h American English children 8-11)  │
│ Params:        88,154,112 (88M) — ALL FROZEN                      │
│ Output dim:    768 per frame                                       │
│ Precision:     float16                                             │
│ Input:         mel spectrogram (1, 80, 3000) — NOT raw audio!     │
│ Output:        (1, 1500, 768) → mask to valid → (1, T_valid, 768) │
│                                                                    │
│ Why this model: Knows what children's speech "sounds like."        │
│ Trained on real children → understands child voice characteristics │
│ (higher pitch, reading hesitations, pronunciation patterns).       │
│ Student learns "acoustic patterns unique to children's speech"     │
├─────────────────────────────────────────────────────────────────────┤
│ MMS-300M       │ facebook/mms-300m                 │ Student       │
│                │                                   │ (TRAINS)      │
│                │                                   │               │
│ Architecture:  wav2vec2 (CNN + 24 Transformer layers)              │
│ Pretrained on: 1,100+ languages, ~500K hours unlabeled audio      │
│ Params:        315,438,720 (315M) — ALL TRAINABLE                 │
│ + Projection:  Linear(1024→768) = 787,200 params — TRAINABLE      │
│ Output dim:    1024 per frame                                      │
│ Precision:     float32 (needs gradient accuracy)                   │
│ Memory saving: gradient checkpointing (saves ~40% activation mem) │
│ Input:         raw audio waveform (normalized)                     │
│ Output:        (1, T, 1024)                                        │
│                ├→ avg_pool → (1, 1024) [semantic branch]           │
│                └→ Linear(1024→768) → (1, T, 768) [acoustic branch]│
│                                                                    │
│ Why this model as STUDENT (not teacher):                           │
│ 1. Broadest foundation — covers 1100+ languages including our 3   │
│ 2. Same architecture as Teacher A → natural representation match  │
│ 3. Large capacity (315M) → can absorb knowledge from both teachers│
│ 4. Already understands speech broadly, just needs to learn         │
│    children's patterns + Indian language specifics                 │
│                                                                    │
│ Why not as TEACHER:                                                │
│ - It's a generalist, not a specialist                              │
│ - Doesn't know children's speech (never saw children during train) │
│ - Doesn't have deep Hindi/Marathi knowledge (1100 lang = shallow) │
│ Teachers are SPECIALISTS, student is the GENERALIST who learns.   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

```
TRAINING:   asr_train.csv — 13,765 clips (46.4 hours)
            Sentence + Paragraph + Story reading levels ONLY
            Hindi: 6,304 clips (28.6h)
            Marathi: 2,865 clips (13.5h)
            English: 4,596 clips (4.3h)

DEV SET:    asr_dev.csv — 1,775 clips (6.0 hours)
            Same reading levels (Sentence + Paragraph + Story)
            Used for overfitting detection ONLY — never trained on
            Speaker-independent: different children than training set

TEST SET:   asr_test.csv — 1,695 clips (5.8 hours)
            Reserved for final evaluation after Phase 2
            Never touched during Phase 1

WHY ASR SUBSET (not full 48,757 clips)?
  Full set includes Capital Letter, Small Letter, Word, Letter levels.
  A child saying "B" = 0.3 seconds = ~5 audio frames.
  MSE between 5 frames = noise, not learning signal.
  Sentence/Paragraph/Story = 2-30 seconds = 60-900 frames = meaningful.
```

---

## Results

### Baselines (Zero-Shot — No Training)

```
Corpus-level WER (lower = better):
┌──────────────────┬────────┬─────────┬─────────┬─────────┐
│ Model            │ Hindi  │ Marathi │ English │ Overall │
├──────────────────┼────────┼─────────┼─────────┼─────────┤
│ Whisper Small    │  98.0% │ 124.7%  │  53.6%  │ 101.2%  │
│ Kid-Whisper-EN   │ 185.5% │ 272.0%  │  57.2%  │ 197.2%  │
└──────────────────┴────────┴─────────┴─────────┴─────────┘
Both models fail on Indian children's speech → research gap confirmed.
```

### Phase 1 Knowledge Distillation (Verification)

```
1000 clips x 3 epochs, tested with 2 different random seeds:

                 SEED 42          SEED 123 (+ dev eval)
Epoch 1:         1.9086           1.9042 (dev: 1.7235)
Epoch 2:         1.5979           1.5903 (dev: 1.4521)
Epoch 3:         1.3943           1.3838 (dev: 1.2724)
────────────────────────────────────────────────────────
Reduction:       -26.9%           -27.3%
Semantic (A):    -19.9%           -21.7%
Acoustic (B):    -27.1%           -27.5%
Train-Dev gap:     N/A            -0.11 to -0.18 (no overfitting)

LEARNING CONFIRMED — consistent across both seeds.
Dev loss lower than train loss → model is generalizing well.
```

---

## GPU Memory Budget (RTX 4060, 8GB)

```
Component                    Memory
─────────────────────────── ──────
XLSR teacher (float16)       602 MB
Kid-Whisper teacher (f16)    168 MB
MMS student (float32)      1,203 MB
Projection layer               3 MB
Optimizer states (AdamW)   2,400 MB
Activations (checkpointed) 2,500 MB
─────────────────────────── ──────
Peak total:                6,900 MB / 7,932 MB (87% utilization)
```
