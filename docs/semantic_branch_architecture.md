# Semantic Branch — Full Architecture & Pseudocode
# Indian Children ASR — Knowledge Distillation from IndicConformer
# Date: 2026-09-08

---

## Goal

Teach Whisper Small's encoder to understand Hindi/Marathi speech
by learning from IndicConformer's output predictions (logits).

This is MSE loss at LOGITS level — the student learns to produce
the same per-frame predictions as the teacher.

---

## Models Involved

```
TEACHER: IndicConformer 600M (ai4bharat/indic-conformer-600m-multilingual)
  - Architecture: Conformer encoder + CTC decoder (ONNX)
  - Encoder hidden dim: 1024
  - CTC output: 5633 total classes (all 22 Indian languages)
  - Per-language: 257 classes (256 BPE subwords + 1 blank)
  - Hindi tokens at positions 1536-1791 + blank at 5632
  - Marathi tokens at positions 2816-3071 + blank at 5632
  - Frame rate: 12.5 frames/sec (79.9 ms/frame)
  - Output frames: VARIABLE (depends on audio length, NO padding)
  - State: FROZEN (never trains, pre-computed offline)
  - No English support

STUDENT: Whisper Small (openai/whisper-small)
  - Architecture: Transformer encoder
  - Encoder hidden dim: 768
  - Frame rate: 50.0 frames/sec (20.0 ms/frame)
  - Output frames: ALWAYS 1500 (Whisper pads all audio to 30 seconds)
  - Real frames: depends on audio length (e.g., 683 for 13.66s clip)
  - State: UNFROZEN (learns from teacher)
  - Starting point: Fresh pretrained weights (no acoustic branch checkpoint)

NEW LAYER: CTC Head 2 — Linear(768 → 257)
  - Purpose: Convert student's 768-dim features to teacher's 257-dim logit space
  - Same role as Projection(768→1024) in acoustic branch — a bridge layer
  - Separate head for Hindi and Marathi (different BPE vocabularies)
  - Params: ~197K per head (tiny vs 88M encoder)
  - State: UNFROZEN (trained from scratch)

DATA: ASER training set
  - Hindi: 6,304 clips
  - Marathi: 2,865 clips
  - English: 4,596 clips → SKIPPED (no IndicConformer support)
  - Total for semantic branch: 9,169 clips
```

---

## Why Hindi and Marathi Need Separate CTC Heads

```
IndicConformer's CTC decoder outputs 5633 scores per frame.
These 5633 positions are PARTITIONED by language.

When decoding Hindi, a boolean mask selects 257 positions:
  Position 0 in masked output = full vocab index 1536 = "▁क"
  Position 1 in masked output = full vocab index 1537 = "▁स"
  ...
  Position 256 = blank

When decoding Marathi, a DIFFERENT boolean mask selects 257 positions:
  Position 0 in masked output = full vocab index 2816 = "या"
  Position 1 in masked output = full vocab index 2817 = "्या"
  ...
  Position 256 = blank

SAME index (0) means "▁क" in Hindi but "या" in Marathi!

If one CTC head tries to learn both:
  Hindi clip pushes output[0] towards "▁क" score
  Marathi clip pushes output[0] towards "या" score
  → Contradictory gradients → head cannot learn properly

Solution: Two separate Linear(768→257) heads.
  ctc_head_hi: trained only on Hindi clips
  ctc_head_mr: trained only on Marathi clips
  Both share the same encoder — only the last layer differs.

Extra cost: 197K × 2 = 394K params (0.4% of 88M encoder). Negligible.
```

---

## Full Architecture Diagram

```
            Audio Input: child saying "नीतू के घर में गाय है"
            (218,560 samples = 13.66 seconds at 16kHz)
                              │
         ┌────────────────────┴────────────────────────┐
         │                                             │
         │  TEACHER (pre-computed, loaded from disk)   │  STUDENT (runs during training)
         │                                             │
         ▼                                             ▼
┌─────────────────────┐                    ┌───────────────────────┐
│   Preprocessor.ts   │                    │ WhisperFeatureExtract │
│   (TorchScript)     │                    │   (mel spectrogram)   │
└────────┬────────────┘                    └───────────┬───────────┘
         │                                             │
         ▼                                             ▼
   (1, 80, 1367)                                 (1, 80, 3000)
   80 freq bins                                  80 freq bins
   1367 mel frames                               3000 mel frames
   (variable, no pad)                            (ALWAYS padded to 30s)
         │                                             │
         ▼                                             ▼
┌─────────────────────┐                    ┌───────────────────────┐
│  Encoder (ONNX)     │                    │  Whisper Encoder      │
│  17 Conformer blocks│                    │  12 Transformer layers│
│  hidden_dim = 1024  │                    │  hidden_dim = 768     │
│  FROZEN             │                    │  UNFROZEN (trains)    │
└────────┬────────────┘                    └───────────┬───────────┘
         │                                             │
         ▼                                             ▼
   (1, 1024, 171)                                (1, 1500, 768)
   171 frames                                    1500 frames always
   12.5 fps                                      50 fps
         │                                             │
         ▼                                             │
┌─────────────────────┐                                │
│  CTC Decoder (ONNX) │                                │
│  Linear(1024→5633)  │                                │
│  FROZEN             │                                │
└────────┬────────────┘                                │
         │                                             │
         ▼                                             │
   (1, 171, 5633)                                      │
   5633 = all 22 langs                                 │
         │                                             │
         ▼                                             ▼
┌─────────────────────┐                    ┌───────────────────────┐
│  Language Mask      │                    │  Slice Real Frames    │
│  Hindi: select 257  │                    │  Keep first 683       │
│  out of 5633        │                    │  Discard 817 padding  │
│  (boolean indexing) │                    │  frames               │
└────────┬────────────┘                    └───────────┬───────────┘
         │                                             │
         ▼                                             ▼
   (171, 257)                                    (1, 683, 768)
   171 teacher frames                            683 student frames
   257 Hindi BPE logits                          768 hidden features
         │                                             │
         │                                             ▼
         │                                 ┌───────────────────────┐
         │                                 │  CTC Head 2 (Hindi)  │
         │                                 │  Linear(768 → 257)   │
         │                                 │  UNFROZEN (trains)    │
         │                                 └───────────┬───────────┘
         │                                             │
         │                                             ▼
         │                                       (1, 683, 257)
         │                                       683 student frames
         │                                       257 Hindi BPE logits
         │                                             │
         ▼                                             │
┌──────────────────────────────────────────────────────┤
│  TIME ALIGNMENT (Interpolation)                      │
│                                                      │
│  Teacher: 171 frames (12.5 fps)                      │
│  Student: 683 frames (50.0 fps)                      │
│  Ratio: 3.99x                                        │
│                                                      │
│  F.interpolate(teacher, size=683, mode='linear')     │
│                                                      │
│  Teacher (171, 257) → reshape → (1, 257, 171)       │
│  → interpolate → (1, 257, 683)                       │
│  → reshape back → (1, 683, 257)                      │
│                                                      │
│  Each teacher frame is "stretched" across ~4          │
│  student frames using linear blending.               │
└────────┬─────────────────────────────────────────────┘
         │
         ▼
   (1, 683, 257)                             (1, 683, 257)
   teacher aligned                           student logits
         │                                       │
         └──────────────┬────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │    MSE LOSS      │
              │                  │
              │  mean((student   │
              │   - teacher)^2)  │
              │                  │
              │  across all      │
              │  683 × 257       │
              │  = 175,531 vals  │
              │                  │
              │  = single scalar │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  loss.backward() │
              │                  │
              │  Gradients flow  │
              │  to:             │
              │   - CTC Head 2   │
              │   - Whisper enc  │
              │                  │
              │  NOT to:         │
              │   - Teacher      │
              │   (pre-computed) │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  optimizer.step()│
              │  Update weights  │
              └──────────────────┘
```

---

## Verified Numbers (from actual model runs on HI_S1_P_0.wav, 13.66s)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step                          │ Shape              │ Notes          │
├───────────────────────────────┼────────────────────┼────────────────┤
│ TEACHER                       │                    │                │
│ [T1] Preprocessor input       │ (1, 218560)        │ raw waveform   │
│ [T1] Preprocessor output      │ (1, 80, 1367)      │ mel spec       │
│ [T2] Encoder output           │ (1, 1024, 171)     │ 171 frames     │
│ [T3] CTC decoder output       │ (1, 171, 5633)     │ all langs      │
│ [T4] After language mask       │ (171, 257)         │ Hindi only     │
│                               │                    │                │
│ STUDENT                       │                    │                │
│ [S1] Mel spectrogram          │ (1, 80, 3000)      │ padded to 30s  │
│ [S2] Encoder output           │ (1, 1500, 768)     │ always 1500    │
│ [S3] Real frames              │ 683                │ of 1500 total  │
│ [S4] After slicing            │ (1, 683, 768)      │ real only      │
│ [S5] CTC Head 2 output        │ (1, 683, 257)      │ Hindi logits   │
│                               │                    │                │
│ ALIGNMENT                     │                    │                │
│ [A1] Teacher before interp    │ (171, 257)         │ 12.5 fps       │
│ [A2] Teacher after interp     │ (1, 683, 257)      │ stretched 4x   │
│ [A2] Student logits           │ (1, 683, 257)      │ 50 fps         │
│ [A3] Shapes match?            │ YES ✓              │                │
│ [A4] MSE (random init)        │ 44.96              │ will decrease  │
│                               │                    │                │
│ VERIFICATION                  │                    │                │
│ Decode from saved logits       │ ✓ 100% match      │ 5/5 clips OK   │
│ Interpolation first frame      │ ✓ exact match     │ no distortion  │
│ Interpolation last frame       │ ✓ exact match     │ no distortion  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pseudocode

### SCRIPT 1: step0_precompute_teacher_logits.py (DONE, VERIFIED)

```
PURPOSE: Run IndicConformer on all Hindi/Marathi clips ONCE.
         Save raw logits to disk. Never run teacher during training.

INPUT:   ASER-Dataset/splits/asr_train.csv (9,169 Hindi+Marathi clips)
OUTPUT:  teacher_logits/train/{clip_name}.pt  (one file per clip)

ALREADY WRITTEN AND TESTED. Verification: 5/5 clips decode correctly.

──────────────────────────────────────────────────────────────

LOAD IndicConformer model

FOR each Hindi/Marathi clip in training set:

    # Load audio
    wav = torchaudio.load(audio_path)       # (1, num_samples)
    resample to 16kHz if needed
    trim to 30 seconds max

    # Run teacher pipeline
    audio_signal = preprocessor(wav)         # (1, 80, T_mel)
    encoder_out = encoder(audio_signal)      # (1, 1024, T_enc)
    raw_logits = ctc_decoder(encoder_out)    # (1, T_enc, 5633)

    # Select this language's tokens
    IF language == Hindi:
        mask = language_masks["hi"]          # boolean, 5633 long, 257 True
    ELIF language == Marathi:
        mask = language_masks["mr"]          # different 257 positions

    logits = raw_logits[0, :T_enc, mask]     # (T_enc, 257) — raw, no log_softmax

    # Verify: decode from logits and compare with normal output
    decoded = greedy_ctc_decode(logits, vocab[lang])
    normal = model.forward(wav, lang)
    ASSERT decoded == normal                 # must match!

    # Save
    torch.save({
        'logits': logits.float16(),          # (T_enc, 257)
        'num_frames': T_enc,                 # int
        'language': lang_code                # "hi" or "mr"
    }, f"teacher_logits/train/{clip_name}.pt")
```

### SCRIPT 2: step1_semantic_mse_training.py (TO WRITE)

```
PURPOSE: Train Whisper Small encoder + CTC Head 2 to match
         IndicConformer's logits using MSE loss.

INPUT:   teacher_logits/train/*.pt   (pre-computed)
         ASER-Dataset/splits/asr_train.csv
         ASER-Dataset/splits/asr_dev.csv
OUTPUT:  checkpoints/semantic/semantic_best_dev.pt

──────────────────────────────────────────────────────────────

# ═══ SETUP ═══

LOAD Whisper Small encoder (pretrained, UNFROZEN)
    encoder = WhisperModel("openai/whisper-small").encoder
    encoder.train()
    # All 88M params receive gradients

CREATE CTC Head 2 for Hindi
    ctc_head_hi = Linear(768, 257)           # 197K params, random init

CREATE CTC Head 2 for Marathi
    ctc_head_mr = Linear(768, 257)           # 197K params, random init

LOAD WhisperFeatureExtractor
    feat_ext = WhisperFeatureExtractor("openai/whisper-small")

LOAD clips from asr_train.csv (Hindi + Marathi only, skip English)
    → 9,169 clips

optimizer = AdamW(
    params = [encoder.parameters(), ctc_head_hi.parameters(), ctc_head_mr.parameters()],
    lr = 3e-5
)
scheduler = linear warmup (500 steps) then linear decay
max_grad_norm = 1.0
patience = 5
max_epochs = 20

# ═══ TRAINING LOOP ═══

best_dev_loss = infinity
patience_counter = 0

FOR epoch = 1 to max_epochs:

    SHUFFLE training clips
    epoch_losses = []

    FOR each clip in training clips:

        # ─── Load audio ───
        wav = torchaudio.load(clip.audio_path)     # (1, samples)
        wav_1d = wav.squeeze()                      # (samples,)
        samples = wav_1d.shape[0]

        # ─── Student forward pass ───
        mel = feat_ext(wav_1d, return_tensors="pt")  # (1, 80, 3000) padded
        features = encoder(mel).last_hidden_state     # (1, 1500, 768)

        # Calculate real frames (discard padding)
        real_frames = samples // 160 // 2             # e.g., 683 for 13.66s
        real_features = features[:, :real_frames, :]  # (1, 683, 768)

        # Forward through language-specific CTC head
        IF clip.language == "hi":
            student_logits = ctc_head_hi(real_features)  # (1, 683, 257)
        ELIF clip.language == "mr":
            student_logits = ctc_head_mr(real_features)  # (1, 683, 257)

        # ─── Load teacher logits (pre-computed) ───
        teacher_data = torch.load(f"teacher_logits/train/{clip_name}.pt")
        teacher_logits = teacher_data['logits'].float()  # (T_teacher, 257)
        teacher_frames = teacher_data['num_frames']       # e.g., 171

        # ─── Time alignment: interpolate teacher → student frame count ───
        # Teacher has fewer frames (12.5 fps) than student (50 fps)
        # Stretch teacher logits to match student's frame count

        teacher_interp = teacher_logits.T.unsqueeze(0)     # (1, 257, 171)
        teacher_aligned = F.interpolate(
            teacher_interp,
            size=real_frames,                               # 683
            mode='linear',
            align_corners=False
        )                                                   # (1, 257, 683)
        teacher_aligned = teacher_aligned.squeeze(0).T      # (683, 257)
        teacher_aligned = teacher_aligned.unsqueeze(0)      # (1, 683, 257)

        # ─── MSE Loss ───
        loss = F.mse_loss(student_logits, teacher_aligned)

        # ─── Backward + update ───
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.item())

    # ─── Dev evaluation ───
    dev_loss = evaluate_dev(encoder, ctc_head_hi, ctc_head_mr, dev_clips)

    # ─── Early stopping ───
    IF dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        patience_counter = 0
        SAVE checkpoint (encoder + ctc_head_hi + ctc_head_mr)
    ELSE:
        patience_counter += 1
        IF patience_counter >= patience:
            PRINT "Early stopping at epoch {epoch}"
            STOP

    PRINT epoch, avg_train_loss, dev_loss, patience_counter
```

### SCRIPT 3: step2_semantic_evaluate.py (TO WRITE)

```
PURPOSE: Check if the trained encoder actually learned to transcribe.
         Decode from CTC Head 2 and compute WER.

INPUT:   checkpoints/semantic/semantic_best_dev.pt
         ASER-Dataset/splits/asr_test.csv
OUTPUT:  WER per language (Hindi, Marathi)

──────────────────────────────────────────────────────────────

LOAD best checkpoint (encoder + ctc_head_hi + ctc_head_mr)
LOAD IndicConformer's vocab (for decoding BPE tokens)
    vocab_hi = model.vocab["hi"]    # 257 tokens: ["<unk>", "▁क", "▁स", ...]
    vocab_mr = model.vocab["mr"]    # 257 tokens: ["<unk>", "या", "्या", ...]

FOR each Hindi/Marathi test clip:

    # Student forward pass
    mel = feat_ext(wav, return_tensors="pt")
    features = encoder(mel).last_hidden_state       # (1, 1500, 768)
    real_features = features[:, :real_frames, :]    # (1, T, 768)

    IF Hindi:
        logits = ctc_head_hi(real_features)         # (1, T, 257)
    ELIF Marathi:
        logits = ctc_head_mr(real_features)         # (1, T, 257)

    # Greedy CTC decode
    predicted_ids = argmax(logits, dim=-1)           # (T,) — one ID per frame
    collapsed = unique_consecutive(predicted_ids)    # remove repeats
    tokens = [vocab[lang][id] for id in collapsed if id != 256]  # remove blanks
    predicted_text = ''.join(tokens).replace('▁', ' ').strip()

    # Compare with ground truth
    clip_wer = WER(predicted_text, ground_truth)

COMPUTE overall WER per language
COMPARE:
    Whisper Small baseline:     Hindi 103.6%, Marathi 132.7%
    IndicConformer (teacher):   Hindi  32.4%, Marathi  44.9%
    Our student (after KD):     Hindi  ??.?%, Marathi  ??.?%
    SUCCESS = student WER improves compared to Whisper Small baseline
```

---

## Step-by-Step Execution Plan

```
STEP 0: Pre-compute teacher logits              STATUS: DONE ✓
  Script: step0_precompute_teacher_logits.py
  Run on: HPC (GPU), 46 min for 9,169 train + 25 min for 1,171 dev
  Verified: 10,340/10,340 clips decoded correctly (100%)
  Output: teacher_logits/train/ (943.7 MB), teacher_logits/dev/ (122.2 MB)

STEP 1: Train semantic MSE                       STATUS: DONE ✓
  Script: step1_semantic_mse_training.py
  16 epochs, early stopped at epoch 16 (patience 5)
  Best dev MSE: 1.5418 (epoch 11)
  Train MSE: 4.2345 → 0.4617 (89.1% reduction)
  Training time: 712 min on V100

STEP 2: Evaluate WER                             STATUS: DONE ✓
  Script: step2_semantic_evaluate.py
  Decoded from CTC Head 2 using teacher's BPE vocab
  Results (best_dev.pt, epoch 11):
    Hindi:   39.72% WER (baseline 161.81%, teacher 39.01%)
    Marathi: 68.11% WER (baseline 170.78%, teacher 44.61%)
    Overall: 47.86% WER

STEP 3: Joint training (MSE + CTC)              STATUS: NEXT (PLANNED)
  Script: step3_semantic_joint_training.py (TO WRITE)
  Warm start: encoder + CTC Head 2 from step1 best_dev.pt
  New component: CTC Head 1 (768→85) for CTC loss with ground truth
  Combined loss = α × MSE_semantic + (1-α) × CTC
  α schedule: 0.3 → 0.2 → 0.1 (decay over epochs)
  Improvements: SpecAugment, dropout(0.1), freeze bottom 6 layers, grad_accum=4
  Dev metric: WER from CTC Head 1 (not MSE)
  Detailed architecture & pseudocode: see "Step 3" section above

STEP 4: Evaluate joint model                    STATUS: AFTER STEP 3
  Script: step4_semantic_joint_evaluate.py
  Decode from CTC Head 1 (85 char vocab), compute WER
  Can now evaluate ALL 3 languages (English too!)

STEP 5: Merge with acoustic branch              STATUS: LATER
  Combine acoustic MSE + semantic MSE + CTC
  3-way loss
  Only after both branches independently verified
```

---

## Step 3: Joint Training — Architecture & Pseudocode

### What is Joint Training?

```
In Steps 0-2, we trained the encoder + CTC Head 2 using ONLY MSE loss.
The student learned to mimic the teacher's logits — but never saw actual text.

CTC Head 2 (768→257) is a BRIDGE layer — it projects into the teacher's
BPE vocabulary space. It can decode text using the teacher's BPE tokens,
but that's not our final goal.

Our actual goal: transcribe children's speech using our OWN character
vocabulary (85 tokens: Devanagari + English + special tokens, from vocab.json).

Joint training adds a SECOND head — CTC Head 1 (768→85) — that predicts
our character vocabulary, trained with CTC loss against ground truth text.

The encoder now receives gradients from BOTH losses:
  1. MSE loss (via CTC Head 2): "keep producing features the teacher likes"
  2. CTC loss (via CTC Head 1): "also make features that decode into correct text"

This is exactly what the CARE paper does: dual-loss knowledge distillation.
```

### Why Not Just Use CTC Loss Alone?

```
We tried CTC-only in the acoustic branch (Sequential CTC → 81.86% WER).
The encoder was frozen and CTC head had to work with whatever features existed.

Joint training in the acoustic branch gave 19.97% WER — a massive improvement.
The same principle applies here in the semantic branch.

Without MSE loss: encoder forgets IndicConformer knowledge, CTC optimizes alone.
Without CTC loss: encoder matches teacher logits but can't transcribe in our vocab.
With BOTH: encoder keeps teacher knowledge AND learns to transcribe. Best of both.
```

### Architecture Diagram

```
            Audio Input: child saying "नीतू के घर में गाय है"
            (218,560 samples = 13.66 seconds at 16kHz)
                              │
                              ▼
                  ┌───────────────────────┐
                  │ WhisperFeatureExtract │
                  │   (mel spectrogram)   │
                  └───────────┬───────────┘
                              │
                              ▼
                        (1, 80, 3000)
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Whisper Encoder      │
                  │  12 Transformer layers│
                  │  hidden_dim = 768     │
                  │  UNFROZEN             │
                  │  (bottom 6 FROZEN*)   │
                  └───────────┬───────────┘
                              │
                              ▼
                        (1, 1500, 768)
                              │
                       ┌──────┴──────┐
                       │             │
                  Slice real     Slice real
                  frames         frames
                       │             │
                       ▼             ▼
                 (1, 683, 768)  (1, 683, 768)
                       │             │
          ┌────────────┘             └────────────┐
          │                                       │
          ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────┐
│  CTC Head 2 (Hindi)  │              │  CTC Head 1          │
│  Linear(768 → 257)   │              │  Linear(768 → 85)    │
│  + Dropout(0.1)*     │              │  + Dropout(0.1)*     │
│  WARM START from     │              │  TRAINED FROM SCRATCH│
│  step1 checkpoint    │              │  (random init)       │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                     │
           ▼                                     ▼
     (1, 683, 257)                         (1, 683, 85)
     student logits                        char logits
           │                                     │
           │                                     │
  ┌────────┴────────────┐                ┌───────┴───────────┐
  │  TEACHER LOGITS     │                │  GROUND TRUTH     │
  │  (pre-computed)     │                │  TEXT              │
  │                     │                │                    │
  │  teacher_logits.pt  │                │  "नीतू के घर में  │
  │  (T_teacher, 257)   │                │   गाय है"         │
  │       │             │                │       │            │
  │       ▼             │                │       ▼            │
  │  F.interpolate      │                │  text_to_indices() │
  │  (171→683 frames)   │                │  [12, 5, ...]     │
  │       │             │                │       │            │
  │       ▼             │                │       ▼            │
  │  (1, 683, 257)      │                │  target_indices    │
  │  teacher aligned    │                │  (variable length) │
  └────────┬────────────┘                └───────┬───────────┘
           │                                     │
           ▼                                     ▼
  ┌──────────────────┐                  ┌──────────────────┐
  │    MSE LOSS      │                  │    CTC LOSS      │
  │                  │                  │                  │
  │  mean((student   │                  │ CTCLoss(blank=0) │
  │   - teacher)^2)  │                  │ log_softmax →    │
  │  over 683×257    │                  │ sum over all     │
  │                  │                  │ valid alignments │
  └────────┬─────────┘                  └────────┬─────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │   COMBINED LOSS      │
               │                      │
               │   L = α × MSE       │
               │     + (1-α) × CTC   │
               │                      │
               │   α = 0.3 (start)    │
               │   α decays to 0.1    │
               │   over training      │
               └──────────┬───────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │  loss.backward()     │
               │                      │
               │  Gradients flow to:  │
               │   ✅ CTC Head 1     │
               │   ✅ CTC Head 2     │
               │   ✅ Encoder top 6  │
               │   ❌ Encoder bot 6* │
               │   ❌ Teacher (pre-  │
               │      computed)       │
               └──────────┬───────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │  optimizer.step()    │
               │  Update weights     │
               └──────────────────────┘

    * = Training improvements (see below)
```

### What Gets Updated

```
COMPONENT                    PARAMS      STATUS          WARM START FROM
─────────────────────────────────────────────────────────────────────────
Encoder layers 0-5           ~44M        FROZEN*         step1 best_dev.pt
Encoder layers 6-11          ~44M        UNFROZEN        step1 best_dev.pt
CTC Head 2 (Hindi, 768→257)  ~197K       UNFROZEN        step1 best_dev.pt
CTC Head 2 (Marathi, 768→257)~197K       UNFROZEN        step1 best_dev.pt
CTC Head 1 (768→85)          ~65K        UNFROZEN        random init (new)
─────────────────────────────────────────────────────────────────────────
Total trainable:             ~44.5M      (50% of encoder + both heads)
Total frozen:                ~44M        (bottom 6 encoder layers)

* Layer freezing: bottom 6 encoder layers learned low-level speech features
  during step1. These are already good. Freezing them:
  - Prevents catastrophic forgetting of learned representations
  - Reduces GPU memory (no gradients for 50% of encoder)
  - Speeds up training (~1.5x)
  Top 6 layers are unfrozen because they do high-level feature extraction
  that benefits from joint optimization with CTC.
```

### Training Improvements (Saved from Step 1 Analysis)

```
ISSUE 1: No SpecAugment
  Step 1 had no data augmentation → model memorizes training clips.
  FIX: Apply SpecAugment (frequency masking + time masking) on mel input.
  - freq_mask: 2 masks, max width 15 bins (out of 80)
  - time_mask: 2 masks, max width 50 frames
  Same as Whisper/Wav2Vec 2.0 standard augmentation.

ISSUE 2: No Dropout
  Step 1 had no dropout → CTC heads overfit on training data.
  FIX: Add Dropout(0.1) before each CTC head.
  Small enough to not hurt convergence, enough to regularize.

ISSUE 3: Full Encoder Unfrozen
  Step 1 trained all 12 layers → bottom layers drift from pretrained features.
  FIX: Freeze bottom 6 layers. Only train top 6 + heads.
  Pretrained low-level features are already good for mel → phonemes.

ISSUE 4: Batch Size = 1
  Step 1 processed one clip at a time → noisy gradients, slow convergence.
  FIX: Batch size 4-8 with gradient accumulation if GPU memory limited.
  More stable gradients → smoother loss curve → better final model.
  Implementation: accumulate gradients over N clips, step every N.
```

### Alpha Schedule

```
α controls the balance between MSE (teacher knowledge) and CTC (transcription).

APPROACH: Start with higher α (more teacher guidance), decay over training.

  Epoch 1-5:   α = 0.3  (30% MSE, 70% CTC)
  Epoch 6-15:  α = 0.2  (20% MSE, 80% CTC)
  Epoch 16+:   α = 0.1  (10% MSE, 90% CTC)

WHY start at 0.3, not 0.5?
  - CTC Head 1 is randomly initialized → needs strong CTC gradient early
  - Encoder already learned from teacher in step1 → doesn't need as much MSE
  - In acoustic branch joint training, α=0.5 worked but encoder was freshly
    combined. Here we have a warm-started encoder that already knows teacher patterns.

WHY decay to 0.1?
  - As CTC head learns, the model should optimize more for actual transcription
  - MSE keeps the encoder "anchored" to teacher features but shouldn't dominate
  - Fully removing MSE (α=0) risks catastrophic forgetting of teacher knowledge
```

### Pseudocode: step3_semantic_joint_training.py

```
PURPOSE: Joint training with MSE (teacher) + CTC (ground truth) losses.
         Encoder receives gradients from BOTH — learns features that are
         useful for matching teacher AND for transcription.

INPUT:   checkpoints/semantic_mse/best_dev.pt  (from step1)
         teacher_logits/train/*.pt             (from step0)
         ASER-Dataset/splits/asr_train.csv
         ASER-Dataset/splits/asr_dev.csv
         ASER-Dataset/vocab.json               (85 char tokens)
OUTPUT:  checkpoints/semantic_joint/best_dev.pt

──────────────────────────────────────────────────────────────

# ═══ SETUP ═══

# Load character vocabulary (for CTC Head 1)
char_to_idx = load("ASER-Dataset/vocab.json")    # 85 tokens
idx_to_char = reverse(char_to_idx)
BLANK_IDX = char_to_idx["<blank>"]                # 0

# Load Whisper encoder + CTC Head 2 from step1 checkpoint
checkpoint = torch.load("checkpoints/semantic_mse/best_dev.pt")
encoder = WhisperModel("whisper-small").encoder
encoder.load_state_dict(checkpoint["encoder_state_dict"])

ctc_head_hi = Linear(768, 257)     # bridge layer (teacher BPE space)
ctc_head_hi.load_state_dict(checkpoint["ctc_head_hi_state_dict"])

ctc_head_mr = Linear(768, 257)     # bridge layer (teacher BPE space)
ctc_head_mr.load_state_dict(checkpoint["ctc_head_mr_state_dict"])

# Create CTC Head 1 — NEW, random init (our char vocab)
ctc_head_char = Linear(768, 85)    # decoding head (our vocabulary)
dropout = Dropout(0.1)

# Freeze bottom 6 encoder layers
FOR layer in encoder.layers[:6]:
    layer.requires_grad_(False)
# encoder.layers[6:11] remain unfrozen

# Trainable params: encoder layers 6-11 + ctc_head_hi + ctc_head_mr + ctc_head_char
optimizer = AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
scheduler = linear_warmup(1000 steps) + cosine_decay
ctc_loss_fn = CTCLoss(blank=0, zero_infinity=True)
grad_accum_steps = 4   # effective batch size = 4

# ═══ DATA LOADING ═══

# Load training clips (Hindi + Marathi only)
train_clips = load_csv("asr_train.csv")
train_clips = [c for c in train_clips if c.language in ("Hindi", "Marathi")]
# → 9,169 clips

dev_clips = load_csv("asr_dev.csv")
dev_clips = [c for c in dev_clips if c.language in ("Hindi", "Marathi")]

# ═══ SPECAUGMENT ═══

def apply_spec_augment(mel_features):
    """
    Apply SpecAugment to mel spectrogram (1, 80, 3000).
    - 2 frequency masks (max width 15 out of 80 bins)
    - 2 time masks (max width 50 out of 3000 frames)
    """
    FOR i in range(2):
        f = random(0, 15)
        f0 = random(0, 80 - f)
        mel_features[:, f0:f0+f, :] = 0      # mask frequency band

    FOR i in range(2):
        t = random(0, 50)
        t0 = random(0, 3000 - t)
        mel_features[:, :, t0:t0+t] = 0      # mask time band

    RETURN mel_features

# ═══ ALPHA SCHEDULE ═══

def get_alpha(epoch):
    IF epoch <= 5:   RETURN 0.3
    IF epoch <= 15:  RETURN 0.2
    RETURN 0.1

# ═══ TRAINING LOOP ═══

best_dev_wer = infinity
patience_counter = 0
max_epochs = 30
patience = 7

FOR epoch = 1 to max_epochs:

    alpha = get_alpha(epoch)
    SHUFFLE train_clips
    epoch_mse_losses = []
    epoch_ctc_losses = []
    optimizer.zero_grad()

    FOR i, clip in enumerate(train_clips):

        # ─── Load audio ───
        wav = torchaudio.load(clip.audio_path)
        wav = resample_to_16k_if_needed(wav)
        wav = mono(wav).squeeze()       # (samples,)
        num_samples = len(wav)

        # ─── Mel spectrogram + SpecAugment ───
        mel = feat_ext(wav, return_tensors="pt")    # (1, 80, 3000)
        mel.input_features = apply_spec_augment(mel.input_features)

        # ─── Encoder forward ───
        features = encoder(mel).last_hidden_state    # (1, 1500, 768)
        real_frames = min(num_samples // 160 // 2, 1500)
        real_features = features[:, :real_frames, :]  # (1, T, 768)

        # ─── BRANCH A: MSE Loss (teacher distillation via CTC Head 2) ───
        # Select language-specific CTC Head 2
        IF clip.lang_code == "hi":
            student_logits_257 = ctc_head_hi(dropout(real_features))  # (1, T, 257)
        ELIF clip.lang_code == "mr":
            student_logits_257 = ctc_head_mr(dropout(real_features))  # (1, T, 257)

        # Load + align pre-computed teacher logits
        teacher = torch.load(f"teacher_logits/train/{clip_uid}.pt")
        teacher_logits = teacher['logits'].float()    # (T_teacher, 257)
        teacher_aligned = F.interpolate(
            teacher_logits.T.unsqueeze(0),             # (1, 257, T_teacher)
            size=real_frames,
            mode='linear'
        ).squeeze(0).T.unsqueeze(0)                    # (1, T, 257)

        mse_loss = F.mse_loss(student_logits_257, teacher_aligned)

        # ─── BRANCH B: CTC Loss (ground truth via CTC Head 1) ───
        char_logits = ctc_head_char(dropout(real_features))  # (1, T, 85)
        log_probs = char_logits.log_softmax(dim=-1)          # (1, T, 85)
        log_probs = log_probs.permute(1, 0, 2)               # (T, 1, 85) for CTC

        # Convert ground truth text → character indices
        target_indices = text_to_indices(clip.ground_truth, char_to_idx)
        targets = torch.tensor(target_indices)
        input_lengths = torch.tensor([real_frames])
        target_lengths = torch.tensor([len(target_indices)])

        ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        # ─── Combined loss ───
        total_loss = alpha * mse_loss + (1 - alpha) * ctc_loss
        total_loss = total_loss / grad_accum_steps   # normalize for accumulation

        total_loss.backward()

        # ─── Gradient accumulation step ───
        IF (i + 1) % grad_accum_steps == 0:
            clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_mse_losses.append(mse_loss.item())
        epoch_ctc_losses.append(ctc_loss.item())

    # ─── Dev evaluation (WER from CTC Head 1) ───
    # This is what matters — can the model actually transcribe?
    dev_wer = evaluate_dev_wer(
        encoder, ctc_head_char, dev_clips,
        char_to_idx, idx_to_char
    )
    dev_mse = evaluate_dev_mse(
        encoder, ctc_head_hi, ctc_head_mr, dev_clips
    )

    PRINT f"Epoch {epoch}: MSE={avg(epoch_mse_losses):.4f} "
          f"CTC={avg(epoch_ctc_losses):.4f} "
          f"DevWER={dev_wer:.2f}% DevMSE={dev_mse:.4f} α={alpha}"

    # ─── Checkpoint on best dev WER (not MSE!) ───
    IF dev_wer < best_dev_wer:
        best_dev_wer = dev_wer
        patience_counter = 0
        SAVE {
            "encoder_state_dict": encoder.state_dict(),
            "ctc_head_hi_state_dict": ctc_head_hi.state_dict(),
            "ctc_head_mr_state_dict": ctc_head_mr.state_dict(),
            "ctc_head_char_state_dict": ctc_head_char.state_dict(),
            "epoch": epoch,
            "dev_wer": dev_wer,
            "dev_mse": dev_mse,
            "alpha": alpha,
        } → "checkpoints/semantic_joint/best_dev.pt"
    ELSE:
        patience_counter += 1
        IF patience_counter >= patience:
            PRINT "Early stopping at epoch {epoch}"
            STOP

# ═══ DEV EVALUATION FUNCTION ═══

def evaluate_dev_wer(encoder, ctc_head_char, dev_clips, char_to_idx, idx_to_char):
    """
    Greedy CTC decode from CTC Head 1 → compute WER against ground truth.
    This tells us if the model can actually transcribe speech.
    """
    encoder.eval()
    ctc_head_char.eval()
    all_refs, all_hyps = [], []

    FOR clip in dev_clips:
        mel = feat_ext(wav, return_tensors="pt")
        features = encoder(mel).last_hidden_state
        real_features = features[:, :real_frames, :]

        logits = ctc_head_char(real_features)          # (1, T, 85)
        indices = argmax(logits, dim=-1).squeeze()      # (T,)
        collapsed = unique_consecutive(indices)
        chars = [idx_to_char[idx] for idx in collapsed if idx != BLANK_IDX]
        predicted = ''.join(chars).replace('<space>', ' ').strip()

        all_refs.append(clip.ground_truth)
        all_hyps.append(predicted)

    RETURN compute_corpus_wer(all_refs, all_hyps) * 100

    encoder.train()
    # Re-freeze bottom 6 layers after eval
```

### Data Flow Summary

```
                    ┌────────────────────────────────┐
                    │   WHAT EACH COMPONENT DOES     │
                    └────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│  ENCODER (Whisper Small, 12 layers)                              │
│                                                                   │
│  Layers 0-5: FROZEN — extract low-level speech features          │
│              (mel → phonemes, speaker-invariant patterns)         │
│              Already trained in step1, preserved.                 │
│                                                                   │
│  Layers 6-11: UNFROZEN — high-level feature extraction           │
│               Receives gradients from BOTH losses.                │
│               Learns to produce features useful for:              │
│                 - Matching teacher logits (MSE branch)            │
│                 - Predicting characters (CTC branch)              │
└───────────────────────────────────────────────────────────────────┘
                              │
                              │ (1, T, 768)
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
     ┌──────────────────────┐  ┌──────────────────────┐
     │  CTC HEAD 2          │  │  CTC HEAD 1          │
     │  (Bridge Layer)      │  │  (Decoding Head)     │
     │                      │  │                      │
     │  768 → 257           │  │  768 → 85            │
     │  Teacher BPE vocab   │  │  Our char vocab      │
     │  Per-language (Hi/Mr)│  │  Shared (all langs)  │
     │  Warm-started        │  │  Random init         │
     │                      │  │                      │
     │  PURPOSE:            │  │  PURPOSE:            │
     │  Keep encoder        │  │  Learn to actually   │
     │  aligned with        │  │  transcribe speech   │
     │  IndicConformer      │  │  in our vocabulary   │
     │                      │  │                      │
     │  LOSS: MSE           │  │  LOSS: CTC           │
     │  (vs teacher logits) │  │  (vs ground truth)   │
     │                      │  │                      │
     │  After training:     │  │  After training:     │
     │  DISCARDED           │  │  KEPT for inference  │
     └──────────────────────┘  └──────────────────────┘
```

### Evaluation After Joint Training (Step 4)

```
Evaluate the jointly trained model using CTC Head 1 (NOT Head 2).

CTC Head 1 outputs our character vocabulary (85 tokens).
Decoding: greedy CTC → character sequence → text.

Script: step4_semantic_joint_evaluate.py

Expected results:
  - Hindi:   < 39.72% (step2 result via Head 2)
  - Marathi: < 68.11% (step2 result via Head 2)
  - English: NOW POSSIBLE — CTC Head 1 supports all 3 languages
             (vocab.json has both English and Devanagari characters)

Key difference from step2:
  - step2 decoded from CTC Head 2 using teacher's BPE vocab (257 tokens)
  - step4 decodes from CTC Head 1 using our char vocab (85 tokens)
  - CTC Head 1 is the final decoding head — this is the real WER
```

---

## Comparison: Acoustic Branch vs Semantic Branch

```
                          ACOUSTIC BRANCH          SEMANTIC BRANCH
                          (DONE)                   (THIS PLAN)
──────────────────────────────────────────────────────────────────
Teacher                   Kid-Whisper Medium       IndicConformer 600M
Teacher output            Encoder features         CTC logits
Distillation level        Hidden representations   Output predictions
Teacher dim               (T, 1024)                (T', 257) per lang
Teacher frame rate        50.0 fps                 12.5 fps
Bridge layer              Projection(768→1024)     CTC Head 2(768→257)
Bridge layer purpose      Match feature dims       Match logit space
Time alignment            NOT needed               YES — interpolate 4x
                          (same Whisper family)     (different architectures)
Per-language heads?       No (1 projection)        Yes (Hindi + Marathi)
Languages trained         All 3 (Hi+Mr+En)         Hindi + Marathi only
Training clips            13,765                   9,169
Loss                      MSE on features          MSE on logits
Encoder start             Pretrained Whisper       Pretrained Whisper
                          Small (fresh)            Small (fresh)
Pre-compute teacher?      Yes (features)           Yes (logits)
──────────────────────────────────────────────────────────────────
```
