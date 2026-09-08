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
  Run on: HPC (GPU), ~10 hours for 9,169 clips
  Verified: 5/5 clips decode correctly (100%)
  Output: teacher_logits/train/*.pt

STEP 1: Train semantic MSE                       STATUS: TO WRITE
  Script: step1_semantic_mse_training.py
  What to watch:
    - Train MSE loss going down?
    - Dev MSE loss going down?
    - Train-Dev gap not growing? (overfitting check)
  Test first: 50 clips, 3 epochs (~5 min locally)
  Full run: 9,169 clips, 20 epochs on HPC

STEP 2: Evaluate WER                             STATUS: TO WRITE
  Script: step2_semantic_evaluate.py
  Decode from CTC Head 2 using teacher's BPE vocab
  Compare with baselines (Whisper Small, IndicConformer)

STEP 3: Joint training (MSE + CTC)              STATUS: LATER
  Add CTC Head 1 (768→85) for CTC loss with ground truth
  Combined loss = alpha * MSE_semantic + (1-alpha) * CTC
  Only after Steps 1-2 confirm semantic MSE works

STEP 4: Merge with acoustic branch              STATUS: MUCH LATER
  Combine acoustic MSE + semantic MSE + CTC
  3-way loss
  Only after both branches independently verified
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
