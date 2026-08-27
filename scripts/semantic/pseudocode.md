# Semantic Branch: CTC-based Knowledge Distillation — Pseudocode

## Goal
Train Whisper Small's encoder to transcribe Hindi/Marathi by learning from
IndicConformer's transcriptions using CTC loss.

## Overview

```
STEP 0: Generate teacher transcripts (one-time, offline)
STEP 1: Build student model (Whisper encoder + CTC head)
STEP 2: Prepare training data (tokenize teacher transcripts)
STEP 3: Training loop
STEP 4: Evaluation (CTC greedy decode → WER)
```

---

## STEP 0: Generate Teacher Transcripts

```
PURPOSE:
    IndicConformer takes ~4 seconds per clip.
    Training runs multiple epochs (each epoch = all clips again).
    If we run IndicConformer during training:
        9169 clips × 4 sec × 20 epochs = ~204 hours wasted.
    Pre-computing once takes ~10 hours, then we reuse forever.

INPUT:
    asr_train.csv → 13,765 clips total
    Filter to Hindi (6,304) + Marathi (2,865) = 9,169 clips
    WHY skip English: IndicConformer does not support English.
                      English clips will only be used in acoustic branch.

PROCESS:
    LOAD IndicConformer model (ai4bharat/indic-conformer-600m-multilingual)

    FOR each Hindi/Marathi training clip:
        audio = LOAD audio file (16kHz, mono)
        language_code = "hi" for Hindi, "mr" for Marathi
        teacher_transcript = IndicConformer.forward(audio, lang=language_code)
        SAVE: audio_path, teacher_transcript, ground_truth, language, duration

OUTPUT:
    teacher_transcripts.csv
    Columns: audio_path, teacher_transcript, ground_truth, language, duration
```

---

## STEP 1: Build Student Model

```
PURPOSE:
    Whisper Small's encoder understands audio but has no CTC capability.
    We add a single Linear layer (the "CTC head") on top of the encoder.
    This converts encoder features (768 dims) into vocabulary predictions.

ARCHITECTURE:
    ┌─────────────────────────────────────┐
    │         Whisper Small Encoder        │
    │    (12 transformer layers, 768 dim)  │
    │         TRAINABLE — 242M params      │
    └──────────────┬──────────────────────┘
                   │
                   ▼
            encoder features
            shape: (1500 frames, 768 dims)
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │    Dropout(0.1)                      │
    │    WHY: prevents overfitting of the  │
    │         CTC head to training data    │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │    CTC Head: Linear(768 → 51865)    │
    │    This is ONE matrix multiplication │
    │    768 = encoder hidden dimension    │
    │    51865 = Whisper's vocabulary size  │
    │    TRAINABLE — 39.8M params (new)    │
    └──────────────┬──────────────────────┘
                   │
                   ▼
            logits
            shape: (1500 frames, 51865 vocab)
            Each frame now has a score for every possible token

    WHY this exact pattern:
        This is identical to how HuggingFace implements Wav2Vec2ForCTC,
        HubertForCTC, and WavLMForCTC. We follow the same standard:
        encoder → dropout → single Linear layer → CTC loss.
        Reference: HuggingFace transformers, Wav2Vec2ForCTC source code.

    TOTAL TRAINABLE PARAMETERS:
        Whisper encoder: ~242M params (fine-tuned)
        CTC head:        ~39.8M params (trained from scratch)
        Total:           ~282M params

*** IMPORTANT PARAMETER: vocab_size = 51865 ***
    WHY 51865: This is Whisper Small's full vocabulary size (multilingual).
    It includes Hindi, Marathi, English, and 96 other languages.
    The CTC head must match this size so we can use Whisper's tokenizer
    to convert between text and token IDs.

*** IMPORTANT PARAMETER: blank_id = 0 ***
    WHY 0: CTC needs a special "blank" token that means "no prediction
    at this frame." By convention (following Wav2Vec2ForCTC), we use
    token index 0. This is the same as Whisper's padding token.
```

---

## STEP 2: Prepare Training Data

```
PURPOSE:
    CTC loss needs:
      1. Audio → mel spectrogram → input to encoder
      2. Target text → token IDs (integers)
      3. Input length (how many real frames, excluding padding)
      4. Target length (how many tokens in the target)

    We must prepare all 4 for each training clip.

PROCESS:
    LOAD teacher_transcripts.csv from Step 0
    LOAD Whisper's processor (feature extractor + tokenizer)

    FOR each clip:
        ── Audio Preparation ──
        audio = LOAD audio file (16kHz, mono, max 30 seconds)
        mel_spectrogram = WhisperFeatureExtractor(audio)
            shape: (80 frequency bins, 3000 frames)
            WHY 3000: Whisper ALWAYS pads audio to 30 seconds.
            A 5-second clip gets 25 seconds of silence padding.

        ── Calculate Real Frame Count ──
        audio_samples = length of audio in samples (before padding)
        real_encoder_frames = audio_samples ÷ 160 ÷ 2

        *** IMPORTANT: WHY ÷ 160 ÷ 2 ***
            ÷ 160: Whisper's mel spectrogram uses hop_length = 160 samples
                   16000 samples/sec ÷ 160 = 100 mel frames per second
            ÷ 2:   Whisper encoder has a conv layer with stride 2
                   This halves the frame count: 100 ÷ 2 = 50 frames/sec

            Example: 10-second clip
                samples = 160,000
                mel frames = 160,000 ÷ 160 = 1,000
                encoder frames = 1,000 ÷ 2 = 500 real frames
                (remaining 1000 frames out of 1500 are padding)

        ── Tokenize Target Text ──
        target_tokens = WhisperTokenizer(teacher_transcript)
            Example: "नमस्ते" → [34567, 12890, 45123]  (3 BPE tokens)

        *** IMPORTANT: target_length MUST be ≤ real_encoder_frames ***
            WHY: CTC aligns T input frames to N output tokens.
            If N > T, there aren't enough frames to assign one per token.
            CTC loss becomes infinite → training crashes.

            In practice: shortest clip ≈ 0.5 sec = 25 frames.
            Shortest target ≈ 1-3 tokens. So 25 ≥ 3. We are safe.
            But we should CHECK this and skip any violating clips.

OUTPUT:
    For each clip, we have:
        input_features:     (80, 3000)  — mel spectrogram
        real_frame_count:   integer     — e.g., 500 for a 10-sec clip
        target_token_ids:   list[int]   — e.g., [34567, 12890, 45123]
        target_length:      integer     — e.g., 3
```

---

## STEP 3: Training Loop

```
*** IMPORTANT PARAMETERS ***
    learning_rate   = 1e-4
        WHY: Standard for CTC fine-tuning (Wav2Vec2ForCTC tutorials).
        Too high → CTC loss explodes. Too low → no learning.

    warmup_steps    = 500
        WHY: CTC loss is unstable at the start because the CTC head
        is randomly initialized. Warmup lets it stabilize gradually.
        Without warmup, early gradients can be too large.

    max_grad_norm   = 1.0
        WHY: CTC loss can produce very large gradients, especially
        when the model is very wrong (early training). Gradient clipping
        prevents these spikes from destroying the model weights.
        This is CRITICAL for CTC — almost all CTC papers use this.

    epochs          = 20
    patience        = 5
        WHY: Early stopping. If dev loss doesn't improve for 5 epochs,
        the model has converged or is overfitting. Stop and use best.

    seed            = 42
        WHY: Reproducibility. Same seed = same results every run.

    ctc_zero_infinity = True
        WHY: If any clip has an impossible CTC alignment (target longer
        than input), the loss becomes infinity. This flag replaces
        infinity with zero, preventing NaN gradients from crashing
        training. Safety net — should rarely trigger if data is clean.


SETUP:
    student_model = WhisperEncoder + Dropout + CTC_Head
    student_model.to(GPU)

    optimizer = AdamW(
        params = all parameters of student_model,
        lr = learning_rate
    )

    scheduler = LinearWarmup(
        warmup_steps = 500,
        then linear decay to 0
    )

    ctc_loss_fn = CTCLoss(
        blank = 0,
        zero_infinity = True,
        reduction = "mean"
    )
        WHY reduction="mean" instead of "sum":
            "sum" makes the loss proportional to sequence length.
            Long clips would dominate training, short clips ignored.
            "mean" treats all clips equally regardless of length.

    SET random seed for reproducibility
    DISABLE cudnn for CTC loss
        WHY: Known PyTorch bug — cudnn's CTC implementation can produce
        incorrect gradients on some GPU architectures. Disabling it uses
        PyTorch's native (correct) implementation. Slightly slower but
        guarantees correctness. Referenced in Wav2Vec2ForCTC source code.


TRAINING:
    best_dev_loss = infinity
    patience_counter = 0

    FOR epoch = 1 to max_epochs:
        student_model.train()
        SHUFFLE all training clips
        epoch_losses = []

        FOR each clip in training clips:
            ── Forward Pass ──

            1. LOAD mel spectrogram from audio file
               input_features shape: (1, 80, 3000)

            2. COMPUTE real frame count
               real_frames = audio_samples ÷ 160 ÷ 2

            3. RUN through Whisper encoder
               encoder_output = student_model.encoder(input_features)
               shape: (1, 1500, 768)

            4. APPLY dropout
               encoder_output = Dropout(encoder_output)

            5. RUN through CTC head
               logits = CTC_Head(encoder_output)
               shape: (1, 1500, 51865)

            6. COMPUTE log probabilities

               *** IMPORTANT: Must cast to float32 ***
               WHY: CTC loss does NOT support float16/bfloat16.
               The log_softmax computation needs full precision to
               avoid numerical underflow. Even if training with mixed
               precision, this step MUST be float32.

               log_probs = log_softmax(logits.float(), dim=-1)
               shape: (1, 1500, 51865)

            7. TRANSPOSE to time-first format

               *** IMPORTANT: CTC expects (Time, Batch, Vocab) ***
               WHY: This is PyTorch's CTCLoss API requirement.
               Most models output (Batch, Time, Vocab).
               We must transpose: (1, 1500, 51865) → (1500, 1, 51865)

               log_probs = log_probs.transpose(0, 1)

            ── Compute Loss ──

            8. PREPARE targets
               target_ids = tokenized teacher transcript  (1D tensor)
               input_length = tensor([real_frames])       (not 1500!)
               target_length = tensor([number of tokens])

               *** IMPORTANT: input_length = real_frames, NOT 1500 ***
               WHY: If we pass 1500 for a 5-second clip, CTC will try
               to align the target text against 1250 padding frames.
               The padding frames have near-zero features → random logits.
               CTC would learn that "silence = random Hindi text" which
               is completely wrong.
               We pass only the real frame count so CTC only looks at
               frames that contain actual speech.

            9. COMPUTE CTC loss
               loss = ctc_loss_fn(log_probs, target_ids, input_length, target_length)

            ── Backward Pass ──

            10. optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            11. RECORD loss value
                epoch_losses.append(loss.item())

            12. CLEAR GPU cache
                WHY: Whisper processes one clip at a time (batch_size=1
                due to variable lengths). GPU memory fragments over time.
                Clearing cache prevents out-of-memory errors mid-epoch.

        ── End of Epoch ──

        13. PRINT epoch summary
            avg_train_loss = mean(epoch_losses)
            PRINT: epoch number, avg loss, time taken, GPU memory

        ── Dev Evaluation ──

        14. student_model.eval()
            dev_losses = []
            WITH no_grad:
                FOR each clip in dev set (Hindi/Marathi only):
                    COMPUTE forward pass (steps 1-9 above)
                    dev_losses.append(loss.item())
            avg_dev_loss = mean(dev_losses)

        15. PRINT train vs dev comparison
            PRINT: train_loss, dev_loss, gap (dev - train)
            WHY: If gap grows → overfitting. If both decrease → healthy.

        ── Early Stopping ──

        16. IF avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                patience_counter = 0
                SAVE model as "best_dev_model.pt"
            ELSE:
                patience_counter += 1
                IF patience_counter >= patience:
                    PRINT "Early stopping at epoch {epoch}"
                    STOP training

        ── Save Checkpoints ──

        17. SAVE "best_train_model.pt"  (if this epoch has lowest train loss)
            SAVE "epoch_{N}.pt"         (every epoch, for analysis)
```

---

## STEP 4: Evaluation

```
PURPOSE:
    After training, measure how well the student transcribes Hindi/Marathi.
    Use CTC greedy decoding to convert logits → text, then compute WER.

PROCESS:
    LOAD best_dev_model.pt
    student_model.eval()

    FOR each clip in asr_test.csv (Hindi + Marathi only):

        ── CTC Greedy Decode ──

        1. audio → encoder → CTC head → logits (1500, 51865)

        2. predicted_ids = argmax(logits, dim=-1) at each frame
           result: [0, 0, 34567, 34567, 0, 12890, 0, 0, ...]
           These are: [blank, blank, token1, token1, blank, token2, ...]

        3. COLLAPSE consecutive repeated tokens
           [34567, 34567] → [34567]

        4. REMOVE blank tokens (id=0)
           [0, 34567, 0, 12890, 0] → [34567, 12890]

        5. DECODE token IDs to text using Whisper tokenizer
           [34567, 12890] → "नमस्ते"

        ── Compute WER ──

        6. predicted_text = decoded text from step 5
           ground_truth = original transcript from asr_test.csv
           clip_wer = WER(predicted_text, ground_truth)

    COMPUTE overall WER per language (Hindi, Marathi)
    COMPARE with baseline:
        Whisper Small baseline:     Hindi 161.81%, Marathi 170.78%
        IndicConformer (teacher):   Hindi 39.01%,  Marathi 44.61%
        Our student (after KD):     Hindi ??,      Marathi ??
        SUCCESS = student WER closer to teacher than baseline

OUTPUT:
    experiments/semantic_distillation/
        results_summary.txt
        predictions_all.txt
        predictions_mismatches.txt
```

---

## Key Decisions and Their Justification

```
DECISION 1: Use Whisper's own BPE tokenizer (51865 tokens) for CTC
    WHY:   Simplest approach — no need to create a custom tokenizer.
           Whisper's tokenizer already handles Hindi, Marathi, English.
    RISK:  Large vocabulary (51865) can make CTC training harder.
           CTC typically works better with smaller vocabularies (500-5000).
    PLAN:  Try this first. If training is unstable or WER is poor,
           switch to a character-level tokenizer (~200 tokens).

DECISION 2: Fine-tune entire encoder (not just CTC head)
    WHY:   The encoder needs to learn Indian children's speech patterns.
           If we freeze the encoder and only train the CTC head, the
           head can only work with Whisper's existing representations —
           which are poor for Hindi/Marathi (161-170% WER baseline).
    RISK:  Overfitting if training data is small. We have 9169 clips
           (46 hours Hindi+Marathi) which should be sufficient.

DECISION 3: Teacher transcripts (not ground truth) as CTC targets
    WHY:   This is knowledge distillation. The student learns from
           the teacher's output, not from ground truth labels.
           IndicConformer's Hindi WER is 39% — not perfect, but
           much better than Whisper's 161%. The student learns
           the teacher's strengths while the acoustic branch (later)
           adds children's speech understanding.
    NOTE:  We can also experiment with mixing teacher transcripts and
           ground truth labels. But pure teacher labels is the standard
           KD approach (Kim & Rush, 2016).

DECISION 4: CTC loss reduction = "mean" (not "sum")
    WHY:   Our clips vary from 0.5s to 30s. With "sum", a 30-second
           clip contributes 60× more to the loss than a 0.5-second clip.
           "mean" normalizes by sequence length, treating all clips equally.

DECISION 5: Disable cudnn for CTC loss computation
    WHY:   PyTorch's cudnn CTC implementation has known gradient bugs
           on certain GPU architectures. HuggingFace's Wav2Vec2ForCTC
           explicitly disables it. We follow the same practice.
           Source: HuggingFace transformers, Wav2Vec2ForCTC, line ~1725.
```

---

## GPU Memory Estimate (RTX 4060, 8GB)

```
Whisper Small encoder (float32):   ~968 MB
CTC head Linear(768, 51865):       ~159 MB
Optimizer states (AdamW, 2× model): ~2254 MB
Activations + gradients (1 clip):  ~2000-3000 MB
────────────────────────────────────────────────
Estimated peak:                    ~5400-6400 MB
Available:                         7932 MB
Headroom:                          ~1500-2500 MB ✓

WHY batch_size=1:
    With batch_size=2, activations double → ~8400+ MB → OOM.
    We process one clip at a time. This is fine because CTC loss
    is computed per-clip anyway (each clip has different length).
```

---

## Files This Pipeline Will Create

```
scripts/
    step0_generate_teacher_transcripts.py   ← Run IndicConformer on training set
    step1_semantic_distillation.py          ← Main training script

benchmarks/
    teacher_transcripts.csv                 ← IndicConformer's predictions on train set

checkpoints/semantic/
    best_dev_model.pt                       ← Best model (lowest dev loss)
    best_train_model.pt                     ← Best model (lowest train loss)
    epoch_1.pt ... epoch_N.pt               ← Per-epoch checkpoints

experiments/semantic_distillation/
    results_summary.txt                     ← WER numbers
    predictions_all.txt                     ← GT vs Prediction for all test clips
    predictions_mismatches.txt              ← Only errors with +Added/-Removed
```

---

## Command Reference

```bash
# Step 0: Generate teacher transcripts (one-time, ~10 hours)
# Quick test first:
python scripts/step0_generate_teacher_transcripts.py --max_clips 20
# Full run:
python scripts/step0_generate_teacher_transcripts.py

# Step 1: Semantic distillation training
# Quick test (20 clips, 2 epochs):
python scripts/step1_semantic_distillation.py --max_clips 20 --epochs 2
# Full run (~20 epochs with early stopping):
python scripts/step1_semantic_distillation.py --epochs 20 --patience 5
```

---

## References

```
[1] Kim & Rush, "Sequence-Level Knowledge Distillation" (EMNLP 2016)
    — Foundation for using teacher transcripts as training targets

[2] Watanabe et al., "Hybrid CTC/Attention Architecture" (2017)
    — Adding CTC head to encoder-decoder models is standard practice

[3] Baevski et al., "wav2vec 2.0" (NeurIPS 2020)
    — CTC fine-tuning with attention masks for variable-length audio

[4] Radford et al., "Robust Speech Recognition via Large-Scale
    Weak Supervision" (Whisper paper, 2023)
    — Whisper pads all audio to 30 seconds → must handle padding

[5] HuggingFace Transformers — Wav2Vec2ForCTC source code
    — Our CTC head follows this exact implementation pattern:
      encoder → dropout → Linear → CTC loss
```
