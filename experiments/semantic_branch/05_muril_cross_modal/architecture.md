# Semantic Branch — MuRIL Cross-Modal Knowledge Distillation

## Goal
Train Whisper Small encoder to produce features that capture **word MEANING**
(from MuRIL text model), not just acoustic/phonological patterns.

This replaces IndicConformer (speech teacher) with MuRIL (text teacher) so
that the semantic encoder learns genuinely different features from the
acoustic encoder.

## Why Replace IndicConformer with MuRIL?

Current problem:
```
Acoustic encoder  → trained from Kid-Whisper (speech model)  → learns SOUND patterns
Semantic encoder  → trained from IndicConformer (speech model) → learns SOUND patterns
                                                                 ^^^ SAME TYPE!
Both encoders capture acoustic/phonological features = HIGH OVERLAP
Gated fusion has little complementary information to work with.
```

Proposed:
```
Acoustic encoder  → trained from Kid-Whisper (speech model)  → learns SOUND patterns
Semantic encoder  → trained from MuRIL (TEXT model)          → learns MEANING patterns
                                                                ^^^ DIFFERENT TYPE!
One captures HOW words sound. Other captures WHAT words mean = LOW OVERLAP
Gated fusion gets genuinely complementary features.
```

## What is MuRIL?

MuRIL = Multilingual Representations for Indian Languages (Google Research)
- Architecture: BERT-base (12 layers, 110M params)
- Output: 768-dim embedding per token (SAME as Whisper Small!)
- Trained on: Wikipedia + CommonCrawl in 17 Indian languages
- Includes: Hindi, Marathi, English (all 3 of our languages)
- HuggingFace: `google/muril-base-cased`
- Key feature: trained on PARALLEL translated text, so it knows that
  Hindi "घर" = Marathi "घर" = English "house" (cross-lingual)


## Architecture Diagram

```
TEACHER SIDE (MuRIL — frozen, runs on ground truth text):

    Ground truth text: "मीना पढ़ने जाती है"
        |
        v
    MuRIL Tokenizer
        |
        v
    Tokens: [CLS] ▁मीना ▁पढ़ने ▁जाती ▁है [SEP]
        |
        v
    MuRIL Encoder (frozen, 12 BERT layers)
        |
        v
    Token embeddings: (N_tokens, 768)
        [CLS]  = [0.12, -0.34, ...]    ← sentence context
        ▁मीना  = [0.78, 0.23, ...]     ← "Meena" in context
        ▁पढ़ने  = [-0.45, 0.67, ...]    ← "to study" in context
        ▁जाती  = [0.22, -0.89, ...]    ← "goes" (feminine) in context
        ▁है    = [0.11, 0.05, ...]     ← "is" in context

    Drop [CLS] and [SEP] → (4 tokens, 768)
                                |
                                |  These are the TARGETS
                                |
                                v

─── ALIGNMENT STEP ─────────────────────────────────────────────────

    We need to match 4 text tokens to ~200 audio frames.
    Use our EXISTING trained CTC model to find which frames
    correspond to which words:

    CTC alignment output:
        "मीना" → frames 1-40
        "पढ़ने" → frames 41-95
        "जाती" → frames 96-155
        "है"   → frames 156-185
        (rest) → silence/padding → IGNORED

    Now for each word, average the student's audio frames:
        speech_emb("मीना") = mean(student_frames[1:40])    → (768,)
        speech_emb("पढ़ने") = mean(student_frames[41:95])   → (768,)
        speech_emb("जाती") = mean(student_frames[96:155])  → (768,)
        speech_emb("है")   = mean(student_frames[156:185]) → (768,)

─── MSE LOSS ───────────────────────────────────────────────────────

    For each word:
        MSE(speech_emb("मीना"), muril_emb("मीना"))
        MSE(speech_emb("पढ़ने"), muril_emb("पढ़ने"))
        MSE(speech_emb("जाती"), muril_emb("जाती"))
        MSE(speech_emb("है"),   muril_emb("है"))

    Average over all words → MSE loss for this clip

────────────────────────────────────────────────────────────────────


STUDENT SIDE (Whisper encoder — trainable):

    Audio (.wav) — Hindi, Marathi, OR English
        |
        v
    Mel Spectrogram (1, 80, 3000) → SpecAugment
        |
        v
    Whisper Small Encoder (TRAINABLE, bottom 6 frozen)
        |
        v
    Frame-level features (1, T_frames, 768)
        |
        v
    Slice to real frames → (1, T, 768)
        |
        ├─── word-level avg pooling (via CTC alignment) ──→ MSE Loss vs MuRIL
        |
        └─── CTC Head 1: Linear(768→85) ──→ CTC Loss vs ground truth text


TOTAL LOSS:
    loss = α × MSE_muril + (1-α) × CTC_loss

    α starts at 0.3, decays to 0.1 over training
    (MSE is harder to optimize, so CTC should dominate)
```


## Step-by-Step: What Happens for ONE Training Clip

Example clip: Hindi child reading "मीना पढ़ने जाती है"

```
STEP 1: Get MuRIL embeddings (teacher side)
    Input text: "मीना पढ़ने जाती है"
    → MuRIL tokenizer → [CLS, ▁मीना, ▁पढ़ने, ▁जाती, ▁है, SEP]
    → MuRIL encoder (frozen) → 6 token embeddings of (768,)
    → Drop CLS and SEP → 4 word embeddings of (768,)
    These are PRE-COMPUTED offline (like we did for IndicConformer logits)

STEP 2: Get student encoder features (student side)
    Load audio → mel spectrogram → SpecAugment
    → Whisper encoder → frame features (1, T, 768)
    Example: T = 200 frames for a 4-second clip

STEP 3: CTC alignment (which frames = which words)
    Use our pre-trained acoustic CTC model to decode:
    → frame-level character predictions → group into words
    → "मीना" = frames 1-40, "पढ़ने" = frames 41-95, etc.

STEP 4: Word-level pooling
    speech_emb("मीना") = mean(encoder_output[1:40])     → (768,)
    speech_emb("पढ़ने") = mean(encoder_output[41:95])    → (768,)
    speech_emb("जाती") = mean(encoder_output[96:155])   → (768,)
    speech_emb("है")   = mean(encoder_output[156:185])  → (768,)

STEP 5: MSE loss
    loss_mse = average of:
        MSE(speech_emb("मीना"), muril_emb("मीना"))
        MSE(speech_emb("पढ़ने"), muril_emb("पढ़ने"))
        MSE(speech_emb("जाती"), muril_emb("जाती"))
        MSE(speech_emb("है"),   muril_emb("है"))

STEP 6: CTC loss (same as before)
    encoder_output → CTC Head 1 (768→85) → logits
    → CTC loss vs character-level ground truth

STEP 7: Combined loss
    total_loss = 0.3 × loss_mse + 0.7 × loss_ctc
    total_loss.backward()
    → Gradients flow into: encoder (top 6 layers) + CTC Head 1
```


## Pre-computation (Offline, One-Time)

Two things must be pre-computed before training:

### 1. MuRIL embeddings (NEW)
```
For each clip in train/dev/test:
    text = ground truth transcript
    tokens = muril_tokenizer(text)
    embeddings = muril_model(tokens).last_hidden_state   → (N_tokens, 768)
    Drop [CLS] and [SEP]
    Save: {
        "embeddings": tensor (N_words, 768),
        "tokens": list of token strings,
        "word_count": int
    }
    → Saved to: muril_embeddings/train/{child_id}_{clip_name}.pt
```

### 2. CTC alignment maps (NEW)
```
For each clip in train/dev:
    audio → mel → acoustic_encoder (from joint_best_wer.pt)
    → CTC Head → frame-level character predictions
    → Group consecutive same-characters into segments
    → Map character segments to words (using ground truth text)
    → Save: {
        "word_boundaries": [(start_frame, end_frame, word), ...],
        "num_frames": int
    }
    → Saved to: ctc_alignments/train/{child_id}_{clip_name}.pt
```

Both steps run ONCE. MuRIL is fast (~500 clips/min on CPU).
CTC alignment uses our existing acoustic model.


## What Gets Updated During Training

| Component | Params | Status |
|-----------|--------|--------|
| Whisper encoder layers 0-5 | ~44M | FROZEN |
| Whisper encoder layers 6-11 | ~44M | TRAINABLE |
| CTC Head 1 (768→85) | ~65K | TRAINABLE |
| MuRIL | 110M | FROZEN (teacher, not updated) |
| Acoustic CTC model | ~88M | FROZEN (only for alignment, not updated) |
| **Total trainable** | **~44M** | |


## Key Differences from IndicConformer Approach (02_mse_distillation)

| Aspect | IndicConformer (current) | MuRIL (proposed) |
|--------|------------------------|-------------------|
| Teacher model | Speech model (600M) | Text model (110M) |
| Teacher input | Audio | Ground truth text |
| MSE computed on | Logits (257-dim per frame) | Embeddings (768-dim per word) |
| Alignment method | F.interpolate (frame stretching) | CTC forced alignment (word boundaries) |
| Granularity | Per-frame | Per-word |
| Languages | Hindi + Marathi only | Hindi + Marathi + English (all 3!) |
| What encoder learns | "which BPE token sounds like this" | "what does this word MEAN in context" |
| No. of bridge heads | 2 (ctc_head_hi, ctc_head_mr) | 0 (no bridge needed, both are 768-dim) |
| CTC Head 1 | Trained separately (sequential) | Trained jointly (α×MSE + CTC) |


## Training Config

| Param | Value | Rationale |
|-------|-------|-----------|
| LR | 2e-5 | Standard for encoder fine-tuning |
| Warmup | 500 steps | |
| Epochs | 30 | |
| Patience | 7 | Early stopping on dev WER (from CTC Head 1) |
| α (MSE weight) | 0.3 → 0.1 (decay) | CTC should dominate; MSE is regularizer |
| Frozen layers | Bottom 6 of 12 | Top 6 adapt; bottom 6 preserve acoustics |
| Dropout | 0.1 | Before CTC head |
| Grad clip | 1.0 | |
| Grad accum | 4 | Effective batch size |


## Evaluation

WER computed from CTC Head 1 (our 85-char vocab) — same as all other branches.
This is the FAIR comparison metric across all experiments.

| System | Expected WER | Notes |
|--------|-------------|-------|
| IndicConformer MSE (current) | 47.86% | CTC Head 2 decode, Hi+Mr only |
| IndicConformer sequential CTC | 49.02% | CTC Head 1 decode, Hi+Mr+En |
| MuRIL cross-modal (proposed) | 45-55% | CTC Head 1 decode, Hi+Mr+En |

Standalone WER might be similar or slightly worse — BUT the features will be
MORE COMPLEMENTARY to acoustic branch for gated fusion.


## Checkpoints

Saved to: `checkpoints/semantic_muril/`
Keys: `encoder_state_dict`, `ctc_head_state_dict`
(No separate language heads needed — MuRIL works for all 3 languages)


## Files to Create

```
experiments/semantic_branch/05_muril_cross_modal/
    architecture.md          ← this file
    pseudocode.md            ← training pseudocode
    step0_precompute_muril_embeddings.py   ← one-time: text → MuRIL embeddings
    step1_precompute_ctc_alignments.py     ← one-time: audio → word boundaries
    step2_muril_training.py                ← main training script
    step3_muril_evaluate.py                ← test set WER evaluation
```
