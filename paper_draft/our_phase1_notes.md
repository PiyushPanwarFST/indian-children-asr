# Our Architecture — Phase 1 Pre-Training
# Indian Children Speech Recognition (ICASSP 2026)

**Based on:** CARE framework (Dutta & Ganapathy, IEEE TASLP 2025)  
**Our adaptation for:** Hindi + Marathi + Indian English children's speech  
**Dataset:** ASER (81,423 clips, 123.72 hours, 5,260 children)

---

## Key Difference from CARE Phase 1 — Before We Start

```
CARE Phase 1                          OUR Phase 1
──────────────────────────────────────────────────────────────
Semantic Teacher:                     Semantic Teacher:
  Whisper  →  RoBERTa                   IndicWav2Vec
  audio→text→meaning                    audio→Indian lang meaning
  TWO model steps                       ONE model step (simpler)

Acoustic Teacher:                     Acoustic Teacher:
  PASE+  (256-dim output)               Kid-Whisper  (768-dim output)
  adult speech features                 children's speech features

Common Encoder:                       Common Encoder:
  WavLM  (English only)                 MMS-300M  (1000+ languages)
  12 layers, 768-dim                    12 layers, 768-dim

FC Layer in Acoustic Branch:          FC Layer:
  NEEDED  (768→256 to match PASE+)      NOT NEEDED
                                        (Kid-Whisper is 768-dim,
                                         same as branch output)
```

**Why no FC layer needed in our design:**
In CARE, PASE+ gave 256-dim acoustic features, but the acoustic branch
transformer outputs 768-dim. FC was added to shrink 768→256 so MSE loss
could compare them. Kid-Whisper encoder outputs 768-dim, same as our
acoustic branch. No size mismatch. No FC needed. Simpler.

---

## Running Example

```
Child saying: "राधा के पास एक तोता है"
(Radha has a parrot — ASER Hindi reading clip)
5 second audio clip at 16kHz sampling rate
= 80,000 raw numbers
```

---

## Phase 1 Diagram — Full Overview

```
                         RAW AUDIO
           [0.002, 0.008, -0.003, 0.012, ...]
                    <- 80,000 numbers ->
                            |
            |---------------|---------------|
            |               |               |
            v               v               v
        PATH A          PATH C           PATH B
     (Semantic          (OUR             (Acoustic
      Teacher)          MODEL)           Teacher)
        |               |                    |
        v               v                    v
  [IndicWav2Vec]  [CNN Feature]        [Kid-Whisper]
    frozen         Extractor              frozen
                   TRAINS                    |
        |               |                    v
        v          [Common Encoder       50 rows
    25 x 768        MMS-300M              x 768
    (frames)        layers 1-6]               |
        |           TRAINS             [Downsample
        v               |                  by 2]
   [Avg Pool]      25 x 768                 |
        |               |                   v
        v           |-------|            Y_kid
    Y_indic         v       v           25 x 768
    1 x 768    [Acoustic] [Semantic]    CORRECT
    CORRECT     Branch    Branch        ACOUSTIC
    SEMANTIC  MMS-300M  IndicWav2Vec     ANSWER
     ANSWER   layers    layers
              7-12      7-12
              TRAINS    TRAINS
                |           |
                v           v
           25 x 768     25 x 768
                |           |
                v           v
          [No FC here]  [Avg Pool]
          (768 already   25 rows
           matches        → 1 row
           Y_kid)             |
                |             v
                v         Y_sem_hat
           Y_acoust_hat    1 x 768
           25 x 768
                |             |
                v             v
         Compare with    Compare with
           Y_kid            Y_indic
                |             |
                v             v
           L_acoust        L_sem
           (MSE)           (MSE)
                |             |
                v             v
          L_total = L_sem + L_acoust
          Update PATH C only.
          Teachers A and B stay frozen.
```

---

## PATH A — Producing the Correct Semantic Answer (Y_indic)

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw audio numbers:
[0.002, 0.008, -0.003, 0.012, ...]
<- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[IndicWav2Vec — FROZEN]
Trained by AI4Bharat on 40 Indian languages.
Takes raw audio directly. No transcription step needed.
Internally: CNN + 12 transformer layers.
        |
OUTPUT (frame level)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 rows x 768 numbers  (simplified; one row per ~40ms)

Row 1:  [0.21, -0.43, 0.87, ...] <- 768 numbers
Row 2:  [0.33, -0.21, 0.54, ...] <- 768 numbers
...
Row 25: [0.67, -0.89, 0.23, ...] <- 768 numbers

Each row = what the model understood at that time point
in the context of Indian languages (Hindi/Marathi/English)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Average Pool — collapse 25 rows into 1 row]
        |
OUTPUT = Y_indic  ✓  THE CORRECT SEMANTIC ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE row: [0.45, -0.62, 0.58, ...]
         <------- 768 numbers ------->

This is Y_indic.
Shape: 1 x 768
Meaning: mathematical summary of the LINGUISTIC CONTENT
         of this speech clip in Indian language context.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY IndicWav2Vec instead of Whisper → RoBERTa (like CARE)?
──────────────────────────────────────────────────────────
CARE used Whisper to transcribe audio to text first,
then RoBERTa to read that text and produce meaning vectors.
Two models. Two steps. Designed for English.

IndicWav2Vec takes AUDIO directly and was trained on 40 Indian
languages including Hindi and Marathi. One step. No transcription
error can propagate. Much better for our task.
```

---

## PATH B — Producing the Correct Acoustic Answer (Y_kid)

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Same raw audio numbers:
[0.002, 0.008, -0.003, 0.012, ...]
<- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Kid-Whisper Encoder — FROZEN]
Whisper fine-tuned on children's speech data.
Internally: CNN on mel spectrogram + transformer encoder.
Captures: children's higher pitch, irregular rhythm,
          mispronunciations, shorter vocal tracts.
        |
OUTPUT (before downsampling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
50 rows x 768 numbers  (5 sec clip, 1 row per 20ms)

Row 1  (0-20ms):   [0.12, 0.87, -0.34, ...] <- 768 numbers
Row 2  (20-40ms):  [0.45, 0.23, -0.67, ...] <- 768 numbers
...
Row 50 (end):      [0.09, 0.56, -0.12, ...] <- 768 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Downsample by 2 — keep every other row]
        |
OUTPUT = Y_kid  ✓  THE CORRECT ACOUSTIC ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 rows x 768 numbers

Row 1:  [0.12, 0.87, -0.34, ...] <- 768 numbers
Row 3:  [0.44, 0.21, -0.65, ...] <- 768 numbers
...
Row 25: [0.08, 0.53, -0.11, ...] <- 768 numbers

This is Y_kid.
Shape: 25 x 768
Meaning: how this child sounds at each time point
         (pitch, energy, rhythm — in a children's voice model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY Kid-Whisper instead of PASE+ (like CARE)?
─────────────────────────────────────────────
PASE+ was trained on adult speech (Librispeech, VoxCeleb).
It never heard a child speak. It gives acoustic features
calibrated to adult vocal characteristics.

Children have: higher fundamental frequency (pitch),
               faster decay, different formant positions,
               more pronunciation variability.

Kid-Whisper was fine-tuned specifically on children's speech.
Its encoder representations are naturally tuned to capture
these child-specific acoustic properties.
```

---

## PATH C — Our Model Producing Attempts (Y_sem_hat and Y_acoust_hat)

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Same raw audio numbers (third time):
[0.002, 0.008, -0.003, 0.012, ...]
<- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[CNN Feature Extractor — part of MMS-300M]
TRAINS. Compresses raw waveform into frames.
Stride of 320 samples at 16kHz = one frame every 20ms.
        |
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 rows x 768 numbers  (one row every 40ms, simplified)

Row 1:  [0.33, 0.71, -0.22, ...] <- 768 numbers
...
Row 25: [0.28, 0.59, -0.18, ...] <- 768 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Common Encoder — MMS-300M layers 1 to 6]
TRAINS. Learns multilingual phonemes and basic patterns.
Trained across Hindi, Marathi, Indian English simultaneously.
Captures: phoneme boundaries, syllable structure,
          multilingual sound patterns.
        |
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Still 25 rows x 768 numbers
BUT now each row is richer — contains multilingual
phoneme-level understanding of the audio.

Row 1:  [0.51, 0.38, -0.44, ...] <- 768 numbers
...
Row 25: [0.47, 0.42, -0.39, ...] <- 768 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        |--------------------------|
        |                          |
        v                          v
[ACOUSTIC BRANCH]          [SEMANTIC BRANCH]
MMS-300M layers 7-12       IndicWav2Vec layers 7-12
TRAINS                     TRAINS
        |                          |
        v                          v
   25 x 768                   25 x 768
(transformer outputs          (transformer outputs
 768 naturally)                768 naturally)
        |                          |
        v                          v
  [NO FC LAYER]              [Avg Pool]
  (768 already matches        25 rows → 1 row
   Y_kid's 768-dim)           (collapse time into
        |                      single vector)
        v                          |
   Y_acoust_hat                    v
   25 x 768                   Y_sem_hat
                               1 x 768

Y_acoust_hat example:              Y_sem_hat example:
Row 1:  [0.08, 0.79, -0.41, ...]   [0.39, -0.71, 0.49, ...]
Row 2:  [0.32, 0.19, -0.71, ...]    <- 768 numbers ->
...
Row 25: [0.03, 0.49, -0.18, ...]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY MMS-300M as Common Encoder instead of WavLM?
─────────────────────────────────────────────────
WavLM was pretrained on English speech only (LibriLight).
MMS-300M was pretrained on 1000+ languages by Meta,
explicitly covering Hindi, Marathi, and Indian English.
Same architecture (wav2vec2 base: 12 layers, 768-dim).
Plug-in replacement. Better multilingual representations.

WHY IndicWav2Vec layers for Semantic Branch init?
──────────────────────────────────────────────────
CARE initialized semantic branch from RoBERTa (text model),
needed conv adapters to bridge speech-to-text domain gap.
We initialize from IndicWav2Vec's upper layers (speech model
trained on Indian languages). Same domain as our backbone.
No conv adapters needed. Cleaner design.
```

---

## Final Summary — All 4 Outputs of Our Phase 1

| Name | What It Is | Shape | Produced By | Role |
|---|---|---|---|---|
| Y_indic | Indian language content summary | 1 x 768 | IndicWav2Vec (frozen teacher) | Correct semantic answer |
| Y_kid | Children's voice quality over time | 25 x 768 | Kid-Whisper (frozen teacher) | Correct acoustic answer |
| Y_sem_hat | Model's attempt at content | 1 x 768 | Semantic branch (trains) | Model's semantic output |
| Y_acoust_hat | Model's attempt at voice quality | 25 x 768 | Acoustic branch (trains) | Model's acoustic output |

---

## Loss Calculation

```
SEMANTIC LOSS (L_sem):
Correct answer  Y_indic    : [0.45, -0.62, 0.58, ...]  <- 768 numbers
Model attempt   Y_sem_hat  : [0.39, -0.71, 0.49, ...]  <- 768 numbers

MSE = average of (each number difference)²
    = (0.45-0.39)² + (-0.62-(-0.71))² + (0.58-0.49)² + ...
    = one single number  e.g. 0.041

ACOUSTIC LOSS (L_acoust):
Correct answer  Y_kid       : 25 rows x 768 numbers
Model attempt   Y_acoust_hat: 25 rows x 768 numbers

MSE = average difference² across all 25 rows and 768 columns
    = one single number  e.g. 0.068

TOTAL LOSS:
L_total = 0.041 + 0.068 = 0.109

Only PATH C (our model) updates. Teachers stay frozen.
After thousands of ASER clips → L_total near zero → Phase 1 done.
```

---

## What is Different from CARE Phase 1 — Summary Table

| Component | CARE Phase 1 | Our Phase 1 | Why We Changed |
|---|---|---|---|
| Semantic Teacher | Whisper → RoBERTa | IndicWav2Vec | One step, Indian languages, no transcription error |
| Acoustic Teacher | PASE+ (256-dim) | Kid-Whisper (768-dim) | Children-specific acoustic model |
| Common Encoder | WavLM (English) | MMS-300M (1000+ langs) | Hindi + Marathi + English coverage |
| Semantic Branch Init | RoBERTa layers + conv adapters | IndicWav2Vec upper layers | Same domain, no adapter needed |
| Acoustic Branch FC | YES (768→256) | NO (768=768 already) | Kid-Whisper and branch both 768-dim |
| Y_sem shape | 1 x 768 | 1 x 768 | Same |
| Y_acoust shape | 25 x 256 | 25 x 768 | Matches our teacher dimension |
| Loss | MSE | MSE | Same |
| Labels needed | NO | NO | Same (self-supervised) |

---

## What Phase 1 Produces — In Simple Words

> After Phase 1, our model is a trained backbone that takes raw audio of
> an Indian child speaking and produces TWO things simultaneously:
>
> - A 768-number vector summarizing the LINGUISTIC CONTENT in Indian language context
> - A 25×768 grid of numbers describing HOW that child sounds over time
>
> IndicWav2Vec and Kid-Whisper teachers are no longer needed.
> Our model learned to do their jobs — from raw audio alone.
> This backbone is now ready for Phase 2 (ASR transcription with RNN-T).
