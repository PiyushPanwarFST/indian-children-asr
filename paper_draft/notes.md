# Indian Children Speech Recognition — My Understanding Notes

**Base Paper:** CARE — Dutta & Ganapathy, IEEE TASLP 2025  
**Target:** ICASSP 2026

---

## Phase 1 — Pre-Training

**Paper location:** Section III-B, pages 3680–3681

**Running example:** Audio clip — child saying "राधा के पास एक तोता है"  
(Radha has a parrot — Hindi paragraph clip from ASER)

---

### Phase 1 Diagram

```
                        RAW AUDIO
          [0.002, 0.008, -0.003, 0.012, ...]
                   <- 80,000 numbers ->
                           |
           |---------------|---------------|
           |               |               |
           v               v               v
       PATH A          PATH C           PATH B
    (Semantic          (CARE            (Acoustic
     Teacher)          Model)           Teacher)
       |               |                   |
       v               v                   v
   [WHISPER]      [CNN Feature]         [PASE+]
   frozen         Extractor             frozen
       |           TRAINS                   |
       v               |                    v
   [RoBERTa]      [Common                 50 rows
   frozen          Encoder                 x 256
       |           WavLM 1-6]                |
       v           TRAINS              [Downsample
   12 rows x 768       |                  by 2]
       |           25 rows x 768           |
       v               |                   v
   [Avg Pool]      |-------|             Y_pase
       |           v       v           25 x 256
       v      [Acoustic] [Semantic]     CORRECT
   Y_text      Branch    Branch        ACOUSTIC
   768-dim    WavLM 7-12  RoBERTa       ANSWER
   CORRECT    TRAINS    +Conv Ada
   SEMANTIC       |     TRAINS
    ANSWER        |         |
                  v         v
              25 x 768   25 x 768
                  |         |
                  v         v
              [FC Layer] [Avg Pool]
              768 -> 256  25 rows
              (match      -> 1 row
               PASE+ dim) (match
                  |        Y_text)
                  v         v
              Y_acoust  Y_sem
              _hat      _hat
              25 x 256  768-dim
                |         |
                v         v
         Compare with  Compare with
           Y_pase        Y_text
                |         |
                v         v
           L_acoust    L_sem
           (MSE)       (MSE)
                |         |
                v         v
         L_total = L_sem + L_acoust
         Update PATH C only.
         Teachers A and B stay frozen.
```

---

### What the Raw Audio Actually Looks Like

Before anything — understand what raw audio IS as data:

```
Raw Audio File (.mp3 / .wav)
= just a list of numbers representing air pressure over time

[0.002, 0.008, -0.003, 0.012, -0.007, 0.019, ...]
 <-------------- 16,000 numbers per second ------------->

A 5 second clip = 80,000 numbers in a list
Nothing else. Just numbers.

This raw list of 80,000 numbers goes into Phase 1.
```

---

### PATH A — Producing the Correct Semantic Answer (Y_text)

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw audio numbers:
[0.002, 0.008, -0.003, 0.012, ...]
<- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[WHISPER — converts audio to text]
        |
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plain text string:
"राधा के पास एक तोता है"
<- just a sentence, like a WhatsApp message ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[RoBERTa — reads text, converts to meaning numbers]
        |
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12 rows of vectors (one per RoBERTa layer),
each row = 768 numbers

Layer 1:  [0.21, -0.43, 0.87, 0.12, ...]  <- 768 numbers
Layer 2:  [0.33, -0.21, 0.54, 0.09, ...]  <- 768 numbers
...
Layer 12: [0.67, -0.89, 0.23, 0.45, ...]  <- 768 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Average Pool — collapse 12 rows into 1 row]
        |
OUTPUT = Y_text  ✓  THE CORRECT SEMANTIC ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE row: [0.45, -0.62, 0.58, 0.24, ...]
         <------ 768 numbers ------>

This is Y_text.
Shape: 768 numbers in a single row.
Meaning: mathematical summary of what this sentence MEANS.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### PATH B — Producing the Correct Acoustic Answer (Y_pase)

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Same raw audio numbers:
[0.002, 0.008, -0.003, 0.012, ...]
<- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[PASE+ — analyzes voice quality every 20ms]
        |
OUTPUT (before downsampling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
50 rows x 256 numbers  (5 sec clip, 1 row per 20ms)

Row 1  (0-20ms):   [0.12, 0.87, -0.34, ...] <- 256 numbers
Row 2  (20-40ms):  [0.45, 0.23, -0.67, ...] <- 256 numbers
...
Row 50 (end):      [0.09, 0.56, -0.12, ...] <- 256 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Downsample by 2 — keep every other row]
        |
OUTPUT = Y_pase  ✓  THE CORRECT ACOUSTIC ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 rows x 256 numbers

Row 1:  [0.12, 0.87, -0.34, ...] <- 256 numbers
Row 3:  [0.44, 0.21, -0.65, ...] <- 256 numbers
...
Row 25: [0.08, 0.53, -0.11, ...] <- 256 numbers

This is Y_pase.
Shape: 25 rows, each row has 256 numbers.
Meaning: voice quality measurements at 25 time points.
         (pitch, energy, speed at each moment in time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### PATH C — CARE Model Producing Its Attempts (Y_sem_hat and Y_acoust_hat)

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Same raw audio numbers (third time):
[0.002, 0.008, -0.003, 0.012, ...]
<- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[CNN Feature Extractor — compress waveform to frames]
        |
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 rows x 768 numbers  (one row every 40ms)

Row 1:  [0.33, 0.71, -0.22, ...] <- 768 numbers
Row 2:  [0.41, 0.68, -0.31, ...] <- 768 numbers
...
Row 25: [0.28, 0.59, -0.18, ...] <- 768 numbers

Shape: 25 rows x 768 numbers
Meaning: compressed audio, one chunk per 40ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[Common Encoder — WavLM layers 1 to 6]
Learns basic patterns: phonemes, pitch boundaries
        |
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Still 25 rows x 768 numbers
BUT now each row is richer — contains phoneme-level
and basic acoustic understanding

Row 1:  [0.51, 0.38, -0.44, ...] <- 768 numbers
...
Row 25: [0.47, 0.42, -0.39, ...] <- 768 numbers

Shape: 25 rows x 768 numbers  (same shape, richer content)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        |--------------------------------|
        |                                |
        v                                v
[ACOUSTIC BRANCH]                [SEMANTIC BRANCH]
WavLM layers 7-12                RoBERTa layers + conv adapters
        |                                |
        v                                v
   25 x 768                         25 x 768
(transformer always                (transformer always
 outputs 768)                       outputs 768)
        |                                |
        v                                v
  [FC LAYER]                      [Average Pool]
  768 -> 256                      25 rows -> 1 row
  (resize to match                (collapse time into
   PASE+ teacher)                  single vector)
        |                                |
OUTPUT                           OUTPUT
================================ ================================
25 rows x 256 numbers            ONE row x 768 numbers

Y_acoust_hat:                    Y_sem_hat:
Row 1:  [0.08, 0.79, -0.41, ...]  [0.39, -0.71, 0.49, ...]
Row 2:  [0.32, 0.19, -0.71, ...]   <- 768 numbers ->
...
Row 25: [0.03, 0.49, -0.18, ...]
================================ ================================
```

---

### What is the FC Layer?

FC = **Fully Connected Layer** (also called a Linear Layer)

It is just a matrix multiplication that resizes a vector from one size to another.

```
ACOUSTIC BRANCH TRANSFORMER OUTPUTS:  25 rows x 768 numbers
PASE+ TEACHER OUTPUTS:                 25 rows x 256 numbers

Problem: Cannot compare 768-dim with 256-dim. MSE needs same size.
Solution: Add FC layer to shrink 768 -> 256.

HOW FC WORKS (for one row):

Input:  [a, b, c, d, ...]  <- 768 numbers

FC has a weight matrix: 768 rows x 256 columns (learnable numbers)
Matrix multiplication:

  output[0] = a*w00 + b*w10 + c*w20 + ... (weighted sum of all 768 inputs)
  output[1] = a*w01 + b*w11 + c*w21 + ...
  ...
  output[255] = a*w0,255 + b*w1,255 + ...

Output: [x, y, z, ...]  <- 256 numbers

The FC layer learns which combination of the 768 acoustic features
best matches the 256 PASE+ numbers. This is what gets trained.
```

**Why only acoustic branch has FC, not semantic?**

```
Semantic Branch target = Y_text  = 768 numbers  (from RoBERTa)
Acoustic Branch target = Y_pase  = 256 numbers  (from PASE+)

Semantic branch already outputs 768 -> same size as Y_text -> NO FC needed
Acoustic branch outputs  768 -> DIFFERENT from Y_pase 256 -> FC needed
```

**Why FC is skipped in Phase 2:**

```
In Phase 2 there is no teacher comparison happening.
FC only existed to match PASE+ (256-dim).
Without that constraint, acoustic branch just outputs its natural 768-dim.
So FC is skipped -> acoustic branch gives 768 -> same as semantic 768
-> concatenation gives 768+768 = 1536.
```

---

### Final Summary — All 4 Outputs of Phase 1

| Name | What It Is | Shape | Produced By | Role |
|---|---|---|---|---|
| Y_text | Meaning of sentence in numbers | 768 numbers, 1 row | RoBERTa (frozen teacher) | Correct semantic answer |
| Y_pase | Voice quality over time | 25 rows x 256 numbers | PASE+ (frozen teacher) | Correct acoustic answer |
| Y_sem_hat | CARE's attempt at meaning | 768 numbers, 1 row | Semantic branch (trains) | Model's semantic output |
| Y_acoust_hat | CARE's attempt at voice quality | 25 rows x 256 numbers | Acoustic branch (trains) | Model's acoustic output |

---

### Loss Calculation — Comparing the Pairs

```
SEMANTIC LOSS (L_sem):
Correct answer  Y_text     : [0.45, -0.62, 0.58, ...]  <- 768 numbers
Model attempt   Y_sem_hat  : [0.39, -0.71, 0.49, ...]  <- 768 numbers

MSE = average of (each number difference)²
    = (0.45-0.39)² + (-0.62-(-0.71))² + (0.58-0.49)² + ...
    = one single number  e.g. 0.043

ACOUSTIC LOSS (L_acoust):
Correct answer  Y_pase      : 25 rows x 256 numbers
Model attempt   Y_acoust_hat: 25 rows x 256 numbers

MSE = average difference² across all 25 rows and 256 columns
    = one single number  e.g. 0.071

TOTAL LOSS:
L_total = 0.043 + 0.071 = 0.114

Both losses are just one single number each. Not vectors. Just a score saying how wrong the model is.

Model updates itself to make this number smaller.
After thousands of clips -> L_total near zero -> Phase 1 done -> model saved.
```

---

### What Phase 1 Produced — In Simple Words

> After Phase 1, CARE is a trained model that can take raw audio and produce TWO things simultaneously:
> - A 768-number vector summarizing the MEANING of what was said
> - A 25x256 grid of numbers describing HOW it was said over time
>
> The teachers (RoBERTa and PASE+) are no longer needed. CARE learned to do their jobs itself — from audio alone.

---

## Phase 2 — Fine-Tuning (Downstream Task)

*To be added*

---

## Our Proposed Architecture — Indian Children ASR

*To be added*

---

## Planning and Next Steps

*To be added*
