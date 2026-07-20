# Our Architecture — Phase 2 (ASR Decoding with RNN-T)
# Indian Children Speech Recognition (ICASSP 2026)

**Input to Phase 2:** Trained backbone from Phase 1 (frozen)  
**Goal:** Convert children's speech audio → text transcript  
**Decoder:** RNN-T (Recurrent Neural Network Transducer)

---

## Key Difference from CARE Phase 2 — Before We Start

```
CARE Phase 2 output:
  [0.76, 0.05, 0.17, 0.02]
  4 probabilities → pick one → ANGRY
  Fixed output. Always 4 numbers. CANNOT do ASR.

OUR Phase 2 output:
  "राधा के पास एक तोता है"
  Variable length. Different every clip.
  Must output ONE CHARACTER AT A TIME.
  This is why we use RNN-T instead of classifier.

CARE Phase 2 last step:   Mean Pool → Linear(1536→256) → Linear(256→4)
OUR Phase 2 last step:    NO Mean Pool → RNN-T (Encoder + Prediction + Joiner)

WHY NO MEAN POOL:
  CARE needed mean pool to get a fixed 1×1536 vector for its classifier.
  RNN-T works on sequences — it WANTS all 25 frames separately.
  Mean pool would destroy the time information RNN-T needs.
  So we skip it entirely.
```

---

## Running Example

```
Child says:  "राधा के पास एक तोता है"
Audio:       80,000 raw numbers (5 second clip at 16kHz)
Expected output text: "राधा के पास एक तोता है"
```

---

## Phase 2 Full Diagram

```
                        RAW AUDIO
          [0.002, 0.008, -0.003, 0.012, ...]
                   <- 80,000 numbers ->
                           |
                           v
              FROZEN BACKBONE (from Phase 1)
         ┌─────────────────────────────────┐
         │  CNN Feature Extractor  FROZEN  │
         │           |                     │
         │      25 × 768                   │
         │           |                     │
         │  Common Encoder (MMS-300M       │
         │  layers 1-6)  FROZEN            │
         │           |                     │
         │      25 × 768                   │
         │      |         |                │
         │      v         v                │
         │ [Acoustic]  [Semantic]          │
         │  Branch      Branch             │
         │  FROZEN      FROZEN             │
         │      |         |                │
         │  25×768    25×768               │
         └─────────────────────────────────┘
                           |
                           v
              COLLECT ALL 13 LAYER OUTPUTS
         ┌─────────────────────────────────────┐
         │ Layer 0 (CNN)       : 25 × 768      │
         │ Layer 1 (Common L1) : 25 × 768      │
         │ Layer 2 (Common L2) : 25 × 768      │
         │ Layer 3 (Common L3) : 25 × 768      │
         │ Layer 4 (Common L4) : 25 × 768      │
         │ Layer 5 (Common L5) : 25 × 768      │
         │ Layer 6 (Common L6) : 25 × 768      │
         │      ↓ duplicate each (768→1536)    │
         │ Layer 0-6 become    : 25 × 1536     │
         │                                     │
         │ Layer 7  (Branch L1): 25 × 1536     │ ← acoustic 768
         │ Layer 8  (Branch L2): 25 × 1536     │   + semantic 768
         │ Layer 9  (Branch L3): 25 × 1536     │   concatenated
         │ Layer 10 (Branch L4): 25 × 1536     │
         │ Layer 11 (Branch L5): 25 × 1536     │
         │ Layer 12 (Branch L6): 25 × 1536     │
         │                                     │
         │ TOTAL: 13 grids, each 25 × 1536     │
         └─────────────────────────────────────┘
                           |
                           v
              CONVEX COMBINATION (13 learned weights)
         ┌─────────────────────────────────────┐
         │ w0+w1+...+w12 = 1.0  (via softmax) │
         │ These 13 weights TRAIN in Phase 2   │
         │                                     │
         │ Combined = w0×L0 + w1×L1 + ...      │
         │            + w12×L12                │
         └─────────────────────────────────────┘
                           |
                           v
                    ONE grid: 25 × 1536
              (13 grids collapsed into 1 grid)

          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          NO MEAN POOL HERE  (unlike CARE Phase 2)
          We keep all 25 frames for RNN-T encoder
          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           |
                           v
         ┌─────────────────────────────────────────────────────┐
         │                    RNN-T                            │
         │                                                     │
         │  AUDIO SIDE                    TEXT SIDE            │
         │  ──────────────────────        ──────────────────── │
         │  25 frames × 1536              Ground truth chars   │
         │        |                       (training only)      │
         │        v                             |              │
         │  [Linear: 1536→512]           [Embedding lookup]   │
         │        |                       char → 512 numbers  │
         │        v                             |              │
         │  [Transformer attention]        [LSTM]              │
         │  frames see each other          remembers history   │
         │        |                             |              │
         │        v                             v              │
         │  h[1]...h[25]               p[0], p[1], p[2]...    │
         │  25 vectors × 512           grows by 1 per char     │
         │        |                             |              │
         │        └──────────┬─────────────────┘              │
         │                   v                                 │
         │              [JOINER]                               │
         │           h[t] + p[u]                               │
         │           Linear → ~150 scores                      │
         │           Softmax → probabilities                   │
         │                   |                                 │
         │                   v                                 │
         │           character OR blank                        │
         │           IF char  → emit, stay on frame t         │
         │           IF blank → move to frame t+1             │
         └─────────────────────────────────────────────────────┘
                           |
                           v
              'र','ा','ध','ा',' ','क','े',' ','प','ा','स'...
                           |
                           v
              "राधा के पास एक तोता है"
```

---

## Step 1 — Frozen Backbone Pass

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw audio: [0.002, 0.008, -0.003, ...]
           <- 80,000 numbers ->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[CNN Feature Extractor]  FROZEN
        |
        v
[Common Encoder MMS-300M layers 1-6]  FROZEN
        |
        v
[Acoustic Branch + Semantic Branch]  FROZEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Everything frozen. Nothing updates.
Just produces feature outputs from all 13 layers.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 2 — Collect All 13 Layer Outputs

```
WHY ALL 13 AND NOT JUST LAST?
─────────────────────────────
Layer 0-2   → basic sound patterns, raw phonemes
Layer 3-6   → syllable boundaries, rhythm
Layer 7-9   → speaker characteristics, language identity
Layer 10-12 → high level linguistic and acoustic content

Using only last layer = throwing away 12 layers of learned info.
Using all 13 = using everything the backbone learned in Phase 1.

WHAT WE COLLECT:
─────────────────────────────────────────────────────────
Layer 0  (CNN output)       :  25 rows × 768  numbers
Layer 1  (Common layer 1)   :  25 rows × 768  numbers
Layer 2  (Common layer 2)   :  25 rows × 768  numbers
Layer 3  (Common layer 3)   :  25 rows × 768  numbers
Layer 4  (Common layer 4)   :  25 rows × 768  numbers
Layer 5  (Common layer 5)   :  25 rows × 768  numbers
Layer 6  (Common layer 6)   :  25 rows × 768  numbers
         ↓ All 7 are 768-dim. Must become 1536 to match branch layers.

DUPLICATION (for layers 0-6):
Layer 0 original:    [a, b, c, ...]           ← 768 numbers
Layer 0 duplicated:  [a, b, c, ..., a, b, c, ...] ← 1536 numbers
(just copy the vector and attach it to itself)

AT BRANCH LAYERS (layers 7-12):
Acoustic layer output : [0.08, 0.79, ...] ← 768 numbers
Semantic layer output : [0.54, 0.11, ...] ← 768 numbers
Concatenated          : [0.08, 0.79, ..., 0.54, 0.11, ...] ← 1536 numbers
(placed side by side — NOT added)

Layer 7  (Branch layer 1)   :  25 rows × 1536 numbers
Layer 8  (Branch layer 2)   :  25 rows × 1536 numbers
Layer 9  (Branch layer 3)   :  25 rows × 1536 numbers
Layer 10 (Branch layer 4)   :  25 rows × 1536 numbers
Layer 11 (Branch layer 5)   :  25 rows × 1536 numbers
Layer 12 (Branch layer 6)   :  25 rows × 1536 numbers
─────────────────────────────────────────────────────────
TOTAL: 13 grids, each 25 rows × 1536 numbers
```

---

## Step 3 — Convex Combination

```
WE HAVE: 13 grids (each 25 × 1536)
WE WANT: 1 grid   (25 × 1536)

13 learnable raw values: a0, a1, ..., a12
Apply softmax to get weights that always sum to 1:

  w0  = e^a0  / (e^a0 + ... + e^a12)
  w1  = e^a1  / (e^a0 + ... + e^a12)
  ...
  w12 = e^a12 / (e^a0 + ... + e^a12)
  SUM = 1.00  always

Combined = w0×Grid0 + w1×Grid1 + ... + w12×Grid12

HOW THIS WORKS FOR ONE POSITION (row=1, col=0):
  Grid0[1,0]  = 0.51
  Grid1[1,0]  = 0.48
  ...
  Grid12[1,0] = 0.49

  Combined[1,0] = w0×0.51 + w1×0.48 + ... + w12×0.49
               = one single number

This happens for ALL 25×1536 = 38,400 positions.

OUTPUT: ONE grid → 25 rows × 1536 numbers

These 13 weights (a0 to a12) TRAIN during Phase 2.
Backbone stays frozen. Only these 13 values update via backprop.
They learn which layers are most useful for ASR.
```

---

## Step 4 — Feed into RNN-T (NO Mean Pool)

```
INPUT TO RNN-T:
  25 rows × 1536 numbers
  (one row per audio frame, all 25 frames kept)

WHY NOT MEAN POOL HERE:
  CARE did mean pool to get 1×1536 for its fixed-size classifier.
  RNN-T encoder processes frames one by one — it NEEDS the 25 rows.
  Mean pooling would collapse all time information into 1 row.
  That would destroy the sequential structure RNN-T relies on.
```

---

## Step 5 — RNN-T Encoder

```
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 frames × 1536 numbers
Frame 1:  [0.51, 0.38, -0.44, ...] ← 1536 numbers
Frame 2:  [0.47, 0.42, -0.39, ...] ← 1536 numbers
...
Frame 25: [0.53, 0.35, -0.41, ...] ← 1536 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSIDE ENCODER:

SUB-STEP 1: Linear Projection (same idea as FC layer)
  Each frame: 1536 numbers × weight matrix (1536×512) = 512 numbers
  Frame 1: 1536 → 512
  Frame 2: 1536 → 512
  ...
  Frame 25: 1536 → 512
  (512 is our chosen hidden size — design decision)

SUB-STEP 2: Transformer Attention
  All 25 frames look at each other simultaneously.
  Frame 3 can see what happened in frame 1 and frame 2.
  Frame 10 can see all earlier and later frames.
  This is how each encoder state gets context from full audio.

OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
25 encoder states (one per frame)
  h[1]:  [0.71, 0.19, -0.34, ...] ← 512 numbers
  h[2]:  [0.68, 0.23, -0.41, ...] ← 512 numbers
  ...
  h[25]: [0.74, 0.17, -0.38, ...] ← 512 numbers

All 25 vectors exist from this point forward.
They sit and wait while joiner works through them one by one.

TRAINS in Phase 2.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 6 — Prediction Network

```
INPUT: previously predicted character token
       TRAINING  → ground truth from ASER (que_text column)
       TESTING   → whatever joiner predicted in previous step

HOW CHARACTER BECOMES 512 NUMBERS:
  Each character in vocabulary has a learned embedding vector.
  Lookup table:
    'र'  → [0.33, -0.62, 0.29, ...] ← 512 numbers
    'ा'  → [0.67, -0.44, 0.71, ...] ← 512 numbers
    'ध'  → [0.21, -0.18, 0.54, ...] ← 512 numbers
  (these embeddings are learned during Phase 2 training)

INSIDE LSTM:
  LSTM has internal memory — it remembers ALL previous characters.
  Not just the last one. Full history.

STEP BY STEP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 0:
  Input token  = <START>
  Embedding    = [0.01, 0.01, ...] 512 numbers
  LSTM runs
  p[0]         = [0.33, -0.44, ...] 512 numbers
  Meaning: "I haven't said anything yet"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 (after 'र' emitted by joiner):
  Input token  = 'र'
  Embedding    = [0.33, -0.62, ...] 512 numbers
  LSTM runs (remembers step 0)
  p[1]         = [0.41, -0.31, ...] 512 numbers
  Meaning: "I said 'र', next likely 'ा' to form 'रा'"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2 (after 'ा' emitted):
  Input token  = 'ा'
  Embedding    = [0.67, -0.44, ...] 512 numbers
  LSTM runs (remembers step 0+1)
  p[2]         = [0.28, -0.52, ...] 512 numbers
  Meaning: "I said 'रा', next likely 'ध' to form 'राध'"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prediction states grow by 1 every time a character is emitted.
For "राधा के पास एक तोता है" (28 chars):
  p[0] to p[28] → 29 states total by end of sentence.

TRAINS in Phase 2.
```

---

## Step 7 — Joiner

```
At each (t, u) step, joiner takes:
  h[t] = one encoder vector    (512 numbers)
  p[u] = one prediction vector (512 numbers)

INSIDE JOINER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ADD them:
   h[t]: [0.71, 0.19, -0.34, ...]   512 numbers
   p[u]: [0.33, -0.44, 0.28, ...]   512 numbers
   sum:  [1.04, -0.25, -0.06, ...]  512 numbers

2. LINEAR LAYER:
   sum (512) × weight matrix (512 × vocab_size)
   vocab_size = ~150 (all Hindi+Marathi+English chars + blank)
   output: 150 scores
   [2.3, -0.5, 0.8, 1.2, ...]   one score per character

3. SOFTMAX:
   Scores → probabilities (sum = 1.00)
   'र'  : 73%   ← highest
   'ा'  : 8%
   blank: 10%
   others: small %

4. DECISION:
   Pick highest probability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IF output = character:
  → emit that character
  → prediction network takes it → creates p[u+1]
  → stay on SAME audio frame t
  → joiner next runs at (t, u+1)

IF output = blank:
  → emit nothing
  → move to NEXT audio frame t+1
  → stay on same prediction step u
  → joiner next runs at (t+1, u)

TRAINS in Phase 2.
```

---

## Step 8 — Full Joiner Loop for "राधा"

```
(simplified to 3 audio frames)

Encoder ready: h[1], h[2], h[3]  ← all 3 waiting

t=1, u=0: Joiner(h[1] + p[0]) → 'र'
          emit 'र'  |  prediction network: p[1] created
          stay at frame 1, move to u=1

t=1, u=1: Joiner(h[1] + p[1]) → 'ा'
          emit 'ा'  |  prediction network: p[2] created
          stay at frame 1, move to u=2

t=1, u=2: Joiner(h[1] + p[2]) → blank
          nothing emitted  |  move to frame 2, stay at u=2

t=2, u=2: Joiner(h[2] + p[2]) → 'ध'
          emit 'ध'  |  prediction network: p[3] created
          stay at frame 2, move to u=3

t=2, u=3: Joiner(h[2] + p[3]) → blank
          nothing emitted  |  move to frame 3, stay at u=3

t=3, u=3: Joiner(h[3] + p[3]) → 'ा'
          emit 'ा'  |  prediction network: p[4] created
          stay at frame 3, move to u=4

t=3, u=4: Joiner(h[3] + p[4]) → blank
          no more frames → DONE

FINAL OUTPUT: 'र' + 'ा' + 'ध' + 'ा' = "राधा"
```

---

## Step 9 — Loss in Phase 2

```
CARE Phase 2 loss   = Cross-Entropy  (predicted class vs correct class)
OUR Phase 2 loss    = RNN-T Loss

RNN-T loss is more complex than cross-entropy because output
length is variable. But the idea is the same:

  Ground truth:   "राधा के पास एक तोता है"
  Model output:   "राधा के पश एक तोता है"   ← 1 error (पास→पश)

  RNN-T loss = how much total probability did model assign
               to the CORRECT sequence?
             = high loss if wrong sequence gets more probability
             = low loss if correct sequence gets more probability

After 65,311 ASER training clips → loss reduces → model gets
better at predicting correct Hindi/Marathi/English characters.
```

---

## What Trains and What is Frozen in Phase 2

```
FROZEN (from Phase 1, never changes):
  CNN Feature Extractor
  Common Encoder (MMS-300M layers 1-6)
  Acoustic Branch (layers 7-12)
  Semantic Branch (layers 7-12)

TRAINS in Phase 2:
  13 convex combination weights     ← tiny, just 13 numbers
  RNN-T Encoder (linear + transformer)
  Prediction Network (embedding + LSTM)
  Joiner (linear layer)
```

---

## Full Shape Journey — Phase 2

| Step | Operation | Input Shape | Output Shape |
|---|---|---|---|
| Raw audio | Start | 80,000 numbers | 80,000 numbers |
| CNN | Compress waveform | 80,000 | 25 × 768 |
| Common Encoder | Multilingual features | 25 × 768 | 25 × 768 |
| Branches | Acoustic + Semantic concat | 25 × 768 | 25 × 1536 |
| Duplication | Match CNN/Common layers | 7 × 25×768 | 7 × 25×1536 |
| Collect | All 13 layers | — | 13 grids of 25×1536 |
| Convex Combo | 13 grids → 1 grid | 13 × 25×1536 | 25 × 1536 |
| RNN-T Encoder | Linear + Transformer | 25 × 1536 | 25 × 512 (h states) |
| Prediction Net | Embedding + LSTM | 1 char token | 512 (p state) |
| Joiner | h[t] + p[u] | 512 + 512 | ~150 probabilities |
| Decision | Pick highest | ~150 probs | 1 character OR blank |
| **Final** | **All chars collected** | **sequence** | **"राधा के पास एक तोता है"** |

---

## CARE Phase 2 vs Our Phase 2 — Key Differences

| | CARE Phase 2 | Our Phase 2 |
|---|---|---|
| Task | Emotion classification | ASR (speech to text) |
| Mean Pool | YES (needed for classifier) | NO (RNN-T needs all frames) |
| Output head | Linear → 4 classes | RNN-T (Encoder + Prediction + Joiner) |
| Output type | 1 emotion label | Full text transcript |
| Output length | Always 1 | Variable (depends on clip) |
| Loss | Cross-Entropy | RNN-T Loss |
| Convex Combination | YES (same idea) | YES (same idea, 13 weights) |
| Backbone frozen | YES | YES |

---

## Phase 2 Conclusion

> Phase 2 plugs the RNN-T decoder on top of our frozen Phase 1 backbone.
> The backbone already understands Indian children's speech deeply.
> RNN-T's three parts (Encoder, Prediction Network, Joiner) learn
> how to READ those representations and convert them to text.
>
> Encoder    = processes audio frames from our backbone
> Prediction = remembers previously predicted characters (language model)
> Joiner     = combines both and decides next character or blank
>
> This combination handles Hindi, Marathi, Indian English,
> and code-switching naturally — which a simple CTC decoder cannot.
