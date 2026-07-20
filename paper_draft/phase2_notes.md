# Phase 2 — Fine-Tuning (Downstream Task)

**Paper location:** Section III-B-3 "Inference" (page 3681) + Section III-D-2 (page 3682)

---

## What Changed from Phase 1?

```
Phase 1  =  TEACHING the model  (backbone learns from teachers)
Phase 2  =  USING the model     (backbone frozen, predict emotions)

Teachers (RoBERTa, PASE+)      ->  NO LONGER NEEDED
CARE backbone from Phase 1     ->  LOADED, MOSTLY FROZEN
New things that train          ->  only 13 small weights + tiny classifier
```

**Same running example:** Child saying "राधा के पास एक तोता है"  
**New goal:** Predict emotion → Angry / Happy / Sad / Neutral

---

## Phase 2 Diagram

```
                         RAW AUDIO
               [0.002, 0.008, -0.003, ...]
                      <- 80,000 numbers ->
                              |
                              v
                   [CNN Feature Extractor]
                         FROZEN
                              |
                         25 x 768
                              |
                              v
                   [Common Encoder - 6 layers]
                         FROZEN
                              |
                         25 x 768
                              |
                 |----------------------------|
                 v                            v
       [Acoustic Branch]            [Semantic Branch]
           FROZEN                      FROZEN
              |                            |
        25 x 768                      25 x 768
      (FC layer skipped              (avg pool skipped
       in Phase 2)                    in Phase 2)
              |                            |
              |____________________________|
                             |
                    CONCATENATE at each layer
                    (place side by side)
                    768 + 768 = 1536
                             |
                      25 x 1536  (per branch layer)

NOTE: In Phase 1, acoustic branch used FC to reduce 768->256
      to match PASE+ teacher (256-dim). In Phase 2 that FC
      layer is SKIPPED. Transformer layers always output 768.
      Paper ref: Figure 2 caption, page 3681.
                             |
                             v
              COLLECT ALL 13 LAYER OUTPUTS
              ================================
              7 common layers  x  25 x 768
              (duplicated)     -> 25 x 1536 each
              +
              6 branch layers  x  25 x 1536
              ================================
              Total: 13 grids, each 25 x 1536
                             |
                             v
              CONVEX COMBINATION (learned weights)
              ================================
              w0 + w1 + ... + w12 = 1
              Combined = w0*L0 + w1*L1 + ... + w12*L12
              ================================
              Output: ONE grid  ->  25 x 1536
                             |
                             v
                 MEAN POOL (average 25 rows -> 1)
              ================================
              Row 1:   [0.51, 0.38, ...]
              Row 2:   [0.47, 0.42, ...]   -> average
              ...                              each column
              Row 25:  [0.53, 0.35, ...]
              ================================
              Output: ONE vector  ->  1 x 1536
                             |
                             v
              CLASSIFICATION HEAD
              ================================
              [Linear 1536 -> 256]  TRAINS
                        |
                     [ReLU]
                        |
              [Linear  256 -> 4]    TRAINS
                        |
                   [Softmax]
              ================================
              Output: 4 probabilities
              Angry: 76%  Happy: 5%  Sad: 17%  Neutral: 2%
                             |
                             v
                  PREDICTED EMOTION = ANGRY
```

---

## Step 1 — Pass Audio Through Frozen CARE Backbone

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
[Common Encoder — 6 layers]  FROZEN
        |
        v
[Acoustic Branch + Semantic Branch]  FROZEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Everything here is FROZEN from Phase 1.
Nothing updates. Just produces outputs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 2 — Collect ALL 13 Layer Outputs

**Why all 13 and not just the last one?**

```
Layer 1-3  ->  basic phoneme patterns
Layer 4-6  ->  rhythm, speaking rate
Layer 7-9  ->  speaker characteristics
Layer 10-12 -> high-level emotion content

Using ONLY last layer = throwing away 12 layers of information.
Using ALL 13 layers  = using everything the model learned.
```

**What we collect:**

```
Layer 0  (CNN output)      :  25 rows x 768  numbers
Layer 1  (Common layer 1)  :  25 rows x 768  numbers
Layer 2  (Common layer 2)  :  25 rows x 768  numbers
Layer 3  (Common layer 3)  :  25 rows x 768  numbers
Layer 4  (Common layer 4)  :  25 rows x 768  numbers
Layer 5  (Common layer 5)  :  25 rows x 768  numbers
Layer 6  (Common layer 6)  :  25 rows x 768  numbers
                                <- 7 outputs (all 25 x 768)

At each branch layer, semantic + acoustic are joined side by side:

  Semantic layer 1: [0.21, 0.43, ...]  <- 768 numbers
  Acoustic layer 1: [0.54, 0.11, ...]  <- 768 numbers
  Concatenated:     [0.21, 0.43, ..., 0.54, 0.11, ...]
                    <-------- 1536 numbers -------->
  (NOT addition — placed side by side, all info preserved)

Branch layer 1  :  25 rows x 1536 numbers
Branch layer 2  :  25 rows x 1536 numbers
Branch layer 3  :  25 rows x 1536 numbers
Branch layer 4  :  25 rows x 1536 numbers
Branch layer 5  :  25 rows x 1536 numbers
Branch layer 6  :  25 rows x 1536 numbers
                                <- 6 outputs (all 25 x 1536)

PROBLEM: 7 layers are 768-dim, 6 layers are 1536-dim. Cannot mix.
SOLUTION (from paper Section III-D-2): Duplicate the 768-dim layers.

Layer 0 original:    [a, b, c, ...]          <- 768 numbers
Layer 0 duplicated:  [a, b, c, ..., a, b, c, ...]  <- 1536 numbers

Now ALL 13 layers are 25 x 1536. Same space. Can combine.

TOTAL: 13 grids, each 25 rows x 1536 numbers
```

---

## Step 3 — Convex Combination

**Source:** SUPERB benchmark paper (Yang et al., Interspeech 2021)  
**Paper reference in CARE:** Section III-B-3, page 3681

```
We have 13 grids. We need 1 grid.
Each grid gets a learned weight.

13 raw learnable parameters: a0, a1, a2, ... a12
Apply Softmax to get valid weights (all positive, sum = 1):

  w0  = e^a0  / (e^a0 + e^a1 + ... + e^a12)
  w1  = e^a1  / (e^a0 + e^a1 + ... + e^a12)
  ...
  w12 = e^a12 / (e^a0 + e^a1 + ... + e^a12)

  w0 + w1 + ... + w12 = 1.00  always

Combined = w0 x Layer0 + w1 x Layer1 + ... + w12 x Layer12

Example learned weights after Phase 2 training:
  w0  (CNN)       = 0.02  <- basic, less useful for emotion
  w1  (Common 1)  = 0.03
  w2  (Common 2)  = 0.04
  w3  (Common 3)  = 0.05
  w4  (Common 4)  = 0.07
  w5  (Common 5)  = 0.08
  w6  (Common 6)  = 0.09
  w7  (Branch 1)  = 0.10
  w8  (Branch 2)  = 0.12
  w9  (Branch 3)  = 0.14  <- highest -> most useful for emotion
  w10 (Branch 4)  = 0.11
  w11 (Branch 5)  = 0.09
  w12 (Branch 6)  = 0.06
  SUM             = 1.00

These 13 weights (a0 to a12) are LEARNABLE.
They update via backpropagation during Phase 2 training.

OUTPUT: ONE grid  ->  25 rows x 1536 numbers
(13 grids collapsed into 1 grid of same size)
```

---

## Step 4 — Mean Pool Across Time

```
INPUT: 25 rows x 1536 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Row 1  : [0.51, 0.38, -0.44, ...]  <- 1536 numbers
Row 2  : [0.47, 0.42, -0.39, ...]  <- 1536 numbers
Row 3  : [0.53, 0.31, -0.41, ...]  <- 1536 numbers
...
Row 25 : [0.49, 0.40, -0.42, ...]  <- 1536 numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average each column across all 25 rows:

  Position 0  : (0.51 + 0.47 + 0.53 + ... + 0.49) / 25 = 0.50
  Position 1  : (0.38 + 0.42 + 0.31 + ... + 0.40) / 25 = 0.38
  Position 2  : (-0.44 + -0.39 + -0.41 + ... + -0.42) / 25 = -0.41
  ...
  Position 1535: average of 25 values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT: ONE row  ->  1 x 1536 numbers
[0.50, 0.38, -0.41, ...]
<-------- 1536 numbers -------->

This ONE vector represents the ENTIRE audio clip.
25 time-step vectors collapsed into 1 summary vector.

WHY: Classifier needs fixed-size input.
     3-sec clip has fewer rows than 10-sec clip.
     Mean pool gives standard 1536-dim for any audio length.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 5 — Classification Head

**Paper reference:** Section III-D-2, last paragraph (page 3682)

```
INPUT:  1 vector x 1536 numbers
[0.50, 0.38, -0.41, 0.22, ...]
<-------- 1536 numbers -------->
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[LINEAR LAYER 1  :  1536 -> 256]   TRAINS in Phase 2
Weight matrix: 1536 x 256 numbers (learnable)
Each of 256 outputs = weighted sum of all 1536 inputs

OUTPUT: 1 vector x 256 numbers
[0.23, -0.14, 0.67, -0.89, ...]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[ReLU ACTIVATION]
Rule: if number < 0, make it 0. If >= 0, keep it.

INPUT : [0.23, -0.14,  0.67, -0.89,  0.11]
OUTPUT: [0.23,  0.00,  0.67,  0.00,  0.11]

Purpose: lets model learn complex non-linear patterns.
Without ReLU, two linear layers = one linear layer (useless).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[LINEAR LAYER 2  :  256 -> 4]   TRAINS in Phase 2
Weight matrix: 256 x 4 numbers (learnable)

OUTPUT: 1 vector x 4 numbers
[2.3, -0.5, 0.8, -1.2]
 Angry Happy  Sad  Neutral

These 4 numbers = RAW SCORES for each emotion.
Higher score = model more confident about that emotion.
NOT probabilities yet (can be negative, do not sum to 1).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
[SOFTMAX]
Converts 4 raw scores into 4 probabilities.

  e^2.3  = 9.97   |   Angry   = 9.97  / 13.11 = 0.76  (76%)
  e^-0.5 = 0.61   |   Happy   = 0.61  / 13.11 = 0.05  ( 5%)
  e^0.8  = 2.23   |   Sad     = 2.23  / 13.11 = 0.17  (17%)
  e^-1.2 = 0.30   |   Neutral = 0.30  / 13.11 = 0.02  ( 2%)
  Sum    = 13.11  |   SUM     = 1.00

OUTPUT: [0.76, 0.05, 0.17, 0.02]
         Angry Happy  Sad  Neutral
         Sum = 1.00  always
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        |
        v
Pick highest probability  ->  ANGRY (76%)
PREDICTED EMOTION = ANGRY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Step 6 — Loss in Phase 2 (Different from Phase 1)

```
Phase 1 loss  =  MSE   (comparing number vectors to teacher outputs)
Phase 2 loss  =  Cross-Entropy  (predicted class vs correct class label)

Ground truth label : ANGRY
Model prediction   : Angry 76%, Happy 5%, Sad 17%, Neutral 2%

Cross-Entropy Loss = -log(probability given to correct class)
                   = -log(0.76)
                   = 0.27   <- small = good prediction

Bad example:
Model predicts: Angry 10%, Happy 60%, Sad 20%, Neutral 10%
Loss = -log(0.10) = 2.30   <- large = bad prediction

Only 13 weights + classification head update via backpropagation.
Backbone stays completely frozen.
Loss reduces over 50 epochs -> model gets better at predicting emotions.
```

---

## Full Shape Journey — Phase 2

| Step | Operation | Input Shape | Output Shape |
|---|---|---|---|
| Raw audio | Start | 80,000 numbers | 80,000 numbers |
| CNN | Compress waveform | 80,000 | 25 x 768 |
| Common Encoder | Enrich features | 25 x 768 | 25 x 768 |
| Branches | Semantic + Acoustic | 25 x 768 | 25 x 1536 (concatenated) |
| Duplication | Match dimensions | 7 layers x 25x768 | 7 layers x 25x1536 |
| Collect | All 13 layers | 13 grids of 25x1536 | 13 grids of 25x1536 |
| Convex Combo | 13 grids -> 1 grid | 13 x 25 x 1536 | 25 x 1536 |
| Mean Pool | 25 rows -> 1 row | 25 x 1536 | 1 x 1536 |
| Linear 1 | Compress | 1 x 1536 | 1 x 256 |
| ReLU | Remove negatives | 1 x 256 | 1 x 256 |
| Linear 2 | Emotion scores | 1 x 256 | 1 x 4 |
| Softmax | Scores -> probabilities | 1 x 4 | 1 x 4 (sum=1) |
| **Final** | **Pick highest** | **1 x 4** | **1 emotion label** |

---

## Phase 1 vs Phase 2 — Key Differences

| | Phase 1 | Phase 2 |
|---|---|---|
| Goal | Teach the backbone | Use backbone to predict |
| Teachers needed | YES (RoBERTa + PASE+) | NO |
| Labels needed | NO (unsupervised) | YES (emotion labels) |
| What trains | Full backbone (CNN + Common + branches) | Only 13 weights + classification head |
| Loss type | MSE (vector comparison) | Cross-entropy (class prediction) |
| Output | Y_sem_hat, Y_acoust_hat | Emotion probabilities |

---

## Phase 2 Conclusion

> Phase 2 is lightweight on purpose.  
> The heavy work (understanding speech) was already done in Phase 1.  
> Phase 2 just learns HOW TO READ those representations for the specific task.  
>
> In CARE: task = emotion classification  
> In our Indian children project: task = text transcription (ASR)  
> The Phase 2 structure stays almost the same — only the final head changes.
