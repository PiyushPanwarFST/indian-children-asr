# Acoustic Branch — Pseudocode

## What is this branch doing?

Kid-Whisper Medium has been trained on children's speech (MyST dataset).
It has learned HOW children sound — their pitch, pronunciation style,
breathing patterns, mumbling. We want to transfer this acoustic knowledge
into our Whisper Small student.

Unlike the semantic branch (which transfers TEXT knowledge via CTC loss),
the acoustic branch transfers SOUND knowledge via MSE loss on encoder features.

---

## Why MSE on encoder features?

The encoder's job is to convert audio → features (a compressed understanding).

Kid-Whisper's encoder produces 1024 numbers per frame.
Whisper Small's encoder produces 768 numbers per frame.

If we make the student's features SIMILAR to the teacher's features,
the student learns to "hear" audio the way the teacher does — including
the acoustic patterns specific to children's speech.

MSE (Mean Squared Error) = average of (student_feature - teacher_feature)²
Small MSE = student and teacher "hear" similarly = good
Large MSE = student and teacher "hear" differently = need more training

---

## Why no Step 0 (offline pre-computation)?

Semantic branch needed Step 0 because IndicConformer is SLOW (~6 seconds per clip).
Running it during training would make each epoch take hours.

Kid-Whisper is FAST (~0.01 seconds per clip) — it's a standard Whisper model
running entirely on GPU. So we run teacher and student together during training.

---

## Step 2: Acoustic Distillation

```
INPUT:
    All training clips (Hindi + Marathi + English)     *** ALL languages, not just Hindi/Marathi
    Dev clips for validation
    
MODELS:
    Student: Whisper Small encoder (768 dims)          *** Same encoder as semantic branch
    Teacher: Kid-Whisper Medium MyST encoder (1024 dims) *** FROZEN — never updated
    Projection: Linear layer (768 → 1024)              *** Maps student space to teacher space

WHY ALL LANGUAGES?
    Acoustic patterns (how children sound) are language-independent.
    A child's voice characteristics are the same whether speaking
    Hindi, Marathi, or English. So we use ALL clips for maximum data.

WHY A PROJECTION LAYER?
    Student produces 768 numbers per frame.
    Teacher produces 1024 numbers per frame.
    We can't directly compare 768 numbers with 1024 numbers.
    The projection layer learns to convert 768 → 1024.
    Think of it as a translator between two different "languages" of features.

WHY IS THE TEACHER FROZEN?
    Teacher already knows how children sound — we don't want to change that.
    We only update the student (to learn FROM the teacher).
    If we updated the teacher too, both would change and we'd lose
    the knowledge we're trying to transfer.
```

---

## Training Loop (per epoch)

```
FOR each audio clip in training set:

    1. LOAD audio → mel spectrogram (80 × 3000)          *** Same as semantic branch
    
    2. COUNT real frames using attention mask              *** Same padding handling
       real_frames = attention_mask.sum() ÷ 2
    
    3. STUDENT FORWARD PASS:
       audio → Whisper Small encoder → 768-dim features   (1500 frames × 768)
       768-dim features → Projection layer → 1024-dim     (1500 frames × 1024)
    
    4. TEACHER FORWARD PASS (no gradients):
       audio → Kid-Whisper encoder → 1024-dim features    (1500 frames × 1024)
       *** Teacher output is DETACHED — no gradients flow back to teacher
    
    5. COMPUTE MSE LOSS:
       loss = average of (student_projected - teacher_features)²
       *** Only on REAL frames [0 : real_frames]
       *** Padding frames are EXCLUDED from loss
    
    6. BACKWARD + UPDATE:
       loss.backward()                                    → compute gradients
       clip_grad_norm_(max_norm=1.0)                      → prevent exploding gradients
       optimizer.step()                                   → update student + projection
       scheduler.step()                                   → adjust learning rate

END FOR
```

---

## Key Differences from Semantic Branch

```
                    SEMANTIC                         ACOUSTIC
                    ─────────────────────────────────────────────────
Teacher:            IndicConformer                   Kid-Whisper Medium
What's learned:     Hindi/Marathi transcription      Children's acoustic patterns
Loss function:      CTC loss (on text logits)        MSE loss (on encoder features)
Languages:          Hindi + Marathi only             All languages
Teacher speed:      Slow (6s/clip, run offline)      Fast (0.01s/clip, run online)
Extra layer:        CTC head (768 → 51866 vocab)     Projection (768 → 1024 dims)
Output checked:     Decoded text vs ground truth     Feature similarity (MSE value)
```

---

## Hyperparameters

```
Learning rate:      3e-5 (0.00003)                    *** Same as semantic fine-tuning
Warmup:             500 steps                         *** Gradual increase to avoid crash
Optimizer:          AdamW                             *** Same as semantic
Gradient clipping:  max_norm = 1.0                    *** Prevents exploding gradients
Dropout:            0.1 (10%)                         *** Before projection layer
Early stopping:     patience = 5                      *** Stop if dev loss doesn't improve
Epochs:             20 maximum                        *** Early stopping usually triggers before
```

---

## What does success look like?

```
GOOD training:
    Train MSE: 0.50 → 0.30 → 0.20 → 0.15 → 0.12    (decreasing)
    Dev MSE:   0.55 → 0.35 → 0.25 → 0.22 → 0.21    (decreasing, small gap)
    
    → Student's features becoming similar to teacher's features
    → Student learning to "hear" like a children's speech expert

BAD training (overfitting):
    Train MSE: 0.50 → 0.10 → 0.01                    (dropping too fast)
    Dev MSE:   0.55 → 0.45 → 0.50                    (going up)
    
    → Student memorizing training audio, not generalizing

EXPECTED in full run:
    Both losses should decrease for many epochs because:
    - MSE is a smoother loss than CTC (less noisy gradients)
    - No teacher transcript quality issue (features are always correct)
    - Using ALL languages gives more training data
```

---

## GPU Memory Estimate

```
Teacher encoder (frozen):    ~1200 MB     (Kid-Whisper Medium, no gradients)
Student encoder:              ~350 MB     (Whisper Small)
Projection layer:                ~3 MB     (768 × 1024 = ~800K params)
Forward pass activations:    ~2000 MB     (both encoders + gradients)
─────────────────────────────────────────
Total:                        ~3500 MB     (fits in 8GB RTX 4060)
```
