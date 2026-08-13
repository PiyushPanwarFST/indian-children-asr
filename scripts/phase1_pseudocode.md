# Phase 1: Multi-Teacher Knowledge Distillation — Pseudocode

## High-Level Flow

```
PARSE command-line args (--verify, --epochs, --patience, --seed, --lr, etc.)

STEP 1: LOAD DATA
    Read asr_train.csv → keep clips ≤ 30s
    Split by language: hindi[], marathi[], english[]
    IF --verify: sample N clips proportionally, set epochs=3
    Read asr_dev.csv → dev_clips (for overfitting detection)

STEP 2: LOAD 3 MODELS
    Teacher A  = XLSR-53 (315M params, FROZEN, float16)    → Hindi/Marathi semantic targets
    Teacher B  = Kid-Whisper encoder (88M params, FROZEN, float16) → acoustic targets
    Student C  = MMS-300M (315M params, TRAINS, float32)   → learns from both teachers
    Projection = Linear(1024 → 768)                        → maps student dim to Teacher B dim

STEP 3: VERIFY COMPONENTS (one test clip)
    Run one Hindi clip through all 3 paths
    Check: teachers produce no gradients, student produces gradients
    Check: loss computes and backward() works
    Print peak GPU memory

STEP 4: SETUP
    Optimizer = AdamW(student_params + projection_params, lr=3e-5)
    Set random seeds for reproducibility

STEP 5: TRAINING LOOP
    FOR each epoch:
        Shuffle training clips
        FOR each clip:
            1. load_audio(clip) → 16kHz mono numpy array
            2. get_teacher_targets(audio, language):
                 Teacher B: audio → mel spectrogram → Whisper encoder → Y_kid (1, T_valid, 768)
                            T_valid = num_samples // 160 // 2  (real speech frames only)
                 Teacher A: IF Hindi/Marathi:
                            audio → XLSR-53 → avg_pool → Y_indic (1, 1024)
                            ELSE: Y_indic = None
            3. get_student_outputs(audio):
                 audio → MMS-300M → frames (1, T_student, 1024)
                 Semantic branch: avg_pool(frames) → Y_sem_hat (1, 1024)
                 Acoustic branch: Linear(frames) → Y_ac_hat (1, T_student, 768)
            4. COMPUTE LOSS:
                 T = min(T_valid, T_student)
                 L_acoustic = MSE(Y_ac_hat[:T], Y_kid[:T])     ← all clips
                 L_semantic = MSE(Y_sem_hat, Y_indic)           ← Hindi/Marathi only
                 L_total = L_acoustic + L_semantic
            5. optimizer.zero_grad()
               L_total.backward()
               clip_grad_norm(max=1.0)
               optimizer.step()
            6. Clear GPU cache

        END FOR (clips)

        PRINT epoch summary (avg losses, time, GPU peak)

        DEV EVALUATION:
            Switch student to eval mode
            Run all dev clips (no gradients) → compute avg dev loss
            Print train vs dev gap
            IF patience > 0:
                IF dev_loss < best_dev_loss: reset patience counter
                ELSE: increment patience counter
                IF counter >= patience: STOP TRAINING

        SAVE CHECKPOINTS:
            best_train_model.pt  (lowest training loss so far)
            best_dev_model.pt    (lowest dev loss so far)
            epoch_N.pt           (every epoch)

    END FOR (epochs)

STEP 6: PRINT RESULTS
    Epoch-by-epoch loss table (train, dev, semantic, acoustic, gap)
    Overall reduction percentages
    First-10 vs last-10 step comparison
    Timing stats + full run estimate
```

## Key Functions

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `load_audio(path, max_sec)` | file path | numpy array (16kHz mono) | Load + resample + trim audio |
| `get_teacher_targets(audio, lang)` | audio array, language string | (Y_indic, Y_kid) | Run frozen teachers, return target representations |
| `get_student_outputs(audio)` | audio array | (Y_sem_hat, Y_ac_hat) | Run trainable student, return both branches |
| `evaluate_dev(dev_clips)` | list of clip dicts | (avg_total, avg_sem, avg_ac) | Dev set loss (no training) |

## Key Dimensions

```
Audio: (num_samples,)         e.g., 10s clip = (160000,)

Teacher A (XLSR-53):
  Input:  (1, num_samples)
  Output: (1, T_xlsr, 1024)   → avg_pool → (1, 1024)

Teacher B (Kid-Whisper encoder):
  Input:  (1, 80, 3000)       mel spectrogram (always padded to 30s = 3000 frames)
  Output: (1, 1500, 768)      always 1500 frames (30s padded)
  Masked: (1, T_valid, 768)   T_valid = num_samples // 160 // 2

Student (MMS-300M):
  Input:  (1, num_samples)
  Output: (1, T_mms, 1024)
  Semantic: avg_pool → (1, 1024)        matches Teacher A
  Acoustic: Linear → (1, T_mms, 768)    matches Teacher B (after min-T alignment)

Projection: Linear(1024, 768)  — 787,200 params
```

## GPU Memory Budget (RTX 4060, 8GB)

```
Teacher A (XLSR-53, fp16):     ~602 MB
Teacher B (Kid-Whisper, fp16): ~168 MB
Student (MMS-300M, fp32):     ~1203 MB
Optimizer states (AdamW):     ~2400 MB  (2x model size for momentum+variance)
Activations + gradients:      ~2000-3000 MB (with gradient checkpointing)
─────────────────────────────────────────
Total peak:                   ~6500-7000 MB / 7932 MB available
```

## Command Reference

```bash
# Quick verify (2 min)
python scripts/phase1_combined.py --verify

# Verify with different seed
python scripts/phase1_combined.py --verify --seed 123

# Full training with early stopping (~9h)
python scripts/phase1_combined.py --epochs 20 --patience 5

# Full training, log to file
nohup python scripts/phase1_combined.py --epochs 20 --patience 5 \
    2>&1 | tee phase1_full_training.log &
```
