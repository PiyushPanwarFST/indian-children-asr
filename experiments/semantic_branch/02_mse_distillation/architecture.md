# Semantic MSE Distillation - Architecture

## Goal
Train Whisper Small encoder to produce same CTC logits as IndicConformer (teacher) - learning what speech MEANS (token predictions), not just how it sounds.

## Architecture Diagram
```
TEACHER (offline, precomputed in step0):
    Audio -> IndicConformer encoder -> (T_teacher, 1024)
          -> IndicConformer CTC decoder -> (T_teacher, 5633)
          -> language mask -> teacher_logits (T_teacher, 257)
    Saved to: teacher_logits/train/*.pt

STUDENT (trained in step1):
    Audio (.wav)
        |
        v
    Mel Spectrogram (1, 80, 3000)
        |
        v
    Whisper Encoder (TRAINABLE)
        |
        v
    Encoder features (1, T_student, 768)
        |
        v
    Slice to real frames (1, T, 768)
        |
        +------ Hindi clip? ------+------ Marathi clip? ----+
        |                         |
        v                         v
    CTC Head 2 Hindi          CTC Head 2 Marathi
    Linear(768->257)          Linear(768->257)
    TRAINABLE                 TRAINABLE
        |                         |
        v                         v
    student_logits (1, T, 257)
        |
        v
    F.interpolate(teacher_logits, size=T_student)  <- align frame rates
        |                         |
        v                         v
    student (1, T, 257)       teacher_aligned (1, T, 257)
        |                         |
        +-------- MSE Loss -------+
```

## Frame Rate Alignment
- IndicConformer: 12.5 fps (encoder stride = 80ms)
- Whisper Small: 50 fps (encoder stride = 20ms)
- Solution: F.interpolate(teacher, size=student_frames, mode='linear')
- Stretches teacher logits to match student frame count (lossless)

## Why Separate CTC Head 2 per Language
- IndicConformer has 5633 shared vocab internally
- Language mask selects 257 tokens per language
- Token index 5 in Hindi = different character than index 5 in Marathi
- Must decode with correct language-specific vocabulary

## What Gets Updated
- Encoder: ALL layers (~241M params)
- CTC Head 2 Hindi: Linear(768, 257)
- CTC Head 2 Marathi: Linear(768, 257)
- English: SKIPPED (IndicConformer has no English support)

## Key Details
- Loss: MSE (number matching, ground truth text NOT used)
- Teacher logits precomputed offline (step0) to save time
- CTC Head 2 outputs in IndicConformer's BPE vocab space
- Only Hindi + Marathi (no teacher for English)

## Evaluation (step2)
- Decode from CTC Head 2 using IndicConformer's BPE vocabulary
- Result: Hindi 39.72% | Marathi 68.11% | English N/A | Overall 47.86%
- This is a diagnostic result (proves encoder learned good features)
- Not the final system (uses teacher's vocab, not ours)

## Checkpoints
- Saved to: `checkpoints/semantic_mse/`
- Keys: `encoder_state_dict`, `ctc_head_hi_state_dict`, `ctc_head_mr_state_dict`
