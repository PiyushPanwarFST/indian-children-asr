# Acoustic MSE Distillation - Architecture

## Goal
Train Whisper Small encoder to produce same hidden features as Kid-Whisper Medium (teacher) for children's speech.

## Architecture Diagram
```
Audio (.wav)
    |
    v
Mel Spectrogram (1, 80, 3000)
    |
    +--------------------------+
    |                          |
    v                          v
Student Encoder            Teacher Encoder (FROZEN)
(Whisper Small)            (Kid-Whisper Medium)
12 layers, 768-dim         24 layers, 1024-dim
    |                          |
    v                          v
(B, 1500, 768)            (B, 1500, 1024)
    |
    v
Projection Layer
Linear(768 -> 1024) + Dropout(0.1)
    |                          |
    v                          v
(B, 1500, 1024)           (B, 1500, 1024)
    |                          |
    +--------- MSE Loss ------+
    (only on REAL frames, not padding)
```

## What Gets Updated
- Student Encoder: ALL layers trainable (~241M params)
- Projection Layer: Linear(768, 1024) trainable (~787K params)
- Teacher: FROZEN (never updated)

## Key Details
- Teacher: Kid-Whisper Medium (fine-tuned on MyST children's speech)
- Loss: MSE on encoder hidden states (not logits)
- All 3 languages used (acoustic patterns are language-independent)
- Real frames only (padding excluded via attention mask)

## Evaluation
- Trained encoder + Whisper's original decoder (beam search)
- Result: Hindi 177.42% | Marathi 153.12% | English 130.97% | Overall 167.10%
- High WER expected (encoder optimized for MSE, not for Whisper decoder)

## Checkpoints
- Saved to: `checkpoints/acoustic/`
- Key: `model_state_dict` (encoder + projection weights)
