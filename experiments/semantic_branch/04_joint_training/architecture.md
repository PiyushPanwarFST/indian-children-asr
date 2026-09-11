# Semantic Joint Training - Architecture

## Goal
Co-optimize encoder with dual loss: MSE (match IndicConformer) + CTC (predict text). Both CTC heads warm-started. This is the CARE paper equivalent for the semantic branch.

## Architecture Diagram
```
Audio (.wav) -- Hindi + Marathi only (need teacher logits for MSE)
    |
    v
Mel Spectrogram (1, 80, 3000)
    |
    v
SpecAugment (2 freq masks <= 15 bins, 2 time masks <= 50 frames)
    |
    v
+----------------------------------------------------+
| WHISPER ENCODER (12 transformer layers)             |
|   Layers 0-5:  FROZEN (no gradients)                |
|   Layers 6-11: UNFROZEN (receive grads from both)   |
|   Conv1, Conv2: FROZEN                              |
|   Gradient checkpointing: ENABLED                   |
+----------------------------------------------------+
    |
    v
Encoder features (1, T, 768)
    |
    +--- Dropout(0.1) ---+          +--- Dropout(0.1) ---+
    |                     |          |                     |
    v                     |          v                     |
CTC Head 2               |      CTC Head 1               |
Hindi: Linear(768->257)   |      Linear(768->85)          |
OR Marathi: 768->257      |      (our custom vocab)       |
warm from step1           |      warm from step2b          |
    |                     |          |                     |
    v                     |          v                     |
student_logits (1,T,257)  |      char_logits (1,T,85)     |
    |                     |          |                     |
    v                     |          v                     |
MSE Loss                  |      CTC Loss                 |
<-- teacher logits        |      <-- ground truth text     |
(precomputed,             |      (training transcripts)   |
 interpolated to T)       |                                |
    |                     |          |                     |
    +---------- + ------------------+                     |
                |                                          |
                v                                          |
    total = alpha * MSE + (1-alpha) * CTC                  |
                |                                          |
                v                                          |
    Backprop -> Encoder top 6 + CTC Head 2 + CTC Head 1   |
```

## Alpha Schedule
| Epochs | Alpha | MSE Weight | CTC Weight |
|--------|-------|------------|------------|
| 1-5    | 0.3   | 30%        | 70%        |
| 6-15   | 0.2   | 20%        | 80%        |
| 16+    | 0.1   | 10%        | 90%        |

CTC dominates later -> optimize for text prediction.

## Warm Start Sources
| Component | Source | Checkpoint |
|-----------|--------|------------|
| Encoder | Step 1 MSE | checkpoints/semantic_mse/best_dev.pt |
| CTC Head 2 Hi | Step 1 MSE | same checkpoint |
| CTC Head 2 Mr | Step 1 MSE | same checkpoint |
| CTC Head 1 | Step 2b sequential CTC | checkpoints/semantic_ctc/best_wer.pt |

## Training Improvements
1. SpecAugment (regularization on mel input)
2. Dropout(0.1) before both heads
3. Bottom 6/12 encoder layers frozen
4. Gradient accumulation = 4 (effective batch size = 4)
5. Cosine LR decay with 1000-step warmup
6. Gradient clipping (max_norm=1.0)

## What Gets Updated
- Encoder layers 6-11 (from BOTH MSE and CTC gradients)
- CTC Head 2 Hi/Mr (from MSE gradient only)
- CTC Head 1 (from CTC gradient only)
- Encoder layers 0-5, Conv1, Conv2: FROZEN

## Dev Evaluation
- WER computed from CTC Head 1 ONLY (greedy decode, 85-token vocab)
- MSE from CTC Head 2 logged as secondary metric
- Early stopping on dev WER (patience=7)

## CTC-only Fine-tuning Mode (--ctc_only)
Joint training produced 52.45% WER — WORSE than sequential CTC (49.02%).
Root cause: MSE constrains encoder features, preventing CTC from adapting them.

CTC-only mode drops MSE entirely:
- alpha = 0.0 always, CTC Head 2 not loaded
- All 3 languages (Hindi + Marathi + English)
- Encoder top 6 layers unfrozen, CTC Head 1 warm from step2b
- IndicConformer knowledge already embedded in encoder from step1
- Checkpoints saved to: `checkpoints/semantic_ctc_finetune/`

## Results
- Joint training (MSE+CTC): 52.45% WER (worse than sequential)
- Sequential CTC (frozen encoder): 49.02% WER (Hi 37.95% | Mr 74.36% | En 56.82%)
- CTC-only fine-tuning: pending

## Checkpoints
- Saved to: `checkpoints/semantic_joint/`
- Keys: `encoder_state_dict`, `ctc_head_hi_state_dict`, `ctc_head_mr_state_dict`, `ctc_head_char_state_dict`
