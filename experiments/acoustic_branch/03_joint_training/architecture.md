# Acoustic Joint Training - Architecture

## Goal
Co-optimize encoder + CTC head with dual loss (MSE + CTC). Encoder gets gradients from both acoustic teacher matching AND text prediction.

## Architecture Diagram
```
Audio (.wav)
    |
    v
Mel Spectrogram (1, 80, 3000)
    |
    v
Whisper Encoder (UNFROZEN - all layers trainable)
warm from: checkpoints/acoustic/hpc_best_dev_model.pt
    |
    v
Encoder features (1, 1500, 768)
    |
    +---------------------------+
    |                           |
    v                           v
Projection + Dropout        CTC Head 1 (768->85)
Linear(768->1024)            warm from: sequential CTC
warm from: acoustic MSE      checkpoint
    |                           |
    v                           v
(1, T, 1024)                (1, T, 85)
    |                           |
    v                           v
MSE Loss                    CTC Loss
<-- Teacher features        <-- Ground truth text
(Kid-Whisper, FROZEN)        (training set transcripts)
    |                           |
    +----------- + -----------+
                 |
                 v
    total = alpha * MSE + (1-alpha) * CTC
                 |
                 v
    Backprop -> Encoder + Projection + CTC Head
```

## Warm Start Sources
| Component | Source |
|-----------|--------|
| Encoder | checkpoints/acoustic/hpc_best_dev_model.pt (step 1) |
| Projection | same checkpoint |
| CTC Head 1 | checkpoints/semantic/hpc_ctc_best_wer.pt (step 2) |

## What Gets Updated
- Encoder: ALL layers unfrozen (~241M params)
- Projection: Linear(768, 1024)
- CTC Head: Linear(768, 85)
- Teacher: FROZEN

## Key Details
- Alpha = 0.5 (equal weight MSE and CTC)
- LR = 3e-5 with cosine decay
- Dev metric: WER from CTC Head (greedy decode)
- All languages used for CTC, Hindi+Marathi for MSE

## Results
- Best dev WER: 19.97%
- Hindi: 15.72% | Marathi: 26.37% | English: 30.03%

## Checkpoints
- Saved to: `checkpoints/joint/`
- Keys: `encoder_state_dict`, `projection_state_dict`, `ctc_head_state_dict`
