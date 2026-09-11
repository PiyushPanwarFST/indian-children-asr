# Combined Branch — Gated Feature Fusion

## Goal
Fuse features from two independently trained Whisper Small encoders:
- **Acoustic** (19.97% WER): Kid-Whisper distillation — captures children's voice patterns
- **Semantic** (49.02% WER): IndicConformer distillation — captures Indian language phonetics

Neither encoder alone uses both knowledge sources. Gated fusion learns per-frame weighting.

## Architecture Diagram
```
Audio (.wav) -- All languages (Hindi + Marathi + English)
    |
    v
Mel Spectrogram (1, 80, 3000) → SpecAugment (training only)
    |
    ├──────────────────────────────────┐
    v                                  v
+----------------------------+  +----------------------------+
| ACOUSTIC ENCODER (FROZEN)  |  | SEMANTIC ENCODER (FROZEN)  |
| Whisper Small, 12 layers   |  | Whisper Small, 12 layers   |
| From: joint_best_wer.pt    |  | From: semantic_mse/best.pt |
| Teacher: Kid-Whisper Med   |  | Teacher: IndicConformer    |
+----------------------------+  +----------------------------+
    |                                  |
    v                                  v
feat_a (1, T, 768)                feat_s (1, T, 768)
    |                                  |
    +──────── concatenate ─────────────+
                    |
                    v
            (1, T, 1536)
                    |
                    v
    +-------------------------------+
    | GATED FUSION (TRAINABLE)      |
    |                               |
    | gate = σ(Linear(1536→768))    |
    | fused = gate * feat_a         |
    |       + (1-gate) * feat_s     |
    | → LayerNorm(768)              |
    | → Dropout(0.1)               |
    +-------------------------------+
                    |
                    v
        fused features (1, T, 768)
                    |
                    v
    +-------------------------------+
    | CTC Head: Linear(768→85)      |
    | Warm from acoustic CTC head   |
    +-------------------------------+
                    |
                    v
    CTC Loss ← ground truth text
                    |
                    v
    Backprop → Fusion + CTC Head ONLY
    (encoders stay frozen)
```

## What Gets Updated
| Component | Params | Status |
|-----------|--------|--------|
| Acoustic encoder | ~88M | FROZEN |
| Semantic encoder | ~88M | FROZEN |
| Gated fusion (gate_linear + layer_norm) | ~1.18M | TRAINABLE |
| CTC head | ~65K | TRAINABLE (warm-started) |
| **Total trainable** | **~1.25M** | |

## Why Gated Fusion
- Simple concatenation treats all frames equally
- Gate learns per-frame, per-dimension weighting between encoders
- Sigmoid gate initialized near 0.5 → starts with equal weighting
- Over training, gate learns: "use acoustic for prosody-heavy frames, semantic for linguistic frames"

## Checkpoint Sources
| Component | Checkpoint | Key |
|-----------|------------|-----|
| Acoustic encoder | checkpoints/joint/joint_best_wer.pt | encoder_state_dict |
| CTC head (warm) | checkpoints/joint/joint_best_wer.pt | ctc_head_state_dict |
| Semantic encoder | checkpoints/semantic_mse/best_dev.pt | encoder_state_dict |

## Training Config
| Param | Value | Rationale |
|-------|-------|-----------|
| LR | 1e-3 | Only ~1.25M trainable (like step2b CTC head) |
| Warmup | 200 steps | Small model, fast convergence |
| Epochs | 30 | Standard |
| Patience | 7 | Early stopping on dev WER |
| Grad clip | 1.0 | Consistent |
| Dropout | 0.1 | In fusion layer |
| SpecAugment | 2 freq ≤15, 2 time ≤50 | Same mel to both encoders |

## Memory (~1-2 GB total)
- Two frozen encoders: ~704 MB (no gradient graph)
- Trainable fusion + CTC + optimizer: ~30 MB
- No teacher encoder needed (unlike joint training)

## Results
- Pending

## Checkpoints
- Saved to: `checkpoints/combined/`
- Keys: `fusion_state_dict`, `ctc_head_state_dict`
