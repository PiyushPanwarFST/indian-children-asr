# Combined Branch — Unfrozen Gated Feature Fusion

## Goal
Fuse features from two independently trained Whisper Small encoders **while fine-tuning both encoders end-to-end**:
- **Acoustic encoder** (19.97% WER): Kid-Whisper distillation — captures children's voice patterns
- **Semantic encoder** (49.02% WER): IndicConformer distillation — captures Indian language phonetics

Unlike the frozen variant, both encoders are **unfrozen** and updated with a small learning rate (1e-5) so they can co-adapt their representations to the fusion module. The fusion + CTC head trains with a larger learning rate (1e-3). This **differential LR** strategy prevents destroying pre-trained features while still allowing end-to-end optimization.

## Architecture Diagram
```
Audio (.wav) — All languages (Hindi + Marathi + English)
    │
    v
Mel Spectrogram (1, 80, 3000) → SpecAugment (training only)
    │                               2 freq masks ≤15, 2 time masks ≤50
    │
    ├──────────────────────────────────┐
    v                                  v
+----------------------------+  +----------------------------+
│ ACOUSTIC ENCODER (UNFROZEN)│  │ SEMANTIC ENCODER (UNFROZEN)│
│ Whisper Small, 12 layers   │  │ Whisper Small, 12 layers   │
│ From: joint_best_wer.pt    │  │ From: best_dev.pt          │
│ Teacher: Kid-Whisper Medium│  │ Teacher: IndicConformer     │
│ LR: 1e-5 (slow adaptation)│  │ LR: 1e-5 (slow adaptation) │
│ ~88M params                │  │ ~88M params                 │
+----------------------------+  +----------------------------+
    │                                  │
    │  ┌─ Trim to real frames ─────────┤
    v  v                               v
feat_a (1, T, 768)                feat_s (1, T, 768)
    │                                  │
    └──────── concatenate ─────────────┘
                    │
                    v
            (1, T, 1536)
                    │
                    v
    +-------------------------------+
    │ GATED FUSION (TRAINABLE)      │
    │ LR: 1e-3 (fast learning)     │
    │                               │
    │ gate = σ(Linear(1536→768))    │
    │ fused = gate ⊙ feat_a        │
    │       + (1-gate) ⊙ feat_s    │
    │ → LayerNorm(768)              │
    │ → Dropout(0.1)               │
    │ ~1.18M params                 │
    +-------------------------------+
                    │
                    v
        fused features (1, T, 768)
                    │
                    v
    +-------------------------------+
    │ CTC Head: Linear(768→85)      │
    │ Warm from acoustic CTC head   │
    │ LR: 1e-3 (fast learning)     │
    │ ~65K params                   │
    +-------------------------------+
                    │
                    v
    CTC Loss ← ground truth text
                    │
                    v
    ┌───────────────────────────────────────────┐
    │         GRADIENT FLOW (end-to-end)        │
    │                                           │
    │  CTC Loss                                 │
    │    ↓                                      │
    │  CTC Head          (lr = 1e-3)            │
    │    ↓                                      │
    │  Gated Fusion      (lr = 1e-3)            │
    │    ↓              ↓                       │
    │  Acoustic Enc    Semantic Enc  (lr = 1e-5)│
    │                                           │
    │  ALL parameters receive gradients         │
    └───────────────────────────────────────────┘
```

## What Gets Updated
| Component | Params | LR | Status |
|-----------|--------|-----|--------|
| Acoustic encoder (12 Whisper layers) | ~88M | 1e-5 | TRAINABLE (slow) |
| Semantic encoder (12 Whisper layers) | ~88M | 1e-5 | TRAINABLE (slow) |
| Gated fusion (gate_linear + layer_norm) | ~1.18M | 1e-3 | TRAINABLE (fast) |
| CTC head | ~65K | 1e-3 | TRAINABLE (fast, warm-started) |
| **Total trainable** | **~177M** | | **All end-to-end** |

## Differential Learning Rate — Why?
- **Encoder LR = 1e-5**: 100x smaller than fusion LR. The encoders already carry useful pre-trained knowledge from distillation. A tiny LR lets them co-adapt to fusion without destroying those features.
- **Fusion + CTC LR = 1e-3**: These layers are either randomly initialized (fusion gate) or warm-started (CTC head). They need to learn faster to catch up.
- This is a standard technique in transfer learning / fine-tuning (similar to discriminative learning rates in ULMFiT).

## Why Gated Fusion
- Simple concatenation treats all frames equally
- Gate learns per-frame, per-dimension weighting between encoders
- Sigmoid gate initialized near 0.5 → starts with equal weighting
- Over training, gate learns: "use acoustic for prosody-heavy frames, semantic for linguistic frames"
- With unfrozen encoders, the gate and encoders co-evolve — the encoders can specialize further knowing how the gate will use them

## Checkpoint Sources (Warm Start)
| Component | Checkpoint | Key |
|-----------|------------|-----|
| Acoustic encoder | checkpoints/joint/joint_best_wer.pt | encoder_state_dict |
| CTC head (warm) | checkpoints/joint/joint_best_wer.pt | ctc_head_state_dict |
| Semantic encoder | checkpoints/semantic_mse/best_dev.pt | encoder_state_dict |

## Training Config
| Param | Value | Rationale |
|-------|-------|-----------|
| Fusion + CTC LR | 1e-3 | Fast learning for new layers |
| Encoder LR | 1e-5 | Slow adaptation, preserve pre-trained features |
| Warmup | 500 steps | Longer warmup for stable encoder updates |
| LR schedule | Linear warmup → cosine decay to 0 | Smooth convergence |
| Epochs | 30 | Standard |
| Patience | 10 | Early stopping on dev WER (more patience for slow encoder convergence) |
| Grad clip | 1.0 | Prevent gradient explosion through deep encoder stack |
| Dropout | 0.1 | In fusion layer |
| Weight decay | 0.01 | AdamW regularization |
| SpecAugment | 2 freq ≤15, 2 time ≤50 | Same augmented mel to both encoders |

## Memory Estimate (V100 32GB)
- Two unfrozen encoders + gradients: ~88M × 2 × (4B weights + 4B grads) ≈ ~1.4 GB
- AdamW optimizer states (2 moments per param): ~177M × 8B ≈ ~1.4 GB
- Activations for backprop through 24 layers: ~2-4 GB
- **Total estimate: ~5-8 GB** (fits easily on V100 32GB)
- Optional: `--grad_checkpoint` trades compute for memory if needed

## Checkpoint Format (Saved Every Epoch)
```python
{
    "epoch": int,
    # Model weights — ALL components saved
    "fusion_state_dict": fusion.state_dict(),
    "ctc_head_state_dict": ctc_head.state_dict(),
    "acoustic_encoder_state_dict": acoustic_encoder.state_dict(),
    "semantic_encoder_state_dict": semantic_encoder.state_dict(),
    # Optimizer state (for resume)
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    # Metrics
    "train_loss": float,
    "dev_wer": float,
    "dev_loss": float,
    "lang_wers": {"hi": float, "mr": float, "en": float},
    "vocab_size": 85,
    "global_step": int,
    "args": dict,
}
```
Saved to: `checkpoints/combined_unfrozen/best_wer.pt` + per-epoch `epoch_N.pt`

## Frozen vs Unfrozen — Comparison
| Aspect | Frozen | Unfrozen |
|--------|--------|----------|
| Encoder gradients | None | Yes (lr=1e-5) |
| Trainable params | ~1.25M | ~177M |
| GPU memory | ~1-2 GB | ~5-8 GB |
| Training speed | Fast (~2 min/epoch) | Slow (~25 min/epoch) |
| Encoder co-adaptation | No — encoders are fixed | Yes — encoders specialize for fusion |
| Expected WER | ~17% | ~13-15% (lower) |
| Warmup | 200 steps | 500 steps |
| Patience | 7 | 10 |

## Results
- Dev WER: 13.47% at epoch 8 (still training on HPC)
- HPC training in progress
