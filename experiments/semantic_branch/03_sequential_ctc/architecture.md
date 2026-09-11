# Semantic Sequential CTC - Architecture

## Goal
Train CTC Head 1 on FROZEN semantic encoder to learn character-to-frame alignment using our custom 85-token vocab. Same approach that worked in acoustic branch.

## Architecture Diagram
```
Audio (.wav) -- ALL languages (Hindi + Marathi + English)
    |
    v
Mel Spectrogram (1, 80, 3000)
    |
    v
Semantic Encoder (FROZEN)
loaded from: checkpoints/semantic_mse/best_dev.pt
requires_grad=False, wrapped in torch.no_grad()
    |
    v
Encoder features (1, 1500, 768)
    |
    v
CTC Head 1: Linear(768, 85) <-- ONLY TRAINABLE THING
65,365 parameters, random init, LR=1e-3
    |
    v
Logits (1, T, 85)
    |
    v
CTC Loss <-- Ground truth text (training set transcripts)
    |         text -> char indices via vocab.json (85 tokens)
    v
Backprop -> CTC Head 1 ONLY
```

## Difference from 02_mse_distillation
| Aspect | 02 MSE Distillation | 03 Sequential CTC |
|--------|--------------------|--------------------|
| What trains | Encoder + CTC Head 2 | CTC Head 1 only |
| Loss | MSE (match teacher numbers) | CTC (match text) |
| Vocab | IndicConformer BPE (257) | Our chars (85) |
| English | No | Yes |
| Ground truth text used | No | Yes |
| Encoder state | Trainable | FROZEN |

## Key Details
- Encoder frozen = stable features for CTC to learn alignment
- Custom 85-token vocab supports all 3 languages
- LR=1e-3 (high, standard for small CTC head)
- CTC Head 2 (Hi/Mr): NOT present here, not used

## Results
- Best dev WER: 49.02%
- Hindi: 37.95% | Marathi: 74.36% | English: 56.82%

## Checkpoints
- Saved to: `checkpoints/semantic_ctc/`
- Key: `ctc_head_state_dict`
