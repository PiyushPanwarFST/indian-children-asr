# Acoustic Sequential CTC - Architecture

## Goal
Train a CTC head on top of the FROZEN acoustic encoder to learn character-to-frame alignment.

## Architecture Diagram
```
Audio (.wav)
    |
    v
Mel Spectrogram (1, 80, 3000)
    |
    v
Acoustic Encoder (FROZEN)
(from 01_mse_distillation checkpoint)
requires_grad=False, wrapped in torch.no_grad()
    |
    v
Encoder features (1, 1500, 768)
    |
    v
CTC Head 1: Linear(768, 85) <-- ONLY TRAINABLE THING
65,365 parameters, LR=1e-3
    |
    v
Logits (1, T, 85)
    |
    v
CTC Loss <-- Ground truth transcript (training set text)
    |         text -> char indices via vocab.json (85 tokens)
    v
Backprop -> CTC Head ONLY (encoder gets zero gradient)
```

## Vocabulary
- 85 tokens from ASER-Dataset/vocab.json
- 3 special: blank (0), space (1), unk (2)
- 22 English characters (a-z)
- 60 Devanagari characters

## What Gets Updated
- CTC Head: Linear(768, 85) - 65,365 params TRAINABLE
- Encoder: FROZEN (0 trainable params)

## Key Details
- Encoder frozen with `torch.no_grad()` - stable features for CTC to learn on
- CTC loss aligns characters to audio frames
- Higher LR (1e-3) because only small CTC head is training
- All 3 languages (Hindi + Marathi + English)

## Results
- Best dev WER: 81.86% (acoustic branch)
- Hindi: best per-language WER tracked during training

## Checkpoints
- Saved to: `checkpoints/semantic/` (acoustic CTC head)
- Key: `ctc_head_state_dict`
