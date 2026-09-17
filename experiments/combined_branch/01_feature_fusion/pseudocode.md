# Combined Branch — Unfrozen Gated Feature Fusion Pseudocode

## Overview
Both encoders are UNFROZEN and trained end-to-end with the fusion module.
Differential learning rate: encoders (1e-5) vs fusion+CTC (1e-3).

```
# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════

# Step 1: Load pre-trained encoders (UNFROZEN — all params trainable)
acoustic_enc = WhisperSmall.encoder()
acoustic_enc.load_weights("checkpoints/joint/joint_best_wer.pt" → encoder_state_dict)
# Source: Kid-Whisper distillation → knows children's voice patterns
# All 12 transformer layers remain TRAINABLE (not frozen)

semantic_enc = WhisperSmall.encoder()
semantic_enc.load_weights("checkpoints/semantic_mse/best_dev.pt" → encoder_state_dict)
# Source: IndicConformer distillation → knows Indian language phonetics
# All 12 transformer layers remain TRAINABLE (not frozen)

# Step 2: Create trainable fusion module
gate_linear = Linear(1536, 768)      # Input: concat of both encoders
layer_norm  = LayerNorm(768)
dropout     = Dropout(0.1)

# Step 3: CTC head — warm-started from acoustic branch
ctc_head = Linear(768, 85)           # 85 = character vocab size
ctc_head.load_weights("checkpoints/joint/joint_best_wer.pt" → ctc_head_state_dict)
# Warm start: CTC head already knows char→CTC mapping from acoustic training

# Step 4: Optimizer with DIFFERENTIAL learning rates
#   - Encoders: tiny LR to preserve pre-trained knowledge
#   - Fusion + CTC: larger LR since these layers need to learn/adapt faster
optimizer = AdamW([
    {params: acoustic_enc + semantic_enc, lr: 1e-5},   # 100x smaller
    {params: gate_linear + layer_norm + ctc_head, lr: 1e-3},
], weight_decay=0.01)

# Step 5: LR schedule — linear warmup then cosine decay to zero
scheduler = LinearWarmup(500 steps) + CosineDecay(to 0)

ctc_loss_fn = CTCLoss(blank=0, zero_infinity=True)

# Total trainable: ~177M params
#   Acoustic encoder: ~88M
#   Semantic encoder: ~88M
#   Fusion module:    ~1.18M
#   CTC head:         ~65K

# Optional: gradient checkpointing for memory savings
if grad_checkpoint:
    acoustic_enc.enable_gradient_checkpointing()
    semantic_enc.enable_gradient_checkpointing()


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

best_dev_wer = infinity
patience_counter = 0

# Set ALL modules to training mode (encoders are NOT frozen)
acoustic_enc.train()
semantic_enc.train()
fusion.train()
ctc_head.train()

FOR epoch in 1..30:

    shuffle(train_clips)       # Clips from all 3 languages mixed

    FOR each clip in train_clips:

        # ── 1. Load and validate audio ──
        wav = load_audio(clip.audio_path, max_sec=30.0)
        if len(wav) < 8000:    # Skip clips < 0.5 seconds
            skip; continue

        num_samples = len(wav)

        # ── 2. Extract mel spectrogram + SpecAugment ──
        mel = whisper_feature_extractor(wav)       # (1, 80, 3000)
        mel = spec_augment(mel)                     # Data augmentation
            # 2 frequency masks (width ≤ 15 mel bins)
            # 2 time masks (width ≤ 50 frames)
        mel = mel.to(GPU)

        # ── 3. Forward through BOTH encoders (WITH gradients!) ──
        # Key difference from frozen: no torch.no_grad() here
        a_out = acoustic_enc(mel).last_hidden_state    # (1, 1500, 768)
        s_out = semantic_enc(mel).last_hidden_state    # (1, 1500, 768)

        # Trim to real frames (remove padding beyond actual audio length)
        real_frames = ceil(num_samples / 320)          # Whisper: 320 samples per frame
        real_frames = min(real_frames, 1500)            # Cap at max sequence length
        a_feat = a_out[:, :real_frames, :]              # (1, T, 768)
        s_feat = s_out[:, :real_frames, :]              # (1, T, 768)

        # ── 4. Gated fusion ──
        concat = cat([a_feat, s_feat], dim=-1)          # (1, T, 1536)
        gate   = sigmoid(gate_linear(concat))           # (1, T, 768)
        # gate ≈ 1 → trust acoustic encoder
        # gate ≈ 0 → trust semantic encoder
        fused  = gate * a_feat + (1 - gate) * s_feat    # (1, T, 768)
        fused  = dropout(layer_norm(fused))

        # ── 5. CTC head + loss computation ──
        logits    = ctc_head(fused)                     # (1, T, 85)
        log_probs = log_softmax(logits, dim=-1)         # (1, T, 85)
        log_probs = log_probs.permute(1, 0, 2)          # (T, 1, 85) — CTC format

        gt_text = normalize_text(clip.ground_truth)     # lowercase, strip punctuation
        targets = text_to_indices(gt_text, char_vocab)  # list of int indices

        if targets is empty OR real_frames <= len(targets):
            skip; continue      # CTC requires T > target length

        loss = CTC_loss(log_probs, targets,
                        input_lengths=[real_frames],
                        target_lengths=[len(targets)])

        if isnan(loss) or isinf(loss):
            skip; continue

        # ── 6. Backward pass — gradients flow through EVERYTHING ──
        optimizer.zero_grad()
        loss.backward()
        #
        # Gradient flow:
        #   CTC Loss
        #     → CTC Head (768→85)        — lr = 1e-3
        #     → Gated Fusion             — lr = 1e-3
        #       → gate_linear (1536→768)
        #       → layer_norm
        #     → Acoustic Encoder         — lr = 1e-5 (100x smaller)
        #       → 12 transformer layers
        #       → conv positional embed
        #     → Semantic Encoder         — lr = 1e-5 (100x smaller)
        #       → 12 transformer layers
        #       → conv positional embed
        #
        clip_grad_norm(ALL_trainable_params, max_norm=1.0)  # Prevent explosion
        optimizer.step()     # Different LR applied per param group
        scheduler.step()     # Update both LRs (warmup → cosine)

        # ── 7. Free GPU memory ──
        del a_out, s_out, a_feat, s_feat, fused, logits
        cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # END OF EPOCH — Dev Evaluation
    # ═══════════════════════════════════════════════════════════

    acoustic_enc.eval()
    semantic_enc.eval()
    fusion.eval()
    ctc_head.eval()

    dev_wer, dev_loss = 0, 0
    all_refs, all_hyps = [], []
    lang_wers = {}

    with no_grad():
        FOR each clip in dev_clips:
            wav = load_audio(clip.audio_path)
            mel = whisper_feature_extractor(wav).to(GPU)
            # No SpecAugment during evaluation

            a_feat = acoustic_enc(mel).last_hidden_state[:, :real_frames]
            s_feat = semantic_enc(mel).last_hidden_state[:, :real_frames]
            fused  = fusion(a_feat, s_feat)
            logits = ctc_head(fused)

            # CTC greedy decoding
            pred_indices = argmax(logits, dim=-1)       # (1, T)
            pred_text    = ctc_greedy_decode(pred_indices, idx_to_char)
            #   → collapse repeated chars, remove blanks

            all_refs.append(clip.ground_truth)
            all_hyps.append(pred_text)

    # Compute WER (word error rate)
    dev_wer = compute_corpus_wer(all_refs, all_hyps)
    lang_wers = {
        "hi": wer(hindi_refs, hindi_hyps),
        "mr": wer(marathi_refs, marathi_hyps),
        "en": wer(english_refs, english_hyps),
    }

    print(f"Epoch {epoch}: Dev WER = {dev_wer}%")
    print(f"  Hindi: {lang_wers['hi']}%")
    print(f"  Marathi: {lang_wers['mr']}%")
    print(f"  English: {lang_wers['en']}%")
    print(f"  Encoder LR: {scheduler.get_lr()[0]}")
    print(f"  Fusion LR:  {scheduler.get_lr()[1]}")

    # ═══════════════════════════════════════════════════════════
    # CHECKPOINT SAVING
    # ═══════════════════════════════════════════════════════════

    checkpoint = {
        "epoch": epoch,
        # ALL model weights saved (including both encoders)
        "acoustic_encoder_state_dict": acoustic_enc.state_dict(),
        "semantic_encoder_state_dict": semantic_enc.state_dict(),
        "fusion_state_dict": fusion.state_dict(),
        "ctc_head_state_dict": ctc_head.state_dict(),
        # Optimizer + scheduler (for resume)
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        # Metrics
        "train_loss": avg_loss,
        "dev_wer": dev_wer,
        "lang_wers": lang_wers,
    }

    if dev_wer < best_dev_wer:
        best_dev_wer = dev_wer
        patience_counter = 0
        save(checkpoint, "checkpoints/combined_unfrozen/best_wer.pt")
    else:
        patience_counter += 1
        if patience_counter >= 10:      # patience = 10
            EARLY STOP
            break

    save(checkpoint, f"checkpoints/combined_unfrozen/epoch_{epoch}.pt")

    # Back to training mode for next epoch
    acoustic_enc.train()
    semantic_enc.train()
    fusion.train()
    ctc_head.train()
```

## Key Differences from Frozen Version

| Aspect | Frozen | Unfrozen (this) |
|--------|--------|-----------------|
| `acoustic_enc.freeze()` | Yes | **No** — all params trainable |
| `with no_grad():` around encoders | Yes | **No** — gradients flow through |
| Optimizer params | fusion + CTC only | **encoders + fusion + CTC** |
| Learning rate | 1e-3 (single) | **1e-5 encoders, 1e-3 fusion** (differential) |
| Trainable params | ~1.25M | **~177M** |
| Checkpoint saves encoders | No | **Yes** — encoder weights change |
| Warmup steps | 200 | **500** |
| Patience | 7 | **10** |
| GPU memory | ~1-2 GB | **~5-8 GB** |
| Why better | Quick, stable | Encoders co-adapt to fusion task |
