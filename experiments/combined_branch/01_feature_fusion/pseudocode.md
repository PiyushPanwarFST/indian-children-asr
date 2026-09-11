# Combined Branch — Gated Feature Fusion Pseudocode

```
INIT:
    # Load two frozen encoders
    acoustic_enc = load_whisper_small("checkpoints/joint/joint_best_wer.pt")
    acoustic_enc.freeze()   # ALL params frozen

    semantic_enc = load_whisper_small("checkpoints/semantic_mse/best_dev.pt")
    semantic_enc.freeze()   # ALL params frozen

    # Trainable fusion
    gate_linear = Linear(1536, 768)
    layer_norm  = LayerNorm(768)
    dropout     = Dropout(0.1)

    # CTC head warm-started from acoustic branch
    ctc_head = Linear(768, 85)
    ctc_head.load("checkpoints/joint/joint_best_wer.pt" -> ctc_head_state_dict)

    optimizer = AdamW([gate_linear, layer_norm, ctc_head], lr=1e-3)
    scheduler = warmup(200) + cosine_decay
    ctc_loss  = CTCLoss(blank=0)

FOR each epoch in 1..30:
    shuffle(train_clips)   # All 3 languages

    FOR each clip in train_clips:
        wav = load_audio(clip)
        mel = extract_mel(wav)
        mel = spec_augment(mel)    # Same augmented mel for both encoders
        real_frames = compute_real_frames(wav)

        # Both encoders (no gradients)
        with no_grad():
            feat_a = acoustic_enc(mel)[:, :real_frames]   # (1, T, 768)
            feat_s = semantic_enc(mel)[:, :real_frames]    # (1, T, 768)

        # Gated fusion (with gradients)
        concat = cat([feat_a, feat_s], dim=-1)             # (1, T, 1536)
        gate   = sigmoid(gate_linear(concat))              # (1, T, 768)
        fused  = gate * feat_a + (1 - gate) * feat_s       # (1, T, 768)
        fused  = dropout(layer_norm(fused))

        # CTC loss
        logits  = ctc_head(fused)                          # (1, T, 85)
        targets = text_to_indices(clip.transcript)
        loss    = CTC(logits, targets)

        loss.backward()    # Only fusion + ctc_head get gradients
        clip_grad_norm(max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Dev evaluation
    dev_wer = greedy_decode_all_dev(dev_clips)
    per_lang_wer = {Hindi, Marathi, English}
    save_checkpoint_if_best(dev_wer)
    early_stop_if_no_improvement(patience=7)
```
