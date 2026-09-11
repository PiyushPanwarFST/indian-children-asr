# Acoustic Joint Training - Pseudocode

```
INIT:
    # Warm start all components
    encoder    = load("checkpoints/acoustic/hpc_best_dev_model.pt")  # trained encoder
    projection = load(same_checkpoint)                                # trained projection
    ctc_head   = load("checkpoints/semantic/hpc_ctc_best_wer.pt")    # trained CTC head
    teacher    = KidWhisperMedium.encoder  # FROZEN

    encoder.train()  # UNFROZEN - all layers receive gradients
    optimizer = AdamW([encoder + projection + ctc_head], lr=3e-5)
    alpha = 0.5

FOR each epoch in 1..30:
    FOR each clip in train_set:
        wav = load_audio(clip)
        mel = extract_mel(wav)
        real_frames = compute_real_frames(wav)

        # Student forward (gradients flow)
        features = encoder(mel)                         # (1, 1500, 768)
        projected = projection(dropout(features))       # (1, 1500, 1024)

        # Teacher forward (frozen)
        with no_grad():
            teacher_features = teacher(mel)             # (1, 1500, 1024)

        # ── BRANCH A: MSE loss ──
        mse_loss = MSE(projected[:, :real_frames], teacher_features[:, :real_frames])

        # ── BRANCH B: CTC loss ──
        logits = ctc_head(features)                     # (1, 1500, 85)
        targets = text_to_indices(clip.transcript)
        ctc_loss = CTC(logits[:, :real_frames], targets)

        # Combined
        total = alpha * mse_loss + (1 - alpha) * ctc_loss
        total.backward()    # gradients flow to encoder from BOTH losses
        optimizer.step()

    # Dev evaluation
    dev_wer = greedy_decode_and_compute_wer(dev_clips)  # from CTC Head
    save_checkpoint_if_best(dev_wer)
    early_stop_if_no_improvement(patience=7)
```
