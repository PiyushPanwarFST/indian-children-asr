# Semantic Joint Training - Pseudocode

```
INIT:
    # Warm start all components
    ckpt = load("checkpoints/semantic_mse/best_dev.pt")
    encoder     = WhisperSmall.encoder
    encoder.load(ckpt["encoder_state_dict"])         # trained in step1
    ctc_head_hi = Linear(768, 257)
    ctc_head_hi.load(ckpt["ctc_head_hi_state_dict"]) # trained in step1
    ctc_head_mr = Linear(768, 257)
    ctc_head_mr.load(ckpt["ctc_head_mr_state_dict"]) # trained in step1

    ctc_ckpt = load("checkpoints/semantic_ctc/best_wer.pt")
    ctc_head_char = Linear(768, 85)
    ctc_head_char.load(ctc_ckpt["ctc_head_state_dict"])  # trained in step2b

    # Freeze bottom half
    freeze(encoder.conv1, encoder.conv2, encoder.layers[0:6])
    unfreeze(encoder.layers[6:12])

    dropout   = Dropout(0.1)
    optimizer = AdamW(all_trainable_params, lr=2e-5)
    scheduler = warmup(1000) + cosine_decay
    ctc_loss  = CTCLoss(blank=0)

FOR each epoch in 1..30:
    alpha = get_alpha(epoch)   # 0.3 -> 0.2 -> 0.1
    shuffle(train_clips)       # Hindi + Marathi only

    FOR each clip in train_clips:
        wav = load_audio(clip)
        mel = extract_mel(wav)
        mel = spec_augment(mel)
        real_frames = compute_real_frames(wav)

        # Encoder forward (top 6 layers get gradients)
        features = encoder(mel)[:, :real_frames]    # (1, T, 768)

        # ── BRANCH A: MSE loss ──
        if clip.lang == "hi":
            student_257 = ctc_head_hi(dropout(features))
        else:
            student_257 = ctc_head_mr(dropout(features))

        teacher_257 = load_precomputed(clip.logit_path)
        teacher_257 = interpolate(teacher_257, to=real_frames)
        mse_loss = MSE(student_257, teacher_257)

        # ── BRANCH B: CTC loss ──
        char_logits = ctc_head_char(dropout(features))  # (1, T, 85)
        targets = text_to_indices(clip.transcript)
        ctc_loss_val = CTC(char_logits, targets)

        # Combined
        total = (alpha * mse_loss + (1-alpha) * ctc_loss_val) / grad_accum
        total.backward()

        # Update every 4 clips (gradient accumulation)
        if step % 4 == 0:
            clip_grad_norm(max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Dev evaluation
    dev_wer = greedy_decode_from_ctc_head_1(dev_clips)  # WER metric
    dev_mse = compute_mse_from_ctc_head_2(dev_clips)     # secondary
    save_checkpoint_if_best(dev_wer)
    early_stop_if_no_improvement(patience=7)
```
