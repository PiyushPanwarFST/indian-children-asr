# Semantic MSE Distillation - Pseudocode

## Step 0: Precompute teacher logits (one-time)
```
model = load_indicconformer()

FOR each clip in train_set (Hindi + Marathi only):
    wav = load_audio(clip)
    encoder_out = model.encode(wav)               # (1, 1024, T_enc)
    raw_logits = model.ctc_decoder(encoder_out)    # (1, T_enc, 5633)
    lang_mask = model.language_masks[clip.lang]     # boolean[5633]
    logits = raw_logits[:, :, lang_mask]            # (T_enc, 257)
    save(logits, f"teacher_logits/train/{clip_uid}.pt")
```

## Step 1: MSE training
```
INIT:
    encoder     = WhisperSmall.encoder        # 768-dim, TRAINABLE
    ctc_head_hi = Linear(768, 257)            # Hindi bridge head
    ctc_head_mr = Linear(768, 257)            # Marathi bridge head
    optimizer   = AdamW([encoder + heads], lr=3e-5)

FOR each epoch in 1..20:
    FOR each clip in train_set:   # Hindi + Marathi only
        wav = load_audio(clip)
        mel = extract_mel(wav)
        real_frames = compute_real_frames(wav)

        # Student forward
        features = encoder(mel)                    # (1, 1500, 768)
        real_features = features[:, :real_frames]  # (1, T, 768)

        if clip.lang == "hi":
            student_logits = ctc_head_hi(real_features)  # (1, T, 257)
        else:
            student_logits = ctc_head_mr(real_features)  # (1, T, 257)

        # Teacher logits (precomputed)
        teacher_logits = load(clip.logit_path)     # (T_teacher, 257)
        teacher_aligned = F.interpolate(teacher_logits, size=real_frames)

        # MSE loss
        loss = MSE(student_logits, teacher_aligned)
        loss.backward()
        optimizer.step()

    dev_mse = evaluate_dev(dev_clips)
    save_checkpoint_if_best(dev_mse)

## Step 2: Evaluation
    FOR each test clip (Hindi or Marathi):
        features = trained_encoder(mel)
        logits = ctc_head_hi_or_mr(features)   # (1, T, 257)
        text = greedy_ctc_decode(logits, indicconformer_bpe_vocab)
        wer = compute_wer(text, ground_truth)
```
