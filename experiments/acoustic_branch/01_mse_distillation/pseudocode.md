# Acoustic MSE Distillation - Pseudocode

```
INIT:
    student_encoder = WhisperSmall.encoder          # 768-dim
    teacher_encoder = KidWhisperMedium.encoder      # 1024-dim, FROZEN
    projection      = Linear(768, 1024)
    dropout         = Dropout(0.1)
    optimizer       = AdamW([student_encoder.params, projection.params], lr=3e-5)

FOR each epoch in 1..20:
    FOR each clip in train_set:     # all languages
        wav = load_audio(clip)
        mel = extract_mel(wav)

        # Student forward
        student_features = student_encoder(mel)     # (1, 1500, 768)
        projected = projection(dropout(student_features))  # (1, 1500, 1024)

        # Teacher forward (no gradients)
        with no_grad():
            teacher_features = teacher_encoder(mel)  # (1, 1500, 1024)

        # MSE on real frames only
        real_frames = compute_real_frames(wav)
        loss = MSE(projected[:, :real_frames], teacher_features[:, :real_frames])

        loss.backward()
        optimizer.step()

    # Dev evaluation: compute MSE on dev set
    dev_mse = evaluate_dev_mse(dev_clips)
    save_checkpoint_if_best(dev_mse)
```
