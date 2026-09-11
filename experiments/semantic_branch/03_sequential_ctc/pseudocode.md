# Semantic Sequential CTC - Pseudocode

```
INIT:
    encoder = load_frozen_encoder("checkpoints/semantic_mse/best_dev.pt")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False     # FROZEN

    ctc_head = Linear(768, 85)          # random init, TRAINABLE
    vocab    = load("ASER-Dataset/vocab.json")
    optimizer = AdamW(ctc_head.params, lr=1e-3)
    ctc_loss  = CTCLoss(blank=0, zero_infinity=True)

FOR each epoch in 1..30:
    shuffle(train_clips)    # all 3 languages

    FOR each clip in train_set:
        wav = load_audio(clip)
        mel = extract_mel(wav)
        real_frames = compute_real_frames(wav)

        # Frozen encoder - no gradients
        with no_grad():
            features = encoder(mel).last_hidden_state  # (1, 1500, 768)

        # CTC head - trainable
        logits = ctc_head(features)     # (1, 1500, 85)

        # CTC loss vs ground truth
        targets = text_to_indices(clip.transcript, vocab)
        log_probs = logits[:, :real_frames].log_softmax()
        loss = ctc_loss(log_probs, targets, [real_frames], [len(targets)])

        loss.backward()     # only ctc_head gets gradients
        clip_grad_norm(ctc_head, max_norm=5.0)
        optimizer.step()

    # Dev evaluation with per-language WER
    dev_wer, lang_wers = evaluate_dev(dev_clips)
    save_checkpoint_if_best(dev_wer)
    early_stop_if_no_improvement(patience=5)
```
