# Acoustic Sequential CTC - Pseudocode

```
INIT:
    encoder  = load_frozen_encoder("checkpoints/acoustic/hpc_best_dev_model.pt")
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    ctc_head  = Linear(768, 85)     # random init, TRAINABLE
    vocab     = load("ASER-Dataset/vocab.json")  # 85 tokens
    optimizer = AdamW(ctc_head.params, lr=1e-3)
    ctc_loss  = CTCLoss(blank=0, zero_infinity=True)

FOR each epoch in 1..30:
    FOR each clip in train_set:     # all languages
        wav = load_audio(clip)
        mel = extract_mel(wav)

        # Frozen encoder - no gradients computed
        with no_grad():
            features = encoder(mel).last_hidden_state  # (1, 1500, 768)

        # CTC head - gradients flow here
        logits = ctc_head(features)     # (1, 1500, 85)

        # CTC loss vs ground truth text
        targets = text_to_indices(clip.transcript, vocab)
        real_frames = compute_real_frames(wav)
        log_probs = logits[:, :real_frames].log_softmax()

        loss = ctc_loss(log_probs, targets, [real_frames], [len(targets)])
        loss.backward()     # gradients flow to ctc_head ONLY
        optimizer.step()

    # Dev evaluation
    dev_wer = greedy_decode_and_compute_wer(dev_clips)
    save_checkpoint_if_best(dev_wer)
    early_stop_if_no_improvement(patience=5)
```
