# Semantic Branch — MuRIL Cross-Modal KD — Pseudocode

## Overview
3 phases: (1) precompute MuRIL embeddings, (2) precompute CTC alignments, (3) train.

---

## Phase 1: Pre-compute MuRIL Embeddings (step0 — run once)

```
# Load MuRIL (Google's Indian language BERT)
muril_tokenizer = AutoTokenizer("google/muril-base-cased")
muril_model     = AutoModel("google/muril-base-cased")
muril_model.eval()    # Frozen — never trained

FOR each clip in [train_clips + dev_clips]:

    text = clip.ground_truth       # e.g., "मीना पढ़ने जाती है"

    # Tokenize
    tokens = muril_tokenizer(text, return_tensors="pt")
    # tokens.input_ids = [CLS, ▁मीना, ▁पढ़ने, ▁जाती, ▁है, SEP]
    #                      ↑                              ↑
    #                   special tokens (will be dropped)

    # Get contextual embeddings
    with no_grad():
        output = muril_model(**tokens)
        embeddings = output.last_hidden_state    # (1, N_tokens, 768)

    # Drop [CLS] (first) and [SEP] (last)
    word_embeddings = embeddings[0, 1:-1, :]     # (N_words, 768)

    # Handle subword tokens: MuRIL may split one word into multiple tokens
    # Example: "पढ़ने" might become ["▁पढ़", "ने"]
    # We need to GROUP subword tokens back into words and average them
    #
    # MuRIL uses "▁" prefix for word-start tokens (like SentencePiece)
    # Token starting with "▁" = start of new word
    # Token WITHOUT "▁" = continuation of previous word
    #
    # Example:
    #   tokens: [▁मीना, ▁पढ़, ने, ▁जाती, ▁है]
    #   word 1: ▁मीना           → mean(emb[0])         = (768,)
    #   word 2: ▁पढ़ + ने       → mean(emb[1], emb[2]) = (768,)
    #   word 3: ▁जाती           → mean(emb[3])         = (768,)
    #   word 4: ▁है             → mean(emb[4])         = (768,)
    #
    # Result: 4 word embeddings, each (768,)

    word_groups = group_subword_tokens_into_words(tokens, word_embeddings)
    # word_groups = [
    #     {"word": "मीना",  "embedding": tensor(768)},
    #     {"word": "पढ़ने",  "embedding": tensor(768)},
    #     {"word": "जाती",  "embedding": tensor(768)},
    #     {"word": "है",    "embedding": tensor(768)},
    # ]

    # Save to disk
    save({
        "word_embeddings": stack([w["embedding"] for w in word_groups]),  # (N_words, 768)
        "words": [w["word"] for w in word_groups],                        # list of strings
        "num_words": len(word_groups),
    }, path=f"muril_embeddings/train/{clip.uid}.pt")

# Time estimate: ~500 clips/min on CPU (MuRIL is small, text-only)
# Storage: ~2KB per clip × 65K clips ≈ 130 MB total
```

---

## Phase 2: Pre-compute CTC Alignments (step1 — run once)

```
# Load our EXISTING trained acoustic model (from acoustic branch)
acoustic_encoder = load_whisper_encoder("checkpoints/joint/joint_best_wer.pt")
ctc_head = load_ctc_head("checkpoints/joint/joint_best_wer.pt")  # Linear(768→85)
acoustic_encoder.eval()    # Frozen — just using it for alignment
ctc_head.eval()

char_to_idx = load("ASER-Dataset/vocab.json")    # 85 chars
idx_to_char = reverse(char_to_idx)

FOR each clip in [train_clips + dev_clips]:

    # Step A: Get frame-level CTC predictions
    wav = load_audio(clip.audio_path)
    mel = whisper_feature_extractor(wav)

    with no_grad():
        features = acoustic_encoder(mel).last_hidden_state   # (1, T, 768)
        logits = ctc_head(features)                           # (1, T, 85)
        frame_predictions = argmax(logits, dim=-1)            # (1, T)
        # frame_predictions[t] = index of most likely character at frame t
        # Example: [0, 0, 0, 12, 12, 12, 34, 34, 0, 0, 56, 56, ...]
        #           blank    म        ी       blank  न

    # Step B: CTC collapse (remove blanks and repeated)
    # Convert frame-level predictions to character segments with boundaries
    segments = []
    prev_idx = -1
    seg_start = 0

    for t in range(T):
        idx = frame_predictions[t]
        if idx != prev_idx:
            if prev_idx != BLANK and prev_idx != -1:
                segments.append({
                    "char": idx_to_char[prev_idx],
                    "start_frame": seg_start,
                    "end_frame": t - 1,
                })
            seg_start = t
            prev_idx = idx

    # segments = [
    #   {"char": "म", "start_frame": 3, "end_frame": 5},
    #   {"char": "ी", "start_frame": 6, "end_frame": 7},
    #   {"char": "न", "start_frame": 10, "end_frame": 11},
    #   {"char": "ा", "start_frame": 12, "end_frame": 14},
    #   ...
    # ]

    # Step C: Group characters into words
    # Ground truth: "मीना पढ़ने जाती है"
    # Words: ["मीना", "पढ़ने", "जाती", "है"]
    #
    # Match CTC character segments to ground truth words sequentially:
    #   "म"+"ी"+"न"+"ा" → "मीना" → frames 3-14
    #   "प"+"ढ़"+"न"+"े" → "पढ़ने" → frames 20-45
    #   ...
    #
    # Use greedy matching: consume CTC chars left-to-right, matching
    # against ground truth words left-to-right.

    gt_words = clip.ground_truth.split()
    word_boundaries = align_chars_to_words(segments, gt_words)
    # word_boundaries = [
    #   {"word": "मीना",  "start_frame": 3,   "end_frame": 14},
    #   {"word": "पढ़ने",  "start_frame": 20,  "end_frame": 45},
    #   {"word": "जाती",  "start_frame": 50,  "end_frame": 78},
    #   {"word": "है",    "start_frame": 82,  "end_frame": 90},
    # ]

    # Some words may not align (CTC made errors) → skip those words
    # Only keep words where CTC output matches ground truth characters

    save({
        "word_boundaries": word_boundaries,
        "num_frames": T,
        "num_words_aligned": len(word_boundaries),
        "num_words_total": len(gt_words),
    }, path=f"ctc_alignments/train/{clip.uid}.pt")

# Time estimate: ~100 clips/min on GPU
# Storage: ~1KB per clip × 65K clips ≈ 65 MB total
```

---

## Phase 3: Training (step2 — main training loop)

```
# ═══════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════

# Load student encoder (warm start from MSE distillation)
encoder = WhisperSmall.encoder()
encoder.load_weights("checkpoints/semantic_mse/best_dev.pt" → encoder_state_dict)
# Why warm start? The encoder already has some Indian language knowledge
# from IndicConformer distillation. We build on top of it.

# Freeze bottom 6 layers (preserve low-level acoustics)
for layer in encoder.layers[0:6]:
    layer.freeze()
# Top 6 layers are trainable (adapt for semantic features)

# CTC Head 1 — for decoding (warm start from sequential CTC)
ctc_head = Linear(768, 85)
ctc_head.load_weights("checkpoints/semantic_ctc/best_wer.pt" → ctc_head_state_dict)
# OR random init if no sequential checkpoint available

# Dropout
head_dropout = Dropout(0.1)

# Optimizer — only trainable params
trainable_params = [
    encoder.layers[6:12].parameters(),    # top 6 encoder layers
    ctc_head.parameters(),                 # CTC head
]
optimizer = AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
scheduler = LinearWarmup(500) + CosineDecay

ctc_loss_fn = CTCLoss(blank=0, zero_infinity=True)

# Alpha schedule for MSE weight
alpha_start = 0.3
# Decays: epoch 1-10 → 0.3, epoch 11-20 → 0.2, epoch 21-30 → 0.1


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

best_dev_wer = infinity
patience_counter = 0

FOR epoch in 1..30:

    # Update alpha
    if epoch <= 10:   alpha = 0.3
    elif epoch <= 20: alpha = 0.2
    else:             alpha = 0.1

    shuffle(train_clips)

    FOR each clip in train_clips:

        # ── 1. Load pre-computed data ──
        muril_data = load(f"muril_embeddings/train/{clip.uid}.pt")
        # muril_data["word_embeddings"] = (N_words, 768)
        # muril_data["words"] = ["मीना", "पढ़ने", "जाती", "है"]

        align_data = load(f"ctc_alignments/train/{clip.uid}.pt")
        # align_data["word_boundaries"] = [
        #   {"word": "मीना", "start_frame": 3, "end_frame": 14},
        #   ...
        # ]

        # Skip if alignment failed (too few words matched)
        if align_data["num_words_aligned"] < 2:
            skip; continue

        # ── 2. Forward through student encoder ──
        wav = load_audio(clip.audio_path)
        mel = whisper_feature_extractor(wav)
        mel = spec_augment(mel)
        encoder_output = encoder(mel)         # (1, T, 768)
        real_frames = compute_real_frames(wav)
        features = encoder_output[:, :real_frames, :]  # (1, T, 768)

        # ── 3. Compute word-level speech embeddings ──
        speech_word_embs = []
        muril_word_embs = []

        for i, boundary in enumerate(align_data["word_boundaries"]):
            start = boundary["start_frame"]
            end   = boundary["end_frame"]
            word  = boundary["word"]

            # Find matching MuRIL word
            # (match by index — both come from same ground truth)
            if i >= muril_data["num_words"]:
                break

            # Average student frames for this word
            word_speech_emb = mean(features[0, start:end+1, :])  # (768,)
            word_muril_emb  = muril_data["word_embeddings"][i]    # (768,)

            speech_word_embs.append(word_speech_emb)
            muril_word_embs.append(word_muril_emb)

        # ── 4. MSE loss (speech vs MuRIL, word-level) ──
        if len(speech_word_embs) >= 2:
            speech_stack = stack(speech_word_embs)    # (N_aligned, 768)
            muril_stack  = stack(muril_word_embs)     # (N_aligned, 768)
            mse_loss = MSE(speech_stack, muril_stack)
        else:
            mse_loss = None

        # ── 5. CTC loss (standard character-level decoding) ──
        char_logits = ctc_head(head_dropout(features))  # (1, T, 85)
        log_probs = log_softmax(char_logits).permute(1, 0, 2)  # (T, 1, 85)

        gt_text = normalize_text(clip.ground_truth)
        targets = text_to_indices(gt_text, char_vocab)

        if targets and real_frames > len(targets):
            ctc_loss = CTC(log_probs, targets, [real_frames], [len(targets)])
        else:
            ctc_loss = None

        # ── 6. Combined loss ──
        if ctc_loss is not None and mse_loss is not None:
            total_loss = alpha * mse_loss + (1 - alpha) * ctc_loss
        elif ctc_loss is not None:
            total_loss = ctc_loss
        elif mse_loss is not None:
            total_loss = mse_loss
        else:
            skip; continue

        # ── 7. Backward + update ──
        total_loss.backward() / grad_accum
        if (step + 1) % grad_accum == 0:
            clip_grad_norm(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # ═══════════════════════════════════════════════════════════
    # DEV EVALUATION — WER from CTC Head 1
    # ═══════════════════════════════════════════════════════════

    encoder.eval()
    ctc_head.eval()

    with no_grad():
        FOR each clip in dev_clips:
            wav = load_audio(clip.audio_path)
            mel = whisper_feature_extractor(wav)     # NO SpecAugment
            features = encoder(mel).last_hidden_state[:, :real_frames]

            char_logits = ctc_head(features)          # (1, T, 85)
            pred_text = ctc_greedy_decode(char_logits, idx_to_char)

            refs.append(clip.ground_truth)
            hyps.append(pred_text)

    dev_wer = compute_corpus_wer(refs, hyps)
    per_lang_wer = {Hindi, Marathi, English}

    # Early stopping on dev WER
    if dev_wer < best_dev_wer:
        best_dev_wer = dev_wer
        patience_counter = 0
        save_checkpoint("checkpoints/semantic_muril/best_wer.pt")
    else:
        patience_counter += 1
        if patience_counter >= 7:
            EARLY STOP

    encoder.train()
    ctc_head.train()
```

---

## Key Matching Between MuRIL Words and CTC-Aligned Words

This is the trickiest part. MuRIL tokenizes text differently than our CTC vocab.

```
Ground truth: "मीना पढ़ने जाती है"

MuRIL tokens:     [▁मीना, ▁पढ़, ने, ▁जाती, ▁है]  → grouped: [मीना, पढ़ने, जाती, है]
CTC characters:   [म, ी, न, ा, _, प, ढ़, न, े, ...] → grouped: [मीना, पढ़ने, जाती, है]
Ground truth words: [मीना, पढ़ने, जाती, है]

All three produce the SAME word list (from the same ground truth).
So we match by word INDEX: word[0] in MuRIL = word[0] in CTC alignment.

If CTC alignment misses a word (recognition error), we skip that word's
MSE loss. We don't force bad alignments.
```

---

## What the Encoder Learns (Compared)

```
BEFORE (IndicConformer MSE, current):
    Frame 15 → "this 40ms sounds like BPE token ▁गाय with 78% probability"
    Frame 16 → "this 40ms sounds like BPE token ▁गाय with 82% probability"
    → Per-frame, local, phonological

AFTER (MuRIL cross-modal, proposed):
    Frames 10-25 (word "गाय") → averaged embedding should be close to
        MuRIL's embedding of "गाय" which encodes:
        - "गाय" means "cow" (semantic)
        - it's the subject of this sentence (syntactic)
        - in context of "दूध देती है" = gives milk (contextual)
    → Per-word, global context, semantic meaning
```

---

## Comparison: Both Semantic Approaches

| | IndicConformer (02_mse) | MuRIL (05_cross_modal) |
|---|---|---|
| Pre-compute step | Teacher logits (10 hrs GPU) | MuRIL embeddings (30 min CPU) + CTC alignments (2 hrs GPU) |
| Training loss | MSE on 257-dim logits per frame | MSE on 768-dim embeddings per word |
| Languages | Hindi + Marathi only | Hindi + Marathi + English |
| Bridge heads needed | 2 × Linear(768→257) | None (both 768-dim) |
| Teacher cost at training | Zero (pre-computed) | Zero (pre-computed) |
| Alignment method | F.interpolate (time stretching) | CTC-based word boundaries |
| Standalone WER | ~48% (Hi+Mr) | ~45-55% (Hi+Mr+En) |
| Feature type | Phonological | Semantic/contextual |
| Complementary to acoustic? | Partially | Fully |
