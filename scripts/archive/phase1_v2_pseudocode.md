# Phase 1 v2: Dual-Teacher Knowledge Distillation — Pseudocode
# Architecture: Whisper Small student, IndicConformer semantic teacher, Kid-Whisper acoustic teacher

## What Changed from v1

| | Phase 1 v1 (old) | Phase 1 v2 (new) |
|---|---|---|
| **Student** | MMS-300M (encoder only) | Whisper Small (encoder + decoder) |
| **Semantic Teacher** | XLSR-53 (encoder only) | IndicConformer 600M (full ASR model) |
| **Acoustic Teacher** | Kid-Whisper Small (encoder) | Kid-Whisper Small (encoder) |
| **Semantic Loss** | MSE on embedding vectors | Cross-entropy on text (sequence-level distillation) |
| **Acoustic Loss** | MSE on last hidden layer | MSE on configurable intermediate layer |
| **Semantic target** | Avg-pooled embedding (1, 1024) | Text transcript (token sequence) |
| **After Phase 1** | Just embeddings, cannot evaluate ASR | Working ASR model, can compute WER |

---

## High-Level Flow

```
PARSE command-line args:
    --verify          : quick test with small subset (default: False)
    --verify_n        : number of clips for verify mode (default: 50)
    --epochs          : number of training epochs (default: 20)
    --patience        : early stopping patience (default: 5)
    --seed            : random seed (default: 42)
    --lr              : learning rate (default: 1e-5, smaller than v1 because Whisper)
    --acoustic_layer  : which encoder layer for acoustic MSE (default: 8)
    --alpha           : weight for semantic loss (default: 0.5)
    --beta            : weight for acoustic loss (default: 0.5)
    --precompute      : pre-generate teacher transcripts offline (default: True)
    --max_duration    : max clip duration in seconds (default: 30)

STEP 0: PRE-GENERATE SEMANTIC TARGETS (run once, save to disk)
    Load IndicConformer 600M
    For each clip in asr_train.csv + asr_dev.csv:
        audio → IndicConformer → transcript text
        Save to teacher_transcripts.csv
    Unload IndicConformer (free GPU memory)
    WHY: IndicConformer uses NeMo framework. Loading it during training
         alongside Whisper models wastes GPU. Generate all transcripts
         once, save as text file, use during training.

STEP 1: LOAD DATA
    Read asr_train.csv → training clips
    Read asr_dev.csv → dev clips
    Read teacher_transcripts.csv → teacher transcripts (from Step 0)
    Filter: keep clips ≤ max_duration seconds
    IF --verify: sample N clips proportionally by language, set epochs=3

STEP 2: LOAD 2 MODELS (not 3 — IndicConformer already done in Step 0)
    Teacher B  = Kid-Whisper Small (244M params, FROZEN, float16)
                 Model: "aadel4/kid-whisper-small-myst"
                 Load: WhisperModel.from_pretrained() → extract .encoder
                 Purpose: acoustic teacher, provides frame-level targets
                 
    Student C  = Whisper Small (244M params, TRAINS, float32)
                 Model: "openai/whisper-small"
                 Load: WhisperForConditionalGeneration.from_pretrained()
                 Purpose: learns from both teachers
                 Note: Load FULL model (encoder + decoder) because
                       semantic branch needs decoder for text generation

    Tokenizer  = WhisperTokenizer.from_pretrained("openai/whisper-small")
                 Purpose: convert teacher transcripts → token IDs for cross-entropy

    Processor  = WhisperProcessor.from_pretrained("openai/whisper-small")
                 Purpose: convert audio → mel spectrogram for both Whisper models

STEP 3: VERIFY COMPONENTS (one test clip)
    Pick one Hindi clip
    
    3a. Verify acoustic branch:
        audio → mel spectrogram → Kid-Whisper encoder (frozen)
            → output_hidden_states=True
            → extract layer [acoustic_layer] → Y_kid (1, 1500, 768)
            → apply attention_mask → Y_kid (1, T_valid, 768)
        audio → mel spectrogram → Student encoder
            → output_hidden_states=True  
            → extract layer [acoustic_layer] → Y_ac_hat (1, 1500, 768)
            → apply attention_mask → Y_ac_hat (1, T_valid, 768)
        L_acoustic = MSE(Y_ac_hat, Y_kid)
        Check: Kid-Whisper has NO gradients, Student HAS gradients
        
    3b. Verify semantic branch:
        teacher_text = teacher_transcripts[clip_id]  (pre-generated)
        target_ids = tokenizer.encode(teacher_text)  → (N_tokens,)
        audio → mel spectrogram → Student full model (labels=target_ids)
            → output.loss = cross-entropy (computed automatically by HuggingFace)
        L_semantic = output.loss
        Check: loss is a scalar, backward() works
        
    3c. Verify total loss:
        L_total = alpha * L_semantic + beta * L_acoustic
        L_total.backward()
        Check: student parameters have gradients
        Print peak GPU memory

STEP 4: SETUP
    Optimizer = AdamW(student.parameters(), lr=1e-5, weight_decay=0.01)
    Scheduler = linear warmup (optional, 500 steps warmup)
    Set random seeds for reproducibility

STEP 5: TRAINING LOOP
    FOR each epoch (1 to num_epochs):
        student.train()
        Shuffle training clips
        Initialize: running_sem_loss=0, running_ac_loss=0, running_total=0
        
        FOR each clip in training_clips:
            
            ─── 5a. LOAD AUDIO ───
            audio_np = load_audio(clip.audio_path, max_sec=30)
            INPUT:  file path (string)
            OUTPUT: numpy array, shape (num_samples,), dtype float32
                    e.g., 10s clip → (160000,)
            HOW:    torchaudio.load() → resample to 16kHz → mono → numpy
            
            ─── 5b. PREPARE MEL SPECTROGRAM ───
            mel_input = processor(audio_np, sampling_rate=16000,
                                  return_tensors="pt",
                                  return_attention_mask=True)
            INPUT:  numpy array (num_samples,)
            OUTPUT: mel_input.input_features  → (1, 80, 3000)  mel spectrogram
                    mel_input.attention_mask   → (1, 3000)      1=real, 0=padding
            WHY:    Whisper ALWAYS pads audio to 30 seconds before computing mel.
                    A 10s clip gets 20s of silence padding.
                    attention_mask tells us which frames are real vs padding.
            NOTE:   This mel is used for BOTH teacher and student (same input)
            
            ─── 5c. GET ACOUSTIC TEACHER TARGET ───
            with torch.no_grad():                    ← teacher is frozen
                kid_out = kid_whisper_encoder(
                    mel_input.input_features.to(device, dtype=float16),
                    output_hidden_states=True        ← return ALL layer outputs
                )
            
            INPUT:  mel spectrogram (1, 80, 3000)
            OUTPUT: kid_out.hidden_states = tuple of 13 tensors
                    hidden_states[0]  = embedding layer output  (1, 1500, 768)
                    hidden_states[1]  = layer 1 output          (1, 1500, 768)
                    hidden_states[2]  = layer 2 output          (1, 1500, 768)
                    ...
                    hidden_states[12] = layer 12 output         (1, 1500, 768)
            
            # Select the configured acoustic layer
            Y_kid_full = kid_out.hidden_states[acoustic_layer]  → (1, 1500, 768)
            
            # Remove padding frames using attention mask
            enc_mask = mel_input.attention_mask[:, ::2]   → (1, 1500)
            WHY ::2: mel has 3000 frames, encoder has 1500 (stride-2 conv in Whisper)
            
            valid_frames = int(enc_mask.sum().item())
            Y_kid = Y_kid_full[:, :valid_frames, :].float()   → (1, T_valid, 768)
            
            EXAMPLE: 10s clip → T_valid = 500 frames
                     Y_kid shape = (1, 500, 768)
            
            ─── 5d. GET SEMANTIC TEACHER TARGET ───
            teacher_text = teacher_transcripts[clip.id]
            target_ids = tokenizer.encode(teacher_text)
            target_ids = torch.tensor([target_ids]).to(device)  → (1, N_tokens)
            
            INPUT:  teacher text string, e.g., "राधा के पास एक तोता है"
            OUTPUT: target_ids, e.g., [50258, 2345, 892, 1456, ..., 50257]
                    50258 = <|startoftranscript|> (Whisper special token)
                    50257 = <|endoftext|> (Whisper special token)
                    Middle tokens = the actual text in Whisper's BPE vocabulary
            N_tokens: varies by transcript length, typically 5-50 tokens
            
            WHY NOT run IndicConformer here:
                1. IndicConformer uses NeMo framework (different from HuggingFace)
                2. Loading it alongside Whisper wastes GPU memory
                3. Teacher output (text) is deterministic — same audio always → same text
                4. So we pre-generate all transcripts ONCE in Step 0
            
            ─── 5e. GET STUDENT OUTPUTS (BOTH BRANCHES) ───
            
            # SEMANTIC BRANCH: full model forward pass with labels
            student_out = student_model(
                input_features=mel_input.input_features.to(device),
                labels=target_ids,                    ← teacher's transcript as target
                output_hidden_states=True             ← need encoder hidden states too
            )
            
            INPUT:  mel spectrogram (1, 80, 3000) + target token IDs (1, N_tokens)
            OUTPUT: student_out.loss           → scalar (cross-entropy, auto-computed)
                    student_out.logits         → (1, N_tokens, vocab_size=51865)
                    student_out.encoder_hidden_states → tuple of 13 tensors
            
            HOW THE LOSS IS COMPUTED (inside HuggingFace automatically):
                At each token position, the model predicts a probability 
                distribution over 51,865 possible tokens.
                
                Position 1: model predicts [0.01, 0.003, ..., 0.15, ...]
                            target says token should be "राधा" (token ID 2345)
                            loss += -log(probability of token 2345)
                
                Position 2: model predicts [0.02, 0.008, ..., 0.22, ...]
                            target says token should be "के" (token ID 892)
                            loss += -log(probability of token 892)
                
                ... for all N_tokens positions
                
                Final cross-entropy = average of all position losses
            
            L_semantic = student_out.loss                → scalar
            
            # ACOUSTIC BRANCH: extract student encoder intermediate layer
            Y_ac_hat_full = student_out.encoder_hidden_states[acoustic_layer]
                                                          → (1, 1500, 768)
            Y_ac_hat = Y_ac_hat_full[:, :valid_frames, :] → (1, T_valid, 768)
            
            EXAMPLE: 10s clip → T_valid = 500
                     Y_ac_hat shape = (1, 500, 768)
                     Y_kid shape    = (1, 500, 768)  ← same! can compare directly
            
            ─── 5f. COMPUTE ACOUSTIC LOSS ───
            L_acoustic = MSE(Y_ac_hat, Y_kid)           → scalar
            
            INPUT:  Y_ac_hat (1, T_valid, 768) — student's layer N output
                    Y_kid    (1, T_valid, 768) — teacher's layer N output
            OUTPUT: single number — average squared difference across all values
            
            EXAMPLE:
                Y_ac_hat[0][0] = [0.12, 0.87, -0.34, ...]  ← student's frame 1
                Y_kid[0][0]    = [0.15, 0.82, -0.31, ...]  ← teacher's frame 1
                MSE = mean of (0.12-0.15)² + (0.87-0.82)² + (-0.34-(-0.31))² + ...
                
            WHY SAME LAYER from both models:
                Layer N in teacher captures certain acoustic features.
                We want the student's layer N to capture the SAME features.
                Comparing same-layer outputs is the most natural alignment.
            
            ─── 5g. COMPUTE TOTAL LOSS ───
            L_total = alpha * L_semantic + beta * L_acoustic
            
            DEFAULT: alpha=0.5, beta=0.5 → equal weight to both branches
            
            WHY WEIGHTS:
                Cross-entropy (semantic) and MSE (acoustic) have different scales.
                alpha/beta let us balance them.
                We may need to tune these based on initial experiments.
                If semantic loss is ~5.0 and acoustic is ~0.5, they're on different
                scales. Weights help normalize.
            
            ─── 5h. BACKWARD + UPDATE ───
            optimizer.zero_grad()
            L_total.backward()
            
            WHAT GETS UPDATED:
                Student encoder weights  — YES (acoustic branch gradients flow here)
                Student decoder weights  — YES (semantic branch gradients flow here)
                Kid-Whisper weights      — NO  (frozen, no gradients)
                IndicConformer weights   — N/A (not even loaded)
            
            clip_grad_norm_(student.parameters(), max_norm=1.0)
            WHY: prevent exploding gradients — if any gradient is too large,
                 scale all gradients down proportionally
            
            optimizer.step()
            
            ─── 5i. LOGGING ───
            running_sem_loss += L_semantic.item()
            running_ac_loss += L_acoustic.item()
            running_total += L_total.item()
            
            # Print every 100 clips
            IF clip_num % 100 == 0:
                PRINT f"Clip {clip_num}/{total}: sem={avg_sem:.4f} ac={avg_ac:.4f}"
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        END FOR (clips)
        
        ─── 5j. EPOCH SUMMARY ───
        avg_train_sem = running_sem_loss / num_clips
        avg_train_ac  = running_ac_loss / num_clips
        avg_train_total = running_total / num_clips
        PRINT epoch summary table
        
        ─── 5k. DEV EVALUATION ───
        student.eval()
        
        # PART 1: Dev Loss (same as training but no gradient updates)
        with torch.no_grad():
            FOR each clip in dev_clips:
                Same as 5b-5f but no backward()
                Accumulate dev_sem_loss, dev_ac_loss
            
            avg_dev_sem = dev_sem_loss / num_dev_clips
            avg_dev_ac  = dev_ac_loss / num_dev_clips
            avg_dev_total = avg_dev_sem + avg_dev_ac
        
        # PART 2: WER Evaluation (NEW in v2 — we can now evaluate ASR!)
        wer_scores = []
        with torch.no_grad():
            FOR each clip in dev_clips (or a subset):
                audio → mel → student.generate() → predicted_text
                ground_truth = clip.transcript
                wer = compute_wer(predicted_text, ground_truth)
                wer_scores.append(wer)
        
        avg_wer = mean(wer_scores)
        PRINT f"Dev WER: {avg_wer:.2f}%"
        PRINT f"Baseline WER comparison: vanilla Whisper Small = {baseline_wer}%"
        
        WHY WER IS POSSIBLE NOW:
            In v1, student was encoder-only (MMS-300M) → could not generate text
            In v2, student is full Whisper (encoder+decoder) → can generate text
            → We can compare with baseline Whisper Small immediately!
        
        HOW student.generate() WORKS:
            INPUT:  mel spectrogram (1, 80, 3000)
            PROCESS: encoder produces embeddings
                     decoder generates tokens one by one (autoregressive)
                     each token is chosen by: argmax over vocabulary probabilities
            OUTPUT: list of token IDs → decode to text string
            EXAMPLE: [50258, 2345, 892, ...] → "राधा के पास एक तोता है"
        
        ─── 5l. EARLY STOPPING ───
        IF avg_dev_total < best_dev_loss:
            best_dev_loss = avg_dev_total
            patience_counter = 0
            Save best_dev_model.pt
        ELSE:
            patience_counter += 1
            IF patience_counter >= patience:
                PRINT "Early stopping triggered"
                BREAK
        
        ─── 5m. SAVE CHECKPOINTS ───
        Save: checkpoints/phase1_v2_best_dev.pt     (lowest dev loss)
        Save: checkpoints/phase1_v2_best_train.pt   (lowest train loss)
        Save: checkpoints/phase1_v2_best_wer.pt     (lowest dev WER)
        Save: checkpoints/phase1_v2_epoch_N.pt       (every epoch)
        
        WHAT IS SAVED:
            student.state_dict()      — all model weights
            optimizer.state_dict()    — optimizer momentum/variance
            epoch number
            best losses
            best WER
    
    END FOR (epochs)

STEP 6: FINAL EVALUATION
    Load best_wer checkpoint
    Run on full dev set → compute WER per language
    
    PRINT results table:
        | Language | Baseline WER | Phase 1 WER | Improvement |
        |----------|-------------|-------------|-------------|
        | Hindi    | X%          | Y%          | (X-Y)%      |
        | Marathi  | X%          | Y%          | (X-Y)%      |
        | English  | X%          | Y%          | (X-Y)%      |
    
    PRINT loss summary:
        Epoch-by-epoch: train_loss, dev_loss, semantic, acoustic, WER
        Overall reduction percentages

STEP 7: COMPUTE BASELINE (can be run separately before training)
    Load vanilla Whisper Small ("openai/whisper-small")
    FOR each clip in dev_clips:
        audio → mel → whisper.generate() → predicted_text
        wer = compute_wer(predicted_text, clip.transcript)
    PRINT "Baseline Whisper Small WER: {avg_wer}%"
    Save baseline WER for comparison in Step 6
```

---

## Key Functions — Detailed Input/Output

| # | Function | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | `load_audio(path, max_sec)` | file path (string) | numpy array (num_samples,) float32 | Load audio, resample to 16kHz, convert to mono |
| 2 | `prepare_mel(audio_np)` | numpy array (num_samples,) | input_features (1,80,3000), attention_mask (1,3000) | Convert audio to mel spectrogram + padding mask |
| 3 | `get_acoustic_target(mel, mask, layer)` | mel features, mask, layer number | Y_kid (1, T_valid, 768) | Run frozen Kid-Whisper, extract layer N, remove padding |
| 4 | `get_student_output(mel, target_ids, layer)` | mel features, token IDs, layer number | (L_semantic, Y_ac_hat of shape (1, T_valid, 768)) | Run student, get CE loss + encoder layer N output |
| 5 | `compute_wer(predicted, reference)` | two strings | float (0.0 to 1.0) | Word Error Rate between predicted and reference text |
| 6 | `evaluate_dev(dev_clips, transcripts)` | clip list, teacher transcripts | (avg_loss, avg_sem, avg_ac, avg_wer) | Full dev evaluation with loss + WER |
| 7 | `precompute_transcripts(clips, model)` | clip list, IndicConformer model | dict {clip_id: transcript} | Generate teacher transcripts for all clips |
| 8 | `compute_baseline_wer(dev_clips)` | clip list | float (avg WER) | Run vanilla Whisper Small on dev set |

---

## Key Dimensions — Every Tensor Explained

```
AUDIO:
  Raw audio:              (num_samples,)          e.g., 10s = (160000,)
  
MEL SPECTROGRAM:
  input_features:         (1, 80, 3000)           always padded to 30s
  attention_mask:          (1, 3000)               1=real audio, 0=padding
  WHY 80:  80 mel frequency bins
  WHY 3000: 30 seconds × 100 frames/second = 3000 mel frames

ENCODER (both Kid-Whisper and Student Whisper):
  Each layer output:      (1, 1500, 768)          1500 = 3000/2 (stride-2 conv)
  After removing padding: (1, T_valid, 768)       T_valid = real speech frames
  
  WHY 1500: Whisper encoder has a stride-2 conv1d as first layer
            3000 mel frames → 1500 encoder frames
  
  WHY 768:  Whisper Small hidden dimension = 768
            Every frame is represented by 768 numbers
  
  EXAMPLE (10 second clip):
    mel frames:     10s × 100 = 1000 real + 2000 padding = 3000
    encoder frames: 1000/2 = 500 real + 1000 padding = 1500
    After masking:  (1, 500, 768) → only real speech frames kept
  
  hidden_states tuple:
    hidden_states[0]  = conv embedding output     (1, 1500, 768)
    hidden_states[1]  = transformer layer 1 out   (1, 1500, 768)
    hidden_states[2]  = transformer layer 2 out   (1, 1500, 768)
    ...
    hidden_states[12] = transformer layer 12 out  (1, 1500, 768)  = last_hidden_state

DECODER:
  input:  target_ids      (1, N_tokens)           N_tokens = transcript length
  output: logits           (1, N_tokens, 51865)   probability over vocabulary
  
  WHY 51865: Whisper vocabulary size
             Each position predicts one of 51,865 possible tokens
  
  Cross-entropy loss:  scalar (single number)
    = average of -log(P(correct_token)) at each position

TEACHER TRANSCRIPT (pre-generated):
  Just a text string, e.g., "राधा के पास एक तोता है"
  Tokenized: [50258, 2345, 892, 1456, 3201, 1589, 445, 50257]
  Size: typically 5-50 tokens per clip
```

---

## Model Comparison: What's on GPU During Training

```
STEP 0 (pre-compute, done separately):
  IndicConformer 600M (float16):  ~1200 MB  ← loaded alone, then unloaded
  
STEP 5 (training):
  Kid-Whisper encoder (float16):  ~244 MB   ← frozen teacher
  Student Whisper Small (float32): ~976 MB   ← trains (encoder + decoder)
  Optimizer states (AdamW):       ~1952 MB   ← 2x student size
  Activations + gradients:        ~2000 MB   ← intermediate computation
  ──────────────────────────────────────────
  Total peak:                     ~5172 MB / 7932 MB available
  Headroom:                       ~2760 MB ✅ (comfortable fit on RTX 4060)
  
  WHY SO MUCH LESS THAN v1:
    v1: MMS-300M (1200MB) + XLSR-53 (602MB) + Kid-Whisper (168MB) = 1970 MB for models
    v2: Whisper Small (976MB) + Kid-Whisper (244MB) = 1220 MB for models
    Plus: no need for IndicConformer during training (pre-computed!)
```

---

## Step 0 Detail: Pre-Computing Teacher Transcripts

```python
# This runs ONCE before training starts. Can be a separate script.
# Output: teacher_transcripts.csv with columns [clip_id, teacher_transcript]

FUNCTION precompute_transcripts(csv_path, output_path):
    
    # Load IndicConformer (NeMo framework)
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual"
    )
    model.eval()
    model.to(device)
    
    # Read all clips
    clips = read_csv(csv_path)  # asr_train.csv + asr_dev.csv
    
    results = []
    FOR each clip in clips:
        # NeMo transcription
        transcript = model.transcribe([clip.audio_path])
        # transcript = "राधा के पास एक तोता है"
        
        results.append({
            "clip_id": clip.id,
            "audio_path": clip.audio_path,
            "teacher_transcript": transcript[0],
            "ground_truth": clip.transcript,    # for comparison
            "language": clip.language
        })
    
    # Save to CSV
    save_csv(results, output_path)
    # output_path = "ASER-Dataset/teacher_transcripts.csv"
    
    # Unload model
    del model
    torch.cuda.empty_cache()
    
    PRINT stats:
        Total clips processed: N
        Per language: Hindi=X, Marathi=Y, English=Z
        Sample transcripts (first 5)

    INPUT:  asr_train.csv + asr_dev.csv paths
    OUTPUT: teacher_transcripts.csv file on disk
    TIME:   ~30-60 minutes (one-time cost)
    GPU:    ~1200 MB (IndicConformer only, then freed)
```

---

## Acoustic Layer Experiment — Finding Best Layer

```
BEFORE full training, run a small experiment:

FOR layer_num in [4, 6, 8, 10, 12]:
    Train for 3 epochs on 200 clips with acoustic_layer=layer_num
    Record: dev acoustic loss, dev WER
    
PRINT results:
    | Layer | Dev Acoustic Loss | Dev WER |
    |-------|-------------------|---------|
    | 4     | 0.XX              | XX%     |
    | 6     | 0.XX              | XX%     |
    | 8     | 0.XX              | XX%     |
    | 10    | 0.XX              | XX%     |
    | 12    | 0.XX              | XX%     |

Pick the layer with lowest dev WER for full training.

Command: python scripts/phase1_v2.py --verify --verify_n 200 --epochs 3 --acoustic_layer 8
```

---

## WER Computation Detail

```python
FUNCTION compute_wer(predicted_text, reference_text):
    """
    Word Error Rate = (Substitutions + Insertions + Deletions) / Reference Words
    
    EXAMPLE:
      Reference:  "राधा के पास एक तोता है"     → 6 words
      Predicted:  "राधा के पस एक तोता"          → 5 words
      
      Substitutions: 1 ("पास" → "पस")
      Deletions:     1 ("है" missing)
      Insertions:    0
      
      WER = (1 + 0 + 1) / 6 = 33.3%
    """
    
    # Use jiwer library (standard for ASR evaluation)
    import jiwer
    
    # Normalize: lowercase, remove punctuation
    pred = normalize_text(predicted_text)
    ref  = normalize_text(reference_text)
    
    wer = jiwer.wer(ref, pred)
    
    INPUT:  two text strings
    OUTPUT: float between 0.0 and 1.0+ (can exceed 1.0 if many insertions)
```

---

## Complete Training Example — One Clip Walkthrough

```
CLIP: HI_S1_P_0.wav (13.66 seconds, Hindi)
GROUND TRUTH: "राधा के पास एक तोता है उसकी चोंच लाल है वह बहुत बोलता है सब को हँसाता है"

─── Step 5a: LOAD AUDIO ───
audio_np = load_audio("HI_S1_P_0.wav")
→ numpy array, shape (218560,)    [13.66s × 16000 samples/s]

─── Step 5b: PREPARE MEL ───
mel_input = processor(audio_np, return_attention_mask=True, ...)
→ input_features: (1, 80, 3000)   [padded to 30s]
→ attention_mask:  (1, 3000)       [first 1366 = 1, rest = 0]

─── Step 5c: ACOUSTIC TEACHER ───
kid_out = kid_whisper_encoder(mel, output_hidden_states=True)
→ hidden_states[8]: (1, 1500, 768)  [layer 8 output]
enc_mask = attention_mask[:, ::2]    → (1, 1500)  [first 683 = 1, rest = 0]
valid_frames = 683
Y_kid = hidden_states[8][:, :683, :] → (1, 683, 768)

─── Step 5d: SEMANTIC TEACHER TARGET ───
teacher_text = "राधा के पास एक तोता है उसकी चोंच लाल है वह बहुत बोलता है सब को हँसाता है"
(pre-generated by IndicConformer in Step 0)
target_ids = tokenizer.encode(teacher_text) → (1, 28)  [28 BPE tokens]

─── Step 5e: STUDENT FORWARD PASS ───
student_out = student_model(
    input_features=mel,        # (1, 80, 3000)
    labels=target_ids,         # (1, 28)
    output_hidden_states=True
)

→ student_out.loss = 3.2451   [cross-entropy on 28 token positions]
→ student_out.encoder_hidden_states[8]: (1, 1500, 768)

L_semantic = 3.2451
Y_ac_hat = encoder_hidden_states[8][:, :683, :] → (1, 683, 768)

─── Step 5f: ACOUSTIC LOSS ───
L_acoustic = MSE(Y_ac_hat, Y_kid)
           = MSE between (1, 683, 768) and (1, 683, 768)
           = 0.4521

─── Step 5g: TOTAL LOSS ───
L_total = 0.5 × 3.2451 + 0.5 × 0.4521
        = 1.6226 + 0.2261
        = 1.8486

─── Step 5h: UPDATE ───
optimizer.zero_grad()
L_total.backward()     → gradients flow to student encoder AND decoder
clip_grad_norm_(max=1.0)
optimizer.step()        → student weights updated

NEXT CLIP...
```

---

## File Structure

```
scripts/
├── phase1_v2_precompute.py     ← Step 0: generate teacher transcripts
├── phase1_v2.py                ← Steps 1-7: main training script
├── phase1_v2_pseudocode.md     ← this file
├── phase1_combined.py          ← old v1 (kept for reference)
└── phase1_pseudocode.md        ← old v1 pseudocode

ASER-Dataset/
├── splits/
│   ├── asr_train.csv
│   ├── asr_dev.csv
│   └── asr_test.csv
└── teacher_transcripts.csv     ← generated by phase1_v2_precompute.py

checkpoints/
├── phase1_v2_best_dev.pt
├── phase1_v2_best_wer.pt
├── phase1_v2_best_train.pt
└── phase1_v2_epoch_N.pt
```

---

## Dependencies

```
torch                 ← PyTorch
torchaudio            ← audio loading + resampling
transformers          ← Whisper models + tokenizer + processor
nemo_toolkit[asr]     ← IndicConformer (Step 0 only)
jiwer                 ← WER computation
pandas                ← CSV handling
numpy                 ← array operations
```

---

## Command Reference

```bash
# Step 0: Pre-compute teacher transcripts (run once)
python scripts/phase1_v2_precompute.py

# Quick verify (5 min)
python scripts/phase1_v2.py --verify

# Layer experiment (find best acoustic layer)
for layer in 4 6 8 10 12; do
    python scripts/phase1_v2.py --verify --verify_n 200 --epochs 3 --acoustic_layer $layer
done

# Full training with early stopping
python scripts/phase1_v2.py --epochs 20 --patience 5 --acoustic_layer 8

# Full training with logging
nohup python scripts/phase1_v2.py --epochs 20 --patience 5 \
    2>&1 | tee phase1_v2_training.log &

# Compute baseline WER (for comparison)
python scripts/phase1_v2.py --baseline_only
```
