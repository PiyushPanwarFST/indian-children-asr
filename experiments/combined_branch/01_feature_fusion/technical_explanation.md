# Combined Branch — Deep Technical Explanation

## Table of Contents
1. [How Other Papers Combine Branches](#1-how-other-papers-combine-branches)
2. [How Our Base CARE Paper Does It](#2-how-our-base-care-paper-does-it)
3. [Frame-Level vs Logit-Level Representations](#3-frame-level-vs-logit-level-representations)
4. [What "Acoustic Features" and "Linguistic Features" Actually Mean](#4-what-acoustic-features-and-linguistic-features-actually-mean)
5. [Gated Fusion — Line by Line Execution](#5-gated-fusion--line-by-line-execution)
6. [How the Combined Model Becomes an "Indian Children's Speech" Model](#6-how-the-combined-model-becomes-an-indian-childrens-speech-model)

---

## 1. How Other Papers Combine Branches

In speech and NLP research, there are several standard ways to combine two streams of information:

### Method A: Feature Concatenation (simplest)
```
feat_a (T, 768) + feat_s (T, 768) → concat → (T, 1536) → Linear(1536, 768) → output
```
Used by: Early multi-modal systems, simple ensemble models.
**Problem**: Treats all frames equally. Cannot learn "trust acoustic more for this frame."

### Method B: Late Fusion / Score Averaging
```
Model A → logits_a (T, 85)
Model B → logits_b (T, 85)
Final = 0.5 * logits_a + 0.5 * logits_b
```
Used by: Ensemble ASR systems, ROVER (Recognizer Output Voting Error Reduction).
**Problem**: Fixed 50/50 weighting. Both models must use the same vocabulary.

### Method C: Attention-Based Fusion (Cross-Attention)
```
feat_a → Q (query)
feat_s → K, V (key, value)
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```
Used by: Transformer-based multi-modal models, VisualBERT, LXMERT.
**Problem**: Expensive (O(T^2) computation), complex to train.

### Method D: Gated Fusion (what we use)
```
gate = σ(W @ [feat_a; feat_s] + b)     # learn per-frame weighting
fused = gate * feat_a + (1-gate) * feat_s
```
Used by: Highway Networks, LSTM gates, Gated Multi-Modal Fusion (Arevalo et al. 2017).
**Why we chose this**: Simple, fast, learns per-frame AND per-dimension weighting with minimal parameters. Proven effective in multi-modal fusion tasks.

### Method E: FiLM (Feature-wise Linear Modulation)
```
gamma, beta = MLP(feat_s)
modulated = gamma * feat_a + beta
```
Used by: FiLM (Perez et al. 2018), visual question answering.
One stream modulates another. Asymmetric — one stream is "primary", other is "conditioning."

### Method F: Mixture of Experts (MoE)
```
router = softmax(W @ input)   # which expert to use
output = sum(router[i] * expert_i(input))
```
Used by: Switch Transformer, GShard.
**Problem**: Overkill for 2 streams. Better for many diverse experts.

---

## 2. How Our Base CARE Paper Does It

The CARE paper (Dutta & Ganapathy, IEEE TASLP 2025) is designed for **emotion classification** from children's speech (4 classes: Angry/Happy/Sad/Neutral), NOT for ASR. This is important because their combination strategy is designed for classification, not frame-level prediction.

### CARE's Architecture

CARE uses a SINGLE shared encoder with branch-specific layers:

```
Raw Audio
    |
    v
Shared Encoder (7 common layers)
    |
    +-----> Acoustic Branch (6 layers) ← distilled from PASE+ teacher
    |
    +-----> Semantic Branch (6 layers) ← distilled from RoBERTa teacher
```

Total: 7 shared + 6 acoustic + 6 semantic = 19 layers in one model.

### CARE Phase 1: Branch Training
- Acoustic branch: MSE loss matching PASE+ encoder features (frame-level acoustic representations)
- Semantic branch: MSE loss matching RoBERTa text embeddings (sentence-level semantic representations)
- Both losses train the shared encoder + respective branch layers

### CARE Phase 2: Combination (Learnable Convex Combination)
```
For each of the 13 layer outputs (7 shared + 6 branch):
    weighted_output = sum(alpha_i * layer_i_output)
    where alpha = softmax(learnable_weights)    # 13 learned scalars

pooled = mean_pool(weighted_output)     # collapse time → single vector
emotion = classifier(pooled)            # Linear → 4 classes
```

**Key differences from what we do:**
| Aspect | CARE Paper | Our Project |
|--------|-----------|-------------|
| Task | Emotion classification (4 classes) | ASR (character sequence) |
| Output | Single label per utterance | Frame-level character predictions |
| Combination | Weighted sum of layer outputs | Gated fusion of final encoder outputs |
| Pooling | Mean pool (collapses time) | No pooling (need frame-level for CTC) |
| Encoders | 1 shared encoder with branches | 2 separate encoders |
| Acoustic teacher | PASE+ (acoustic features) | Kid-Whisper (encoder features) |
| Semantic teacher | RoBERTa (text embeddings) | IndicConformer (CTC logits) |

**Why we can't use CARE's exact combination:**
CARE does mean pooling — it collapses 1500 frames into 1 vector. This works for classification ("is this utterance happy?") but destroys the temporal information we need for ASR ("what character appears at each frame?"). For ASR, we need frame-level fusion, which is why we use gated fusion instead of convex layer combination.

---

## 3. Frame-Level vs Logit-Level Representations

This is a critical difference between our two branches.

### Acoustic Branch: FRAME-LEVEL (Encoder Features)

```
Audio → Kid-Whisper Encoder → hidden states (T, 1024)
                                    ↑
                        These are INTERMEDIATE representations
                        BEFORE the decoder / CTC layer
                        
Audio → Student Encoder → hidden states (T, 768)
                                    ↑
                        Student learns to MATCH these
                        via MSE: ||student_768 - teacher_1024||²
                        (after projection 768→1024)
```

**What are encoder features?**
They are the output of the transformer's self-attention layers. Each frame (20ms of audio) gets a 768-dimensional vector that represents a compressed, abstract summary of the audio at that time. These are NOT predictions — they're intermediate representations that contain information about:
- Pitch and fundamental frequency at that moment
- Energy and loudness
- Spectral shape (which phoneme is being spoken)
- Context from surrounding frames (via self-attention)
- Speaker characteristics (voice quality, age, accent)

**Analogy**: Think of encoder features as "notes" a student writes while listening to a lecture. They capture the raw information but haven't been organized into answers yet.

### Semantic Branch: LOGIT-LEVEL (CTC Decoder Output)

```
Audio → IndicConformer Encoder → hidden states (T, 1024)
        → IndicConformer CTC Decoder → logits (T, 257)
                                            ↑
                                These are FINAL predictions
                                Probability of each BPE token at each frame
                                
Audio → Student Encoder → hidden states (T, 768)
        → CTC Head 2 → logits (T, 257)
                            ↑
                Student learns to MATCH these
                via MSE: ||student_257 - teacher_257||²
```

**What are logits?**
They are the raw scores (before softmax) that the CTC decoder outputs. Each frame gets a 257-dimensional vector where each dimension represents "how likely is BPE token X at this frame?" These ARE predictions — they represent the teacher's opinion about what text corresponds to each audio frame.

**Analogy**: Think of logits as "answers" the teacher writes on an exam. The student is copying the teacher's answers rather than the teacher's understanding.

### What other papers do

| Paper | Acoustic Teacher | What they distill | Level |
|-------|-----------------|-------------------|-------|
| **CARE** (Dutta 2025) | PASE+ | Encoder features | Frame-level |
| **CARE** (Dutta 2025) | RoBERTa | Text embeddings | Sentence-level |
| **FitNets** (Romero 2015) | Large CNN | Intermediate layers | Layer-level |
| **DistilBERT** (Sanh 2019) | BERT-Large | Last hidden states + logits | Both |
| **DistilHuBERT** (Chang 2022) | HuBERT | Hidden states from multiple layers | Multi-layer |
| **Wav2Vec 2.0** (Baevski 2020) | Self (contrastive) | Quantized features | Frame-level |
| **Our Acoustic** | Kid-Whisper | Encoder features | Frame-level |
| **Our Semantic** | IndicConformer | CTC logits | Logit-level |

**Key insight**: Distilling from encoder features (frame-level) gives the student more freedom — it learns the teacher's "thinking process." Distilling from logits (output-level) gives the student the teacher's "final answers" — more constrained but more directly useful for the same task.

**Why our semantic branch distills from logits, not features:**
IndicConformer's encoder has 1024-dim features, but these features are designed for a Conformer architecture with different attention patterns. Matching these features directly would force our Whisper encoder to think like a Conformer, which is architecturally difficult. Instead, we match the CTC decoder output — the teacher's predictions about text. This is architecture-agnostic: regardless of how the teacher arrived at its predictions, we just match the predictions themselves.

---

## 4. What "Acoustic Features" and "Linguistic Features" Actually Mean

### Acoustic Encoder Features: "How it sounds"

When the acoustic encoder processes a child saying "मेरा नाम" (mera naam), the 768-dim vector at each frame captures:

```
Frame 50 (at 1.0 second — the "m" in "mera"):
    dims 0-100:   pitch information (fundamental frequency ~300Hz for children)
    dims 100-200: energy pattern (how loud this phoneme is)
    dims 200-400: spectral envelope (formant frequencies that distinguish /m/ from /n/)
    dims 400-600: temporal context (what came before and after, via self-attention)
    dims 600-768: speaker characteristics (child voice, accent, speaking rate)
```

(Note: these dimension ranges are approximate — in practice, information is distributed and entangled across all 768 dimensions.)

The acoustic encoder learned these from **Kid-Whisper**, which was trained on English-speaking children. So it knows:
- What children's voices sound like (higher pitch, more variability)
- How children pronounce sounds (less precise articulation)
- Speaking patterns of children (pauses, repetitions, self-corrections)

**What it DOESN'T know**: Indian language phonetics. It never saw IndicConformer's knowledge about Hindi/Marathi sound systems.

### Semantic Encoder Features: "What is being said"

When the semantic encoder processes the same "मेरा नाम", its 768-dim vector captures:

```
Frame 50 (at 1.0 second):
    The encoder was trained to produce features that, when passed through
    CTC Head 2 (Linear 768→257), reproduce IndicConformer's prediction:
    
    IndicConformer says: P(token="मे")=0.85, P(token="रा")=0.02, P(blank)=0.10, ...
    
    So the encoder features are shaped to encode:
    "At this frame, the most likely Hindi BPE token is 'मे'"
```

The semantic encoder learned these from **IndicConformer**, which was trained on adult Hindi/Marathi speech. So it knows:
- Hindi phoneme inventory (retroflex consonants, aspirated stops, nasals)
- Marathi-specific sounds (like the /ɭ/ retroflex lateral)
- Indian language phonotactics (which sounds can follow which)
- Common word patterns in Hindi/Marathi

**What it DOESN'T know**: Children's voice characteristics. It was trained to match an adult speech model's outputs.

### Why combining them creates an "Indian Children's Speech" model

```
Acoustic encoder alone:  Knows CHILDREN's voices, NOT Indian languages
                         → 19.97% WER (good at voice, guesses at language)

Semantic encoder alone:  Knows INDIAN languages, NOT children's voices  
                         → 49.02% WER (good at language, struggles with voices)

Combined:                Knows BOTH children's voices AND Indian languages
                         → Should be < 19.97% WER
```

**Technical mechanism**: At each frame, the gated fusion asks:
- "Is this frame where voice quality matters more (child speaking clearly but unusual word)?" → Weight acoustic higher
- "Is this frame where linguistic knowledge matters more (child mumbling but common Hindi word)?" → Weight semantic higher

The gate learns this automatically from the CTC loss during training.

---

## 5. Gated Fusion — Line by Line Execution

Here is the complete GatedFusion module with every line explained:

```python
class GatedFusion(nn.Module):
    def __init__(self, dim=768, dropout=0.1):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
```

### Line: `self.gate_linear = nn.Linear(dim * 2, dim)`

**What it creates**: A fully-connected linear layer with:
- Input size: `dim * 2 = 768 * 2 = 1536`
- Output size: `dim = 768`
- Learnable parameters:
  - Weight matrix `W_g` of shape `(768, 1536)` → 1,179,648 numbers
  - Bias vector `b_g` of shape `(768,)` → 768 numbers
  - Total: **1,180,416 parameters** (this is 95% of all trainable params)

**What it does mathematically**: For input vector `x` of size 1536:
```
output = W_g @ x + b_g
```
This is a matrix multiplication followed by bias addition. Each of the 768 output values is a weighted sum of all 1536 input values.

**Why input is 1536**: Because we will concatenate acoustic (768) + semantic (768) = 1536. The gate needs to SEE BOTH feature vectors to decide how to weight them. If the gate only saw one, it couldn't compare.

**Why output is 768**: Because the final fused vector should be 768-dim (to match the CTC head's expected input size).

**Initialization**: PyTorch initializes the weights using Kaiming uniform initialization. This means the initial weights are small random numbers centered around 0. Since these get passed through sigmoid later, initial gate values will be near σ(0) = 0.5, meaning equal weighting of both encoders at the start.

### Line: `self.layer_norm = nn.LayerNorm(dim)`

**What it creates**: A Layer Normalization module that normalizes across the 768 feature dimensions.

**Parameters**: 
- `gamma` (scale): 768 learnable values, initialized to 1.0
- `beta` (shift): 768 learnable values, initialized to 0.0

**What it does mathematically**: For input vector `x` of size 768:
```
mean = (1/768) * sum(x)
variance = (1/768) * sum((x - mean)²)
output = gamma * (x - mean) / sqrt(variance + 1e-5) + beta
```

**Why we need it**: The two encoders were trained independently. One might produce features with mean=2.5 and std=0.8, the other with mean=-0.3 and std=1.5. After the gated combination, the fused features could have unpredictable scale. LayerNorm brings them to zero mean, unit variance — a stable distribution that the CTC head can reliably work with.

### Line: `self.dropout = nn.Dropout(dropout)`

**What it creates**: A dropout layer with probability 0.1.

**What it does**: During training, randomly sets 10% of values to zero. During eval, does nothing.

**Why**: Regularization. With only ~1.25M trainable params and potentially 10,000+ training clips, the fusion layer could overfit. Dropout forces the model to not rely too heavily on any single feature dimension.

---

Now the forward pass:

```python
def forward(self, acoustic_feat, semantic_feat):
    # acoustic_feat: (B, T, 768)
    # semantic_feat: (B, T, 768)
```

**Inputs**:
- `acoustic_feat`: Output from the frozen acoustic encoder. Shape `(B, T, 768)` where:
  - `B` = batch size (1 in our case, we process one clip at a time)
  - `T` = number of real frames (e.g., 683 for a 13.66-second clip)
  - `768` = Whisper Small hidden dimension
- `semantic_feat`: Output from the frozen semantic encoder. Same shape.

---

### Line: `concat = torch.cat([acoustic_feat, semantic_feat], dim=-1)`

**What happens**: Concatenates along the last dimension (feature dimension).

```
acoustic_feat:  [B, T, 768]    →   first 768 values
semantic_feat:  [B, T, 768]    →   last 768 values
concat:         [B, T, 1536]   →   all 1536 values
```

**Concrete example** (single frame, single batch):
```
acoustic_feat[0, 50, :] = [0.2, -0.5, 1.3, ..., 0.8]   # 768 values from acoustic
semantic_feat[0, 50, :] = [-0.1, 0.7, 0.4, ..., -0.3]   # 768 values from semantic
concat[0, 50, :]        = [0.2, -0.5, 1.3, ..., 0.8, -0.1, 0.7, 0.4, ..., -0.3]  # 1536 values
```

**Why concatenate instead of add?** Adding would mix the features irreversibly. Concatenation preserves both feature vectors intact, letting the gate_linear learn WHICH specific acoustic and semantic dimensions are important and how they interact.

---

### Line: `gate = torch.sigmoid(self.gate_linear(concat))`

This is the CORE of gated fusion. Two operations:

**Step 1: `self.gate_linear(concat)`**

The linear layer transforms each frame's 1536-dim concatenated vector to 768-dim:
```
raw_gate = W_g @ concat + b_g     # shape: (B, T, 768)
```

Each of the 768 output values is a weighted sum of all 1536 input features. For example:
```
raw_gate[0, 50, 0] = W_g[0, 0]*acoustic[0] + W_g[0, 1]*acoustic[1] + ... + W_g[0, 767]*acoustic[767]
                    + W_g[0, 768]*semantic[0] + W_g[0, 769]*semantic[1] + ... + W_g[0, 1535]*semantic[767]
                    + b_g[0]
```

This raw gate value can be any real number (-∞ to +∞).

**Step 2: `torch.sigmoid(...)`**

The sigmoid function squashes each value to the range [0, 1]:
```
sigmoid(x) = 1 / (1 + e^(-x))

sigmoid(-5) ≈ 0.007   (strongly favor semantic)
sigmoid(-2) ≈ 0.12    (mostly semantic)
sigmoid(0)  = 0.50     (equal weighting)
sigmoid(2)  ≈ 0.88    (mostly acoustic)
sigmoid(5)  ≈ 0.993   (strongly favor acoustic)
```

**Result**: `gate` has shape `(B, T, 768)` where every value is between 0 and 1.

**What the gate means**:
- `gate[0, 50, d] = 0.9` → For frame 50, dimension d: use 90% acoustic + 10% semantic
- `gate[0, 50, d] = 0.1` → For frame 50, dimension d: use 10% acoustic + 90% semantic
- `gate[0, 50, d] = 0.5` → For frame 50, dimension d: use equal parts

**Why sigmoid and not softmax?**
- Sigmoid makes each dimension INDEPENDENT. Gate[d=0]=0.9 doesn't affect gate[d=1].
- Softmax would make dimensions compete: if gate[d=0] is high, other dimensions must be lower. This constraint is unnecessary — there's no reason why trusting acoustic for pitch (dimension 0) should prevent trusting acoustic for energy (dimension 1).

**Why sigmoid and not ReLU?**
- ReLU gives values in [0, ∞). We need [0, 1] for weighted average.
- Sigmoid gives exact [0, 1] range, which is interpretable as "percentage from acoustic."

---

### Line: `fused = gate * acoustic_feat + (1 - gate) * semantic_feat`

**What happens**: Element-wise weighted average.

For each frame t, each dimension d:
```
fused[t, d] = gate[t, d] * acoustic[t, d] + (1 - gate[t, d]) * semantic[t, d]
```

**Concrete example**:
```
Frame 50, dimension 0:
    gate = 0.8
    acoustic value = 1.5
    semantic value = -0.3
    fused = 0.8 * 1.5 + 0.2 * (-0.3) = 1.2 - 0.06 = 1.14

Frame 50, dimension 100:
    gate = 0.2
    acoustic value = 0.4
    semantic value = 2.1
    fused = 0.2 * 0.4 + 0.8 * 2.1 = 0.08 + 1.68 = 1.76
```

**Mathematical property**: This is a convex combination. For any gate value g ∈ [0, 1]:
```
fused = g * a + (1-g) * s
```
The fused value always lies BETWEEN the acoustic and semantic values (or equals one of them at the extremes). This prevents the fusion from producing values wildly different from either encoder — a stability guarantee.

**Why `(1 - gate)` instead of a separate gate?**
Using `(1 - gate)` ensures the weights always sum to 1.0 at each dimension:
```
weight_acoustic + weight_semantic = gate + (1 - gate) = 1.0
```
If we used two separate gates, the weights could sum to >1 (amplification) or <1 (information loss). The convex combination preserves the scale of the features.

---

### Line: `fused = self.layer_norm(fused)`

Normalizes the fused features across 768 dimensions:
```
For each frame t:
    mean = average of fused[t, 0..767]
    std = standard deviation of fused[t, 0..767]
    fused[t, d] = gamma[d] * (fused[t, d] - mean) / std + beta[d]
```

This ensures the CTC head receives inputs with a consistent distribution regardless of how the gate weights changed the feature magnitudes.

---

### Line: `fused = self.dropout(fused)`

During training: randomly zero out 10% of the 768 dimensions at each frame.
During evaluation: pass through unchanged.

---

### Full data flow example

For a 10-second audio clip of a Hindi child saying "मेरा नाम पियुष है":

```
1. Audio → Mel (1, 80, 3000) → SpecAugment

2. Acoustic Encoder (FROZEN):
   Mel → 12 transformer layers → (1, 1500, 768)
   Slice to real frames → (1, 500, 768)
   
   Each frame: 768 values encoding children's voice patterns
   Frame 50: [0.2, -0.5, 1.3, ..., 0.8]  ← "sounds like a child saying /m/"

3. Semantic Encoder (FROZEN):
   Same mel → 12 transformer layers → (1, 1500, 768)  
   Slice to real frames → (1, 500, 768)
   
   Each frame: 768 values encoding Indian language patterns
   Frame 50: [-0.1, 0.7, 0.4, ..., -0.3]  ← "likely a Hindi nasal consonant /m/"

4. Gated Fusion:
   concat → (1, 500, 1536)
   gate = σ(W @ concat + b) → (1, 500, 768)
   
   Frame 50 gate: [0.7, 0.3, 0.5, ..., 0.8]
   ↑ mostly acoustic for dims related to voice quality
   ↑ mostly semantic for dims related to phoneme identity
   
   fused = gate * acoustic + (1-gate) * semantic → (1, 500, 768)
   → LayerNorm → Dropout

5. CTC Head:
   fused (1, 500, 768) → Linear(768, 85) → logits (1, 500, 85)
   
   Frame 50 logits: [0.1, 0.0, 0.0, ..., 8.5, ..., 0.2]
                                          ↑ index for "म" has highest score
   
6. CTC Decode:
   argmax per frame → collapse repeats → remove blanks → "मेरा नाम पियुष है"
```

---

## 6. How the Combined Model Becomes an "Indian Children's Speech" Model

The core insight of the dual-encoder approach:

```
                  CHILDREN knowledge          INDIAN LANGUAGE knowledge
                  (from Kid-Whisper)           (from IndicConformer)
                        |                              |
                        v                              v
                  Acoustic Encoder              Semantic Encoder
                  "This sounds like            "This sounds like
                   a child's /m/"               Hindi nasal /m/"
                        |                              |
                        +------→ Gate ←────────────────+
                                  |
                                  v
                          "This is definitely /m/
                           spoken by a child in Hindi"
                                  |
                                  v
                              CTC Head → "म"
```

### Why this works better than either alone:

**Problem 1: Child says "कम" (kam) but with heavy accent**
- Acoustic encoder: "I hear a child's voice, the /k/ is aspirated (children often do this), and the vowel sounds like /a/... probably 'kam'"  ← Correct because it knows children
- Semantic encoder: "The spectral pattern matches Hindi /k/ + /a/ + /m/... but the pitch and energy are unusual..." ← Confused by child's voice
- Combined: Gate trusts acoustic for voice quality, semantic for phoneme confirmation → "कम" ✓

**Problem 2: Child mumbles "अच्छा" (achchha — "good") quickly**
- Acoustic encoder: "I hear something like 'ah-chh-ah' from a child, but the consonant cluster is unclear..."  ← Struggles because the sound is ambiguous
- Semantic encoder: "The pattern matches IndicConformer's prediction for 'अच्छा' — this is a very common Hindi word with a characteristic geminate consonant"  ← Knows the language pattern
- Combined: Gate trusts semantic for linguistic identification → "अच्छा" ✓

**Problem 3: Child says "school" (English word in Hindi sentence)**
- Acoustic encoder: "This is clearly a child saying an English word 'school'" ← Kid-Whisper trained on English children
- Semantic encoder: "IndicConformer doesn't handle English well..." ← No English in teacher
- Combined: Gate trusts acoustic for English segments → "school" ✓

### The training process that makes this happen

During training, the CTC loss computes the error between the model's prediction and the ground truth text. This error flows backward:

```
CTC Loss: "you predicted 'कन' but correct is 'कम'"
    |
    v (backward through CTC Head)
CTC Head: "the features at frame 50 led me astray"
    |
    v (backward through LayerNorm + gate)
Gate: "I weighted acoustic 0.5 and semantic 0.5 at frame 50"
      "The acoustic features had the right answer (0.8 for /m/)
       but semantic features had wrong answer (0.6 for /n/)"
      "→ I should increase gate toward acoustic for this type of frame"
    |
    v (update W_g weights via AdamW optimizer)
Next time: gate for similar frames → 0.7 (more acoustic)
```

Over thousands of training steps, the gate learns a complex mapping:
- "When acoustic and semantic agree → gate ≈ 0.5 (doesn't matter)"
- "When they disagree on Hindi → gate ≈ 0.4 (trust semantic more for Hindi phonemes)"
- "When they disagree on English → gate ≈ 0.8 (trust acoustic more for English)"
- "When the child's voice is unclear → gate ≈ 0.3 (trust semantic's language model)"
- "When the linguistic context is ambiguous → gate ≈ 0.7 (trust acoustic's hearing)"

**This is why the combined model is an "Indian Children's Speech" model**: it learns to dynamically use children's voice knowledge (acoustic) and Indian language knowledge (semantic) depending on what each frame of audio needs.
