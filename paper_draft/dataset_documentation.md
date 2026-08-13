# ASER Dataset — Preparation and Split Methodology
# Indian Children Speech Recognition (ICASSP 2026)

---

## 1. Original Dataset (Before Our Work)

The ASER dataset was introduced by Agarwal et al. (ICASSP 2020) for oral reading
fluency assessment of Indian school children. It contains recordings of children
(ages 6–14) reading standardized text at different levels (letters, words, sentences,
paragraphs, stories) in Hindi, Marathi, and English.

**Original state:**
- 81,423 audio clips in WAV format (distributed as ZIP archives)
- Metadata CSV with fields: child_id, audio_path, transcript, script_language,
  reading_level, region, duration_sec
- NO official train/dev/test splits provided
- NO preprocessing applied (raw field recordings)

**Recording conditions:** Smartphones in rural school settings (noisy environment,
variable recording distance, variable device quality).

---

## 2. Prior ASR Work on This Dataset

After a systematic literature review of ICASSP, Interspeech, ACL Anthology, and
ArXiv proceedings (2020–2025), **no published ASR work with documented splits,
preprocessing, or WER numbers exists for this dataset.**

The original paper benchmarks a reading-level CLASSIFIER (not ASR). Subsequent
citations are in educational technology or oral reading fluency contexts, without
ASR pipelines.

**This thesis represents the first ASR system trained and evaluated on the ASER
dataset.** As no prior split convention exists, we establish one here with full
documentation so that future work can reproduce and extend our results.

---

## 3. Round 1 — Extraction and Split Creation

### 3.1 Extraction

All ZIP archives were extracted to a single directory. A unified manifest CSV was
built with one row per audio clip containing all metadata fields. This produced
**81,423 clips** across Hindi (Rajasthan/UP), Marathi (Maharashtra), and Indian
English recordings.

### 3.2 Train/Dev/Test Split Methodology

We created speaker-independent splits following the standard protocol for speaker
verification and child ASR datasets (e.g., MyST, CMU Kids):

**Core rule: all clips from one child appear in exactly one split.**

This is required because:
- A child's voice is distinctive (pitch, accent, pronunciation patterns)
- Putting the same child in both train and test would let the model "memorize"
  that child's voice, inflating accuracy artificially
- Speaker-independent evaluation measures generalization to unseen children —
  the real-world use case (assessing a child the model has never heard)

**Split ratios:** 80% train / 10% dev / 10% test

**Stratification:** Children were assigned to splits stratified by region
(RJ = Rajasthan Hindi, UP = Uttar Pradesh Hindi, MH = Maharashtra Marathi)
to ensure each split reflects the full geographic and linguistic diversity of
the dataset.

**Seed:** random_state=42 (fixed, reproducible — any researcher running our
script will get identical splits)

**Result after Round 1:**

| Split | Children | Clips |
|-------|----------|-------|
| train | ~4,208 | 65,311 |
| dev   | ~526  | 7,987  |
| test  | ~526  | 8,125  |

**ASR-only subset:** For ASR training, we exclude single-letter and single-word
clips (reading_level = Letter, Capital Letter, Small Letter, Word) because these
are too short and lack the acoustic context needed for sequence-to-sequence
transcription. We retain only:
- Sentence (English)
- Paragraph
- Story

This produces asr_train / asr_dev / asr_test splits (10,939 / 1,404 / 1,530 clips).

**Comparison to Pratham's own baseline:** Pratham's internal benchmark used random
splitting without speaker-independence constraint. This means the same child's clips
could appear in both train and test, giving inflated accuracy. Our split is the
correct methodology for a publishable ASR benchmark.

---

## 4. Round 2 — Audio Cleaning

### 4.1 Filtering Bad Clips

Before cleaning, we removed statistically invalid clips:

| Reason | Clips Removed |
|--------|--------------|
| Zero duration (corrupted files) | 84 |
| Too short (< 0.5 seconds) | 142 |
| Too long (> 30 seconds, recording errors) | 2,995 |
| Invalid reading level label "Cl" | 175 |
| **Total removed** | **3,396** |
| **Remaining** | **78,027** |

### 4.2 Transcript Normalization

Removed punctuation that has no phonetic value: `। . , : ; ! ? " ' ( ) [ ] { } – — - / \`

Collapsed multiple whitespace. This is standard preprocessing for ASR text targets —
punctuation does not appear in speech and should not be in the target vocabulary.

### 4.3 Audio Processing Pipeline (ffmpeg)

Each audio file was processed with:

```
silenceremove=start_periods=1:start_threshold=-40dB:start_duration=0.05
             :stop_periods=-1:stop_threshold=-40dB:stop_duration=0.2
dynaudnorm=f=150:g=15
aresample=16000
```

**silenceremove:** Trims silence from the beginning and end of each recording.
Field recordings of children often have 0.5–2 seconds of room noise before the
child starts speaking. This silence is not meaningful for ASR and wastes model
capacity. Threshold -40dB, minimum silence duration 0.05s at start / 0.2s at end.

**dynaudnorm:** Single-pass volume normalization. Field recordings vary widely
in loudness (some children spoke close to the microphone, others from a distance).
Normalizing volume ensures the model trains on consistent amplitude ranges.

**aresample=16000:** Resample to 16kHz. Standard for wav2vec2 / MMS / Whisper
family models. Original recordings vary in sample rate.

Processing used 27 parallel workers (CPU count − 1). All 78,027 files processed
in approximately 15 minutes. Zero fallbacks (all files processed with full
VAD + normalization, none required the fallback pipeline).

### 4.4 Duration Recomputation and Final Filter

After VAD trimming, clip durations were recomputed using ffprobe. Clips that
became shorter than 0.5 seconds after silence removal (the speech content itself
was too short) were excluded:

- Removed after VAD: 8,661 clips
- Final clip count: **69,366 clips**

### 4.5 Updated Splits

All six splits (train/dev/test, asr_train/asr_dev/asr_test) were updated to
include only clips that survived cleaning, with updated audio paths pointing
to the cleaned audio directory and updated duration_sec values.

---

## 5. Final Dataset Statistics

### Overall

| Metric | Value |
|--------|-------|
| Original clips | 81,423 |
| Final clips | 69,366 |
| Total removed | 12,057 (14.8%) |
| Total duration | 45.42 hours |
| Unique children | 5,241 |

### By Language

| Language | Clips | Duration |
|----------|-------|----------|
| English | 52,936 | 24.26 hrs |
| Hindi | 13,308 | 16.84 hrs |
| Marathi | 3,122 | 4.32 hrs |

### By Reading Level

| Level | Clips | ASR? |
|-------|-------|------|
| Capital Letter (English) | 18,740 | No |
| Small Letter (English) | 15,116 | No |
| Word (English) | 12,977 | No |
| Sentence (English) | 6,103 | **Yes** |
| Letter (Devanagari) | 6,071 | No |
| Word (Devanagari) | 5,726 | No |
| Paragraph | 4,076 | **Yes** |
| Story | 557 | **Yes** |

### ASR Subset (Connected Speech Only)

| Split | Clips | Notes |
|-------|-------|-------|
| asr_train | 8,454 | For model training |
| asr_dev | 1,062 | Hyperparameter tuning |
| asr_test | 1,220 | Final evaluation (never touched during training) |
| **Total** | **10,736** | **19.91 hours** |

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `manifest.csv` | Original 81,423 clips |
| `manifest_clean.csv` | All 69,366 cleaned clips |
| `asr_manifest_clean.csv` | 10,736 ASR-only cleaned clips |
| `train_clean.csv` | 55,777 training clips (all levels) |
| `dev_clean.csv` | 6,729 dev clips |
| `test_clean.csv` | 6,860 test clips |
| `asr_train_clean.csv` | 8,454 ASR training clips |
| `asr_dev_clean.csv` | 1,062 ASR dev clips |
| `asr_test_clean.csv` | 1,220 ASR test clips |
| `audio_clean/` | Processed audio (16kHz, VAD trimmed, normalized) |

---

## 7. Reproducibility

All preprocessing code is in `prepare_dataset.py` (Round 1) and `clean_dataset.py`
(Round 2). The split creation uses `random_state=42` throughout. Any researcher
with access to the original ASER ZIP files can run these scripts and obtain
identical splits and statistics.
