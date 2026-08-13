# ASER Dataset Preprocessing
# Indian Children Speech Recognition — ICASSP 2026

---

## Raw Dataset (Before Any Processing)

The ASER dataset (Agarwal et al., ICASSP 2020) was distributed as **5,301 ZIP files**
organized into three region folders: Hindi RJ (Rajasthan), Hindi UP (Uttar Pradesh),
and Marathi MH (Maharashtra). Each ZIP corresponds to one child's complete recording
session. No train/dev/test splits were provided. No preprocessing was applied.

- **Total clips:** 81,423
- **Total children:** 5,301
- **Total duration:** ~123 hours (estimated)
- **Audio format:** AMR-WB codec, 16 kHz, mono (smartphone field recordings)

Each ZIP contains: one JSON file (child metadata + transcripts) and 10–25 audio clips
(the child reading letters, words, paragraphs, or stories at varying difficulty levels).

**Note on English:** There is no separate English region. English reading tasks
(capital letters, small letters, English words, English sentences) are embedded
within every Hindi and Marathi session — all children were tested on English
literacy as part of the ASER assessment.

---

## Round 1 — Extraction and Split Creation

**Script:** `scripts/prepare_dataset.py` + `scripts/eda_and_splits.py`

### Step 1 — Extract and Build Manifest

All 5,301 ZIP files were extracted. A unified manifest CSV (`manifest.csv`) was
constructed with one row per audio clip, consolidating all metadata from the
per-child JSON files.

Each row contains: audio file path, transcript, language (Hindi/Marathi/English),
region, reading level, child ID, age group, school grade, recording date,
correctness label, and clip duration (computed via ffprobe).

**Result:** manifest.csv — 81,423 rows

### Step 2 — Speaker-Independent Train/Dev/Test Splits

Splits were created **by child ID**, not by clip. All recordings of a given child
appear in exactly one split. This is the standard protocol for child ASR evaluation
(used in MyST, CMU Kids, and related corpora) and prevents speaker leakage —
the test set contains only voices the model has never heard during training.

**Method:**
- Children grouped by region (Hindi RJ / Hindi UP / Marathi MH)
- Within each region, children shuffled with fixed seed (random_state = 42)
- Split 80% train / 10% dev / 10% test by child count within each region
- Clips assigned to the split of their child

**Why stratified by region?** To ensure each split reflects the same
linguistic/geographic mix as the full dataset. Without stratification, random
chance could concentrate Marathi speakers in one split.

**Speaker leakage verification:** Programmatic check confirmed zero overlap
between train, dev, and test child IDs.

| Split | Children | Clips |
|-------|----------|-------|
| train | ~4,208   | 65,311 |
| dev   | ~526     | 7,987  |
| test  | ~526     | 8,125  |

### Step 3 — ASR Subset Splits

Single-character and single-word clips (reading levels: Letter, Capital Letter,
Small Letter, Word) are excluded from ASR training. These clips lack sufficient
acoustic context for sequence-to-sequence transcription. Only connected speech
levels are retained: **Sentence (English), Paragraph, and Story**.

| Split     | Clips  |
|-----------|--------|
| asr_train | 10,939 |
| asr_dev   | 1,404  |
| asr_test  | 1,530  |

---

## Round 2 — Audio and Text Cleaning

**Script:** `scripts/clean_dataset.py`

### Step 1 — Remove Invalid Clips

Clips failing basic validity checks were removed before any audio processing:

| Reason | Clips Removed |
|--------|--------------|
| Zero duration (corrupted files) | 84 |
| Duration < 0.5 seconds (no real speech) | 142 |
| Duration > 30 seconds (recording error*) | 2,995 |
| Invalid reading level label "Cl" | 175 |
| **Total removed** | **3,396** |
| **Remaining** | **78,027** |

*Clips exceeding 30 seconds are recording errors where the device was not stopped
after the child finished reading. The speech content itself is under 30 seconds
in legitimate clips. This threshold also matches the maximum context window of
wav2vec2 and Whisper-family models.

### Step 2 — Transcript Normalization

Punctuation with no phonetic value was removed from all transcripts:
`। . , : ; ! ? " ' ( ) [ ] { } – — - / \`

Multiple whitespace was collapsed. This is standard for ASR text targets —
punctuation is never spoken and should not appear in the model's output vocabulary.

### Step 3 — Audio Processing (VAD + Normalization + Resampling)

Each of the 78,027 audio files was processed using a single ffmpeg pass with
three operations applied in sequence:

**a) Silence Trimming (Voice Activity Detection)**
Leading and trailing silence was removed using ffmpeg's `silenceremove` filter
(threshold: -40 dB, minimum silence duration: 50 ms at start, 200 ms at end).
Field recordings of children typically contain 0.5–2 seconds of ambient noise
before the child begins reading. Removing this silence reduces irrelevant context
that the model would otherwise need to learn to ignore.

**b) Volume Normalization**
Volume was normalized using ffmpeg's `dynaudnorm` filter (Dynamic Audio Normalizer,
frame length 150 ms, Gaussian smoothing over 15 frames). Field recordings vary
widely in loudness due to differences in recording distance, device sensitivity,
and ambient noise. Normalization ensures consistent amplitude across clips,
which stabilizes model training.

**c) Resampling**
All audio was resampled to 16,000 Hz using ffmpeg's `aresample` filter. The source
codec (AMR-WB) already records at 16 kHz, so this step standardizes the output
codec to MP3 and guarantees format consistency regardless of source device.

Processing was parallelized across 27 workers (CPU count − 1). All 78,027 files
completed with zero fallbacks (no file required the VAD-skip fallback pipeline).

### Step 4 — Post-VAD Duration Filter

After silence trimming, clip durations were recomputed using ffprobe. Clips that
fell below 0.5 seconds after trimming (indicating that most audio was silence
with minimal real speech) were excluded.

- **Removed after VAD:** 8,661 clips
- **Final clip count:** 69,366 clips

### Step 5 — Updated Splits

All six split files (train/dev/test, asr_train/asr_dev/asr_test) were regenerated
to reflect the cleaned clip set, with updated audio paths and durations.

---

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Final clips (all levels) | 69,366 |
| Final duration (all levels) | 45.42 hours |
| ASR clips (connected speech) | 10,736 |
| ASR duration | 19.91 hours |
| Unique children | 5,241 |

**By language:**

| Language | Clips | Duration |
|----------|-------|----------|
| English  | 52,936 | 24.26 hrs |
| Hindi    | 13,308 | 16.84 hrs |
| Marathi  | 3,122  | 4.32 hrs  |

**ASR splits (final, after cleaning):**

| Split     | Clips | Duration |
|-----------|-------|----------|
| asr_train | 8,454 | ~16.4 hrs |
| asr_dev   | 1,062 | ~2.1 hrs  |
| asr_test  | 1,220 | ~2.4 hrs  |

---

## Reproducibility

All splits use `random_state = 42`. Any researcher with access to the original
ASER ZIP files can run `scripts/prepare_dataset.py` followed by
`scripts/eda_and_splits.py` and `scripts/clean_dataset.py` to obtain
identical splits and statistics.
