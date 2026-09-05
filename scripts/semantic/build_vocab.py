"""
Build CTC Vocabulary from Training Transcripts
================================================

WHAT THIS SCRIPT DOES:
    Scans all transcripts in asr_train.csv, collects every unique CHARACTER,
    and saves a vocabulary file (vocab.json) mapping each character to an index.

WHY CHARACTER-LEVEL (not word-level)?
    CTC works at the frame level — 1500 encoder frames for 30s audio.
    Each frame predicts ONE token. If vocab = words, each frame would need
    to predict a full word, which is too hard (vocab would be 50,000+ words).
    With characters, vocab is small (~120 chars for Hindi+Marathi+English)
    and CTC can spread one word across multiple frames:
        "हलो" = frame1→ह, frame2→ल, frame3→ो

    This is the standard approach used in:
    - Wav2Vec 2.0 (Baevski et al., NeurIPS 2020) — character-level CTC
    - HuggingFace CTC fine-tuning tutorial (Patrick von Platen)
    - CARE paper (Rumberg et al., 2022) — character-level CTC on children's speech

SPECIAL TOKENS:
    <blank> (index 0): CTC blank token — means "no output at this frame"
        Required by torch.nn.CTCLoss(blank=0). This is NOT a space character.
        CTC uses blanks to separate repeated characters:
            Output: [ह, ह, <blank>, ह] → collapsed: "हह" (two ह characters)
            Output: [ह, ह, ह, ह]       → collapsed: "ह"  (one ह character)

    <space> (index 1): Word boundary — the actual space between words.
        "राधा के पास" has spaces between words.
        We need a token for it so CTC can predict word boundaries.

    <unk> (index 2): Unknown token — for any character not in vocab during inference.
        Safety net. Should rarely be used if vocab is built from training data.

TEXT NORMALIZATION:
    We apply the SAME normalization as our WER module (scripts/utils/wer.py):
    - Lowercase (for English consistency)
    - Remove punctuation (English + Hindi/Marathi danda ।॥)
    This ensures the vocab matches what the model will be evaluated on.

OUTPUT:
    ASER-Dataset/vocab.json — JSON dict mapping character → index
    Example: {"<blank>": 0, "<space>": 1, "<unk>": 2, "a": 3, "b": 4, ..., "अ": 30, ...}

REFERENCE:
    HuggingFace blog: "Fine-Tune Wav2Vec2 for English ASR with Transformers"
    https://huggingface.co/blog/fine-tune-wav2vec2-english
    They build vocab the exact same way: scan training text → unique chars → vocab.json

Usage:
    python scripts/semantic/build_vocab.py
"""

import csv
import json
import os
import sys

# ─── Add project root to path so we can import utils ───
# This lets us do: from scripts.utils.wer import normalize_text
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.wer import normalize_text


# ─── Config ───
TRAIN_CSV = os.path.join(PROJECT_ROOT, "ASER-Dataset", "splits", "asr_train.csv")
VOCAB_OUTPUT = os.path.join(PROJECT_ROOT, "ASER-Dataset", "vocab.json")


def build_vocab():
    """
    Scan all training transcripts and build character vocabulary.

    STEPS:
        1. Read every transcript from asr_train.csv
        2. Apply same normalization as WER module (lowercase + remove punctuation)
        3. Collect every unique character across all transcripts
        4. Sort characters (Devanagari first by Unicode, then English a-z)
        5. Assign index to each: {<blank>: 0, <space>: 1, <unk>: 2, char1: 3, ...}
        6. Save as vocab.json

    Returns:
        dict: The vocabulary mapping {character: index}
    """

    # ── Step 1: Read all transcripts ──
    print("Reading training transcripts...")
    transcripts = []
    lang_counts = {"Hindi": 0, "Marathi": 0, "English": 0}

    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row["transcript"]
            language = row["language"]
            transcripts.append(transcript)
            if language in lang_counts:
                lang_counts[language] += 1

    print(f"  Total transcripts: {len(transcripts)}")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")

    # ── Step 2: Normalize all transcripts ──
    # Using the SAME normalize_text() from scripts/utils/wer.py
    # This ensures vocab matches what WER evaluation will see
    print("\nNormalizing transcripts (lowercase + remove punctuation)...")
    normalized = [normalize_text(t) for t in transcripts]

    # Remove empty transcripts (if any after normalization)
    normalized = [t for t in normalized if t.strip()]
    print(f"  Non-empty after normalization: {len(normalized)}")

    # ── Step 3: Collect unique characters ──
    # We treat each character individually. Space is handled separately.
    print("\nCollecting unique characters...")
    all_chars = set()
    for text in normalized:
        for char in text:
            if char != " ":  # Space gets its own special token <space>
                all_chars.add(char)

    # ── Step 4: Sort characters ──
    # sorted() on Unicode puts Devanagari (U+0900-097F) before Latin (U+0061-007A)
    # This makes the vocab file organized: Devanagari block, then English block
    sorted_chars = sorted(all_chars)

    # ── Step 5: Build vocab dict with special tokens ──
    # Index 0 = <blank> (required by CTCLoss, blank=0 is PyTorch default)
    # Index 1 = <space> (word boundary)
    # Index 2 = <unk> (unknown character fallback)
    # Index 3+ = actual characters from training data
    vocab = {
        "<blank>": 0,
        "<space>": 1,
        "<unk>": 2,
    }

    for i, char in enumerate(sorted_chars):
        vocab[char] = i + 3  # Start from index 3

    print(f"\n  Vocabulary size: {len(vocab)}")
    print(f"    Special tokens: 3 (<blank>, <space>, <unk>)")
    print(f"    Unique characters: {len(sorted_chars)}")

    # ── Print character breakdown ──
    devanagari = [c for c in sorted_chars if '\u0900' <= c <= '\u097F']
    english = [c for c in sorted_chars if 'a' <= c <= 'z']
    digits = [c for c in sorted_chars if c.isdigit()]
    other = [c for c in sorted_chars if c not in devanagari and c not in english and c not in digits]

    print(f"\n  Character breakdown:")
    print(f"    Devanagari: {len(devanagari)} characters")
    print(f"    English a-z: {len(english)} characters")
    print(f"    Digits 0-9: {len(digits)} characters")
    if other:
        print(f"    Other: {len(other)} characters → {other}")

    # ── Show some examples ──
    print(f"\n  First 10 Devanagari: {devanagari[:10]}")
    print(f"  English: {english}")

    # ── Step 6: Save vocab.json ──
    print(f"\nSaving vocabulary to: {VOCAB_OUTPUT}")
    with open(VOCAB_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(vocab)} entries.")

    # ── Verification ──
    # Quick check: encode a sample transcript and decode it back
    print(f"\n{'='*60}")
    print("VERIFICATION — encode and decode a sample transcript")
    print(f"{'='*60}")

    # Pick one Hindi and one English sample
    sample_hi = None
    sample_en = None
    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"] == "Hindi" and sample_hi is None:
                sample_hi = row["transcript"]
            if row["language"] == "English" and sample_en is None:
                sample_en = row["transcript"]
            if sample_hi and sample_en:
                break

    # Reverse vocab for decoding: index → character
    idx_to_char = {v: k for k, v in vocab.items()}

    for label, sample in [("Hindi", sample_hi), ("English", sample_en)]:
        if not sample:
            continue
        normalized_sample = normalize_text(sample)
        print(f"\n  [{label}]")
        print(f"  Original:   '{sample}'")
        print(f"  Normalized: '{normalized_sample}'")

        # Encode: text → list of indices
        indices = []
        for char in normalized_sample:
            if char == " ":
                indices.append(vocab["<space>"])
            elif char in vocab:
                indices.append(vocab[char])
            else:
                indices.append(vocab["<unk>"])

        print(f"  Encoded:    {indices[:20]}{'...' if len(indices) > 20 else ''}")

        # Decode: list of indices → text
        decoded = ""
        for idx in indices:
            token = idx_to_char[idx]
            if token == "<space>":
                decoded += " "
            elif token == "<blank>" or token == "<unk>":
                continue
            else:
                decoded += token

        print(f"  Decoded:    '{decoded}'")
        print(f"  Match: {'YES ✓' if decoded == normalized_sample else 'NO ✗ — BUG!'}")

    return vocab


if __name__ == "__main__":
    vocab = build_vocab()
    print(f"\nDone. Vocab size = {len(vocab)}")
