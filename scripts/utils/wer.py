"""
Standard WER (Word Error Rate) module for ALL experiments.
============================================================
USE THIS for every evaluation script — never write your own WER logic.

Why this file exists:
  - Every experiment must use the SAME normalization + SAME WER method
  - Otherwise results are not comparable across experiments

Two functions:
  1. normalize_text()  → clean text before WER comparison
  2. compute_corpus_wer() → corpus-level WER (standard for ASR papers)

What is corpus-level WER?
  - Pass ALL references and ALL predictions as lists
  - jiwer treats each list item as a separate sentence, computes WER across all of them
  - Longer sentences (more words) naturally carry more weight
  - This is what Whisper paper, CARE paper, and all ASR papers use

Why NOT average-of-clips WER?
  - That computes WER per clip, then averages
  - A 2-word clip with 1 error = 50% WER, a 100-word clip with 5 errors = 5% WER
  - Average = 27.5%, but corpus-level = 6/102 = 5.9%
  - Short clips inflate the number unfairly
"""

from jiwer import wer as jiwer_wer


def normalize_text(text):
    """
    Normalize text before WER comparison.

    Why we need this:
      Our ground truth transcripts are already clean (no punctuation).
      But MODEL OUTPUTS add punctuation, capitalization, extra spaces.
      Example: GT = "what is the time", Whisper = "What is the time?"
      Without normalization: "time" vs "time?" = error. With: both become "what is the time" = 0% WER.

    What it does:
      1. Strip whitespace
      2. Lowercase everything
      3. Remove ALL punctuation (English + Hindi/Marathi danda ।॥)
      4. Collapse multiple spaces into one
    """
    if not text:
        return ""
    text = text.strip().lower()
    # Remove punctuation — covers English, Hindi, Marathi
    for ch in '.,!?;:"\'-()[]{}—–_।॥…·।':
        text = text.replace(ch, "")
    return " ".join(text.split())


def compute_corpus_wer(references, predictions):
    """
    Compute corpus-level WER — the STANDARD metric for ASR papers.

    Args:
        references:  list of ground truth strings, one per clip
        predictions: list of predicted strings, one per clip (same order)

    Returns:
        float: WER (0.0 = perfect, 1.0 = 100% error, can be >1.0)

    How it works:
        jiwer receives two lists of equal length. It computes:
        WER = (substitutions + insertions + deletions) / total_reference_words
        across ALL sentences combined.

    Example:
        refs  = ["what is the time", "I like to play cricket with my friends"]
        preds = ["what is the time", "I like to play"]
        WER = 3 deletions / 12 total words = 25%
    """
    # Normalize both sides
    refs_clean = [normalize_text(r) for r in references]
    preds_clean = [normalize_text(p) for p in predictions]

    # Remove pairs where reference is empty (can't compute WER on empty reference)
    filtered_refs = []
    filtered_preds = []
    for r, p in zip(refs_clean, preds_clean):
        if r:  # skip empty references
            filtered_refs.append(r)
            filtered_preds.append(p if p else "")  # empty prediction = 100% error

    if not filtered_refs:
        return 0.0

    return jiwer_wer(filtered_refs, filtered_preds)
