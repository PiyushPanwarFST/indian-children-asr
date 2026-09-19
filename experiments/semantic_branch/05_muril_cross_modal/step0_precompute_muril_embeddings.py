"""
Step 0: Pre-Compute MuRIL Word Embeddings (One-Time Offline)
=============================================================
Runs MuRIL (google/muril-base-cased) on the ground-truth text of every clip
and saves per-word contextual embeddings to disk.

WHY pre-compute?
    MuRIL is frozen during cross-modal distillation — its weights never change.
    Running it inside the training loop wastes GPU time on redundant forward
    passes. Pre-computing once means training only loads .pt files from disk.

WHAT we save per clip:
    - word_embeddings: (N_words, 768) — contextual word embeddings
      N_words = number of whitespace-delimited words in the ground truth.
      Each word embedding is the AVERAGE of its WordPiece subword embeddings
      from MuRIL's last hidden state.
    - words: list[str] — the actual words (for verification / alignment)
    - num_words: int — len(words)

WHY average subword embeddings?
    MuRIL uses WordPiece tokenization, so a single word like "बोलता" may
    become ["बो", "##लता"]. Averaging the subword vectors gives us one
    embedding per word, which aligns naturally with word-level ASR outputs.
    We use tokenizer.word_ids() for robust subword-to-word grouping.

VERIFICATION built in:
    --verify mode processes 5 clips and prints full details: original text,
    WordPiece tokens, word_ids mapping, resulting word embeddings shape.
    Use this to sanity-check before a full run.

Usage:
    # Quick verify — 5 clips with detailed output
    python step0_precompute_muril_embeddings.py --verify

    # Process all training clips
    python step0_precompute_muril_embeddings.py --split train

    # Process all splits
    python step0_precompute_muril_embeddings.py --split all

    # Test with limited clips on CPU
    python step0_precompute_muril_embeddings.py --split train --max_clips 20 --device cpu
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASER_ROOT = PROJECT_ROOT / "ASER-Dataset"

SPLIT_CSV = {
    "train": ASER_ROOT / "splits" / "asr_train.csv",
    "dev":   ASER_ROOT / "splits" / "asr_dev.csv",
    "test":  ASER_ROOT / "splits" / "asr_test.csv",
}

MURIL_MODEL_NAME = "google/muril-base-cased"
EMBEDDING_DIM = 768

LANGUAGE_MAP = {"Hindi": "hi", "Marathi": "mr", "English": "en"}

# Output directory (relative to this script's experiment folder)
OUTPUT_BASE = Path(__file__).resolve().parent / "muril_embeddings"


# ─────────────────────────────────────────────
# LOAD CLIPS FROM CSV
# ─────────────────────────────────────────────

def load_clips(csv_path, max_clips=None):
    """
    Load clips from an ASER split CSV file.

    Reads all rows and extracts audio path, transcript, language, and child_id.
    The transcript field may be named "transcript", "que_text", or "text"
    depending on the CSV version — all three are checked.

    Args:
        csv_path: Path to the CSV file.
        max_clips: If set, return at most this many clips.

    Returns:
        list of dicts with keys:
            audio_path, language, lang_code, ground_truth, clip_uid
    """
    clips = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row.get("language", "")
            lang_code = LANGUAGE_MAP.get(lang)
            if lang_code is None:
                continue

            audio_path = row["audio_path"]

            # Extract transcript — try multiple possible column names
            ground_truth = row.get("transcript",
                           row.get("que_text",
                           row.get("text", "")))

            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "ground_truth": ground_truth,
                "clip_uid": clip_uid,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips


# ─────────────────────────────────────────────
# MURIL EMBEDDING EXTRACTION
# ─────────────────────────────────────────────

def load_muril(device="cpu"):
    """
    Load MuRIL model and tokenizer from HuggingFace.

    MuRIL (Multi-lingual Representations for Indian Languages) is a BERT-based
    model pre-trained on 17 Indian languages + English. We use it as a frozen
    text encoder for cross-modal knowledge distillation.

    Args:
        device: "cpu" or "cuda".

    Returns:
        model: BertModel (frozen, eval mode)
        tokenizer: BertTokenizerFast
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MURIL_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL_NAME)
    model = AutoModel.from_pretrained(MURIL_MODEL_NAME)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print(f"  Loaded MuRIL on {device}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    return model, tokenizer


def extract_word_embeddings(model, tokenizer, text, device="cpu"):
    """
    Tokenize text with MuRIL and extract per-word contextual embeddings.

    WordPiece splits words into subwords (e.g., "बोलता" -> ["बो", "##लता"]).
    We group subword embeddings by word_id and average them to get one
    768-dim embedding per original word.

    Special tokens [CLS] and [SEP] have word_id=None and are dropped.

    Args:
        model: MuRIL BertModel (frozen).
        tokenizer: BertTokenizerFast.
        text: Ground-truth transcript string.
        device: "cpu" or "cuda".

    Returns:
        word_embeddings: torch.Tensor of shape (N_words, 768), float32.
        words: list[str] — the whitespace-delimited words from the text.
        token_details: dict with debugging info (tokens, word_ids) — only
                       used in --verify mode.
    """
    text = text.strip()

    # Edge case: empty text
    if not text:
        return (
            torch.zeros(0, EMBEDDING_DIM, dtype=torch.float32),
            [],
            {"tokens": [], "word_ids": []},
        )

    words = text.split()

    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get token strings and word_ids for debugging
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_ids = encoding.word_ids(batch_index=0)

    # Forward pass — get last hidden state
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state[0]  # (seq_len, 768)

    # Group subword embeddings by word_id and average
    # word_ids: [None, 0, 0, 1, 2, 2, 2, None]  (None = special tokens)
    word_to_embeds = defaultdict(list)
    for idx, wid in enumerate(word_ids):
        if wid is not None:
            word_to_embeds[wid].append(hidden_states[idx])

    # Build output tensor: one averaged embedding per word
    num_words = len(word_to_embeds)
    if num_words == 0:
        return (
            torch.zeros(0, EMBEDDING_DIM, dtype=torch.float32),
            [],
            {"tokens": tokens, "word_ids": word_ids},
        )

    word_embeddings = torch.zeros(num_words, EMBEDDING_DIM, dtype=torch.float32)
    for wid in range(num_words):
        embeds = word_to_embeds[wid]
        word_embeddings[wid] = torch.stack(embeds).mean(dim=0).cpu().float()

    # Sanity check: word count from tokenizer should match whitespace split
    # MuRIL's WordPiece may merge/split differently, but word_ids should be
    # contiguous 0..N-1 matching the tokenizer's notion of words.
    # If there's a mismatch, we trust word_ids (it's more robust).
    if num_words != len(words):
        # Re-derive words from tokenizer's grouping
        derived_words = []
        for wid in range(num_words):
            # Collect all tokens for this word_id
            word_tokens = [
                tokens[idx] for idx, w in enumerate(word_ids) if w == wid
            ]
            # Join, removing "##" prefix from continuation tokens
            word_str = "".join(
                t[2:] if t.startswith("##") else t for t in word_tokens
            )
            derived_words.append(word_str)
        words = derived_words

    token_details = {"tokens": tokens, "word_ids": word_ids}
    return word_embeddings, words, token_details


# ─────────────────────────────────────────────
# VERIFY MODE
# ─────────────────────────────────────────────

def run_verify(clips, model, tokenizer, device, num_clips=5):
    """
    Process a few clips and print detailed output for debugging.

    Shows the full tokenization pipeline: original text -> WordPiece tokens ->
    word_ids mapping -> averaged word embeddings. Useful for confirming the
    subword-to-word grouping is correct.

    Args:
        clips: List of clip dicts.
        model: MuRIL model.
        tokenizer: MuRIL tokenizer.
        device: "cpu" or "cuda".
        num_clips: Number of clips to verify.
    """
    verify_clips = clips[:num_clips]
    print(f"\n{'='*70}")
    print(f"  VERIFY MODE — processing {len(verify_clips)} clips")
    print(f"{'='*70}\n")

    for i, clip in enumerate(verify_clips):
        text = clip["ground_truth"]
        print(f"--- Clip {i+1}/{len(verify_clips)}: {clip['clip_uid']} ---")
        print(f"  Language:    {clip['language']} ({clip['lang_code']})")
        print(f"  Text:        \"{text}\"")

        word_embeddings, words, details = extract_word_embeddings(
            model, tokenizer, text, device
        )

        print(f"  Tokens:      {details['tokens']}")
        print(f"  Word IDs:    {details['word_ids']}")
        print(f"  Words:       {words}")
        print(f"  Num words:   {len(words)}")
        print(f"  Embed shape: {tuple(word_embeddings.shape)}")

        if word_embeddings.shape[0] > 0:
            norms = word_embeddings.norm(dim=-1)
            print(f"  Embed norms: min={norms.min():.3f}, "
                  f"max={norms.max():.3f}, mean={norms.mean():.3f}")

            # Show per-word details
            for j, word in enumerate(words):
                subwords = [
                    details["tokens"][idx]
                    for idx, wid in enumerate(details["word_ids"])
                    if wid == j
                ]
                print(f"    word[{j}] = \"{word}\" "
                      f"<- {subwords} "
                      f"norm={norms[j]:.3f}")
        print()

    print(f"Verify complete. All {len(verify_clips)} clips processed.\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute MuRIL word embeddings for ASER clips"
    )
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["train", "dev", "test", "all"],
        help="Which split to process. Default: all.",
    )
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Max clips to process per split (for testing). Default: all.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run MuRIL on. Default: cpu.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify mode: process 5 clips with detailed output, then exit.",
    )
    args = parser.parse_args()

    # Determine which splits to process
    if args.split == "all":
        splits = ["train", "dev", "test"]
    else:
        splits = [args.split]

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Load MuRIL
    model, tokenizer = load_muril(device=args.device)
    print()

    # Process each split
    for split in splits:
        csv_path = SPLIT_CSV[split]
        if not csv_path.exists():
            print(f"WARNING: CSV not found: {csv_path} — skipping {split}")
            continue

        output_dir = OUTPUT_BASE / split
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load clips
        clips = load_clips(csv_path, max_clips=args.max_clips)

        # Count by language
        lang_counts = defaultdict(int)
        for c in clips:
            lang_counts[c["lang_code"]] += 1
        lang_str = ", ".join(f"{cnt} {lc.upper()}" for lc, cnt in sorted(lang_counts.items()))

        print(f"{'='*70}")
        print(f"  PRE-COMPUTE MURIL WORD EMBEDDINGS")
        print(f"  Split:  {split}")
        print(f"  CSV:    {csv_path}")
        print(f"  Output: {output_dir}/")
        print(f"  Clips:  {len(clips)} total ({lang_str})")
        print(f"  Device: {args.device}")
        print(f"{'='*70}")

        # Verify mode: detailed output for 5 clips, then skip to next split
        if args.verify:
            run_verify(clips, model, tokenizer, args.device)
            continue

        # Process all clips
        success = 0
        errors = 0
        skipped = 0
        total_words = 0
        empty_texts = 0
        start_time = time.time()

        for i, clip in enumerate(clips):
            clip_uid = clip["clip_uid"]
            save_path = output_dir / f"{clip_uid}.pt"

            # Skip if already computed
            if save_path.exists():
                if i < 3:
                    print(f"  [{i+1}/{len(clips)}] SKIP (exists): {clip_uid}")
                skipped += 1
                success += 1
                continue

            try:
                text = clip["ground_truth"]

                # Handle empty text
                if not text or not text.strip():
                    empty_texts += 1
                    word_embeddings = torch.zeros(0, EMBEDDING_DIM, dtype=torch.float32)
                    words = []
                else:
                    word_embeddings, words, _ = extract_word_embeddings(
                        model, tokenizer, text, args.device
                    )

                # Save to disk
                torch.save({
                    "word_embeddings": word_embeddings,  # (N_words, 768)
                    "words": words,                       # list[str]
                    "num_words": len(words),              # int
                }, save_path)

                total_words += len(words)
                success += 1

                # Progress logging
                if (i + 1) % 100 == 0 or i == len(clips) - 1 or i < 5:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(clips) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1:5d}/{len(clips)}] {clip_uid:30s} "
                        f"lang={clip['lang_code']} words={len(words):3d} "
                        f"shape={tuple(word_embeddings.shape)} "
                        f"[{elapsed:.0f}s, ETA {eta:.0f}s]"
                    )

            except Exception as e:
                print(f"  [{i+1}/{len(clips)}] ERROR {clip_uid}: {e}")
                errors += 1

        # Summary
        elapsed = time.time() - start_time
        processed = success - skipped
        avg_words = total_words / processed if processed > 0 else 0

        print()
        print(f"{'='*70}")
        print(f"  DONE — {split}")
        print(f"  Time:        {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Saved:       {success}/{len(clips)} clips "
              f"({skipped} skipped, {processed} newly computed)")
        print(f"  Errors:      {errors}")
        print(f"  Empty texts: {empty_texts}")
        print(f"  Total words: {total_words:,} (avg {avg_words:.1f} words/clip)")
        print(f"  Output:      {output_dir}/")

        # Disk space
        if processed > 0:
            dir_size = sum(
                f.stat().st_size for f in output_dir.iterdir() if f.suffix == ".pt"
            )
            print(f"  Disk usage:  {dir_size / 1024 / 1024:.1f} MB "
                  f"(~{dir_size / max(success, 1) / 1024:.1f} KB per clip)")

        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
