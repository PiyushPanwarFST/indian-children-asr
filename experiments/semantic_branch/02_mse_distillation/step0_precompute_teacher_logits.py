"""
Step 0: Pre-Compute IndicConformer Teacher Logits (One-Time Offline)
====================================================================
Runs IndicConformer on all Hindi/Marathi clips and saves the RAW LOGITS
(per-frame scores over the 257-token vocabulary) to disk.

WHY pre-compute?
    IndicConformer takes ~2-4 seconds per clip via ONNX.
    If we run it during training: 9,169 clips × 4s × 20 epochs = ~200 hours.
    Pre-computing once takes ~10 hours. Then training just loads from disk.
    (Same approach we used for Kid-Whisper features in acoustic branch.)

WHAT we save per clip:
    - logits: (T_enc, 257) — raw CTC decoder scores BEFORE log_softmax
      T_enc varies by audio length (e.g., 63 frames for 5s audio)
      257 = 256 BPE subword tokens + 1 blank (per language)
    - num_frames: int — actual encoder output frames
    - language: str — "hi" or "mr"

WHY raw logits (not log_softmax)?
    MSE distillation works on raw logits — this preserves the teacher's
    full confidence landscape. log_softmax compresses values into [-inf, 0],
    losing relative magnitudes. Hinton et al. (2015) recommend raw logits.

VERIFICATION built in:
    After saving, the script decodes each clip from saved logits and
    compares with IndicConformer's normal transcription to confirm
    the logits are correct.

Usage:
    # Quick test — verify with 5 clips
    python scripts/semantic/step0_precompute_teacher_logits.py --max_clips 5

    # Full run — all Hindi/Marathi training clips
    python scripts/semantic/step0_precompute_teacher_logits.py

    # Also pre-compute dev set
    python scripts/semantic/step0_precompute_teacher_logits.py --eval_set dev
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torchaudio

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds — Whisper's max, also reasonable for Conformer

TRAIN_CSV = "ASER-Dataset/splits/asr_train.csv"
DEV_CSV = "ASER-Dataset/splits/asr_dev.csv"
TEST_CSV = "ASER-Dataset/splits/asr_test.csv"

# Output directory for saved logits
OUTPUT_DIR = "teacher_logits"

# ─────────────────────────────────────────────
# LOAD INDICCONFORMER
# ─────────────────────────────────────────────

def load_indicconformer():
    """
    Load IndicConformer 600M model.

    This model uses ONNX runtime internally (not PyTorch).
    It has separate components: preprocessor, encoder, ctc_decoder.
    We need to access these components directly to extract raw logits
    (the model.forward() method only returns decoded text, not logits).

    Returns:
        model: IndicASRModel instance
        vocab: dict — per-language token lists, e.g., vocab["hi"] = ["<unk>", "▁क", ...]
        language_masks: dict — per-language boolean masks over 5633-dim output
    """
    from transformers import AutoModel

    print("Loading ai4bharat/indic-conformer-600m-multilingual...")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    )
    print(f"  Loaded. Device: {model.d}")
    print(f"  Languages: {list(model.vocab.keys())}")
    print(f"  Hindi vocab size: {len(model.vocab['hi'])}")
    print(f"  Marathi vocab size: {len(model.vocab['mr'])}")
    print(f"  BLANK_ID: {model.config.BLANK_ID}")

    return model


def extract_teacher_logits(model, wav, lang):
    """
    Run IndicConformer on audio and extract RAW logits before log_softmax.

    The normal model.forward() returns decoded text only.
    We need to manually call encoder + ctc_decoder to get logits.

    Pipeline:
        wav → preprocessor.ts → (1, 80, T_mel)
            → encoder.onnx → (1, 1024, T_enc)
            → ctc_decoder.onnx → (1, T_enc, 5633) raw scores
            → language_mask → (1, T_enc, 257) per-language logits

    Args:
        model: IndicASRModel instance
        wav: torch.Tensor, shape (1, num_samples)
        lang: str, "hi" or "mr"

    Returns:
        logits: torch.Tensor, shape (T_enc, 257) — raw scores before log_softmax
        num_frames: int — T_enc (actual encoder frames)
        transcript: str — decoded text (for verification)
    """
    # Step 1: Encode audio → encoder features
    # model.encode() runs preprocessor + encoder ONNX models
    encoder_outputs, encoded_lengths = model.encode(wav)
    # encoder_outputs: numpy array (1, 1024, T_enc)
    # encoded_lengths: numpy array (1,) — actual frame count

    num_frames = int(encoded_lengths[0])

    # Step 2: Run CTC decoder → raw logits over full 5633-dim vocab
    raw_logits_np = model.models['ctc_decoder'].run(
        ['logprobs'],
        {'encoder_output': encoder_outputs}
    )[0]
    # raw_logits_np: numpy array (1, T_enc, 5633)

    # Step 3: Apply language mask to select this language's 257 tokens
    # language_masks[lang] is a boolean list of length 5633
    # True at positions belonging to this language (256 tokens + blank)
    #
    # IMPORTANT: Must convert to torch first, then apply boolean mask.
    # numpy advanced boolean indexing on raw[0, :T, mask] transposes the result!
    # torch [:, mask] correctly gives (T, 257).
    mask = model.language_masks[lang]
    logits_all = torch.from_numpy(raw_logits_np[0, :num_frames, :])  # (T_enc, 5633)
    logits = logits_all[:, mask]  # (T_enc, 257)
    # logits: (T_enc, 257) — raw scores, NOT log_softmax'd
    # This is what we save for MSE distillation

    # Step 4: Also decode for verification
    # The normal decoding pipeline does log_softmax AFTER masking,
    # then argmax → collapse repeats → remove blanks → join tokens
    log_probs = logits.log_softmax(dim=-1)
    indices = torch.argmax(log_probs, dim=-1)
    collapsed = torch.unique_consecutive(indices, dim=-1)
    transcript = ''.join([
        model.vocab[lang][idx] for idx in collapsed
        if idx != model.config.BLANK_ID
    ]).replace('▁', ' ').strip()

    return logits.to(torch.float16), num_frames, transcript


# ─────────────────────────────────────────────
# LOAD CLIPS FROM CSV
# ─────────────────────────────────────────────

def load_clips(csv_path, max_clips=None):
    """
    Load Hindi/Marathi clips from ASER split CSV.
    Skips English clips (IndicConformer has no English support).

    Returns:
        list of dicts with keys: audio_path, language, lang_code, ground_truth, filename, clip_uid
    """
    clips = []
    lang_map = {"Hindi": "hi", "Marathi": "mr"}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row.get("language", "")
            lang_code = lang_map.get(lang)
            if lang_code is None:
                # Skip English and unknown languages
                continue

            audio_path = row["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join("ASER-Dataset", audio_path)

            # Use child_id + basename as unique identifier
            # Multiple children read the same passage → same filename
            # e.g., 571 clips all named HI_S1_P_0.wav from different children
            # basename includes chunk suffix (e.g., HI_S1_ST_0_chunk0) for uniqueness
            child_id = row.get("child_id", "")
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            clip_uid = f"{child_id}_{basename}" if child_id else basename

            clips.append({
                "audio_path": audio_path,
                "language": lang,
                "lang_code": lang_code,
                "ground_truth": row.get("transcript", row.get("que_text", row.get("text", ""))),
                "filename": os.path.basename(audio_path),
                "clip_uid": clip_uid,
            })

    if max_clips and max_clips < len(clips):
        clips = clips[:max_clips]

    return clips


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute IndicConformer teacher logits")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Max clips to process (for testing). Default: all.")
    parser.add_argument("--eval_set", type=str, default="train",
                        choices=["train", "dev", "test"],
                        help="Which split to process. Default: train.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing logits, don't compute new ones.")
    args = parser.parse_args()

    # Select CSV based on eval_set
    csv_map = {"train": TRAIN_CSV, "dev": DEV_CSV, "test": TEST_CSV}
    csv_path = csv_map[args.eval_set]

    # Output subdirectory per split
    output_dir = os.path.join(OUTPUT_DIR, args.eval_set)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  PRE-COMPUTE TEACHER LOGITS (IndicConformer 600M)")
    print(f"  Split: {args.eval_set}")
    print(f"  CSV: {csv_path}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}")

    # Load clips
    clips = load_clips(csv_path, max_clips=args.max_clips)
    hi_count = sum(1 for c in clips if c['lang_code'] == 'hi')
    mr_count = sum(1 for c in clips if c['lang_code'] == 'mr')
    print(f"  Clips: {len(clips)} total ({hi_count} Hindi, {mr_count} Marathi)")
    print()

    # Load model
    model = load_indicconformer()
    print()

    # Process each clip
    success = 0
    errors = 0
    verify_matches = 0
    verify_total = 0
    total_frames = 0
    start_time = time.time()

    for i, clip in enumerate(clips):
        clip_name = clip['clip_uid']
        save_path = os.path.join(output_dir, f"{clip_name}.pt")

        if args.verify_only:
            # Just verify existing files
            if not os.path.exists(save_path):
                print(f"  [{i+1}/{len(clips)}] MISSING: {save_path}")
                errors += 1
                continue
            saved = torch.load(save_path, weights_only=True)
            print(f"  [{i+1}/{len(clips)}] {clip_name}: shape={tuple(saved['logits'].shape)}, "
                  f"frames={saved['num_frames']}, lang={saved['language']}")
            success += 1
            continue

        # Skip if already computed
        if os.path.exists(save_path):
            if i < 3:  # Print first few skips
                print(f"  [{i+1}/{len(clips)}] SKIP (exists): {clip_name}")
            success += 1
            continue

        try:
            # Load audio
            wav, sr = torchaudio.load(clip['audio_path'])
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav[:, :MAX_DURATION * SAMPLE_RATE]

            # Extract logits
            logits, num_frames, transcript = extract_teacher_logits(
                model, wav, clip['lang_code']
            )
            total_frames += num_frames

            # Save to disk
            torch.save({
                'logits': logits,           # (T_enc, 257) float16
                'num_frames': num_frames,   # int
                'language': clip['lang_code'],  # "hi" or "mr"
            }, save_path)

            # Verification: also run normal forward pass and compare
            normal_transcript = model.forward(wav, lang=clip['lang_code'], decoding='ctc')
            verify_total += 1
            if transcript == normal_transcript:
                verify_matches += 1
                match_str = "✓ MATCH"
            else:
                match_str = f"✗ MISMATCH"

            # Progress logging
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(clips) - i - 1) / rate if rate > 0 else 0

            if i < 10 or (i + 1) % 100 == 0 or i == len(clips) - 1:
                print(f"  [{i+1:5d}/{len(clips)}] {clip_name:25s} "
                      f"lang={clip['lang_code']} frames={num_frames:4d} "
                      f"logits={tuple(logits.shape)} {match_str} "
                      f"[{elapsed:.0f}s, ETA {eta:.0f}s]")
                if match_str.startswith("✗"):
                    print(f"           Normal:  {normal_transcript[:80]}")
                    print(f"           FromLog: {transcript[:80]}")

            success += 1

        except Exception as e:
            print(f"  [{i+1}/{len(clips)}] ERROR {clip_name}: {e}")
            errors += 1

    # Summary
    elapsed = time.time() - start_time
    print()
    print(f"{'='*70}")
    print(f"  DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Saved: {success}/{len(clips)} clips")
    print(f"  Errors: {errors}")
    print(f"  Total frames saved: {total_frames:,}")
    if verify_total > 0:
        print(f"  Verification: {verify_matches}/{verify_total} decoded correctly "
              f"({100*verify_matches/verify_total:.1f}%)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}")

    # Disk space estimate
    if success > 0 and not args.verify_only:
        dir_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir) if f.endswith('.pt')
        )
        print(f"  Disk usage: {dir_size / 1024 / 1024:.1f} MB "
              f"(~{dir_size / success / 1024:.1f} KB per clip)")


if __name__ == "__main__":
    main()
