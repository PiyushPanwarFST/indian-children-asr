#!/usr/bin/env python3
"""Generate experiment folder files (results_summary, predictions_all, predictions_mismatches)
from CSV evaluation results."""

import csv
import os
import re
import string
from collections import Counter
from datetime import date

import jiwer

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = "/home/hp/Indain_children_spech"
DATE_STR = "2026-09-07"

EXPERIMENTS = [
    {
        "folder": "acoustic_distillation",
        "csv_path": f"{BASE}/results/acoustic/three_model_hpc_epoch16_test_1695clips.csv",
        "clip_col": "clip",
        "lang_col": "language",
        "gt_col": "ground_truth",
        "pred_col": "student_trained_whisper_small",
        "model_name": "Acoustic Distillation Student (244M)",
        "model_id": "Whisper Small encoder distilled from Kid-Whisper Medium, decoded with Whisper decoder",
    },
    {
        "folder": "sequential_ctc",
        "csv_path": f"{BASE}/results/semantic/ctc_eval_test_1695clips.csv",
        "clip_col": "file",
        "lang_col": "language",
        "gt_col": "ground_truth",
        "pred_col": "student_ctc",
        "model_name": "Sequential CTC Student (244M)",
        "model_id": "Frozen distilled encoder + CTC head (768→85), greedy decoding",
    },
    {
        "folder": "joint_training",
        "csv_path": f"{BASE}/results/joint/joint_eval_test_1695clips.csv",
        "clip_col": "file",
        "lang_col": "language",
        "gt_col": "ground_truth",
        "pred_col": "student_joint",
        "model_name": "Joint Training Student (244M)",
        "model_id": "MSE+CTC joint training, warm start from acoustic+CTC checkpoints",
    },
]

LANG_ORDER = ["Hindi", "Marathi", "English"]

# ── Normalisation ──────────────────────────────────────────────────────────────

# Punctuation to strip: ASCII punctuation but keep Devanagari characters
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lowercase, strip ASCII punctuation (keep Devanagari), collapse whitespace."""
    text = str(text).lower().strip()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── WER helpers ────────────────────────────────────────────────────────────────


def clip_wer(gt: str, pred: str) -> float:
    """Per-clip WER using jiwer after normalisation."""
    g = normalise(gt)
    p = normalise(pred)
    if not g:
        return 0.0 if not p else 100.0
    return jiwer.wer(g, p) * 100.0


def corpus_wer(gt_list: list[str], pred_list: list[str]) -> float:
    """Corpus-level WER: total edits / total reference words."""
    g = [normalise(t) for t in gt_list]
    p = [normalise(t) for t in pred_list]
    # Filter out empty references
    pairs = [(gi, pi) for gi, pi in zip(g, p) if gi]
    if not pairs:
        return 0.0
    g2, p2 = zip(*pairs)
    return jiwer.wer(list(g2), list(p2)) * 100.0


# ── Word diff ──────────────────────────────────────────────────────────────────


def word_diff(gt: str, pred: str):
    """Return (added_words, removed_words) lists."""
    gt_words = normalise(gt).split()
    pred_words = normalise(pred).split()
    gt_counts = Counter(gt_words)
    pred_counts = Counter(pred_words)
    added = []
    removed = []
    all_words = set(gt_words) | set(pred_words)
    for w in list(pred_words):
        if pred_counts[w] > gt_counts.get(w, 0):
            added.append(w)
            pred_counts[w] -= 1
        else:
            pred_counts[w] -= 1
    # Recount for removed
    gt_counts2 = Counter(gt_words)
    pred_counts2 = Counter(pred_words)
    for w in list(gt_words):
        if gt_counts2[w] > pred_counts2.get(w, 0):
            removed.append(w)
            gt_counts2[w] -= 1
        else:
            gt_counts2[w] -= 1
    return added, removed


# ── Load CSV ───────────────────────────────────────────────────────────────────


def load_csv(cfg: dict) -> list[dict]:
    rows = []
    with open(cfg["csv_path"], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt = r[cfg["gt_col"]]
            pred = r[cfg["pred_col"]]
            rows.append({
                "file": r[cfg["clip_col"]],
                "language": r[cfg["lang_col"]],
                "gt": gt,
                "pred": pred,
                "wer": clip_wer(gt, pred),
            })
    return rows


# ── Generate files ─────────────────────────────────────────────────────────────


def generate(cfg: dict):
    rows = load_csv(cfg)
    out_dir = os.path.join(BASE, "experiments", cfg["folder"])
    os.makedirs(out_dir, exist_ok=True)

    # Group by language
    by_lang = {lang: [] for lang in LANG_ORDER}
    for r in rows:
        lang = r["language"]
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append(r)

    # Compute corpus-level WER per language
    lang_wers = {}
    for lang in LANG_ORDER:
        clips = by_lang.get(lang, [])
        if clips:
            lang_wers[lang] = corpus_wer([c["gt"] for c in clips], [c["pred"] for c in clips])
        else:
            lang_wers[lang] = None

    overall_wer = corpus_wer([r["gt"] for r in rows], [r["pred"] for r in rows])
    total_clips = len(rows)

    # ── results_summary.txt ──
    lines = []
    lines.append(f"# {cfg['model_name']}")
    lines.append(f"# Model ID: {cfg['model_id']}")
    lines.append(f"# Date: {DATE_STR}")
    lines.append(f"# Test set: ASER-Dataset/splits/asr_test.csv")
    lines.append("")
    lines.append("=" * 50)
    lines.append("  Language      Clips        WER")
    lines.append("  -----------------------------------")
    for lang in LANG_ORDER:
        clips = by_lang.get(lang, [])
        n = len(clips)
        w = lang_wers[lang]
        if w is not None:
            lines.append(f"  {lang:<14}{n:>4}    {w:>6.2f}%")
        else:
            lines.append(f"  {lang:<14}{n:>4}       N/A")
    lines.append("  -----------------------------------")
    lines.append(f"  {'Overall':<14}{total_clips:>4}    {overall_wer:>6.2f}%")
    lines.append("=" * 50)
    lines.append("")

    with open(os.path.join(out_dir, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> {out_dir}/results_summary.txt")

    # ── predictions_all.txt ──
    pa = []
    pa.append(f"# {cfg['model_name']} — All Predictions")
    pa.append(f"# Overall WER: {overall_wer:.2f}%")
    pa.append(f"# Test set: {total_clips} clips")
    pa.append(f"# Date: {DATE_STR}")
    pa.append("")
    for lang in LANG_ORDER:
        clips = by_lang.get(lang, [])
        if not clips:
            continue
        w = lang_wers[lang]
        pa.append("=" * 80)
        pa.append(f"  {lang} — {len(clips)} clips — WER: {w:.2f}%")
        pa.append("=" * 80)
        pa.append("")
        for c in clips:
            pa.append(f"File: {c['file']}")
            pa.append(f"WER:  {c['wer']:.1f}%")
            pa.append(f"GT:   {c['gt']}")
            pa.append(f"PRED: {c['pred']}")
            pa.append("-" * 80)
        pa.append("")

    with open(os.path.join(out_dir, "predictions_all.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pa))
    print(f"  -> {out_dir}/predictions_all.txt")

    # ── predictions_mismatches.txt ──
    pm = []
    pm.append(f"# {cfg['model_name']} — Mismatches Only")
    pm.append(f"# Overall WER: {overall_wer:.2f}%")
    pm.append(f"# Only showing clips where prediction ≠ ground truth")
    pm.append("")
    for lang in LANG_ORDER:
        clips = by_lang.get(lang, [])
        if not clips:
            continue
        mismatches = [c for c in clips if normalise(c["gt"]) != normalise(c["pred"])]
        w = lang_wers[lang]
        pm.append("=" * 80)
        pm.append(f"  {lang} — {len(mismatches)}/{len(clips)} mismatches — WER: {w:.2f}%")
        pm.append("=" * 80)
        pm.append("")
        for c in mismatches:
            added, removed = word_diff(c["gt"], c["pred"])
            pm.append(f"File: {c['file']}")
            pm.append(f"WER:  {c['wer']:.1f}%")
            pm.append(f"GT:   {c['gt']}")
            pm.append(f"PRED: {c['pred']}")
            pm.append(f"+ Added:   {added}")
            pm.append(f"- Removed: {removed}")
            pm.append("-" * 80)
        pm.append("")

    with open(os.path.join(out_dir, "predictions_mismatches.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pm))
    print(f"  -> {out_dir}/predictions_mismatches.txt")

    return {
        "model_name": cfg["model_name"],
        "lang_wers": lang_wers,
        "overall_wer": overall_wer,
    }


# ── Update COMPARISON.txt ─────────────────────────────────────────────────────


def update_comparison(results: list[dict]):
    comp_path = os.path.join(BASE, "experiments", "COMPARISON.txt")

    # Read existing content
    with open(comp_path, "r", encoding="utf-8") as f:
        existing = f.read()

    # Build new rows for the 3 experiments
    new_rows = []
    for res in results:
        name = res["model_name"]
        hi = res["lang_wers"].get("Hindi")
        mr = res["lang_wers"].get("Marathi")
        en = res["lang_wers"].get("English")
        ov = res["overall_wer"]
        hi_s = f"{hi:.2f}%" if hi is not None else "N/A"
        mr_s = f"{mr:.2f}%" if mr is not None else "N/A"
        en_s = f"{en:.2f}%" if en is not None else "N/A"
        ov_s = f"{ov:.2f}%"
        new_rows.append(f"  {name:<44}{hi_s:>9}{mr_s:>10}{en_s:>10}{ov_s:>10}")

    # Build updated file
    lines = []
    lines.append("# ASR Model Comparison on ASER Test Set")
    lines.append(f"# Date: {DATE_STR}")
    lines.append("# Test set: 1695 clips (782 Hindi, 366 Marathi, 547 English)")
    lines.append("")
    lines.append("=" * 85)
    lines.append("  Model                                         Hindi    Marathi    English    Overall")
    lines.append("  " + "-" * 75)
    # Old baselines (keep as-is)
    lines.append("  Whisper Small (244M)                        161.81%    170.78%     47.42%    126.83%")
    lines.append("  IndicConformer 600M                          39.01%     44.61%        N/A     59.90%")
    lines.append("  Kid-Whisper Medium MyST (769M)              454.66%    656.45%     37.35%    363.56%")
    lines.append("  Kid-Whisper Medium MyST+CSLU (769M)         714.32%   1083.20%     58.46%    582.31%")
    lines.append("  " + "-" * 75)
    lines.append("  # CARE Pipeline Experiments (Phase 1-3)")
    lines.append("  " + "-" * 75)
    for row in new_rows:
        lines.append(row)
    lines.append("=" * 85)
    lines.append("")

    # Preserve the role assignment section from existing
    role_idx = existing.find("# Role Assignment")
    if role_idx != -1:
        lines.append(existing[role_idx:].rstrip())
        lines.append("")

    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  -> Updated {comp_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []
    for cfg in EXPERIMENTS:
        print(f"\n[{cfg['folder']}]")
        res = generate(cfg)
        all_results.append(res)

    update_comparison(all_results)
    print("\nDone.")
