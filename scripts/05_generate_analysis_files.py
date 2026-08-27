"""
Script 5: Generate Analysis Files from baseline_results.csv

Generates 4 files for professor review (no re-running models needed):
  1. predictions_whisper.csv   — all Whisper predictions with WER
  2. predictions_kidwhisper.csv — all Kid-Whisper predictions with WER
  3. mismatches_worst.csv      — 200 worst predictions per model (highest WER)
  4. summary_report.txt        — full text report with tables and examples
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from jiwer import wer as compute_wer

ASER_ROOT   = Path("/home/hp/Indain_children_spech/ASER-Dataset")
RESULTS_CSV = ASER_ROOT / "baseline_results.csv"
OUT_DIR     = ASER_ROOT / "baseline_analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
with open(RESULTS_CSV, encoding="utf-8") as f:
    all_results = list(csv.DictReader(f))

for r in all_results:
    r["wer"] = float(r["wer"])

print(f"Loaded {len(all_results)} rows from {RESULTS_CSV}")

models    = ["Whisper-Small", "Kid-Whisper-EN"]
languages = ["Hindi", "Marathi", "English"]

# ── File 1 & 2: Per-model prediction files ───────────────────────────────────
fields = ["clip_id", "language", "reading_level", "duration_sec",
          "reference", "hypothesis", "wer", "audio_path"]

# Need reading_level — load from asr_test.csv
test_path_to_meta = {}
with open(ASER_ROOT / "splits" / "asr_test.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        test_path_to_meta[row["audio_path"]] = row.get("reading_level", "")

for model in models:
    model_rows = [r for r in all_results if r["model"] == model]
    safe_name  = model.lower().replace("-", "_").replace(" ", "_")
    out_path   = OUT_DIR / f"predictions_{safe_name}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(model_rows):
            writer.writerow({
                "clip_id":       i + 1,
                "language":      r["language"],
                "reading_level": test_path_to_meta.get(r["audio_path"], ""),
                "duration_sec":  r["duration_sec"],
                "reference":     r["reference"],
                "hypothesis":    r["hypothesis"],
                "wer":           f"{r['wer']:.4f}",
                "audio_path":    r["audio_path"],
            })
    print(f"Saved: {out_path} ({len(model_rows)} rows)")

# ── File 3: Worst mismatches (highest WER, non-empty hypothesis only) ─────────
mismatch_fields = ["rank", "model", "language", "wer", "reference", "hypothesis",
                   "duration_sec", "audio_path"]
mismatch_path = OUT_DIR / "mismatches_worst200.csv"

# Collect worst 100 per model where hypothesis is non-empty
worst = []
for model in models:
    model_rows = [r for r in all_results
                  if r["model"] == model and r["hypothesis"].strip()]
    model_rows_sorted = sorted(model_rows, key=lambda x: x["wer"], reverse=True)
    worst.extend(model_rows_sorted[:100])

worst = sorted(worst, key=lambda x: x["wer"], reverse=True)

with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=mismatch_fields)
    writer.writeheader()
    for rank, r in enumerate(worst, 1):
        writer.writerow({
            "rank":       rank,
            "model":      r["model"],
            "language":   r["language"],
            "wer":        f"{r['wer']:.4f}",
            "reference":  r["reference"],
            "hypothesis": r["hypothesis"],
            "duration_sec": r["duration_sec"],
            "audio_path": r["audio_path"],
        })
print(f"Saved: {mismatch_path} ({len(worst)} rows)")

# ── File 4: Full summary text report ─────────────────────────────────────────
report_path = OUT_DIR / "summary_report.txt"

def corpus_wer(rows):
    refs = " ".join(r["reference"] for r in rows)
    hyps = " ".join(r["hypothesis"] for r in rows)
    if not refs.strip() or not hyps.strip():
        return None
    try:
        return compute_wer(refs, hyps)
    except:
        return None

def avg_wer(rows):
    if not rows:
        return None
    return np.mean([r["wer"] for r in rows])

def empty_rate(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if not r["hypothesis"].strip()) / len(rows)

with open(report_path, "w", encoding="utf-8") as f:

    f.write("=" * 75 + "\n")
    f.write("  ASER DATASET — BASELINE EVALUATION REPORT\n")
    f.write("  Models: Whisper Small (OpenAI) | Kid-Whisper-EN (Attia et al.)\n")
    f.write("  Test set: asr_test.csv — 1,695 clips | 5.79 hours\n")
    f.write("  Languages: Hindi, Marathi, Indian-accented English\n")
    f.write("=" * 75 + "\n\n")

    # ── Section 1: Test set composition ──────────────────────────────────────
    f.write("SECTION 1: TEST SET COMPOSITION\n")
    f.write("-" * 40 + "\n")
    for lang in languages:
        lang_rows = [r for r in all_results
                     if r["language"] == lang and r["model"] == "Whisper-Small"]
        total_dur = sum(float(r["duration_sec"]) for r in lang_rows) / 3600
        f.write(f"  {lang:<10} {len(lang_rows):>5} clips | {total_dur:.2f} hours\n")
    total_clips = len(all_results) // len(models)
    total_dur_all = sum(float(r["duration_sec"]) for r in all_results
                        if r["model"] == "Whisper-Small") / 3600
    f.write(f"  {'TOTAL':<10} {total_clips:>5} clips | {total_dur_all:.2f} hours\n\n")

    # ── Section 2: Corpus-level WER table ────────────────────────────────────
    f.write("SECTION 2: CORPUS-LEVEL WER (standard ASR paper metric)\n")
    f.write("  (All reference text concatenated, all hypothesis text concatenated,\n")
    f.write("   single WER computed. Gives equal weight per word, not per clip.)\n")
    f.write("-" * 75 + "\n")
    f.write(f"  {'Model':<20} {'Hindi':>10} {'Marathi':>10} {'English':>10} {'Overall':>10}\n")
    f.write(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")

    for model in models:
        f.write(f"  {model:<20}")
        all_refs, all_hyps = [], []
        for lang in languages:
            rows = [r for r in all_results
                    if r["model"] == model and r["language"] == lang]
            cw = corpus_wer(rows)
            if cw is not None:
                f.write(f" {cw:>9.1%}")
                all_refs.extend(r["reference"] for r in rows)
                all_hyps.extend(r["hypothesis"] for r in rows)
            else:
                f.write(f" {'N/A (empty)':>10}")
        # Overall
        full_ref = " ".join(all_refs)
        full_hyp = " ".join(all_hyps)
        if full_ref.strip() and full_hyp.strip():
            ow = compute_wer(full_ref, full_hyp)
            f.write(f" {ow:>9.1%}\n")
        else:
            f.write(f" {'N/A':>10}\n")

    f.write("\n")

    # ── Section 3: Average per-clip WER ──────────────────────────────────────
    f.write("SECTION 3: AVERAGE PER-CLIP WER\n")
    f.write("-" * 75 + "\n")
    f.write(f"  {'Model':<20} {'Hindi':>10} {'Marathi':>10} {'English':>10} {'Overall':>10}\n")
    f.write(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")

    for model in models:
        f.write(f"  {model:<20}")
        all_wers = []
        for lang in languages:
            rows = [r for r in all_results
                    if r["model"] == model and r["language"] == lang]
            aw = avg_wer(rows)
            if aw is not None:
                f.write(f" {aw:>9.1%}")
                all_wers.extend(r["wer"] for r in rows)
            else:
                f.write(f" {'N/A':>10}")
        if all_wers:
            f.write(f" {np.mean(all_wers):>9.1%}\n")
        else:
            f.write(f" {'N/A':>10}\n")

    f.write("\n")

    # ── Section 4: Empty output rate ─────────────────────────────────────────
    f.write("SECTION 4: EMPTY OUTPUT RATE (model produced no text)\n")
    f.write("-" * 75 + "\n")
    f.write(f"  {'Model':<20} {'Hindi':>10} {'Marathi':>10} {'English':>10}\n")
    f.write(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}\n")

    for model in models:
        f.write(f"  {model:<20}")
        for lang in languages:
            rows = [r for r in all_results
                    if r["model"] == model and r["language"] == lang]
            er = empty_rate(rows)
            f.write(f" {er:>9.1%}")
        f.write("\n")

    f.write("\n")

    # ── Section 5: Sample predictions ────────────────────────────────────────
    f.write("SECTION 5: SAMPLE PREDICTIONS (5 per model per language)\n")
    f.write("=" * 75 + "\n")

    for model in models:
        for lang in languages:
            rows = [r for r in all_results
                    if r["model"] == model and r["language"] == lang]
            f.write(f"\n  [{model}] [{lang}] — {len(rows)} clips total\n")
            f.write("  " + "-" * 70 + "\n")
            # Show mix: some good, some bad (sort by WER, pick spread)
            rows_sorted = sorted(rows, key=lambda x: x["wer"])
            picks = []
            if len(rows_sorted) >= 5:
                # best, 25th pct, median, 75th pct, worst with non-empty hyp
                non_empty = [r for r in rows_sorted if r["hypothesis"].strip()]
                if non_empty:
                    n = len(non_empty)
                    picks = [
                        non_empty[0],
                        non_empty[n//4],
                        non_empty[n//2],
                        non_empty[3*n//4],
                        non_empty[-1],
                    ]
                else:
                    picks = rows_sorted[:5]
            else:
                picks = rows_sorted

            for i, r in enumerate(picks):
                label = ["BEST", "25th%", "MEDIAN", "75th%", "WORST"][i] if len(picks) == 5 else str(i+1)
                f.write(f"\n  [{label}] WER: {r['wer']:.0%} | dur: {float(r['duration_sec']):.1f}s\n")
                f.write(f"  REF: {r['reference'][:90]}\n")
                f.write(f"  HYP: {r['hypothesis'][:90] if r['hypothesis'].strip() else '(empty)'}\n")

    f.write("\n\n")

    # ── Section 6: WER distribution histogram ────────────────────────────────
    f.write("SECTION 6: WER DISTRIBUTION\n")
    f.write("-" * 75 + "\n")
    buckets = [(0, 0.25, "0-25%"), (0.25, 0.5, "25-50%"), (0.5, 0.75, "50-75%"),
               (0.75, 1.0, "75-100%"), (1.0, 999, ">100%")]

    for model in models:
        f.write(f"\n  {model}:\n")
        rows = [r for r in all_results if r["model"] == model]
        total = len(rows)
        f.write(f"  {'WER Range':<12} {'Count':>8} {'%':>8}\n")
        for lo, hi, label in buckets:
            count = sum(1 for r in rows if lo <= r["wer"] < hi)
            f.write(f"  {label:<12} {count:>8} {count/total:>7.1%}\n")

    f.write("\n")
    f.write("=" * 75 + "\n")
    f.write("  END OF REPORT\n")
    f.write("  Generated from: baseline_results.csv (3,390 predictions)\n")
    f.write("=" * 75 + "\n")

print(f"Saved: {report_path}")
print(f"\nAll files in: {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name:<45} {size_kb:.1f} KB")
