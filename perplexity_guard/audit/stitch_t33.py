"""Replicate the §4.2 stitch sweep on `facebook/esm2_t33_650M_UR50D`.

Why: the original stitch experiment used `--fast` (t12_35M) for tractability.
Replicating on t33_650M kills the most obvious "different model" critique
without changing any experimental design — same 280 mosaic constructs,
same per-window calibration recipe, just a bigger pLM backbone.

This script:
  1. Builds the per-window calibration on t33_650M (or loads cached).
  2. Re-derives all 280 mosaic constructs (deterministic seed=37) and
     scores each with whole-sequence + sliding-window perplexity.
  3. Writes outputs to *_t33.csv paths so they don't clobber t12 results.
  4. Prints a side-by-side comparison: detection rate by prefix fraction
     and by screen, t12 vs t33.

Runtime: ~30-40 min on Apple M5 / MPS for 280 sequences at t33_650M.

Usage:
    PYTHONPATH=. uv run python -m perplexity_guard.audit.stitch_t33
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from perplexity_guard.audit.screen_variants import (
    VARIANTS,
    apply_variants,
    detection_rate,
)
from perplexity_guard.audit.sliding_window import (
    build_window_calibration,
    score_stitch,
)
from perplexity_guard.core.esm_analysis import DEFAULT_MODEL, ESM2Scorer
from perplexity_guard.tests.run_e2e_demo import DEFAULT_PDBS


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "perplexity_guard" / "calibration"
AUDIT_DIR = REPO_ROOT / "perplexity_guard" / "audit" / "results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--prefix-fracs", type=float, nargs="+",
                        default=[0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0])
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[t33] model={DEFAULT_MODEL}  window={args.window_size}  "
          f"reps={args.replicates}  ETA ~30-40 min on M5/MPS")

    scorer = ESM2Scorer(
        model_name=DEFAULT_MODEL,
        device="cpu" if args.cpu else None,
        calibration_dir=CALIB_DIR,
        auto_calibrate=True,
    )
    window_cal = build_window_calibration(scorer, window_size=args.window_size)
    print(f"[t33] window calibration: mean={window_cal.mean:.2f} "
          f"std={window_cal.std:.2f} n_windows={window_cal.n_windows}")

    t0 = time.time()
    df = score_stitch(
        scorer, window_cal, DEFAULT_PDBS, args.replicates,
        args.prefix_fracs, args.seed,
    )
    print(f"\n[t33] scored {len(df)} rows in {time.time()-t0:.0f}s")

    df = apply_variants(df)
    out_csv = AUDIT_DIR / "stitch_attack_sw_t33.csv"
    df.to_csv(out_csv, index=False)
    print(f"[t33] wrote {out_csv}")

    # Per-screen × prefix detection rate.
    survey = df[df["replicate"] >= 0].copy()
    rows = []
    for v in VARIANTS:
        col = f"verdict_{v.name}"
        for f, sub in survey.groupby("prefix_frac"):
            rows.append({
                "screen": v.name,
                "prefix_frac": float(f),
                "n": int(len(sub)),
                "detect_rate": detection_rate(sub[col]),
                "block_rate": float((sub[col] == "BLOCK").sum() / max(len(sub), 1)),
            })
    summary = pd.DataFrame(rows).sort_values(["screen", "prefix_frac"])
    summary_csv = AUDIT_DIR / "stitch_attack_t33_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[t33] wrote {summary_csv}")

    pivot = (summary
             .pivot(index="screen", columns="prefix_frac", values="detect_rate")
             .reindex(index=[v.name for v in VARIANTS])
             .fillna(0) * 100)
    print("\n=== t33_650M detection rate (%) by prefix fraction ===")
    print(pivot.round(1).to_string())

    # Side-by-side with t12 if present.
    t12_path = AUDIT_DIR / "stitch_attack_v2_summary.csv"
    if t12_path.exists():
        t12 = pd.read_csv(t12_path)
        t12_pivot = (t12
                     .pivot(index="screen", columns="prefix_frac", values="detect_rate")
                     .reindex(index=[v.name for v in VARIANTS])
                     .fillna(0) * 100)
        print("\n=== t12_35M detection rate (%) by prefix fraction (for comparison) ===")
        print(t12_pivot.round(1).to_string())
        print("\n=== Δ (t33 − t12), positive = t33 detects more ===")
        delta = pivot.sub(t12_pivot)
        print(delta.round(1).to_string())
        print()
        print("Headline cells for §4.2 t33 replication paragraph:")
        for screen in ["perplexity_only", "or_gate", "sliding_window_perplexity", "or_gate_v2"]:
            for f in [0.0, 0.50]:
                t12_v = t12_pivot.loc[screen, f] if screen in t12_pivot.index else None
                t33_v = pivot.loc[screen, f] if screen in pivot.index else None
                if t12_v is not None and t33_v is not None:
                    print(f"  {screen:<28} @ {int(f*100):>3}% prefix: "
                          f"t12={t12_v:5.1f}%  t33={t33_v:5.1f}%  Δ={t33_v-t12_v:+5.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
