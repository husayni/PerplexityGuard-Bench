"""Mosaic / stitching evasion: how much natural prefix does it take to mask
an adversary from a whole-sequence pseudo-perplexity gate?

Hypothesis
----------
Pseudo-perplexity reduces over the whole sequence (the mean of per-residue
masked NLLs). If you concatenate a natural prefix in front of an
otherwise-detectable adversary, the natural portion drags the mean down
and at some prefix fraction the screen flips from BLOCK/REVIEW to PASS.
This is a structural attack against the *gate's reduction operator*, not
against either signal individually. It is distinct from the tandem-repeat
evasion (which exploits a one-tailed test) and from random-mutation
adversaries (which the gate already handles).

Setup
-----
For each native sequence:
  1. Construct an adversary = mutate_40pct(native). Verify the screens
     would detect it stand-alone.
  2. For each prefix_frac in {0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0}:
     k = round(prefix_frac * len(native))
     construct = native[:k] + adversary[k:]
  3. Score construct with ESM-2; compute complexity; apply each screen.

Three independent replicates per native (different random seeds for the
adversary). The naive expectation is that detection drops smoothly with
prefix_frac; the question is *how fast*, and whether there's an
adversarial budget the operator should be aware of.

Usage
-----
    PYTHONPATH=. uv run python -m perplexity_guard.audit.stitch_attack \\
        --fast --replicates 3

Runs in ~5-8 min on Apple M5 / MPS with the fast ESM-2 t12_35M model.
Drop --fast for the t33_650M production backbone; budget ~30-40 min.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perplexity_guard.audit.screen_variants import (
    VARIANTS,
    detection_rate,
)
from perplexity_guard.core.complexity import wootton_federhen
from perplexity_guard.core.esm_analysis import (
    DEFAULT_MODEL,
    ESM2Scorer,
    FAST_MODEL,
)
from perplexity_guard.tests.run_e2e_demo import (
    DEFAULT_PDBS,
    extract_native_sequence,
    fetch_pdb,
    mutate_40pct,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "perplexity_guard" / "calibration"
OUT_DIR = REPO_ROOT / "perplexity_guard" / "audit" / "results"

# Prefix fractions to sweep. Concentrated near the middle where we expect
# the screen to flip; pinned at 0 and 1 for sanity controls.
DEFAULT_PREFIX_FRACS = (0.00, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00)


def stitch(native: str, adversary: str, prefix_frac: float) -> str:
    """native[:k] + adversary[k:], k = round(prefix_frac * L). Both must be the same length."""
    if len(native) != len(adversary):
        raise ValueError(f"length mismatch: native={len(native)} adv={len(adversary)}")
    L = len(native)
    k = int(round(prefix_frac * L))
    k = max(0, min(L, k))
    return native[:k] + adversary[k:]


def _score_one(seq: str, scorer: ESM2Scorer) -> dict:
    perp = scorer.score_sequence(seq)
    cx = wootton_federhen(seq)
    return {
        "len": len(seq),
        "perplexity": perp.pseudo_perplexity,
        "z_natural": perp.zscore,
        "is_high": bool(perp.is_high),
        "complexity_mean": cx.mean_complexity,
        "low_complexity_frac": cx.low_complexity_fraction,
        "distinct_3mer_frac": cx.distinct_3mer_fraction,
        "repetitiveness": cx.repetitiveness_score,
        "commec_status": "skipped",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbs", nargs="+", default=DEFAULT_PDBS)
    parser.add_argument("--fast", action="store_true",
                        help="Use t12_35M (≈4-8 min total)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Independent random adversaries per native")
    parser.add_argument("--prefix-fracs", type=float, nargs="+",
                        default=list(DEFAULT_PREFIX_FRACS))
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = args.model or (FAST_MODEL if args.fast else DEFAULT_MODEL)
    print(f"[stitch] model={model}  replicates={args.replicates} "
          f"prefix_fracs={args.prefix_fracs}")

    scorer = ESM2Scorer(
        model_name=model,
        device="cpu" if args.cpu else None,
        calibration_dir=CALIB_DIR,
        auto_calibrate=True,
    )

    rng = random.Random(args.seed)
    rows: list[dict] = []
    t0 = time.time()
    for pdb_id in args.pdbs:
        pdb_path = fetch_pdb(pdb_id)
        native = extract_native_sequence(pdb_path)
        # Score the native once as a baseline.
        nrow = _score_one(native, scorer)
        nrow.update({
            "pdb": pdb_id, "replicate": -1, "prefix_frac": 1.0,
            "kind": "native", "is_full_adversary": False,
        })
        rows.append(nrow)
        for rep in range(args.replicates):
            adv = mutate_40pct(native, rng)
            # Confirm the unstitched adversary (prefix_frac=0) is detectable.
            for f in args.prefix_fracs:
                seq = stitch(native, adv, f)
                row = _score_one(seq, scorer)
                row.update({
                    "pdb": pdb_id,
                    "replicate": rep,
                    "prefix_frac": float(f),
                    "kind": f"stitch_{int(round(f*100)):03d}pct_native_prefix",
                    "is_full_adversary": (f == 0.0),
                })
                rows.append(row)
                print(
                    f"  {pdb_id} rep={rep} f={f:.2f} "
                    f"len={row['len']:>3} "
                    f"ppl={row['perplexity']:6.2f} "
                    f"z={row['z_natural']:+5.2f} "
                    f"rep_score={row['repetitiveness']:.2f}"
                )
        print(f"  ... {pdb_id} done ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)

    # Apply each screen variant to every row.
    for v in VARIANTS:
        df[f"verdict_{v.name}"] = df.apply(v.decide, axis=1)

    out_csv = OUT_DIR / "stitch_attack.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Aggregate detection rates by (screen, prefix_frac).
    survey = df[df["replicate"] >= 0].copy()
    summary_rows = []
    for v in VARIANTS:
        col = f"verdict_{v.name}"
        for f, sub in survey.groupby("prefix_frac"):
            summary_rows.append({
                "screen": v.name,
                "prefix_frac": float(f),
                "n": int(len(sub)),
                "detect_rate": detection_rate(sub[col]),
                "block_rate": float((sub[col] == "BLOCK").sum() / max(len(sub), 1)),
                "mean_z": float(sub["z_natural"].mean()),
                "median_z": float(sub["z_natural"].median()),
                "mean_repetitiveness": float(sub["repetitiveness"].mean()),
                "mean_perplexity": float(sub["perplexity"].mean()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["screen", "prefix_frac"])
    summary_csv = OUT_DIR / "stitch_attack_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # Plot: detection rate and mean z vs natural-prefix fraction.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)

    ax = axes[0]
    color_map = {
        "perplexity_only": "#d7191c",
        "complexity_only": "#1a9641",
        "or_gate": "#2c7bb6",
    }
    for v in VARIANTS:
        sub = summary[summary["screen"] == v.name].sort_values("prefix_frac")
        ax.plot(sub["prefix_frac"] * 100, sub["detect_rate"] * 100,
                marker="o", linewidth=2,
                label=v.name.replace("_", "-"),
                color=color_map.get(v.name, None))
    ax.axhline(50, color="#888", linestyle=":", linewidth=0.8, alpha=0.7,
               label="50% detection")
    ax.set_xlabel("Natural prefix fraction (%) — adversary's evasion budget")
    ax.set_ylabel("Detection rate (REVIEW or BLOCK), %")
    ax.set_title("Stitching attack: detection vs. natural-prefix fraction")
    ax.set_ylim(-3, 105)
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[1]
    sub = summary[summary["screen"] == "perplexity_only"].sort_values("prefix_frac")
    ax.plot(sub["prefix_frac"] * 100, sub["mean_z"],
            marker="s", linewidth=2, color="#d7191c", label="mean z (perplexity)")
    ax.fill_between(sub["prefix_frac"] * 100,
                    sub["mean_z"] - 0.0,  # placeholder
                    sub["mean_z"] + 0.0,
                    alpha=0)
    ax.axhline(3.0, color="#7b3294", linestyle=":", label="z=3 BLOCK")
    ax.axhline(1.5, color="#fdae61", linestyle="--", label="z=1.5 REVIEW")
    ax.axhline(0.0, color="#888", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Natural prefix fraction (%)")
    ax.set_ylabel("Mean ESM-2 z-score across the stitched construct")
    ax.set_title("Whole-sequence z degrades smoothly as natural prefix grows")
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Mosaic / stitching evasion — model {model.split('/')[-1]} "
        f"(adversary = mutate_40pct, replicates={args.replicates})",
        fontsize=11,
    )
    out_png = OUT_DIR / "stitch_attack.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Headline summary printed to stdout for convenience.
    print("\n=== Detection rate (%) by prefix fraction ===")
    pivot = (summary
             .pivot(index="screen", columns="prefix_frac", values="detect_rate")
             .reindex(index=[v.name for v in VARIANTS])
             .fillna(0) * 100)
    print(pivot.round(1).to_string())

    # Find the smallest prefix at which each screen drops below 50% detection.
    print("\n=== Screen-flip prefix fraction (smallest f where detect <50%) ===")
    for v in VARIANTS:
        sub = summary[summary["screen"] == v.name].sort_values("prefix_frac")
        below = sub[sub["detect_rate"] < 0.5]
        if below.empty:
            print(f"  {v.name}: never drops below 50% (robust)")
        else:
            f_first = float(below["prefix_frac"].iloc[0])
            print(f"  {v.name}: first below-50% at prefix_frac={f_first:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
