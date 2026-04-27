"""Position-revert attack: measure each screen's detection vs. evasion budget.

Setup
-----
For each native sequence, construct an adversary by replacing 40% of
positions with random AAs (the canonical mutated_40pct proxy used in the
ESM-2 validation). Then walk the adversary back toward the native by
reverting K of the mutated positions, where K varies from 0 (full
adversary) to 40% (back to native). At each K, score every variant with
ESM-2 and apply each screen.

Output
------
A CSV (`revert_attack.csv`) with one row per (native, K, replicate) and
a figure (`revert_attack.png`) showing detection rate as a function of
the fraction of mutations that survived (the attacker's "evasion
budget").

Usage
-----
    PYTHONPATH=. uv run python -m perplexity_guard.audit.revert_attack \\
        --fast --replicates 3 --steps 5
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
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "perplexity_guard" / "calibration"
OUT_DIR = REPO_ROOT / "perplexity_guard" / "audit" / "results"

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def make_adversary_and_track(
    native: str, frac: float, rng: random.Random
) -> tuple[str, list[int], list[str]]:
    """Mutate ``frac`` of positions; return mutated seq + the positions
    + the original residues at those positions, so we can revert later.
    """
    chars = list(native)
    n_mut = max(1, int(frac * len(native)))
    mut_positions = rng.sample(range(len(native)), n_mut)
    original_residues = [native[p] for p in mut_positions]
    for p in mut_positions:
        new_aa = rng.choice([a for a in AA20 if a != native[p]])
        chars[p] = new_aa
    return "".join(chars), mut_positions, original_residues


def revert_K(
    adversary: str, mut_positions: list[int], original_residues: list[str], k: int
) -> str:
    """Revert ``k`` of the mutated positions back to native."""
    if k <= 0:
        return adversary
    chars = list(adversary)
    for p, orig in zip(mut_positions[:k], original_residues[:k]):
        chars[p] = orig
    return "".join(chars)


def _score_row(seq: str, scorer: ESM2Scorer) -> dict:
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
                        help="Use t12_35M (faster, smaller calibration mismatch)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Independent random adversaries per native")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of revert-fraction steps in [0, 1.0]")
    parser.add_argument("--start-frac", type=float, default=0.40,
                        help="Initial mutation fraction (default: 0.40)")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = args.model or (FAST_MODEL if args.fast else DEFAULT_MODEL)
    print(f"[revert] model={model}  replicates={args.replicates} steps={args.steps}")

    scorer = ESM2Scorer(
        model_name=model,
        device="cpu" if args.cpu else None,
        calibration_dir=CALIB_DIR,
        auto_calibrate=True,
    )

    # Revert fractions: 0 = full adversary; 1.0 = back to native.
    revert_fracs = np.linspace(0.0, 1.0, args.steps + 1)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    t0 = time.time()
    for pdb_id in args.pdbs:
        pdb_path = fetch_pdb(pdb_id)
        native = extract_native_sequence(pdb_path)
        # Score the native once as a baseline.
        nrow = _score_row(native, scorer)
        nrow.update({
            "pdb": pdb_id, "replicate": -1, "revert_frac": float("nan"),
            "n_surviving_muts": 0, "kind": "native",
        })
        rows.append(nrow)
        for rep in range(args.replicates):
            adv, positions, originals = make_adversary_and_track(
                native, args.start_frac, rng,
            )
            n_total_muts = len(positions)
            for f in revert_fracs:
                k = int(round(f * n_total_muts))
                seq = revert_K(adv, positions, originals, k)
                row = _score_row(seq, scorer)
                row.update({
                    "pdb": pdb_id,
                    "replicate": rep,
                    "revert_frac": float(f),
                    "n_surviving_muts": n_total_muts - k,
                    "frac_surviving_muts": (n_total_muts - k) / max(n_total_muts, 1),
                    "kind": f"revert_{int(f*100):03d}pct",
                })
                rows.append(row)
                print(
                    f"  {pdb_id} rep={rep} f={f:.2f} "
                    f"surv={n_total_muts - k:>2} "
                    f"ppl={row['perplexity']:6.2f} z={row['z_natural']:+5.2f} "
                    f"rep={row['repetitiveness']:.2f}"
                )
        print(f"  ... {pdb_id} done ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)

    # Apply each screen variant to every row.
    for v in VARIANTS:
        df[f"verdict_{v.name}"] = df.apply(v.decide, axis=1)

    out_csv = OUT_DIR / "revert_attack.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Aggregate: detection rate vs. surviving-mutations fraction.
    survey = df[df["replicate"] >= 0].copy()
    survey["bucket"] = (survey["frac_surviving_muts"] * 100).round().astype(int)
    summary_rows = []
    for v in VARIANTS:
        col = f"verdict_{v.name}"
        for bucket, sub in survey.groupby("bucket"):
            summary_rows.append({
                "screen": v.name,
                "surviving_muts_pct": int(bucket),
                "n": int(len(sub)),
                "detect_rate": detection_rate(sub[col]),
                "mean_z": float(sub["z_natural"].mean()),
                "mean_repetitiveness": float(sub["repetitiveness"].mean()),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["screen", "surviving_muts_pct"])
    summary_csv = OUT_DIR / "revert_attack_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # Plot: detection vs. evasion budget.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)

    ax = axes[0]
    for v in VARIANTS:
        sub = summary[summary["screen"] == v.name]
        ax.plot(sub["surviving_muts_pct"], sub["detect_rate"] * 100,
                marker="o", linewidth=2, label=v.name.replace("_", "-"))
    ax.set_xlabel("Mutations surviving (%) — attacker's evasion budget")
    ax.set_ylabel("Detection rate (REVIEW or BLOCK)")
    ax.set_title("Detection vs. attacker evasion budget")
    ax.set_ylim(-3, 105)
    ax.set_xlim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    # Mean z-score (perplexity signal) vs. surviving mutations
    for v in VARIANTS[:1]:  # only perplexity for clarity here
        sub = summary[summary["screen"] == v.name]
        ax.plot(sub["surviving_muts_pct"], sub["mean_z"],
                marker="s", linewidth=2, color="#d7191c", label="mean z-score")
    ax.axhline(3.0, color="#7b3294", linestyle=":", label="z=3 BLOCK")
    ax.axhline(1.5, color="#fdae61", linestyle="--", label="z=1.5 REVIEW")
    ax.axhline(0.0, color="#888", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Mutations surviving (%)")
    ax.set_ylabel("Mean ESM-2 z-score (vs. natural)")
    ax.set_title("Perplexity signal degrades smoothly as attacker reverts")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"Position-revert attack — model {model.split('/')[-1]}, "
        f"start mutation fraction = {args.start_frac:.0%}",
        fontsize=11,
    )
    out_png = OUT_DIR / "revert_attack.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    print("\n=== Summary ===")
    print((summary
           .pivot(index="screen", columns="surviving_muts_pct", values="detect_rate")
           .fillna(0) * 100)
          .round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
