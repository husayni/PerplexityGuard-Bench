"""Run the screen × evasion audit on the existing demo CSV.

For each (screen variant, evasion class) pair, compute the detection rate
(P(REVIEW or BLOCK)) and block rate (P(BLOCK)). Output a CSV + a
publication-quality figure.

Usage:
    uv run python -m perplexity_guard.audit.run_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perplexity_guard.audit.screen_variants import (
    VARIANTS,
    apply_variants,
    block_rate,
    detection_rate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = REPO_ROOT / "perplexity_guard" / "demo_results" / "e2e_demo.csv"
OUT_DIR = REPO_ROOT / "perplexity_guard" / "audit" / "results"


# Display order for the evasion-class axis.
KIND_ORDER = [
    "native",
    "mpnn_T=0.1",
    "mpnn_T=1.0",
    "mutated_40pct",
    "shuffled",
    "repeat_motif",
]
KIND_PRETTY = {
    "native": "Native\n(control)",
    "mpnn_T=0.1": "ProteinMPNN\nT=0.1",
    "mpnn_T=1.0": "ProteinMPNN\nT=1.0",
    "mutated_40pct": "40%-mutated\nproxy",
    "shuffled": "Shuffled\nproxy",
    "repeat_motif": "Tandem\nrepeat",
}


def _matrix(df: pd.DataFrame, kinds: list[str]) -> pd.DataFrame:
    """Build the (variant × kind) detection-rate matrix.

    Skips screen variants whose required input columns are missing from
    ``df`` — for example, ``sliding_window_perplexity`` and ``or_gate_v2``
    need a ``max_window_z`` column populated by ``audit/sliding_window.py``,
    so they are skipped when ``df`` is the legacy ``e2e_demo.csv`` that
    only carries whole-sequence signals. This avoids misleading 0%-detection
    rows that would falsely imply the sliding-window patch regresses on
    the §4.1 main matrix when it simply wasn't computed there.
    """
    rows = []
    has_max_window = "max_window_z" in df.columns
    sliding_screens = {
        "sliding_window_perplexity",
        "sliding_window_perplexity_tight",
        "or_gate_v2",
        "or_gate_v3_tight",
    }
    for v in VARIANTS:
        if v.name in sliding_screens and not has_max_window:
            continue
        col = f"verdict_{v.name}"
        for k in kinds:
            sub = df[df["kind"] == k]
            rows.append({
                "screen": v.name,
                "kind": k,
                "n": int(len(sub)),
                "detect_rate": detection_rate(sub[col]),
                "block_rate": block_rate(sub[col]),
                "review_rate": float(((sub[col] == "REVIEW")).sum() / max(len(sub), 1)),
                "pass_rate": float(((sub[col] == "PASS")).sum() / max(len(sub), 1)),
            })
    return pd.DataFrame(rows)


_SCREEN_LABELS = {
    "perplexity_only": "perplexity-only\n(whole-sequence z)",
    "complexity_only": "complexity-only\n(WF + distinct-k-mer)",
    "or_gate": "OR-gate\n(perplexity + complexity)",
    "sliding_window_perplexity": "sliding-window\n(per-window z)",
    "or_gate_v2": "OR-gate v2\n(sliding-window + cx)",
}


def _plot_matrix(matrix: pd.DataFrame, kinds: list[str], out: Path) -> None:
    """Heatmap of detection rate by (screen variant × evasion class).

    Y-axis labels are generated from the screens actually present in the
    matrix (dropping any with no rows), so the script supports both the
    original 3-screen audit and the extended 5-screen one with sliding-
    window variants registered in ``screen_variants.VARIANTS``.
    """
    pivot = matrix.pivot(index="screen", columns="kind", values="detect_rate")
    pivot = pivot.reindex(columns=kinds)
    # Only show rows for variants that actually have data in this run.
    present_variants = [v for v in VARIANTS if v.name in matrix["screen"].unique()]
    pivot = pivot.reindex(index=[v.name for v in present_variants])

    fig, ax = plt.subplots(figsize=(11, max(2.0, 0.7 * len(present_variants) + 1.0)),
                            constrained_layout=True)
    data = pivot.to_numpy()
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([KIND_PRETTY.get(k, k) for k in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [_SCREEN_LABELS.get(v.name, v.name.replace("_", " ")) for v in present_variants],
        fontsize=9,
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(
                j, i, f"{v:.0%}",
                ha="center", va="center",
                color="white" if (0.25 < v < 0.75) else "black",
                fontsize=10, fontweight="bold",
            )
    ax.set_title(
        "Detection rate by screen variant × evasion class\n"
        "(green = high detection, red = miss)",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("P(REVIEW or BLOCK)", fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _consistency_check(df: pd.DataFrame) -> dict:
    """Sanity-check: our re-derived OR-gate should ~match the live verdict.

    Small mismatches are OK if commec was variable per-row; the live
    verdict logic in ensemble.py is reproduced here, so a 100% match is
    the expected case. Any mismatch is reported so we know.
    """
    df2 = df.copy()
    df2["match"] = df2["verdict_or_gate"] == df2["verdict"]
    matched = int(df2["match"].sum())
    total = int(len(df2))
    mismatches = df2[~df2["match"]][
        ["pdb", "kind", "verdict", "verdict_or_gate", "z_natural", "repetitiveness"]
    ]
    return {
        "n_total": total,
        "n_matched": matched,
        "match_rate": matched / max(total, 1),
        "mismatches": mismatches.to_dict("records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEMO_CSV))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = apply_variants(df)

    kinds = [k for k in KIND_ORDER if k in df["kind"].unique()]
    matrix = _matrix(df, kinds)

    matrix_csv = out_dir / "audit_matrix.csv"
    matrix.to_csv(matrix_csv, index=False)
    print(f"Wrote {matrix_csv}")

    figure = out_dir / "audit_matrix.png"
    _plot_matrix(matrix, kinds, figure)
    print(f"Wrote {figure}")

    consistency = _consistency_check(df)
    consistency_path = out_dir / "or_gate_consistency.json"
    consistency_path.write_text(json.dumps(consistency, indent=2, default=str))
    print(
        f"OR-gate replication match rate: "
        f"{consistency['n_matched']}/{consistency['n_total']} "
        f"({consistency['match_rate']:.1%}). "
        f"See {consistency_path.name} for any mismatches."
    )

    print("\n=== Detection-rate matrix (REVIEW or BLOCK) ===")
    pivot = matrix.pivot(index="screen", columns="kind", values="detect_rate")
    pivot = pivot.reindex(columns=kinds, index=[v.name for v in VARIANTS])
    print((pivot * 100).round(1).to_string())
    print("\n=== Block-rate matrix (BLOCK only) ===")
    pivot_b = matrix.pivot(index="screen", columns="kind", values="block_rate")
    pivot_b = pivot_b.reindex(columns=kinds, index=[v.name for v in VARIANTS])
    print((pivot_b * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
