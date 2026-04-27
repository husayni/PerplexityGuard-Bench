"""Re-run the audit + stitch analyses with sliding-window screens included.

Reads the augmented CSVs produced by ``sliding_window.py``:
  - perplexity_guard/demo_results/e2e_demo_sw.csv  (if present; main matrix)
  - perplexity_guard/audit/results/stitch_attack_sw.csv (stitch experiment)

For each, applies all five ``ScreenVariants`` (perplexity-only,
complexity-only, OR-gate, sliding-window, OR-gate-v2) and emits an
updated matrix CSV plus a publication-quality figure.

Usage:
    PYTHONPATH=. uv run python -m perplexity_guard.audit.run_audit_v2
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
DEMO_SW = REPO_ROOT / "perplexity_guard" / "demo_results" / "e2e_demo_sw.csv"
DEMO_LEGACY = REPO_ROOT / "perplexity_guard" / "demo_results" / "e2e_demo.csv"
STITCH_SW = REPO_ROOT / "perplexity_guard" / "audit" / "results" / "stitch_attack_sw.csv"
OUT_DIR = REPO_ROOT / "perplexity_guard" / "audit" / "results"


KIND_ORDER = [
    "native", "mpnn_T=0.1", "mpnn_T=1.0",
    "mutated_40pct", "shuffled", "repeat_motif",
]
KIND_PRETTY = {
    "native": "Native\n(control)",
    "mpnn_T=0.1": "ProteinMPNN\nT=0.1",
    "mpnn_T=1.0": "ProteinMPNN\nT=1.0",
    "mutated_40pct": "40%-mutated\nproxy",
    "shuffled": "Shuffled\nproxy",
    "repeat_motif": "Tandem\nrepeat",
}
SCREEN_LABELS = {
    "perplexity_only": "perplexity-only\n(whole-sequence z)",
    "complexity_only": "complexity-only\n(WF + distinct-k-mer)",
    "or_gate": "OR-gate\n(perplexity + complexity)",
    "sliding_window_perplexity": "sliding-window\n(per-window z)",
    "or_gate_v2": "OR-gate v2\n(sliding-window + complexity)",
}


def _matrix(df: pd.DataFrame, kinds: list[str]) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        col = f"verdict_{v.name}"
        for k in kinds:
            sub = df[df["kind"] == k]
            if sub.empty:
                continue
            rows.append({
                "screen": v.name,
                "kind": k,
                "n": int(len(sub)),
                "detect_rate": detection_rate(sub[col]),
                "block_rate": block_rate(sub[col]),
            })
    return pd.DataFrame(rows)


def _heatmap(matrix: pd.DataFrame, kinds: list[str], out: Path, title: str) -> None:
    pivot = matrix.pivot(index="screen", columns="kind", values="detect_rate")
    pivot = pivot.reindex(columns=[k for k in kinds if k in pivot.columns])
    pivot = pivot.reindex(index=[v.name for v in VARIANTS if v.name in pivot.index])

    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    data = pivot.to_numpy()
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([KIND_PRETTY.get(k, k) for k in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([SCREEN_LABELS.get(s, s) for s in pivot.index], fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                    color="white" if (0.25 < v < 0.75) else "black",
                    fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Detection rate (REVIEW or BLOCK)", fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _stitch_curves(df: pd.DataFrame, out_csv: Path, out_png: Path) -> pd.DataFrame:
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
                "block_rate": block_rate(sub[col]),
            })
    summary = pd.DataFrame(rows).sort_values(["screen", "prefix_frac"])
    summary.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    color_map = {
        "perplexity_only": "#d7191c",
        "complexity_only": "#1a9641",
        "or_gate": "#fdae61",
        "sliding_window_perplexity": "#542788",
        "or_gate_v2": "#2c7bb6",
    }
    style_map = {
        "perplexity_only": "--",
        "complexity_only": ":",
        "or_gate": "--",
        "sliding_window_perplexity": "-",
        "or_gate_v2": "-",
    }
    for v in VARIANTS:
        sub = summary[summary["screen"] == v.name]
        if sub.empty:
            continue
        ax.plot(sub["prefix_frac"] * 100, sub["detect_rate"] * 100,
                marker="o", linewidth=2.0,
                linestyle=style_map.get(v.name, "-"),
                color=color_map.get(v.name, None),
                label=v.name.replace("_", "-"))
    ax.axhline(50, color="#888", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Natural prefix fraction (%) — adversary's evasion budget")
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("Stitching attack: detection vs. natural-prefix fraction (5 screens)")
    ax.set_ylim(-3, 105); ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-csv", default=str(DEMO_SW),
                        help="Path to the augmented main-matrix CSV")
    parser.add_argument("--stitch-csv", default=str(STITCH_SW),
                        help="Path to the augmented stitch CSV")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_path = Path(args.main_csv)
    if main_path.exists():
        df = pd.read_csv(main_path)
        df = apply_variants(df)
        kinds = [k for k in KIND_ORDER if k in df["kind"].unique()]
        mtx = _matrix(df, kinds)
        mtx_csv = out_dir / "audit_matrix_v2.csv"
        mtx.to_csv(mtx_csv, index=False)
        fig = out_dir / "audit_matrix_v2.png"
        _heatmap(
            mtx, kinds, fig,
            "Detection rate by screen variant × attack class (sliding-window included)",
        )
        print(f"[v2] main-matrix v2: {mtx_csv} + {fig}")
        print(
            (mtx.pivot(index="screen", columns="kind", values="detect_rate")
                .reindex(columns=kinds, index=[v.name for v in VARIANTS])
                .fillna(0) * 100).round(1).to_string()
        )
    else:
        print(f"[v2] {main_path} not found; skip main matrix.")

    stitch_path = Path(args.stitch_csv)
    if stitch_path.exists():
        df = pd.read_csv(stitch_path)
        df = apply_variants(df)
        sm_csv = out_dir / "stitch_attack_v2_summary.csv"
        sm_png = out_dir / "stitch_attack_v2.png"
        summary = _stitch_curves(df, sm_csv, sm_png)
        print(f"\n[v2] stitch-curve v2: {sm_csv} + {sm_png}")
        pivot = (summary
                 .pivot(index="screen", columns="prefix_frac", values="detect_rate")
                 .reindex(index=[v.name for v in VARIANTS])
                 .fillna(0) * 100)
        print(pivot.round(1).to_string())
    else:
        print(f"[v2] {stitch_path} not found; skip stitch curve.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
