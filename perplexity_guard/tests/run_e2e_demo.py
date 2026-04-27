"""End-to-end demo: download benign PDBs → ProteinMPNN at multiple temperatures
+ synthetic AI proxies → ESM-2 → screen → metrics.

Conditions per PDB
------------------
1. native            : the wild-type sequence extracted from ATOM records.
                       FPR control — should always PASS.
2. mpnn_low_temp     : ProteinMPNN samples at T=0.1 (high recovery, native-like).
                       Realistic "AI redesign that preserves the fold."
3. mpnn_high_temp    : ProteinMPNN samples at T=1.0 (more divergent, fewer
                       constraints from the prior).
4. mutated_40pct     : 40% of residues randomly substituted to AAs.
                       Matches the validated proxy from the original signal study.
5. shuffled          : composition-preserving shuffle of the native.
                       Strong positive control.

Output
------
demo_results/e2e_demo.{csv,json}  — per-sequence verdicts + run summary.
demo_results/e2e_summary.png      — boxplot + verdict-count chart.

Run with:
    PYTHONPATH=. uv run python -m perplexity_guard.tests.run_e2e_demo --fast
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perplexity_guard.core import (
    ESM2Scorer,
    EnsembleVerdict,
    ProteinMPNNRunner,
    commec_available,
    ensemble_screen,
    proteinmpnn_available,
)
from perplexity_guard.core.esm_analysis import DEFAULT_MODEL, FAST_MODEL


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "perplexity_guard" / "tests" / "sample_pdbs"
DEMO_DIR = REPO_ROOT / "perplexity_guard" / "demo_results"
CALIB_DIR = REPO_ROOT / "perplexity_guard" / "calibration"
DEFAULT_PDBS = [
    "1UBQ",  # ubiquitin (76 aa, β-grasp)
    "2LYZ",  # hen egg-white lysozyme (129 aa, α/β)
    "1MBO",  # sperm-whale myoglobin (153 aa, all-α)
    "1ENH",  # engrailed homeodomain (54 aa, all-α)
    "1PGA",  # Streptococcus protein G B1 (56 aa, β-grasp)
    "1UZC",  # cold-shock protein (67 aa, β)
    "2IGD",  # immunoglobulin-binding domain (61 aa, β)
    "3ICB",  # bovine calbindin D9k (75 aa, EF-hand)
    "1BPI",  # bovine pancreatic trypsin inhibitor (58 aa, mixed)
    "1CTF",  # E. coli L7/L12 C-terminal (68 aa, all-α)
]
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
AA20 = "ACDEFGHIKLMNPQRSTVWY"


# ---- I/O ----------------------------------------------------------------


def fetch_pdb(pdb_id: str) -> Path:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLE_DIR / f"{pdb_id.upper()}.pdb"
    if out.exists() and out.stat().st_size > 1024:
        return out
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        out.write_bytes(r.read())
    return out


def extract_native_sequence(pdb_path: Path, chain: str | None = None) -> str:
    """Pull the longest-chain AA sequence out of CA atom records.

    Handles NMR ensembles by only consuming MODEL 1 (or the implicit single
    model in X-ray entries) — otherwise we'd concatenate the same residues
    across every NMR conformer.
    """
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seq_by_chain: dict[str, list[str]] = {}
    last_res: dict[str, int] = {}
    saw_model = False
    for line in pdb_path.read_text().splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            continue
        if line.startswith("ENDMDL") and saw_model:
            break  # NMR / multi-model: only take the first model
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        # Skip alternate-location indicators other than '' or 'A'.
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        c = line[21]
        resname = line[17:20].strip()
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        if resname not in three_to_one or last_res.get(c) == resnum:
            continue
        last_res[c] = resnum
        seq_by_chain.setdefault(c, []).append(three_to_one[resname])
    if not seq_by_chain:
        raise RuntimeError(f"No CA atoms parsed from {pdb_path}")
    if chain is not None and chain in seq_by_chain:
        return "".join(seq_by_chain[chain])
    longest = max(seq_by_chain.items(), key=lambda kv: len(kv[1]))
    return "".join(longest[1])


# ---- Synthetic proxies (matches validation script) ---------------------


def mutate_40pct(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    n_mut = max(1, int(0.4 * len(seq)))
    for i in rng.sample(range(len(seq)), n_mut):
        chars[i] = rng.choice(AA20)
    return "".join(chars)


def shuffle_seq(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def repeat_motif(seq: str, rng: random.Random) -> str:
    """Tile a 4-8mer drawn from the source sequence to the original length.

    Tandem-repeat designs are the evasion case identified in the validation
    phase: they score *below* natural pseudo-perplexity but should be flagged
    by the low-complexity detector.
    """
    L = len(seq)
    k = rng.randint(4, 8)
    start = rng.randint(0, max(0, L - k))
    motif = seq[start : start + k] or "GA"
    return (motif * (L // len(motif) + 1))[:L]


# ---- Driver -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbs", nargs="+", default=DEFAULT_PDBS)
    parser.add_argument("--num-sequences", type=int, default=4,
                        help="ProteinMPNN samples per (PDB, temperature)")
    parser.add_argument("--fast", action="store_true", help="Use t12_35M instead of t33_650M")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-commec", action="store_true")
    parser.add_argument(
        "--no-auto-calibrate", action="store_true",
        help="Skip the auto-calibration step (use the fallback distribution)",
    )
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    if not proteinmpnn_available():
        print("[error] ProteinMPNN not cloned at ./external/ProteinMPNN", file=sys.stderr)
        return 2

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    model_name = args.model or (FAST_MODEL if args.fast else DEFAULT_MODEL)
    print(f"[demo] model={model_name}  pdbs={args.pdbs}  n_per_temp={args.num_sequences}")

    avail = commec_available()
    print(f"[demo] commec usable={avail.usable} reason={avail.runtime_error}")

    scorer = ESM2Scorer(
        model_name=model_name,
        device="cpu" if args.cpu else None,
        calibration_dir=CALIB_DIR,
        auto_calibrate=not args.no_auto_calibrate,
    )
    runner = ProteinMPNNRunner(model_name="v_48_020")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    timings: dict[str, dict] = {}

    for pdb_id in args.pdbs:
        pdb_path = fetch_pdb(pdb_id)
        print(f"\n[demo] === {pdb_id} ({pdb_path.name}, "
              f"{pdb_path.stat().st_size/1024:.0f} KB) ===")
        native_seq = extract_native_sequence(pdb_path)

        items: list[tuple[str, str, dict]] = []
        items.append((f"{pdb_id} native", native_seq, {"kind": "native", "sample_idx": -1}))

        # Two MPNN temperature points to span recovery regimes.
        mpnn_t0 = time.time()
        for temp in (0.1, 1.0):
            designs = runner.design(
                pdb_path,
                num_sequences=args.num_sequences,
                temperature=temp,
                seed=args.seed,
            )
            for d in designs:
                if d.is_native:
                    continue  # de-dup against the extracted native
                kind = f"mpnn_T={temp}"
                label = f"{pdb_id} {kind} sample={d.sample_index}"
                items.append((label, d.sequence, {
                    "kind": kind,
                    "sample_idx": d.sample_index,
                    "mpnn_score": d.score,
                    "temperature": d.temperature,
                }))
        mpnn_seconds = time.time() - mpnn_t0

        # Synthetic proxies that mirror the original signal validation.
        items.append((f"{pdb_id} mutated_40pct",
                      mutate_40pct(native_seq, rng),
                      {"kind": "mutated_40pct", "sample_idx": -1}))
        items.append((f"{pdb_id} shuffled",
                      shuffle_seq(native_seq, rng),
                      {"kind": "shuffled", "sample_idx": -1}))
        # Repeat-motif: the evasion case the low-complexity detector now closes.
        items.append((f"{pdb_id} repeat_motif",
                      repeat_motif(native_seq, rng),
                      {"kind": "repeat_motif", "sample_idx": -1}))

        pdb_t0 = time.time()
        for label, seq, meta in items:
            v = ensemble_screen(
                seq, scorer,
                run_commec=not args.no_commec,
                name=label.replace(" ", "_"),
                metadata={"label": label, "pdb": pdb_id, **meta},
            )
            rows.append(_row(v, pdb_id, meta))
            print(
                f"  {meta['kind']:<14} sample={meta.get('sample_idx', -1):>2} "
                f"len={v.sequence_length:>3} "
                f"ppl={v.perplexity.pseudo_perplexity:6.2f} "
                f"z={v.perplexity.zscore:+5.2f} "
                f"rep={v.complexity.repetitiveness_score:.2f} "
                f"d3={v.complexity.distinct_3mer_fraction:.2f} "
                f"verdict={v.verdict:<6} ({v.runtime_seconds:.1f}s)"
            )
        timings[pdb_id] = {
            "wall_seconds": round(time.time() - pdb_t0, 2),
            "mpnn_seconds": round(mpnn_seconds, 2),
        }

    df = pd.DataFrame(rows)
    csv_path = DEMO_DIR / "e2e_demo.csv"
    df.to_csv(csv_path, index=False)

    # ---- Metrics summary ----
    summary = _summarise(df, timings, scorer, model_name, avail)
    json_path = DEMO_DIR / "e2e_demo.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    # ---- Plot ----
    plot_path = DEMO_DIR / "e2e_summary.png"
    _plot(df, plot_path, model_name, scorer)

    # ---- Console report ----
    print("\n" + "=" * 72)
    print(f"E2E demo complete. {len(df)} sequences across {len(args.pdbs)} PDBs.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")
    print("\nPer-condition verdicts:")
    pivot = (
        df.groupby(["kind", "verdict"]).size().unstack(fill_value=0)
        .reindex(columns=["PASS", "REVIEW", "BLOCK"], fill_value=0)
    )
    print(pivot.to_string())
    print("\nDetection rate (REVIEW or BLOCK) per condition:")
    det = (df.groupby("kind")["verdict"]
             .apply(lambda s: float(((s != "PASS").sum()) / len(s))))
    print(det.round(3).to_string())
    print(f"\nFalse positive rate on natives: "
          f"{summary['natives']['false_positive_rate']:.2%}  "
          f"(target: <5%)")
    print(f"Detection rate on mutated_40pct: "
          f"{summary.get('mutated_40pct', {}).get('detection_rate', 0):.2%}")
    print(f"Detection rate on shuffled    : "
          f"{summary.get('shuffled', {}).get('detection_rate', 0):.2%}")
    return 0


def _row(v: EnsembleVerdict, pdb_id: str, meta: dict) -> dict:
    return {
        "pdb": pdb_id,
        "kind": meta["kind"],
        "sample_idx": meta.get("sample_idx", -1),
        "len": v.sequence_length,
        "perplexity": round(v.perplexity.pseudo_perplexity, 2),
        "z_natural": round(v.perplexity.zscore, 2),
        "is_high": v.perplexity.is_high,
        "complexity_mean": round(v.complexity.mean_complexity, 3),
        "low_complexity_frac": round(v.complexity.low_complexity_fraction, 3),
        "distinct_3mer_frac": round(v.complexity.distinct_3mer_fraction, 3),
        "repetitiveness": round(v.complexity.repetitiveness_score, 3),
        "ppl_runtime_s": round(v.perplexity.runtime_seconds, 2),
        "commec_status": v.commec.status,
        "verdict": v.verdict,
        "flags": "|".join(v.flags),
        "mpnn_score": meta.get("mpnn_score"),
        "temperature": meta.get("temperature"),
        "total_runtime_s": round(v.runtime_seconds, 2),
        "explanation": v.explanation,
    }


def _summarise(df: pd.DataFrame, timings: dict, scorer, model_name: str, avail) -> dict:
    natives = df[df["kind"] == "native"]
    summary: dict = {
        "model": model_name,
        "device": str(scorer.device),
        "calibration": scorer.calibration.to_dict(),
        "commec_available": avail.__dict__,
        "n_total": int(len(df)),
        "natives": {
            "n": int(len(natives)),
            "verdict_counts": natives["verdict"].value_counts().to_dict(),
            "false_positive_rate": float(((natives["verdict"] != "PASS").sum())
                                         / max(len(natives), 1)),
            "ppl_mean": float(natives["perplexity"].mean()) if len(natives) else None,
            "ppl_median": float(natives["perplexity"].median()) if len(natives) else None,
        },
        "timings": timings,
        "ppl_runtime_per_seq_seconds": {
            "mean": float(df["ppl_runtime_s"].mean()),
            "median": float(df["ppl_runtime_s"].median()),
            "max": float(df["ppl_runtime_s"].max()),
        },
    }
    for kind, sub in df.groupby("kind"):
        summary[kind] = {
            "n": int(len(sub)),
            "ppl_mean": float(sub["perplexity"].mean()),
            "ppl_median": float(sub["perplexity"].median()),
            "z_mean": float(sub["z_natural"].mean()),
            "verdict_counts": sub["verdict"].value_counts().to_dict(),
            "detection_rate": float(((sub["verdict"] != "PASS").sum()) / max(len(sub), 1)),
        }
    return summary


def _plot(df: pd.DataFrame, out: Path, model_name: str, scorer) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    kind_order = [
        "native", "mpnn_T=0.1", "mpnn_T=1.0",
        "mutated_40pct", "shuffled", "repeat_motif",
    ]
    kinds = [k for k in kind_order if k in df["kind"].unique()]
    palette = {
        "native": "#2c7fb8",
        "mpnn_T=0.1": "#a6cee3",
        "mpnn_T=1.0": "#fdae61",
        "mutated_40pct": "#d7191c",
        "shuffled": "#7b3294",
        "repeat_motif": "#1a9641",
    }

    # Panel 1: PPL by condition.
    ax = axes[0]
    data = [df[df["kind"] == k]["perplexity"].values for k in kinds]
    bp = ax.boxplot(data, tick_labels=kinds, showfliers=True, patch_artist=True)
    for patch, k in zip(bp["boxes"], kinds):
        patch.set_facecolor(palette.get(k, "#888"))
        patch.set_alpha(0.6)
    ax.axhline(scorer.calibration.p95, color="orange", linestyle="--",
               label=f"p95 = {scorer.calibration.p95:.1f}")
    ax.axhline(scorer.calibration.mean + 3 * scorer.calibration.std, color="red",
               linestyle=":", label="z=3 BLOCK")
    ax.set_ylabel("ESM-2 pseudo-perplexity")
    ax.set_title("Perplexity by condition")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)

    # Panel 2: Repetitiveness score (entropy ∪ k-mer-redundancy).
    ax = axes[1]
    data_rep = [df[df["kind"] == k]["repetitiveness"].values for k in kinds]
    bp = ax.boxplot(data_rep, tick_labels=kinds, showfliers=True, patch_artist=True)
    for patch, k in zip(bp["boxes"], kinds):
        patch.set_facecolor(palette.get(k, "#888"))
        patch.set_alpha(0.6)
    ax.axhline(0.20, color="orange", linestyle="--", label="REVIEW (≥0.20)")
    ax.axhline(0.50, color="red", linestyle=":", label="BLOCK (≥0.50)")
    ax.set_ylabel("repetitiveness score")
    ax.set_title("Low-complexity / repeat detector by condition")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)

    # Panel 3: Verdicts stacked.
    ax = axes[2]
    pivot = (
        df.groupby(["kind", "verdict"]).size().unstack(fill_value=0)
        .reindex(columns=["PASS", "REVIEW", "BLOCK"], fill_value=0)
        .reindex(index=kinds)
    )
    bottom = np.zeros(len(pivot))
    verdict_colors = {"PASS": "#1a9641", "REVIEW": "#fdae61", "BLOCK": "#d7191c"}
    for verdict in ("PASS", "REVIEW", "BLOCK"):
        ax.bar(pivot.index, pivot[verdict], bottom=bottom,
               color=verdict_colors[verdict], label=verdict)
        bottom += pivot[verdict].values
    ax.set_ylabel("count")
    ax.set_title("Verdicts by condition")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"PerplexityGuard E2E demo — {model_name} "
        f"(calibration: mean={scorer.calibration.mean:.2f} std={scorer.calibration.std:.2f}, "
        f"n={scorer.calibration.n})",
        fontsize=11,
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
