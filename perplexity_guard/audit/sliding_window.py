"""Compute sliding-window pseudo-perplexity for the audit's sequences.

This is the structural patch motivated by §4.2 of the report. The whole-
sequence perplexity gate is dilution-vulnerable to natural prefixes; a
position-resolved gate that flags any 30-residue window whose local z-
score exceeds threshold closes that attack by construction.

The script does three things:

1. **Build a per-window calibration.** Reads
   `calibration/uniprot_sample.fasta` (cached on first auto-calibration),
   scores each UniProt sequence with per-position NLLs, accumulates per-
   window mean PPLs across all sliding windows of the chosen size, and
   persists `calibration/<safe(model)>_window<W>.json`.

2. **Score the main-matrix sequences with sliding windows.** Re-runs the
   full e2e demo (deterministic seed) but augments each row with three new
   columns: ``max_window_z``, ``min_window_z``, ``argmax_window_pos``.
   Output: ``demo_results/e2e_demo_sw.csv``.

3. **Score the stitch-attack sequences with sliding windows.** Same idea
   for the stitching experiment. Output:
   ``audit/results/stitch_attack_sw.csv``.

Then `audit/run_audit.py` and `audit/stitch_attack.py` (or the bundled
`run_with_sliding_window.py` driver added below) can be re-applied with
the four-screen ``VARIANTS`` list including ``sliding_window_perplexity``
and ``or_gate_v2``.

Usage:
    # Build calibration only (~5 min on t12_35M):
    PYTHONPATH=. uv run python -m perplexity_guard.audit.sliding_window \\
        --fast --calibrate-only

    # Calibrate + re-score the stitch experiment (~10 min on t12):
    PYTHONPATH=. uv run python -m perplexity_guard.audit.sliding_window \\
        --fast --replicates 3 --score-stitch

    # Calibrate + re-score the main matrix (~30 min on t33):
    PYTHONPATH=. uv run python -m perplexity_guard.audit.sliding_window \\
        --score-main --num-sequences 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from perplexity_guard.core.complexity import wootton_federhen
from perplexity_guard.core.esm_analysis import (
    DEFAULT_MODEL,
    ESM2Scorer,
    FAST_MODEL,
    WindowCalibration,
    windowed_z_max,
    _safe,
)
from perplexity_guard.core.proteinmpnn_wrapper import (
    ProteinMPNNRunner,
    proteinmpnn_available,
)
from perplexity_guard.tests.run_e2e_demo import (
    DEFAULT_PDBS,
    extract_native_sequence,
    fetch_pdb,
    mutate_40pct,
    repeat_motif,
    shuffle_seq,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "perplexity_guard" / "calibration"
DEMO_DIR = REPO_ROOT / "perplexity_guard" / "demo_results"
AUDIT_DIR = REPO_ROOT / "perplexity_guard" / "audit" / "results"

DEFAULT_WINDOW = 30


# ---- Window calibration ------------------------------------------------


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                seq = "".join(chunks).upper()
                if seq and set(seq) <= AA20:
                    out.append((header, seq))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None:
        seq = "".join(chunks).upper()
        if seq and set(seq) <= AA20:
            out.append((header, seq))
    return out


def build_window_calibration(
    scorer: ESM2Scorer,
    window_size: int = DEFAULT_WINDOW,
    n_calibration_sequences: int = 50,
    overwrite: bool = False,
) -> WindowCalibration:
    """Aggregate per-window PPLs across UniProt naturals → WindowCalibration."""
    cal_path = CALIB_DIR / f"{_safe(scorer.model_name)}_window{window_size}.json"
    if cal_path.exists() and not overwrite:
        data = json.loads(cal_path.read_text())
        return WindowCalibration(
            model=scorer.model_name,
            window_size=int(data["window_size"]),
            mean=float(data["mean"]),
            std=float(data["std"]),
            p95=float(data["p95"]),
            p99=float(data["p99"]),
            n_windows=int(data.get("n_windows", 0)),
            n_sequences=int(data.get("n_sequences", 0)),
            source=str(cal_path),
        )

    fasta_path = CALIB_DIR / "uniprot_sample.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"{fasta_path} not found. Run the main demo or whole-sequence "
            "auto-calibration first to populate it."
        )
    records = _parse_fasta(fasta_path)
    # Filter for sequences long enough to host at least one window.
    records = [(h, s) for (h, s) in records if len(s) >= window_size]
    records = records[:n_calibration_sequences]
    if not records:
        raise RuntimeError("No calibration sequences passed length filter.")

    print(
        f"[swcal] building window calibration: model={scorer.model_name} "
        f"window={window_size} n={len(records)}"
    )
    all_window_ppls: list[float] = []
    t0 = time.time()
    for i, (h, s) in enumerate(records):
        try:
            r = scorer.score_sequence(s)
        except Exception as e:
            print(f"[swcal] skip {h[:30]}: {e}")
            continue
        nlls = r.per_position_nlls
        if len(nlls) < window_size:
            continue
        for j in range(len(nlls) - window_size + 1):
            mean_nll = sum(nlls[j : j + window_size]) / window_size
            ppl = math.exp(min(mean_nll, 50.0))
            all_window_ppls.append(ppl)
        if (i + 1) % 5 == 0 or i == len(records) - 1:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(
                f"[swcal] {i+1}/{len(records)} L={len(s):>3} "
                f"n_windows={len(nlls) - window_size + 1} "
                f"({rate:.2f} seq/s)"
            )

    if not all_window_ppls:
        raise RuntimeError("No window PPLs collected for calibration.")
    arr = np.array(all_window_ppls, dtype=np.float64)
    cal = WindowCalibration(
        model=scorer.model_name,
        window_size=window_size,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        n_windows=int(arr.size),
        n_sequences=len(records),
        source=str(cal_path),
    )
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(cal.to_dict(), indent=2))
    print(
        f"[swcal] wrote {cal_path.name}: "
        f"mean={cal.mean:.2f} std={cal.std:.2f} p95={cal.p95:.2f} "
        f"p99={cal.p99:.2f} n_windows={cal.n_windows}"
    )
    return cal


# ---- Per-sequence scoring helper ----------------------------------------


def score_with_window(seq: str, scorer: ESM2Scorer, window_cal: WindowCalibration):
    """Run pLM scoring once; return whole-sequence + sliding-window stats."""
    perp = scorer.score_sequence(seq)
    cx = wootton_federhen(seq)
    max_z, min_z, argmax_pos = windowed_z_max(perp.per_position_nlls, window_cal)
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
        "max_window_z": max_z,
        "min_window_z": min_z,
        "argmax_window_pos": argmax_pos,
    }


# ---- Stitch experiment with sliding window ------------------------------


def score_stitch(scorer: ESM2Scorer, window_cal: WindowCalibration,
                 pdbs: list[str], replicates: int,
                 prefix_fracs: list[float], seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []
    t0 = time.time()
    for pdb_id in pdbs:
        pdb_path = fetch_pdb(pdb_id)
        native = extract_native_sequence(pdb_path)
        nrow = score_with_window(native, scorer, window_cal)
        nrow.update({"pdb": pdb_id, "replicate": -1, "prefix_frac": 1.0,
                     "kind": "native", "is_full_adversary": False})
        rows.append(nrow)
        for rep in range(replicates):
            adv = mutate_40pct(native, rng)
            for f in prefix_fracs:
                k = int(round(f * len(native)))
                seq = native[:k] + adv[k:]
                row = score_with_window(seq, scorer, window_cal)
                row.update({
                    "pdb": pdb_id, "replicate": rep, "prefix_frac": float(f),
                    "kind": f"stitch_{int(round(f*100)):03d}pct_native_prefix",
                    "is_full_adversary": (f == 0.0),
                })
                rows.append(row)
                print(
                    f"  {pdb_id} rep={rep} f={f:.2f} L={row['len']:>3} "
                    f"ppl={row['perplexity']:6.2f} z_whole={row['z_natural']:+5.2f} "
                    f"max_window_z={row['max_window_z']:+5.2f}"
                )
        print(f"  ... {pdb_id} done ({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(rows)


# ---- Main matrix with sliding window ------------------------------------


def score_main(scorer: ESM2Scorer, window_cal: WindowCalibration,
               pdbs: list[str], num_mpnn: int, seed: int) -> pd.DataFrame:
    if not proteinmpnn_available():
        print("[error] ProteinMPNN not cloned; cannot rebuild main matrix.")
        return pd.DataFrame()
    runner = ProteinMPNNRunner(model_name="v_48_020")
    rng = random.Random(seed)
    rows: list[dict] = []
    t0 = time.time()
    for pdb_id in pdbs:
        pdb_path = fetch_pdb(pdb_id)
        native = extract_native_sequence(pdb_path)
        items: list[tuple[str, str]] = [(f"{pdb_id} native:native", native)]
        for temp in (0.1, 1.0):
            designs = runner.design(pdb_path, num_sequences=num_mpnn,
                                    temperature=temp, seed=seed)
            for d in designs:
                if d.is_native:
                    continue
                items.append((f"{pdb_id} mpnn_T={temp}:mpnn_T={temp}", d.sequence))
        items.append((f"{pdb_id} mutated_40pct:mutated_40pct", mutate_40pct(native, rng)))
        items.append((f"{pdb_id} shuffled:shuffled", shuffle_seq(native, rng)))
        items.append((f"{pdb_id} repeat_motif:repeat_motif", repeat_motif(native, rng)))
        for label, seq in items:
            row = score_with_window(seq, scorer, window_cal)
            kind = label.rsplit(":", 1)[1]
            row.update({"pdb": pdb_id, "kind": kind, "label": label.rsplit(":", 1)[0]})
            rows.append(row)
            print(
                f"  {pdb_id} {kind:<14} L={row['len']:>3} "
                f"ppl={row['perplexity']:6.2f} z_whole={row['z_natural']:+5.2f} "
                f"max_window_z={row['max_window_z']:+5.2f}"
            )
        print(f"  ... {pdb_id} done ({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(rows)


# ---- Driver -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use t12_35M (faster)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Build the per-window calibration and exit")
    parser.add_argument("--score-stitch", action="store_true",
                        help="Re-score stitch sequences with sliding window")
    parser.add_argument("--score-main", action="store_true",
                        help="Re-score main-matrix sequences with sliding window")
    parser.add_argument("--pdbs", nargs="+", default=DEFAULT_PDBS)
    parser.add_argument("--replicates", type=int, default=3,
                        help="Stitch replicates per native")
    parser.add_argument("--num-sequences", type=int, default=4,
                        help="ProteinMPNN samples per (PDB, temperature)")
    parser.add_argument("--prefix-fracs", type=float, nargs="+",
                        default=[0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0])
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--overwrite-calibration", action="store_true")
    args = parser.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    model = args.model or (FAST_MODEL if args.fast else DEFAULT_MODEL)
    print(f"[sw] model={model}  window={args.window_size}")

    scorer = ESM2Scorer(
        model_name=model,
        device="cpu" if args.cpu else None,
        calibration_dir=CALIB_DIR,
        auto_calibrate=True,
    )
    window_cal = build_window_calibration(
        scorer, window_size=args.window_size,
        overwrite=args.overwrite_calibration,
    )

    if args.calibrate_only:
        print("[sw] calibrate-only mode; exiting.")
        return 0

    if args.score_stitch or (not args.score_main and not args.score_stitch):
        # Default action when neither flag is set: score stitch (the relevant
        # experiment for the structural-patch demonstration).
        print("\n[sw] === scoring stitch experiment with sliding-window ===")
        df = score_stitch(scorer, window_cal, args.pdbs, args.replicates,
                          args.prefix_fracs, args.seed)
        out = AUDIT_DIR / "stitch_attack_sw.csv"
        df.to_csv(out, index=False)
        print(f"[sw] wrote {out} ({len(df)} rows)")

    if args.score_main:
        print("\n[sw] === scoring main matrix with sliding-window ===")
        df = score_main(scorer, window_cal, args.pdbs, args.num_sequences, args.seed)
        out = DEMO_DIR / "e2e_demo_sw.csv"
        df.to_csv(out, index=False)
        print(f"[sw] wrote {out} ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
