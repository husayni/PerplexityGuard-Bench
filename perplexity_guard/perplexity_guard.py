"""PerplexityGuard CLI + Gradio app.

Subcommands
-----------
screen   Run the full pipeline (PDB → ProteinMPNN → ESM-2 → commec) and write
         a JSON + CSV report.
score    Score one or more amino-acid sequences from a FASTA (skips MPNN).
calibrate
         Compute mean/std/p95/p99 of pseudo-perplexity over a corpus of
         natural sequences and write a calibration JSON.
ui       Launch the Gradio web app.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from perplexity_guard.core import (
    ESM2Scorer,
    EnsembleVerdict,
    ProteinMPNNRunner,
    commec_available,
    ensemble_screen,
    proteinmpnn_available,
    risk_color,
)
from perplexity_guard.core.esm_analysis import (
    DEFAULT_MODEL,
    FAST_MODEL,
    Calibration,
    PerplexityResult,
)


REPO_ROOT = Path(__file__).resolve().parent
CALIB_DIR = REPO_ROOT / "calibration"


# ---- Helpers ------------------------------------------------------------


def _load_scorer(args: argparse.Namespace, auto_calibrate: bool = True) -> ESM2Scorer:
    return ESM2Scorer(
        model_name=args.model,
        device="cpu" if args.cpu else None,
        calibration_dir=CALIB_DIR,
        auto_calibrate=auto_calibrate and not getattr(args, "no_auto_calibrate", False),
    )


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None:
        out.append((header, "".join(chunks)))
    return out


def _verdicts_to_df(verdicts: list[EnsembleVerdict]) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(verdicts):
        seq_disp = v.sequence if len(v.sequence) <= 36 else v.sequence[:32] + "…"
        rows.append({
            "idx": i,
            "label": v.metadata.get("label", f"seq_{i}"),
            "len": v.sequence_length,
            "sequence": seq_disp,
            "perplexity": round(v.perplexity.pseudo_perplexity, 2),
            "z_natural": round(v.perplexity.zscore, 2),
            "low_complexity_frac": round(v.complexity.low_complexity_fraction, 3),
            "ppl_runtime_s": round(v.perplexity.runtime_seconds, 2),
            "commec_status": v.commec.status,
            "commec_verdict": (
                v.commec.verdict if len(v.commec.verdict) <= 60
                else v.commec.verdict[:57] + "…"
            ),
            "verdict": v.verdict,
            "flags": ", ".join(v.flags),
            "total_runtime_s": round(v.runtime_seconds, 2),
            "explanation": v.explanation,
        })
    return pd.DataFrame(rows)


def _summarise_run(verdicts: list[EnsembleVerdict]) -> dict:
    n = len(verdicts)
    if n == 0:
        return {"n": 0}
    counts = {"PASS": 0, "REVIEW": 0, "BLOCK": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    runtimes = [v.runtime_seconds for v in verdicts]
    ppls = [v.perplexity.pseudo_perplexity for v in verdicts]
    zs = [v.perplexity.zscore for v in verdicts]
    return {
        "n": n,
        "verdict_counts": counts,
        "fpr_estimate_for_naturals": (
            (counts["REVIEW"] + counts["BLOCK"]) / max(n, 1)
        ),
        "runtime_seconds": {
            "mean": float(np.mean(runtimes)),
            "median": float(np.median(runtimes)),
            "max": float(np.max(runtimes)),
        },
        "perplexity": {
            "mean": float(np.mean(ppls)),
            "median": float(np.median(ppls)),
            "min": float(np.min(ppls)),
            "max": float(np.max(ppls)),
        },
        "zscore": {
            "mean": float(np.mean(zs)),
            "max": float(np.max(zs)),
        },
    }


# ---- CLI: screen --------------------------------------------------------


def cmd_screen(args: argparse.Namespace) -> int:
    if not proteinmpnn_available():
        print("[error] ProteinMPNN not cloned at ./external/ProteinMPNN", file=sys.stderr)
        return 2
    avail = commec_available()
    if not avail.usable and not args.no_commec:
        print(
            f"[warn] commec unusable ({avail.runtime_error}); "
            f"continuing perplexity-only. Pass --no-commec to silence.",
            file=sys.stderr,
        )

    scorer = _load_scorer(args)
    runner = ProteinMPNNRunner(
        model_name=args.mpnn_model, use_soluble=args.soluble
    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_paths = [Path(p).resolve() for p in args.pdb]
    all_verdicts: list[EnsembleVerdict] = []
    per_pdb: dict[str, dict] = {}

    for pdb in pdb_paths:
        pdb_t0 = time.time()
        print(f"\n[screen] === {pdb.name} ===")
        designs = runner.design(
            pdb,
            num_sequences=args.num_sequences,
            temperature=args.mpnn_temp,
            seed=args.seed,
        )
        verdicts: list[EnsembleVerdict] = []
        for d in designs:
            v = ensemble_screen(
                d.sequence,
                scorer,
                run_commec=not args.no_commec,
                name=f"{pdb.stem}_s{d.sample_index}",
                metadata={
                    "label": f"{pdb.stem} sample={d.sample_index}"
                            + (" (native)" if d.is_native else ""),
                    "pdb": pdb.name,
                    "sample_index": d.sample_index,
                    "is_native": d.is_native,
                    "mpnn_score": d.score,
                    "temperature": d.temperature,
                },
            )
            verdicts.append(v)
            print(
                f"  sample={d.sample_index:>2}{' (native)' if d.is_native else '':<9} "
                f"len={v.sequence_length:>3} "
                f"ppl={v.perplexity.pseudo_perplexity:6.2f} "
                f"z={v.perplexity.zscore:+5.2f} "
                f"commec={v.commec.status:<8} verdict={v.verdict:<6} "
                f"({v.runtime_seconds:.1f}s)"
            )
        all_verdicts.extend(verdicts)
        per_pdb[pdb.name] = {
            "n_designs": len(verdicts),
            "wall_seconds": round(time.time() - pdb_t0, 2),
            "summary": _summarise_run(verdicts),
        }

    df = _verdicts_to_df(all_verdicts)
    csv_path = out_dir / "screen_results.csv"
    json_path = out_dir / "screen_results.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "model": args.model,
        "mpnn_model": args.mpnn_model,
        "commec": commec_available().__dict__,
        "calibration": scorer.calibration.to_dict(),
        "per_pdb": per_pdb,
        "overall_summary": _summarise_run(all_verdicts),
        "verdicts": [v.to_dict() | {"label": v.metadata.get("label")} for v in all_verdicts],
    }, indent=2, default=str))

    print()
    print("=" * 68)
    print(f"Screened {len(all_verdicts)} sequences across {len(pdb_paths)} PDBs.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    print(df[["label", "len", "perplexity", "z_natural", "commec_status", "verdict"]]
          .to_string(index=False))
    return 0


# ---- CLI: score ---------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    scorer = _load_scorer(args)
    if args.aa:
        records = [(args.label or "input", args.aa)]
    elif args.fasta:
        records = _read_fasta(Path(args.fasta))
    else:
        print("[error] pass --aa <SEQUENCE> or --fasta <PATH>", file=sys.stderr)
        return 2

    verdicts: list[EnsembleVerdict] = []
    for header, seq in records:
        v = ensemble_screen(
            seq,
            scorer,
            run_commec=not args.no_commec,
            name=header.split()[0] if header else "query",
            metadata={"label": header},
        )
        verdicts.append(v)
        print(
            f"  {header[:32]:<32}  "
            f"len={v.sequence_length:>3} "
            f"ppl={v.perplexity.pseudo_perplexity:6.2f} "
            f"z={v.perplexity.zscore:+5.2f} "
            f"verdict={v.verdict}"
        )
    df = _verdicts_to_df(verdicts)
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")
    return 0


# ---- CLI: calibrate -----------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    fasta = Path(args.fasta).resolve()
    records = _read_fasta(fasta)
    if args.n and args.n < len(records):
        records = records[: args.n]
    print(f"[calib] scoring {len(records)} sequences from {fasta.name}")
    scorer = _load_scorer(args, auto_calibrate=False)
    ppls = []
    for i, (h, s) in enumerate(records):
        try:
            r = scorer.score_sequence(s)
            ppls.append(r.pseudo_perplexity)
        except Exception as e:
            print(f"[calib] skip {h}: {e}")
            continue
        if (i + 1) % 5 == 0 or i == len(records) - 1:
            print(f"[calib] {i+1}/{len(records)} latest={ppls[-1]:.2f}")
    arr = np.array(ppls, dtype=np.float64)
    out = {
        "model": args.model,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "source_fasta": str(fasta),
    }
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIB_DIR / f"{args.model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(out, indent=2))
    return 0


# ---- Gradio UI ----------------------------------------------------------


def build_gradio_app(scorer: ESM2Scorer, mpnn_runner: ProteinMPNNRunner | None):
    import gradio as gr  # imported lazily so CLI users don't pay for it

    def _color_verdict(verdict: str) -> str:
        return f"<span style='color:{risk_color(verdict)};font-weight:600'>{verdict}</span>"

    def screen_pdb_handler(
        pdb_file,
        num_sequences: int,
        mpnn_temp: float,
        run_commec: bool,
    ):
        if pdb_file is None:
            return None, "Upload a PDB file first."
        if mpnn_runner is None:
            return None, "ProteinMPNN is not available — cannot generate from a PDB."
        path = Path(pdb_file.name)
        designs = mpnn_runner.design(
            path,
            num_sequences=int(num_sequences),
            temperature=float(mpnn_temp),
        )
        verdicts: list[EnsembleVerdict] = []
        for d in designs:
            v = ensemble_screen(
                d.sequence, scorer,
                run_commec=run_commec,
                name=f"{path.stem}_s{d.sample_index}",
                metadata={
                    "label": f"sample={d.sample_index}{' (native)' if d.is_native else ''}",
                    "is_native": d.is_native,
                    "mpnn_score": d.score,
                },
            )
            verdicts.append(v)
        df = _verdicts_to_df(verdicts).copy()
        df["verdict"] = df["verdict"].map(_color_verdict)
        summary = _summarise_run(verdicts)
        md = (
            f"**Designs screened:** {summary['n']}  \n"
            f"**Verdict counts:** {summary['verdict_counts']}  \n"
            f"**Mean PPL:** {summary['perplexity']['mean']:.2f}  "
            f"(min {summary['perplexity']['min']:.2f}, max {summary['perplexity']['max']:.2f})  \n"
            f"**Mean runtime/design:** {summary['runtime_seconds']['mean']:.1f}s  \n"
            f"**Calibration source:** {scorer.calibration.source}"
        )
        return df, md

    def screen_seq_handler(aa_sequence: str, run_commec: bool):
        seq = (aa_sequence or "").strip()
        if not seq:
            return None, "Paste an amino acid sequence first."
        v = ensemble_screen(
            seq, scorer, run_commec=run_commec, name="ad_hoc",
            metadata={"label": "ad-hoc input"},
        )
        df = _verdicts_to_df([v]).copy()
        df["verdict"] = df["verdict"].map(_color_verdict)
        md = (
            f"**Verdict:** {_color_verdict(v.verdict)}  \n"
            f"**Flags:** {', '.join(v.flags) or '(none)'}  \n"
            f"**Why:** {v.explanation}"
        )
        return df, md

    avail = commec_available()
    commec_md = (
        "✅ commec configured and ready"
        if avail.usable
        else f"⚠️ commec not configured: {avail.runtime_error}. "
             "Perplexity-only mode is still functional."
    )

    with gr.Blocks(title="PerplexityGuard") as app:
        gr.Markdown("# 🛡️ PerplexityGuard\n"
                    "**ESM-2 perplexity + commec homology** screen for AI-generated protein sequences.\n"
                    f"Backbone: `{scorer.model_name}`  •  Device: `{scorer.device}`  •  {commec_md}")
        with gr.Tabs():
            with gr.Tab("Screen from PDB"):
                with gr.Row():
                    pdb_in = gr.File(label="PDB structure", file_types=[".pdb"])
                    with gr.Column():
                        n_seq = gr.Slider(1, 16, value=8, step=1, label="Designs per PDB")
                        temp = gr.Slider(0.05, 1.0, value=0.1, step=0.05, label="Sampling temperature")
                        commec_chk = gr.Checkbox(value=avail.usable, label="Run commec homology screen")
                        go_btn = gr.Button("Generate & Screen", variant="primary")
                pdb_table = gr.Dataframe(label="Per-design results", datatype="markdown")
                pdb_summary = gr.Markdown()
                go_btn.click(
                    screen_pdb_handler,
                    inputs=[pdb_in, n_seq, temp, commec_chk],
                    outputs=[pdb_table, pdb_summary],
                )
            with gr.Tab("Screen a sequence"):
                seq_in = gr.Textbox(
                    label="Amino acid sequence",
                    placeholder="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                    lines=4,
                )
                commec_chk2 = gr.Checkbox(value=avail.usable, label="Run commec homology screen")
                go_btn2 = gr.Button("Screen sequence", variant="primary")
                seq_table = gr.Dataframe(label="Result", datatype="markdown")
                seq_summary = gr.Markdown()
                go_btn2.click(
                    screen_seq_handler,
                    inputs=[seq_in, commec_chk2],
                    outputs=[seq_table, seq_summary],
                )
        gr.Markdown(
            "### Verdict legend\n"
            f"<span style='color:{risk_color('PASS')};font-weight:600'>PASS</span>"
            " — natural-looking and clean; "
            f"<span style='color:{risk_color('REVIEW')};font-weight:600'>REVIEW</span>"
            " — high perplexity OR commec warn; "
            f"<span style='color:{risk_color('BLOCK')};font-weight:600'>BLOCK</span>"
            " — extreme perplexity OR regulated-pathogen homology hit.",
        )
    return app


def cmd_ui(args: argparse.Namespace) -> int:
    scorer = _load_scorer(args)
    runner = ProteinMPNNRunner(model_name=args.mpnn_model) if proteinmpnn_available() else None
    app = build_gradio_app(scorer, runner)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


# ---- argparse -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="perplexity_guard")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"ESM-2 model name (default: {DEFAULT_MODEL})")
    p.add_argument("--cpu", action="store_true", help="Force CPU (skip MPS/CUDA)")
    p.add_argument("--no-auto-calibrate", action="store_true",
                   help="Skip auto-calibration on first use of a new model")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="Run full pipeline on PDB(s)")
    s.add_argument("pdb", nargs="+", help="One or more PDB files")
    s.add_argument("--num-sequences", type=int, default=8)
    s.add_argument("--mpnn-temp", type=float, default=0.1)
    s.add_argument("--mpnn-model", default="v_48_020")
    s.add_argument("--soluble", action="store_true")
    s.add_argument("--seed", type=int, default=37)
    s.add_argument("--out", default="demo_results")
    s.add_argument("--no-commec", action="store_true")
    s.set_defaults(func=cmd_screen)

    s = sub.add_parser("score", help="Score amino-acid sequence(s)")
    s.add_argument("--aa")
    s.add_argument("--fasta")
    s.add_argument("--label")
    s.add_argument("--out")
    s.add_argument("--no-commec", action="store_true")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("calibrate", help="Compute calibration JSON from a FASTA")
    s.add_argument("fasta")
    s.add_argument("--n", type=int, default=0, help="Subset to first N sequences (0 = all)")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("ui", help="Launch the Gradio web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--share", action="store_true")
    s.add_argument("--mpnn-model", default="v_48_020")
    s.set_defaults(func=cmd_ui)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
