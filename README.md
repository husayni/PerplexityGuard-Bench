# PerplexityGuard-Bench

**An adversarial-robustness benchmark for sequence-naturalness DNA synthesis screens.**

AIxBio Hackathon 2026 — Track 1: DNA Screening & Synthesis Controls. Submitted by Hussain (solo team).

## TL;DR

- **The paper:** [`REPORT.pdf`](REPORT.pdf) — 11-page research report. Source: [`REPORT.md`](REPORT.md).
- **The benchmark:** `perplexity_guard/audit/` — a frozen attack battery + a five-screen evaluator + the figures and tables in the paper.
- **The reference implementation:** `perplexity_guard/core/` — ESM-2 pseudo-perplexity (whole-sequence and sliding-window) + Wootton-Federhen complexity + commec wrapper.
- **The headline result:** sliding-window pseudo-perplexity is the *structural patch* that closes the §4.2 mosaic-stitching evasion. At 50% natural-prefix budget, it recovers detection from 30% (whole-sequence OR-gate) to **70%** (OR-gate v2). The patch costs a 10% native-FPR, manageable via threshold retuning.
- **Commec status:** the `commec` homology arm is wired up **and was run end-to-end** on the n=120 main matrix (`commec_status=clear` for every row of `perplexity_guard/demo_results/e2e_demo.csv`). All test sequences are benign-derived, so commec correctly returns `clear` on every one — its empirical contribution to the OR-gate on this benign-derived corpus is therefore null, but the homology arm is part of the measured pipeline rather than a paper-only claim. To reproduce, install commec with the BLAST databases configured (see step 3 in *Reproducing the paper* below); without that the homology arm degrades to a structured `skipped` diagnostic and the perplexity + complexity arms carry the screen. See [`REPORT.md`](REPORT.md) §6.1.

## Reproducing the paper

Every number in the report is reproducible from this repository. Total wall-clock from a clean clone: **~20 min** for the headline experiments on Apple M5 / MPS using the fast model; **~60 min** if you also re-run the main matrix on the production t33_650M backbone.

```bash
# 1. Set up the Python environment (uv recommended).
uv sync

# 2. Clone ProteinMPNN (the structure-conditioned protein design model used
#    as one of the "real generator" attack classes in the §4.1 main matrix).
git clone --depth 1 https://github.com/dauparas/ProteinMPNN external/ProteinMPNN

# 3. Install commec for the homology arm. This was configured for the paper's
#    n=120 main matrix; every benign-derived sequence returned `clear`, so
#    commec contributes the orthogonal homology signal without altering
#    detection numbers on this corpus. Without commec the OR-gate's homology
#    arm degrades to a structured `skipped` diagnostic (perplexity + complexity
#    still run and carry the screen).
#    See https://commec.readthedocs.io for the BLAST-database setup; the
#    regulated-only mini-DB is a few GB, the full setup is ~50 GB.
#
# uv add "git+https://github.com/ibbis-screening/common-mechanism"
# brew install taxonkit && uv add pytaxonkit
# export COMMEC_DB=/path/to/commec_dbs
```

### The four reproduction commands

```bash
# A. Main matrix — 10 PDBs × 12 conditions = 120 sequences scored under
#    perplexity-only, complexity-only, OR-gate. Apple M5 / MPS, t33_650M, ~9 min.
uv run python -m perplexity_guard.tests.run_e2e_demo --num-sequences 4

# B. Cross-screen audit — re-derives detection-rate matrix from the per-sequence CSV.
#    Instant. Output: audit_matrix.csv + audit_matrix.png (Figure 1, Table 1 in REPORT.md).
uv run python -m perplexity_guard.audit.run_audit

# C. §4.2 stitching experiment — 10 PDBs × 3 reps × 9 prefix fractions = 280 mosaics.
#    Apple M5 / MPS, t12_35M (--fast), ~5-8 min.
uv run python -m perplexity_guard.audit.stitch_attack --fast --replicates 3

# D. §4.3 sliding-window patch — builds per-window calibration, scores all 280
#    stitch sequences with both whole-sequence and sliding-window perplexity,
#    applies all five screens, produces stitch_attack_v2_summary.csv +
#    stitch_attack_v2.png (Figure 3, Table 3 in REPORT.md). ~10 min on t12.
uv run python -m perplexity_guard.audit.sliding_window --fast --replicates 3 --score-stitch
uv run python -m perplexity_guard.audit.run_audit_v2
```

### Optional: t33 replication of the §4.2 stitch sweep

The original §4.2 stitch sweep was on `--fast` (t12_35M) for tractability. To reproduce on the production t33_650M backbone (~30–40 min on M5/MPS):

```bash
uv run python -m perplexity_guard.audit.stitch_t33
```

The §4.3 lemma predicts the structural averaging-vulnerability is independent of pLM size, so we expect the stitching curve to replicate within noise. The script writes outputs to `*_t33.csv` paths (so it doesn't clobber the t12 results) and prints a side-by-side comparison.

## Expected outputs

After running A–D above, you should see:

```
perplexity_guard/
├── calibration/
│   ├── facebook_esm2_t33_650M_UR50D.json         # whole-sequence calibration (mean=4.67 std=3.10)
│   ├── facebook_esm2_t12_35M_UR50D.json          # whole-sequence calibration (mean=12.09 std=3.06)
│   └── facebook_esm2_t12_35M_UR50D_window30.json # per-window calibration (mean=12.40 std=4.35 n=5,723)
├── demo_results/
│   ├── e2e_demo.csv                              # 120 rows, per-sequence signals + verdicts
│   ├── e2e_demo.json                             # run summary + per-PDB timings
│   └── e2e_summary.png                           # Figure 1 (Table 1 underlying)
└── audit/
    └── results/
        ├── audit_matrix.csv                      # 3 × 6 detection-rate matrix
        ├── audit_matrix.png                      # Figure 1
        ├── or_gate_consistency.json              # 120/120 OR-gate replication check
        ├── stitch_attack.csv                     # 280 rows, raw stitch signals
        ├── stitch_attack_summary.csv             # 3 × 9 stitch detection rates
        ├── stitch_attack.png                     # Figure 2
        ├── stitch_attack_sw.csv                  # 280 rows, with sliding-window column
        ├── stitch_attack_v2_summary.csv          # 5 × 9 stitch detection rates
        └── stitch_attack_v2.png                  # Figure 3
```

A **120/120 OR-gate replication match** between the deployed ensemble logic in `core/ensemble.py` and the audit's re-derived `verdict_or_gate` (in `audit/screen_variants.py`) is the in-paper sanity check that the audit is faithful to the deployed screen.

## Headline numbers

(All cross-checked against the source CSVs above; see `REPORT.md` for full tables.)

| screen | mpnn_T=0.1 | repeat_motif | stitch @ 50% prefix | stitch native-FPR |
|---|---:|---:|---:|---:|
| perplexity-only | 2.5% | 0% | 20% | 0% |
| complexity-only | 0% | 100% | 0% | 0% |
| OR-gate (whole-seq) | 10% | 100% | 30% | 0% |
| sliding-window (§4.3) | — | — | 57% | 10% |
| **OR-gate v2 (§4.3)** | — | — | **70%** | 10% |

The §4.3 *Lemma (Whole-sequence dilution)* proves the 30% cell is structural for any whole-sequence-averaging gate, not a tuning artifact. The §4.3 *Theorem (Position-resolved gates evade the lemma)* proves the +40 pp recovery in the bottom row is also structural — sliding-window reductions cannot exhibit the dilution behavior the corollary forces on whole-sequence gates.

## What's in the box

- `REPORT.md` / `REPORT.pdf` — the 11-page research report. The paper contains the full §3 methodology, §4 results, §4.3 theoretical lemma + structural patch, §5 discussion, §6 limitations & dual-use considerations, and the §A reproducibility appendix.
- `perplexity_guard/core/` — the reference implementation. ESM-2 scorer with per-position NLLs, auto-calibrating both whole-sequence and per-window. Wootton-Federhen + distinct-k-mer complexity detector. Back-translation + commec wrapper with structured "skipped" diagnostics. OR-gate verdict logic.
- `perplexity_guard/audit/` — the benchmark. `screen_variants.py` defines the five screens; `run_audit.py` produces the cross-screen matrix; `stitch_attack.py` and `sliding_window.py` run the §4.2 / §4.3 experiments; `stitch_t33.py` replicates §4.2 on the production model; `patch_report*.py` auto-fill data-dependent numbers into the report.
- `perplexity_guard/tests/` — 17 unit tests covering back-translation correctness, ensemble decision corner cases, and complexity-detector behavior. `run_e2e_demo.py` is the main matrix driver.
- `validate_perplexity.py` — the original signal-validation experiment (Cohen's d=2.76 on 40%-mutated UniProt; synthetic-proxy validation, see §6.1 for what this does not establish).
- `external/ProteinMPNN/` — generator used as one of the §4.1 attack classes; cloned during setup.

## Submission contact

- **Author:** Hussain (`hussainsyed.dev@gmail.com`)
- **Title:** *PerplexityGuard-Bench: An Adversarial-Robustness Benchmark for Sequence-Naturalness Synthesis Screens*
- **Track:** AIxBio Hackathon 2026 — Track 1 (DNA Screening & Synthesis Controls)
- **License:** MIT (this package). ProteinMPNN and commec retain their upstream licenses.
