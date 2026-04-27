# PerplexityGuard

**An audit harness and reference implementation for pLM-based DNA synthesis screens.**

> *"PerplexityGuard audits pLM-based synthesis screens, identifies their structural failure modes, and validates a sliding-window patch."*

## What it does

This package is two things at once:

1. **PerplexityGuard-Bench** (`audit/`): a frozen attack battery, a standardized signal-table format, and a one-command evaluator that produces a cross-screen detection-rate matrix. Adding a new screen is a `decide(row) → verdict` function under 30 LOC. See `REPORT.md` (top of repo) for the published audit results.
2. **PerplexityGuard** (`core/`): a reference pLM-naturalness screen for proteins about to be ordered as synthetic genes. It composes ESM-2 pseudo-perplexity, a Wootton-Federhen + distinct-k-mer low-complexity detector, and a wrapper around the IBBIS Common Mechanism (`commec`).

**What the audit found** (full numbers in `REPORT.md`): single-signal pLM perplexity is structurally vulnerable to mosaic-stitching adversaries (a 50% natural-prefix attack drops detection from 80% → 20%), and even a whole-sequence OR-gate over perplexity + low-complexity detects only 10% of low-temperature ProteinMPNN designs — the operationally-relevant adversarial setting. A sliding-window perplexity gate (the §4.3 structural patch) recovers detection at 50% prefix from 30% to 70% on the stitching attack, at a 10% native-FPR cost.

**What this package is *not*:** a ready-for-deployment defense. The 10% T=0.1 ProteinMPNN gap is unpatched, and the `commec` integration is wired up but not empirically validated in this work (see *Honest limitations* below).

The naturalness signal was validated separately with `validate_perplexity.py`
at the repo root: ESM-2 t12_35M separates 60 reviewed UniProt sequences from
40%-mutated variants with **Cohen's d = 2.76**, **AUROC = 0.987**, **p = 1.9e-20** (synthetic-proxy validation only — see `REPORT.md` §6.1 for what this does *not* establish).

## Pipeline

```
PDB structure
    │
    ▼
┌──────────────────────┐
│  ProteinMPNN         │  generate N candidate sequences from the backbone
│  (subprocess wrap)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐    ┌────────────────────────┐
│  ESM-2 pseudo-PPL    │    │  back-translate AA→DNA │
│  (ESM2Scorer, MPS/   │    └──────────┬─────────────┘
│   CUDA, cached)      │               │
└──────────┬───────────┘               ▼
           │             ┌──────────────────────┐
           │             │  commec homology     │
           │             │  (skipped if DBs not │
           │             │   configured)        │
           │             └──────────┬───────────┘
           ▼                        ▼
        ┌──────────────────────────────┐
        │  Ensemble verdict            │
        │  PASS / REVIEW / BLOCK       │
        └──────────────────────────────┘
```

## Repository layout

```
perplexity_guard/
├── perplexity_guard.py     # CLI: screen / score / calibrate / ui
├── core/
│   ├── esm_analysis.py     # ESM-2 pseudo-perplexity + calibration
│   ├── proteinmpnn_wrapper.py
│   ├── screening.py        # back-translation + commec wrapper + stub
│   └── ensemble.py         # PASS / REVIEW / BLOCK fusion logic
├── tests/
│   ├── test_validation.py  # unit tests (12 passing)
│   ├── run_e2e_demo.py     # downloads PDBs, runs full pipeline
│   └── sample_pdbs/        # populated on first run
├── calibration/            # model-specific reference distributions (JSON)
├── demo_results/           # CSV + JSON dumps from e2e runs
├── requirements.txt
└── README.md
```

External dependencies live in `../external/`:
- `external/ProteinMPNN/` — cloned from https://github.com/dauparas/ProteinMPNN

## Setup

The repo uses [uv](https://github.com/astral-sh/uv).

```bash
# 1. Clone & install Python deps
git clone <this repo>
cd aixbio
uv sync

# 2. Clone ProteinMPNN (model weights are bundled in the repo)
mkdir -p external && cd external
git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git
cd ..

# 3. (Optional) install commec for real homology screening.
#    NB: commec needs `taxonkit` (Go binary) + ~50 GB of NCBI/UniProt BLAST DBs.
#    The pipeline runs perplexity-only when commec is not configured.
uv add "git+https://github.com/ibbis-screening/common-mechanism"
# Then follow https://commec.readthedocs.io to set up databases and export
# the COMMEC_DB env var.
```

ESM-2 weights download automatically on first use into the HuggingFace cache.

## Quick start

### Unit tests

```bash
uv run python -m unittest perplexity_guard.tests.test_validation -v
```

### End-to-end demo

```bash
uv run python -m perplexity_guard.tests.run_e2e_demo --fast --num-sequences 4
```

This downloads three benign PDBs (1UBQ ubiquitin, 2LYZ lysozyme, 1MBO myoglobin),
ProteinMPNN-redesigns each, scores every native + design with ESM-2, and writes
`demo_results/e2e_demo.{csv,json}` with the verdict table and runtime metrics.

`--fast` uses `facebook/esm2_t12_35M_UR50D` (35M params, ~1 s/sequence on M-series Mac).
Drop the flag to run the production-grade `facebook/esm2_t33_650M_UR50D` (~5–10 s/sequence).

### Calibrate against your own corpus

The default calibration is conservative. To build a model-specific reference:

```bash
uv run python -m perplexity_guard.perplexity_guard \
    --model facebook/esm2_t33_650M_UR50D \
    calibrate path/to/natural_proteins.fasta --n 200
# writes perplexity_guard/calibration/facebook_esm2_t33_650M_UR50D.json
```

### Command-line screening

```bash
# Full pipeline on PDBs
uv run python -m perplexity_guard.perplexity_guard screen \
    perplexity_guard/tests/sample_pdbs/1UBQ.pdb \
    --num-sequences 8 --out my_results/

# Score one or more amino acid sequences directly
uv run python -m perplexity_guard.perplexity_guard score \
    --aa MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

### Gradio web app

```bash
uv run python -m perplexity_guard.perplexity_guard ui --port 7860
# open http://127.0.0.1:7860
```

The app has two tabs:
- **Screen from PDB** — upload a `.pdb`, choose number of designs and sampling
  temperature, hit *Generate & Screen*.
- **Screen a sequence** — paste an amino acid sequence directly.

Verdicts are colour-coded: green PASS, amber REVIEW, red BLOCK.

## Verdict logic

Three orthogonal signals, two-tailed gate to close evasion paths:

```
BLOCK    commec regulated-pathogen homology hit, OR
         perplexity z-score ≥ 3 (≈ 99.9th percentile of natural), OR
         low-complexity fraction ≥ 0.50 (closes the repeat-motif evasion).

REVIEW   perplexity ≥ p95 threshold OR z ≥ 1.5,
         OR commec returned "warn",
         OR low-complexity fraction ≥ 0.20,
         OR commec skipped AND perplexity z ≥ 1.0.

PASS     otherwise.
```

Why three signals: the validation phase showed pseudo-perplexity catches
out-of-distribution sequences but is **evaded** by tandem-repeat designs,
which score *below* natural perplexity. A complementary Wootton-Federhen
low-complexity check closes that hole — together they form a two-tailed
gate. commec adds an orthogonal homology axis that catches sequences
resembling regulated agents regardless of how natural they look to a pLM.

Thresholds are stored in `core/esm_analysis.py::_FALLBACK_CALIBRATION` and
the `LOW_COMPLEXITY_*` constants in `core/ensemble.py`. Per-model
calibration JSON files in `calibration/` override the fallback (see the
auto-calibration section below).

## Auto-calibration

The first time you instantiate `ESM2Scorer(..., auto_calibrate=True,
calibration_dir=…)` for a given model, the scorer will:

1. Look for `calibration/<safe(model_name)>.json`. If present, load it and
   skip the rest.
2. Otherwise fetch ~80 reviewed UniProt sequences (length 80–200) into
   `calibration/uniprot_sample.fasta` (cached).
3. Score the first 50 with the loaded ESM-2 model, compute mean / std /
   p50 / p95 / p99, and persist to JSON.

The fallback distribution shipped in code is intentionally loose; the
auto-calibrated JSON is what the verdict logic actually uses once
present. Auto-calibration takes ~5 min for `t33_650M` on M-series MPS,
~30 s for `t12_35M`. You can also run it explicitly:

```bash
uv run python -m perplexity_guard.perplexity_guard \
    --model facebook/esm2_t33_650M_UR50D \
    calibrate calibration/uniprot_sample.fasta --n 50
```

## What commec adds when configured

Real commec runs nucleotide and protein BLAST against IBBIS-curated
regulated-pathogen and benign-protein databases, and parses the hits into
a structured "regulated / benign / unclear" verdict. PerplexityGuard treats
a regulated hit as a hard BLOCK, regardless of perplexity. Without commec,
the system runs in perplexity-only mode and flags everything that looks
statistically out-of-distribution for review.

## Honest limitations

- **Validated on synthetic AI proxies, not real generators.** The original
  signal validation used 40%-mutated UniProt sequences, not real ProteinMPNN /
  RFdiffusion output. Replacing those proxies with real generator dumps is
  the most important next step; expect the effect size to shrink.
- **Repeat-motif designs evade.** The same validation found that
  tandem-repeat sequences score *below* natural perplexity. The verdict
  logic here treats `z ≥ 3` only — a determined adversary could in
  principle stay below the gate. Pair with a low-complexity / repeat
  detector if that threat matters.
- **commec setup is heavy.** The runtime detects missing taxonkit /
  databases and falls back gracefully, but real homology screening is
  only available when the operator has done the database setup.
- **Defensive tool, not an oracle.** This is one tripwire among many. False
  positives on natural intrinsically-disordered domains are inevitable.

## Performance (Apple M5, 24 GB)

ESM-2 t12_35M, batch_size=16, sequence length ~150 residues:

| stage                         | wall-time      |
|-------------------------------|----------------|
| ESM-2 model load (cold)       | ~8 s           |
| ESM-2 score per sequence      | ~1.0–1.5 s     |
| ProteinMPNN per PDB (4 seqs)  | ~20–30 s       |
| commec screen (when enabled)  | tens of seconds (DB-dependent) |

Numbers for the production t33_650M backbone run ~5–10× slower per
sequence; budget ~5–8 s per design on M-series and a few minutes for the
full 3-PDB demo.

## License

This package is released under the MIT license. ProteinMPNN and commec
retain their upstream licenses; see their repositories.
