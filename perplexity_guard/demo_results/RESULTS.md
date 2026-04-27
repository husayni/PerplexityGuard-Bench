# PerplexityGuard E2E demo — final results

**Configuration**: ESM-2 `t33_650M_UR50D` (production model), Apple M5 / MPS,
auto-calibrated on 50 reviewed UniProt sequences (mean PPL = 4.67, std = 3.10,
p95 = 12.39, p99 = 14.42).

**Dataset**: 10 benign, well-characterised PDBs spanning all-α, all-β, α/β, and
EF-hand folds; lengths 54–153 aa.

| PDB  | protein                                  | length | fold        |
|------|------------------------------------------|-------:|-------------|
| 1UBQ | ubiquitin                                |   76   | β-grasp     |
| 2LYZ | hen egg-white lysozyme                   |  129   | α/β         |
| 1MBO | sperm-whale myoglobin                    |  153   | all-α       |
| 1ENH | engrailed homeodomain                    |   54   | all-α       |
| 1PGA | Streptococcus protein G B1 domain        |   56   | β-grasp     |
| 1UZC | cold-shock protein                       |   69   | β           |
| 2IGD | immunoglobulin-binding domain            |   61   | β           |
| 3ICB | bovine calbindin D9k                     |   75   | EF-hand     |
| 1BPI | bovine pancreatic trypsin inhibitor      |   58   | mixed       |
| 1CTF | E. coli L7/L12 ribosomal C-terminal      |   68   | all-α       |

**Per PDB**: 1 native + 4 ProteinMPNN samples at T=0.1 + 4 at T=1.0 + 1
mutated_40pct + 1 shuffled + 1 repeat_motif = **12 sequences**, **120 total**.

---

## Verdict matrix

| condition       | n  | PASS | REVIEW | BLOCK | detection rate |
|-----------------|---:|-----:|-------:|------:|---------------:|
| **native**      | 10 | 10   |   0    |   0   | **0%** (FPR ✓) |
| mpnn_T=0.1      | 40 | 36   |   4    |   0   | 10%            |
| mpnn_T=1.0      | 40 |  3   |  31    |   6   | **92.5%**      |
| mutated_40pct   | 10 |  0   |   3    |   7   | **100%**       |
| shuffled        | 10 |  0   |   1    |   9   | **100%**       |
| repeat_motif    | 10 |  0   |   0    |  10   | **100%** ← evasion closed |

**Headline numbers**

- **False-positive rate on natives: 0.00%** (target <5%, n=10 across 8 species).
- **Legitimate near-native MPNN redesigns (T=0.1) pass at 90%** — the system
  does not over-flag computational designs that recapitulate the native fold.
- **Aggressive MPNN redesigns (T=1.0): 92.5% caught.**
- **Synthetic AI proxies (mutated_40pct, shuffled, repeat_motif): all 100%.**
- **Tandem-repeat evasion path is closed.** repeat_motif scores PPL ≈ 1.0
  (below natural) yet is BLOCKed because the Wootton-Federhen entropy and
  distinct-3-mer signal both fire (repetitiveness 0.75–0.91, distinct-3-mer
  fraction 0.05–0.12).

---

## Mean perplexity / repetitiveness by condition

| condition       |  n  | mean PPL | mean z   | mean repetitiveness | distinct-3-mer mean |
|-----------------|----:|---------:|---------:|--------------------:|---------------------:|
| native          | 10  |   3.91   | −0.25    |        0.00         |  ~0.98               |
| mpnn_T=0.1      | 40  |   5.25   | +0.18    |        0.01         |  ~0.97               |
| mpnn_T=1.0      | 40  |  11.07   | +2.07    |        0.00         |  ~0.99               |
| mutated_40pct   | 10  |  15.66   | +3.55    |        0.00         |  ~1.00               |
| shuffled        | 10  |  17.89   | +4.27    |        0.00         |  ~1.00               |
| repeat_motif    | 10  |   1.01   | −1.18    |        0.85         |  ~0.08               |

The two signals are orthogonal: high-PPL conditions cluster on the right of
panel 1 (perplexity), the repeat condition lights up panel 2
(repetitiveness). Panel 3 (verdicts) shows the OR-gate doing its job.

---

## Runtime

ESM-2 t33_650M on Apple M5 (24 GB, MPS):

| stat                   | value     |
|------------------------|-----------|
| model load (warm cache)|  ~3 s     |
| ESM PPL per seq (mean) | **4.12 s**|
| ESM PPL per seq (median)|  2.56 s   |
| ESM PPL per seq (max)  | 13.45 s   |
| ProteinMPNN per PDB    |  1.5–4 s  |
| total wall-clock       | **~9 min** for 120 sequences |

Per-PDB wall (full pipeline, 12 sequences each):

| PDB  | wall (s) |
|------|---------:|
| 1ENH |   21.2   |
| 1PGA |   23.0   |
| 1BPI |   24.0   |
| 2IGD |   26.4   |
| 1UZC |   33.1   |
| 1CTF |   33.0   |
| 3ICB |   39.5   |
| 1UBQ |   40.2   |
| 2LYZ |  114.0   |
| 1MBO |  162.4   |

The two long PDBs (2LYZ at 129 aa, 1MBO at 153 aa) dominate wall-clock
because ESM-2 pseudo-perplexity is *L* forward passes per sequence; for the
median 60-aa protein, full screening completes in ~25 s. **Well under the
<15 s/design target on every condition.**

---

## What the four submission-readiness fixes did

1. **Auto-calibration** (`ESM2Scorer(auto_calibrate=True)`).
   First time the production model is used, the scorer fetches an 80-seq
   UniProt sample, scores 50, and writes a model-specific JSON to
   `calibration/`. Subsequent runs hit the cache. Fixed the previous
   miscalibration where t33 fallback (mean=9.0) caused mutated_40pct
   detection to drop to 33%; with the empirical mean=4.67 it climbs to
   **100%**.

2. **Wootton-Federhen + k-mer redundancy detector** (`core/complexity.py`).
   Two-tailed gate that closes the repeat-motif evasion. Repetitiveness
   score combines per-window normalized Shannon entropy (catches
   homopolymers, low-residue runs) with distinct-3-mer fraction (catches
   tandem-repeat tiles whose motif has 4+ residues — the case the
   entropy-only detector missed). Detection on `repeat_motif` went from
   **0% → 100%** with this fix.

3. **commec diagnostics for Mac** (`core/screening.py`).
   When commec is unusable, the wrapper now reports a structured
   `diagnosis` and `fix_hint`. On this machine: *"commec needs the
   pytaxonkit Python wrapper plus the taxonkit Go binary. On macOS, both
   are easy to add but require a system install, not just pip install
   commec.* — fix: `brew install taxonkit && uv add pytaxonkit`*". The
   ensemble keeps running in perplexity-only mode and labels every result
   with `homology_skipped` so the operator knows.

4. **10 PDBs spanning four folds** (defaults in `tests/run_e2e_demo.py`).
   Plus a side-fix to the PDB parser that previously concatenated all NMR
   models in 1UZC into a 1587-aa "sequence"; the parser now stops at the
   first `ENDMDL`.

---

## Comparison to the previous 3-PDB run

The earlier run on 1UBQ + 2LYZ + 1MBO with t33_650M reported:
- mutated_40pct: 100%, shuffled: 100%, mpnn_T=1.0: 91.7%, native FPR: 0%

The new 10-PDB run **reproduces those numbers and adds the closed
repeat_motif evasion**. The natives FPR holds at 0% as the panel
expands, which is the most important calibration sanity check.

---

## Unit-test coverage

`uv run python -m unittest perplexity_guard.tests.test_validation` — **17 tests pass**:

- 5 back-translation tests (round-trip, stop codon handling, lowercase, unknown residues)
- 8 ensemble decision tests (natural→PASS, high PPL→REVIEW, extreme PPL→BLOCK,
  regulated commec→BLOCK, warn commec→REVIEW, borderline+skipped→REVIEW,
  **repeat_motif→BLOCK** *(new — closes the evasion path)*)
- 4 complexity tests (natural high, repeat low, homopolymer zero, empty default)

---

## Caveats (still honest, still apply)

- **Synthetic proxies are not real generators.** The 100% on
  mutated_40pct / shuffled / repeat_motif is on *constructed* AI-failure
  modes. Real RFdiffusion+MPNN designs on novel backbones are the next
  test set and the most important next milestone.
- **commec didn't run.** The diagnostics are clearer now, but the
  ensemble's homology arm is unmeasured. If you can spin up the IBBIS
  regulated-pathogen mini-DB before submission, you'll have an
  actual measurement to put in the writeup.
- **Calibration is corpus-specific.** The 50-seq UniProt sample (length
  80–200, reviewed only) defines "natural"; narrower domains
  (intrinsically-disordered, antibody, etc.) need their own calibration
  via `perplexity_guard calibrate <fasta>`.

## Reproducibility

```bash
uv sync
git clone --depth 1 https://github.com/dauparas/ProteinMPNN external/ProteinMPNN

# One command — auto-calibrates on first use, runs the matrix:
uv run python -m perplexity_guard.tests.run_e2e_demo --num-sequences 4
```

Outputs in `perplexity_guard/demo_results/`:
- `e2e_demo.csv` — 120-row per-sequence matrix.
- `e2e_demo.json` — full structured summary with calibration + timings.
- `e2e_summary.png` — three-panel plot (perplexity, repetitiveness, verdicts).
- `e2e_t33_final.log` — full console transcript.
