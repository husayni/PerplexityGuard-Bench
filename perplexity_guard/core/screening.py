"""Back-translation + commec (Common Mechanism) homology screening.

The Common Mechanism is the IBBIS-maintained, open-source bioseq screening
tool. It runs nucleotide and protein BLAST against curated regulated-pathogen
databases and a benign protein database, and returns a structured verdict.

Setup is heavy: it needs `taxonkit` (Go binary), pytaxonkit, and tens of GB of
NCBI/UniProt BLAST databases. The wrapper here detects whether the runtime is
configured and either runs commec or returns a transparent ``skipped`` result.
This keeps the demo runnable on a laptop without preventing a properly
configured operator from seeing real homology hits.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Most-frequent codons across the universal genetic code, with stop=TAA.
# Choice biased toward E. coli / human consensus so screening tools see DNA
# that resembles real ordered synthetic genes.
_CODON_TABLE: dict[str, str] = {
    "A": "GCC", "R": "CGC", "N": "AAC", "D": "GAC", "C": "TGC",
    "E": "GAG", "Q": "CAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}
_STOP_CODON = "TAA"


def back_translate_aa_to_dna(aa_sequence: str, append_stop: bool = True) -> str:
    """Most-frequent-codon back-translation of an amino acid sequence to DNA.

    This is intentionally deterministic — homology-based screens key off
    sequence identity, not codon noise, so a deterministic mapping keeps the
    screening output reproducible across re-runs.
    """
    seq = aa_sequence.upper().strip()
    out: list[str] = []
    for aa in seq:
        if aa == "*":
            continue
        codon = _CODON_TABLE.get(aa)
        if codon is None:
            raise ValueError(f"non-standard residue {aa!r} cannot be back-translated")
        out.append(codon)
    if append_stop:
        out.append(_STOP_CODON)
    return "".join(out)


# ---- commec availability ------------------------------------------------


@dataclass
class CommecAvailability:
    cli_present: bool
    cli_runnable: bool
    db_configured: bool
    runtime_error: str | None
    db_path: str | None
    diagnosis: str = ""           # human-friendly explanation
    fix_hint: str = ""            # actionable next step

    @property
    def usable(self) -> bool:
        return self.cli_present and self.cli_runnable and self.db_configured


def _check_pytaxonkit_import() -> str | None:
    """Returns None if pytaxonkit imports cleanly, else the error message."""
    try:
        import pytaxonkit  # noqa: F401
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _check_taxonkit_binary() -> str | None:
    """Returns None if `taxonkit` Go binary is on PATH, else a hint string."""
    if shutil.which("taxonkit") is None:
        return "taxonkit Go binary not on PATH"
    return None


def commec_available() -> CommecAvailability:
    """Probe whether the commec CLI is installed AND runnable AND has DBs.

    On Mac the most common failure is that ``pytaxonkit`` / ``taxonkit`` is
    missing (Go toolchain), so we check that explicitly and surface a
    Mac-friendly install hint instead of leaving the user to read tracebacks.
    """
    cli = shutil.which("commec")
    if cli is None:
        return CommecAvailability(
            cli_present=False, cli_runnable=False, db_configured=False,
            runtime_error="commec binary not on PATH", db_path=None,
            diagnosis="commec is not installed in this environment",
            fix_hint=(
                "Install with: uv add 'git+https://github.com/"
                "ibbis-screening/common-mechanism' (pulls Python deps)"
            ),
        )
    pytaxonkit_err = _check_pytaxonkit_import()
    taxonkit_err = _check_taxonkit_binary()
    try:
        proc = subprocess.run(
            [cli, "--help"], capture_output=True, text=True, timeout=20
        )
    except Exception as e:
        return CommecAvailability(
            cli_present=True, cli_runnable=False, db_configured=False,
            runtime_error=f"--help failed: {e}", db_path=None,
            diagnosis="commec CLI is installed but failed to start",
            fix_hint="Inspect `commec --help` output manually to see the trace.",
        )
    if proc.returncode != 0:
        err_text = (proc.stderr or proc.stdout).strip()
        first = err_text.splitlines()[-1] if err_text else "non-zero exit"
        diagnosis: str
        fix_hint: str
        if "pytaxonkit" in err_text or "taxonkit" in err_text.lower():
            diagnosis = (
                "commec needs the pytaxonkit Python wrapper plus the taxonkit "
                "Go binary. On macOS, both are easy to add but require a system "
                "install, not just `pip install commec`."
            )
            fix_hint = (
                "brew install taxonkit  # Go binary\n"
                "uv add pytaxonkit       # Python wrapper\n"
                "Then retry `commec --help`."
            )
        elif "ModuleNotFoundError" in err_text:
            diagnosis = "commec is missing a Python dependency."
            fix_hint = f"Install the missing module then retry. Last line: {first}"
        else:
            diagnosis = "commec CLI exited non-zero on --help."
            fix_hint = f"Last error line: {first}"
        return CommecAvailability(
            cli_present=True, cli_runnable=False, db_configured=False,
            runtime_error=first, db_path=None,
            diagnosis=diagnosis, fix_hint=fix_hint,
        )
    if pytaxonkit_err is not None or taxonkit_err is not None:
        # `commec --help` succeeded but the screening backend will trip on it.
        details = "; ".join(x for x in (pytaxonkit_err, taxonkit_err) if x)
        return CommecAvailability(
            cli_present=True, cli_runnable=False, db_configured=False,
            runtime_error=details, db_path=None,
            diagnosis=(
                "commec --help works but the BLAST backend depends on taxonkit, "
                "which is not importable. Real screening calls would fail."
            ),
            fix_hint="brew install taxonkit && uv add pytaxonkit",
        )
    db = os.environ.get("COMMEC_DB") or os.environ.get("COMMON_MECHANISM_DB")
    if not db:
        return CommecAvailability(
            cli_present=True, cli_runnable=True, db_configured=False,
            runtime_error="COMMEC_DB env var not set", db_path=None,
            diagnosis=(
                "commec runs but has no database to screen against — it needs "
                "the regulated-pathogen + benign BLAST DBs locally."
            ),
            fix_hint=(
                "Follow https://commec.readthedocs.io database setup, then\n"
                "  export COMMEC_DB=/path/to/commec_dbs\n"
                "Minimal setup (regulated only) is ~few GB; full setup ~50 GB."
            ),
        )
    if not Path(db).exists():
        return CommecAvailability(
            cli_present=True, cli_runnable=True, db_configured=False,
            runtime_error=f"COMMEC_DB={db} does not exist", db_path=db,
            diagnosis=f"COMMEC_DB points at {db}, which is not a directory.",
            fix_hint="Check the path; rerun setup if the DBs were never built.",
        )
    return CommecAvailability(
        cli_present=True, cli_runnable=True, db_configured=True,
        runtime_error=None, db_path=db,
        diagnosis="commec is fully configured and ready to screen.",
        fix_hint="",
    )


# ---- Result schema ------------------------------------------------------


@dataclass
class CommecResult:
    status: str  # "clear", "flag", "warn", "skipped", "error"
    verdict: str  # human-readable
    runtime_seconds: float
    hits: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    skipped_reason: str | None = None
    db_path: str | None = None

    @property
    def is_skipped(self) -> bool:
        return self.status == "skipped"

    @property
    def is_clear(self) -> bool:
        return self.status == "clear"

    @property
    def is_hit(self) -> bool:
        return self.status in ("flag", "warn")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "runtime_seconds": self.runtime_seconds,
            "hits": self.hits,
            "skipped_reason": self.skipped_reason,
            "db_path": self.db_path,
        }


# ---- commec invocation --------------------------------------------------


def run_commec_screening(
    dna_sequence: str,
    name: str = "query",
    work_dir: Path | None = None,
    timeout_s: int = 1200,
) -> CommecResult:
    """Run commec screen on a DNA sequence; return a structured result.

    If commec is not installed/configured, returns ``status="skipped"`` with
    the reason. Callers should treat ``skipped`` as a non-veto in the
    ensemble — perplexity then carries the screening burden.
    """
    avail = commec_available()
    if not avail.usable:
        reason = avail.runtime_error or "unknown"
        return CommecResult(
            status="skipped",
            verdict=f"commec unavailable: {reason}",
            runtime_seconds=0.0,
            skipped_reason=reason,
            db_path=avail.db_path,
        )

    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="commec_"))
        cleanup = True
    else:
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        fasta = work_dir / f"{_safe(name)}.fasta"
        fasta.write_text(f">{name}\n{dna_sequence}\n")
        out_prefix = work_dir / "screen"
        # commec v0.3.2 takes the FASTA as a positional argument; --fast
        # skips the NR/NT BLAST steps that the minimal biorisk-only DB
        # cannot serve. -F overwrites prior outputs in this work_dir
        # (idempotent reruns). The biorisk HMM scan and benign-DB clearance
        # still execute, which is exactly the "regulated-pathogen mini-DB"
        # screening operators want.
        cmd = [
            "commec",
            "screen",
            "--fast",
            "-F",
            "--databases", str(avail.db_path),
            "--output", str(out_prefix),
            str(fasta),
        ]
        env = os.environ.copy()
        # EMBOSS' transeq needs ACDROOT set; do it here so callers don't
        # have to remember to export it from the shell.
        if "EMBOSS_ACDROOT" not in env:
            for candidate in (
                "/opt/homebrew/share/EMBOSS/acd",
                "/usr/local/share/EMBOSS/acd",
            ):
                if Path(candidate).exists():
                    env["EMBOSS_ACDROOT"] = candidate
                    break
        t0 = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
            env=env,
        )
        elapsed = time.time() - t0
        raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            return CommecResult(
                status="error",
                verdict=f"commec exited {proc.returncode}: {raw.strip()[-300:]}",
                runtime_seconds=elapsed,
                raw_output=raw,
                db_path=avail.db_path,
            )
        result_file = _find_screen_output(out_prefix)
        hits, status, verdict = _parse_commec_output(result_file, raw)
        return CommecResult(
            status=status,
            verdict=verdict,
            runtime_seconds=elapsed,
            hits=hits,
            raw_output=raw,
            db_path=avail.db_path,
        )
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:64] or "query"


def _find_screen_output(prefix: Path) -> Path | None:
    """commec writes per-query .screen and JSON files; pick the JSON if present."""
    candidates = [
        prefix.with_suffix(".screen.json"),
        prefix.with_suffix(".json"),
        Path(str(prefix) + ".screen"),
    ]
    for c in candidates:
        if c.exists():
            return c
    parent = prefix.parent
    json_files = sorted(parent.glob(f"{prefix.name}*.json"))
    if json_files:
        return json_files[0]
    return None


def _parse_commec_output(
    result_file: Path | None,
    raw_text: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Best-effort parse of commec output into (hits, status, verdict).

    For v0.3.2 the .screen log uses the markers:
      - "Biorisks: no hits detected, PASS"        → clear
      - "Biorisks: HIT" / "FAIL" / "biorisk concerns found" → flag
      - "review" / "WARN"                          → warn
    Substring matches are case-sensitive on the *original* text so that
    unrelated log strings like "no regulated regions to clear" don't
    accidentally trigger a flag verdict.
    """
    if result_file is not None and result_file.suffix.endswith("json"):
        try:
            data = json.loads(result_file.read_text())
            hits = data.get("hits", []) or []
            verdict_text = (data.get("verdict") or data.get("recommendation") or "").lower()
            if "block" in verdict_text or "regulated pathogen" in verdict_text or "biorisk" in verdict_text:
                return hits, "flag", str(data.get("verdict", "homology hit"))
            if "warn" in verdict_text or "review" in verdict_text:
                return hits, "warn", str(data.get("verdict", "review recommended"))
            if "clear" in verdict_text or "pass" in verdict_text or not hits:
                return hits, "clear", "no regulated-pathogen homology"
            return hits, "warn", str(data.get("verdict", "review recommended"))
        except Exception:
            pass
    # v0.3.2 plain-text log fallback. Use the exact verdict markers commec emits.
    if "Biorisks: HIT" in raw_text or "biorisk concerns found" in raw_text:
        return [], "flag", "commec biorisk HMM hit"
    if "no biorisk concerns found" in raw_text or "no hits detected, PASS" in raw_text:
        return [], "clear", "no biorisk HMM hit"
    if " FAIL" in raw_text:  # leading space avoids matching --fail-on, etc.
        return [], "flag", "commec FAIL verdict"
    if "WARN" in raw_text or "review recommended" in raw_text:
        return [], "warn", "commec WARN / review"
    # Default to clear when commec exited 0 and emitted no hit markers.
    return [], "clear", "no regulated-pathogen homology detected"
