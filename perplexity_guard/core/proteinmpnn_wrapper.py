"""Subprocess wrapper around dauparas/ProteinMPNN.

The upstream repo ships ``protein_mpnn_run.py`` with bundled vanilla weights.
We invoke it as a subprocess (rather than importing internals, which mutate
sys.path and globals) and parse the FASTA it writes into a structured list of
designs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Resolved at import time so the path is visible in error messages.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MPNN_DIR = _REPO_ROOT / "external" / "ProteinMPNN"


class ProteinMPNNError(RuntimeError):
    """Raised when ProteinMPNN subprocess execution fails."""


@dataclass
class MPNNDesign:
    """One sampled sequence from ProteinMPNN."""

    sequence: str
    score: float | None  # negative log-likelihood from ProteinMPNN, lower=better
    sample_index: int
    seed: int | None
    temperature: float
    source_pdb: str
    is_native: bool = False  # True for the line tagged as the original native sequence


def proteinmpnn_available(mpnn_dir: Path | None = None) -> bool:
    p = (mpnn_dir or DEFAULT_MPNN_DIR) / "protein_mpnn_run.py"
    return p.exists()


class ProteinMPNNRunner:
    """Wraps protein_mpnn_run.py.

    Parameters
    ----------
    mpnn_dir : Path | None
        Path to the cloned ProteinMPNN repository. Defaults to
        ``<repo>/external/ProteinMPNN``.
    model_name : str
        ProteinMPNN model variant (v_48_002, v_48_010, v_48_020, v_48_030).
        Higher noise → more diverse designs; v_48_020 is the typical default.
    use_soluble : bool
        Use the soluble-protein-only weights.
    """

    def __init__(
        self,
        mpnn_dir: Path | None = None,
        model_name: str = "v_48_020",
        use_soluble: bool = False,
    ) -> None:
        self.mpnn_dir = (mpnn_dir or DEFAULT_MPNN_DIR).resolve()
        if not (self.mpnn_dir / "protein_mpnn_run.py").exists():
            raise ProteinMPNNError(
                f"ProteinMPNN not found at {self.mpnn_dir}. "
                "Clone https://github.com/dauparas/ProteinMPNN into ./external/."
            )
        self.model_name = model_name
        self.use_soluble = use_soluble

    def design(
        self,
        pdb_path: Path | str,
        num_sequences: int = 8,
        temperature: float = 0.1,
        chains: Sequence[str] | None = None,
        seed: int = 37,
        out_dir: Path | None = None,
        timeout_s: int = 600,
    ) -> list[MPNNDesign]:
        pdb_path = Path(pdb_path).resolve()
        if not pdb_path.exists():
            raise ProteinMPNNError(f"PDB not found: {pdb_path}")
        cleanup = False
        if out_dir is None:
            out_dir = Path(tempfile.mkdtemp(prefix="mpnn_"))
            cleanup = True
        else:
            out_dir = Path(out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._run(pdb_path, out_dir, num_sequences, temperature, chains, seed, timeout_s)
            return self._collect(out_dir, pdb_path, temperature, seed)
        finally:
            if cleanup:
                shutil.rmtree(out_dir, ignore_errors=True)

    def _run(
        self,
        pdb_path: Path,
        out_dir: Path,
        num_sequences: int,
        temperature: float,
        chains: Sequence[str] | None,
        seed: int,
        timeout_s: int,
    ) -> None:
        cmd = [
            sys.executable,
            str(self.mpnn_dir / "protein_mpnn_run.py"),
            "--pdb_path", str(pdb_path),
            "--out_folder", str(out_dir),
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", f"{temperature}",
            "--seed", str(seed),
            "--batch_size", "1",
            "--model_name", self.model_name,
            "--suppress_print", "1",
        ]
        if self.use_soluble:
            cmd.append("--use_soluble_model")
        if chains:
            cmd += ["--pdb_path_chains", " ".join(chains)]
        env = os.environ.copy()
        # Make the bundled protein_mpnn_utils import deterministic.
        env["PYTHONPATH"] = f"{self.mpnn_dir}{os.pathsep}{env.get('PYTHONPATH','')}"
        t0 = time.time()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.mpnn_dir,
                env=env,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise ProteinMPNNError(f"ProteinMPNN timed out after {timeout_s}s") from e
        if res.returncode != 0:
            raise ProteinMPNNError(
                f"ProteinMPNN exited {res.returncode}\n"
                f"--- stdout ---\n{res.stdout}\n"
                f"--- stderr ---\n{res.stderr}"
            )
        elapsed = time.time() - t0
        print(f"[mpnn] designed {num_sequences} sequences for {pdb_path.name} in {elapsed:.1f}s")

    @staticmethod
    def _collect(
        out_dir: Path,
        pdb_path: Path,
        temperature: float,
        seed: int,
    ) -> list[MPNNDesign]:
        # ProteinMPNN writes <out>/seqs/<stem>.fa with paired header/sequence lines.
        stem = pdb_path.stem
        fa_path = out_dir / "seqs" / f"{stem}.fa"
        if not fa_path.exists():
            raise ProteinMPNNError(f"Expected output not found: {fa_path}")
        text = fa_path.read_text()
        designs: list[MPNNDesign] = []
        header: str | None = None
        seq_chunks: list[str] = []
        records: list[tuple[str, str]] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
        for idx, (h, s) in enumerate(records):
            score = _parse_score(h)
            sample_idx = _parse_sample(h)
            # ProteinMPNN's FASTA convention: the first record is the input
            # native re-scored under the model and has no `sample=` tag; all
            # actual samples have `sample=N`.
            is_native = sample_idx is None
            designs.append(
                MPNNDesign(
                    sequence=s,
                    score=score,
                    sample_index=sample_idx if sample_idx is not None else -1,
                    seed=seed,
                    temperature=temperature,
                    source_pdb=pdb_path.name,
                    is_native=is_native,
                )
            )
        return designs


_SCORE_RE = re.compile(r"score=([-+]?\d*\.?\d+)")
_SAMPLE_RE = re.compile(r"sample=(\d+)")


def _parse_score(header: str) -> float | None:
    m = _SCORE_RE.search(header)
    return float(m.group(1)) if m else None


def _parse_sample(header: str) -> int | None:
    m = _SAMPLE_RE.search(header)
    return int(m.group(1)) if m else None
