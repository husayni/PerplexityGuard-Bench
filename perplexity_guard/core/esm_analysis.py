"""ESM-2 pseudo-perplexity scoring and natural-distribution calibration.

Pseudo-perplexity for an L-residue sequence S is

    PPL(S) = exp( -1/L * sum_i log P(s_i | s_1..s_{i-1}, MASK, s_{i+1}..s_L) ),

i.e. each residue is masked one at a time and the model's log-prob on the
ground-truth residue is averaged over the sequence. We use this as a
"naturalness" score; lower is more natural.

Calibration: the ESM-2 family produces different absolute PPL ranges across
sizes (t12_35M ≈ 8–18, t33_650M tighter). The validation that motivated
PerplexityGuard ran on t12_35M; the production scorer here defaults to
t33_650M and supports loading model-specific calibration JSON written by
``calibrate.py``.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
FAST_MODEL = "facebook/esm2_t12_35M_UR50D"

# UniProt REST endpoint used to bootstrap auto-calibration.
_UNIPROT_FASTA_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=reviewed:true+AND+length:[80+TO+200]+AND+fragment:false"
    "&format=fasta&size=80"
)

# Fallback calibration if no JSON has been computed for the active model.
# Used purely so the demo can run before the user has run calibrate.py.
_FALLBACK_CALIBRATION: dict[str, dict[str, float]] = {
    "facebook/esm2_t12_35M_UR50D": {
        "mean": 11.54, "std": 3.65, "p95": 17.0, "p99": 18.5,
    },
    "facebook/esm2_t33_650M_UR50D": {
        # Conservative defaults; users should run calibrate.py for their corpus.
        "mean": 9.0, "std": 3.0, "p95": 14.5, "p99": 16.5,
    },
}


@dataclass
class Calibration:
    model: str
    mean: float
    std: float
    p95: float
    p99: float
    n: int = 0
    source: str = "fallback"

    @classmethod
    def for_model(cls, model_name: str, calibration_dir: Path | None = None) -> "Calibration":
        if calibration_dir is not None:
            path = calibration_dir / f"{_safe(model_name)}.json"
            if path.exists():
                data = json.loads(path.read_text())
                return cls(
                    model=model_name,
                    mean=float(data["mean"]),
                    std=float(data["std"]),
                    p95=float(data["p95"]),
                    p99=float(data["p99"]),
                    n=int(data.get("n", 0)),
                    source=str(path),
                )
        if model_name in _FALLBACK_CALIBRATION:
            d = _FALLBACK_CALIBRATION[model_name]
            return cls(model=model_name, source="fallback", **d)
        # Last-ditch: assume the t33 defaults.
        d = _FALLBACK_CALIBRATION[DEFAULT_MODEL]
        return cls(model=model_name, source="fallback-default", **d)

    def zscore(self, ppl: float) -> float:
        if self.std <= 0:
            return 0.0
        return (ppl - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "mean": self.mean,
            "std": self.std,
            "p95": self.p95,
            "p99": self.p99,
            "n": self.n,
            "source": self.source,
        }


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


@dataclass
class WindowCalibration:
    """Per-sliding-window pseudo-perplexity calibration.

    The whole-sequence calibration (``Calibration``) describes the
    distribution of full-sequence PPL values across natural proteins.
    Window-level PPLs are noisier (variance scales as 1/W vs 1/L), so a
    sliding-window screen needs its *own* calibration aggregated over all
    windows of all natural sequences in the calibration set.
    """
    model: str
    window_size: int
    mean: float
    std: float
    p95: float
    p99: float
    n_windows: int = 0
    n_sequences: int = 0
    source: str = "fallback"

    def zscore(self, window_ppl: float) -> float:
        if self.std <= 0:
            return 0.0
        return (window_ppl - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "window_size": self.window_size,
            "mean": self.mean,
            "std": self.std,
            "p95": self.p95,
            "p99": self.p99,
            "n_windows": self.n_windows,
            "n_sequences": self.n_sequences,
            "source": self.source,
        }


def windowed_z_max(
    per_position_nlls: list[float],
    window_calibration: "WindowCalibration",
) -> tuple[float, float, int]:
    """Sliding-window z-score: max and min window-z plus the offending window index.

    Walks per_position_nlls in stride-1 windows of size
    ``window_calibration.window_size``, computes each window's mean NLL →
    PPL, z-scores against the window calibration, and returns
    (max_z, min_z, argmax_z_position). Sequences shorter than the window
    fall back to the whole-sequence z (computed against this calibration's
    own mean/std, which is a slight over-estimate of variance for very
    short sequences but stays in the right direction).
    """
    L = len(per_position_nlls)
    W = window_calibration.window_size
    if L == 0:
        return 0.0, 0.0, 0
    if L < W:
        mean_nll = sum(per_position_nlls) / L
        ppl = math.exp(min(mean_nll, 50.0))
        z = window_calibration.zscore(ppl)
        return z, z, 0
    zs: list[float] = []
    for i in range(L - W + 1):
        window = per_position_nlls[i : i + W]
        mean_nll = sum(window) / W
        ppl = math.exp(min(mean_nll, 50.0))
        zs.append(window_calibration.zscore(ppl))
    arr = np.asarray(zs, dtype=np.float64)
    return float(arr.max()), float(arr.min()), int(arr.argmax())


@dataclass
class PerplexityResult:
    sequence_length: int
    pseudo_perplexity: float
    mean_nll: float
    zscore: float
    p95_threshold: float
    p99_threshold: float
    is_high: bool
    runtime_seconds: float
    # Per-residue masked NLLs in sequence order (residues only, no special tokens).
    # Used by sliding-window screens; whole-sequence screens can ignore.
    per_position_nlls: list[float] = field(default_factory=list)


class ESM2Scorer:
    """Loads an ESM-2 MLM once; scores sequences on demand."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        calibration: Calibration | None = None,
        calibration_dir: Path | None = None,
        auto_calibrate: bool = False,
        auto_calibrate_n: int = 50,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        print(f"[esm] loading {model_name} on {self.device}")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"[esm] {self.device} placement failed ({e}); falling back to CPU")
            self.device = torch.device("cpu")
            self.model.to(self.device)
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self._calibration_dir = calibration_dir
        if calibration is not None:
            self.calibration = calibration
        else:
            self.calibration = Calibration.for_model(model_name, calibration_dir)
        print(
            f"[esm] loaded in {time.time()-t0:.1f}s; "
            f"calibration source={self.calibration.source} "
            f"mean={self.calibration.mean:.2f} std={self.calibration.std:.2f}"
        )
        if (
            auto_calibrate
            and calibration is None
            and calibration_dir is not None
            and self.calibration.source.startswith("fallback")
        ):
            new_cal = self._auto_calibrate(calibration_dir, n=auto_calibrate_n)
            if new_cal is not None:
                self.calibration = new_cal
                print(
                    f"[esm] auto-calibration done: mean={new_cal.mean:.2f} "
                    f"std={new_cal.std:.2f} p95={new_cal.p95:.2f} "
                    f"(n={new_cal.n}, source={Path(new_cal.source).name})"
                )

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.no_grad()
    def score_sequence(self, sequence: str, batch_size: int = 16) -> PerplexityResult:
        seq = sequence.upper().strip()
        if not seq:
            raise ValueError("empty sequence")
        t0 = time.time()
        enc = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(self.device)
        attn = enc["attention_mask"][0].to(self.device)
        L_full = int(input_ids.shape[0])
        positions = [
            i for i in range(L_full) if input_ids[i].item() not in (self.cls_id, self.eos_id)
        ]
        if not positions:
            raise ValueError("sequence has no scoreable positions")
        nlls: list[float] = []
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start : start + batch_size]
            bsz = len(batch_pos)
            batch_inputs = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
            batch_attn = attn.unsqueeze(0).repeat(bsz, 1)
            for row, pos in enumerate(batch_pos):
                batch_inputs[row, pos] = self.mask_id
            out = self.model(input_ids=batch_inputs, attention_mask=batch_attn)
            logits = out.logits
            pos_t = torch.tensor(batch_pos, device=self.device, dtype=torch.long)
            rows = torch.arange(bsz, device=self.device, dtype=torch.long)
            slice_logits = logits[rows, pos_t]
            logp = torch.log_softmax(slice_logits.float(), dim=-1)
            true_ids = input_ids[pos_t]
            nll = -logp[rows, true_ids]
            nlls.extend(nll.detach().cpu().tolist())
        mean_nll = float(np.mean(nlls))
        ppl = float(math.exp(min(mean_nll, 50.0)))
        z = self.calibration.zscore(ppl)
        # Map nlls back to sequence order; the `positions` list was built in
        # order, so nlls[i] aligns with positions[i] which is the i-th
        # residue (excluding special tokens) of the input.
        return PerplexityResult(
            sequence_length=len(seq),
            pseudo_perplexity=ppl,
            mean_nll=mean_nll,
            zscore=z,
            p95_threshold=self.calibration.p95,
            p99_threshold=self.calibration.p99,
            is_high=ppl >= self.calibration.p95,
            runtime_seconds=time.time() - t0,
            per_position_nlls=list(nlls),
        )

    def score_batch(
        self,
        sequences: Iterable[str],
        batch_size: int = 16,
    ) -> list[PerplexityResult]:
        return [self.score_sequence(s, batch_size=batch_size) for s in sequences]

    def _auto_calibrate(self, calibration_dir: Path, n: int) -> Calibration | None:
        """Compute and persist a per-model calibration JSON from a UniProt sample.

        Bootstrapping: fetches a one-off `uniprot_sample.fasta` if missing,
        scores up to ``n`` sequences with this scorer, writes
        `<safe(model)>.json`. No-ops if calibration already exists.
        """
        calibration_dir.mkdir(parents=True, exist_ok=True)
        out_path = calibration_dir / f"{_safe(self.model_name)}.json"
        if out_path.exists():
            data = json.loads(out_path.read_text())
            return Calibration(
                model=self.model_name,
                mean=float(data["mean"]),
                std=float(data["std"]),
                p95=float(data["p95"]),
                p99=float(data["p99"]),
                n=int(data.get("n", 0)),
                source=str(out_path),
            )
        sample_path = calibration_dir / "uniprot_sample.fasta"
        if not sample_path.exists() or sample_path.stat().st_size < 1024:
            try:
                print(f"[esm] fetching UniProt sample → {sample_path.name}")
                with urllib.request.urlopen(_UNIPROT_FASTA_URL, timeout=60) as r:
                    sample_path.write_bytes(r.read())
            except Exception as e:
                print(f"[esm] auto-calibration: UniProt fetch failed ({e}); "
                      "keeping fallback calibration")
                return None
        records = _parse_fasta(sample_path)[:n]
        if not records:
            print("[esm] auto-calibration: no sequences parsed; keeping fallback")
            return None
        print(f"[esm] auto-calibrating on {len(records)} UniProt sequences "
              "(this may take a few minutes on the production model)")
        ppls: list[float] = []
        t0 = time.time()
        for i, (h, s) in enumerate(records):
            try:
                r = self.score_sequence(s)
                ppls.append(r.pseudo_perplexity)
            except Exception as e:
                print(f"[esm] calib skip {h[:30]}: {e}")
                continue
            if (i + 1) % 5 == 0 or i == len(records) - 1:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"[esm] calib {i+1}/{len(records)} latest={ppls[-1]:.2f} "
                      f"rate={rate:.2f}/s")
        if not ppls:
            return None
        arr = np.array(ppls, dtype=np.float64)
        data = {
            "model": self.model_name,
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "source_fasta": str(sample_path),
        }
        out_path.write_text(json.dumps(data, indent=2))
        return Calibration(
            model=self.model_name,
            mean=data["mean"],
            std=data["std"],
            p95=data["p95"],
            p99=data["p99"],
            n=data["n"],
            source=str(out_path),
        )


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
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
    # Filter to standard 20 AAs to keep tokenizer happy.
    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    return [(h, s.upper()) for h, s in out if s and set(s.upper()) <= AA20]


def esm_pseudo_perplexity(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sequence: str,
    batch_size: int = 16,
) -> float:
    """Functional API matching the spec: returns pseudo-perplexity for one sequence.

    Prefer ``ESM2Scorer`` for repeated calls — it caches device placement and
    calibration. This helper is retained so existing notebooks can call it
    directly.
    """
    device = next(model.parameters()).device
    seq = sequence.upper().strip()
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].to(device)
    attn = enc["attention_mask"][0].to(device)
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    positions = [
        i for i in range(int(input_ids.shape[0])) if input_ids[i].item() not in (cls_id, eos_id)
    ]
    if not positions:
        return float("nan")
    nlls: list[float] = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start : start + batch_size]
            bsz = len(batch_pos)
            batch_inputs = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
            batch_attn = attn.unsqueeze(0).repeat(bsz, 1)
            for row, pos in enumerate(batch_pos):
                batch_inputs[row, pos] = mask_id
            out = model(input_ids=batch_inputs, attention_mask=batch_attn)
            logits = out.logits
            pos_t = torch.tensor(batch_pos, device=device, dtype=torch.long)
            rows = torch.arange(bsz, device=device, dtype=torch.long)
            logp = torch.log_softmax(logits[rows, pos_t].float(), dim=-1)
            nll = -logp[rows, input_ids[pos_t]]
            nlls.extend(nll.cpu().tolist())
    return float(math.exp(min(float(np.mean(nlls)), 50.0)))
