"""
PerplexityGuard signal validation.

Tests whether ESM-2 pseudo-perplexity assigns systematically higher scores to
synthetic / AI-distorted protein sequences than to natural UniProt-curated ones.
Outputs Cohen's d, AUROC, p-values, distribution plots, and a GO/NO-GO decision.

Usage:
    uv run python validate_perplexity.py
    uv run python validate_perplexity.py --model facebook/esm2_t30_150M_UR50D --n 80
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from scipy import stats
from transformers import AutoModelForMaskedLM, AutoTokenizer

AA20 = "ACDEFGHIKLMNPQRSTVWY"


# ---- 1. UniProt fetch ----------------------------------------------------


def parse_fasta(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
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


def get_uniprot_sequences(
    n_sequences: int = 60,
    min_len: int = 80,
    max_len: int = 200,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Fetch diverse reviewed Swiss-Prot sequences within length range."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"reviewed:true AND length:[{min_len} TO {max_len}] AND fragment:false"
    params = {"query": query, "format": "fasta", "size": str(min(500, n_sequences * 10))}
    print(f"[fetch] UniProt query: {query}")
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            text = r.text
            break
        except Exception as e:
            last_err = e
            print(f"[fetch] attempt {attempt+1} failed: {e}, retrying...")
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"UniProt fetch failed: {last_err}")
    seqs = parse_fasta(text)
    cleaned = [
        (h, s.upper())
        for h, s in seqs
        if min_len <= len(s) <= max_len and set(s.upper()) <= set(AA20)
    ]
    rng = random.Random(seed)
    rng.shuffle(cleaned)
    out = cleaned[:n_sequences]
    if len(out) < n_sequences:
        print(
            f"[fetch] WARNING: only {len(out)} sequences passed filters "
            f"(requested {n_sequences})"
        )
    return out


# ---- 2. ESM-2 pseudo-perplexity -------------------------------------------


@dataclass
class ESM2:
    name: str
    device: torch.device
    tokenizer: Any
    model: Any


def load_esm2(model_name: str, prefer_mps: bool = True) -> ESM2:
    if prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[esm] loading {model_name} on {device}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    try:
        model.to(device)
    except Exception as e:
        print(f"[esm] {device} placement failed ({e}); falling back to CPU")
        device = torch.device("cpu")
        model.to(device)
    print(f"[esm] loaded in {time.time()-t0:.1f}s")
    return ESM2(name=model_name, device=device, tokenizer=tok, model=model)


@torch.no_grad()
def esm_pseudo_perplexity(esm: ESM2, sequence: str, batch_size: int = 16) -> float:
    """Compute pseudo-perplexity = exp(-1/L sum_i log P(s_i | s_\\i)) for one sequence."""
    seq = sequence.upper()
    enc = esm.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].to(esm.device)
    attn = enc["attention_mask"][0].to(esm.device)
    cls_id = esm.tokenizer.cls_token_id
    eos_id = esm.tokenizer.eos_token_id
    mask_id = esm.tokenizer.mask_token_id
    L_full = int(input_ids.shape[0])
    positions = [
        i for i in range(L_full) if input_ids[i].item() not in (cls_id, eos_id)
    ]
    if not positions:
        return float("nan")
    nlls: list[float] = []
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        bsz = len(batch_positions)
        batch_inputs = input_ids.unsqueeze(0).repeat(bsz, 1).clone()
        batch_attn = attn.unsqueeze(0).repeat(bsz, 1)
        for row, pos in enumerate(batch_positions):
            batch_inputs[row, pos] = mask_id
        out = esm.model(input_ids=batch_inputs, attention_mask=batch_attn)
        logits = out.logits  # [B, L+2, V]
        pos_t = torch.tensor(batch_positions, device=esm.device, dtype=torch.long)
        rows = torch.arange(bsz, device=esm.device, dtype=torch.long)
        slice_logits = logits[rows, pos_t]  # [B, V]
        logp = torch.log_softmax(slice_logits.float(), dim=-1)
        true_ids = input_ids[pos_t]  # [B]
        nll = -logp[rows, true_ids]  # [B]
        nlls.extend(nll.detach().cpu().tolist())
    mean_nll = float(np.mean(nlls))
    # Clamp to avoid overflow on weird inputs.
    return float(math.exp(min(mean_nll, 50.0)))


# ---- 3. Synthetic generators ---------------------------------------------


def generate_synthetic_sequences(
    natural: list[tuple[str, str]],
    seed: int = 13,
) -> dict[str, list[tuple[str, str]]]:
    """Build four flavours of length-matched synthetic sequence."""
    rng = random.Random(seed)
    pool = [aa for _, s in natural for aa in s]
    composition = list(pool)
    rng.shuffle(composition)

    shuffled: list[tuple[str, str]] = []
    mutated: list[tuple[str, str]] = []
    repeated: list[tuple[str, str]] = []
    unigram: list[tuple[str, str]] = []
    for header, s in natural:
        L = len(s)
        # 1) composition-preserving shuffle.
        sh = list(s)
        rng.shuffle(sh)
        shuffled.append((f"shuffle|{header}", "".join(sh)))
        # 2) heavy random-mutation: replace 40% of residues with random AAs.
        chars = list(s)
        n_mut = max(1, int(0.4 * L))
        for i in rng.sample(range(L), n_mut):
            chars[i] = rng.choice(AA20)
        mutated.append((f"mutate|{header}", "".join(chars)))
        # 3) repeat-motif: tile a short k-mer drawn from this sequence.
        k = rng.randint(4, 8)
        start = rng.randint(0, max(0, L - k))
        motif = s[start : start + k] or "GA"
        rep = (motif * (L // len(motif) + 1))[:L]
        repeated.append((f"repeat|{header}", rep))
        # 4) corpus-unigram sample (length matched, AA-frequency matched).
        uni = "".join(rng.choices(composition, k=L))
        unigram.append((f"unigram|{header}", uni))

    return {
        "shuffled": shuffled,
        "mutated_40pct": mutated,
        "repeat_motif": repeated,
        "unigram_corpus": unigram,
    }


# ---- 4. Statistics and plots ---------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled == 0:
        return float("nan")
    return float((b.mean() - a.mean()) / pooled)


def auroc(natural: np.ndarray, synthetic: np.ndarray) -> float:
    """Probability that a random synthetic scores higher than a random natural."""
    y = np.concatenate([np.zeros(len(natural)), np.ones(len(synthetic))])
    s = np.concatenate([natural, synthetic])
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    pos = float(y_sorted.sum())
    neg = float(len(y_sorted) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    tp = 0.0
    auc = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return float(auc / (pos * neg))


def bootstrap_d_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        ds[i] = cohens_d(a_s, b_s)
    lo, hi = np.quantile(ds, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def analyze_separation(
    natural: np.ndarray,
    bundles: dict[str, np.ndarray],
) -> dict[str, dict]:
    res: dict[str, dict] = {}
    for name, arr in bundles.items():
        d = cohens_d(natural, arr)
        d_lo, d_hi = bootstrap_d_ci(natural, arr, seed=hash(name) & 0xFFFF)
        u_stat, p_u = stats.mannwhitneyu(natural, arr, alternative="less")
        t_stat, p_t = stats.ttest_ind(natural, arr, equal_var=False)
        a = auroc(natural, arr)
        res[name] = {
            "n_natural": int(len(natural)),
            "n_synthetic": int(len(arr)),
            "mean_natural": float(natural.mean()),
            "mean_synthetic": float(arr.mean()),
            "median_natural": float(np.median(natural)),
            "median_synthetic": float(np.median(arr)),
            "cohens_d": float(d),
            "cohens_d_ci95": [d_lo, d_hi],
            "auroc": float(a),
            "mannwhitney_p_one_sided": float(p_u),
            "welch_t": float(t_stat),
            "welch_p_two_sided": float(p_t),
        }
    return res


def plot_results(
    natural: np.ndarray,
    bundles: dict[str, np.ndarray],
    out_path: Path,
    model_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    colors = ["#fdae61", "#d7191c", "#7b3294", "#1a9641"]

    ax = axes[0]
    bins = np.geomspace(
        max(min(natural.min(), *(b.min() for b in bundles.values())), 1.0),
        max(natural.max(), *(b.max() for b in bundles.values())) * 1.05,
        30,
    )
    ax.hist(natural, bins=bins, alpha=0.7, label=f"natural (n={len(natural)})", color="#2c7fb8")
    for (name, arr), c in zip(bundles.items(), colors):
        ax.hist(arr, bins=bins, alpha=0.45, label=f"{name} (n={len(arr)})", color=c)
    ax.set_xlabel("ESM-2 pseudo-perplexity")
    ax.set_ylabel("count")
    ax.set_xscale("log")
    ax.set_title("Distributions")
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [natural] + list(bundles.values())
    labels = ["natural"] + list(bundles.keys())
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("pseudo-perplexity")
    ax.set_yscale("log")
    ax.set_title("Per-condition spread")
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    ax = axes[2]
    for (name, arr), c in zip(bundles.items(), colors):
        s = np.concatenate([natural, arr])
        y = np.concatenate([np.zeros_like(natural), np.ones_like(arr)])
        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order]
        tpr = np.cumsum(y_sorted) / max(y.sum(), 1)
        fpr = np.cumsum(1 - y_sorted) / max(len(y_sorted) - y.sum(), 1)
        a = auroc(natural, arr)
        ax.plot(
            np.concatenate([[0.0], fpr]),
            np.concatenate([[0.0], tpr]),
            label=f"{name} (AUROC {a:.2f})",
            color=c,
            linewidth=1.6,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: synthetic vs natural")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[3]
    names = list(bundles.keys())
    ds = [cohens_d(natural, bundles[n]) for n in names]
    bars = ax.bar(names, ds, color=colors)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="d=0.5 GO threshold")
    ax.axhline(0.8, color="black", linestyle=":", linewidth=1, alpha=0.5, label="d=0.8 large effect")
    for bar, val in zip(bars, ds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:+.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Cohen's d (synthetic - natural)")
    ax.set_title("Effect sizes")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.legend(fontsize=8)

    fig.suptitle(f"ESM-2 pseudo-perplexity signal validation — {model_name}", fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---- 5. Decision logic ---------------------------------------------------


def make_decision(stats_by_condition: dict[str, dict]) -> tuple[str, str]:
    """GO if at least the realistic AI-failure mode separates with d>0.5 and p<0.05."""
    target = "mutated_40pct"
    target_stats = stats_by_condition.get(target, {})
    target_d = target_stats.get("cohens_d", 0.0)
    target_p = target_stats.get("mannwhitney_p_one_sided", 1.0)
    target_auroc = target_stats.get("auroc", 0.5)
    passes = {
        name: (s["cohens_d"] > 0.5 and s["mannwhitney_p_one_sided"] < 0.05)
        for name, s in stats_by_condition.items()
    }
    if target_d > 0.5 and target_p < 0.05:
        return (
            "GO",
            f"Realistic AI failure mode '{target}' separates: d={target_d:.2f} "
            f"(95% CI {target_stats['cohens_d_ci95'][0]:.2f}–{target_stats['cohens_d_ci95'][1]:.2f}), "
            f"AUROC={target_auroc:.2f}, p={target_p:.1e}.",
        )
    winners = [n for n, ok in passes.items() if ok]
    if winners:
        return (
            "WEAK_GO",
            f"Realistic '{target}' did not pass (d={target_d:.2f}, p={target_p:.1e}); "
            f"only crude conditions separate: {winners}. PerplexityGuard may need a "
            f"stronger backbone or auxiliary signals.",
        )
    return (
        "NO_GO",
        "No synthetic condition shows reliable separation. Pseudo-perplexity alone is insufficient.",
    )


# ---- 6. Driver ----------------------------------------------------------


def score_group(
    esm: ESM2,
    group: list[tuple[str, str]],
    label: str,
    batch_size: int,
) -> np.ndarray:
    out = np.zeros(len(group), dtype=np.float64)
    t0 = time.time()
    for i, (_, s) in enumerate(group):
        out[i] = esm_pseudo_perplexity(esm, s, batch_size=batch_size)
        if (i + 1) % 5 == 0 or i == len(group) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(group) - i - 1) / max(rate, 1e-6)
            print(
                f"[score:{label}] {i+1}/{len(group)} "
                f"ppl={out[i]:7.2f} rate={rate:.2f}/s eta={eta:.0f}s"
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--min-len", type=int, default=80)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--model", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU (skip MPS).")
    args = p.parse_args()

    args.out.mkdir(exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[cfg] model={args.model} n={args.n} "
        f"len=[{args.min_len},{args.max_len}] batch={args.batch_size}"
    )

    natural = get_uniprot_sequences(args.n, args.min_len, args.max_len, args.seed)
    print(f"[fetch] obtained {len(natural)} natural sequences "
          f"(median len {int(np.median([len(s) for _, s in natural]))})")
    syn = generate_synthetic_sequences(natural, seed=args.seed + 1)
    for name, lst in syn.items():
        print(f"[gen] {name}: {len(lst)} sequences")

    esm = load_esm2(args.model, prefer_mps=not args.cpu)

    natural_scores = score_group(esm, natural, "natural", args.batch_size)
    bundles = {
        name: score_group(esm, lst, name, args.batch_size) for name, lst in syn.items()
    }

    stats_by_condition = analyze_separation(natural_scores, bundles)
    plot_path = args.out / "perplexity_validation.png"
    plot_results(natural_scores, bundles, plot_path, args.model)

    decision, reason = make_decision(stats_by_condition)

    summary = {
        "model": args.model,
        "device": str(esm.device),
        "n_sequences": len(natural),
        "length_range": [args.min_len, args.max_len],
        "natural_summary": {
            "mean": float(natural_scores.mean()),
            "median": float(np.median(natural_scores)),
            "std": float(natural_scores.std(ddof=1)),
            "min": float(natural_scores.min()),
            "max": float(natural_scores.max()),
        },
        "conditions": stats_by_condition,
        "decision": decision,
        "reason": reason,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        args.out / "scores.npz",
        natural=natural_scores,
        **{k: v for k, v in bundles.items()},
    )

    print()
    print("=" * 64)
    print(f"DECISION: {decision}")
    print(f"REASON:   {reason}")
    print("=" * 64)
    print()
    print(
        f"Natural pseudo-perplexity: mean={natural_scores.mean():.2f} "
        f"median={np.median(natural_scores):.2f} "
        f"std={natural_scores.std(ddof=1):.2f}"
    )
    print()
    print("Per-condition statistics:")
    print(
        f"  {'condition':>16}  {'d':>7}  {'CI95':>17}  {'AUROC':>5}  "
        f"{'mean_nat':>8}  {'mean_syn':>8}  {'p':>9}"
    )
    for name, s in stats_by_condition.items():
        ci = s["cohens_d_ci95"]
        print(
            f"  {name:>16}  {s['cohens_d']:+7.2f}  "
            f"[{ci[0]:+5.2f},{ci[1]:+5.2f}]  {s['auroc']:.3f}  "
            f"{s['mean_natural']:8.2f}  {s['mean_synthetic']:8.2f}  "
            f"{s['mannwhitney_p_one_sided']:.1e}"
        )
    print()
    print(f"Wrote {args.out / 'summary.json'}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {args.out / 'scores.npz'}")
    return 0 if decision in ("GO", "WEAK_GO") else 1


if __name__ == "__main__":
    sys.exit(main())
