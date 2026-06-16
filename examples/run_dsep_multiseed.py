"""Multi-seed driver for the stratified d-separation experiment — mean ± std per difficulty stratum.

The single-seed follow-up showed the causal graph transformer learns colliders and multi-hop paths
under a stratified curriculum, but with notable seed variance on the traded-off strata. This driver
makes that rigorous: it trains K seeds (stratified curriculum) and reports the difficulty-stratified
diagnostic (natural distribution, unseen sizes) aggregated as mean ± std across seeds.

PREPARED, NOT YET RUN (it trains K full models). Launch with::

    uv run --extra torch python examples/run_dsep_multiseed.py --seeds 5 --layers 6 --epochs 50
    uv run --extra torch python examples/run_dsep_multiseed.py --smoke   # 2 tiny seeds, seconds

It reuses the trained-and-audited training/diagnostic code, so there is nothing new to validate —
only the aggregation across seeds.
"""

from __future__ import annotations

import argparse
import random
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

import torch
from causal_graph_transformer import (
    STRATA,
    CausalGraphTransformer,
    Config,
    GraphDataset,
    collate,
)
from causal_graph_transformer import (
    train as cgt_train,
)
from causal_graph_transformer_diagnose import generate
from torch.utils.data import DataLoader


def stratified_diagnostic(cfg: Config, ckpt: Path, sizes: list[int], n: int, seed: int
                          ) -> dict[str, float]:
    """Per-stratum accuracy of a checkpoint on a natural (unbalanced) test set."""
    device = cfg.resolved_device()
    model = CausalGraphTransformer(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    rng = random.Random(seed)
    pool = list(string.ascii_uppercase[: cfg.max_nodes])
    examples, strata = [], []
    for _ in range(n):
        ex, st = generate(cfg, rng.choice(sizes), rng, pool)
        examples.append(ex)
        strata.append(st)
    counts: Counter[str] = Counter(strata)
    correct: dict[str, int] = defaultdict(int)
    preds: list[int] = []
    with torch.no_grad():
        for batch in DataLoader(GraphDataset(examples), batch_size=512, collate_fn=collate):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(model(batch).argmax(-1).tolist())
    for ex, st, pred in zip(examples, strata, preds, strict=True):
        correct[st] += int(pred == int(ex.label))
    return {st: (correct[st] / counts[st] if counts[st] else float("nan")) for st in STRATA}


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed stratified d-separation experiment.")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--diag-sizes", type=int, nargs="+", default=[6, 7])
    p.add_argument("--diag-n", type=int, default=8000)
    p.add_argument("--out", default="runs/dsep_multiseed")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.seeds, a.layers, a.d_model, a.n_train, a.epochs, a.diag_n = 2, 2, 64, 600, 3, 500

    per_stratum: dict[str, list[float]] = defaultdict(list)
    for s in range(a.seeds):
        out = Path(a.out) / f"seed{s}"
        cfg = Config(
            curriculum="stratified", n_train=a.n_train, epochs=a.epochs, d_model=a.d_model,
            n_layers=a.layers, seed=s, eval_every=max(1, a.epochs), out=str(out),
        )
        print(f"\n===== seed {s}/{a.seeds - 1} =====")
        cgt_train(cfg)
        accs = stratified_diagnostic(cfg, out / "best.pt", a.diag_sizes, a.diag_n, seed=10_000 + s)
        print(f"  seed {s}: " + "  ".join(f"{st}={accs[st]:.3f}" for st in STRATA))
        for st in STRATA:
            per_stratum[st].append(accs[st])

    print(f"\n=== stratified diagnostic over {a.seeds} seeds (natural dist) ===")
    print(f"{'stratum':<16}{'mean':>8}{'std':>8}")
    for st in STRATA:
        vals = per_stratum[st]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{st:<16}{mean:>8.3f}{std:>8.3f}")


if __name__ == "__main__":
    main()
