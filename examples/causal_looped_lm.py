# STATUS: research · R3 of the ladder — LATENT iteration: a weight-tied looped block on the same STRUCTONLY substrate · design: docs/causal_llm/LADDER.md · map: CAUSAL_LLM.md
"""R3 — does causal computation fit in ITERATED weights, with no token tape?

R2 measured the token tape: writing the closure as tokens lifts in-dist `cause` 0.731 -> 0.926 but
leaves s4 at ~0.7 and the confounded trap at 0.42 (the shortcut corrupts the written computation).
R3 asks the twin question with the OTHER tape: a single pre-LN GPT-2-style block, weight-tied and
applied T times (latent recurrence), one-shot answer, NO trace. Iteration replaces depth; state
lives in hidden activations instead of tokens (the CoT-vs-looped separation of arXiv:2605.30757,
instantiated on causal semantics).

Substrate, data, eval sets, and sequences are IDENTICAL to `causal_pure_twostage.py`'s STRUCTONLY
arm (imported from it): ``query <g> TRUE graph </g> <ans>``, loss on the answer token only. The
model is ~160K params — 5x FEWER than the 809K 4L baseline (if iteration wins at 1/5 the
parameters, capacity is not the story; the R4 8L/192d control already showed more params alone do
nothing).

Falsifiable checks this script produces:
  * a test-time T-sweep (T_EVAL) on s3/s4 — looped models can trade compute for depth at eval;
    if competence tracks iteration count (and s4 needs more T than s3, graph diameter being
    larger), the latent tape is doing real iterative work;
  * the confounded trap read beside balanced `cause`, as always (constant-"no" scores 1.000 there).

Run::

    SEEDS=0 FAST=1 uv run --extra torch python examples/causal_looped_lm.py   # smoke
    SEEDS=0,1,2   uv run --extra torch python examples/causal_looped_lm.py

Knobs: SEEDS, TTRAIN (loop count in training, default 8), TEVAL (comma list, default
2,4,8,12,16,24), EMBD/HEADS, FAST=1.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy  # data substrate (audited labels)
import causal_pure_twostage as pt  # vocab, sequence builders, eval-set selectors

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
FAST = os.environ.get("FAST") == "1"
T_TRAIN = int(os.environ.get("TTRAIN", "8"))
# Randomize the train-time loop count per batch (uniform 4..12). The literature's standard recipe
# for test-time T-scaling: a model trained at ONE fixed T converges to a fixpoint and the eval
# T-sweep is flat (observed in the smoke); jitter forces T-robust iterative computation.
T_JIT = os.environ.get("TJIT") == "1"
T_EVAL = [int(x) for x in os.environ.get("TEVAL", "2,4,8,12,16,24").split(",")]
EMBD = int(os.environ.get("EMBD", "128"))
HEADS = int(os.environ.get("HEADS", "4"))
NPOS = 128  # structonly sequences are short (<=60); no trace tokens in this arm


class Block(nn.Module):
    """One pre-LN GPT-2-style block (causal self-attention + 4x GELU MLP). This is the ENTIRE
    reasoning stack — it is applied T times with tied weights."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        q = self.ln1(x)
        a, _ = self.attn(q, q, q, attn_mask=causal, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class LoopedLM(nn.Module):
    """Embeddings + ONE weight-tied block iterated T times + LN + (tied) LM head.

    Padding note: sequences are packed left-aligned (real tokens first), so the causal mask alone
    keeps real positions from ever attending a pad; pad-position outputs are ignored by the loss.
    No step/iteration encoding — pure weight tying."""

    def __init__(self, vocab: int, d: int, h: int):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(NPOS, d)
        self.block = Block(d, h)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight  # GPT-2-style tying
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, ids: torch.Tensor, t: int) -> torch.Tensor:
        _n, ln = ids.shape
        pos = torch.arange(ln, device=ids.device)
        x = self.wte(ids) + self.wpe(pos)[None]
        causal = torch.triu(torch.ones(ln, ln, dtype=torch.bool, device=ids.device), 1)
        for _ in range(t):
            x = self.block(x, causal)
        return self.head(self.ln_f(x))


def train(model: LoopedLM, corpus, epochs: int, lr: float = 5e-4, bs: int = 32) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    tag = "looped T~U(4,12)" if T_JIT else f"looped T={T_TRAIN}"
    for ep in range(epochs):
        rng.shuffle(corpus)
        tot = nb = 0.0
        for i in range(0, len(corpus), bs):
            ids, _attn, lab = pt.pack(corpus[i : i + bs])
            t = rng.randint(4, 12) if T_JIT else T_TRAIN
            logits = model(ids, t)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=pt.IGN
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"    [{tag}] epoch {ep + 1}/{epochs}  loss {tot / nb:.4f}", flush=True)


@torch.no_grad()
def acc_at(model: LoopedLM, data, t: int) -> float:
    """STRUCTONLY read at loop count t: answer margin at the last real position."""
    if not data:
        return float("nan")
    prompts = [pt.seq_structonly(e)[0][:-1] for e in data]  # drop the gold answer token
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pt.PAD, dtype=torch.long)
    last = torch.tensor([len(p) - 1 for p in prompts])
    for j, p in enumerate(prompts):
        ids[j, : len(p)] = torch.tensor(p)
    ok = 0
    n = 0
    bs = 256
    for i in range(0, len(prompts), bs):
        logits = model(ids[i : i + bs], t)
        row = logits[torch.arange(len(logits)), last[i : i + bs]]
        m = row[:, pt.YES] - row[:, pt.NO]
        lab = torch.tensor([float(e["label"]) for e in data[i : i + bs]])
        ok += int(((m > 0) == (lab > 0.5)).sum())
        n += len(logits)
    return ok / n


def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    n = 2000 if FAST else 8000
    epochs = 4 if FAST else 12
    nt = 400 if FAST else 1500

    train_data = hy.build(n, sizes=[2, 3], seed=seed)
    t3 = hy.build(nt, [3], seed + 50)
    t4 = hy.build(nt, [4], seed + 60)
    c3, c4 = pt.confounded(t3), pt.confounded(t4)
    cause3 = [e for e in t3 if e["is_causal"]]
    cause4 = [e for e in t4 if e["is_causal"]]

    model = LoopedLM(len(pt.WORDS), EMBD, HEADS)
    nparam = sum(p.numel() for p in model.parameters())
    corpus = [pt.seq_structonly(e) for e in train_data]
    print(f"\n  seed {seed}: LoopedLM {nparam:,} params, train T={T_TRAIN}", flush=True)
    train(model, corpus, epochs)
    model.eval()

    out: dict[str, float] = {"params": float(nparam)}
    for t in T_EVAL:
        out[f"T{t}_cause_s3"] = acc_at(model, cause3, t)
        out[f"T{t}_cause_s4"] = acc_at(model, cause4, t)
        out[f"T{t}_conf_s3"] = acc_at(model, c3, t)
        out[f"T{t}_conf_s4"] = acc_at(model, c4, t)
    return out


def main() -> None:
    print("R3 — latent iteration (weight-tied looped block), same STRUCTONLY substrate")
    print(f"  seeds {SEEDS}{'  [FAST smoke]' if FAST else ''}; train T={T_TRAIN}; eval T {T_EVAL}")
    rows = [run_seed(s) for s in SEEDS]

    def agg(key):
        vals = [r[key] for r in rows if key in r and not math.isnan(r[key])]
        if not vals:
            return float("nan"), 0.0
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), sd

    print("\n" + "=" * 78)
    print("  confounded = all-negative set (constant-'no' scores 1.000); read beside `cause`.")
    print(f"  {'eval T':<8}{'cause s3':>16}{'cause s4':>16}{'conf s3':>16}{'conf s4':>16}")
    for t in T_EVAL:
        cells = []
        for key in ("cause_s3", "cause_s4", "conf_s3", "conf_s4"):
            mean, sd = agg(f"T{t}_{key}")
            cells.append(f"{mean:.3f}+/-{sd:.3f}")
        star = "  <- train T" if t == T_TRAIN else ""
        print(f"  {t:<8}" + "".join(f"{c:>16}" for c in cells) + star)
    print("\n  references (same substrate, 3 seeds): R4 one-shot 4L = 0.731/0.581 cause,")
    print("  0.190/0.160 conf · R2 token tape = 0.926/0.681 cause, 0.421/0.190 conf ·")
    print("  GNN = 1.000/0.952 cause, 1.000/0.893 conf.")
    print(
        "\n  Reading: the block is applied T times with TIED weights -- iteration replaces"
        "\n  depth, state lives in activations. If accuracy climbs with eval T (and s4 wants"
        "\n  more T than s3), latent iteration does real work; where it plateaus vs R2 says"
        "\n  which tape -- tokens or activations -- carries causal computation further."
    )


if __name__ == "__main__":
    main()
