# STATUS: canonical (end-state superseded by causal_hybrid_twostage.py) · Act 4 Coupling — fully-learned hybrid FAILS confounding (0.43)  ·  map: CAUSAL_LLM.md
"""Fully-learned hybrid: GPT-2 perception + a LEARNED reasoner -- nothing hand-coded.

The audit (AUDIT.md) showed the earlier hybrid's reasoning was a hard-coded reachability/do() formula;
only perception was learned. This is the honest end-state: a real GPT-2 reads the prose (learned
perception -> entity reps -> soft adjacency), and a LEARNED GNN reasoner (initialised from the GPT-2
entity reps, message-passing over the learned adjacency) produces the answer. The reachability + do()
+ back-door logic is NOT wired in -- the GNN must learn it. Compared, same data/seeds, against:

  * VANILLA GPT-2 (the pure path) -- a real LM answering from its LM head;
  * references (cited, not retrained here): the HAND-CODED hybrid (~1.0 in-dist / ~0.91 held-out)
    and a learned GNN reasoner GIVEN the true structure (in-dist ~1.0, size-5 ~0.8-0.9, see
    causal_core_learned_reasoning.py).

So this stacks two genuinely-learned, genuinely-hard halves (perception AND reasoning). Expect it to
be harder and more fragile than either alone -- we report what actually happens, multi-seed, on the
confounded-cause query (correlated but not causal). The aux edge loss (true adjacency, train-time
only) grounds the perception half; the answer comes only through the learned GNN.

CPU-sized (slow: several GPT-2 trainings).  Run::

    uv run --extra torch python examples/causal_hybrid_learned.py
"""

from __future__ import annotations

import os
import random
import statistics
import sys

import torch
from torch import nn
from transformers import GPT2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy  # data, GPT-2 config, VanillaLM, pack/acc/train

SEEDS = [0, 1]


def confounded(data):
    return [dict(e, is_causal=1, label=e["cause"]) for e in data if e["corr"] and not e["cause"]]


class HybridLearnedLM(nn.Module):
    """GPT-2 perception (-> adjacency) + a LEARNED GNN reasoner. No hand-coded causal formula."""

    def __init__(self, steps=5):
        super().__init__()
        cfg = hy.gpt2()
        self.gpt = GPT2Model(cfg)
        d = cfg.n_embd
        self.steps = steps
        self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        # learned reasoner (message passing over the learned soft adjacency)
        self.win = nn.Linear(d, d)
        self.wout = nn.Linear(d, d)
        self.w0 = nn.Linear(d, d)
        self.read = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, ids, attn, last, entw, xs, ys, is_causal, present):
        h = self.gpt(input_ids=ids, attention_mask=attn).last_hidden_state
        slots = []
        for s in range(hy.NE):
            m = ((ids == entw[:, s : s + 1]) & (attn == 1)).float().unsqueeze(-1)
            slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        hv = torch.stack(slots, 1)  # (B,NE,d) entity reps from GPT-2
        hi = hv.unsqueeze(2).expand(-1, hy.NE, hy.NE, -1)
        hj = hv.unsqueeze(1).expand(-1, hy.NE, hy.NE, -1)
        edge = self.edge(torch.cat([hi, hj], -1)).squeeze(-1)  # adjacency logits (perception)
        pm = present.unsqueeze(2) * present.unsqueeze(1)
        a = torch.sigmoid(edge.masked_fill(torch.eye(hy.NE).bool(), -30)) * pm
        # LEARNED reasoning: GNN message passing over the learned adjacency, init from entity reps
        g = hv * present.unsqueeze(-1)
        for _ in range(self.steps):
            m_in = torch.bmm(a.transpose(1, 2), self.win(g))
            m_out = torch.bmm(a, self.wout(g))
            g = torch.relu(self.w0(g) + m_in + m_out) * present.unsqueeze(-1)
        idx = torch.arange(ids.size(0))
        ans = self.read(torch.cat([g[idx, xs], g[idx, ys], is_causal.unsqueeze(1)], 1)).squeeze(-1)
        return ans, edge, pm


def train_learned(model, data, epochs=12, lr=5e-4, bs=64, lam=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            ids, attn, last, entw, xs, ys, isc, lab, adj, pres = hy.pack(data[i : i + bs])
            ans, edge, pm = model(ids, attn, last, entw, xs, ys, isc, pres)
            loss = (
                bce(ans, lab)
                + lam
                * (
                    nn.functional.binary_cross_entropy_with_logits(edge, adj, reduction="none") * pm
                ).sum()
                / pm.sum()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def acc_learned(model, data) -> float:
    if not data:
        return float("nan")
    ids, attn, last, entw, xs, ys, isc, lab, _adj, pres = hy.pack(data)
    ans, _, _ = model(ids, attn, last, entw, xs, ys, isc, pres)
    return float(((ans > 0) == (lab > 0.5)).float().mean())


def run_seed(seed):
    torch.manual_seed(seed)
    train = hy.build(8000, sizes=[2, 3], seed=seed)
    t3, t4 = hy.build(1500, [3], seed + 50), hy.build(1500, [4], seed + 60)
    c3, c4 = confounded(t3), confounded(t4)

    van = hy.VanillaLM()
    hy.train(van, train, hybrid=False, epochs=12)
    van.eval()

    learned = HybridLearnedLM()
    train_learned(learned, train, epochs=12)
    learned.eval()

    return {
        "vanilla_conf_s3": hy.acc(van, c3, False),
        "vanilla_conf_s4": hy.acc(van, c4, False),
        "learned_hybrid_conf_s3": acc_learned(learned, c3),
        "learned_hybrid_conf_s4": acc_learned(learned, c4),
        # also the balanced cause column, to rule out a constant-"no" artifact on confounded
        "learned_hybrid_cause_s3": acc_learned(learned, [e for e in t3 if e["is_causal"]]),
        "learned_hybrid_cause_s4": acc_learned(learned, [e for e in t4 if e["is_causal"]]),
    }


def main() -> None:
    print(f"Fully-learned hybrid (GPT-2 perception + LEARNED GNN reasoner), seeds {SEEDS}")
    rows = [run_seed(s) for s in SEEDS]
    print("\n  metric                      mean +/- std       per-seed")
    for key in rows[0]:
        vals = [r[key] for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(
            f"  {key:26s}  {statistics.mean(vals):.3f} +/- {sd:.3f}   {[round(v, 3) for v in vals]}"
        )
    print(
        "\n  references (cited): hand-coded hybrid ~1.000/~0.914 (confounded in/held-out); "
        "vanilla ~0.15; learned GNN reasoner GIVEN structure ~1.0 in-dist / ~0.8-0.9 size-5"
    )

    print(
        "\nReading: this is the honest end-state -- NOTHING hand-coded in the reasoning. If the "
        "fully-learned hybrid beats vanilla on confounded but trails the hand-coded hybrid (and is "
        "seed-fragile), then stacking learned perception + learned reasoning works partially but not "
        "as well as the wired-in algorithm -- exactly the honest boundary the audit pointed to. The "
        "cause column confirms the confounded number isn't a constant-'no' artifact."
    )


if __name__ == "__main__":
    main()
