# STATUS: canonical · Act 4 Coupling — the FIX: decoupled two-stage -> 1.000 confounded in-dist  ·  map: CAUSAL_LLM.md
"""Two-stage fix: decoupled training closes the confounding gap the joint hybrid couldn't.

causal_perception_bottleneck.py localized the fully-learned hybrid's confounding failure (~0.43) to
END-TO-END JOINT TRAINING -- not perception (edge F1 ~0.86; the perceived graph drives the exact
algorithm to ~1.0) and not the reasoner's capacity (a GNN trained on clean structure scores ~1.0 on
the perceived graph). This script applies the implied fix and measures it end-to-end from prose:

  Stage A (perception):  train GPT-2 + an edge MLP on the edge-recovery loss ONLY -> soft adjacency.
  Stage B (reasoning):   train a structure-only GNN reasoner on CLEAN (teacher-forced) structure
                         + the answer loss -- the regime where the ablation hit ~1.0.
  Inference:             prose -> frozen Stage-A perception -> thresholded adjacency -> Stage-B
                         reasoner -> answer.  Nothing hand-coded; the two halves are trained apart.

Teacher-forcing the structure for Stage B is a legitimate training device (privileged info at train
time, like scheduled sampling), not a hand-coded reasoner: at inference the reasoner sees only the
*perceived* graph. Baseline for contrast: the naive JOINT hybrid (causal_hybrid_learned.py), same
data/seeds (~0.43 on confounded).

If the two-stage system reaches ~1.0 on confounded-cause from prose where the joint hybrid sits at
~0.43, the branch's "fully-learned systems can't go beyond correlation" end-state was a training
artifact, not a capability limit.

CPU-sized (two GPT-2 trainings per seed).  Run::

    SEEDS=0 uv run --extra torch python examples/causal_hybrid_twostage.py   # fast smoke
    uv run --extra torch python examples/causal_hybrid_twostage.py           # default 2 seeds
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
import causal_hybrid_lm as hy
from causal_hybrid_learned import HybridLearnedLM, acc_learned, confounded, train_learned

NE = hy.NE
EYE = torch.eye(NE).bool()
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]


# --------------------------------------------------------------------------- Stage A: perception
class PerceptionNet(nn.Module):
    """GPT-2 reads prose -> soft adjacency. Trained on the edge-recovery loss only (no answer)."""

    def __init__(self):
        super().__init__()
        cfg = hy.gpt2()
        self.gpt = GPT2Model(cfg)
        d = cfg.n_embd
        self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, ids, attn, entw, present):
        h = self.gpt(input_ids=ids, attention_mask=attn).last_hidden_state
        slots = []
        for s in range(NE):
            m = ((ids == entw[:, s : s + 1]) & (attn == 1)).float().unsqueeze(-1)
            slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        hv = torch.stack(slots, 1)
        hi = hv.unsqueeze(2).expand(-1, NE, NE, -1)
        hj = hv.unsqueeze(1).expand(-1, NE, NE, -1)
        edge = self.edge(torch.cat([hi, hj], -1)).squeeze(-1)
        pm = present.unsqueeze(2) * present.unsqueeze(1)
        return edge, pm

    @torch.no_grad()
    def adjacency(self, ids, attn, entw, present, threshold=True):
        edge, pm = self.forward(ids, attn, entw, present)
        a = torch.sigmoid(edge.masked_fill(EYE, -30)) * pm
        return (a > 0.5).float() * pm if threshold else a


def train_perception(model, data, epochs=12, lr=5e-4, bs=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            ids, attn, _, entw, _, _, _, _, adj, pres = hy.pack(data[i : i + bs])
            edge, pm = model(ids, attn, entw, pres)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(edge, adj, reduction="none") * pm
            ).sum() / pm.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()


# --------------------------------------------------------------------------- Stage B: reasoning
class GNNReasoner(nn.Module):
    """Structure-only learned message passing over a given adjacency -> query readout."""

    def __init__(self, d=48, steps=5):
        super().__init__()
        self.steps = steps
        self.init = nn.Parameter(torch.randn(d) * 0.1)
        self.win = nn.Linear(d, d)
        self.wout = nn.Linear(d, d)
        self.w0 = nn.Linear(d, d)
        self.read = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, a, xs, ys, isc, present):
        b = a.size(0)
        h = self.init.view(1, 1, -1).expand(b, NE, -1) * present.unsqueeze(-1)
        for _ in range(self.steps):
            m_in = torch.bmm(a.transpose(1, 2), self.win(h))
            m_out = torch.bmm(a, self.wout(h))
            h = torch.relu(self.w0(h) + m_in + m_out) * present.unsqueeze(-1)
        idx = torch.arange(b)
        return self.read(torch.cat([h[idx, xs], h[idx, ys], isc.unsqueeze(1)], 1)).squeeze(-1)


def train_reasoner(model, data, epochs=25, lr=2e-3, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            _, _, _, _, xs, ys, isc, lab, adj, pres = hy.pack(data[i : i + bs])
            pm = pres.unsqueeze(2) * pres.unsqueeze(1)
            loss = bce(model(adj * pm, xs, ys, isc, pres), lab)  # CLEAN (teacher-forced) structure
            opt.zero_grad()
            loss.backward()
            opt.step()


# --------------------------------------------------------------------------- compose + evaluate
@torch.no_grad()
def twostage_acc(perception, reasoner, data, threshold=True) -> float:
    if not data:
        return float("nan")
    ids, attn, _, entw, xs, ys, isc, lab, _, pres = hy.pack(data)
    a = perception.adjacency(ids, attn, entw, pres, threshold=threshold)
    out = reasoner(a, xs, ys, isc, pres)
    return float(((out > 0) == (lab > 0.5)).float().mean())


@torch.no_grad()
def edge_f1(perception, data) -> float:
    ids, attn, _, entw, _, _, _, _, adj, pres = hy.pack(data)
    pm = pres.unsqueeze(2) * pres.unsqueeze(1)
    pred = perception.adjacency(ids, attn, entw, pres, threshold=True) > 0.5
    true = adj > 0.5
    mask = pm.bool() & ~EYE
    tp = float((pred & true & mask).sum())
    fp = float((pred & ~true & mask).sum())
    fn = float((~pred & true & mask).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return 2 * prec * rec / (prec + rec) if prec + rec else float("nan")


def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    train = hy.build(8000, sizes=[2, 3], seed=seed)
    t3 = hy.build(1500, [3], seed + 50)
    t4 = hy.build(1500, [4], seed + 60)
    c3, c4 = confounded(t3), confounded(t4)
    cause3 = [e for e in t3 if e["is_causal"]]
    cause4 = [e for e in t4 if e["is_causal"]]

    # baseline -- the naive joint hybrid (~0.43 on confounded)
    joint = HybridLearnedLM()
    train_learned(joint, train, epochs=12)
    joint.eval()

    # the fix -- Stage A perception (edge loss) + Stage B reasoner (clean structure), composed
    perc = PerceptionNet()
    train_perception(perc, train, epochs=12)
    perc.eval()
    reas = GNNReasoner()
    train_reasoner(reas, train, epochs=25)
    reas.eval()

    return {
        "edge_f1": edge_f1(perc, c3 + c4),
        "joint_conf_s3": acc_learned(joint, c3),
        "joint_conf_s4": acc_learned(joint, c4),
        "two_conf_s3": twostage_acc(perc, reas, c3),
        "two_conf_s4": twostage_acc(perc, reas, c4),
        "two_conf_soft_s3": twostage_acc(perc, reas, c3, threshold=False),
        "two_cause_s3": twostage_acc(perc, reas, cause3),
        "two_cause_s4": twostage_acc(perc, reas, cause4),
    }


def main() -> None:
    print(f"Two-stage (decoupled) vs joint hybrid, end-to-end from prose, seeds {SEEDS}")
    rows = [run_seed(s) for s in SEEDS]

    def agg(key):
        vals = [r[key] for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), sd, [round(v, 3) for v in vals]

    m, sd, per = agg("edge_f1")
    print(f"\n  Stage-A perception edge F1: {m:.3f} +/- {sd:.3f}   {per}")

    print("\n  Confounded-cause accuracy (correlated but NOT causal; perfect = 1.0), from prose:")
    print("    system                 size 3 (in-dist)     size 4 (held-out)")
    for sys_key, label in (("joint", "JOINT hybrid (base)"), ("two", "TWO-STAGE (fix)")):
        m3, sd3, _ = agg(f"{sys_key}_conf_s3")
        m4, sd4, _ = agg(f"{sys_key}_conf_s4")
        print(f"    {label:21s}  {m3:.3f} +/- {sd3:.3f}      {m4:.3f} +/- {sd4:.3f}")

    cm3, csd3, _ = agg("two_cause_s3")
    cm4, csd4, _ = agg("two_cause_s4")
    sm3, ssd3, _ = agg("two_conf_soft_s3")
    print(
        f"\n  Two-stage `cause` query (balanced):  s3 {cm3:.3f}+/-{csd3:.3f}  "
        f"s4 {cm4:.3f}+/-{csd4:.3f}"
        f"\n  Two-stage confounded on SOFT (un-thresholded) graph, s3: {sm3:.3f}+/-{ssd3:.3f}"
    )

    j3, _, _ = agg("joint_conf_s3")
    t3v, _, _ = agg("two_conf_s3")
    print(
        f"\n  Reading: nothing is hand-coded; the two systems differ ONLY in training schedule. "
        f"If two-stage ({t3v:.2f} confounded in-dist) clears the joint hybrid ({j3:.2f}), a "
        "fully-learned causal LM DOES go beyond correlation on confounded cases -- the joint "
        "hybrid's failure was an optimization artifact, fixed by decoupling perception from "
        "reasoning. The balanced `cause` column rules out a constant-'no'."
    )


if __name__ == "__main__":
    main()
