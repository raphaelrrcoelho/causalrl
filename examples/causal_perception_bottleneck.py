# STATUS: canonical · Act 4 Coupling — diagnosis: the 0.43 failure is joint TRAINING, not perception  ·  map: CAUSAL_LLM.md
"""Ablation: is PERCEPTION (not reasoning) the bottleneck on the confounding trap?

Two earlier results bracket the question:
  * causal_core_learned_reasoning.py -- structure GIVEN, reasoning LEARNED: a GNN gets confounded
    ~1.0 in-dist (0.92 @ size 4). So learned causal reasoning, incl. back-door, is NOT the blocker.
  * causal_hybrid_learned.py -- structure LEARNED (GPT-2 perception) + reasoning LEARNED: confounded
    collapses to ~0.43 (below chance), while the easy `cause` query stays ~1.0.

The only thing that changed between them is perception quality. This script isolates it. We take the
fully-learned hybrid's *perceived* soft adjacency, measure how good it is (edge recovery), then feed
TRUE vs PERCEIVED structure into two reasoners whose causal logic is known-good:

  * HARDWIRED  the exact reachability + do()/back-door formula (NOT trained). The clean causal test:
               run the correct algorithm on the perceived graph vs the true graph; any drop is due
               purely to edge errors, with no training/representation confound.
  * GNN        a learned reasoner trained on TRUE structure (perception removed), tested on both.

The confounded-cause query is the test (correlated but NOT causal; a perfect reasoner says "no" ->
1.0; saying "causes" on a merely-correlated pair drives it down).

FINDING (refutes the title's hypothesis): the perceived graph (edge F1 ~0.86), fed to the exact
algorithm OR to a GNN trained on clean structure, scores ~1.0 -- soft or thresholded. The end-to-end
hybrid, using that SAME perception, fails at ~0.43; thresholding its own reasoner does NOT help. So
perception is NOT the bottleneck: the failure is end-to-end joint TRAINING of the reasoner, and a
decoupled / two-stage recipe (train the reasoner on clean structure, then plug perception in)
already reaches ~1.0 on the perceived graph.

CPU-sized (trains one GPT-2 per seed + a tiny GNN).  Run::

    SEEDS=0 uv run --extra torch python examples/causal_perception_bottleneck.py   # fast smoke
    uv run --extra torch python examples/causal_perception_bottleneck.py           # default 2 seeds
"""

from __future__ import annotations

import os
import random
import statistics
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy  # data, GPT-2 config, pack/acc/train
from causal_hybrid_learned import HybridLearnedLM, acc_learned, confounded, train_learned

NE = hy.NE
EYE = torch.eye(NE).bool()
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]


# ---------------------------------------------------------------------------
# Perception: pull the fully-learned hybrid's soft adjacency out, exactly as it
# builds it internally (sigmoid of edge logits, diagonal suppressed, present-masked).
# ---------------------------------------------------------------------------
@torch.no_grad()
def perceive(hybrid: HybridLearnedLM, data: list[dict]):
    ids, attn, last, entw, xs, ys, isc, lab, adj, pres = hy.pack(data)
    _ans, edge, pm = hybrid(ids, attn, last, entw, xs, ys, isc, pres)
    a_hat = torch.sigmoid(edge.masked_fill(EYE, -30)) * pm
    return a_hat, adj * pm, pm, xs, ys, isc, lab, pres


def edge_recovery(a_hat, adj_true, pm) -> dict:
    """Quality of the perceived adjacency vs the truth, over present off-diagonal pairs."""
    mask = pm.bool() & ~EYE
    pred = (a_hat > 0.5) & mask
    true = (adj_true > 0.5) & mask
    tp = float((pred & true).sum())
    fp = float((pred & ~true & mask).sum())
    fn = float((~pred & true & mask).sum())
    correct = float(((a_hat > 0.5) == (adj_true > 0.5))[mask].sum())
    total = float(mask.sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"edge_acc": correct / total, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Reasoners that operate on a GIVEN adjacency (perception removed).
# ---------------------------------------------------------------------------
class Hardwired:
    """The exact hand-coded reachability + back-door formula. NOT learned (it is the algorithm)."""

    def __init__(self, steps: int = 5):
        self.steps = steps

    @torch.no_grad()
    def predict(self, a, xs, ys, isc, present):
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(a.size(0))
        fwd, bwd = r[idx, xs, ys], r[idx, ys, xs]
        rzx, rzy = r[idx, :, xs], r[idx, :, ys]
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * present).max(1).values
        score = isc * fwd + (1 - isc) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        return score > 0.5


class GNNReasoner(nn.Module):
    """Learned K-step message passing over a given adjacency -> query readout (no perception)."""

    def __init__(self, d: int = 48, steps: int = 5):
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

    @torch.no_grad()
    def predict(self, a, xs, ys, isc, present):
        return self.forward(a, xs, ys, isc, present) > 0


def train_gnn(model, data, epochs=25, lr=2e-3, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            _, _, _, _, xs, ys, isc, lab, adj, pres = hy.pack(data[i : i + bs])
            pm = pres.unsqueeze(2) * pres.unsqueeze(1)
            loss = bce(model(adj * pm, xs, ys, isc, pres), lab)  # TRUE structure
            opt.zero_grad()
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Evaluate a reasoner's accuracy on a subset, feeding TRUE vs PERCEIVED structure.
# `hard` thresholds the perceived adjacency to {0,1} so the structure-trained GNN
# sees the same domain it trained on; HARDWIRED is also shown on the soft graph.
# ---------------------------------------------------------------------------
@torch.no_grad()
def hybrid_own_reasoner(hybrid, subset, threshold: bool) -> float:
    """Run the hybrid's OWN trained GNN reasoner, but feed it the perceived graph soft (=end-to-end,
    a sanity check vs the anchor) or thresholded. Disambiguates 'soft input confuses it' (threshold
    -> jumps to ~1.0, fix is trivial) vs 'the jointly-trained weights are bad' (stays ~0.43)."""
    if not subset:
        return float("nan")
    ids, attn, _, entw, xs, ys, isc, lab, _, pres = hy.pack(subset)
    h = hybrid.gpt(input_ids=ids, attention_mask=attn).last_hidden_state
    slots = []
    for s in range(NE):
        m = ((ids == entw[:, s : s + 1]) & (attn == 1)).float().unsqueeze(-1)
        slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
    hv = torch.stack(slots, 1)
    hi = hv.unsqueeze(2).expand(-1, NE, NE, -1)
    hj = hv.unsqueeze(1).expand(-1, NE, NE, -1)
    edge = hybrid.edge(torch.cat([hi, hj], -1)).squeeze(-1)
    pm = pres.unsqueeze(2) * pres.unsqueeze(1)
    a = torch.sigmoid(edge.masked_fill(EYE, -30)) * pm
    if threshold:
        a = (a > 0.5).float() * pm
    g = hv * pres.unsqueeze(-1)
    for _ in range(hybrid.steps):
        m_in = torch.bmm(a.transpose(1, 2), hybrid.win(g))
        m_out = torch.bmm(a, hybrid.wout(g))
        g = torch.relu(hybrid.w0(g) + m_in + m_out) * pres.unsqueeze(-1)
    idx = torch.arange(ids.size(0))
    ans = hybrid.read(torch.cat([g[idx, xs], g[idx, ys], isc.unsqueeze(1)], 1)).squeeze(-1)
    return float(((ans > 0) == (lab > 0.5)).float().mean())


@torch.no_grad()
def reasoner_acc(reasoner, hybrid, subset, source, hard=True) -> float:
    if not subset:
        return float("nan")
    ids, attn, last, entw, xs, ys, isc, lab, adj, pres = hy.pack(subset)
    pm = pres.unsqueeze(2) * pres.unsqueeze(1)
    if source == "true":
        a = adj * pm
    else:
        _ans, edge, _pm = hybrid(ids, attn, last, entw, xs, ys, isc, pres)
        a_hat = torch.sigmoid(edge.masked_fill(EYE, -30)) * pm
        a = (a_hat > 0.5).float() * pm if hard else a_hat
    pred = reasoner.predict(a, xs, ys, isc, pres)
    return float((pred == (lab > 0.5)).float().mean())


def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    train = hy.build(8000, sizes=[2, 3], seed=seed)
    t3 = hy.build(1500, [3], seed + 50)
    t4 = hy.build(1500, [4], seed + 60)
    c3, c4 = confounded(t3), confounded(t4)
    cause3 = [e for e in t3 if e["is_causal"]]
    cause4 = [e for e in t4 if e["is_causal"]]

    # 1) the fully-learned hybrid (perception + learned GNN reasoner) -- the 0.43 anchor
    hybrid = HybridLearnedLM()
    train_learned(hybrid, train, epochs=12)
    hybrid.eval()

    # 2) perception quality (the missing number)
    a_hat, adj_true, pm, *_ = perceive(hybrid, c3 + c4)
    eq = edge_recovery(a_hat, adj_true, pm)

    # 3) reasoners with known-good logic, on TRUE vs PERCEIVED structure
    hard = Hardwired()
    gnn = GNNReasoner()
    train_gnn(gnn, train)
    gnn.eval()

    out = {**{f"edge_{k}": v for k, v in eq.items()}}
    out["e2e_hybrid_conf_s3"] = acc_learned(hybrid, c3)  # sanity: ~0.43
    out["e2e_hybrid_conf_s4"] = acc_learned(hybrid, c4)
    # disambiguation: the hybrid's OWN reasoner on soft (=anchor) vs thresholded perceived graph
    out["own_soft_s3"] = hybrid_own_reasoner(hybrid, c3, threshold=False)
    out["own_hard_s3"] = hybrid_own_reasoner(hybrid, c3, threshold=True)
    out["own_soft_s4"] = hybrid_own_reasoner(hybrid, c4, threshold=False)
    out["own_hard_s4"] = hybrid_own_reasoner(hybrid, c4, threshold=True)
    for name, reasoner in (("hard", hard), ("gnn", gnn)):
        out[f"{name}_conf_true_s3"] = reasoner_acc(reasoner, hybrid, c3, "true")
        out[f"{name}_conf_perc_s3"] = reasoner_acc(reasoner, hybrid, c3, "perceived")
        out[f"{name}_conf_true_s4"] = reasoner_acc(reasoner, hybrid, c4, "true")
        out[f"{name}_conf_perc_s4"] = reasoner_acc(reasoner, hybrid, c4, "perceived")
        # the easy `cause` query under the SAME perceived structure (robustness contrast)
        out[f"{name}_cause_perc_s3"] = reasoner_acc(reasoner, hybrid, cause3, "perceived")
        out[f"{name}_cause_perc_s4"] = reasoner_acc(reasoner, hybrid, cause4, "perceived")
    # HARDWIRED on the SOFT perceived graph = what the hand-coded hybrid effectively computes
    out["hard_conf_soft_s3"] = reasoner_acc(hard, hybrid, c3, "perceived", hard=False)
    out["hard_conf_soft_s4"] = reasoner_acc(hard, hybrid, c4, "perceived", hard=False)
    return out


def main() -> None:
    print(f"Perception-bottleneck ablation, seeds {SEEDS}")
    rows = [run_seed(s) for s in SEEDS]

    def agg(key):
        vals = [r[key] for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), sd, [round(v, 3) for v in vals]

    print("\n  Perceived adjacency quality (fully-learned hybrid's GPT-2 perception):")
    for k in ("edge_edge_acc", "edge_precision", "edge_recall", "edge_f1"):
        m, sd, per = agg(k)
        print(f"    {k.replace('edge_edge', 'edge'):16s}  {m:.3f} +/- {sd:.3f}   {per}")

    print("\n  Anchor -- fully-learned hybrid end-to-end (perception + learned reasoning):")
    for k in ("e2e_hybrid_conf_s3", "e2e_hybrid_conf_s4"):
        m, sd, per = agg(k)
        print(f"    {k:24s}  {m:.3f} +/- {sd:.3f}   {per}")

    print("\n  Decisive 2x2 -- confounded-cause accuracy by reasoner x structure source:")
    print("    reasoner            TRUE adj            PERCEIVED adj (thresholded)")
    for name, label in (("hard", "HARDWIRED (algo)"), ("gnn", "GNN (trained/true)")):
        for s in (3, 4):
            tm, tsd, _ = agg(f"{name}_conf_true_s{s}")
            pm_, psd, _ = agg(f"{name}_conf_perc_s{s}")
            tag = "in-dist" if s == 3 else "held-out"
            print(
                f"    {label:18s} s{s} {tag:8s}  {tm:.3f} +/- {tsd:.3f}     {pm_:.3f} +/- {psd:.3f}"
            )
    hm3, hsd3, _ = agg("hard_conf_soft_s3")
    hm4, hsd4, _ = agg("hard_conf_soft_s4")
    print(
        f"    HARDWIRED on SOFT perceived graph (= hand-coded hybrid): "
        f"s3 {hm3:.3f}+/-{hsd3:.3f}  s4 {hm4:.3f}+/-{hsd4:.3f}"
    )

    print("\n  Disambiguation -- the hybrid's OWN trained reasoner, soft vs thresholded perceived:")
    for s in (3, 4):
        sm, ssd, _ = agg(f"own_soft_s{s}")
        hm, hsd, _ = agg(f"own_hard_s{s}")
        tag = "in-dist" if s == 3 else "held-out"
        print(
            f"    s{s} {tag:8s}  soft(=end-to-end) {sm:.3f}+/-{ssd:.3f}   "
            f"thresholded {hm:.3f}+/-{hsd:.3f}"
        )

    print("\n  Contrast -- easy `cause` query under the SAME perceived structure:")
    for name, label in (("hard", "HARDWIRED"), ("gnn", "GNN")):
        m3, sd3, _ = agg(f"{name}_cause_perc_s3")
        m4, sd4, _ = agg(f"{name}_cause_perc_s4")
        print(
            f"    {label:10s}  cause@perceived  s3 {m3:.3f}+/-{sd3:.3f}   s4 {m4:.3f}+/-{sd4:.3f}"
        )

    anchor, _, _ = agg("e2e_hybrid_conf_s3")
    th, _, _ = agg("hard_conf_true_s3")
    ph, _, _ = agg("hard_conf_perc_s3")
    os_, _, _ = agg("own_soft_s3")
    oh, _, _ = agg("own_hard_s3")
    verdict = (
        "PERCEPTION is the bottleneck"
        if ph < 0.7
        else "the END-TO-END-TRAINED reasoner is the bottleneck (perception is fine)"
    )
    fix = (
        "thresholding the perceived graph at inference recovers it (a trivial fix)"
        if oh - os_ > 0.2
        else "thresholding does NOT help -- the fix is in TRAINING (decouple/two-stage: a reasoner "
        "trained on clean structure already reaches ~1.0 on the perceived graph, per the 2x2)"
    )
    print(
        f"\n  Reading (s3): end-to-end hybrid {anchor:.2f}; the exact algorithm scores {th:.2f} "
        f"on TRUE edges and {ph:.2f} on the hybrid's PERCEIVED edges. Since perceived-edge "
        f"reasoning ({ph:.2f}) far exceeds the end-to-end hybrid ({anchor:.2f}), {verdict}. The "
        f"hybrid's own reasoner goes soft {os_:.2f} -> thresholded {oh:.2f}, so {fix}."
    )


if __name__ == "__main__":
    main()
