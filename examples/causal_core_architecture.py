"""An *embedded* causal reasoning core -- the first architectural brick of a causal LM.

Our results showed a vanilla transformer (a) encodes the causal structure but does not route the
answer through it (presence != mediation) and (b) fails to size-generalize the reachability reasoning
(chance on graphs larger than trained). The fix is architectural: *embed* the causal machinery in the
weights, not bolt on an external engine.

This module does that, minimally and end-to-end-differentiable:

    text  --[transformer encoder]-->  per-variable reps
          --[pairwise edge MLP]----->  a soft adjacency matrix A   (causal structure, made explicit)
          --[K-step propagation]---->  soft transitive closure R   (reachability = causal ancestry)
          --[affine on R[x,y]]------>  answer

Three architectural commitments, each targeting one observed deficit:
  * the answer is a function ONLY of R (which comes only from A) -> the computation is *routed through*
    the causal structure (mediation by construction, not hope);
  * reachability is computed by **iterative propagation** (R <- clamp(A + R@A)), an algorithm that is
    size-invariant -> it extrapolates to graph sizes never seen in training;
  * A is an explicit, inspectable, intervenable object (you can do() on it by zeroing a column).

We ground A with an auxiliary loss (predict the true edges) -- the learned "perception/discovery" half
-- while the propagation is a fixed inductive bias -- the "causal algorithm" half. Trained on 2- and
3-variable graphs (random variable names), tested on **4- and 5-variable graphs held out by size**.
The vanilla transformer struct-only model collapsed to ~chance there (curriculum experiment); a core
with the causal algorithm embedded should not.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_core_architecture.py
"""

from __future__ import annotations

import random

import torch
from torch import nn

LETTERS = "ABCDE"
VOCAB = {"<pad>": 0, ">": 1, "?": 2, "A": 3, "B": 4, "C": 5, "D": 6, "E": 7}
MAXLEN = 48
torch.set_num_threads(4)


# ==============================================================================================
# 1. Data: random DAGs -> edge-list tokens + query + (label, true adjacency). Random variable names.
# ==============================================================================================


def reachable(adj: list[list[int]], i: int, j: int) -> bool:
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for v in range(5):
            if adj[u][v] and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def make(sizes: list[int], rng: random.Random) -> dict:
    k = rng.choice(sizes)
    letters = rng.sample(LETTERS, k)  # random subset -> every name appears at every size
    order = letters[:]
    rng.shuffle(order)
    adj = [[0] * 5 for _ in range(5)]
    edges = []
    for a in range(k):
        for b in range(a + 1, k):
            if rng.random() < 0.45:
                u, v = order[a], order[b]
                adj[LETTERS.index(u)][LETTERS.index(v)] = 1
                edges.append((u, v))
    x, y = rng.sample(letters, 2)
    toks = []
    for u, v in edges:
        toks += [u, ">", v]
    toks += ["?", x, y]
    present = [chr(65 + i) in letters for i in range(5)]
    return {
        "ids": [VOCAB[t] for t in toks][:MAXLEN],
        "xs": LETTERS.index(x),
        "ys": LETTERS.index(y),
        "label": int(reachable(adj, LETTERS.index(x), LETTERS.index(y))),
        "adj": adj,
        "present": present,
    }


def build(n: int, sizes: list[int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    want = n // 2
    cnt = {0: 0, 1: 0}
    out = []
    tries = 0
    while len(out) < 2 * want and tries < n * 400:
        tries += 1
        e = make(sizes, rng)
        if cnt[e["label"]] >= want:
            continue
        out.append(e)
        cnt[e["label"]] += 1
    rng.shuffle(out)
    return out


def batch(items: list[dict]):
    width = max(len(e["ids"]) for e in items)
    ids = torch.zeros(len(items), width, dtype=torch.long)
    pad = torch.ones(len(items), width, dtype=torch.bool)  # True = pad
    for j, e in enumerate(items):
        ids[j, : len(e["ids"])] = torch.tensor(e["ids"])
        pad[j, : len(e["ids"])] = False
    xs = torch.tensor([e["xs"] for e in items])
    ys = torch.tensor([e["ys"] for e in items])
    lab = torch.tensor([e["label"] for e in items], dtype=torch.float)
    adj = torch.tensor([e["adj"] for e in items], dtype=torch.float)  # (B,5,5)
    pres = torch.tensor([e["present"] for e in items], dtype=torch.float)  # (B,5)
    return ids, pad, xs, ys, lab, adj, pres


# ==============================================================================================
# 2. The embedded causal core.
# ==============================================================================================


class CausalCore(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2, steps: int = 5):
        super().__init__()
        self.steps = steps
        self.emb = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(MAXLEN, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.read = nn.Linear(1, 1)  # affine on the reachability score R[x, y]

    def adjacency(self, ids, pad):
        b, t = ids.shape
        h = self.emb(ids) + self.pos(torch.arange(t))
        h = self.encoder(h, src_key_padding_mask=pad)  # (B,T,d)
        # per-variable representation: mean of token reps at that letter's positions
        slots = []
        for v in range(5):  # letters A..E -> vocab ids 3..7
            m = ((ids == v + 3) & ~pad).float().unsqueeze(-1)  # (B,T,1)
            slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        hv = torch.stack(slots, dim=1)  # (B,5,d)
        hi = hv.unsqueeze(2).expand(-1, 5, 5, -1)
        hj = hv.unsqueeze(1).expand(-1, 5, 5, -1)
        logits = self.edge(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # (B,5,5)
        eye = torch.eye(5).bool()
        logits = logits.masked_fill(eye, -30.0)  # no self-loops
        return logits

    def forward(self, ids, pad, xs, ys, pres):
        logits = self.adjacency(ids, pad)
        pair_mask = pres.unsqueeze(2) * pres.unsqueeze(1)  # only present-present edges
        a = torch.sigmoid(logits) * pair_mask  # soft adjacency (B,5,5)
        # iterative reachability propagation: R <- clamp(A + R @ A) -- size-invariant algorithm
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(ids.size(0))
        score = r[idx, xs, ys].unsqueeze(-1)  # reachability x -> y
        return self.read(score).squeeze(-1), logits, pair_mask


def train(model, data, epochs=16, lr=2e-3, bs=128, lam=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        tot_a = tot_e = nb = 0.0
        for i in range(0, len(data), bs):
            ids, pad, xs, ys, lab, adj, pres = batch(data[i : i + bs])
            ans_logit, edge_logit, pmask = model(ids, pad, xs, ys, pres)
            ans_loss = bce(ans_logit, lab)
            edge_loss = (
                nn.functional.binary_cross_entropy_with_logits(edge_logit, adj, reduction="none")
                * pmask
            ).sum() / pmask.sum()
            loss = ans_loss + lam * edge_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_a += float(ans_loss.detach())
            tot_e += float(edge_loss.detach())
            nb += 1
        print(f"    epoch {ep + 1}/{epochs}  answer {tot_a / nb:.3f}  edge {tot_e / nb:.3f}")


@torch.no_grad()
def accuracy(model, data, bs=256) -> tuple[float, float]:
    correct = edge_ok = total = epairs = 0.0
    for i in range(0, len(data), bs):
        ids, pad, xs, ys, lab, adj, pres = batch(data[i : i + bs])
        ans_logit, edge_logit, pmask = model(ids, pad, xs, ys, pres)
        correct += int(((ans_logit > 0) == (lab > 0.5)).sum())
        total += lab.numel()
        edge_pred = (edge_logit > 0).float()
        edge_ok += float(((edge_pred == adj) * pmask).sum())
        epairs += float(pmask.sum())
    return correct / total, edge_ok / epairs


def main() -> None:
    torch.manual_seed(0)
    print("building data: train sizes {2,3}, held-out test sizes {4,5} (random variable names) ...")
    train_data = build(12000, sizes=[2, 3], seed=1)
    tests = {s: build(1500, sizes=[s], seed=10 + s) for s in (2, 3, 4, 5)}

    model = CausalCore()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"embedded CausalCore: {n_params / 1e3:.0f}K params (encoder + edge-MLP + fixed propagation)")
    train(model, train_data)
    model.eval()

    print("\n                accuracy (does X cause Y, via the embedded core)")
    print("                model    edge-recovery    (size in training? )")
    for s in (2, 3, 4, 5):
        acc, edge = accuracy(model, tests[s])
        seen = "trained" if s in (2, 3) else "HELD-OUT"
        print(f"  size {s}:        {acc:.3f}        {edge:.3f}        {seen}")

    print("\n  reference (vanilla transformer struct-only, from the curriculum experiment):")
    print("    size 3 ~0.83 (trained), size 4 ~0.55 (held-out, ~chance)")
    print("\nReading: if the embedded core stays high on the HELD-OUT sizes 4 and 5 (where the vanilla "
          "transformer was at chance), then embedding the causal algorithm -- explicit adjacency + "
          "iterative reachability propagation, with the answer routed through it -- is what gives "
          "size-general causal reasoning. That is the architectural piece a causal LM needs.")


if __name__ == "__main__":
    main()
