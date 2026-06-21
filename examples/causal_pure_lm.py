"""The PURE path: can a real GPT-2 internalise the causal computation in its own weights?

The hybrid bolts an explicit causal core onto GPT-2. The "pure" alternative is a real LM that does
the causal reasoning *in its own weights* -- no bespoke module. We give that path its best shot and
test it with multi-seed robustness, on the same natural-language causal QA (correlation vs causation):

  * PURE-DIRECT    GPT2LMHeadModel reads the prose and predicts yes/no from its LM head.
  * PURE-GROUNDED  same GPT-2 + a training-time auxiliary loss pushing its hidden states to encode the
    true edges (an IIT/probe-style grounding). Crucially the ANSWER still comes only from the LM head
    -- the aux head is a scaffold, discarded at test. Tests whether grounding *pressure* makes the LM
    internalise and *use* the structure.
  * reference: the HYBRID (explicit routed core) -- confounded ~1.0 in-dist, ~0.91 held-out.

Our arc predicts the pure path is weaker: a feature can be present in the representation yet not
mediate the output (presence != mediation), so grounding pressure need not make the LM head route
through it. We report confounded-cause accuracy (correlated but not causal) in-dist and held-out,
mean +/- std across seeds -- the honest test of whether a real LM can internalise causal reasoning.

CPU-sized (slow: several GPT-2 trainings).  Run::

    uv run --extra torch python examples/causal_pure_lm.py
"""

from __future__ import annotations

import os
import random
import statistics
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy  # data, GPT-2 config, VanillaLM (= pure-direct), pack/acc

SEEDS = [0, 1]
YES, NO = hy.VOCAB["yes"], hy.VOCAB["no"]


def confounded(data):
    return [dict(e, is_causal=1, label=e["cause"]) for e in data if e["corr"] and not e["cause"]]


class PureGroundedLM(nn.Module):
    """A real GPT-2: the answer comes from the LM head; an aux head only adds grounding pressure."""

    def __init__(self):
        super().__init__()
        from transformers import GPT2LMHeadModel

        self.lm = GPT2LMHeadModel(hy.gpt2())
        d = self.lm.config.n_embd
        self.aux = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, ids, attn, last, entw, present):
        out = self.lm(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        ans = out.logits[torch.arange(ids.size(0)), last]
        ans = ans[:, YES] - ans[:, NO]  # answer from the LM head only
        h = out.hidden_states[-1]
        slots = []
        for s in range(hy.NE):
            m = ((ids == entw[:, s : s + 1]) & (attn == 1)).float().unsqueeze(-1)
            slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        hv = torch.stack(slots, 1)
        hi = hv.unsqueeze(2).expand(-1, hy.NE, hy.NE, -1)
        hj = hv.unsqueeze(1).expand(-1, hy.NE, hy.NE, -1)
        edge = self.aux(torch.cat([hi, hj], -1)).squeeze(-1)
        pm = present.unsqueeze(2) * present.unsqueeze(1)
        return ans, edge, pm


def train_grounded(model, data, epochs=12, lr=5e-4, bs=64, lam=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            ids, attn, last, entw, _xs, _ys, _isc, lab, adj, pres = hy.pack(data[i : i + bs])
            ans, edge, pm = model(ids, attn, last, entw, pres)
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
def acc_grounded(model, data) -> float:
    if not data:
        return float("nan")
    ids, attn, last, entw, _xs, _ys, _isc, lab, adj, pres = hy.pack(data)
    ans, _, _ = model(ids, attn, last, entw, pres)
    return float(((ans > 0) == (lab > 0.5)).float().mean())


def run_seed(seed):
    torch.manual_seed(seed)
    train = hy.build(8000, sizes=[2, 3], seed=seed)
    t3 = hy.build(1500, sizes=[3], seed=seed + 50)
    t4 = hy.build(1500, sizes=[4], seed=seed + 60)
    c3, c4 = confounded(t3), confounded(t4)

    direct = hy.VanillaLM()
    hy.train(direct, train, hybrid=False, epochs=12)
    direct.eval()

    grounded = PureGroundedLM()
    train_grounded(grounded, train, epochs=12)
    grounded.eval()

    return {
        "pure_direct_s3": hy.acc(direct, c3, False),
        "pure_direct_s4": hy.acc(direct, c4, False),
        "pure_grounded_s3": acc_grounded(grounded, c3),
        "pure_grounded_s4": acc_grounded(grounded, c4),
    }


def main() -> None:
    print(f"PURE path: can a real GPT-2 internalise causal reasoning? (confounded, seeds {SEEDS})")
    rows = [run_seed(s) for s in SEEDS]
    print("\n  metric                 mean +/- std        per-seed")
    for key in rows[0]:
        vals = [r[key] for r in rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(
            f"  {key:20s}  {statistics.mean(vals):.3f} +/- {sd:.3f}   {[round(v, 3) for v in vals]}"
        )
    print(
        "\n  reference (hybrid, explicit routed core): confounded ~1.000 in-dist, ~0.914 held-out"
    )

    print(
        "\nReading: if PURE-DIRECT and PURE-GROUNDED stay well below the hybrid on confounded -- "
        "especially held-out -- then a real LM does NOT internalise causal reasoning from this "
        "training, even with grounding pressure: presence != mediation, the explicit routed core is "
        "what delivers it. If PURE-GROUNDED nears the hybrid, the LM can internalise it after "
        "all -- a surprise worth chasing. Either way, this is the honest test of the pure path."
    )


if __name__ == "__main__":
    main()
