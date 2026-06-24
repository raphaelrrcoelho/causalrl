# STATUS: foundational (imported) · by-construction core · Act 4 Coupling — GPT-2 + hand-coded core; data/utils others reuse  ·  map: CAUSAL_LLM.md
"""Hybrid: a real GPT-2 + the embedded causal core (the original 'plug it into a real LM' step).

Earlier couplings used a bespoke tiny module. This is the real thing: a genuine GPT-2 decoder
(``transformers.GPT2Model``) is the language backbone, and the embedded causal core is a reasoning
head on top of its hidden states. The contrast is the experiment -- the SAME real GPT-2, with vs
without the core:

  * VANILLA  GPT2LMHeadModel reads the prose and predicts the yes/no answer token directly (a normal
    LM doing the reasoning in its weights).
  * HYBRID   GPT2Model encodes the prose; we gather a representation per entity from its hidden
    states, an edge MLP builds a soft adjacency, and the embedded core (K-step reachability + the
    do() switch) produces the answer. GPT-2 does the *reading*; the core does the *causal reasoning*.

Both train end-to-end on natural-language causal QA (correlation vs causation), 2/3-entity prose for
training, 4-entity prose held out by size, including confounded scenarios (correlated, not causal).
If the hybrid beats vanilla GPT-2 -- especially out of size and on confounded cases -- then bolting
the embedded core onto a real LM is what buys causal reasoning beyond correlation.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_hybrid_lm.py
"""

from __future__ import annotations

import random

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

ENTITIES = [
    "smoking",
    "tar",
    "cancer",
    "rain",
    "grass",
    "slippery",
    "stress",
    "sleep",
    "exercise",
    "fitness",
]
VERBS = ["causes", "triggers", "increases"]
WORDS = (
    ENTITIES + VERBS + ["does", "cause", "are", "and", "correlated", ".", "?", "yes", "no", "<pad>"]
)
VOCAB = {w: i for i, w in enumerate(WORDS)}
NE = len(ENTITIES)
MAXLEN = 48
torch.set_num_threads(4)


def reachable(adj, i, j) -> bool:
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for v in range(NE):
            if adj[u][v] and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def make(sizes, rng) -> dict:
    k = rng.choice(sizes)
    ents = sorted(rng.sample(range(NE), k))
    slot = {e: s for s, e in enumerate(ents)}
    order = ents[:]
    rng.shuffle(order)
    adj = [[0] * NE for _ in range(NE)]
    toks = []
    for a in range(k):
        for b in range(a + 1, k):
            if rng.random() < 0.5:
                u, v = order[a], order[b]
                adj[slot[u]][slot[v]] = 1
                toks += [ENTITIES[u], rng.choice(VERBS), ENTITIES[v], "."]
    xe, ye = rng.sample(ents, 2)
    sx, sy = slot[xe], slot[ye]
    cause = reachable(adj, sx, sy)
    desc = {s: {t for t in range(k) if t != s and reachable(adj, s, t)} for s in range(k)}
    common = any(z not in (sx, sy) and sx in desc[z] and sy in desc[z] for z in range(k))
    corr = cause or reachable(adj, sy, sx) or common
    is_causal = rng.random() < 0.5
    if is_causal:
        toks += ["does", ENTITIES[xe], "cause", ENTITIES[ye], "?"]
    else:
        toks += ["are", ENTITIES[xe], "and", ENTITIES[ye], "correlated", "?"]
    label = int(cause if is_causal else corr)
    return {
        "ids": [VOCAB[t] for t in toks][:MAXLEN],
        "entw": [ents[s] for s in range(k)],  # entity vocab id per slot
        "xs": sx,
        "ys": sy,
        "is_causal": int(is_causal),
        "cause": int(cause),
        "corr": int(corr),
        "label": label,
        "adj": adj,
        "present": [s < k for s in range(NE)],
    }


def build(n, sizes, seed) -> list[dict]:
    rng = random.Random(seed)
    out, tries = [], 0
    cnt = {(t, lab): 0 for t in (0, 1) for lab in (0, 1)}
    cap = n // 4
    while len(out) < 4 * cap and tries < n * 800:
        tries += 1
        e = make(sizes, rng)
        key = (e["is_causal"], e["label"])
        if cnt[key] >= cap:
            continue
        out.append(e)
        cnt[key] += 1
    rng.shuffle(out)
    return out


def pack(items):
    width = max(len(e["ids"]) for e in items)
    ids = torch.full((len(items), width), VOCAB["<pad>"], dtype=torch.long)
    attn = torch.zeros(len(items), width, dtype=torch.long)
    last = torch.tensor([len(e["ids"]) - 1 for e in items])
    entw = torch.full((len(items), NE), VOCAB["<pad>"], dtype=torch.long)
    for j, e in enumerate(items):
        ids[j, : len(e["ids"])] = torch.tensor(e["ids"])
        attn[j, : len(e["ids"])] = 1
        for s, w in enumerate(e["entw"]):
            entw[j, s] = w
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        ids,
        attn,
        last,
        entw,
        g("xs", torch.long),
        g("ys", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["adj"] for e in items], dtype=torch.float),
        torch.tensor([e["present"] for e in items], dtype=torch.float),
    )


def gpt2(d=96, layers=3, heads=3):
    return GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=MAXLEN + 2,
        n_ctx=MAXLEN + 2,
        n_embd=d,
        n_layer=layers,
        n_head=heads,
        bos_token_id=VOCAB["<pad>"],
        eos_token_id=VOCAB["<pad>"],
    )


# ---- vanilla: real GPT-2 predicts the yes/no answer token directly ----
class VanillaLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = GPT2LMHeadModel(gpt2())

    def forward(self, ids, attn, last):
        logits = self.lm(input_ids=ids, attention_mask=attn).logits
        last_logits = logits[torch.arange(ids.size(0)), last]  # predict token after "?"
        return last_logits[:, VOCAB["yes"]] - last_logits[:, VOCAB["no"]]


# ---- hybrid: real GPT-2 backbone + embedded causal core head ----
class HybridLM(nn.Module):
    def __init__(self, steps=5):
        super().__init__()
        cfg = gpt2()
        self.gpt = GPT2Model(cfg)
        d = cfg.n_embd
        self.steps = steps
        self.edge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.read = nn.Linear(1, 1)

    def forward(self, ids, attn, last, entw, xs, ys, is_causal, present):
        h = self.gpt(input_ids=ids, attention_mask=attn).last_hidden_state  # (B,T,d)
        # gather a representation per entity slot from GPT-2's hidden states
        slots = []
        for s in range(NE):
            m = ((ids == entw[:, s : s + 1]) & (attn == 1)).float().unsqueeze(-1)  # (B,T,1)
            slots.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        hv = torch.stack(slots, 1)  # (B,NE,d)
        hi = hv.unsqueeze(2).expand(-1, NE, NE, -1)
        hj = hv.unsqueeze(1).expand(-1, NE, NE, -1)
        logit = (
            self.edge(torch.cat([hi, hj], -1)).squeeze(-1).masked_fill(torch.eye(NE).bool(), -30)
        )
        a = torch.sigmoid(logit) * present.unsqueeze(2) * present.unsqueeze(1)
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        idx = torch.arange(ids.size(0))
        fwd, bwd = r[idx, xs, ys], r[idx, ys, xs]
        rzx, rzy = r[idx, :, xs], r[idx, :, ys]
        notxy = torch.ones_like(rzx)
        notxy[idx, xs] = 0.0
        notxy[idx, ys] = 0.0
        common = (rzx * rzy * notxy * present).max(dim=1).values
        score = is_causal * fwd + (1 - is_causal) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        pm = present.unsqueeze(2) * present.unsqueeze(1)
        return self.read(score.unsqueeze(-1)).squeeze(-1), logit, pm


def train(model, data, hybrid, epochs=12, lr=5e-4, bs=64, lam=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        tot = nb = 0.0
        for i in range(0, len(data), bs):
            ids, attn, last, entw, xs, ys, isc, lab, adj, pres = pack(data[i : i + bs])
            if hybrid:
                ans, elog, pm = model(ids, attn, last, entw, xs, ys, isc, pres)
                loss = (
                    bce(ans, lab)
                    + lam
                    * (
                        nn.functional.binary_cross_entropy_with_logits(elog, adj, reduction="none")
                        * pm
                    ).sum()
                    / pm.sum()
                )
            else:
                loss = bce(model(ids, attn, last), lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"    epoch {ep + 1}/{epochs}  loss {tot / nb:.3f}")


@torch.no_grad()
def acc(model, data, hybrid) -> float:
    if not data:
        return float("nan")
    ids, attn, last, entw, xs, ys, isc, lab, adj, pres = pack(data)
    ans = model(ids, attn, last, entw, xs, ys, isc, pres)[0] if hybrid else model(ids, attn, last)
    return float(((ans > 0) == (lab > 0.5)).float().mean())


def report(name, model, tests, hybrid):
    print(f"\n  {name}")
    print("  size    observational   interventional   confounded-cause   seen?")
    for s in (2, 3, 4):
        corr = [e for e in tests[s] if not e["is_causal"]]
        cause = [e for e in tests[s] if e["is_causal"]]
        conf = [
            dict(e, is_causal=1, label=e["cause"]) for e in tests[s] if e["corr"] and not e["cause"]
        ]
        seen = "trained" if s in (2, 3) else "HELD-OUT"
        print(
            f"  size {s}:    {acc(model, corr, hybrid):.3f}          {acc(model, cause, hybrid):.3f}"
            f"           {acc(model, conf, hybrid):.3f}        {seen}"
        )


def main() -> None:
    torch.manual_seed(0)
    train_data = build(8000, sizes=[2, 3], seed=1)
    tests = {s: build(1800, sizes=[s], seed=10 + s) for s in (2, 3, 4)}
    print(
        f"natural-language causal QA; vocab {len(VOCAB)} words; train 2/3-entity, test 4 held-out"
    )

    vanilla = VanillaLM()
    print(
        f"\ntraining VANILLA GPT-2 ({vanilla.lm.num_parameters() / 1e6:.2f}M) -> predict yes/no ..."
    )
    train(vanilla, train_data, hybrid=False)
    vanilla.eval()
    report("VANILLA GPT-2 (LM predicts the answer)", vanilla, tests, hybrid=False)

    hybrid = HybridLM()
    np_ = sum(p.numel() for p in hybrid.parameters())
    print(f"\ntraining HYBRID GPT-2 + causal core ({np_ / 1e6:.2f}M) ...")
    train(hybrid, train_data, hybrid=True)
    hybrid.eval()
    report("HYBRID GPT-2 + embedded causal core", hybrid, tests, hybrid=True)

    print(
        "\nReading: same real GPT-2 backbone. If the hybrid (GPT-2 reads prose, the embedded core "
        "reasons) beats vanilla GPT-2 -- especially on the held-out size and confounded pairs -- "
        "then plugging the causal core into a real LM is what delivers reasoning beyond correlation; "
        "vanilla GPT-2 doing the reasoning in its own weights is the harder, weaker path."
    )


if __name__ == "__main__":
    main()
