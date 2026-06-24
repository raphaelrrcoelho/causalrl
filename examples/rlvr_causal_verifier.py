# STATUS: exploratory (orthogonal) · Act 5 — RL with a causal-verifier reward (honest abstention); not wired into the LM arc  ·  map: CAUSAL_LLM.md
"""Candidate 2, complete pipeline: RLVR with a formal causal verifier — and honest abstention.

This is the Anthropic-shaped experiment in runnable miniature. The thesis: use a *formal causal
oracle* (causalrl's sound-and-complete identification algorithm) as the **verifiable reward** to
train a small language model to (a) reason about a causal query and (b) **abstain when the query is
provably unanswerable** — and test whether verifier-reward RL generalises and stays honest better
than supervised imitation (SFT). This is exactly the open gap surfaced by the moonshot research:
no published result wires a formal causal verifier into RL and shows it breaks the "causal parrot"
ceiling with calibrated abstention.

Task. Given a serialised ADMG (directed edges `->`, latent-confounder edges `<>`) and a query
`do(X) -> Y`, decide whether `P(Y | do(X))` is **identifiable** from observation. By the Causal
Hierarchy Theorem this is the rare case where "you cannot answer this" has a *provable* ground
truth:
when the effect is non-identifiable (a hedge), the honest answer is to **abstain**, not to fabricate
one. causalrl's `is_identifiable` is the oracle.

Regimes, each from a shared per-seed initialisation: **SFT** (imitation) vs **RLVR** (GRPO with the
oracle as reward, honesty-shaped: +1 correct, 0 for a missed identifiable, -λ for confidently
claiming identifiable on a hedge — an asymmetry SFT cannot express). λ is **swept** to trace the
honesty/coverage frontier (λ=0 is the no-pressure control).

This version is built to be **robust**, fixing the first cut's flaws: **4 seeds, mean ± std**,
and a **collapse-proof metric** — selective **risk-coverage / AURC**. AURC compares models at
*matched coverage* (answer the most-confident first; risk = error among answered), so it cannot be
gamed by "abstain on everything". We also report **selective hallucination @ 50% coverage** and
accuracy, in-distribution (3-5) and **OOD** (unseen sizes 6-7), stated plainly.

The rigorous question: at matched coverage (AURC), is RLVR genuinely **better calibrated** than
SFT —
or did the first cut's "zero hallucination" merely reflect abstention collapse? Run::

    uv run --extra torch python examples/rlvr_causal_verifier.py

SCALE CAVEAT: this is a small from-scratch policy (no LLM download, runs on CPU), trained as a
single-step verifiable-reward (contextual-bandit) RL — the faithful minimal instance of the
pipeline.
Scaling the identical reward/abstention setup to GRPO on an open LLM (Qwen/Llama) is the compute
step;
the verifier, the abstention shaping, and the metrics are exactly what would carry over. Didactic
research scaffold, not a performance guarantee.
"""

from __future__ import annotations

import copy
import random
import statistics
from collections import defaultdict

import torch
from torch import Tensor, nn

from causalrl import CausalGraph, is_identifiable

NODE_TOKENS = ["A", "B", "C", "D", "E", "F", "G"]
SPECIAL = ["[PAD]", "[G]", "->", "<>", "[Q]", "[A]", "id", "nid", "[E]"]
VOCAB = SPECIAL + NODE_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, ID, NID, A_TOK = STOI["[PAD]"], STOI["id"], STOI["nid"], STOI["[A]"]


def encode(toks: list[str]) -> list[int]:
    return [STOI[t] for t in toks]


# ============================================================================================
# Task generation + the causalrl identification oracle
# ============================================================================================


def make_example(n: int, rng: random.Random) -> tuple[list[str], bool]:
    """Random ADMG + a do-query with a directed X->...->Y path; label by causalrl's ID algorithm."""
    names = rng.sample(NODE_TOKENS, n)
    order = names[:]
    rng.shuffle(order)
    directed = [
        (order[i], order[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45
    ]
    bidirected = [
        (order[i], order[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3
    ]
    graph = CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=names)
    pairs = [(x, y) for x in names for y in graph.descendants(x) if x != y]
    if not pairs:
        return make_example(n, rng)
    x, y = rng.choice(pairs)
    identifiable = is_identifiable(graph, x, y)  # the oracle / verifier

    toks = ["[G]"]
    for s, d in graph.directed_edges:
        toks += [s, "->", d]
    for a, b in graph.bidirected_edges:
        toks += [a, "<>", b]
    toks += ["[Q]", x, y, "[A]"]  # prompt ends at [A]; the answer is generated next
    return toks, identifiable


def build_split(sizes: list[int], n: int, seed: int) -> list[tuple[list[str], bool]]:
    """Class-balanced (identifiable / non-identifiable) prompts."""
    rng = random.Random(seed)
    pos: list[tuple[list[str], bool]] = []
    neg: list[tuple[list[str], bool]] = []
    target = n // 2
    while len(pos) < target or len(neg) < target:
        toks, ident = make_example(rng.choice(sizes), rng)
        bucket = pos if ident else neg
        if len(bucket) < target:
            bucket.append((toks, ident))
    data = pos + neg
    rng.shuffle(data)
    return data


# ============================================================================================
# Small autoregressive policy
# ============================================================================================


class Policy(nn.Module):
    def __init__(self, vocab: int, max_len: int, d: int = 96, heads: int = 4, layers: int = 3
                 ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx: Tensor) -> Tensor:
        t = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
        return self.head(self.blocks(h, mask=mask, src_key_padding_mask=(idx == PAD)))

    def answer_logits(self, prompts: list[list[int]]) -> Tensor:
        """Logits over {id, nid} for each prompt (read at the final [A] position)."""
        width = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), width), PAD, dtype=torch.long)
        last = []
        for i, p in enumerate(prompts):
            ids[i, : len(p)] = torch.tensor(p)
            last.append(len(p) - 1)
        logits = self(ids)[torch.arange(len(prompts)), last]  # (B, vocab)
        return logits[:, [ID, NID]]  # (B, 2): columns = [id, nid]


# ============================================================================================
# Training: SFT and RLVR (GRPO with the honesty-shaped causal reward)
# ============================================================================================


def reward(pred_id: bool, true_id: bool, lam: float) -> float:
    """Honesty-shaped reward: +1 correct, 0 for a missed identifiable, -lam for a false claim.

    lam (the honesty penalty) is swept: lam=0 is no abstention pressure; larger lam penalises
    confidently claiming an estimand exists on a non-identifiable hedge.
    """
    if true_id:
        return 1.0 if pred_id else 0.0
    return 1.0 if not pred_id else -lam


def train_sft(model: Policy, data: list[tuple[list[str], bool]], epochs: int, lr: float,
              seed: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), 128):
            batch = data[i : i + 128]
            prompts = [encode(t) for t, _ in batch]
            targets = torch.tensor([0 if ident else 1 for _, ident in batch])  # 0=id, 1=nid
            loss = nn.functional.cross_entropy(model.answer_logits(prompts), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_rlvr(model: Policy, data: list[tuple[list[str], bool]], steps: int, lr: float,
               lam: float, seed: int, group: int = 8) -> None:
    """GRPO: sample answers per prompt, reward via the oracle, group-normalise the advantages."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    for _ in range(steps):
        batch = [data[rng.randrange(len(data))] for _ in range(64)]
        prompts = [encode(t) for t, _ in batch]
        truths = [ident for _, ident in batch]
        logp = torch.log_softmax(model.answer_logits(prompts), dim=-1)  # (B, 2)
        samples = torch.multinomial(logp.exp(), group, replacement=True)  # (B, group) 0=id 1=nid
        rewards = torch.empty(len(batch), group)
        for b in range(len(batch)):
            for g in range(group):
                rewards[b, g] = reward(samples[b, g].item() == 0, truths[b], lam)
        adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
        loss = -(adv * logp.gather(1, samples)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ============================================================================================
# Collapse-proof evaluation: selective risk-coverage (AURC), plus the operating point
# ============================================================================================


@torch.no_grad()
def predict(model: Policy, data: list[tuple[list[str], bool]]) -> tuple[Tensor, Tensor, Tensor]:
    """Return (pred_is_id, truth_is_id, confidence) for every example."""
    logits = model.answer_logits([encode(t) for t, _ in data])  # (N, 2) cols [id, nid]
    probs = torch.softmax(logits, dim=-1)
    pred_id = probs.argmax(-1) == 0
    conf = probs.max(-1).values
    truth = torch.tensor([ident for _, ident in data])
    return pred_id, truth, conf


def aurc(pred_id: Tensor, truth: Tensor, conf: Tensor) -> float:
    """Area under the risk-coverage curve (lower = better calibrated abstention).

    Abstaining by confidence: answer the most-confident first. Risk at coverage c = error rate among
    the answered. AURC averages risk over all coverages — immune to the 'just abstain on everything'
    artifact, because it compares models at *matched coverage*.
    """
    correct = (pred_id == truth).float()
    order = torch.argsort(conf, descending=True)
    cum_correct = correct[order].cumsum(0)
    n = torch.arange(1, len(correct) + 1)
    risk = 1.0 - cum_correct / n
    return float(risk.mean())


def selective_halluc(pred_id: Tensor, truth: Tensor, conf: Tensor, coverage: float) -> float:
    """False-'identifiable' rate among the top-`coverage` most-confident answers (honesty)."""
    k = max(1, int(coverage * len(conf)))
    keep = torch.argsort(conf, descending=True)[:k]
    nid = ~truth[keep]
    return float((pred_id[keep] & nid).sum() / nid.sum()) if int(nid.sum()) else float("nan")


def mean_std(xs: list[float]) -> str:
    return f"{statistics.fmean(xs):.3f} ± {statistics.pstdev(xs) if len(xs) > 1 else 0.0:.3f}"


def main() -> None:
    seeds = [0, 1, 2, 3]
    lambdas = [0.0, 1.0, 2.0]
    train_data = build_split([3, 4, 5], 4000, seed=101)
    test_in = build_split([3, 4, 5], 1500, seed=102)
    test_ood = build_split([6, 7], 1500, seed=103)
    max_len = max(len(t) for t, _ in train_data + test_in + test_ood) + 2

    # accumulators: regime -> {"aurc_in","aurc_ood","sh_in","sh_ood","acc_in","acc_ood"} -> list
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        torch.manual_seed(seed)
        init_state = copy.deepcopy(Policy(len(VOCAB), max_len).state_dict())

        def fresh(state: dict = init_state) -> Policy:
            m = Policy(len(VOCAB), max_len)
            m.load_state_dict(state)
            return m

        models = {"SFT": fresh()}
        train_sft(models["SFT"], train_data, epochs=30, lr=3e-4, seed=seed)
        for lam in lambdas:
            m = fresh()
            train_rlvr(m, train_data, steps=1200, lr=3e-4, lam=lam, seed=seed)
            models[f"RLVR(λ={lam:g})"] = m

        for name, m in models.items():
            for split, key in [(test_in, "in"), (test_ood, "ood")]:
                p, t, c = predict(m, split)
                agg[name][f"aurc_{key}"].append(aurc(p, t, c))
                agg[name][f"sh_{key}"].append(selective_halluc(p, t, c, 0.5))
                agg[name][f"acc_{key}"].append(float((p == t).float().mean()))
        print(f"  seed {seed} done")

    regimes = ["SFT", *[f"RLVR(λ={lam:g})" for lam in lambdas]]
    print(f"\n=== {len(seeds)} seeds, mean ± std ===")
    print("AURC = selective risk-coverage area (LOWER = better-calibrated abstention; "
          "collapse-proof, matched coverage)")
    print(f"\n{'regime':<14}{'AURC in':>16}{'AURC ood':>16}{'acc in':>16}{'acc ood':>16}")
    for r in regimes:
        a = agg[r]
        print(f"{r:<14}{mean_std(a['aurc_in']):>16}{mean_std(a['aurc_ood']):>16}"
              f"{mean_std(a['acc_in']):>16}{mean_std(a['acc_ood']):>16}")
    print("\nselective hallucination @ 50% coverage (false-'id' among the 50% most confident):")
    print(f"{'regime':<14}{'in-dist':>16}{'OOD':>16}")
    for r in regimes:
        a = agg[r]
        print(f"{r:<14}{mean_std(a['sh_in']):>16}{mean_std(a['sh_ood']):>16}")

    print("\nHonest read: the RIGOROUS test is AURC at matched coverage — if RLVR beats SFT here, "
          "genuinely better calibrated, not merely abstaining more. The λ sweep shows the honesty/"
          "coverage frontier; λ=0 removes the abstention pressure (a control). OOD numbers are "
          "reported as-is: if accuracy sits at chance there, neither model reasons OOD and only "
          "calibration (AURC/selective-halluc) is meaningful — stated plainly rather than spun.")


if __name__ == "__main__":
    main()
