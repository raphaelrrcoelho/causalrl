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

Three regimes from one shared initialisation:
  * **base**  — untrained policy (reference point).
  * **SFT**   — supervised cross-entropy on the correct answer (imitation).
  * **RLVR**  — GRPO with the causal oracle as reward, shaped for honesty: +1 for a correct verdict,
                0 for a missed identifiable case, and **-1 for confidently claiming identifiable
                on a
                non-identifiable query** (hallucination). SFT cannot express this asymmetry.

Metrics, in-distribution (sizes 3-5) and **OOD** (unseen sizes 6-7):
  * accuracy vs the oracle;
  * **hallucination rate** = P(say "identifiable" | truly non-identifiable) — the honesty number.

The Anthropic-relevant question: does RLVR with a formal verifier generalise OOD and **hallucinate
less** than SFT? Run::

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


def reward(pred_id: bool, true_id: bool) -> float:
    if true_id:
        return 1.0 if pred_id else 0.0
    return 1.0 if not pred_id else -1.0  # hallucinating "identifiable" on a hedge is penalised


def train_sft(model: Policy, data: list[tuple[list[str], bool]], epochs: int, lr: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), 128):
            batch = data[i : i + 128]
            prompts = [encode(t) for t, _ in batch]
            targets = torch.tensor([0 if ident else 1 for _, ident in batch])  # 0=id, 1=nid
            logits = model.answer_logits(prompts)
            loss = nn.functional.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_rlvr(model: Policy, data: list[tuple[list[str], bool]], steps: int, lr: float,
               group: int = 8) -> None:
    """GRPO: sample answers per prompt, reward via the oracle, group-normalise the advantages."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    for _ in range(steps):
        batch = [data[rng.randrange(len(data))] for _ in range(64)]
        prompts = [encode(t) for t, _ in batch]
        truths = [ident for _, ident in batch]
        logits = model.answer_logits(prompts)  # (B, 2)
        logp = torch.log_softmax(logits, dim=-1)
        probs = logp.exp()
        # sample `group` answers per prompt
        samples = torch.multinomial(probs, group, replacement=True)  # (B, group), 0=id 1=nid
        rewards = torch.empty(len(batch), group)
        for b in range(len(batch)):
            for g in range(group):
                rewards[b, g] = reward(pred_id=(samples[b, g].item() == 0), true_id=truths[b])
        adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
        chosen_logp = logp.gather(1, samples)  # (B, group)
        loss = -(adv * chosen_logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ============================================================================================
# Evaluation: accuracy + hallucination rate, in-distribution and OOD
# ============================================================================================


@torch.no_grad()
def evaluate(model: Policy, data: list[tuple[list[str], bool]]) -> tuple[float, float]:
    prompts = [encode(t) for t, _ in data]
    truths = [ident for _, ident in data]
    preds_id = (model.answer_logits(prompts).argmax(-1) == 0).tolist()  # True => predicted "id"
    correct = sum(int(p == t) for p, t in zip(preds_id, truths, strict=True))
    nid = [(p, t) for p, t in zip(preds_id, truths, strict=True) if not t]
    halluc = sum(int(p) for p, _ in nid) / len(nid) if nid else float("nan")
    return correct / len(data), halluc


def main() -> None:
    torch.manual_seed(0)
    train_data = build_split([3, 4, 5], 5000, seed=1)
    test_in = build_split([3, 4, 5], 1500, seed=2)
    test_ood = build_split([6, 7], 1500, seed=3)
    max_len = max(len(t) for t, _ in train_data + test_in + test_ood) + 2

    init = Policy(len(VOCAB), max_len)
    init_state = copy.deepcopy(init.state_dict())
    sft = Policy(len(VOCAB), max_len)
    rlvr = Policy(len(VOCAB), max_len)
    sft.load_state_dict(init_state)
    rlvr.load_state_dict(init_state)

    print("training SFT (imitation) ...")
    train_sft(sft, train_data, epochs=40, lr=3e-4)
    print("training RLVR (GRPO with the causalrl verifier + honesty-shaped reward) ...")
    train_rlvr(rlvr, train_data, steps=1500, lr=3e-4)

    print("\n                       in-dist (3-5)        OOD (6-7, unseen sizes)")
    print("                     acc    halluc.        acc    halluc.")
    for name, m in [("base ", init), ("SFT  ", sft), ("RLVR ", rlvr)]:
        ai, hi = evaluate(m, test_in)
        ao, ho = evaluate(m, test_ood)
        print(f"  {name}              {ai:.3f}   {hi:.3f}          {ao:.3f}   {ho:.3f}")
    print("\nhalluc. = P(says 'identifiable' | truly NON-identifiable) — lower is more honest.")
    print("The Anthropic-relevant read: does RLVR with a formal causal verifier generalise OOD and "
          "hallucinate less than SFT? The honesty-shaped reward (which SFT cannot express) is the "
          "lever — it makes confidently-wrong worse than abstaining, the calibrated honesty the "
          "Causal Hierarchy Theorem makes ground-truth-checkable.")


if __name__ == "__main__":
    main()
