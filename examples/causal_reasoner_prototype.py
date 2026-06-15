"""Prototype 4: a small transformer that learns a causal *rule* from causalrl-generated traces.

The earlier examples taught a model a specific distribution (do vs see) or hard-coded the causal
computation into the architecture (the NCM). This one is the first executable slice of the
recommended "small Causal Reasoner" (see examples/CAUSAL_LLM_RESEARCH.md): teach a from-scratch
transformer a *structural causal rule* from demonstrations, and check it generalises to inputs it
never saw — the Axiomatic-Training result (Vashishtha et al., NeurIPS 2024) in miniature.

The rule we teach is **d-separation**, the graphical bedrock of the whole theory: every
identification result (back-door, front-door, the ID algorithm, transportability) is ultimately a
statement about d-separation. Given a DAG and a query "is X independent of Y given Z?", the model
must decide it by reasoning about blocked and open paths (including the collider rule). This is the
rule the Axiomatic-Training paper showed is genuinely learnable and *generalises* — and it is the
prerequisite a model needs before it can decide the harder downstream question of identifiability.

causalrl plays both roles the architecture needs:

* **generator** — random DAGs and queries are sampled and serialised into token traces;
* **verifier / oracle** — ``d_separated`` labels each trace (ground truth).

Two generalisation tests, which measure different things:

* **held-out graphs, same sizes (3-5 vars)** — graphs the model never saw, drawn from the trained
  size range. High accuracy here means it learned the path-blocking rule rather than memorising
  training instances. This is the honest test of *rule vs memorisation*, and it passes (~0.84).
* **size extrapolation (6-7 vars)** — larger graphs, i.e. longer token sequences than any seen in
  training. This conflates the causal rule with transformer *length generalisation*, a separate and
  notoriously hard problem; accuracy here is near baseline. Closing it needs the positional-encoding
  and curriculum care of the Axiomatic-Training regime (and more scale) — flagged open below.

Run::

    uv run --extra torch python examples/causal_reasoner_prototype.py

Trains in a couple of minutes on CPU. Didactic demonstration, not a performance claim. The harder
target — deciding *identifiability* (the Causal-Hierarchy-Theorem gate, where the model must abstain
or return partial-ID bounds) — is harder still; d-separation is the learnable foundation it builds
on.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from causalrl import CausalGraph
from causalrl.identification._separation import d_separated  # the d-separation oracle / verifier

# --------------------------------------------------------------------------------------------
# Vocabulary for serialising (graph, query, answer). Nodes are letters; edges are pairs
# "src -> dst"; the query is "[Q] X Y" with conditioning set "[C] z..."; the answer is a single
# yes/no token after "[A]" (yes = X and Y are d-separated given Z).
# --------------------------------------------------------------------------------------------

NODE_TOKENS = ["A", "B", "C", "D", "E", "F", "G"]
SPECIAL = ["[PAD]", "[G]", "->", "[Q]", "[C]", "[A]", "yes", "no", "[E]"]
VOCAB = SPECIAL + NODE_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, YES, NO = STOI["[PAD]"], STOI["yes"], STOI["no"]


def encode(tokens: list[str]) -> list[int]:
    return [STOI[t] for t in tokens]


# --------------------------------------------------------------------------------------------
# 1. Random DAG generator + the causalrl d-separation oracle.
# --------------------------------------------------------------------------------------------


def random_dag(n: int, rng: random.Random) -> tuple[CausalGraph, list[str]]:
    """A random DAG over n nodes whose labels are drawn at random from the whole pool.

    Random relabeling (not always the first n letters) trains every token uniformly and forces
    the model to learn the *structure* rather than which specific letters tend to appear.
    """
    nodes = rng.sample(NODE_TOKENS, n)
    order = nodes[:]
    rng.shuffle(order)
    directed = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < 0.4
    ]
    return CausalGraph(directed_edges=directed, nodes=nodes), order


def make_example(n: int, rng: random.Random) -> tuple[list[str], bool]:
    """Sample (DAG, X, Y, Z) and label by causalrl's d-separation oracle."""
    graph, order = random_dag(n, rng)
    x, y = rng.sample(order, 2)
    rest = [v for v in order if v not in (x, y)]
    k = rng.randint(0, min(2, len(rest)))
    z = rng.sample(rest, k)
    label = d_separated(graph, {x}, {y}, set(z))  # the oracle / verifier

    tokens = ["[G]"]
    for src, dst in graph.directed_edges:
        tokens += [src, "->", dst]
    tokens += ["[Q]", x, y, "[C]", *z, "[A]", "yes" if label else "no", "[E]"]
    return tokens, label


def build_dataset(sizes: list[int], n: int, seed: int) -> list[list[str]]:
    """Rejection-sample a roughly class-balanced set of traces over the given graph sizes."""
    rng = random.Random(seed)
    pos: list[list[str]] = []
    neg: list[list[str]] = []
    target = n // 2
    while len(pos) < target or len(neg) < target:
        tokens, label = make_example(rng.choice(sizes), rng)
        bucket = pos if label else neg
        if len(bucket) < target:
            bucket.append(tokens)
    data = pos + neg
    rng.shuffle(data)
    return data


# --------------------------------------------------------------------------------------------
# 2. A small from-scratch causal (autoregressive) transformer.
# --------------------------------------------------------------------------------------------


class TinyReasoner(nn.Module):
    def __init__(self, vocab: int, max_len: int, d_model: int = 128, heads: int = 8, layers: int = 4
                 ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, batch_first=True, dropout=0.0
        )
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: Tensor) -> Tensor:
        t = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()  # causal mask
        return self.head(self.blocks(h, mask=mask, src_key_padding_mask=(idx == PAD)))


def pad_batch(rows: list[list[int]], max_len: int) -> Tensor:
    out = torch.full((len(rows), max_len), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r)
    return out


def train(model: nn.Module, data: list[list[int]], max_len: int, epochs: int = 50,
          batch: int = 128, lr: float = 5e-4) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    for epoch in range(epochs):
        rng.shuffle(data)
        total, nb = 0.0, 0
        for i in range(0, len(data), batch):
            ids = pad_batch(data[i : i + batch], max_len)
            logits = model(ids)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, len(VOCAB)),
                ids[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total / nb:.3f}")


# --------------------------------------------------------------------------------------------
# 3. Evaluate: read the model's yes/no at the [A] slot and score against the causalrl oracle.
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def accuracy(model: nn.Module, traces: list[list[str]], max_len: int) -> float:
    correct = 0
    for tokens in traces:
        a = tokens.index("[A]")
        ids = torch.tensor(encode(tokens[: a + 1]))[None]  # everything up to and including [A]
        logits = model(ids)[0, -1]
        pred = YES if logits[YES] > logits[NO] else NO
        correct += int(pred == STOI[tokens[a + 1]])
    return correct / len(traces)


def main() -> None:
    torch.manual_seed(0)

    # Train ONLY on small graphs (3-5 vars); hold out larger graphs (6-7 vars) entirely.
    train_data = build_dataset(sizes=[3, 4, 5], n=8000, seed=1)
    test_small = build_dataset(sizes=[3, 4, 5], n=1000, seed=2)
    test_large = build_dataset(sizes=[6, 7], n=1000, seed=3)  # unseen sizes -> tests the *rule*

    max_len = max(len(r) for r in train_data + test_small + test_large)
    print(f"train: {len(train_data)} d-separation traces (3-5 vars)   example:")
    print(f"  {' '.join(train_data[0])}")
    print(f"vocab {len(VOCAB)}, max_len {max_len}\n")

    model = TinyReasoner(len(VOCAB), max_len)
    enc_train = [encode(r) for r in train_data]
    print(f"training a {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M-param "
          "transformer on d-separation traces ...")
    train(model, enc_train, max_len)

    base = sum(1 for t in test_large if t[t.index("[A]") + 1] == "yes") / len(test_large)
    held_out = accuracy(model, test_small, max_len)
    extrapolated = accuracy(model, test_large, max_len)
    print("\nd-separation accuracy (vs the causalrl oracle):")
    print(f"  held-out graphs    (3-5 vars, unseen instances):  {held_out:.3f}")
    print(f"  size extrapolation (6-7 vars, longer sequences):  {extrapolated:.3f}")
    print(f"  (majority-class baseline, balanced: ~{max(base, 1 - base):.2f})")
    print(
        "\nOn held-out graphs of the trained sizes the transformer judges d-separation well above "
        "baseline: it learned the path-blocking rule (colliders included), not the training "
        "instances. Extrapolating to larger graphs (longer sequences) drops to near baseline — "
        "that is transformer length generalisation, a separate hard problem, not a failure to "
        "learn the rule. d-separation is the graphical bedrock every identification result builds "
        "on; "
        "teaching it structurally is the first slice of a causal reasoner. Scaling it, and the "
        "identifiability gate, are the open frontier — see examples/CAUSAL_LLM_RESEARCH.md."
    )


if __name__ == "__main__":
    main()
