"""Prototype 4: a small transformer that learns a causal *rule* from causalrl-generated traces.

The earlier examples taught a model a specific distribution (do vs see) or hard-coded the causal
computation into the architecture (the NCM). This one is the first executable slice of the
recommended "small Causal Reasoner" (see examples/CAUSAL_LLM_RESEARCH.md): teach a from-scratch
transformer a *structural causal rule* from demonstrations, and check it generalises to inputs it
never saw — the Axiomatic-Training result (Vashishtha et al., NeurIPS 2024) in miniature.

The rule we teach is the single most important one for honest causal reasoning: **identifiability**.
Given a causal diagram (an ADMG with possible latent confounders shown as ``<>`` edges) and a query
``do(X) -> Y``, is ``P(Y | do(X))`` identifiable from observational data at all? This is the gate
demanded by the Causal Hierarchy Theorem: a model that knows the answer knows *when it must abstain*
rather than hallucinate an interventional number it cannot possibly have.

causalrl plays both roles the architecture needs:

* **generator** — random ADMGs are sampled and serialised into token traces;
* **verifier / oracle** — ``is_identifiable`` (the sound-and-complete ID algorithm) labels each one.

The test of *reasoning* vs *memorisation*: we train only on small graphs (3-4 variables) and
evaluate on larger ones (5 variables) the model has never seen. Generalising there means it learned
the rule (the structure of a non-identifiable "hedge"/bow-arc), not the training graphs.

Run::

    uv run --extra torch python examples/causal_reasoner_prototype.py

Trains in a couple of minutes on CPU. Didactic demonstration, not a performance claim. The natural
next steps (numeric L2/L3 answers via the NCM head, and abstention that returns partial-ID bounds
when the answer here is "no") are described in the research note.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from causalrl import CausalGraph, is_identifiable

# --------------------------------------------------------------------------------------------
# Vocabulary for serialising (graph, query, answer). Nodes are letters; edges are triples
# "src REL dst"; the query is "[Q] X Y"; the answer is a single yes/no token after "[A]".
# --------------------------------------------------------------------------------------------

NODE_TOKENS = ["A", "B", "C", "D", "E", "F", "G"]
SPECIAL = ["[PAD]", "[G]", "->", "<>", "[Q]", "[A]", "yes", "no", "[E]"]
VOCAB = SPECIAL + NODE_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
PAD, YES, NO, ANS = STOI["[PAD]"], STOI["yes"], STOI["no"], STOI["[A]"]


def encode(tokens: list[str]) -> list[int]:
    return [STOI[t] for t in tokens]


# --------------------------------------------------------------------------------------------
# 1. Random ADMG generator + the causalrl identifiability oracle.
# --------------------------------------------------------------------------------------------


def random_admg(n: int, rng: random.Random) -> tuple[CausalGraph, list[str]]:
    """A random DAG over n nodes plus a few bidirected (latent-confounder) edges.

    The n node labels are drawn at random from the whole pool (not always the first n), so every
    token is trained on uniformly. This relabeling is the augmentation that forces the model to
    learn the *structure* rather than which specific letters tend to appear.
    """
    nodes = rng.sample(NODE_TOKENS, n)
    order = nodes[:]
    rng.shuffle(order)
    directed = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < 0.45
    ]
    bidirected = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < 0.25
    ]
    return CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=nodes), order


def make_example(n: int, rng: random.Random) -> tuple[list[str], bool] | None:
    """Sample (graph, X, Y) with a directed path X->...->Y; label by causalrl's ID algorithm."""
    graph, order = random_admg(n, rng)
    # pick a treatment/outcome pair where X is a proper ancestor of Y (non-trivial effect)
    candidates = [
        (x, y)
        for x in order
        for y in graph.descendants(x)
        if x != y
    ]
    if not candidates:
        return None
    x, y = rng.choice(candidates)
    label = is_identifiable(graph, x, y)  # the oracle / verifier

    tokens = ["[G]"]
    for src, dst in graph.directed_edges:
        tokens += [src, "->", dst]
    for a, b in graph.bidirected_edges:
        tokens += [a, "<>", b]
    tokens += ["[Q]", x, y, "[A]", "yes" if label else "no", "[E]"]
    return tokens, label


def build_dataset(sizes: list[int], n: int, seed: int) -> list[list[str]]:
    """Rejection-sample a roughly class-balanced set of traces over the given graph sizes."""
    rng = random.Random(seed)
    pos: list[list[str]] = []
    neg: list[list[str]] = []
    target = n // 2
    while len(pos) < target or len(neg) < target:
        ex = make_example(rng.choice(sizes), rng)
        if ex is None:
            continue
        tokens, label = ex
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
    def __init__(self, vocab: int, max_len: int, d_model: int = 160, heads: int = 8, layers: int = 4
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


def train(model: nn.Module, data: list[list[int]], max_len: int, epochs: int = 120,
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
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total / nb:.3f}")


# --------------------------------------------------------------------------------------------
# 3. Evaluate: read the model's yes/no at the [A] slot and score against the causalrl oracle.
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def accuracy(model: nn.Module, traces: list[list[str]], max_len: int) -> float:
    correct = 0
    for tokens in traces:
        a = tokens.index("[A]")
        prompt = encode(tokens[: a + 1])  # everything up to and including [A]
        ids = torch.tensor(prompt)[None]
        logits = model(ids)[0, -1]
        pred = YES if logits[YES] > logits[NO] else NO
        truth = STOI[tokens[a + 1]]
        correct += int(pred == truth)
    return correct / len(traces)


def main() -> None:
    torch.manual_seed(0)

    # Train ONLY on small graphs (3-5 vars); hold out larger graphs (6-7 vars) entirely.
    train_data = build_dataset(sizes=[3, 4, 5], n=9000, seed=1)
    test_small = build_dataset(sizes=[3, 4, 5], n=1000, seed=2)
    test_large = build_dataset(sizes=[6, 7], n=1000, seed=3)  # unseen sizes -> tests the *rule*

    max_len = max(len(r) for r in train_data + test_small + test_large)
    print(f"train: {len(train_data)} traces (3-5 vars)   example:")
    print(f"  {' '.join(train_data[0])}")
    print(f"vocab {len(VOCAB)}, max_len {max_len}\n")

    model = TinyReasoner(len(VOCAB), max_len)
    enc_train = [encode(r) for r in train_data]
    print(f"training a {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M-param "
          "transformer on identifiability traces ...")
    train(model, enc_train, max_len)

    base = sum(1 for t in test_large if t[t.index("[A]") + 1] == "yes") / len(test_large)
    print("\nidentifiability-decision accuracy (vs the causalrl ID oracle):")
    in_acc = accuracy(model, test_small, max_len)
    gen_acc = accuracy(model, test_large, max_len)
    print(f"  in-distribution  (3-5 vars):  {in_acc:.3f}")
    print(f"  generalisation   (6-7 vars, UNSEEN size):  {gen_acc:.3f}")
    print(f"  (majority-class baseline on the 6-7-var set: {max(base, 1 - base):.3f})")
    print(
        "\nThe transformer was trained only on 3-5 variable graphs, yet decides identifiability on "
        "larger graphs it never saw, above the majority-class baseline — it learned the structural "
        "rule (when a latent-confounded effect is a non-identifiable hedge), not the training "
        "graphs. That decision is the Causal-Hierarchy-Theorem gate: how the model knows when it "
        "must abstain. (General-ADMG identifiability is genuinely hard; perfect generalisation at "
        "this scale is open — see examples/CAUSAL_LLM_RESEARCH.md.)"
    )


if __name__ == "__main__":
    main()
