"""Active causal discovery: choosing interventions to break the Markov-equivalence ceiling.

Observation identifies the graph only up to the Markov-equivalence class: colliders (v-structures)
are oriented, the rest of the skeleton stays undirected -- so causal direction on those edges is
unanswerable from correlation alone (budget 0 below). Interventions orient them, but experiments are
expensive, so *which* interventions you run matters. That is active causal discovery, and it is the
constructive way "beyond correlation": don't just use interventions, *choose* informative ones.

Environment (causalrl-style SCM): a random DAG. Observation gives the CPDAG (collider edges
oriented, the rest undirected). Each intervention do(v) reveals -- and orients -- every
still-undirected edge incident to v. A policy picks a sequence of nodes under a budget; we measure
how much causal structure (and how many causal queries) get resolved per intervention.

Policies compared at each budget:
  * RANDOM   intervene on a random not-yet-used node
  * ACTIVE   intervene on the node incident to the MOST undirected edges (greedy information gain)
  * LEARNED  a small net scores nodes from the current partial structure (trained to imitate ACTIVE)

Headline: ACTIVE resolves the structure (and answers causal queries) with far fewer interventions
than RANDOM, and both rise from the observation-only (budget 0) MEC floor toward 1.0. Choosing
interventions is the lever.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_active_discovery.py
"""

from __future__ import annotations

import random

import torch
from torch import nn

torch.set_num_threads(4)


def gen_dag(n: int, p: float, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    adj = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < p:
                adj[order[a]][order[b]] = 1
    return adj


def adjacent(adj, i, j) -> bool:
    return adj[i][j] == 1 or adj[j][i] == 1


def cpdag(adj, n):
    """Return (skeleton edges, observationally-undirected edges) as sets of frozenset pairs."""
    skel = {frozenset((i, j)) for i in range(n) for j in range(n) if adj[i][j]}
    oriented = set()  # directed edges fixed by colliders
    for k in range(n):
        parents = [p for p in range(n) if adj[p][k]]
        for a in range(len(parents)):
            for b in range(a + 1, len(parents)):
                if not adjacent(
                    adj, parents[a], parents[b]
                ):  # v-structure parents[a]->k<-parents[b]
                    oriented.add((parents[a], k))
                    oriented.add((parents[b], k))
    undirected = {
        e for e in skel if not any((i, j) in oriented for i, j in [tuple(e), tuple(e)[::-1]])
    }
    return skel, undirected


def reach(oriented_adj, i, j, n) -> bool:
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for v in range(n):
            if oriented_adj[u][v] and v not in seen:
                if v == j:
                    return True
                seen.add(v)
                stack.append(v)
    return False


def causal_acc(adj, oriented_adj, n) -> float:
    """Fraction of ordered pairs whose 'does i cause j?' is answered right using oriented edges."""
    ok = tot = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                ok += int(reach(oriented_adj, i, j, n) == reach(adj, i, j, n))
                tot += 1
    return ok / tot


def oriented_from(adj, resolved_undirected, undirected0, n):
    """Build the currently-oriented adjacency: colliders + the undirected edges already resolved."""
    skel, und0 = cpdag(adj, n)
    o = [[0] * n for _ in range(n)]
    for e in skel:
        i, j = tuple(e)
        if e in undirected0 and e not in resolved_undirected:
            continue  # still undirected -> not usable
        # use the true orientation (collider edges, or resolved-by-intervention edges)
        if adj[i][j]:
            o[i][j] = 1
        else:
            o[j][i] = 1
    return o


def node_features(adj, undirected_remaining, intervened, n):
    skel, _ = cpdag(adj, n)
    feats = []
    for v in range(n):
        inc_und = sum(1 for e in undirected_remaining if v in e)
        inc_skel = sum(1 for e in skel if v in e)
        feats.append([inc_und, inc_skel, 1.0 if v in intervened else 0.0])
    return feats


def rollout(adj, n, budget, policy, rng, scorer=None):
    """Run `budget` interventions under a policy; return causal-query accuracy after each step."""
    _, undirected0 = cpdag(adj, n)
    remaining = set(undirected0)
    intervened = set()
    accs = [causal_acc(adj, oriented_from(adj, set(undirected0) - remaining, undirected0, n), n)]
    for _ in range(budget):
        avail = [v for v in range(n) if v not in intervened]
        if not avail:
            accs.append(accs[-1])
            continue
        if policy == "random":
            v = rng.choice(avail)
        elif policy == "active":  # greedy: most incident undirected edges
            v = max(avail, key=lambda u: sum(1 for e in remaining if u in e))
        else:  # learned
            feats = torch.tensor(node_features(adj, remaining, intervened, n), dtype=torch.float)
            scores = scorer(feats).squeeze(-1)
            for u in range(n):
                if u in intervened:
                    scores[u] = -1e9
            v = int(scores.argmax())
        intervened.add(v)
        remaining = {e for e in remaining if v not in e}  # do(v) orients its incident undirected
        accs.append(
            causal_acc(adj, oriented_from(adj, set(undirected0) - remaining, undirected0, n), n)
        )
    return accs


class Scorer(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
        )

    def forward(self, feats):
        return self.net(feats)


def train_scorer(n, rng, steps=3000):
    """Imitate the active (greedy) oracle: pick the node with most incident undirected edges."""
    model = Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(steps):
        adj = gen_dag(n, 0.4, rng)
        _, und0 = cpdag(adj, n)
        if not und0:
            continue
        # a random partial state
        remaining = {e for e in und0 if rng.random() < 0.7}
        intervened = {v for v in range(n) if rng.random() < 0.2}
        feats = torch.tensor(node_features(adj, remaining, intervened, n), dtype=torch.float)
        target = max(range(n), key=lambda u: sum(1 for e in remaining if u in e))
        loss = ce(model(feats).squeeze(-1).unsqueeze(0), torch.tensor([target]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main() -> None:
    torch.manual_seed(0)
    rng = random.Random(0)
    n = 6
    print(f"active causal discovery on random {n}-node DAGs (causal-query accuracy vs budget)\n")
    print("training the LEARNED policy to imitate the active (info-greedy) oracle ...")
    scorer = train_scorer(n, random.Random(1))

    n_graphs, budgets = 600, list(range(0, n + 1))
    agg = {pol: [0.0] * len(budgets) for pol in ("random", "active", "learned")}
    used = 0
    for _ in range(n_graphs):
        adj = gen_dag(n, 0.4, rng)
        _, und0 = cpdag(adj, n)
        if len(und0) < 2:  # only count graphs with real MEC ambiguity to resolve
            continue
        used += 1
        for pol in ("random", "active", "learned"):
            accs = rollout(adj, n, n, pol, rng, scorer)
            for b in budgets:
                agg[pol][b] += accs[b]

    print(f"\n  (averaged over {used} graphs with >=2 undirected edges)")
    print("  budget   random   active   learned")
    for b in budgets:
        r, a, le = (agg[p][b] / used for p in ("random", "active", "learned"))
        tag = "  <- observation only (MEC floor)" if b == 0 else ""
        print(f"    {b}      {r:.3f}    {a:.3f}    {le:.3f}{tag}")

    print(
        "\nReading: budget 0 is the observation-only MEC floor -- causal direction on undirected "
        "edges is unanswerable from correlation. Each intervention orients more; ACTIVE (choose the "
        "most informative experiment) climbs to 1.0 with far fewer interventions than RANDOM, and "
        "the LEARNED policy recovers the active curve. Going beyond correlation efficiently is about "
        "*choosing* interventions -- the active-discovery capability a causal LM needs."
    )


if __name__ == "__main__":
    main()
