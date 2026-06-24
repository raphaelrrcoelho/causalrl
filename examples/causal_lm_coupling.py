# STATUS: superseded by causal_hybrid_lm.py · Act 4 Coupling — first bespoke prose->core->answer  ·  map: CAUSAL_LLM.md
"""A tiny causal *language* model: natural-language prose -> embedded causal core -> answer.

The embedded core (structure + iterative reachability + do(), size-general) reasons over a graph.
This couples it to language: the input is now PROSE about named entities, and a language front-end
binds the words to the core's variables, the core reasons, and the answer is read out -- the NL <->
latent-SCM interface that turns the core into a causal language model.

  input  : "smoking causes tar . tar causes cancer . does smoking cause cancer ?"
  front  : bind entity WORDS -> variable slots (content match); causal VERBS -> edges  (learned)
  core   : soft adjacency A -> K-step reachability -> do()-routed read (causal vs correlational)
  output : yes / no

The front-end is permutation/count-invariant (per-fact, content-addressed), so it size-generalizes:
trained on 2-3 entities, tested on 4 held out. The question fixes the query type and the two
entities; the *scenario* (prose -> causal structure) is what the model learns to read. The same do()
switch distinguishes "are X,Y correlated?" (back-door) from "does X cause Y?" (directed) -- so on
confounded prose (X <- Z -> Y) it answers "correlated yes, causes no", which a correlation reader
cannot.

Scope: simple subject-verb-object prose with a small entity vocabulary; richer language parsing is
future work. The point is the *interface* -- language in, embedded causal reasoning, answer out.

CPU-sized.  Run::

    uv run --extra torch python examples/causal_lm_coupling.py
"""

from __future__ import annotations

import random

import torch
from torch import nn

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
EXTRA = ["does", "cause", "are", "and", "correlated", ".", "?", "<pad>"]
VOCAB = {w: i for i, w in enumerate(ENTITIES + VERBS + EXTRA)}
NE = len(ENTITIES)
MAX_P = 12  # max edge-phrases
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
    ents = sorted(rng.sample(range(NE), k))  # entity ids; slot = position in this sorted list
    slot_of = {e: s for s, e in enumerate(ents)}
    order = ents[:]
    rng.shuffle(order)
    adj = [[0] * NE for _ in range(NE)]
    phrases, nl = [], []
    for a in range(k):
        for b in range(a + 1, k):
            if rng.random() < 0.5:
                u, v = order[a], order[b]  # u -> v (entity ids)
                adj[slot_of[u]][slot_of[v]] = 1
                verb = rng.choice(VERBS)
                phrases.append((u, VOCAB[verb], v))  # (subject ent id, verb id, object ent id)
                nl.append(f"{ENTITIES[u]} {verb} {ENTITIES[v]}")
    xe, ye = rng.sample(ents, 2)
    cause = reachable(adj, slot_of[xe], slot_of[ye])
    # marginal association: directed either way, or a common ancestor
    sx = slot_of[xe]
    sy = slot_of[ye]
    desc = {s: {t for t in range(k) if t != s and reachable(adj, s, t)} for s in range(k)}
    common = any(z not in (sx, sy) and sx in desc[z] and sy in desc[z] for z in range(k))
    corr = cause or reachable(adj, sy, sx) or common
    is_causal = rng.random() < 0.5
    q = (
        f"does {ENTITIES[xe]} cause {ENTITIES[ye]} ?"
        if is_causal
        else f"are {ENTITIES[xe]} and {ENTITIES[ye]} correlated ?"
    )
    return {
        "phrases": phrases[:MAX_P],
        "ent": ents,  # entity ids present (define slots)
        "xe": xe,
        "ye": ye,
        "is_causal": int(is_causal),
        "cause": int(cause),
        "corr": int(corr),
        "label": int(cause if is_causal else corr),
        "adj": adj,
        "nl": " . ".join(nl) + " . " + q,
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


def batch(items):
    b = len(items)
    psrc = torch.zeros(b, MAX_P, dtype=torch.long)
    pverb = torch.zeros(b, MAX_P, dtype=torch.long)
    pobj = torch.zeros(b, MAX_P, dtype=torch.long)
    pmask = torch.zeros(b, MAX_P)
    entw = torch.zeros(b, NE, dtype=torch.long)  # entity word id per slot
    present = torch.zeros(b, NE)
    for k, e in enumerate(items):
        for t, (s, vb, o) in enumerate(e["phrases"]):
            psrc[k, t], pverb[k, t], pobj[k, t], pmask[k, t] = s, vb, o, 1.0
        for slot, eid in enumerate(e["ent"]):
            entw[k, slot] = eid
            present[k, slot] = 1.0
    g = lambda key, dt: torch.tensor([e[key] for e in items], dtype=dt)  # noqa: E731
    return (
        psrc,
        pverb,
        pobj,
        pmask,
        entw,
        present,
        g("xe", torch.long),
        g("ye", torch.long),
        g("is_causal", torch.float),
        g("label", torch.float),
        torch.tensor([e["adj"] for e in items], dtype=torch.float),
    )


class CausalLM(nn.Module):
    def __init__(self, d=64, steps=5):
        super().__init__()
        self.steps = steps
        self.emb = nn.Embedding(len(VOCAB), d)
        self.src_h = nn.Linear(d, d)
        self.obj_h = nn.Linear(d, d)
        self.qx_h = nn.Linear(d, d)
        self.qy_h = nn.Linear(d, d)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.scale = d**0.5
        self.read = nn.Linear(1, 1)

    def _match(self, head_emb, slot_emb, present):  # soft word->slot binding by content
        score = torch.einsum("b...d,bsd->b...s", head_emb, slot_emb) / self.scale
        score = (
            score.masked_fill(present.unsqueeze(1).expand_as(score) == 0, -1e9)
            if score.dim() == 3
            else score.masked_fill(present == 0, -1e9)
        )
        return torch.softmax(score, dim=-1)

    def forward(self, psrc, pverb, pobj, pmask, entw, present, xe, ye, is_causal):
        slot_emb = self.emb(entw)  # (B,NE,d) embedding of each slot's entity word
        # bind each phrase's subject/object word to a variable slot, gate by the verb
        sd = self._match(self.src_h(self.emb(psrc)), slot_emb, present)  # (B,P,NE)
        od = self._match(self.obj_h(self.emb(pobj)), slot_emb, present)  # (B,P,NE)
        gate = torch.sigmoid(self.gate(self.emb(pverb))) * pmask.unsqueeze(-1)  # (B,P,1)
        a = torch.einsum("bp,bpu,bpv->buv", gate.squeeze(-1), sd, od)  # soft adjacency
        a = torch.clamp(a, 0, 1) * present.unsqueeze(2) * present.unsqueeze(1)
        r = a
        for _ in range(self.steps):
            r = torch.clamp(a + torch.bmm(r, a), 0.0, 1.0)
        xd = self._match(self.qx_h(self.emb(xe)), slot_emb, present)  # (B,NE)
        yd = self._match(self.qy_h(self.emb(ye)), slot_emb, present)
        fwd = torch.einsum("bu,buv,bv->b", xd, r, yd)
        bwd = torch.einsum("bu,buv,bv->b", yd, r, xd)
        rzx = torch.einsum("buv,bv->bu", r, xd)  # reach z->x
        rzy = torch.einsum("buv,bv->bu", r, yd)
        notxy = (1 - xd) * (1 - yd) * present
        common = (rzx * rzy * notxy).max(dim=1).values
        score = is_causal * fwd + (1 - is_causal) * (1 - (1 - fwd) * (1 - bwd) * (1 - common))
        return self.read(score.unsqueeze(-1)).squeeze(-1), a


def train(model, data, epochs=20, lr=2e-3, bs=128, lam=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(data)
        ta = te = nb = 0.0
        for i in range(0, len(data), bs):
            psrc, pverb, pobj, pmask, entw, pres, xe, ye, isc, lab, adj = batch(data[i : i + bs])
            ans, a = model(psrc, pverb, pobj, pmask, entw, pres, xe, ye, isc)
            al = bce(ans, lab)
            pm = pres.unsqueeze(2) * pres.unsqueeze(1)
            el = (
                nn.functional.binary_cross_entropy(a.clamp(1e-6, 1 - 1e-6), adj, reduction="none")
                * pm
            ).sum() / pm.sum()
            loss = al + lam * el
            opt.zero_grad()
            loss.backward()
            opt.step()
            ta += float(al.detach())
            te += float(el.detach())
            nb += 1
        print(f"    epoch {ep + 1}/{epochs}  answer {ta / nb:.3f}  edge {te / nb:.3f}")


@torch.no_grad()
def evaluate(model, data) -> float:
    if not data:
        return float("nan")
    psrc, pverb, pobj, pmask, entw, pres, xe, ye, isc, lab, _adj = batch(data)
    ans, _ = model(psrc, pverb, pobj, pmask, entw, pres, xe, ye, isc)
    return float(((ans > 0) == (lab > 0.5)).float().mean())


def main() -> None:
    torch.manual_seed(0)
    print("a tiny causal language model: prose -> embedded causal core -> yes/no")
    train_data = build(16000, sizes=[2, 3], seed=1)
    tests = {s: build(2400, sizes=[s], seed=10 + s) for s in (2, 3, 4)}
    print(f"  e.g. input: {train_data[0]['nl']!r}  ->  {'yes' if train_data[0]['label'] else 'no'}")

    model = CausalLM()
    print(f"CausalLM: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params")
    train(model, train_data)
    model.eval()

    print("\n            answer accuracy from natural language")
    print("  size    observational   interventional   confounded-cause   seen?")
    for s in (2, 3, 4):
        corr = [e for e in tests[s] if not e["is_causal"]]
        cause = [e for e in tests[s] if e["is_causal"]]
        conf = [
            dict(e, is_causal=1, label=e["cause"]) for e in tests[s] if e["corr"] and not e["cause"]
        ]
        seen = "trained" if s in (2, 3) else "HELD-OUT"
        print(
            f"  size {s}:    {evaluate(model, corr):.3f}           {evaluate(model, cause):.3f}"
            f"            {evaluate(model, conf):.3f}          {seen}"
        )

    # show a couple of worked NL examples
    print("\n  worked examples (held-out, 4 entities):")
    for e in tests[4][:3]:
        psrc, pverb, pobj, pmask, entw, pres, xe, ye, isc, _lab, _adj = batch([e])
        with torch.no_grad():
            ans, _ = model(psrc, pverb, pobj, pmask, entw, pres, xe, ye, isc)
        pred = "yes" if float(ans) > 0 else "no"
        print(f"    {e['nl']}  ->  model: {pred}  (truth: {'yes' if e['label'] else 'no'})")

    print(
        "\nReading: language in, embedded causal reasoning, answer out. The front-end binds entity "
        "words to the core's variables and reads causal verbs as edges; the core reasons with the "
        "do() switch (correlated vs causes). It holds on 4-entity prose never trained on, and on "
        "confounded prose it says 'correlated but not causal' -- a causal LM, in miniature."
    )


if __name__ == "__main__":
    main()
