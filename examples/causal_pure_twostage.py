# STATUS: canonical-negative · multi-seed (3 seeds, both capacities) · Act 4 Coupling — the MISSING CELL: decoupled schedule for the PURE LM  ·  map: CAUSAL_LLM.md
"""Can a real LM internalise causal reasoning *in its own weights* if we fix the training SCHEDULE?

The branch's canonical negative is "a real GPT-2 does NOT internalise the causal computation"
(``causal_pure_lm.py``: ~0.37-0.53) and "CoT doesn't help" (``causal_reasoning_scaffold.py``: CoT
0.655 vs 0.818 when *given* the structure). But BOTH were trained under the **joint** schedule --
one loss over structure and answer at once -- and Phase D (``causal_corr2cause_mechanism.py``)
independently showed the joint schedule is the deficient one (+0.22 F1 for decoupling). The fix that
worked (``causal_hybrid_twostage.py``, 1.000 confounded) used an **external GNN**, so the answer
never came from the LM's weights. That leaves one cell of the 2x2 unrun:

                        |  joint schedule           |  decoupled schedule
    external module     |  0.43 (hybrid_learned)    |  1.000 (hybrid_twostage)
    PURE LM weights     |  0.37-0.53 / CoT 0.655    |  <-- THIS SCRIPT

Four arms, ONE architecture (GPT2LMHeadModel), same capacity, same number of gradient steps, same
init. The answer is always the LM's own next token -- no external reasoner, nothing hand-coded.
Essentially only the *loss mask* differs, i.e. only the schedule:

  DIRECT     ``prose ? <ans>``                     -- loss on the answer. No scratchpad.
  JOINT      ``prose ? <g> graph </g> <ans>``      -- loss on graph AND answer together, so the
             answer's gradient can shortcut straight from the prose (this is the scaffold's CoT).
  JOINT-2X   the JOINT mask, but on twice the examples (a second freshly sampled batch of graphs)
             so it MATCHES DECOUPLED's item count and structural diversity. This is the control
             for "decoupled just sees more/more-varied structure->answer pairs"; if DECOUPLED
             still wins, the schedule is what matters, not the data.
  DECOUPLED  same template, two interleaved loss masks that never connect prose to the answer:
               * perception batches: TRUE graph shown, loss on the **graph tokens only**;
               * reasoning batches:  a RANDOM graph shown with the answer **recomputed from that
                 shown graph**, loss on the **answer token only**.
             Because the shown graph is randomized, the prose is uninformative for the answer, so
             the answer computation is forced to route through the emitted structure. This is the
             in-weights analogue of Stage B "train the reasoner on clean structure" -- the template
             is identical to JOINT so the two halves compose at test time.

At test every arm is teacher-free: the model emits its own graph, then its own answer.

Honest metric note: the ``confounded`` set (correlated but NOT causal) is **all-negative**, so a
constant-"no" model scores 1.000 on it. We therefore ALWAYS report the balanced ``cause`` query
alongside it; only high on *both* is meaningful. We also report the self-generated graph's edge F1
(to localize any failure to perception vs reasoning) and a teacher-forced ceiling (answer read with
the TRUE graph inserted) -- the in-model analogue of the scaffold's struct-only 0.818.

CPU-sized (4 GPT-2 trainings per seed).  Run::

    SEEDS=0 FAST=1 uv run --extra torch python examples/causal_pure_twostage.py   # smoke
    uv run --extra torch python examples/causal_pure_twostage.py                  # 2 seeds
"""

from __future__ import annotations

import os
import random
import statistics
import sys

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy  # SCM + prose generator, shared by the whole Act-4 thread

NE = hy.NE
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
FAST = os.environ.get("FAST") == "1"
ARMS = [a for a in os.environ.get("ARMS", "direct,joint,joint2x,decoupled").split(",") if a]
# Capacity knobs -- used to test whether a weak STRUCTONLY is an architectural limit or just a
# too-small model (raise these and see whether reachability-over-tokens actually improves).
LAYERS = int(os.environ.get("LAYERS", "4"))
EMBD = int(os.environ.get("EMBD", "128"))
HEADS = int(os.environ.get("HEADS", "4"))

# Vocab = the shared Act-4 vocab, EXTENDED with scratchpad punctuation. The base words keep their
# ids (list prefix is preserved), so examples built by ``hy`` can be reused verbatim.
WORDS = hy.WORDS + ["<g>", "</g>", "->", ";"]
V = {w: i for i, w in enumerate(WORDS)}
GO, GC, ARROW, SEMI = V["<g>"], V["</g>"], V["->"], V[";"]
YES, NO, PAD = V["yes"], V["no"], V["<pad>"]
MAXLEN = 96
IGN = -100  # cross-entropy ignore_index


def gpt2():
    return GPT2Config(
        vocab_size=len(WORDS),
        n_positions=MAXLEN + 2,
        n_ctx=MAXLEN + 2,
        n_embd=EMBD,
        n_layer=LAYERS,
        n_head=HEADS,
        bos_token_id=PAD,
        eos_token_id=PAD,
    )


# --------------------------------------------------------------------------- SCM helpers
def labels_from(adj, k, sx, sy, is_causal):
    """Recompute (cause, corr, label) from an ARBITRARY adjacency -- hy.make's logic, reused."""
    cause = hy.reachable(adj, sx, sy)
    desc = {s: {t for t in range(k) if t != s and hy.reachable(adj, s, t)} for s in range(k)}
    common = any(z not in (sx, sy) and sx in desc[z] and sy in desc[z] for z in range(k))
    corr = cause or hy.reachable(adj, sy, sx) or common
    return int(cause), int(corr), int(cause if is_causal else corr)


def rand_adj(k, rng):
    """A random DAG over the k occupied slots (hy.make's process, random topological order)."""
    order = list(range(k))
    rng.shuffle(order)
    adj = [[0] * NE for _ in range(NE)]
    for a in range(k):
        for b in range(a + 1, k):
            if rng.random() < 0.5:
                adj[order[a]][order[b]] = 1
    return adj


def graph_tokens(adj, entw, k):
    """Render an adjacency as ``u -> v ;`` tokens, canonical slot order (deterministic target)."""
    out = []
    for u in range(k):
        for v in range(k):
            if adj[u][v]:
                out += [entw[u], ARROW, entw[v], SEMI]
    return out


def parse_graph(toks, entw, k):
    """Inverse of graph_tokens: recover an adjacency from generated tokens (robust to junk)."""
    slot = {e: s for s, e in enumerate(entw[:k])}
    adj = [[0] * NE for _ in range(NE)]
    i = 0
    while i + 2 < len(toks):
        u, arr, v = toks[i], toks[i + 1], toks[i + 2]
        if arr == ARROW and u in slot and v in slot and slot[u] != slot[v]:
            adj[slot[u]][slot[v]] = 1
            i += 4 if i + 3 < len(toks) and toks[i + 3] == SEMI else 3
        else:
            i += 1
    return adj


# --------------------------------------------------------------------------- sequence construction
def seq_direct(e):
    """``prose ? <ans>`` -- supervise the answer only."""
    ids = list(e["ids"]) + [YES if e["label"] else NO]
    sup = [False] * (len(ids) - 1) + [True]
    return ids, sup


def split_query(e):
    """Split ``e['ids']`` into (prose, query). The query starts at the 'does'/'are' token."""
    ids = list(e["ids"])
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] in (V["does"], V["are"]):
            return ids[:i], ids[i:]
    return [], ids


def seq_structonly(e):
    """``query <g> TRUE graph </g> <ans>`` -- NO prose at all, so the answer can ONLY come from
    reachability over the graph tokens. The in-model analogue of the scaffold's struct-only 0.818,
    and the decisive isolation of DECOUPLED's failing component."""
    k = sum(e["present"])
    _prose, query = split_query(e)
    g = graph_tokens(e["adj"], e["entw"], k)
    ids = query + [GO] + g + [GC] + [YES if e["label"] else NO]
    sup = [False] * (len(ids) - 1) + [True]
    return ids, sup


def seq_scratch(e, adj, label, sup_graph, sup_ans):
    """``prose ? <g> graph </g> <ans>`` with selectable supervision over each region."""
    k = sum(e["present"])
    g = graph_tokens(adj, e["entw"], k)
    ids = list(e["ids"]) + [GO] + g + [GC] + [YES if label else NO]
    sup = [False] * len(e["ids"])  # prose + query: never supervised
    sup += [False]  # the <g> marker itself
    sup += [sup_graph] * (len(g) + 1)  # graph tokens and the closing </g>
    sup += [sup_ans]  # the answer
    return ids, sup


def pack(seqs):
    """Left-aligned pack -> (input_ids, attn, labels) for next-token CE with ignore_index."""
    width = max(len(s[0]) for s in seqs)
    n = len(seqs)
    ids = torch.full((n, width), PAD, dtype=torch.long)
    attn = torch.zeros(n, width, dtype=torch.long)
    lab = torch.full((n, width), IGN, dtype=torch.long)
    for j, (s, sup) in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
        attn[j, : len(s)] = 1
        for t, flag in enumerate(sup):
            if flag:
                lab[j, t] = s[t]
    # next-token shift: predict position t from t-1
    return ids[:, :-1], attn[:, :-1], lab[:, 1:]


def lm_loss(model, ids, attn, lab):
    logits = model(input_ids=ids, attention_mask=attn).logits
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=IGN
    )


# --------------------------------------------------------------------------- the three schedules
def build_corpus(arm, data, seed):
    """Materialize (ids, supervision-mask) pairs for one arm. Same examples, different masks."""
    rng = random.Random(seed + 7)
    out = []
    for e in data:
        k = sum(e["present"])
        if arm == "structonly":
            out.append(seq_structonly(e))
        elif arm == "direct":
            out.append(seq_direct(e))
        elif arm == "joint":
            out.append(seq_scratch(e, e["adj"], e["label"], sup_graph=True, sup_ans=True))
        else:  # decoupled -- each example yields BOTH a perception and a reasoning item
            out.append(seq_scratch(e, e["adj"], e["label"], sup_graph=True, sup_ans=False))
            # Balance the reasoning corpus 50/50: resample the shown graph until its recomputed
            # label hits an alternating target, so no constant answer can score well.
            want = rng.randint(0, 1)
            radj, rlab = None, None
            for _ in range(40):
                cand = rand_adj(k, rng)
                _, _, lab = labels_from(cand, k, e["xs"], e["ys"], e["is_causal"])
                radj, rlab = cand, lab
                if lab == want:
                    break
            out.append(seq_scratch(e, radj, rlab, sup_graph=False, sup_ans=True))
    return out


def train(model, corpus, epochs, lr=5e-4, bs=32, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    for ep in range(epochs):
        rng.shuffle(corpus)
        tot = nb = 0.0
        for i in range(0, len(corpus), bs):
            ids, attn, lab = pack(corpus[i : i + bs])
            loss = lm_loss(model, ids, attn, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"    [{tag}] epoch {ep + 1}/{epochs}  loss {tot / nb:.4f}", flush=True)


# ----------------------------------------------------------------------- teacher-free evaluation
@torch.no_grad()
def generate(model, data, max_new=30):
    """Teacher-free: emit the graph, then read the answer. Grouped by prompt length (no padding)."""
    res = {}
    order = sorted(range(len(data)), key=lambda i: len(data[i]["ids"]))
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(len(data[i]["ids"]), []).append(i)
    for _plen, idxs in groups.items():
        prompt = torch.tensor([list(data[i]["ids"]) + [GO] for i in idxs])
        cur = prompt
        done = torch.zeros(len(idxs), dtype=torch.bool)
        gen = [[] for _ in idxs]
        for _ in range(max_new):
            logits = model(input_ids=cur).logits[:, -1, :]
            nxt = logits.argmax(-1)
            for j in range(len(idxs)):
                if not done[j]:
                    t = int(nxt[j])
                    if t == GC:
                        done[j] = True
                    else:
                        gen[j].append(t)
            if bool(done.all()):
                break
            # keep un-finished rows moving; finished rows are padded with </g> (inert)
            step = torch.where(done, torch.full_like(nxt, GC), nxt)
            cur = torch.cat([cur, step.unsqueeze(1)], 1)
        # close every sequence and read the answer logits at the position after </g>
        cur = torch.cat([cur, torch.full((len(idxs), 1), GC, dtype=torch.long)], 1)
        final = model(input_ids=cur).logits[:, -1, :]
        margin = final[:, YES] - final[:, NO]
        for j, i in enumerate(idxs):
            res[i] = (float(margin[j]), gen[j])
    return [res[i] for i in range(len(data))]


@torch.no_grad()
def acc_direct(model, data) -> float:
    """No scratchpad: read the answer token immediately after the prompt."""
    if not data:
        return float("nan")
    seqs = [(list(e["ids"]), [False] * len(e["ids"])) for e in data]
    width = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), width), PAD, dtype=torch.long)
    attn = torch.zeros(len(seqs), width, dtype=torch.long)
    last = torch.tensor([len(s[0]) - 1 for s in seqs])
    for j, (s, _) in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
        attn[j, : len(s)] = 1
    logits = model(input_ids=ids, attention_mask=attn).logits
    row = logits[torch.arange(len(seqs)), last]
    m = row[:, YES] - row[:, NO]
    lab = torch.tensor([float(e["label"]) for e in data])
    return float(((m > 0) == (lab > 0.5)).float().mean())


def acc_generated(model, data) -> tuple[float, float]:
    """Teacher-free accuracy + edge F1 of the graph the model wrote for itself."""
    if not data:
        return float("nan"), float("nan")
    outs = generate(model, data)
    ok = 0
    tp = fp = fn = 0
    for e, (margin, toks) in zip(data, outs, strict=True):
        ok += int((margin > 0) == (e["label"] > 0.5))
        k = sum(e["present"])
        pred = parse_graph(toks, e["entw"], k)
        for u in range(k):
            for v in range(k):
                if u == v:
                    continue
                p, t = pred[u][v], e["adj"][u][v]
                tp += p and t
                fp += p and not t
                fn += t and not p
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return ok / len(data), f1


@torch.no_grad()
def acc_teacher_forced(model, data) -> float:
    """Ceiling: insert the TRUE graph, read the answer (the in-model 'struct-only' condition)."""
    if not data:
        return float("nan")
    seqs = []
    for e in data:
        k = sum(e["present"])
        ids = list(e["ids"]) + [GO] + graph_tokens(e["adj"], e["entw"], k) + [GC]
        seqs.append((ids, [False] * len(ids)))
    width = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), width), PAD, dtype=torch.long)
    attn = torch.zeros(len(seqs), width, dtype=torch.long)
    last = torch.tensor([len(s[0]) - 1 for s in seqs])
    for j, (s, _) in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
        attn[j, : len(s)] = 1
    logits = model(input_ids=ids, attention_mask=attn).logits
    row = logits[torch.arange(len(seqs)), last]
    m = row[:, YES] - row[:, NO]
    lab = torch.tensor([float(e["label"]) for e in data])
    return float(((m > 0) == (lab > 0.5)).float().mean())


@torch.no_grad()
def acc_structonly(model, data) -> float:
    """Answer read after ``query <g> TRUE graph </g>`` with the prose removed entirely."""
    if not data:
        return float("nan")
    seqs = [seq_structonly(e) for e in data]
    prompts = [s[0][:-1] for s in seqs]  # drop the gold answer token
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), PAD, dtype=torch.long)
    attn = torch.zeros(len(prompts), width, dtype=torch.long)
    last = torch.tensor([len(p) - 1 for p in prompts])
    for j, p in enumerate(prompts):
        ids[j, : len(p)] = torch.tensor(p)
        attn[j, : len(p)] = 1
    logits = model(input_ids=ids, attention_mask=attn).logits
    row = logits[torch.arange(len(prompts)), last]
    m = row[:, YES] - row[:, NO]
    lab = torch.tensor([float(e["label"]) for e in data])
    return float(((m > 0) == (lab > 0.5)).float().mean())


def confounded(data):
    """Correlated but NOT causal, asked causally. NB: all labels are 0 (see module docstring)."""
    return [dict(e, is_causal=1, label=e["cause"]) for e in data if e["corr"] and not e["cause"]]


# --------------------------------------------------------------------------- driver
def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    n = 2000 if FAST else 8000
    epochs = 4 if FAST else 12
    nt = 400 if FAST else 1500

    train_data = hy.build(n, sizes=[2, 3], seed=seed)
    t3 = hy.build(nt, [3], seed + 50)
    t4 = hy.build(nt, [4], seed + 60)
    c3, c4 = confounded(t3), confounded(t4)
    cause3 = [e for e in t3 if e["is_causal"]]
    cause4 = [e for e in t4 if e["is_causal"]]

    # JOINT-2X control: the joint schedule with DECOUPLED's item count and structural diversity
    # (a second, freshly sampled batch of graphs). Rules out "decoupled just sees more/more-varied
    # structure->answer pairs" -- if DECOUPLED still wins, the schedule is what matters.
    extra = hy.build(n, sizes=[2, 3], seed=seed + 900)

    out: dict[str, float] = {}
    for arm in ARMS:
        torch.manual_seed(seed)  # identical init across arms
        model = GPT2LMHeadModel(gpt2())
        src = train_data + extra if arm == "joint2x" else train_data
        corpus = build_corpus("joint" if arm == "joint2x" else arm, src, seed)
        # Fairness unit = epochs of supervision PER OBJECTIVE, not gradient steps. JOINT carries
        # both losses on every item, so step-matching would hand DECOUPLED only half the answer
        # supervision (that is exactly what sank the step-matched run: teacher-forced ceiling 0.596
        # vs JOINT 0.871 -- see results/pure_twostage_stepmatched_s0.log). Every arm therefore gets
        # `epochs` passes of each objective; DECOUPLED and JOINT-2X take 2x the gradient steps as a
        # consequence, and JOINT-2X is the step-matched control that *also* gets 2x supervision --
        # a deliberately conservative baseline for DECOUPLED to have to beat.
        ep = epochs
        print(f"\n  training {arm.upper()} ({len(corpus)} items x {ep} epochs) ...", flush=True)
        train(model, corpus, epochs=ep, tag=arm)
        model.eval()

        if arm == "structonly":
            out["structonly_conf_s3"] = acc_structonly(model, c3)
            out["structonly_conf_s4"] = acc_structonly(model, c4)
            out["structonly_cause_s3"] = acc_structonly(model, cause3)
            out["structonly_cause_s4"] = acc_structonly(model, cause4)
        elif arm == "direct":
            out["direct_conf_s3"] = acc_direct(model, c3)
            out["direct_conf_s4"] = acc_direct(model, c4)
            out["direct_cause_s3"] = acc_direct(model, cause3)
            out["direct_cause_s4"] = acc_direct(model, cause4)
        else:
            a3, f3 = acc_generated(model, c3)
            a4, f4 = acc_generated(model, c4)
            g3, _ = acc_generated(model, cause3)
            g4, _ = acc_generated(model, cause4)
            out[f"{arm}_conf_s3"] = a3
            out[f"{arm}_conf_s4"] = a4
            out[f"{arm}_cause_s3"] = g3
            out[f"{arm}_cause_s4"] = g4
            out[f"{arm}_edgef1_s3"] = f3
            out[f"{arm}_edgef1_s4"] = f4
            out[f"{arm}_tf_cause_s3"] = acc_teacher_forced(model, cause3)
            out[f"{arm}_tf_cause_s4"] = acc_teacher_forced(model, cause4)
    return out


def main() -> None:
    print("PURE LM, decoupled vs joint schedule -- the missing cell of the 2x2")
    print(f"  seeds {SEEDS}{'  [FAST smoke]' if FAST else ''}; answer is always the LM's own token")
    rows = [run_seed(s) for s in SEEDS]

    def agg(key):
        vals = [r[key] for r in rows if key in r]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), sd

    print("\n" + "=" * 78)
    print("  confounded = correlated but NOT causal (all-negative set: constant-'no' scores 1.000)")
    print("  cause      = balanced causal query (constant-'no' scores ~0.500)")
    print("  BOTH must be high for the result to mean anything.\n")
    print(f"  {'arm':<12}{'conf s3':>16}{'conf s4':>16}{'cause s3':>16}{'cause s4':>16}")
    for arm in ARMS:
        cells = []
        for key in ("conf_s3", "conf_s4", "cause_s3", "cause_s4"):
            m, sd = agg(f"{arm}_{key}")
            cells.append(f"{m:.3f}+/-{sd:.3f}")
        print(f"  {arm.upper():<12}" + "".join(f"{c:>16}" for c in cells))

    print("\n  self-generated graph edge F1 (is a failure perception, or reasoning?)")
    for arm in [a for a in ARMS if a not in ("direct", "structonly")]:
        m3, s3 = agg(f"{arm}_edgef1_s3")
        m4, s4 = agg(f"{arm}_edgef1_s4")
        print(f"  {arm.upper():<12}  s3 {m3:.3f}+/-{s3:.3f}   s4 {m4:.3f}+/-{s4:.3f}")

    print("\n  teacher-forced ceiling on `cause` (TRUE graph inserted -> answer)")
    for arm in [a for a in ARMS if a not in ("direct", "structonly")]:
        m3, s3 = agg(f"{arm}_tf_cause_s3")
        m4, s4 = agg(f"{arm}_tf_cause_s4")
        print(f"  {arm.upper():<12}  s3 {m3:.3f}+/-{s3:.3f}   s4 {m4:.3f}+/-{s4:.3f}")

    print("\n  reference: external-module route -- joint 0.43 vs two-stage 1.000/0.933 confounded")
    print(
        "\n  Reading: ONE GPT-2, same init/data/capacity/gradient-steps; only the loss mask (the\n"
        "  schedule) differs, and the answer is always the LM's own token. If DECOUPLED clears\n"
        "  JOINT and DIRECT on confounded WHILE holding `cause` high, then the branch's canonical\n"
        "  negative ('a real LM does not internalise causal reasoning') was a JOINT-schedule\n"
        "  artifact, not a capability limit -- causal competence is installable in an LM's own\n"
        "  weights by training it differently, with no external reasoner. If DECOUPLED does NOT\n"
        "  clear them, the external routed module is genuinely load-bearing and the negative\n"
        "  stands on its own -- which sharpens the thesis rather than weakening it.",
    )


if __name__ == "__main__":
    main()
