# STATUS: canonical · Act 1 Diagnosis — presence != mediation; routing through structure (0.66->0.81)  ·  map: CAUSAL_LLM.md
"""Does *installing* the causal structure improve reasoning?  (closing the extraction gap)

The scaffold experiment (``causal_reasoning_scaffold.py``) localized the bottleneck: the model can
REASON OVER a given causal structure (struct-only ~0.82, near the MEC ceiling 0.84) but cannot
EXTRACT one from raw correlations (direct stuck ~0.66). This script tests the program's core move --
*install* (ground) the causal structure in the representation and see if the reasoner improves.

Mechanism (a grounding objective, in the spirit of Phases 0-3): the SAME hidden state the model uses
to answer (its representation at the readout position) gets a dense auxiliary target -- predict the
CPDAG, one 4-way class per variable pair {none, i>j, j>i, undirected}. The answer is still trained
by the ordinary LM loss; the auxiliary loss forces the answer-representation to encode the causal
structure. The per-edge target is decomposed and easy, unlike the global 1-bit answer signal -- so
if extraction is learnable under representational pressure, grounding should lift the answer.

We compare, all same data / tokenizer / model size:

    * plain DIRECT            (premises -> answer)                          baseline (~0.66)
    * GROUNDED DIRECT         (premises -> answer  +  aux: encode the CPDAG)  the test
    * references: struct-only ~0.82 and MEC ceiling ~0.84 (from the scaffold script)

We also report the auxiliary CPDAG accuracy (did the representation actually learn the structure?).

CPU-sized; reuses the transfer + scaffold generators.  Run::

    uv run --extra torch python examples/causal_grounding_install.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_reasoning_scaffold as scf  # sibling examples, imported after sys.path tweak
import causal_transfer_corr2cause as c2c

YES_TOK, NO_TOK, EOS_TOK = c2c.YES_TOK, c2c.NO_TOK, c2c.EOS_TOK

torch.set_num_threads(4)


def cpdag_labels(nodes: list[str], members) -> list[int]:
    """One 4-way class per pair (fixed order): 0 none, 1 i>j, 2 j>i, 3 undirected."""
    out = []
    for i, j in itertools.combinations(nodes, 2):
        if all(j in g.children(i) for g in members):
            out.append(1)
        elif all(i in g.children(j) for g in members):
            out.append(2)
        elif any(j in g.children(i) or i in g.children(j) for g in members):
            out.append(3)
        else:
            out.append(0)
    return out


def build_examples(n_examples: int, n_vars: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    nodes = c2c.LETTERS[:n_vars]
    specs = c2c.premise_specs(nodes)
    index = c2c.fp_index(nodes)
    want = n_examples // 2
    cnt = {True: 0, False: 0}
    out: list[dict] = []
    tries = 0
    while len(out) < 2 * want and tries < n_examples * 400:
        tries += 1
        g = c2c.random_dag(nodes, p=0.5, rng=rng)
        x, y = rng.sample(nodes, 2)
        label = y in g.descendants(x)
        if cnt[label] >= want:
            continue
        members = index[c2c.fingerprint(g, specs)]
        facts = []
        for i, j, z in specs:
            rel = "independent" if c2c.d_separated(g, {i}, {j}, set(z)) else "correlated"
            facts.append(
                f"{i} and {j} are {rel} given {' '.join(sorted(z))}"
                if z
                else f"{i} and {j} are {rel}"
            )
        out.append(
            {
                "premises": " . ".join(facts),
                "nodes": nodes,
                "x": x,
                "y": y,
                "label": label,
                "cpdag": scf.cpdag_str(nodes, members),
                "pairlab": cpdag_labels(nodes, members),
                "corr": not c2c.d_separated(g, {x}, {y}, set()),
                "oracle": sum(y in m.descendants(x) for m in members) >= len(members) / 2,
            }
        )
        cnt[label] += 1
    rng.shuffle(out)
    return out


def _batch(items, pad_id):
    width = max(len(s) for s, _, _ in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for j, (s, _, _) in enumerate(items):
        ids[j, : len(s)] = torch.tensor(s)
        mask[j, : len(s)] = 1
    read = torch.tensor([r for _, r, _ in items])
    lab = torch.tensor([pl for _, _, pl in items])
    return ids, mask, read, lab


def train_grounded(model, aux_head, tok, data, epochs=12, lr=5e-4, lam=1.0, batch_size=64):
    """LM answer loss + auxiliary 'encode the CPDAG at the answer-representation' loss."""
    pad_id = tok.pad_token_id
    items = []
    for e in data:
        full = tok(scf.text_direct(e) + EOS_TOK).input_ids
        plen = len(tok(scf.p_direct(e)).input_ids)
        items.append((full, plen - 1, e["pairlab"]))  # readout = last prompt token ("?")
    opt = torch.optim.AdamW([*model.parameters(), *aux_head.parameters()], lr=lr)
    ce = nn.CrossEntropyLoss()
    rng = random.Random(0)
    for epoch in range(epochs):
        rng.shuffle(items)
        t_lm = t_aux = 0.0
        nb = 0
        for i in range(0, len(items), batch_size):
            ids, mask, read, lab = _batch(items[i : i + batch_size], pad_id)
            labels = ids.masked_fill(mask == 0, -100)
            out = model(
                input_ids=ids, attention_mask=mask, labels=labels, output_hidden_states=True
            )
            h = out.hidden_states[-1][torch.arange(ids.size(0)), read]  # answer-representation
            aux_logits = aux_head(h).view(ids.size(0), lab.size(1), 4)
            aux_loss = ce(aux_logits.reshape(-1, 4), lab.reshape(-1))
            loss = out.loss + lam * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_lm += float(out.loss.detach())
            t_aux += float(aux_loss.detach())
            nb += 1
        print(f"    epoch {epoch + 1}/{epochs}  lm {t_lm / nb:.3f}  aux {t_aux / nb:.3f}")


def classes_to_cpdag(nodes: list[str], classes) -> str:
    parts = []
    for (i, j), c in zip(itertools.combinations(nodes, 2), classes, strict=True):
        if c == 1:
            parts.append(f"{i} > {j}")
        elif c == 2:
            parts.append(f"{j} > {i}")
        elif c == 3:
            parts.append(f"{i} = {j}")
    return " ".join(parts) if parts else "none"


@torch.no_grad()
def pipeline_acc(
    grounded, aux_head, struct_model, tok, data, yes_id, no_id, batch_size=128
) -> float:
    """End-to-end from premises: grounded model EXTRACTS the CPDAG, struct model REASONS over it."""
    pad_id = tok.pad_token_id
    items = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        enc = [tok(scf.p_direct(e)).input_ids for e in batch]
        width = max(len(s) for s in enc)
        ids = torch.full((len(enc), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        read = torch.tensor([len(s) - 1 for s in enc])
        for j, s in enumerate(enc):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        h = grounded(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[
            -1
        ]
        n_pairs = len(data[0]["nodes"]) * (len(data[0]["nodes"]) - 1) // 2
        pred = aux_head(h[torch.arange(len(enc)), read]).view(len(enc), n_pairs, 4).argmax(-1)
        for j, e in enumerate(batch):
            cp = classes_to_cpdag(e["nodes"], pred[j].tolist())
            prompt = f"cpdag {cp} . does {e['x']} cause {e['y']} ?"
            items.append({"prompt": prompt, "label": e["label"]})
    return c2c.accuracy(struct_model, tok, items, yes_id, no_id)


@torch.no_grad()
def aux_structure_acc(model, aux_head, tok, data, n_pairs, batch_size=128) -> float:
    """Per-pair CPDAG accuracy read from the answer-representation (did grounding take?)."""
    pad_id = tok.pad_token_id
    correct = total = 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        enc = [tok(scf.p_direct(e)).input_ids for e in batch]
        width = max(len(s) for s in enc)
        ids = torch.full((len(enc), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        read = torch.tensor([len(s) - 1 for s in enc])
        for j, s in enumerate(enc):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        hr = h[torch.arange(len(enc)), read]
        pred = aux_head(hr).view(len(enc), n_pairs, 4).argmax(-1)
        lab = torch.tensor([e["pairlab"] for e in batch])
        correct += int((pred == lab).sum())
        total += lab.numel()
    return correct / total


def main() -> None:
    torch.manual_seed(0)
    print("generating data (3-var train/val, 4-var OOD) ...")
    train_data = build_examples(8000, n_vars=3, seed=1)
    val_data = build_examples(1500, n_vars=3, seed=2)
    ood_data = build_examples(1500, n_vars=4, seed=3)
    n_pairs = len(train_data[0]["pairlab"])

    tok = c2c.build_tokenizer([scf.text_direct(d) for d in train_data])
    yes_id, no_id = tok.convert_tokens_to_ids(YES_TOK), tok.convert_tokens_to_ids(NO_TOK)

    print("\ntraining PLAIN DIRECT (premises -> answer) ...")
    plain = scf.build_model(tok)
    c2c.train(plain, tok, [scf.text_direct(d) for d in train_data], epochs=12)
    plain.eval()

    print("\ntraining GROUNDED DIRECT (premises -> answer + aux: encode CPDAG) ...")
    grounded = scf.build_model(tok)
    aux_head = nn.Linear(grounded.config.n_embd, n_pairs * 4)
    train_grounded(grounded, aux_head, tok, train_data, epochs=12, lam=1.0)
    grounded.eval()
    aux_head.eval()

    print("\ntraining STRUCT model (CPDAG -> answer) for the extract-then-reason pipeline ...")
    struct = scf.build_model(tok)
    c2c.train(struct, tok, [scf.text_struct_only(d) for d in train_data], epochs=12)
    struct.eval()

    aux_acc = aux_structure_acc(grounded, aux_head, tok, val_data, n_pairs)
    pipe = pipeline_acc(grounded, aux_head, struct, tok, val_data, yes_id, no_id)

    print("\n                            answer accuracy")
    for name, data in [("in-dist (3 vars)", val_data), ("OOD (4 vars)", ood_data)]:
        base = c2c.baselines(data)
        p_acc = scf.acc_teacher_forced(plain, tok, data, scf.p_direct, yes_id, no_id)
        g_acc = scf.acc_teacher_forced(grounded, tok, data, scf.p_direct, yes_id, no_id)
        print(
            f"  {name:16s}  plain-direct={p_acc:.3f}   grounded-direct={g_acc:.3f}   "
            f"corr={base['correlation-heuristic']:.3f}   MEC={base['MEC-oracle (ceiling)']:.3f}"
        )
    print(
        f"\n  grounded aux CPDAG accuracy (in-dist): {aux_acc:.3f} per pair  "
        f"-> the representation extracts the structure"
    )
    print(f"  PIPELINE (extract CPDAG -> reason over it) in-dist answer accuracy: {pipe:.3f}")

    print(
        "\nReading: grounded-direct may match plain-direct even at aux~1.0 -- the structure is "
        "installed but the answer head does not route through it (presence != mediation). "
        "The PIPELINE forces routing: extract the CPDAG (aux), then reason over it. If pipeline >> "
        "plain-direct (toward the MEC ceiling), explicit causal reasoning -- grounding + routing "
        "through the structure -- improves reasoning from raw correlations."
    )


if __name__ == "__main__":
    main()
