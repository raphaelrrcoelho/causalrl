# STATUS: canonical · Act 1 Diagnosis — structure is the missing ingredient; CoT doesn't help  ·  map: CAUSAL_LLM.md
"""Does causal reasoning improve the model's reasoning?  A causal-scaffold (CoT) experiment.

The transfer probe (``causal_transfer_corr2cause.py``) showed a from-scratch LM stuck at ~0.64 on a
Corr2Cause-style task -- below the trivial correlation heuristic, unable to fit the
conditional-independence -> causal-direction mapping. This script asks the on-thesis question:

    if the model does *explicit causal reasoning* -- derive the identifiable causal structure (the
    CPDAG: skeleton + orientations common to the whole Markov-equivalence class) before answering --
    does its accuracy improve toward the information-theoretic (MEC) ceiling?

Same task, same tiny model, three conditions:

  1. DIRECT       premises -> answer                 (extract direction from correlations; ~0.64)
  2. STRUCT-ONLY  the CPDAG alone -> answer          (reason over a GIVEN causal structure)
  3. CoT          premises -> *model generates CPDAG* -> answer  (derive structure, then use it)

(The CPDAG is a deterministic function of the premises, so a struct-given-*with*-premises condition
would be a redundant copy and tells us nothing -- hence STRUCT-ONLY shows the CPDAG *instead of* the
premises, making structure-reasoning the only path to the answer.)

If (2) >> (1) and nears the MEC ceiling, the causal structure is the missing ingredient -- having a
causal model improves reasoning, and *extraction* is the bottleneck (so installing/grounding the
structure should help). If (3) >> (1), self-derived causal reasoning pays off end-to-end.

The CPDAG is computed from causalrl ground truth (the Markov-equivalence class), so it is the exact
identifiable causal structure.

CPU-sized; reuses the transfer generator.  Run::

    uv run --extra torch python examples/causal_reasoning_scaffold.py
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_transfer_corr2cause as c2c  # sibling example, imported after sys.path tweak

YES_TOK, NO_TOK, EOS_TOK, PAD_TOK = c2c.YES_TOK, c2c.NO_TOK, c2c.EOS_TOK, c2c.PAD_TOK

torch.set_num_threads(4)


# ==============================================================================================
# 1. The identifiable causal structure (CPDAG) from the Markov-equivalence class.
# ==============================================================================================


def cpdag_str(nodes: list[str], members) -> str:
    """Skeleton + MEC-invariant orientations as text: 'A > B' directed, 'A = B' undirected."""
    parts = []
    for i, j in itertools.combinations(nodes, 2):
        adjacent = any(j in g.children(i) or i in g.children(j) for g in members)
        if not adjacent:
            continue
        if all(j in g.children(i) for g in members):
            parts.append(f"{i} > {j}")
        elif all(i in g.children(j) for g in members):
            parts.append(f"{j} > {i}")
        else:
            parts.append(f"{i} = {j}")
    return " ".join(parts) if parts else "none"


def build_examples(n_examples: int, n_vars: int, seed: int) -> list[dict]:
    """Balanced examples carrying premises, query, the true CPDAG, label, and baselines."""
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
                "x": x,
                "y": y,
                "label": label,
                "cpdag": cpdag_str(nodes, members),
                "corr": not c2c.d_separated(g, {x}, {y}, set()),
                "oracle": (sum(y in m.descendants(x) for m in members) >= len(members) / 2),
            }
        )
        cnt[label] += 1
    rng.shuffle(out)
    return out


# Prompt formats: the answer token is always the immediate next token after the prompt.
# NB: premises and CPDAG are both functions of the same CIs. To make "give the model the causal
# structure" a real test (not a redundant copy of the premises), the struct-only condition shows the
# CPDAG *instead of* the premises -- so the only path to the answer is reasoning over the structure.
def p_direct(e: dict) -> str:
    return f"{e['premises']} . does {e['x']} cause {e['y']} ?"


def p_struct_only(e: dict) -> str:
    return f"cpdag {e['cpdag']} . does {e['x']} cause {e['y']} ?"


def p_scaffold_prefix(e: dict) -> str:  # model must generate "{cpdag} end" then the answer
    return f"{e['premises']} . does {e['x']} cause {e['y']} ? cpdag"


def text_direct(e: dict) -> str:
    return p_direct(e) + (YES_TOK if e["label"] else NO_TOK)


def text_struct_only(e: dict) -> str:
    return p_struct_only(e) + (YES_TOK if e["label"] else NO_TOK)


def text_scaffold(e: dict) -> str:
    return f"{e['premises']} . does {e['x']} cause {e['y']} ? cpdag {e['cpdag']} end" + (
        YES_TOK if e["label"] else NO_TOK
    )


# ==============================================================================================
# 2. Model + free-generation (chain-of-thought) evaluation.
# ==============================================================================================


def build_model(tok: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=len(tok),
        n_positions=192,
        n_ctx=192,
        n_embd=160,
        n_layer=4,
        n_head=4,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(tok))
    return model


def acc_teacher_forced(model, tok, data, prompt_fn, yes_id, no_id, batch_size=128) -> float:
    """Accuracy reading the answer token right after a (given) prompt."""
    items = [{"prompt": prompt_fn(e), "label": e["label"]} for e in data]
    return c2c.accuracy(model, tok, items, yes_id, no_id, batch_size=batch_size)


@torch.no_grad()
def acc_free_generation(model, tok, data, yes_id, no_id, batch_size=64, max_new=28) -> float:
    """The model generates the CPDAG itself, then we read the first yes/no it emits (causal CoT)."""
    tok.padding_side = "left"
    correct = 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        enc = tok([p_scaffold_prefix(e) for e in batch], padding=True, return_tensors="pt")
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        cont = gen[:, enc.input_ids.shape[1] :]
        for j, e in enumerate(batch):
            row = cont[j].tolist()
            pred = next((t for t in row if t in (yes_id, no_id)), None)
            if pred is not None:
                correct += int((pred == yes_id) == e["label"])
    tok.padding_side = "right"
    return correct / len(data)


def main() -> None:
    torch.manual_seed(0)
    print("generating causalrl Corr2Cause-style data with CPDAG scaffolds (3-var, 4-var OOD) ...")
    train_data = build_examples(8000, n_vars=3, seed=1)
    val_data = build_examples(1500, n_vars=3, seed=2)
    ood_data = build_examples(1500, n_vars=4, seed=3)
    e = train_data[0]
    print(f"  e.g. scaffold: {text_scaffold(e)!r}")

    # shared tokenizer over all three formats so the models use the same vocab
    corpus_direct = [text_direct(d) for d in train_data]
    corpus_struct = [text_struct_only(d) for d in train_data]
    corpus_scaffold = [text_scaffold(d) for d in train_data]
    tok = c2c.build_tokenizer(corpus_direct + corpus_struct + corpus_scaffold)
    yes_id, no_id = tok.convert_tokens_to_ids(YES_TOK), tok.convert_tokens_to_ids(NO_TOK)

    print("\ntraining DIRECT model (premises -> answer) ...")
    direct = build_model(tok)
    c2c.train(direct, tok, corpus_direct, epochs=12)
    direct.eval()

    print("\ntraining STRUCT-ONLY model (CPDAG -> answer; reason over a given structure) ...")
    struct = build_model(tok)
    c2c.train(struct, tok, corpus_struct, epochs=12)
    struct.eval()

    print("\ntraining CoT model (premises -> derive CPDAG -> answer) ...")
    scaffold = build_model(tok)
    c2c.train(scaffold, tok, corpus_scaffold, epochs=12)
    scaffold.eval()

    print("\n                                accuracy")
    for name, data in [("in-dist (3 vars)", val_data), ("OOD (4 vars)", ood_data)]:
        base = c2c.baselines(data)
        d_acc = acc_teacher_forced(direct, tok, data, p_direct, yes_id, no_id)
        s_acc = acc_teacher_forced(struct, tok, data, p_struct_only, yes_id, no_id)
        s_cot = acc_free_generation(scaffold, tok, data[:600], yes_id, no_id)
        print(
            f"  {name:16s}  direct={d_acc:.3f}   struct-only={s_acc:.3f}   "
            f"CoT(self-derived)={s_cot:.3f}   corr={base['correlation-heuristic']:.3f}   "
            f"MEC={base['MEC-oracle (ceiling)']:.3f}"
        )

    print(
        "\nReading: 'direct' (premises->answer) = extract causal direction from raw correlations. "
        "'struct-only' (CPDAG->answer, no premises) = reason over a GIVEN structure -- if it "
        ">> direct and nears the MEC ceiling, having a causal model is the missing ingredient and "
        "*extraction* is the bottleneck (so installing/grounding the structure should help). "
        "'CoT(self-derived)' makes the model derive the structure itself, then answer."
    )


if __name__ == "__main__":
    main()
