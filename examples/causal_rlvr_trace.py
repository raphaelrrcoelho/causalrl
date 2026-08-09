# STATUS: research · ladder §5 — DECOUPLED RLVR: on-policy oracle-verified STRUCTURE rewards vs outcome-only, on the R2 trace substrate · design: docs/causal_llm/LADDER.md · map: CAUSAL_LLM.md
"""Decoupled RLVR — can on-policy structure rewards repair what teacher-forcing cannot see?

R2's decomposition localized the surviving confounded failure: the model CORRUPTS its own written
closure on confounded pairs (own-trace implies the correct "no" only 0.562±0.034) — and supervised
trace training can never fix that, because teacher forcing only ever shows the model TRUE traces,
never its own corrupted ones. On-policy RL sees exactly those. Meanwhile the literature shows
outcome-only RLVR does not make reasoning causally faithful (arXiv:2604.22074). This script runs
the controlled version of both claims on the R2 substrate:

  Phase A  supervised TRACE pretraining (identical to R2's inductive arm) — the shared start.
  Phase B  GRPO from that shared state, TWO reward arms, same budget/samples/seed:
             OUTCOME   r = 1[sampled answer correct]                  (the standard RLVR reward)
             STRUCT    r = 1[answer correct] + F1(sampled trace's final set, TRUE closure)
                       (the decoupled reward — the `causalrl`-oracle-verified intermediate)

RL prompts are the causal-query half of the training set (the trace's final set has a well-defined
target there; the confounded trap IS a causal query). Group-relative advantages (GRPO), no KL term
at this scale. Eval = the full R2 decomposition (acc + own-trace-implied answer) on `cause` and
`confounded`, s3 and s4, BEFORE RL and after each arm. Hypothesis: STRUCT raises the confounded
own-trace correctness (ctans) and with it the trap accuracy; OUTCOME does not (or reward-hacks the
answer while the trace stays corrupt). Abstention/identifiability-gating is deliberately out of
scope here — every query on this substrate is decidable; that leg belongs to the Corr2Cause side.

Run::

    SEEDS=0 FAST=1 uv run --extra torch python examples/causal_rlvr_trace.py   # smoke
    SEEDS=0       uv run --extra torch python examples/causal_rlvr_trace.py

Knobs: SEEDS, RLSTEPS (default 300), KSAMP (group size, default 4), RLLR (default 1e-5), FAST=1.
"""

from __future__ import annotations

import os
import random
import sys

import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy
import causal_pure_twostage as pt

SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
FAST = os.environ.get("FAST") == "1"
RLSTEPS = int(os.environ.get("RLSTEPS", "30" if FAST else "300"))
KSAMP = int(os.environ.get("KSAMP", "4"))
RLLR = float(os.environ.get("RLLR", "1e-5"))
BATCH_PROMPTS = 8
# CONFBOOST=1: draw HALF of all RL batches from confounded-pattern prompts (corr AND NOT cause) —
# the exposure control. In the natural causal-query pool those graphs are a small minority, so a
# reward cannot repair a failure mode it almost never samples; this knob removes that explanation.
CONFBOOST = os.environ.get("CONFBOOST") == "1"


def true_closure(e):
    _steps, s = pt.closure_steps(e["adj"], sum(e["present"]), e["ys"])
    return s


def parse_final_set(toks, e):
    """The final ';'-segment of a sampled trace as a slot set (inductive format)."""
    k = sum(e["present"])
    slot = {t: s for s, t in enumerate(e["entw"][:k])}
    last = toks[len(toks) - toks[::-1].index(pt.SEMI) :] if pt.SEMI in toks else toks
    return {slot[t] for t in last if t in slot}


def set_f1(pred: set, true: set) -> float:
    tp = len(pred & true)
    if not tp:
        return 0.0
    prec, rec = tp / len(pred), tp / len(true)
    return 2 * prec * rec / (prec + rec)


def build_prompt(e):
    k = sum(e["present"])
    _prose, query = pt.split_query(e)
    return query + [pt.GO] + pt.graph_tokens(e["adj"], e["entw"], k) + [pt.GC, pt.TO]


def sample_group(model, prompts, k_samp, max_new=48):
    """Sample k_samp trace+answer continuations per prompt (same-length prompt batch).

    Returns per (prompt, sample): generated trace tokens, sampled answer id, and the sum of
    log-probs of every SAMPLED token (trace tokens, the </t>, and the answer) — the GRPO objective
    differentiates through a teacher-forced re-scoring pass, so sampling here is no_grad."""
    n = len(prompts)
    rows = n * k_samp
    with torch.no_grad():
        cur = torch.tensor([p for p in prompts for _ in range(k_samp)])
        done = torch.zeros(rows, dtype=torch.bool)
        gen = [[] for _ in range(rows)]
        ans = [pt.NO] * rows
        got_ans = torch.zeros(rows, dtype=torch.bool)
        for _ in range(max_new):
            logits = model(input_ids=cur).logits[:, -1, :]
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
            for j in range(rows):
                if done[j]:
                    continue
                t = int(nxt[j])
                if got_ans[j]:  # the token right after </t> is the answer
                    ans[j] = t
                    done[j] = True
                elif t == pt.TC:
                    got_ans[j] = True
                else:
                    gen[j].append(t)
            if bool(done.all()) or cur.size(1) >= pt.NPOS_TRACE - 2:
                break
            step = torch.where(done, torch.full_like(nxt, pt.PAD), nxt)
            cur = torch.cat([cur, step.unsqueeze(1)], 1)
    return gen, ans


def rescore_logprob(model, prompts, gen, ans, k_samp):
    """Teacher-forced re-scoring of each sampled continuation: sum log p(token) with grad."""
    seqs = []
    for j in range(len(gen)):
        p = prompts[j // k_samp]
        seqs.append(p + gen[j] + [pt.TC, ans[j]])
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pt.PAD, dtype=torch.long)
    attn = torch.zeros(len(seqs), width, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
        attn[j, : len(s)] = 1
    logits = model(input_ids=ids, attention_mask=attn).logits
    logp = torch.log_softmax(logits, -1)
    out = []
    for j, s in enumerate(seqs):
        plen = len(prompts[j // k_samp])
        # positions plen-1 .. len(s)-2 predict tokens plen .. len(s)-1 (the sampled region)
        tgt = torch.tensor(s[plen:])
        rowlp = logp[j, plen - 1 : len(s) - 1].gather(1, tgt.unsqueeze(1)).sum()
        out.append(rowlp)
    return torch.stack(out)


def grpo(model, data, steps, use_struct, tag, seed):
    """Group-relative policy optimization over trace+answer samples."""
    opt = torch.optim.AdamW(model.parameters(), lr=RLLR)
    rng = random.Random(seed + 31)
    # bucket the causal-query pool by prompt length (pad-free same-width sampling batches);
    # CONFBOOST keeps a parallel bucket set of confounded-pattern prompts and alternates.
    def make_buckets(rows):
        b: dict[int, list] = {}
        for e in rows:
            b.setdefault(len(build_prompt(e)), []).append(e)
        ws = [w for w, x in b.items() if len(x) >= BATCH_PROMPTS]
        return b, ws, [len(b[w]) for w in ws]

    causal = [e for e in data if e["is_causal"]]
    conf_rows = [e for e in causal if e["corr"] and not e["cause"]]
    pools = [make_buckets(causal)]
    if CONFBOOST and conf_rows:
        pools.append(make_buckets(conf_rows))
        print(f"    [{tag}] CONFBOOST: {len(conf_rows)} confounded-pattern prompts", flush=True)
    model.eval()  # no dropout during sampling/scoring; grads still flow in rescore
    for step in range(steps):
        buckets, widths, weights = pools[step % len(pools)]
        w = rng.choices(widths, weights=weights)[0]
        batch = rng.sample(buckets[w], BATCH_PROMPTS)
        prompts = [build_prompt(e) for e in batch]
        gen, ans = sample_group(model, prompts, KSAMP)
        rewards = torch.zeros(len(gen))
        for j in range(len(gen)):
            e = batch[j // KSAMP]
            r = float((ans[j] == pt.YES) == bool(e["label"]))
            if use_struct:
                r += set_f1(parse_final_set(gen[j], e), true_closure(e))
            rewards[j] = r
        adv = rewards.view(-1, KSAMP)
        adv = (adv - adv.mean(1, keepdim=True)).view(-1)
        if float(adv.abs().sum()) == 0.0:
            continue  # every sample in every group tied — no signal this step
        logp = rescore_logprob(model, prompts, gen, ans, KSAMP)
        loss = -(adv * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 6) == 0:
            print(
                f"    [{tag}] step {step + 1}/{steps}  mean reward {float(rewards.mean()):.3f}",
                flush=True,
            )


def evaluate(model, sets, label):
    print(f"\n  --- {label}")
    out = {}
    for name, data in sets.items():
        acc, _f1, tans, _cons = pt.acc_trace(model, data, False)
        out[f"{name}_acc"], out[f"{name}_tans"] = acc, tans
        print(f"    {name:<10} acc {acc:.3f}   own-trace-implied {tans:.3f}", flush=True)
    return out


def run_seed(seed):
    torch.manual_seed(seed)
    n = 2000 if FAST else 8000
    epochs = 4 if FAST else 12
    nt = 400 if FAST else 1500

    train_data = hy.build(n, sizes=[2, 3], seed=seed)
    t3, t4 = hy.build(nt, [3], seed + 50), hy.build(nt, [4], seed + 60)
    sets = {
        "conf_s3": pt.confounded(t3),
        "conf_s4": pt.confounded(t4),
        "cause_s3": [e for e in t3 if e["is_causal"]],
        "cause_s4": [e for e in t4 if e["is_causal"]],
    }

    print(f"\n  seed {seed}: Phase A — supervised TRACE pretrain", flush=True)
    model = GPT2LMHeadModel(pt.gpt2(pt.NPOS_TRACE))
    corpus = pt.build_corpus("trace", train_data, seed)
    pt.train(model, corpus, epochs=epochs, tag="pretrain")
    model.eval()
    base = evaluate(model, sets, "BASELINE (supervised trace, before RL)")
    state = {k: v.clone() for k, v in model.state_dict().items()}

    results = {"base": base}
    for arm, use_struct in (("OUTCOME", False), ("STRUCT", True)):
        model.load_state_dict(state)
        print(f"\n  seed {seed}: Phase B — GRPO {arm} ({RLSTEPS} steps, K={KSAMP})", flush=True)
        grpo(model, train_data, RLSTEPS, use_struct, arm, seed)
        model.eval()
        results[arm] = evaluate(model, sets, f"AFTER RL — {arm} reward")
    return results


def main():
    print("Decoupled RLVR on the R2 trace substrate — structure rewards vs outcome-only")
    print(f"  seeds {SEEDS}{'  [FAST smoke]' if FAST else ''}; {RLSTEPS} GRPO steps, K={KSAMP}")
    for s in SEEDS:
        run_seed(s)
    print(
        "\n  Reading: BASELINE row = R2's supervised ceiling. If STRUCT lifts the confounded"
        "\n  own-trace-implied answer (the corrupted-computation channel) and its accuracy while"
        "\n  OUTCOME does not, the decoupled-reward mechanism transfers to RL — the RL echo of"
        "\n  Phase D. If neither moves, on-policy exposure alone is not the missing ingredient."
    )


if __name__ == "__main__":
    main()
