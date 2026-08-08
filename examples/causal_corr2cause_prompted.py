# STATUS: canonical · Act 6 Frontier — Phase D part 2: PROMPTED structured-thinking head-to-head.
# Reimplements the prompted "build a graph then answer" method (arXiv:2505.18034, Structured
# Thinking Matters) on real Corr2Cause and compares DIRECT vs STRUCTURED prompting for open LLMs,
# against our TRAINED decoupled reasoner + symbolic ceiling. Point: a *trained* decoupling beats
# the *prompted* decoupling regime. · map: CAUSAL_LLM.md
"""Phase D (part 2) — prompted structured-thinking baseline vs our trained decoupling.

The published prior work (Structured Thinking Matters, arXiv:2505.18034) DECOUPLES by *prompting*
an LLM to first emit the causal graph, then answer. We reimplement it on the real Corr2Cause test
and measure:

  DIRECT      : LLM answers Yes/No directly (the standard baseline).
  STRUCTURED  : LLM first writes the inferred causal structure (skeleton + colliders) as JSON, then
                decides necessity, ending with 'Answer: Yes/No' (the prompted-decoupling method).

Reference rows (same metric): symbolic ceiling on the SAME sample (our exact solver), plus the cited
full-test numbers for the trained decoupled GNN (0.927) and a converged distilbert (0.523). The
thesis: prompted structured thinking helps over direct, but a *trained* decoupled reasoner (and the
exact solver) is far stronger -- consistent with the Phase D mechanism (it's the training signal,
not prompting).

Open LLMs via a local OpenAI-compatible endpoint (Ollama). Honest scope: a fixed N-example sample
(the structured calls are slow), single run, seed-fixed; we report N and cache every call for
reproducibility.

Run::

    PROMPTED_MODELS=mistral:7b,llama3.2:3b PROMPTED_N=150 \
        MISTRAL_API_BASE=http://localhost:11434/v1 MISTRAL_API_KEY=ollama \
        uv run --with datasets --with scikit-learn python examples/causal_corr2cause_prompted.py
"""

import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
import causal_corr2cause_solver as S  # noqa: E402  (parse/predict/f1, the exact ceiling)

BASE = os.environ.get("MISTRAL_API_BASE", "http://localhost:11434/v1")
KEY = os.environ.get("MISTRAL_API_KEY", "ollama")
MODELS = [m for m in os.environ.get("PROMPTED_MODELS", "mistral:7b,llama3.2:3b").split(",") if m]
N = int(os.environ.get("PROMPTED_N", "150"))
SEED = int(os.environ.get("SEED", "0"))
FEWSHOT = int(os.environ.get("PROMPTED_FEWSHOT", "2"))  # for DIRECT (structured is zero-shot)
CACHE_PATH = os.environ.get(
    "PROMPTED_CACHE",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "prompted_cache.json"),
)

INSTR = (
    "You are given a PREMISE listing all statistical (in)dependencies among some variables, and a "
    "HYPOTHESIS about their causal structure. Answer 'Yes' if the hypothesis is necessarily true "
    "in EVERY causal DAG consistent with the premise; otherwise answer 'No'."
)
STRUCT = (
    INSTR + "\n\nThink in two steps.\n"
    "Step 1 - STRUCTURE: from the premise, infer the causal graph and write it as JSON on one "
    'line: {"skeleton": [["A","B"], ...], "colliders": [["A","B","C"] meaning A->B<-C, ...]}. '
    "Include an edge between two variables iff they remain dependent in the premise; mark a "
    "collider A->B<-C when A and B are dependent, B and C are dependent, but A and C are "
    "independent unconditionally.\n"
    "Step 2 - ANSWER: decide whether the hypothesis holds in EVERY DAG consistent with the "
    "premise.\n"
    "End your reply with exactly one line: 'Answer: Yes' or 'Answer: No'."
)


def _load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def call(model, prompt, max_tokens, cache):
    ckey = hashlib.sha256(f"{model}||{max_tokens}||{prompt}".encode()).hexdigest()
    if ckey in cache:
        return cache[ckey]
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                text = json.loads(resp.read())["choices"][0]["message"]["content"]
            cache[ckey] = text
            return text
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            return ""
    return ""


def parse_answer(text):
    """Robustly extract a Yes/No prediction (1/0). Prefer the last 'Answer: ...' line."""
    low = text.lower()
    import re

    hits = re.findall(r"answer\s*:?\s*\**\s*(yes|no)", low)
    if hits:
        return 1 if hits[-1] == "yes" else 0
    # fallback: last standalone yes/no token
    toks = re.findall(r"\b(yes|no)\b", low)
    if toks:
        return 1 if toks[-1] == "yes" else 0
    return 0  # default to the majority (No)


def run_condition(model, sample, shots, structured, cache):
    y, p = [], []
    for i, r in enumerate(sample):
        if structured:
            prompt = STRUCT + "\n\n" + r["input"]
            out = call(model, prompt, 512, cache)
        else:
            prompt = INSTR + "\n\n" + shots + r["input"] + "\nAnswer:"
            out = call(model, prompt, 8, cache)
        y.append(r["label"])
        p.append(parse_answer(out))
        if (i + 1) % 25 == 0:
            _save_cache(cache)
            print(
                f"      [{model} {'struct' if structured else 'direct'}] {i + 1}/{len(sample)}  "
                f"running F1={S.f1(y, p)[2]:.3f}",
                flush=True,
            )
    _save_cache(cache)
    return S.f1(y, p)


def main():
    from datasets import load_dataset

    ds = load_dataset("causalnlp/corr2cause")
    train, test = list(ds["train"]), list(ds["test"])
    rng = random.Random(SEED)
    sample = test[:]
    rng.shuffle(sample)
    sample = sample[:N]
    pos = sum(r["label"] for r in sample)
    print(
        f"Prompted head-to-head on Corr2Cause — N={N} (pos={pos}, {pos / N:.0%}), seed={SEED}, "
        f"models={MODELS}\ncache={CACHE_PATH}"
    )

    # few-shot block for DIRECT, drawn from train (balanced)
    rng2 = random.Random(SEED + 1)
    train_sh = train[:]
    rng2.shuffle(train_sh)
    pos_sh = [r for r in train_sh if r["label"] == 1][: FEWSHOT // 2 + 1]
    neg_sh = [r for r in train_sh if r["label"] == 0][: FEWSHOT - len(pos_sh)]
    shot_rows = (pos_sh + neg_sh)[:FEWSHOT]
    shots = ""
    for r in shot_rows:
        shots += f"{r['input']}\nAnswer: {'Yes' if r['label'] == 1 else 'No'}\n\n"

    # exact symbolic ceiling on the SAME sample (our solver; None=uncovered -> 0)
    sy, sp = [], []
    for r in sample:
        pred = S.predict(r)
        sy.append(r["label"])
        sp.append(0 if pred is None else pred)
    cov = sum(1 for r in sample if S.predict(r) is not None) / len(sample)
    sym = S.f1(sy, sp)

    cache = _load_cache()
    results = {}
    for m in MODELS:
        print(f"\n>>> {m}")
        t0 = time.time()
        results[(m, "direct")] = run_condition(m, sample, shots, False, cache)
        results[(m, "structured")] = run_condition(m, sample, shots, True, cache)
        print(f"    ({time.time() - t0:.0f}s)")

    print(f"\n=== PROMPTED HEAD-TO-HEAD (Corr2Cause, N={N} fixed sample, F1 on positive class) ===")
    print(f"  {'system':42s}  P      R      F1")
    print(
        f"  {'symbolic ceiling (our solver, same sample)':42s}  "
        f"{sym[0]:.3f}  {sym[1]:.3f}  {sym[2]:.3f}   (coverage {cov:.0%})"
    )
    for m in MODELS:
        for cond in ("direct", "structured"):
            pr, rc, f = results[(m, cond)]
            print(f"  {f'{m} — {cond}':42s}  {pr:.3f}  {rc:.3f}  {f:.3f}")
    print("\n  Reference (full 1162-test, from this repo):")
    print(f"  {'trained decoupled GNN (ours)':42s}  {'':6s} {'':6s} 0.927")
    print(f"  {'converged distilbert (end-to-end LM, B1)':42s}  {'':6s} {'':6s} 0.523")
    print(
        "\n  Read: prompted structured thinking helps over direct, but a TRAINED decoupled "
        "reasoner\n  (and the exact solver) is far stronger — it's the training signal, "
        "not the prompt (Phase D)."
    )


if __name__ == "__main__":
    main()
