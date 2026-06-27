# STATUS: canonical · Act 6 Frontier — Phase 2 LEARNED decoupling on real Corr2Cause: end-to-end LM (2a) vs decoupled parse->GNN reasoner (2b) vs symbolic ceiling; + OOD robustness (relabel/paraphrase) + a Mistral LLM baseline  ·  map: CAUSAL_LLM.md
"""Phase 2 — the LEARNED decoupling experiment on the real Corr2Cause benchmark.

Phase 1 (``causal_corr2cause_solver.py``) proved the benchmark is structure-decidable: an exact
parse->MEC->necessity *oracle* scores F1 0.92 (vs GPT-4 ~0.29). That is the ceiling, not a learned
model. Phase 2 asks the branch's actual thesis on real data:

    Does DECOUPLING perception from reasoning beat an end-to-end model -- and where does each break?

Systems, one common test set (the official 1162-example test):

  2a. END-TO-END LM (the "joint" path).  Fine-tune a small pretrained encoder (bert-tiny on CPU,
      distilbert on GPU) directly on raw ``input`` text -> label. Perception + reasoning entangled in
      one set of weights, exactly the regime the synthetic study found brittle.

  2b. DECOUPLED reasoner (the "fix").  PERCEPTION = the deterministic ``parse`` from Phase 1 (premise
      text -> skeleton + v-structure evidence). REASONING = a small, size-agnostic GNN trained to map
      that parsed structure -> label. The halves never share weights; the reasoner learns
      d-separation/necessity, it is not hand-coded.

  LLM baseline.  A zero/few-shot Mistral model prompted for entailment (Yes/No) -- a current open LLM
      standing in for the paper's GPT-4 ~0.29 reference (local GGUF, no API key needed).

Two axes of generalization (the publishable part -- "decoupled beats end-to-end" only on OOD, since a
fine-tuned LM already hits ~0.95 F1 in-distribution; cf. Jin et al. 2024 robustness collapse):

  * SIZE.  The official split is itself a size shift (train is 96% 6-variable graphs; test spans 2..6).
    The GNN is size-agnostic by construction; the LM's patterns are tied to the sizes it saw.
  * INPUT PERTURBATION.  We synthesize two OOD test variants:
      - relabel:    permute the variable letters A..F consistently. The parse->GNN path is EXACTLY
                    invariant (isomorphic graph); a shortcut-learning LM degrades.
      - paraphrase: rewrite the premise's relation phrasing ("correlates with" -> "is associated
                    with", ...). This BREAKS the regex perception (so 2b AND the symbolic oracle drop)
                    -- motivating a *learned* perception (see ``causal_corr2cause_perception.py``).

Honest scope: small models, single seed by default, CPU-sized. The symbolic solver remains the
ceiling/oracle; this script tests *learnability* of the reasoning half + robustness, not frontier scale.

Run::

    SMOKE=1 HF_HOME=/tmp/hf uv run --extra torch --with datasets --with scikit-learn \
        python examples/causal_corr2cause_learned.py            # fast smoke
    HF_HOME=/tmp/hf uv run --extra torch --with datasets --with scikit-learn \
        python examples/causal_corr2cause_learned.py            # full CPU run
    # Mistral baseline auto-runs locally if llama-cpp-python + huggingface_hub are available:
    #   ... --with llama-cpp-python --with huggingface_hub ...   (or set MISTRAL_API_KEY for the API)
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_corr2cause_solver import f1, parse, parse_hyp
from causal_corr2cause_solver import predict as symbolic_predict

# --------------------------------------------------------------------------- config (env knobs)
SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
NMAX = 6
TEMPLATES = [
    "child",
    "has_collider",
    "has_confounder",
    "non-child descendant",
    "non-parent ancestor",
    "parent",
]
TPL_ID = {t: i for i, t in enumerate(TEMPLATES)}

# which systems to run (comma list): ceiling, lex, gnn, lm, mistral
RUN = set(os.environ.get("RUN", "ceiling,lex,gnn,lm,mistral").split(","))
# OOD test variants to evaluate every system on (comma list): relabel, paraphrase ; "" -> clean only
OOD = [v for v in os.environ.get("OOD", "relabel,paraphrase").split(",") if v]

N_TRAIN_GNN = int(os.environ.get("N_TRAIN_GNN", "40000"))
EPOCHS_GNN = int(os.environ.get("EPOCHS_GNN", "30"))
N_TRAIN_LM = int(os.environ.get("N_TRAIN_LM", "20000"))
EPOCHS_LM = int(os.environ.get("EPOCHS_LM", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "384"))
MODEL = os.environ.get("MODEL", "")  # "" -> auto by device
# bert-tiny ships no fast-tokenizer files; reuse the shared uncased WordPiece (identical 30522 vocab).
LM_TOKENIZER = os.environ.get("LM_TOKENIZER", "bert-base-uncased")
# Batch size. On this 6 GB laptop GPU, batch 32 x maxlen 384 OOMs and WSL2 turns that into a silent
# process kill -- keep <=16 (maxlen<=256) for DEVICE=cuda; CPU is fine larger.
BS_LM = int(os.environ.get("BS_LM", "16"))
# Crash-resilient LM training: checkpoint every CKPT_EVERY steps + each epoch so a GPU/WSL wedge
# doesn't lose the run; RESUME auto-continues (device-agnostic -> a crashed GPU run can finish on CPU).
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/tmp/claude-1000/-mnt-c-Users-rapha-Documents-Code-causalrl/8569067c-24af-4b49-b114-c64b15d4c2b4/scratchpad/ckpt",
)
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "300"))
RESUME = os.environ.get("RESUME", "1") not in ("", "0", "false")

N_LLM = int(os.environ.get("N_LLM", "200"))
FEWSHOT = int(os.environ.get("FEWSHOT", "4"))
MISTRAL_MODELS = [m for m in os.environ.get("MISTRAL_MODELS", "mistral-small-latest").split(",") if m]
# backend: auto (local GGUF if no API key, else API) | api | local | off
MISTRAL_BACKEND = os.environ.get("MISTRAL_BACKEND", "auto")
# OpenAI-compatible base for the "api" backend. Point at a local Ollama (http://localhost:11434/v1)
# to use an Ollama-managed model (robust GPU; survives reboots) with MISTRAL_API_KEY=ollama.
MISTRAL_API_BASE = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
# smallest official OPEN-WEIGHT Mistral (Apache-2.0), 4-bit -> CPU-runnable, no API key needed
MISTRAL_GGUF_REPO = os.environ.get("MISTRAL_GGUF_REPO", "bartowski/Mistral-7B-Instruct-v0.3-GGUF")
MISTRAL_GGUF_FILE = os.environ.get("MISTRAL_GGUF_FILE", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
CACHE_DIR = os.environ.get(
    "CACHE_DIR",
    "/tmp/claude-1000/-mnt-c-Users-rapha-Documents-Code-causalrl/8569067c-24af-4b49-b114-c64b15d4c2b4/scratchpad/llmcache",
)

if SMOKE:
    N_TRAIN_GNN, EPOCHS_GNN = 4000, 12
    N_TRAIN_LM, EPOCHS_LM = 1500, 1
    N_LLM = 20


# --------------------------------------------------------------------------- OOD perturbations
LETTERS = "ABCDEF"
# premise relation phrasings -> surface variants (break the exact-phrase regex perception)
_PARAPHRASE = [
    (r"correlates with", ["is correlated with", "covaries with", "is associated with",
                          "shows correlation with"]),
    (r"is independent of", ["is not correlated with", "is unrelated to", "shows no association with"]),
    (r"are independent given", ["are conditionally independent given",
                               "become independent after conditioning on",
                               "are independent once we control for"]),
    (r"Suppose there is a closed system of", ["Consider a closed system of", "Assume a closed system of"]),
    (r"All the statistical relations among these", ["The statistical dependencies among these",
                                                   "The observed relations among these"]),
]


def relabel_row(row, rng):
    """Permute the variable letters A..F consistently across the whole input (premise + hypothesis)."""
    perm = list(LETTERS)
    rng.shuffle(perm)
    mp = dict(zip(LETTERS, perm))
    text = re.sub(r"\b([A-F])\b", lambda m: mp[m.group(1)], row["input"])
    return {**row, "input": text}


def paraphrase_row(row, rng):
    """Rewrite the PREMISE's relation phrasing (hypothesis kept canonical so the query parses)."""
    text = row["input"]
    if "Hypothesis:" not in text:
        return {**row, "input": text}
    prem, hyp = text.split("Hypothesis:", 1)
    for pat, opts in _PARAPHRASE:
        prem = re.sub(pat, lambda m, o=opts: rng.choice(o), prem)
    return {**row, "input": prem + "Hypothesis:" + hyp}


# INDEPENDENT, held-out paraphrases: full-sentence rewrites generated by a local LLM (Mistral-7B via
# Ollama), vetted meaning-preserving, DISJOINT from the _PARAPHRASE connective-swaps used in perception
# training. Used only at EVAL -> tests generalization to phrasings never seen in training, and unlike
# _PARAPHRASE these restructure the whole statement (strictly harder). De-circularizes the paraphrase claim.
_HELDOUT_CORR = [
    "There is a statistical correlation between {a} and {b}",
    "{a} and {b} show a statistical association",
    "{a} and {b} demonstrate a correlation",
    "{b}'s values tend to relate to {a}'s values",
]
_HELDOUT_INDEP = [
    "{a} has no statistical connection with {b}",
    "{a}'s values are unrelated to those of {b}",
    "{a} and {b} are uncorrelated",
    "{a} is not associated with {b}",
]
_HELDOUT_COND = [
    "given {z}, {a} and {b} do not exhibit a statistical correlation",
    "when considering {z}, {a} and {b} are unrelated",
    "in the presence of {z}, there is no association between {a} and {b}",
    "with {z} as a factor, {a} and {b} show independence",
]


def paraphrase_heldout_row(row, rng):
    """Rewrite each premise (in)dependency statement with a held-out LLM phrasing (structure preserved,
    surface form fully restructured). Hypothesis kept canonical so the query still parses."""
    text = row["input"]
    if "Hypothesis:" not in text:
        return {**row, "input": text}
    prem, hyp = text.split("Hypothesis:", 1)
    prem = re.sub(  # most specific first: conditional independence
        r"([A-Z]) and ([A-Z]) are independent given ([A-Z][A-Z ,and]*)",
        lambda m: rng.choice(_HELDOUT_COND).format(a=m.group(1), b=m.group(2), z=m.group(3).strip(" .")),
        prem)
    prem = re.sub(r"([A-Z]) is independent of ([A-Z])",
                  lambda m: rng.choice(_HELDOUT_INDEP).format(a=m.group(1), b=m.group(2)), prem)
    prem = re.sub(r"([A-Z]) correlates with ([A-Z])",
                  lambda m: rng.choice(_HELDOUT_CORR).format(a=m.group(1), b=m.group(2)), prem)
    return {**row, "input": prem + "Hypothesis:" + hyp}


def perturb(rows, kind, seed):
    rng = random.Random(seed)
    fn = {"relabel": relabel_row, "paraphrase": paraphrase_row,
          "paraphrase_heldout": paraphrase_heldout_row}[kind]
    return [fn(r, rng) for r in rows]


# --------------------------------------------------------------------------- 2b perception: parse -> structure
def struct_from_parse(variables, corr, indep):
    """(skeleton S, directed v-structure evidence D, present) over NMAX padded nodes -- the PC output
    before Meek closure; S+D determine the CPDAG, so the reasoner has sufficient info to LEARN from."""
    import numpy as np

    idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    sep_pairs = {p for p, _ in indep}

    def adjacent(a, b):
        p = frozenset((a, b))
        return (p in corr) and (p not in sep_pairs)

    S = np.zeros((NMAX, NMAX), dtype="float32")
    D = np.zeros((NMAX, NMAX), dtype="float32")
    present = np.zeros(NMAX, dtype="float32")
    present[:n] = 1.0
    for a in variables:
        for b in variables:
            if a != b and adjacent(a, b):
                S[idx[a], idx[b]] = 1.0
    for pair, z in indep:
        if len(pair) != 2:
            continue
        i, j = tuple(pair)
        for k in variables:
            if k in (i, j) or k in z:
                continue
            if adjacent(i, k) and adjacent(j, k):
                D[idx[i], idx[k]] = 1.0
                D[idx[j], idx[k]] = 1.0
    return S, D, present, idx


def featurize(row):
    """Premise+hypothesis -> GNN inputs, or None if unparseable (scored as a negative downstream)."""
    import numpy as np

    variables, corr, indep, hyp = parse(row["input"])
    if not variables or len(variables) > NMAX:
        return None
    x, y = parse_hyp(row["template"], hyp)
    if x is None or x not in variables or y not in variables:
        return None
    S, D, present, idx = struct_from_parse(variables, corr, indep)
    isx = np.zeros(NMAX, dtype="float32")
    isy = np.zeros(NMAX, dtype="float32")
    isx[idx[x]] = 1.0
    isy[idx[y]] = 1.0
    return S, D, isx, isy, present, TPL_ID[row["template"]], idx[x], idx[y]


def build_gnn_tensors(rows):
    import numpy as np
    import torch

    feats, labels, keep = [], [], []
    for r in rows:
        f = featurize(r)
        if f is None:
            keep.append(False)
            continue
        keep.append(True)
        feats.append(f)
        labels.append(r["label"])
    if not feats:  # everything failed to parse (e.g. fully paraphrased) -> empty tensors
        return None, keep
    S = torch.tensor(np.stack([f[0] for f in feats]))
    D = torch.tensor(np.stack([f[1] for f in feats]))
    nf = torch.tensor(np.stack([np.stack([f[2], f[3]], -1) for f in feats]))  # (b,N,2)
    present = torch.tensor(np.stack([f[4] for f in feats]))
    tpl = torch.tensor([f[5] for f in feats], dtype=torch.long)
    xi = torch.tensor([f[6] for f in feats], dtype=torch.long)
    yi = torch.tensor([f[7] for f in feats], dtype=torch.long)
    lab = torch.tensor(labels, dtype=torch.float32)
    return (S, D, nf, present, tpl, xi, yi, lab), keep


# --------------------------------------------------------------------------- 2b reasoner: size-agnostic GNN
def make_gnn(d=64, steps=6):
    import torch
    from torch import nn

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(2, d)
            self.init = nn.Parameter(torch.randn(d) * 0.1)
            self.w_self = nn.Linear(d, d)
            self.w_skel = nn.Linear(d, d)
            self.w_din = nn.Linear(d, d)
            self.w_dout = nn.Linear(d, d)
            self.tpl = nn.Embedding(len(TEMPLATES), d)
            self.read = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
            self.steps = steps

        def forward(self, S, D, nf, present, tpl, xi, yi):
            b = S.size(0)
            pm = present.unsqueeze(-1)
            h = (self.init.view(1, 1, -1) + self.embed(nf)) * pm
            for _ in range(self.steps):
                m_skel = torch.bmm(S, self.w_skel(h))
                m_din = torch.bmm(D.transpose(1, 2), self.w_din(h))  # parents -> node
                m_dout = torch.bmm(D, self.w_dout(h))  # node -> children
                h = torch.relu(self.w_self(h) + m_skel + m_din + m_dout) * pm
            ix = torch.arange(b)
            feat = torch.cat([h[ix, xi], h[ix, yi], self.tpl(tpl)], -1)
            return self.read(feat).squeeze(-1)

    return GNN()


def _gnn_predict(model, rows):
    """Predict for EVERY row (parse-fail -> 0), so the GNN is scored on the full test set."""
    import torch

    packed, keep = build_gnn_tensors(rows)
    covered = []
    if packed is not None:
        S, D, nf, present, tpl, xi, yi, _ = packed
        model.eval()
        with torch.no_grad():
            covered = (model(S, D, nf, present, tpl, xi, yi) > 0).long().tolist()
    preds, c = [], 0
    for k in keep:
        preds.append(covered[c] if k else 0)
        c += 1 if k else 0
    return preds


def train_gnn_model(train_rows):
    """Train the GNN reasoner on regex-parsed structure; return the trained model."""
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    packed, _ = build_gnn_tensors(train_rows)
    Str, Dtr, nftr, ptr, ttr, xtr, ytr, ltr = packed
    pos = float(ltr.sum())
    pos_weight = torch.tensor([(len(ltr) - pos) / max(pos, 1.0)])
    model = make_gnn()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bs, n = 256, Str.size(0)
    rng = random.Random(SEED)
    order = list(range(n))
    print(f"  [gnn] train on {n} parsed rows (pos={int(pos)}), {EPOCHS_GNN} epochs")
    for ep in range(EPOCHS_GNN):
        rng.shuffle(order)
        model.train()
        tot = 0.0
        for i in range(0, n, bs):
            b = order[i : i + bs]
            loss = lossf(model(Str[b], Dtr[b], nftr[b], ptr[b], ttr[b], xtr[b], ytr[b]), ltr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep == 0 or ep == EPOCHS_GNN - 1 or (ep + 1) % 10 == 0:
            tr_f1 = f1([r["label"] for r in train_rows[:3000]], _gnn_predict(model, train_rows[:3000]))[2]
            print(f"    epoch {ep + 1:>2d}  loss={tot / n:.4f}  train-gate F1={tr_f1:.3f}")
    return model


def train_gnn(train_rows):
    """Train the GNN and wrap it as a predict_fn(rows)->list[int]."""
    model = train_gnn_model(train_rows)
    return lambda rows: _gnn_predict(model, rows)


# --------------------------------------------------------------------------- 2a end-to-end LM
def train_lm(train_rows):
    """Fine-tune a small encoder text->label; return (predict_fn(rows)->list[int], model_name)."""
    import torch
    from torch import nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # CPU by default: this WSL2 box's CUDA *training* path is unstable (basic matmul works, but a
    # distilbert train step hard-kills the process). GPU is opt-in via DEVICE=cuda.
    device = os.environ.get("DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model_name = MODEL or "google/bert_uncased_L-2_H-128_A-2"  # official tiny BERT (clean config)
    torch.manual_seed(SEED)
    print(f"  [lm] end-to-end fine-tune {model_name} on {device}, "
          f"{N_TRAIN_LM} rows x {EPOCHS_LM} ep, maxlen {MAXLEN}")
    tok = AutoTokenizer.from_pretrained(LM_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    def encode(rows):
        enc = tok([r["input"] for r in rows], truncation=True, padding="max_length",
                  max_length=MAXLEN, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    rng = random.Random(SEED)
    tr = list(train_rows)
    rng.shuffle(tr)
    tr = tr[:N_TRAIN_LM]
    ids, attn = encode(tr)
    y = torch.tensor([r["label"] for r in tr], dtype=torch.long)
    pos = float((y == 1).sum())
    weight = torch.tensor([1.0, (float((y == 0).sum())) / max(pos, 1.0)]).to(device)
    lossf = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    bs, n = BS_LM, ids.size(0)
    eval_bs = 32  # forward-only; small to stay safe on a 6 GB GPU

    # checkpoint/resume so a GPU/WSL wedge mid-run loses at most a partial epoch (weights are saved).
    import re as _re

    tag = _re.sub(r"[^0-9A-Za-z]+", "_", f"{model_name}-{N_TRAIN_LM}-{MAXLEN}-{EPOCHS_LM}-{BS_LM}")
    ckpt = os.path.join(CKPT_DIR, f"lm_{tag}.pt")
    start_epoch = 0
    if RESUME and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch = int(st["epoch"])
        print(f"  [lm] resumed from {ckpt}: {start_epoch}/{EPOCHS_LM} epochs done")

    def _save(epoch_done):
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch_done},
                   ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)  # atomic: a crash during save can't corrupt the checkpoint

    order = list(range(n))
    gstep = 0
    for ep in range(start_epoch, EPOCHS_LM):
        rng.shuffle(order)
        model.train()
        tot = 0.0
        for i in range(0, n, bs):
            bb = torch.tensor(order[i : i + bs])
            out = model(input_ids=ids[bb].to(device), attention_mask=attn[bb].to(device))
            loss = lossf(out.logits, y[bb].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bb)
            gstep += 1
            if CKPT_EVERY and gstep % CKPT_EVERY == 0:
                _save(ep)  # mid-epoch: weights preserved; resume re-runs the rest of epoch ep
        _save(ep + 1)
        print(f"    epoch {ep + 1}/{EPOCHS_LM}  loss={tot / n:.4f}  [ckpt]")

    def predict_fn(rows):
        model.eval()
        tids, tattn = encode(rows)
        preds = []
        with torch.no_grad():
            for i in range(0, tids.size(0), eval_bs):
                out = model(input_ids=tids[i : i + eval_bs].to(device),
                            attention_mask=tattn[i : i + eval_bs].to(device))
                preds.extend(out.logits.argmax(-1).tolist())
        return preds

    return predict_fn, model_name


# --------------------------------------------------------------------------- unified evaluation
def evaluate(predict_fn, rows):
    y = [r["label"] for r in rows]
    p = predict_fn(rows)
    by_t = collections.defaultdict(lambda: [[], []])
    by_n = collections.defaultdict(lambda: [[], []])
    for r, pi in zip(rows, p):
        by_t[r["template"]][0].append(r["label"])
        by_t[r["template"]][1].append(pi)
        by_n[r["num_variables"]][0].append(r["label"])
        by_n[r["num_variables"]][1].append(pi)
    return {"prf": f1(y, p), "by_t": by_t, "by_n": by_n}


# --------------------------------------------------------------------------- Mistral LLM baseline
INSTR = (
    "You are given a PREMISE listing all statistical (in)dependencies among some variables, and a "
    "HYPOTHESIS about their causal structure. Answer 'Yes' if the hypothesis is necessarily true in "
    "EVERY causal DAG consistent with the premise; otherwise answer 'No'. Reply with only Yes or No."
)


def _mistral_call(model, prompt, key, cache):
    import time
    import urllib.error
    import urllib.request

    ckey = hashlib.sha256(f"{model}||{prompt}".encode()).hexdigest()
    if ckey in cache:
        return cache[ckey]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 4}).encode()
    req = urllib.request.Request(
        f"{MISTRAL_API_BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = json.loads(resp.read())["choices"][0]["message"]["content"]
            cache[ckey] = text
            return text
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return ""


def make_local_predictor():
    """Load the open-weight Mistral-7B GGUF (4-bit) locally -- no API key. Returns (predict_one, label)."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(repo_id=MISTRAL_GGUF_REPO, filename=MISTRAL_GGUF_FILE)
    llm = Llama(model_path=path, n_ctx=4096, n_threads=os.cpu_count() or 4, seed=SEED, verbose=False)

    def predict_one(prompt):
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         temperature=0.0, max_tokens=4)
        return out["choices"][0]["message"]["content"]

    return predict_one, f"local {MISTRAL_GGUF_FILE.replace('.gguf', '')}"


def run_mistral(sample, fewshot, predict_one, label):
    shots = ""
    for r in fewshot:
        shots += f"{r['input']}\nAnswer: {'Yes' if r['label'] == 1 else 'No'}\n\n"
    y, p = [], []
    for i, r in enumerate(sample):
        text = predict_one(f"{INSTR}\n\n{shots}{r['input']}\nAnswer:").strip().lower()
        y.append(r["label"])
        p.append(1 if text.startswith("yes") else 0)
        if (i + 1) % 25 == 0:
            print(f"    [{label}] {i + 1}/{len(sample)}  running F1={f1(y, p)[2]:.3f}")
    return {"prf": f1(y, p), "n": len(sample)}


# --------------------------------------------------------------------------- main
def main() -> None:
    from datasets import load_dataset

    print(f"Corr2Cause — Phase 2 (learned decoupling)  seed={SEED}  smoke={SMOKE}  "
          f"run={sorted(RUN)}  ood={OOD}")
    ds = load_dataset("causalnlp/corr2cause")
    train, test = list(ds["train"]), list(ds["test"])
    variants = {"clean": test}
    for k in OOD:
        variants[k] = perturb(test, k, SEED)
    yte = [r["label"] for r in test]

    systems = {}  # name -> predict_fn(rows)->list[int]
    results = {}  # name -> {"clean":(p,r,f), <variant>:(p,r,f), "by_t":, "by_n":, "model":}

    if "ceiling" in RUN:
        systems["symbolic ceiling"] = lambda rows: [(symbolic_predict(r) or 0) for r in rows]
    if "lex" in RUN:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            cap = 8000 if SMOKE else 60000
            vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            clf = LogisticRegression(max_iter=300, class_weight="balanced").fit(
                vec.fit_transform([r["input"] for r in train[:cap]]), [r["label"] for r in train[:cap]])
            systems["lexical TF-IDF+LogReg"] = lambda rows: list(
                clf.predict(vec.transform([r["input"] for r in rows])))
        except ImportError:
            print("  (scikit-learn missing — skipping lexical baseline)")

    if "gnn" in RUN:
        print("\n>>> 2b: decoupled  parse -> GNN reasoner")
        rng = random.Random(SEED)
        sub = train[:]
        rng.shuffle(sub)
        systems["decoupled GNN (2b)"] = train_gnn(sub[:N_TRAIN_GNN])

    if "lm" in RUN:
        print("\n>>> 2a: end-to-end fine-tuned LM")
        try:
            lm_predict, lm_name = train_lm(train)
            systems["end-to-end LM (2a)"] = lm_predict
            results["end-to-end LM (2a)"] = {"model": lm_name}
        except Exception as e:  # noqa: BLE001
            print(f"  [lm] FAILED: {repr(e)[:200]}")

    # evaluate every system on clean + each OOD variant
    print("\n  evaluating systems on:", ", ".join(variants))
    for name, fn in systems.items():
        results.setdefault(name, {})
        for vname, vrows in variants.items():
            ev = evaluate(fn, vrows)
            results[name][vname] = ev["prf"]
            if vname == "clean":
                results[name]["by_t"] = ev["by_t"]
                results[name]["by_n"] = ev["by_n"]
        print(f"    {name}: done")

    # Mistral (clean sample only)
    if "mistral" in RUN and MISTRAL_BACKEND != "off":
        print("\n>>> Mistral LLM baseline (zero/few-shot)")
        key = os.environ.get("MISTRAL_API_KEY")
        backend = "api" if (MISTRAL_BACKEND == "auto" and key) else (
            "local" if MISTRAL_BACKEND == "auto" else MISTRAL_BACKEND)
        rng = random.Random(SEED)
        sample = test[:]
        rng.shuffle(sample)
        sample = sample[:N_LLM]
        shots = [r for r in train if r["num_variables"] <= 4][:FEWSHOT] or train[:FEWSHOT]
        print(f"  backend={backend}  n={len(sample)}  fewshot={len(shots)}")
        if backend == "api" and key:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cpath = os.path.join(CACHE_DIR, "mistral.json")
            cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
            for m in MISTRAL_MODELS:
                try:
                    mres = run_mistral(sample, shots, lambda p, m=m: _mistral_call(m, p, key, cache), m)
                    results[f"Mistral {m} (n={mres['n']})"] = {"clean": mres["prf"], "sample": True}
                except Exception as e:  # noqa: BLE001
                    print(f"    FAILED: {repr(e)[:160]}")
                json.dump(cache, open(cpath, "w"))
        elif backend == "local":
            try:
                predict_one, label = make_local_predictor()
                print(f"  loaded {label}")
                mres = run_mistral(sample, shots, predict_one, label)
                results[f"Mistral {label} (n={mres['n']})"] = {"clean": mres["prf"], "sample": True}
            except Exception as e:  # noqa: BLE001
                print(f"  [local] unavailable: {repr(e)[:200]}")
                print("  enable with: uv run ... --with llama-cpp-python --with huggingface_hub")
        else:
            print("  (no MISTRAL_API_KEY and backend!=local — skipped)")

    # --- main comparison table (clean) --------------------------------------
    print("\n" + "=" * 74)
    print("COMPARISON — F1 on the positive class (full 1162-test, clean; *sample* = subset)")
    print("=" * 74)
    print(f"  {'system':42s}  {'P':>6s} {'R':>6s} {'F1':>6s}")
    priority = ["majority", "lexical", "Mistral", "end-to-end LM (2a)",
                "decoupled GNN (2b)", "symbolic ceiling"]

    def rank(name):
        return next((i for i, pre in enumerate(priority) if name.startswith(pre)), len(priority))

    results["majority(0)"] = {"clean": f1(yte, [0] * len(yte))}
    for name in sorted(results, key=lambda n: (rank(n), n)):
        v = results[name]
        if "clean" not in v:
            continue
        pr, rc, ff = v["clean"]
        note = "  *sample*" if v.get("sample") else ""
        md = f"  [{v['model']}]" if v.get("model") else ""
        print(f"  {name:42s}  {pr:6.3f} {rc:6.3f} {ff:6.3f}{note}{md}")

    # --- OOD robustness table -----------------------------------------------
    if OOD:
        print("\n--- ROBUSTNESS — F1 under input perturbation (full test) ---")
        cols = ["clean"] + OOD
        print(f"  {'system':28s}  " + "  ".join(f"{c:>10s}" for c in cols))
        for name in sorted(results, key=lambda n: (rank(n), n)):
            v = results[name]
            if "clean" not in v or v.get("sample"):
                continue
            cells = [f"{v[c][2]:.3f}" if c in v else "  -- " for c in cols]
            print(f"  {name:28s}  " + "  ".join(f"{c:>10s}" for c in cells))
        print("  (parse->GNN and the symbolic oracle are relabel-INVARIANT by construction; the")
        print("   paraphrase drop is the regex-perception breaking — motivates learned perception.)")

    # --- size cut + per-template (clean) ------------------------------------
    sizes = sorted({r["num_variables"] for r in test})
    print("\n--- F1 by graph size (num_variables), clean — the size-shift tell ---")
    print(f"  {'system':22s}  " + "  ".join(f"N={s}" for s in sizes))
    for name in ("decoupled GNN (2b)", "end-to-end LM (2a)"):
        if name in results and "by_n" in results[name]:
            by_n = results[name]["by_n"]
            cells = [f"{f1(*by_n[s])[2]:.2f}" if s in by_n else " -- " for s in sizes]
            print(f"  {name:22s}  " + "    ".join(cells))

    if "decoupled GNN (2b)" in results and "by_t" in results["decoupled GNN (2b)"]:
        print("\n--- decoupled GNN per-template F1 (clean) ---")
        for t, (yy, pp) in sorted(results["decoupled GNN (2b)"]["by_t"].items()):
            print(f"    {t:22s} n={len(yy):>3d}  F1={f1(yy, pp)[2]:.3f}")


if __name__ == "__main__":
    main()
