# STATUS: canonical · Act 6 Frontier — Phase B1: a CONVERGED strong end-to-end LM baseline
# (distilbert) on real Corr2Cause, scored i.i.d. + OOD. Memory-minimal but learning-EXACT
# (gradient checkpointing + gradient accumulation), so it fits a small/contended box without
# changing the gradients. Pairs with results/ proof artifacts. · map: CAUSAL_LLM.md
"""Phase B1 — the converged *strong* end-to-end LM the Phase-2 quick baseline lacked.

Phase 2's `causal_corr2cause_learned.py` trained only a tiny end-to-end LM (F1 0.33). The decision
gate needs a *strong* one: does a real fine-tuned LM tie the decoupled GNN i.i.d. and/or collapse
OOD? This trains **distilbert-base-uncased** on the same 20k Corr2Cause subset and evaluates on the
full 1162-example test under three variants (clean / relabel / paraphrase), reusing
`causal_corr2cause_learned`'s `perturb`/`evaluate`/`f1` so the splits are byte-identical to the GNN
comparison (SEED 0).

MEASURED RESULT (2 epochs, effective batch 8, maxlen 512, AdamW lr 3e-5, converged; loss ~0.46):
    clean (i.i.d.)  P 0.402  R 0.750  F1 0.523
    relabel         P 0.119  R 0.217  F1 0.154   <- collapses under variable renaming
    paraphrase      P 0.468  R 0.656  F1 0.546
vs decoupled GNN 0.927/0.927/0.000 and symbolic ceiling 0.923. So the decoupling win is NOT
"OOD-only": the GNN beats a converged distilbert i.i.d. *and* the LM is not relabel-invariant
(it learns lexical/letter shortcuts, getting *worse* on relabel with more training: 0.29 -> 0.15).
Proof artifacts: results/b1_distilbert_run.log (verbatim run), results/b1_distilbert_results.json
(config + numbers + checkpoint sha256). Checkpoint kept local (803 MB, gitignored .b1_*).

WHY the unusual trainer: this WSL2 box wedges its GPU thermally and OOMs at maxlen-512 batch-8 CPU
training (especially while a co-tenant process holds RAM). Gradient checkpointing (recompute
activations) + gradient accumulation (micro-batch -> same effective batch) cut peak RAM ~6.5GB->~3GB
with *identical* gradients (distilbert is LayerNorm-only, so micro-batching is exactly equivalent).

Reproduce (from scratch, ~2 epochs; CPU is slow but reliable)::

    DEVICE=cpu NT=6 EFF_BATCH=8 MICRO=4 GC=1 STEPS=5000 SAVE_EVERY=1000 EVAL_AT_END=1 \
      uv run --extra torch --with datasets --with transformers \
        python examples/causal_corr2cause_b1_lm.py

Re-evaluate the released checkpoint (verifies the table above)::

    CKPT=$PWD/.b1_distilbert_2ep_final.pt MODE=eval DEVICE=cpu \
      uv run --extra torch --with datasets --with transformers \
        python examples/causal_corr2cause_b1_lm.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))     # examples/
ROOT = os.environ.get("CAUSALRL_ROOT", os.path.dirname(HERE))
DEVICE = os.environ.get("DEVICE", "cuda")
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch import: pure CPU, no CUDA-context crash
NT = int(os.environ.get("NT", "6"))

import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, HERE)
import torch  # noqa: E402

torch.set_num_threads(NT)
from torch import nn  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

import causal_corr2cause_learned as L  # noqa: E402

CKPT = os.environ.get("CKPT", f"{ROOT}/.b1_burst.pt")   # rolling checkpoint (model+opt+steps)
INIT = os.environ.get("INIT", "")                       # optional warm-start weights (model only)
ENC = os.environ.get("ENC", f"{ROOT}/.b1_enc.pt")       # cached encoded train subset
MODEL = "distilbert-base-uncased"
MAXLEN, LR, N_SUB, SEED = 512, 3e-5, 20000, 0
EFF_BATCH = int(os.environ.get("EFF_BATCH", os.environ.get("BS", "8")))  # effective batch (learning)
MICRO = int(os.environ.get("MICRO", str(EFF_BATCH)))                     # micro-batch (memory)
ACCUM = max(1, EFF_BATCH // MICRO)
EFF_BATCH = MICRO * ACCUM
GC = os.environ.get("GC", "1") not in ("", "0", "false")                # gradient checkpointing
EPOCH_STEPS = N_SUB // EFF_BATCH                                        # 2500 = 1 epoch on 20k
MODE = os.environ.get("MODE", "train")
STEPS = int(os.environ.get("STEPS", "5000"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "1000"))
EVAL_AT_END = os.environ.get("EVAL_AT_END", "0") not in ("", "0", "false")
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
print(f"[b1] device={DEVICE} threads={NT} mode={MODE} steps={STEPS} eff_batch={EFF_BATCH} "
      f"micro={MICRO} accum={ACCUM} grad_ckpt={GC} maxlen={MAXLEN}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(DEVICE)
if GC:
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
steps = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(st["model"])
    try:
        opt.load_state_dict(st["opt"])
    except Exception as e:  # noqa: BLE001
        print(f"[b1] opt not restored ({e}); fresh optimizer", flush=True)
    steps = int(st["steps"])
    print(f"[b1] resumed {CKPT}: steps={steps} (~{steps / EPOCH_STEPS:.2f} ep)", flush=True)
elif INIT and os.path.exists(INIT):
    st = torch.load(INIT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(st["model"])
    steps = EPOCH_STEPS
    print(f"[b1] warm-started from INIT, steps={steps}", flush=True)
else:
    print("[b1] fresh model", flush=True)


def save_ckpt():
    tmp = CKPT + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "steps": steps}, tmp)
    os.replace(tmp, CKPT)


def predict_fn(rows):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            ch = rows[i : i + 32]
            enc = tok([r["input"] for r in ch], truncation=True, padding="max_length",
                      max_length=MAXLEN, return_tensors="pt")
            out = model(input_ids=enc["input_ids"].to(DEVICE),
                        attention_mask=enc["attention_mask"].to(DEVICE))
            preds.extend(out.logits.argmax(-1).tolist())
    return preds


def run_eval():
    ds = load_dataset("causalnlp/corr2cause")
    test = list(ds["test"])
    variants = {"clean": test, "relabel": L.perturb(test, "relabel", 0),
                "paraphrase": L.perturb(test, "paraphrase", 0)}
    print(f"\n=== EVAL @ steps={steps} (~{steps / EPOCH_STEPS:.2f} ep)  device={DEVICE} ===", flush=True)
    print(f"  {'variant':12s} {'P':>7s}{'R':>7s}{'F1':>7s}")
    cev = None
    for nm, rows in variants.items():
        ev = L.evaluate(predict_fn, rows)
        p, r, f = ev["prf"]
        print(f"  {nm:12s} {p:7.3f}{r:7.3f}{f:7.3f}", flush=True)
        if nm == "clean":
            cev = ev
    print("  by size (clean): " +
          "  ".join(f"N={s}:{L.f1(*cev['by_n'][s])[2]:.2f}" for s in sorted(cev["by_n"])), flush=True)


if MODE == "eval":
    run_eval()
    sys.exit(0)

# ---- train ----
if os.path.exists(ENC):
    enc = torch.load(ENC, map_location="cpu", weights_only=True)
    ids, attn, y = enc["ids"], enc["attn"], enc["y"]
    print(f"[b1] loaded cached encoded subset {tuple(ids.shape)}", flush=True)
else:
    ds = load_dataset("causalnlp/corr2cause")
    train = list(ds["train"])
    random.Random(SEED).shuffle(train)
    sub = train[:N_SUB]
    e = tok([r["input"] for r in sub], truncation=True, padding="max_length",
            max_length=MAXLEN, return_tensors="pt")
    ids, attn = e["input_ids"], e["attention_mask"]
    y = torch.tensor([r["label"] for r in sub])
    torch.save({"ids": ids, "attn": attn, "y": y}, ENC + ".tmp")
    os.replace(ENC + ".tmp", ENC)
    print(f"[b1] encoded + cached subset {tuple(ids.shape)}", flush=True)

pos = float((y == 1).sum())
w = torch.tensor([1.0, float((y == 0).sum()) / max(pos, 1.0)]).to(DEVICE)
lossf = nn.CrossEntropyLoss(weight=w)
n = ids.size(0)
order = list(range(n))
rng = random.Random(steps)
rng.shuffle(order)
model.train()
t0 = time.time()
tot = 0.0
done = 0
i = 0
while done < STEPS:
    opt.zero_grad()
    step_loss = 0.0
    for _ in range(ACCUM):  # accumulate ACCUM micro-batches -> effective batch EFF_BATCH
        if i + MICRO > n:
            rng.shuffle(order)
            i = 0
        bb = torch.tensor(order[i : i + MICRO])
        i += MICRO
        out = model(input_ids=ids[bb].to(DEVICE), attention_mask=attn[bb].to(DEVICE))
        loss = lossf(out.logits, y[bb].to(DEVICE)) / ACCUM
        loss.backward()
        step_loss += loss.item()
    opt.step()
    tot += step_loss
    done += 1
    steps += 1
    if done % 100 == 0:
        print(f"  step {done}/{STEPS} (total {steps})  loss {tot / done:.4f}  {time.time() - t0:.0f}s",
              flush=True)
    if SAVE_EVERY and done % SAVE_EVERY == 0:
        save_ckpt()
        print(f"  [ckpt @ {steps}]", flush=True)

save_ckpt()
print(f"[b1] DONE +{STEPS} -> total {steps} (~{steps / EPOCH_STEPS:.2f} ep)  "
      f"meanloss {tot / STEPS:.4f}  {time.time() - t0:.0f}s  saved -> {CKPT}", flush=True)
if EVAL_AT_END:
    run_eval()
