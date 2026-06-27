# STATUS: canonical · Act 6 Frontier — Phase D part 1: the TRAINING-SCHEDULE MECHANISM on real Corr2Cause.
# Same architecture (bert-tiny perception -> GNN) + same data + same query extraction; vary ONLY the
# schedule: DECOUPLED (perception supervised on structure, reasoner on structure, composed) vs JOINT
# (the composed model trained end-to-end on text->label, no structure supervision). decoupled >> joint at
# equal capacity ⇒ the failure is the schedule, not capacity. · map: CAUSAL_LLM.md
"""Phase D (part 1) — does causal reasoning fail for lack of CAPACITY, or of TRAINING SCHEDULE?

The whole side-track's thesis is that decoupling perception from reasoning is what makes causal
reasoning learnable, and that end-to-end ("joint") training is what fails -- a *schedule* effect, not a
capacity/architecture one. The synthetic demonstration is `causal_hybrid_twostage.py` (joint ~0.43 ->
decoupled 1.0). This brings the ablation to the REAL Corr2Cause benchmark, controlling architecture:

  shared:    bert-tiny perception (text -> soft skeleton S + v-structure D) + GNN reasoner; the GNN uses
             S/D as bmm adjacency, so the perception's SOFT (sigmoid) structure feeds it differentiably.
  DECOUPLED: perception trained on STRUCTURE targets (regex S/D), GNN trained on structure -> label,
             then composed (= Phase 2b).
  JOINT:     the SAME perception + GNN trained END-TO-END on text -> label only -- no structure
             supervision; the label gradient must induce the structure on its own.

Both see the SAME N premises and the same (regex) query extraction (template, X, Y); the only difference
is whether the structure is supervised. If decoupled >> joint, the bottleneck is the schedule.

Honest scope: small models, single seed, CPU; the point is the joint-vs-decoupled GAP, not SOTA.

Run::

    DEVICE=cpu PARA_AUG=0 RELABEL_AUG=0 uv run --extra torch --with datasets --with transformers \
        --with scikit-learn python examples/causal_corr2cause_mechanism.py
"""
import os

if os.environ.get("DEVICE", "cpu") == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU on this box wedges; pure CPU
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "6")))
import causal_corr2cause_learned as L  # noqa: E402
import causal_corr2cause_perception as P  # noqa: E402
from causal_corr2cause_learned import (  # noqa: E402
    NMAX,
    TPL_ID,
    evaluate,
    make_gnn,
    train_gnn_model,
)
from causal_corr2cause_perception import (  # noqa: E402
    PERC_MODEL,
    PERC_TOKENIZER,
    make_percept_predictor,
    make_perception,
    premise_of,
    train_perception,
    variables_of,
)
from datasets import load_dataset  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SEED = 0
DEVICE = "cpu" if os.environ.get("DEVICE", "cpu") == "cpu" else "cuda"
N = int(os.environ.get("N_MECH", "6000"))         # SAME data budget for both schedules
EPOCHS_JOINT = int(os.environ.get("EPOCHS_JOINT", "5"))
if L.SMOKE:
    N, EPOCHS_JOINT = 1200, 2


def query_ok(r):
    prem = premise_of(r["input"])
    variables = variables_of(prem)
    if not variables:
        return False
    hyp = r["input"].split("Hypothesis:", 1)[1] if "Hypothesis:" in r["input"] else ""
    x, y = L.parse_hyp(r["template"], hyp)
    return x in variables and y in variables


def build_query(batch, premises):
    b = len(batch)
    nf = torch.zeros(b, NMAX, 2, device=DEVICE)
    tpl = torch.zeros(b, dtype=torch.long, device=DEVICE)
    xi = torch.zeros(b, dtype=torch.long, device=DEVICE)
    yi = torch.zeros(b, dtype=torch.long, device=DEVICE)
    ok = [False] * b
    for bi, r in enumerate(batch):
        variables = variables_of(premises[bi])
        hyp = r["input"].split("Hypothesis:", 1)[1] if "Hypothesis:" in r["input"] else ""
        x, y = L.parse_hyp(r["template"], hyp)
        if x in variables and y in variables:
            idx = {v: j for j, v in enumerate(variables)}
            nf[bi, idx[x], 0] = 1.0
            nf[bi, idx[y], 1] = 1.0
            tpl[bi] = TPL_ID[r["template"]]
            xi[bi], yi[bi] = idx[x], idx[y]
            ok[bi] = True
    return nf, tpl, xi, yi, ok


def _soft_struct(perc, premises):
    hv, present = perc.pool(premises, TOK, DEVICE)  # TOK: module global set in main()
    s, d, pm = perc.edges(hv, present)
    return torch.sigmoid(s) * pm, torch.sigmoid(d) * pm, present


def train_joint(perc, gnn, rows):
    rows = [r for r in rows if query_ok(r)][:N]
    rng = random.Random(SEED)
    y_all = [r["label"] for r in rows]
    pos = float(sum(y_all))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_all) - pos) / max(pos, 1.0)))
    opt = torch.optim.AdamW(list(perc.parameters()) + list(gnn.parameters()), lr=3e-5)
    bs, order = 32, list(range(len(rows)))
    print(f"  [joint] end-to-end perception+GNN on {len(rows)} (text->label, NO structure sup) "
          f"x {EPOCHS_JOINT} ep", flush=True)
    for ep in range(EPOCHS_JOINT):
        rng.shuffle(order)
        perc.train()
        gnn.train()
        tot, t0 = 0.0, time.time()
        for i in range(0, len(rows), bs):
            batch = [rows[j] for j in order[i : i + bs]]
            premises = [premise_of(r["input"]) for r in batch]
            S, D, present = _soft_struct(perc, premises)
            nf, tpl, xi, yi, _ = build_query(batch, premises)
            logit = gnn(S, D, nf, present, tpl, xi, yi)
            yb = torch.tensor([float(r["label"]) for r in batch])
            loss = lossf(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(batch)
        print(f"    [joint] epoch {ep + 1}/{EPOCHS_JOINT}  loss={tot / len(rows):.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    return perc, gnn


def joint_predict_fn(perc, gnn):
    def predict(rows):
        perc.eval()
        gnn.eval()
        preds = []
        for i in range(0, len(rows), 64):
            batch = rows[i : i + 64]
            premises = [premise_of(r["input"]) for r in batch]
            with torch.no_grad():
                S, D, present = _soft_struct(perc, premises)
                nf, tpl, xi, yi, ok = build_query(batch, premises)
                logit = gnn(S, D, nf, present, tpl, xi, yi).tolist()
            preds.extend(int(logit[bi] > 0) if ok[bi] else 0 for bi in range(len(batch)))
        return preds

    return predict


def main():
    global TOK
    ds = load_dataset("causalnlp/corr2cause")
    train, test = list(ds["train"]), list(ds["test"])
    rng = random.Random(SEED)
    sub = train[:]
    rng.shuffle(sub)
    sub = sub[:N]
    TOK = AutoTokenizer.from_pretrained(PERC_TOKENIZER)
    print(f"Mechanism ablation — same bert-tiny+GNN, SAME N={N}, vary only the schedule  "
          f"(para_aug={P.PARA_AUG} relabel_aug={P.RELABEL_AUG})")

    print("\n>>> DECOUPLED — perception supervised on structure + GNN on structure, composed")
    perc_d = make_perception(PERC_MODEL).to(DEVICE)
    train_perception(perc_d, sub, TOK, DEVICE)
    gnn_d = train_gnn_model(sub)  # GNN on regex structure -> label (same N)
    dec_f1 = evaluate(make_percept_predictor(perc_d, gnn_d, TOK, DEVICE), test)["prf"][2]
    ceiling_f1 = evaluate(lambda rows: L._gnn_predict(gnn_d, rows), test)["prf"][2]

    print("\n>>> JOINT — same perception+GNN trained end-to-end on text->label (no structure sup)")
    perc_j = make_perception(PERC_MODEL).to(DEVICE)
    gnn_j = make_gnn().to(DEVICE)
    train_joint(perc_j, gnn_j, sub)
    joint_f1 = evaluate(joint_predict_fn(perc_j, gnn_j), test)["prf"][2]

    print(f"\n=== MECHANISM (real Corr2Cause, same bert-tiny+GNN, N={N}) — clean test F1 ===")
    print(f"  reasoner given regex structure (upper ref)   : {ceiling_f1:.3f}")
    print(f"  DECOUPLED (structure-supervised, two-stage)  : {dec_f1:.3f}")
    print(f"  JOINT (end-to-end, label-only)               : {joint_f1:.3f}")
    print("  Reading: joint << decoupled at EQUAL capacity/data/architecture ⇒ the bottleneck is the")
    print("  TRAINING SCHEDULE (structure supervision), not capacity — the thesis, on real data.")


if __name__ == "__main__":
    main()
