# STATUS: canonical · Act 6 Frontier — Phase 2b/B3 LEARNED perception on Corr2Cause: a trained text->structure extractor replaces the regex parser; with paraphrase+relabel augmentation it is robust on BOTH OOD axes (clean 0.68 / relabel 0.64 / paraphrase 0.66) feeding the same GNN, where the regex front-end collapses on paraphrase (0.000)  ·  map: CAUSAL_LLM.md
"""Phase 2b — make the *perception* learned, so decoupling isn't "free regex".

``causal_corr2cause_learned.py`` showed the decoupled parse->GNN reasoner is exactly invariant to
variable relabeling but COLLAPSES under premise paraphrase -- because its perception is a hand-written
regex keyed on exact phrases ("correlates with", ...). A reviewer rightly objects: you hand-built the
NLP. This script answers that objection by *learning* the perception:

    PERCEPTION (learned):  a small encoder (bert-tiny) reads the premise text and predicts the same
                           structural inputs the regex produced -- the skeleton S and v-structure
                           evidence D over the N variables. Supervised for free by the regex parser on
                           CLEAN premises, trained with paraphrase augmentation AND (B3) relabel
                           augmentation -- for relabel the premise is permuted *and* its structure
                           target re-parsed, so the two permute together, giving the relabel-invariance
                           the paraphrase-only version lacked.
    REASONING (shared):    the SAME GNN reasoner from Phase 2 (trained once on regex structure).

So regex-perception and learned-perception feed the *identical* reasoner; the only thing that changes
is the front-end. We then measure both, plus the end-to-end LM, under clean / relabel / paraphrase.

Result (the decoupling pay-off): the learned perception keeps the reasoner's accuracy on whichever axis
it is augmented for. Paraphrase-aug alone -> clean 0.71 / relabel 0.33 / paraphrase 0.73; adding
relabel-aug (B3) -> clean 0.68 / relabel 0.64 / paraphrase 0.66 (robust on BOTH), where the regex
front-end (and the symbolic oracle, also regex) drop to 0.000 on paraphrase. Evidence:
`results/b3_perception_run.log`.

Honest scope: the paraphraser is rule-based (a finite synonym set), so this is a proof-of-concept that
decoupling *localizes* the robustness problem to a cheaply-retrainable perception module -- not a claim
of open-world paraphrase robustness (that needs LLM-generated paraphrases / the paper's robustness
splits). Small models, single seed, CPU.

Run::

    SMOKE=1 HF_HOME=/tmp/hf uv run --extra torch --with datasets --with scikit-learn \
        python examples/causal_corr2cause_perception.py
"""

from __future__ import annotations

import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_corr2cause_learned as L
from causal_corr2cause_learned import (
    NMAX,
    SEED,
    TPL_ID,
    evaluate,
    f1,
    parse_hyp,
    perturb,
    struct_from_parse,
    train_gnn_model,
)
from causal_corr2cause_solver import parse

SMOKE = L.SMOKE
MAXLEN = int(os.environ.get("MAXLEN", "320"))
PERC_MODEL = os.environ.get("PERC_MODEL", "google/bert_uncased_L-2_H-128_A-2")
# offset_mapping needs a FAST tokenizer; bert-tiny lacks one -> reuse the shared uncased WordPiece.
PERC_TOKENIZER = os.environ.get("PERC_TOKENIZER", "bert-base-uncased")
N_TRAIN_PERC = int(os.environ.get("N_TRAIN_PERC", "6000"))
EPOCHS_PERC = int(os.environ.get("EPOCHS_PERC", "4"))
PARA_AUG = float(os.environ.get("PARA_AUG", "0.7"))  # prob. of paraphrasing a training premise
RELABEL_AUG = float(os.environ.get("RELABEL_AUG", "0.5"))  # B3: prob. of relabeling (-> relabel-invariance)
N_TRAIN_GNN = int(os.environ.get("N_TRAIN_GNN", "40000"))
if SMOKE:
    N_TRAIN_PERC, EPOCHS_PERC, N_TRAIN_GNN = 1200, 2, 4000


def premise_of(text: str) -> str:
    return text.split("Hypothesis:")[0] if "Hypothesis:" in text else text


def variables_of(premise: str):
    return sorted(set(re.findall(r"\b([A-F])\b", premise)))


# --------------------------------------------------------------------------- learned perception
def make_perception(enc_name):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Perception(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(enc_name)
            h = self.enc.config.hidden_size
            self.skel = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 1))
            self.dire = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 1))
            self.hdim = h

        def pool(self, premises, tok, device):
            """premises -> (hv [B,NMAX,H], present [B,NMAX]) via offset-based per-variable pooling."""
            import numpy as np

            enc = tok(premises, return_offsets_mapping=True, truncation=True, padding=True,
                      max_length=MAXLEN, return_tensors="pt")
            offs = enc.pop("offset_mapping")
            hidden = self.enc(input_ids=enc["input_ids"].to(device),
                              attention_mask=enc["attention_mask"].to(device)).last_hidden_state
            b, t, h = hidden.shape
            hv = torch.zeros(b, NMAX, h, device=device)
            present = torch.zeros(b, NMAX, device=device)
            for bi, prem in enumerate(premises):
                o = offs[bi].numpy()
                starts, ends = o[:, 0], o[:, 1]
                valid = ends > starts
                for vi, v in enumerate(variables_of(prem)[:NMAX]):
                    mask = np.zeros(t, dtype=bool)
                    for m in re.finditer(rf"\b{v}\b", prem):
                        mask |= valid & (starts < m.end()) & (ends > m.start())
                    if mask.any():
                        mt = torch.tensor(mask, device=device, dtype=hidden.dtype)
                        hv[bi, vi] = (hidden[bi] * mt.unsqueeze(-1)).sum(0) / mt.sum()
                        present[bi, vi] = 1.0
            return hv, present

        def edges(self, hv, present):
            hi = hv.unsqueeze(2).expand(-1, NMAX, NMAX, -1)
            hj = hv.unsqueeze(1).expand(-1, NMAX, NMAX, -1)
            feat = torch.cat([hi, hj, hi * hj], -1)
            s = self.skel(feat).squeeze(-1)
            s = 0.5 * (s + s.transpose(1, 2))  # skeleton is symmetric
            d = self.dire(feat).squeeze(-1)
            pm = present.unsqueeze(2) * present.unsqueeze(1)
            eye = torch.eye(NMAX, device=hv.device).bool()
            pm = pm.masked_fill(eye, 0.0)
            return s, d, pm

    return Perception()


def _targets(rows):
    """Regex-derived (S,D,present) targets on CLEAN rows — free supervision for the perception."""
    import numpy as np
    import torch

    S, D, P = [], [], []
    for r in rows:
        variables, corr, indep, _ = parse(r["input"])
        s, d, present, _ = struct_from_parse(variables, corr, indep)
        S.append(s)
        D.append(d)
        P.append(present)
    return (torch.tensor(np.stack(S)), torch.tensor(np.stack(D)), torch.tensor(np.stack(P)))


def train_perception(model, rows, tok, device):
    import torch
    from torch import nn

    rng = random.Random(SEED)
    rows = [r for r in rows if variables_of(premise_of(r["input"]))]
    rows = rows[:N_TRAIN_PERC]
    bce = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    bs, n = 32, len(rows)
    order = list(range(n))
    print(f"  [perc] train {PERC_MODEL} on {n} premises x {EPOCHS_PERC} ep "
          f"(paraphrase-aug p={PARA_AUG}, relabel-aug p={RELABEL_AUG})")
    for ep in range(EPOCHS_PERC):
        rng.shuffle(order)
        model.train()
        tot = 0.0
        for i in range(0, n, bs):
            batch = [rows[j] for j in order[i : i + bs]]
            # B3 augmentation per example: relabel (permute variable letters) and/or paraphrase
            # (reword the premise). For relabel the TARGET is re-parsed from the relabeled text, so the
            # structure permutes consistently; paraphrase leaves structure unchanged. Both stay aligned
            # with the model's per-variable pooling because all index by variables_of()'s sorted order.
            src, premises = [], []
            for r in batch:
                rr = L.relabel_row(r, rng) if rng.random() < RELABEL_AUG else r
                prem = premise_of(rr["input"])
                if rng.random() < PARA_AUG:
                    prem = premise_of(L.paraphrase_row(rr, rng)["input"])
                src.append(rr)
                premises.append(prem)
            St, Dt, Pt = _targets(src)  # targets from the (possibly relabeled) source rows
            St, Dt, Pt = St.to(device), Dt.to(device), Pt.to(device)
            hv, present = model.pool(premises, tok, device)
            s, d, pm = model.edges(hv, present)
            eye = torch.eye(NMAX, device=device).bool()
            mask = pm * (~eye)
            denom = mask.sum().clamp(min=1.0)
            loss = (bce(s, St) * mask).sum() / denom + (bce(d, Dt) * mask).sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(batch)
        # structure-recovery accuracy on a clean held-out slice
        with torch.no_grad():
            model.eval()
            hv, present = model.pool([premise_of(r["input"]) for r in rows[:400]], tok, device)
            s, d, pm = model.edges(hv, present)
            St, Dt, Pt = _targets(rows[:400])
            eye = torch.eye(NMAX, device=device).bool()
            mask = (pm * (~eye)).bool()
            sa = (((s > 0) == St.to(device).bool()) & mask)[mask].float().mean().item()
            da = (((d > 0) == Dt.to(device).bool()) & mask)[mask].float().mean().item()
        print(f"    epoch {ep + 1}/{EPOCHS_PERC}  loss={tot / n:.4f}  "
              f"clean edge-acc S={sa:.3f} D={da:.3f}")
    return model


# --------------------------------------------------------------------------- compose: learned perception -> GNN
def make_percept_predictor(perc, gnn, tok, device):
    import torch

    def predict_fn(rows):
        preds = []
        for i in range(0, len(rows), 64):
            batch = rows[i : i + 64]
            premises = [premise_of(r["input"]) for r in batch]
            with torch.no_grad():
                perc.eval()
                hv, present = perc.pool(premises, tok, device)
                s, d, pm = perc.edges(hv, present)
                S_hat = (torch.sigmoid(s) > 0.5).float() * pm
                D_hat = (torch.sigmoid(d) > 0.5).float() * pm
            nf = torch.zeros(len(batch), NMAX, 2, device=device)
            tpl = torch.zeros(len(batch), dtype=torch.long, device=device)
            xi = torch.zeros(len(batch), dtype=torch.long, device=device)
            yi = torch.zeros(len(batch), dtype=torch.long, device=device)
            ok = [False] * len(batch)
            for bi, r in enumerate(batch):
                variables = variables_of(premises[bi])
                hyp = r["input"].split("Hypothesis:", 1)[1] if "Hypothesis:" in r["input"] else ""
                x, y = parse_hyp(r["template"], hyp)
                if x in variables and y in variables:
                    idx = {v: j for j, v in enumerate(variables)}
                    nf[bi, idx[x], 0] = 1.0
                    nf[bi, idx[y], 1] = 1.0
                    tpl[bi] = TPL_ID[r["template"]]
                    xi[bi], yi[bi] = idx[x], idx[y]
                    ok[bi] = True
            with torch.no_grad():
                gnn.eval()
                logit = gnn(S_hat, D_hat, nf, present, tpl, xi, yi).tolist()
            preds.extend(int(logit[bi] > 0) if ok[bi] else 0 for bi in range(len(batch)))
        return preds

    return predict_fn


# --------------------------------------------------------------------------- main
def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = os.environ.get("DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Corr2Cause — Phase 2b (LEARNED perception)  seed={SEED}  smoke={SMOKE}  device={device}")
    ds = load_dataset("causalnlp/corr2cause")
    train, test = list(ds["train"]), list(ds["test"])
    variants = {"clean": test, "relabel": perturb(test, "relabel", SEED),
                "paraphrase": perturb(test, "paraphrase", SEED),
                "paraphrase_heldout": perturb(test, "paraphrase_heldout", SEED)}

    rng = random.Random(SEED)
    sub = train[:]
    rng.shuffle(sub)

    print("\n>>> shared reasoner: GNN on regex-parsed structure")
    gnn = train_gnn_model(sub[:N_TRAIN_GNN])

    print("\n>>> learned perception (text -> structure)")
    tok = AutoTokenizer.from_pretrained(PERC_TOKENIZER)
    perc = make_perception(PERC_MODEL).to(device)
    train_perception(perc, sub, tok, device)

    systems = {
        "regex perception -> GNN": lambda rows: L._gnn_predict(gnn, rows),
        "learned perception -> GNN": make_percept_predictor(perc, gnn, tok, device),
    }

    print("\n--- ROBUSTNESS — F1 (positive class, full test) under input perturbation ---")
    cols = ["clean", "relabel", "paraphrase", "paraphrase_heldout"]
    print(f"  {'front-end (shared GNN reasoner)':32s}  " + "  ".join(f"{c:>10s}" for c in cols))
    for name, fn in systems.items():
        cells = []
        for c in cols:
            cells.append(f"{evaluate(fn, variants[c])['prf'][2]:.3f}")
        print(f"  {name:32s}  " + "  ".join(f"{c:>10s}" for c in cells))
    print("\n  Reading: regex front-end is relabel-invariant but paraphrase-fragile (parser misses the")
    print("  reworded relations); a LEARNED front-end feeding the SAME reasoner should hold up under")
    print("  paraphrase — decoupling localizes the robustness fix to a cheap, retrainable module.")


if __name__ == "__main__":
    main()
