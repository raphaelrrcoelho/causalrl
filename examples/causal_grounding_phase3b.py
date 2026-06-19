"""Causal Grounding (Phase 3b) — *co-trained* carriers to close the interaction.

Phase 3 grounded the regime/treatment interaction only *partially* with two FROZEN 1-D carriers:
composition MAE fell ~40% but did not close, and see/drug regressed. Diagnosis: frozen carriers give
the optimiser no freedom to pick axes that both disentangle tightly AND let the downstream compute
the interaction, and any residual off-diagonal leakage (~0.12-0.14) corrupts composition.

Phase 3b removes that constraint: the orthonormal 2-frame (e_R, e_T) is **warm-started** from the
Phase-2 disentangled frame and then **co-trained with the model** (still orthonormal by construct.).
Same fixed-SCM-truth specs as Phase 3 -- behavioural anchor (each real cell -> truth), disentangle
specs (each carrier transmits its own variable, ignores the other), and composition (set (R,T) via
the two carriers -> the target cell's truth) -- with a heavier behavioural weight so the natural
cells are not sacrificed. Test: does composing interventions on the two carriers reconstruct the
4-cell causal table, including the high-interaction see/nodrug, and does it transfer OOD?

RESULT (negative): it does NOT close -- co-training collapses to a near-constant output (degenerate
minimum, IE=0, MAE ~0.26, worse than frozen Phase 3). Trainable carriers reintroduce the very
collapse that Phases 1b/3 avoided by freezing. Freezing is load-bearing; this run documents why.

CPU-sized; reuses Phases 0-3.  Run::

    CG_CACHE=1 uv run --extra torch python examples/causal_grounding_phase3b.py
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_grounding_das_iit as cg  # sibling examples, imported after sys.path tweak
import causal_grounding_phase2 as p2
import causal_grounding_phase3 as p3

L = cg.TARGET_LAYER


def iit3b_cotrain(
    model,
    tok,
    corpus,
    carriers,
    truth,
    domains,
    rec_id,
    rel_id,
    epochs=6,
    lr=1e-4,
    batch_size=64,
    lam_beh=2.0,
    lam_dis=1.0,
    lam_comp=2.0,
):
    """Co-train the orthonormal carriers together with the model (carriers stay orthonormal)."""
    cells = {name: p3.cells_of(dom, tok) for name, dom in domains}
    enc = [tok(s + cg.EOS_TOK).input_ids for s in corpus]
    pad_id = tok.pad_token_id
    opt = torch.optim.AdamW([*model.parameters(), *carriers.parameters()], lr=lr)
    rng = random.Random(3)

    def src(group):
        return p2.src_mean(model, group, group).detach()

    for epoch in range(epochs):
        rng.shuffle(enc)
        t_lm = t_beh = t_dis = t_comp = 0.0
        nb = 0
        for i in range(0, len(enc), batch_size):
            batch = enc[i : i + batch_size]
            width = max(len(s) for s in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for j, s in enumerate(batch):
                ids[j, : len(s)] = torch.tensor(s)
                mask[j, : len(s)] = 1
            labels = ids.masked_fill(mask == 0, -100)
            lm = model(input_ids=ids, attention_mask=mask, labels=labels).loss

            d_mat = carriers.D()  # recomputed each step -> grads flow into the carriers
            e_r, e_t = d_mat[:, :1], d_mat[:, 1:2]

            beh = torch.zeros(())
            dis = torch.zeros(())
            comp = torch.zeros(())
            for c in cells.values():
                for name, group in c.items():
                    p_beh = cg.p_recover_grad(model, group, rec_id, rel_id)
                    beh = beh + (p_beh - truth[name]).pow(2).mean()

                sd, dd, dn = c["see_drug"], c["do_drug"], c["do_nodrug"]
                dis_specs = [
                    (sd, dd, e_r, truth["do_drug"]),
                    (dd, sd, e_r, truth["see_drug"]),
                    (dd, dn, e_r, truth["do_drug"]),
                    (dn, dd, e_r, truth["do_nodrug"]),
                    (dd, dn, e_t, truth["do_nodrug"]),
                    (dn, dd, e_t, truth["do_drug"]),
                    (sd, dd, e_t, truth["see_drug"]),
                    (dd, sd, e_t, truth["do_drug"]),
                ]
                for base, source, carrier, tgt in dis_specs:
                    p_cf = p3.comp_p_grad(model, base, [(carrier, src(source))], rec_id, rel_id)
                    dis = dis + (p_cf - tgt).pow(2).mean()

                r_do, t_no = src(dd), src(dn)
                comp_specs = [
                    (sd, [], "see_drug"),
                    (sd, [(e_r, r_do)], "do_drug"),
                    (sd, [(e_t, t_no)], "see_nodrug"),
                    (sd, [(e_r, r_do), (e_t, t_no)], "do_nodrug"),
                ]
                for base, patches, name in comp_specs:
                    p_cf = p3.comp_p_grad(model, base, patches, rec_id, rel_id)
                    comp = comp + (p_cf - truth[name]).pow(2).mean()

            loss = lm + lam_beh * beh + lam_dis * dis + lam_comp * comp
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_lm += float(lm.detach())
            t_beh += float(beh.detach())
            t_dis += float(dis.detach())
            t_comp += float(comp.detach())
            nb += 1
        print(
            f"    epoch {epoch + 1}/{epochs}  lm {t_lm / nb:.3f}  beh {t_beh / nb:.4f}  "
            f"disent {t_dis / nb:.4f}  comp {t_comp / nb:.4f}"
        )


def main() -> None:
    torch.manual_seed(0)
    scm = cg.build_scm()
    truth = p2.four_cell_truth(scm)
    print("causalrl 4-cell truth: " + "  ".join(f"{k}={v:.3f}" for k, v in truth.items()) + "\n")

    corpus = cg.build_corpus(scm, cg.DOMAIN_A)
    tok = cg.build_tokenizer(corpus)
    rec_id = tok.convert_tokens_to_ids(cg.REC_TOK)
    rel_id = tok.convert_tokens_to_ids(cg.REL_TOK)
    model = cg.build_model(tok)

    ckpt = "/tmp/cg_base.pt"
    if os.environ.get("CG_CACHE") and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print("loaded cached base model")
    else:
        print("training base GPT-2 on domain A ...")
        cg.lm_train(model, tok, corpus, epochs=10)
        if os.environ.get("CG_CACHE"):
            torch.save(model.state_dict(), ckpt)
    model.eval()

    # warm-start an orthonormal 2-frame from Phase 2's disentangled carriers
    c = p3.cells_of(cg.DOMAIN_A, tok)
    e_r, e_t = p2.das_locate_disentangled(
        model, c["see_drug"], c["do_drug"], c["do_nodrug"], rec_id, rel_id
    )
    carriers = cg.Subspace(model.config.n_embd, 2)
    carriers.warm_start(torch.cat([e_r.t(), e_t.t()], 0))  # rows = the two carrier directions

    print("\n=== before Phase 3b (base model + Phase-2 carriers) ===")
    p3.report(model, tok, e_r, e_t, truth, rec_id, rel_id, "base")

    print("\n=== Phase 3b: co-train carriers + model (IIT) ===")
    domains = [("A", cg.DOMAIN_A), ("C", cg.DOMAIN_C)]
    iit3b_cotrain(model, tok, corpus, carriers, truth, domains, rec_id, rel_id, epochs=6)
    model.eval()

    d_mat = carriers.D().detach()
    e_r, e_t = d_mat[:, :1], d_mat[:, 1:2]
    print("\n=== after Phase 3b ===")
    p3.report(model, tok, e_r, e_t, truth, rec_id, rel_id, "iit3b")

    print(
        "\nReading (NEGATIVE result): co-training the carriers does NOT close the interaction -- "
        "it COLLAPSES. The optimiser hits a degenerate flat minimum (losses freeze after epoch 1), "
        "outputs a near-constant P(rec)~0.60 for every cell, carriers carry nothing (IE=0), and "
        "composition MAE rises to ~0.26 -- worse than frozen Phase 3 (0.10) and even the base. "
        "Trainable carriers reintroduce the collapse that Phases 1b/3 avoided by FREEZING them. "
        "Lesson: freezing is load-bearing. Candidate levers (untested): higher-dim FROZEN "
        "carriers, training only the downstream, or scale."
    )


if __name__ == "__main__":
    main()
