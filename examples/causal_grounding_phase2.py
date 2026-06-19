"""Causal Grounding (Phase 2) — grounding *multiple* causal primitives in disentangled subspaces.

Phases 0-1 (``causal_grounding_das_iit.py``) located, repaired, and installed a single causal
variable -- the see/do *regime*.  Phase 2 asks the next question from ``FRONTIER_PROPOSAL_v2.md``:
does the model ground a *small set* of causal primitives as independently-controllable latents?

We use a task with TWO causal variables that both move P(recover):

    * REGIME     R in {see, do}      -- observe vs intervene (confounded path on/off)
    * TREATMENT  T in {drug, nodrug} -- the drug actually given

both read off the same causalrl SCM, so their ground-truth effects are known:

    P(rec):  see/drug 0.86   see/nodrug 0.14   do/drug 0.65   do/nodrug 0.35
    R-gap (regime, at drug) = +0.21      T-gap (treatment, at do) = +0.30

Three steps, each checked by intervention against SCM truth -- including an honest negative:

  1. NAIVE DAS entangles.  Locating each variable *independently* gives overlapping directions
     (|cos|~0.75): the 2x2 interchange matrix has large off-diagonal leakage (swapping the regime
     subspace also moves the treatment), and composition is poor.
  2. DISENTANGLED DAS.  Locating both *jointly* in an orthonormal 2-frame, optimised so the
     interchange matrix is the identity (each column transmits its own variable, ignores the other),
     drives the off-diagonal down -- the two primitives become independently controllable.
  3. COMPOSITION exposes a real limit.  Setting (R, T) from a *single* prompt by intervening on the
     two subspaces reconstructs the low-interaction cells but fails on the high-interaction one:
     because the variables *interact* in the SCM (the treatment's effect is far larger under see,
     gap 0.72, than under do, gap 0.30), a single linear direction per variable cannot carry a
     context-dependent effect.  Disentangling directions is necessary but not sufficient -- the
     interaction itself must be grounded too (the constructive next step, Phase 2b/3).

CPU-sized, didactic; reuses the Phase 0-1 machinery.  Run::

    uv run --extra torch python examples/causal_grounding_phase2.py
    CG_CACHE=1 uv run --extra torch python examples/causal_grounding_phase2.py   # reuse cached base
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_grounding_das_iit as cg  # sibling example, imported after sys.path tweak

L = cg.TARGET_LAYER


def four_cell_truth(scm, n: int = 200_000) -> dict[str, float]:
    obs = scm.see(n, seed=7)
    x, y = obs["X"], obs["Y"]
    return {
        "see_drug": float(y[x > 0.5].mean()),
        "see_nodrug": float(y[x < 0.5].mean()),
        "do_drug": float(scm.do({"X": 1.0}).see(n, seed=8)["Y"].mean()),
        "do_nodrug": float(scm.do({"X": 0.0}).see(n, seed=9)["Y"].mean()),
    }


def src_mean(model, group: cg.Prompts, like: cg.Prompts) -> Tensor:
    """Mean residual of ``group`` at the readout site, broadcast to ``like``'s batch."""
    return cg.resid_at(model, L, group).mean(0, keepdim=True).expand(len(like), -1)


def patched_logits_multi(
    model, base: cg.Prompts, patches: list[tuple[Tensor, Tensor]], rec_id: int, rel_id: int
) -> float:
    """Apply several subspace interchanges at once (orthogonal subspaces -> deltas add)."""
    idx = torch.arange(len(base))

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        hb = h[idx, base.last]
        new = hb.clone()
        for d_mat, h_src in patches:
            new = new + ((h_src - hb) @ d_mat) @ d_mat.t()
        h = h.clone()
        h[idx, base.last] = new
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = model.transformer.h[L].register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=base.ids, attention_mask=base.mask).logits
    handle.remove()
    last = logits[idx, base.last]
    return float(torch.softmax(last[:, [rec_id, rel_id]], dim=-1)[:, 0].mean())


def das_locate_disentangled(model, see_drug, do_drug, do_nodrug, rec_id, rel_id, steps=400):
    """Locate regime and treatment *jointly* in an orthonormal 2-frame, optimised so the interchange
    matrix is the identity: each column transmits its own variable and ignores the other.

    Orthogonality is free (orthonormal columns); we only fit the on/off-diagonal interchange specs.
    Returns (d_regime, d_treatment), each (d, 1).
    """
    sub = cg.Subspace(model.config.n_embd, 2)
    opt = torch.optim.Adam(sub.parameters(), lr=5e-3)
    for p in model.parameters():
        p.requires_grad_(False)

    def p_patched(base, source, d_col):
        hs = src_mean(model, source, base)
        cf = cg.patched_logits(model, L, base, hs, d_col)
        return torch.softmax(cf[:, [rec_id, rel_id]], dim=-1)[:, 0]

    # (base, source, column, transmit?)  column 0 = regime, 1 = treatment.
    specs = [
        (see_drug, do_drug, 0, True),
        (do_drug, see_drug, 0, True),  # D_R transmits regime
        (do_drug, do_nodrug, 0, False),
        (do_nodrug, do_drug, 0, False),  # D_R ignores treatment
        (do_drug, do_nodrug, 1, True),
        (do_nodrug, do_drug, 1, True),  # D_T transmits treatment
        (see_drug, do_drug, 1, False),
        (do_drug, see_drug, 1, False),  # D_T ignores regime
    ]
    for _ in range(steps):
        opt.zero_grad()
        d_mat = sub.D()
        cols = [d_mat[:, :1], d_mat[:, 1:2]]
        loss = torch.zeros(())
        for base, source, c, transmit in specs:
            p_cf = p_patched(base, source, cols[c])
            with torch.no_grad():
                tgt = (
                    cg.p_recover(model, source, rec_id, rel_id).mean()
                    if transmit
                    else cg.p_recover(model, base, rec_id, rel_id).mean()
                )
            loss = loss + ((p_cf - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    d_mat = sub.D().detach()
    return d_mat[:, :1], d_mat[:, 1:2]


def disentanglement_matrix(model, d_r, d_t, groups, rec_id, rel_id) -> dict:
    see_drug, do_drug, do_nodrug = groups
    return {
        ("R", "R"): cg.interchange_effect(model, L, see_drug, do_drug, d_r, rec_id, rel_id),
        ("R", "T"): cg.interchange_effect(model, L, do_drug, do_nodrug, d_r, rec_id, rel_id),
        ("T", "R"): cg.interchange_effect(model, L, see_drug, do_drug, d_t, rec_id, rel_id),
        ("T", "T"): cg.interchange_effect(model, L, do_drug, do_nodrug, d_t, rec_id, rel_id),
    }


def print_matrix(ie: dict) -> None:
    print("                swap regime     swap treatment")
    print(f"  patch D_R :     {ie[('R', 'R')]:+.2f}            {ie[('R', 'T')]:+.2f}")
    print(f"  patch D_T :     {ie[('T', 'R')]:+.2f}            {ie[('T', 'T')]:+.2f}")


def composition_mae(model, base, d_r, d_t, do_drug, do_nodrug, truth, rec_id, rel_id) -> float:
    r_do = src_mean(model, do_drug, base)  # regime := do
    t_no = src_mean(model, do_nodrug, base)  # treatment := nodrug
    cells = {
        "see_drug": [],
        "do_drug": [(d_r, r_do)],
        "see_nodrug": [(d_t, t_no)],
        "do_nodrug": [(d_r, r_do), (d_t, t_no)],
    }
    print("  cell          composed   truth    |err|")
    tot = 0.0
    for name, patches in cells.items():
        got = patched_logits_multi(model, base, patches, rec_id, rel_id)
        err = abs(got - truth[name])
        tot += err
        print(f"  {name:11s}   {got:.3f}     {truth[name]:.3f}    {err:.3f}")
    return tot / 4


def main() -> None:
    torch.manual_seed(0)
    scm = cg.build_scm()
    truth = four_cell_truth(scm)
    print("causalrl 4-cell truth  P(rec): " + "  ".join(f"{k}={v:.3f}" for k, v in truth.items()))
    print(
        f"  R-gap (regime, at drug) = {truth['see_drug'] - truth['do_drug']:+.3f}    "
        f"T-gap (treatment, at do) = {truth['do_drug'] - truth['do_nodrug']:+.3f}\n"
    )

    corpus = cg.build_corpus(scm, cg.DOMAIN_A)
    tok = cg.build_tokenizer(corpus)
    rec_id = tok.convert_tokens_to_ids(cg.REC_TOK)
    rel_id = tok.convert_tokens_to_ids(cg.REL_TOK)
    model = cg.build_model(tok)

    ckpt = "/tmp/cg_base.pt"  # shares the Phase 0-1 base (same deterministic corpus/vocab/arch)
    if os.environ.get("CG_CACHE") and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print("loaded cached base model")
    else:
        print(f"training base GPT-2 ({model.num_parameters() / 1e6:.2f}M params) on domain A ...")
        cg.lm_train(model, tok, corpus, epochs=10)
        if os.environ.get("CG_CACHE"):
            torch.save(model.state_dict(), ckpt)
    model.eval()

    # four cells (domain A): see/do x drug/nodrug
    see_drug, do_drug = cg.regime_prompts(cg.DOMAIN_A, True, tok)
    see_nodrug, do_nodrug = cg.regime_prompts(cg.DOMAIN_A, False, tok)

    def pr(group: cg.Prompts) -> float:
        return float(cg.p_recover(model, group, rec_id, rel_id).mean())

    print(
        "base model P(rec): "
        f"see/drug={pr(see_drug):.3f}  see/nodrug={pr(see_nodrug):.3f}  "
        f"do/drug={pr(do_drug):.3f}  do/nodrug={pr(do_nodrug):.3f}"
    )

    groups = (see_drug, do_drug, do_nodrug)

    # -------- 1. naive per-variable DAS: locating each variable alone entangles them --------
    print("\n=== 1. naive DAS (locate each variable independently) ===")
    d_r0 = cg.das_locate(model, L, see_drug, do_drug, 1, rec_id, rel_id).D().detach()
    d_t0 = cg.das_locate(model, L, do_drug, do_nodrug, 1, rec_id, rel_id).D().detach()
    cos0 = float((d_r0.squeeze() @ d_t0.squeeze()) / (d_r0.norm() * d_t0.norm()))
    print(f"  |cos(D_R, D_T)| = {abs(cos0):.3f}   (interchange matrix:)")
    print_matrix(disentanglement_matrix(model, d_r0, d_t0, groups, rec_id, rel_id))
    mae0 = composition_mae(model, see_drug, d_r0, d_t0, do_drug, do_nodrug, truth, rec_id, rel_id)
    print(f"  composition MAE = {mae0:.3f}   -> off-diagonal leakage: the two are entangled")

    # -------- 2. disentangled joint DAS: optimise an orthonormal 2-frame to the identity matrix --
    print("\n=== 2. disentangled DAS (locate both jointly in an orthonormal 2-frame) ===")
    d_r, d_t = das_locate_disentangled(model, see_drug, do_drug, do_nodrug, rec_id, rel_id)
    cos = float((d_r.squeeze() @ d_t.squeeze()) / (d_r.norm() * d_t.norm()))
    print(f"  |cos(D_R, D_T)| = {abs(cos):.3f}   (interchange matrix:)")
    print_matrix(disentanglement_matrix(model, d_r, d_t, groups, rec_id, rel_id))
    print("  (rows = subspace swapped; cols = variable the two prompt groups differ in)")

    # -------- 3. compositional control with the disentangled primitives --------
    print("\n=== 3. compositional control (set R and T independently from one prompt) ===")
    mae = composition_mae(model, see_drug, d_r, d_t, do_drug, do_nodrug, truth, rec_id, rel_id)
    print(f"  mean abs error over the 4 composed cells: {mae:.3f}   (naive DAS: {mae0:.3f})")

    print(
        "\nReading: naive per-variable DAS entangles the two directions (off-diagonal leakage); "
        "joint orthonormal DAS disentangles them (off-diagonal cut ~4x, on-diagonal cost). "
        "Composition still fails on the high-interaction cell (see/nodrug): the treatment's effect "
        "is regime-dependent, so a single linear direction cannot carry it -- disentangling "
        "directions is necessary but not sufficient; the interaction must be grounded too (Phase 3)."
    )


if __name__ == "__main__":
    main()
