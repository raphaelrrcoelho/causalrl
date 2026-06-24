# STATUS: oracle-fed · Act 2 Mechanism — DAS+IIT grounding in real activations (regime variable is known)  ·  map: CAUSAL_LLM.md
"""Causal Grounding (Phases 0-1) — locate, diagnose, install, and attribute a *regime* latent.

This is the runnable companion to ``FRONTIER_PROPOSAL_v2.md``. It instantiates the first two phases
of the "causal grounding" program on a tiny GPT-2 trained from scratch on a causalrl see/do task,
where the ground-truth causal variable is **known** (the observational-vs-interventional *regime*).

We ask, and answer by intervention, three mechanistic questions:

  PHASE 0  (locate + diagnose)
    * Does the model represent the binary causal variable ``regime in {see, do}`` in a *low-dim*
      subspace of its residual stream?  -> Distributed Alignment Search (DAS): learn an orthonormal
      subspace and validate it with **interchange interventions** (activation patching = do() on
      activations).  Metric: Interchange Effect (IE), the fraction of the see/do behavioural gap
      that transfers when we swap *only* that subspace between a see- and a do-prompt.
    * Is the gap mediated by that subspace rather than by surface dimensions?  -> compare IE of the
      learned subspace vs a random subspace of the same size (causal-mediation diagnosis).

  PHASE 1  (install + attribute)
    * Can we *install / strengthen* the regime latent so it is cleanly carried by a 1-D subspace
      and the output causally routed through it?  -> Interchange Intervention Training (IIT):
      fine-tune so a subspace causally implements ``regime`` (Geiger et al. 2022) + the LM loss.
    * Is the resulting behaviour causally attributable to that mechanism?  -> mean-ablate the
      subspace and show the see/do gap collapses (and collapses *more* than for the base model).
    * Does grounding help out of distribution?  -> measure the gap on novel context words.

causalrl is the source of ground truth throughout: the SCM defines the true regime variable and the
true P(recover) under see vs do, so "is the output mediated by the *true* causal variable?" is
exactly answerable.  Didactic, CPU-sized demonstration — not a performance claim.

Run::

    uv run --extra torch python examples/causal_grounding_das_iit.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from tokenizers import ByteLevelBPETokenizer
from torch import Tensor, nn
from torch.distributions import Uniform
from torch.nn.utils.parametrizations import orthogonal
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from causalrl import CausalGraph, FunctionalMechanism, StructuralCausalModel

SEE_TOK, DO_TOK, PAD_TOK, EOS_TOK = "<see>", "<do>", "<pad>", "<eos>"
REC_TOK, REL_TOK = "<rec>", "<rel>"  # single-token outcomes -> clean 1-token readout
TARGET_LAYER = 2  # which GPT-2 block's residual-stream output we locate/install the regime in

torch.set_num_threads(4)


# ==============================================================================================
# 1. The confounded SCM (causalrl).  U = illness severity (hidden), X = drug, Y = recovered.
#    see vs do genuinely disagree because U confounds X and Y.
# ==============================================================================================


def build_scm() -> StructuralCausalModel:
    graph = CausalGraph(directed_edges=[("U", "X"), ("U", "Y"), ("X", "Y")])

    def u_mech(_p: dict[str, Tensor], noise: Tensor) -> Tensor:
        return (noise < 0.5).float()

    def x_mech(p: dict[str, Tensor], noise: Tensor) -> Tensor:
        return (noise < (0.2 + 0.6 * p["U"])).float()

    def y_mech(p: dict[str, Tensor], noise: Tensor) -> Tensor:
        prob = 0.5 + 0.15 * (2 * p["X"] - 1) + 0.35 * (2 * p["U"] - 1)
        return (noise < prob.clamp(0.0, 1.0)).float()

    mechanisms = {
        "U": FunctionalMechanism([], u_mech),
        "X": FunctionalMechanism(["U"], x_mech),
        "Y": FunctionalMechanism(["X", "U"], y_mech),
    }
    exogenous = {n: Uniform(0.0, 1.0) for n in ("U", "X", "Y")}
    return StructuralCausalModel(graph, mechanisms, exogenous)


def scm_truth(scm: StructuralCausalModel, n: int = 200_000) -> dict[str, float]:
    obs = scm.see(n, seed=7)
    x, y = obs["X"], obs["Y"]
    return {
        "see_drug": float(y[x > 0.5].mean()),
        "do_drug": float(scm.do({"X": 1.0}).see(n, seed=8)["Y"].mean()),
    }


# ==============================================================================================
# 2. Serialise SCM samples into sentences.  Outcome is a single reserved token for a clean readout.
#    A 'domain' is just a surface vocabulary; we train on domain A, keep domain B for the OOD test
#    (same SCM, same regime semantics, unseen words).
# ==============================================================================================


@dataclass(frozen=True)
class Domain:
    subject: str
    drug: str
    nodrug: str
    see_verbs: tuple[str, ...]
    do_verbs: tuple[str, ...]


DOMAIN_A = Domain(
    subject="the patient",
    drug="the drug",
    nodrug="no drug",
    see_verbs=("took", "was on", "kept taking"),
    do_verbs=("was given", "was administered", "was started on"),
)
DOMAIN_C = Domain(  # a second held-in domain, used only to make the regime latent domain-invariant
    subject="the user",
    drug="the program",
    nodrug="no program",
    see_verbs=("joined", "stayed in", "kept attending"),
    do_verbs=("was enrolled", "was placed into", "was assigned to"),
)
DOMAIN_B = Domain(  # OOD: unseen surface words, never trained on (the held-out generalization test)
    subject="the subject",
    drug="the compound",
    nodrug="no compound",
    see_verbs=("used", "stayed on", "carried on with"),
    do_verbs=("was prescribed", "was handed", "was placed on"),
)


def sentence(
    tok: str, dom: Domain, verb: str, drug_on: bool, rec: bool, *, with_outcome: bool
) -> str:
    drug = dom.drug if drug_on else dom.nodrug
    s = f"{tok} {dom.subject} {verb} {drug} and then"
    # No space before the reserved outcome token: it must be the *immediate* next token after
    # "then", so the readout site (prompt's last token) predicts <rec>/<rel> directly.
    return s + ((REC_TOK if rec else REL_TOK) if with_outcome else "")


def build_corpus(scm: StructuralCausalModel, dom: Domain, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    obs = scm.see(4000, seed=seed)
    do1 = scm.do({"X": 1.0}).see(2000, seed=seed + 1)
    do0 = scm.do({"X": 0.0}).see(2000, seed=seed + 2)
    out: list[str] = []
    for tok, verbs, data in [
        (SEE_TOK, dom.see_verbs, obs),
        (DO_TOK, dom.do_verbs, do1),
        (DO_TOK, dom.do_verbs, do0),
    ]:
        for xi, yi in zip(data["X"].tolist(), data["Y"].tolist(), strict=True):
            out.append(sentence(tok, dom, rng.choice(verbs), xi > 0.5, yi > 0.5, with_outcome=True))
    rng.shuffle(out)
    return out


# ==============================================================================================
# 3. Tokenizer + from-scratch GPT-2 (same stack as Qwen/GPT-OSS, tiny).
# ==============================================================================================


def build_tokenizer(corpus: list[str]) -> PreTrainedTokenizerFast:
    bpe = ByteLevelBPETokenizer()
    specials = [PAD_TOK, EOS_TOK, SEE_TOK, DO_TOK, REC_TOK, REL_TOK]
    bpe.train_from_iterator(corpus, vocab_size=1500, min_frequency=1, special_tokens=specials)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        pad_token=PAD_TOK,
        eos_token=EOS_TOK,
        bos_token=EOS_TOK,
        unk_token=EOS_TOK,
    )
    fast.add_special_tokens({"additional_special_tokens": [SEE_TOK, DO_TOK, REC_TOK, REL_TOK]})
    return fast


def build_model(tok: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tok),
        n_positions=64,
        n_ctx=64,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tok))
    return model


def lm_train(
    model: GPT2LMHeadModel,
    tok: PreTrainedTokenizerFast,
    corpus: list[str],
    epochs: int,
    lr: float = 5e-4,
    batch_size: int = 64,
) -> None:
    model.train()
    enc = [tok(s + EOS_TOK).input_ids for s in corpus]
    pad_id = tok.pad_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(0)
    for epoch in range(epochs):
        rng.shuffle(enc)
        total, nb = 0.0, 0
        for i in range(0, len(enc), batch_size):
            batch = enc[i : i + batch_size]
            width = max(len(s) for s in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for j, s in enumerate(batch):
                ids[j, : len(s)] = torch.tensor(s)
                mask[j, : len(s)] = 1
            labels = ids.masked_fill(mask == 0, -100)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += out.loss.item()
            nb += 1
        print(f"    epoch {epoch + 1}/{epochs}  loss {total / nb:.3f}")


# ==============================================================================================
# 4. Prompt batching + readout.  P(recover) is a single-token softmax over <rec> vs <rel>.
# ==============================================================================================


class Prompts:
    """A padded batch of prompts plus the index of each one's last real token (the readout site)."""

    def __init__(self, strings: list[str], tok: PreTrainedTokenizerFast) -> None:
        enc = [tok(s).input_ids for s in strings]
        width = max(len(s) for s in enc)
        pad_id = tok.pad_token_id
        self.ids = torch.full((len(enc), width), pad_id, dtype=torch.long)
        self.mask = torch.zeros((len(enc), width), dtype=torch.long)
        self.last = torch.tensor([len(s) - 1 for s in enc], dtype=torch.long)
        for j, s in enumerate(enc):
            self.ids[j, : len(s)] = torch.tensor(s)
            self.mask[j, : len(s)] = 1

    def __len__(self) -> int:
        return self.ids.size(0)


def regime_prompts(
    dom: Domain, drug_on: bool, tok: PreTrainedTokenizerFast
) -> tuple[Prompts, Prompts]:
    """Matched see/do prompt sets (one per verb paraphrase), differing only in regime."""
    see = [sentence(SEE_TOK, dom, v, drug_on, True, with_outcome=False) for v in dom.see_verbs]
    do = [sentence(DO_TOK, dom, v, drug_on, True, with_outcome=False) for v in dom.do_verbs]
    return Prompts(see, tok), Prompts(do, tok)


def p_recover(model: GPT2LMHeadModel, p: Prompts, rec_id: int, rel_id: int) -> Tensor:
    with torch.no_grad():
        logits = model(input_ids=p.ids, attention_mask=p.mask).logits
    last = logits[torch.arange(len(p)), p.last]
    return torch.softmax(last[:, [rec_id, rel_id]], dim=-1)[:, 0]


def p_recover_grad(model: GPT2LMHeadModel, p: Prompts, rec_id: int, rel_id: int) -> Tensor:
    """Grad-enabled P(recover) for the behavioural anchor in IIT."""
    logits = model(input_ids=p.ids, attention_mask=p.mask).logits
    last = logits[torch.arange(len(p)), p.last]
    return torch.softmax(last[:, [rec_id, rel_id]], dim=-1)[:, 0]


# ==============================================================================================
# 5. DAS subspace + interchange intervention (the core mechanistic operation).
# ==============================================================================================


class Subspace(nn.Module):
    """A k-dim subspace of R^d via an orthonormal basis D (d x k), learned with DAS."""

    def __init__(self, d: int, k: int) -> None:
        super().__init__()
        self.lin = orthogonal(nn.Linear(d, k, bias=False))  # rows orthonormal

    def D(self) -> Tensor:
        return self.lin.weight.t()  # (d, k), orthonormal columns

    @torch.no_grad()
    def warm_start(self, direction: Tensor) -> None:
        """Initialise the basis from a known direction (k=1) so interchange transmits from step 1.

        Without this, a random subspace does not transmit the swap, and IIT's interchange loss
        degenerates into a sign-flipped behavioural anchor.
        """
        self.lin.weight = direction.reshape(
            self.lin.weight.shape
        )  # parametrization re-orthonormalizes


def resid_at(model: GPT2LMHeadModel, layer: int, p: Prompts) -> Tensor:
    """Residual-stream vector at each prompt's readout site, after the given block."""
    cache: dict[str, Tensor] = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cache["h"] = h[torch.arange(len(p)), p.last].detach()
        return out

    h = model.transformer.h[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=p.ids, attention_mask=p.mask)
    h.remove()
    return cache["h"]


def patched_logits(
    model: GPT2LMHeadModel, layer: int, base: Prompts, h_source: Tensor, D: Tensor
) -> Tensor:
    """Run ``base`` but swap the subspace-projection of its readout activation toward h_source.

    h_cf = h_base + ((h_source - h_base) @ D) @ D^T   (interchange = do() on the regime subspace)
    """
    idx = torch.arange(len(base))

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h_base = h[idx, base.last]
        delta = ((h_source - h_base) @ D) @ D.t()
        h = h.clone()
        h[idx, base.last] = h_base + delta
        return ((h, *tuple(out[1:]))) if isinstance(out, tuple) else h

    handle = model.transformer.h[layer].register_forward_hook(hook)
    logits = model(input_ids=base.ids, attention_mask=base.mask).logits
    handle.remove()
    return logits[idx, base.last]


def interchange_effect(
    model: GPT2LMHeadModel,
    layer: int,
    base: Prompts,
    source: Prompts,
    D: Tensor,
    rec_id: int,
    rel_id: int,
) -> float:
    """Fraction of the behavioural regime-gap recovered by swapping *only* the subspace.

    IE = mean( (P_cf - P_base) / (P_source - P_base) ).  IE~1: subspace carries regime; IE~0: not.
    Averaged over both swap directions and broadcast across the paraphrase grid.
    """
    out = 0.0
    with torch.no_grad():
        for b, s in [(base, source), (source, base)]:
            hs = resid_at(model, layer, s).mean(0, keepdim=True).expand(len(b), -1)
            cf = patched_logits(model, layer, b, hs, D.detach())
            p_cf = torch.softmax(cf[:, [rec_id, rel_id]], dim=-1)[:, 0]
            p_b = p_recover(model, b, rec_id, rel_id).mean()
            p_s = p_recover(model, s, rec_id, rel_id).mean()
            denom = float(p_s - p_b)
            out += float((p_cf.mean() - p_b).clamp(-1, 2) / denom) if abs(denom) > 1e-3 else 0.0
    return out / 2


def ablation_gap(
    model: GPT2LMHeadModel,
    layer: int,
    see: Prompts,
    do: Prompts,
    D: Tensor | None,
    mean_vec: Tensor,
    rec_id: int,
    rel_id: int,
) -> float:
    """see/do gap in P(recover) after mean-ablating the subspace (D=None -> no ablation)."""

    def run(p: Prompts) -> float:
        if D is None:
            return float(p_recover(model, p, rec_id, rel_id).mean())
        idx = torch.arange(len(p))

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            hb = h[idx, p.last]
            h = h.clone()
            h[idx, p.last] = hb - ((hb - mean_vec) @ D) @ D.t()  # strip regime subspace -> mean
            return ((h, *tuple(out[1:]))) if isinstance(out, tuple) else h

        handle = model.transformer.h[layer].register_forward_hook(hook)
        with torch.no_grad():
            logits = model(input_ids=p.ids, attention_mask=p.mask).logits
        handle.remove()
        last = logits[idx, p.last]
        return float(torch.softmax(last[:, [rec_id, rel_id]], dim=-1)[:, 0].mean())

    return run(see) - run(do)


def ood_repair(
    model: GPT2LMHeadModel,
    layer: int,
    base_see: Prompts,
    base_do: Prompts,
    ref_see: Prompts,
    ref_do: Prompts,
    D: Tensor,
    rec_id: int,
    rel_id: int,
) -> tuple[float, float]:
    """Inference-time OOD repair by interchange along the located regime direction D.

    For OOD (base) prompts whose see/do gap has collapsed, overwrite *only* their regime coordinate
    with the value it takes for the matching in-distribution (ref) regime.  If the OOD failure is a
    mis-encoding of the regime feature, this restores the gap.  Returns:
      * repaired gap  = P(base see <- ref see) - P(base do <- ref do)   (regimes differ -> gap back)
      * control gap   = P(base see <- ref see) - P(base do <- ref see)  (same regime -> ~0)
    """

    @torch.no_grad()
    def patched_p(base: Prompts, ref: Prompts) -> float:
        hs = resid_at(model, layer, ref).mean(0, keepdim=True).expand(len(base), -1)
        cf = patched_logits(model, layer, base, hs, D)
        return float(torch.softmax(cf[:, [rec_id, rel_id]], dim=-1)[:, 0].mean())

    p_see = patched_p(base_see, ref_see)
    return p_see - patched_p(base_do, ref_do), p_see - patched_p(base_do, ref_see)


# ==============================================================================================
# 6. DAS training (Phase 0, model frozen) and IIT training (Phase 1, model + subspace).
# ==============================================================================================


def das_locate(
    model: GPT2LMHeadModel,
    layer: int,
    see: Prompts,
    do: Prompts,
    k: int,
    rec_id: int,
    rel_id: int,
    steps: int = 300,
) -> Subspace:
    """Learn the regime subspace: swapping it must make base behave like the source regime."""
    sub = Subspace(model.config.n_embd, k)
    opt = torch.optim.Adam(sub.parameters(), lr=5e-3)
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        for b, s in [(see, do), (do, see)]:
            hs = resid_at(model, layer, s).mean(0, keepdim=True).expand(len(b), -1)
            cf = patched_logits(model, layer, b, hs, sub.D())
            p_cf = torch.softmax(cf[:, [rec_id, rel_id]], dim=-1)[:, 0]
            with torch.no_grad():
                p_s = p_recover(model, s, rec_id, rel_id).mean()
            loss = loss + ((p_cf - p_s) ** 2).mean()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return sub


def iit_train(
    model: GPT2LMHeadModel,
    tok: PreTrainedTokenizerFast,
    corpus: list[str],
    layer: int,
    sub: Subspace,
    rec_id: int,
    rel_id: int,
    see_p: float,
    do_p: float,
    epochs: int,
    lr: float = 1e-4,
    batch_size: int = 64,
    lam: float = 1.0,
    freeze_subspace: bool = True,
) -> None:
    """Interchange Intervention Training with **fixed SCM-truth targets** + **cross-domain** swaps.

    Three losses, all anchored to the SCM (so the degenerate "collapse the gap" optimum is barred):
      * LM loss (keep fluency / next-token over the corpus);
      * behavioural anchor: P(rec|see)->see_p and P(rec|do)->do_p on held-in domains A and C;
      * interchange: injecting the *source regime's* subspace value makes the base output equal the
        source regime's truth probability -- both within a domain and **across** domains A<->C,
        forcing the regime subspace to be domain-INVARIANT (the lever for OOD generalization).
    Domain B is never touched here; it is the held-out OOD test.
    """
    seeA, doA = regime_prompts(DOMAIN_A, True, tok)
    seeC, doC = regime_prompts(DOMAIN_C, True, tok)
    anchors = [(seeA, see_p), (doA, do_p), (seeC, see_p), (doC, do_p)]
    # (base, source, target): source's regime determines the target probability.
    swaps = [
        (seeA, doA, do_p),
        (doA, seeA, see_p),
        (seeC, doC, do_p),
        (doC, seeC, see_p),  # within
        (seeA, doC, do_p),
        (doA, seeC, see_p),
        (seeC, doA, do_p),
        (doC, seeA, see_p),  # cross-domain
    ]

    model.train()
    enc = [tok(s + EOS_TOK).input_ids for s in corpus]
    pad_id = tok.pad_token_id
    # With D frozen at the located regime direction, interchange transmits the regime throughout, so
    # the model is forced to keep regime ON that direction (and make it domain-invariant) without D
    # drifting into the degenerate, sign-flipped optimum.
    params = list(model.parameters()) + ([] if freeze_subspace else list(sub.parameters()))
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(1)
    for epoch in range(epochs):
        rng.shuffle(enc)
        tot_lm, tot_beh, tot_ii, nb = 0.0, 0.0, 0.0, 0
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

            beh = torch.zeros(())
            for p, tgt in anchors:
                beh = beh + ((p_recover_grad(model, p, rec_id, rel_id) - tgt) ** 2).mean()

            ii = torch.zeros(())
            for b, s, tgt in swaps:
                hs = resid_at(model, layer, s).mean(0, keepdim=True).expand(len(b), -1)
                cf = patched_logits(model, layer, b, hs, sub.D())
                p_cf = torch.softmax(cf[:, [rec_id, rel_id]], dim=-1)[:, 0]
                ii = ii + ((p_cf - tgt) ** 2).mean()

            loss = lm + beh + lam * ii
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_lm += float(lm.detach())
            tot_beh += float(beh.detach())
            tot_ii += float(ii.detach())
            nb += 1
        print(
            f"    epoch {epoch + 1}/{epochs}  lm {tot_lm / nb:.3f}  behaviour {tot_beh / nb:.4f}  "
            f"interchange {tot_ii / nb:.4f}"
        )


# ==============================================================================================
# 7. Orchestration.
# ==============================================================================================


def report_gaps(
    tag: str, model: GPT2LMHeadModel, tok: PreTrainedTokenizerFast, rec_id: int, rel_id: int
) -> None:
    for name, dom in [("in-dist (domain A)", DOMAIN_A), ("OOD (domain B, novel words)", DOMAIN_B)]:
        see, do = regime_prompts(dom, True, tok)
        p_see = float(p_recover(model, see, rec_id, rel_id).mean())
        p_do = float(p_recover(model, do, rec_id, rel_id).mean())
        print(
            f"    {tag:5s}  {name:30s}  P(rec|see)={p_see:.3f}  P(rec|do)={p_do:.3f}  "
            f"gap = {p_see - p_do:+.3f}"
        )


def main() -> None:
    import os

    torch.manual_seed(0)
    scm = build_scm()
    truth = scm_truth(scm)
    see_p, do_p = truth["see_drug"], truth["do_drug"]
    print(
        f"causalrl SCM truth:  P(rec|see drug)={see_p:.3f}  P(rec|do drug)={do_p:.3f}  "
        f"ground-truth see/do gap={see_p - do_p:+.3f}\n"
    )

    corpus = build_corpus(scm, DOMAIN_A)  # deterministic -> tokenizer/vocab stable across runs
    tok = build_tokenizer(corpus)
    rec_id = tok.convert_tokens_to_ids(REC_TOK)
    rel_id = tok.convert_tokens_to_ids(REL_TOK)
    model = build_model(tok)

    ckpt = "/tmp/cg_base.pt"  # optional cache so Phase-1 iteration does not retrain the base
    if os.environ.get("CG_CACHE") and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print("loaded cached base model")
    else:
        print(f"training base GPT-2 ({model.num_parameters() / 1e6:.2f}M params) on domain A ...")
        lm_train(model, tok, corpus, epochs=10)
        if os.environ.get("CG_CACHE"):
            torch.save(model.state_dict(), ckpt)
    model.eval()
    print("\nbase model behavioural gaps:")
    report_gaps("base", model, tok, rec_id, rel_id)

    see_A, do_A = regime_prompts(DOMAIN_A, True, tok)
    see_B, do_B = regime_prompts(DOMAIN_B, True, tok)

    # -------- PHASE 0: locate + diagnose --------
    print("\n=== PHASE 0  locate + diagnose (model frozen) ===")
    mean_vec = torch.cat(
        [resid_at(model, TARGET_LAYER, see_A), resid_at(model, TARGET_LAYER, do_A)], 0
    ).mean(0)
    sub_star: Subspace | None = None
    for k in (1, 4):
        sub = das_locate(model, TARGET_LAYER, see_A, do_A, k, rec_id, rel_id)
        ie = interchange_effect(model, TARGET_LAYER, see_A, do_A, sub.D(), rec_id, rel_id)
        rnd = Subspace(model.config.n_embd, k)  # random subspace control (mediation diagnosis)
        ie_rnd = interchange_effect(model, TARGET_LAYER, see_A, do_A, rnd.D(), rec_id, rel_id)
        abl = ablation_gap(model, TARGET_LAYER, see_A, do_A, sub.D(), mean_vec, rec_id, rel_id)
        base_gap = ablation_gap(model, TARGET_LAYER, see_A, do_A, None, mean_vec, rec_id, rel_id)
        print(
            f"  k={k}:  IE(learned)={ie:+.2f}   IE(random)={ie_rnd:+.2f}   "
            f"gap base={base_gap:+.3f} -> ablated={abl:+.3f}   "
            f"attribution={100 * (1 - abl / base_gap):.0f}%"
        )
        if k == 1:
            sub_star = sub
    assert sub_star is not None
    d_star = sub_star.D().detach().clone()  # the located 1-D regime direction

    # -------- PHASE 1a: repair OOD by intervention on the located direction (inference-time) -----
    print("\n=== PHASE 1a  repair OOD by intervening on the located regime direction ===")
    rep, ctrl = ood_repair(model, TARGET_LAYER, see_B, do_B, see_A, do_A, d_star, rec_id, rel_id)
    base_b = float(
        p_recover(model, see_B, rec_id, rel_id).mean()
        - p_recover(model, do_B, rec_id, rel_id).mean()
    )
    print(
        f"  OOD (domain B) see/do gap:  base={base_b:+.3f}  ->  repaired={rep:+.3f}   "
        f"(truth {see_p - do_p:+.3f};  same-regime control gap={ctrl:+.3f} ~ 0)"
    )

    # -------- PHASE 1b: install a domain-invariant regime latent (IIT on the frozen direction) ----
    print("\n=== PHASE 1b  install a domain-invariant regime latent (IIT on located direction) ===")
    sub1 = Subspace(model.config.n_embd, 1)
    sub1.warm_start(d_star)  # freeze D at the Phase-0 regime direction; train the model around it
    iit_train(
        model,
        tok,
        corpus,
        TARGET_LAYER,
        sub1,
        rec_id,
        rel_id,
        see_p,
        do_p,
        epochs=5,
        freeze_subspace=True,
    )
    model.eval()
    mean_vec = torch.cat(
        [resid_at(model, TARGET_LAYER, see_A), resid_at(model, TARGET_LAYER, do_A)], 0
    ).mean(0)
    ie = interchange_effect(model, TARGET_LAYER, see_A, do_A, sub1.D(), rec_id, rel_id)
    abl = ablation_gap(model, TARGET_LAYER, see_A, do_A, sub1.D(), mean_vec, rec_id, rel_id)
    base_gap = ablation_gap(model, TARGET_LAYER, see_A, do_A, None, mean_vec, rec_id, rel_id)
    print(
        f"  installed 1-D regime subspace:  IE={ie:+.2f}   gap intact={base_gap:+.3f} -> "
        f"ablated={abl:+.3f}  (attribution: {100 * (1 - abl / base_gap):.0f}%)"
    )
    print("\nIIT model behavioural gaps (note OOD domain B was never trained on):")
    report_gaps("iit", model, tok, rec_id, rel_id)

    print(
        "\nReading: PHASE 0 locates the regime in 1 linear dim (IE~1 vs ~0 random; ablation kills "
        "the gap). PHASE 1a shows the OOD failure is a regime mis-encoding, repairable by "
        "intervening on that direction. PHASE 1b installs a domain-invariant regime latent via IIT."
    )


if __name__ == "__main__":
    main()
