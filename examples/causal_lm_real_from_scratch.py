# STATUS: canonical (from-scratch sub-thread) · Act 4 Coupling — real GPT-2 trained from scratch (BPE)  ·  map: CAUSAL_LLM.md
"""A *real* small LLM, trained from scratch to be causal — same usage as Qwen / GPT-OSS.

This is the "nivel B" build: unlike the 6-token toy in ``causal_lm_from_scratch.py``, here we
train an honest GPT-style decoder with the standard stack you would use for Qwen or GPT-OSS:

* a real **byte-level BPE tokenizer** trained on the corpus (``tokenizers``),
* a randomly-initialised **GPT-2 decoder** (``transformers.GPT2LMHeadModel``) — trained from
  scratch, not fine-tuned from anything,
* generation via ``model.generate(...)`` — you prompt it and it continues, like any LLM.

What makes it *causal from the foundation* is the **corpus**: every sentence is a natural-language
rendering of a sample from a causalrl :class:`~causalrl.StructuralCausalModel` with a hidden
confounder, tagged with a control token:

* ``<see>`` — observational: "the patient **took** the drug and then recovered"
* ``<do>``  — interventional: "the patient **was given** the drug and then recovered"

The confounder (illness severity) is never written down, so observing and intervening genuinely
disagree. After training, we prompt the model both ways and check it reproduces the SCM's truth:

    truth:  P(recover | do drug)  = 0.65     P(recover | see drug)  = 0.86

A model that merely memorised text would collapse these. One that learned the causal structure
keeps them apart — the difference between *giving* the drug and *noticing who took* it.

Run (a few minutes on CPU, faster on GPU)::

    uv run --extra torch python examples/causal_lm_real_from_scratch.py

causalrl is the source of ground truth throughout. Didactic demonstration, not a perf claim.

Scaling note: this is exactly the shape of a Qwen/GPT-OSS pretraining loop, just tiny. To go
bigger you would (a) grow the GPT-2 config / swap in a Llama config, (b) feed a much larger and
more linguistically varied corpus, and (c) keep causalrl as the generator of interventional /
counterfactual episodes and as the verifier. A genuinely Qwen-scale *from-scratch* run, however,
is a thousands-of-GPU-hour pretraining project, not a single script.
"""

from __future__ import annotations

import random

import torch
from tokenizers import ByteLevelBPETokenizer
from torch import Tensor
from torch.distributions import Uniform
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from causalrl import CausalGraph, FunctionalMechanism, StructuralCausalModel

SEE_TOK, DO_TOK, PAD_TOK, EOS_TOK = "<see>", "<do>", "<pad>", "<eos>"
RECOVER, RELAPSE = " recovered", " relapsed"

# --------------------------------------------------------------------------------------------
# 1. The confounded SCM (causalrl). U = illness severity, hidden from the model.
#    X = received the drug, Y = recovered. Same numbers as the toy example.
# --------------------------------------------------------------------------------------------


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
        "see_nodrug": float(y[x < 0.5].mean()),
        "do_drug": float(scm.do({"X": 1.0}).see(n, seed=8)["Y"].mean()),
        "do_nodrug": float(scm.do({"X": 0.0}).see(n, seed=9)["Y"].mean()),
    }


# --------------------------------------------------------------------------------------------
# 2. Serialise SCM samples into natural-language sentences with <see>/<do> control tokens.
#    A few phrasings give the tokenizer/LM real language to model (not a 6-symbol alphabet).
# --------------------------------------------------------------------------------------------

SEE_TEMPLATES = [
    "{tok} the patient took {drug} and then{outcome} .",
    "{tok} a patient who took {drug} and then{outcome} .",
    "{tok} we observed a patient on {drug} ; they{outcome} .",
]
DO_TEMPLATES = [
    "{tok} the patient was given {drug} and then{outcome} .",
    "{tok} we administered {drug} to a patient and then they{outcome} .",
    "{tok} we randomly assigned {drug} ; the patient{outcome} .",
]


def render(tok: str, templates: list[str], x: Tensor, y: Tensor, rng: random.Random) -> list[str]:
    out: list[str] = []
    for xi, yi in zip(x.tolist(), y.tolist(), strict=True):
        drug = "the drug" if xi > 0.5 else "no drug"
        outcome = RECOVER if yi > 0.5 else RELAPSE
        out.append(rng.choice(templates).format(tok=tok, drug=drug, outcome=outcome))
    return out


def build_corpus(scm: StructuralCausalModel, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    obs = scm.see(4000, seed=seed)
    do1 = scm.do({"X": 1.0}).see(2000, seed=seed + 1)
    do0 = scm.do({"X": 0.0}).see(2000, seed=seed + 2)

    sentences = (
        render(SEE_TOK, SEE_TEMPLATES, obs["X"], obs["Y"], rng)
        + render(DO_TOK, DO_TEMPLATES, do1["X"], do1["Y"], rng)
        + render(DO_TOK, DO_TEMPLATES, do0["X"], do0["Y"], rng)
    )
    rng.shuffle(sentences)
    return sentences


# --------------------------------------------------------------------------------------------
# 3. Train a real byte-level BPE tokenizer, then wrap it for transformers.
# --------------------------------------------------------------------------------------------


def build_tokenizer(corpus: list[str]) -> PreTrainedTokenizerFast:
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        corpus, vocab_size=2000, min_frequency=1,
        special_tokens=[PAD_TOK, EOS_TOK, SEE_TOK, DO_TOK],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,  # the underlying tokenizers object
        pad_token=PAD_TOK, eos_token=EOS_TOK, bos_token=EOS_TOK, unk_token=EOS_TOK,
    )
    fast.add_special_tokens({"additional_special_tokens": [SEE_TOK, DO_TOK]})
    return fast


# --------------------------------------------------------------------------------------------
# 4. A randomly-initialised GPT-2 decoder, trained from scratch with next-token loss.
# --------------------------------------------------------------------------------------------


def build_model(tok: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tok), n_positions=64, n_ctx=64,
        n_embd=128, n_layer=4, n_head=4,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(config)  # random init -> from scratch
    model.resize_token_embeddings(len(tok))
    return model


def train(
    model: GPT2LMHeadModel, tok: PreTrainedTokenizerFast, corpus: list[str],
    epochs: int = 12, batch_size: int = 64, lr: float = 5e-4, device: str = "cpu",
) -> None:
    model.to(device).train()
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
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                        labels=labels.to(device))
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += out.loss.item()
            nb += 1
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / nb:.3f}")


# --------------------------------------------------------------------------------------------
# 5. Read P(recover | prompt) off the trained LM by scoring the two outcome continuations.
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def cont_logprob(model: GPT2LMHeadModel, tok: PreTrainedTokenizerFast, prompt: str, cont: str
                 ) -> float:
    full = tok(prompt + cont, return_tensors="pt").input_ids
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    logits = model(full).logits[0, :-1]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = full[0, 1:]
    token_lp = logp[torch.arange(tgt.shape[0]), tgt]
    return float(token_lp[n_prompt - 1 :].sum())  # only the continuation's tokens


def p_recover(model: GPT2LMHeadModel, tok: PreTrainedTokenizerFast, prompt: str) -> float:
    lp_rec = cont_logprob(model, tok, prompt, RECOVER)
    lp_rel = cont_logprob(model, tok, prompt, RELAPSE)
    return float(torch.softmax(torch.tensor([lp_rec, lp_rel]), dim=0)[0])


def main() -> None:
    torch.manual_seed(0)
    scm = build_scm()
    truth = scm_truth(scm)

    print("building corpus from the causalrl SCM ...")
    corpus = build_corpus(scm)
    print(f"  {len(corpus)} sentences, e.g.: {corpus[0]!r}")

    tok = build_tokenizer(corpus)
    model = build_model(tok)
    print(f"training a {model.num_parameters() / 1e6:.2f}M-param GPT-2 from scratch ...")
    train(model, tok, corpus)
    model.eval()

    prompts = {
        "see_drug":   f"{SEE_TOK} the patient took the drug and then",
        "see_nodrug": f"{SEE_TOK} the patient took no drug and then",
        "do_drug":    f"{DO_TOK} the patient was given the drug and then",
        "do_nodrug":  f"{DO_TOK} the patient was given no drug and then",
    }

    print("\n                              P(recover)")
    print("                          SCM truth   tiny-LLM")
    for key, label in [
        ("see_drug",   "see  drug   (confounded)"),
        ("see_nodrug", "see  no-drug(confounded)"),
        ("do_drug",    "do   drug   (causal)    "),
        ("do_nodrug",  "do   no-drug(causal)    "),
    ]:
        print(f"  {label}  {truth[key]:.3f}      {p_recover(model, tok, prompts[key]):.3f}")

    gap_truth = truth["do_drug"] - truth["do_nodrug"]
    gap_model = (
        p_recover(model, tok, prompts["do_drug"]) - p_recover(model, tok, prompts["do_nodrug"])
    )
    print(f"\n  causal effect of the drug  P(recover|do drug)-P(recover|do no-drug):"
          f"  truth {gap_truth:+.3f}  model {gap_model:+.3f}")

    # Same usage as any LLM: prompt and let it generate.
    gen = model.generate(
        **tok(prompts["do_drug"], return_tensors="pt"),
        max_new_tokens=4, do_sample=True, top_k=10,
        pad_token_id=tok.pad_token_id,
    )
    print(f"\n  sample generation: {tok.decode(gen[0], skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
