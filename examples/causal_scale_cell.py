# STATUS: research · ladder §6 — the SCALE CELL: R4's STRUCTONLY at 0.5B (LoRA) · design: docs/causal_llm/LADDER.md · map: CAUSAL_LLM.md
"""The scale cell — does scale x schedule move the in-weights verdict?

Every in-weights result on this branch is 0.2–3.6M params. This script re-runs the ladder's R4
STRUCTONLY cell (query + true graph as text -> one-shot answer, answer-token supervision only) on a
PRETRAINED ~0.5B LM (Qwen2.5-0.5B) with hand-rolled LoRA — no peft/accelerate (pip installs into
this uv-managed venv get wiped by any sync; 40 lines of LoRA are sturdier). Examples are the SAME
`causal_hybrid_lm` substrate, rendered to text (`" ".join(WORDS[i])`) and tokenized by the model's
own BPE.

References to beat (3 seeds, in-house 809K GPT-2): cause s3 0.731±0.094 / s4 0.581±0.084,
conf s3 0.190±0.150 / s4 0.160±0.133. The five-levers verdict says the confounded trap is
architectural for in-weights computation at small scale — THIS is the cell that could overturn it.

GPU-burst pattern (this box's driver wedges on sustained runs): STEPS per invocation, atomic
checkpoint/resume (LoRA params + opt + step counter), logs to /mnt/c, MODE=eval for a
checkpoint-only read. A zero-shot row is printed before any training.

Run::

    FAST=1 SEEDS=0 STEPS=30 uv run ... python examples/causal_scale_cell.py     # smoke
    SEEDS=0 STEPS=2000 ...                                                      # one burst
    MODE=eval SEEDS=0 ...                                                       # read ckpt

Knobs: SEEDS (one seed per invocation), STEPS, MICRO, LORA_R, MODEL, MODE, CKPT, FAST.
"""

from __future__ import annotations

import math
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_hybrid_lm as hy
import causal_pure_twostage as pt

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
SLUG = MODEL.split("/")[-1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = int(os.environ.get("SEEDS", "0").split(",")[0])
FAST = os.environ.get("FAST") == "1"
STEPS = int(os.environ.get("STEPS", "2000"))
MICRO = int(os.environ.get("MICRO", "8"))
EFF_BATCH = 16
ACCUM = max(1, EFF_BATCH // MICRO)
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = 2 * LORA_R
LR = float(os.environ.get("LR", "1e-4"))
MODE = os.environ.get("MODE", "train")
CKPT = os.environ.get("CKPT", f"{ROOT}/.scale_{SLUG}_s{SEED}.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 2000 if FAST else 8000
NT_EVAL = 400 if FAST else 1500
TOTAL_STEPS = (N // EFF_BATCH) * (4 if FAST else 12)  # in-house budget: 12 epochs


class LoRA(nn.Module):
    """Wraps a frozen nn.Linear with a rank-r fp32 update: y = Wx + (alpha/r) * B(Ax)."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(r, base.in_features) * (1 / math.sqrt(r)))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        lora = (x.float() @ self.a.T @ self.b.T) * self.scale
        return y + lora.to(y.dtype)


def add_lora(model, r, alpha):
    """Wrap every attention/MLP projection Linear in the decoder layers."""
    n_wrapped = 0
    for layer in model.model.layers:
        for mod in (layer.self_attn, layer.mlp):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(mod, name, LoRA(child, r, alpha))
                    n_wrapped += 1
    return n_wrapped


def lora_params(model):
    return [p for n, p in model.named_parameters() if n.endswith((".a", ".b"))]


def render(ids):
    return " ".join(pt.WORDS[i] for i in ids)


def build_rows(data, tok):
    """(input_ids, answer_position, yes_first) rows for the STRUCTONLY text task."""
    rows = []
    for e in data:
        seq, _sup = pt.seq_structonly(e)
        prompt = render(seq[:-1])  # everything up to the answer token
        ids = tok(prompt, return_tensors=None)["input_ids"]
        rows.append((ids, int(e["label"])))
    return rows


def pack_rows(rows, pad_id):
    width = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attn = torch.zeros(len(rows), width, dtype=torch.long)
    last = torch.tensor([len(r[0]) - 1 for r in rows])
    for j, (r, _y) in enumerate(rows):
        ids[j, : len(r)] = torch.tensor(r)
        attn[j, : len(r)] = 1
    y = torch.tensor([r[1] for r in rows])
    return ids.to(DEVICE), attn.to(DEVICE), last.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def acc(model, rows, yes_id, no_id, pad_id, bs=64):
    ok = n = 0
    model.eval()
    for i in range(0, len(rows), bs):
        ids, attn, last, y = pack_rows(rows[i : i + bs], pad_id)
        logits = model(input_ids=ids, attention_mask=attn).logits
        row = logits[torch.arange(len(ids)), last]
        m = row[:, yes_id] - row[:, no_id]
        ok += int(((m > 0) == (y > 0.5)).sum())
        n += len(ids)
    return ok / n


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    yes_id = tok(" yes")["input_ids"]
    no_id = tok(" no")["input_ids"]
    assert len(yes_id) == 1 and len(no_id) == 1, "answer must be single BPE tokens"
    yes_id, no_id = yes_id[0], no_id[0]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    nw = add_lora(model, LORA_R, LORA_ALPHA)
    model.to(DEVICE)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = lora_params(model)
    npar = sum(p.numel() for p in params)
    print(f"[scale] {MODEL} seed={SEED} lora r={LORA_R} on {nw} linears ({npar:,} trainable) "
          f"device={DEVICE} steps/burst={STEPS} total_budget={TOTAL_STEPS}", flush=True)

    data = hy.build(N, sizes=[2, 3], seed=SEED)
    t3, t4 = hy.build(NT_EVAL, [3], SEED + 50), hy.build(NT_EVAL, [4], SEED + 60)
    evals = {
        "conf_s3": build_rows(pt.confounded(t3), tok),
        "conf_s4": build_rows(pt.confounded(t4), tok),
        "cause_s3": build_rows([e for e in t3 if e["is_causal"]], tok),
        "cause_s4": build_rows([e for e in t4 if e["is_causal"]], tok),
    }
    train_rows = build_rows(data, tok)

    opt = torch.optim.AdamW(params, lr=LR)
    step = 0
    if os.path.exists(CKPT):
        st = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        for p, saved in zip(params, st["lora"], strict=True):
            p.data.copy_(saved)
        opt.load_state_dict(st["opt"])
        step = int(st["step"])
        print(f"[scale] resumed {CKPT} at step {step}/{TOTAL_STEPS}", flush=True)
    else:
        print("[scale] fresh LoRA; ZERO-SHOT row first:", flush=True)
        for name, rows in evals.items():
            print(f"    zero-shot {name:<10} {acc(model, rows, yes_id, no_id, pad_id):.3f}",
                  flush=True)

    def save():
        tmp = CKPT + ".tmp"
        torch.save({"lora": [p.data for p in params], "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, CKPT)

    if MODE == "train" and step < TOTAL_STEPS:
        import random

        rng = random.Random(step)  # order differs per resume; every row seen over the run
        order = list(range(len(train_rows)))
        model.train()
        done = 0
        while done < STEPS and step < TOTAL_STEPS:
            rng.shuffle(order)
            for i in range(0, len(order) - EFF_BATCH + 1, EFF_BATCH):
                if done >= STEPS or step >= TOTAL_STEPS:
                    break
                opt.zero_grad()
                for m0 in range(0, EFF_BATCH, MICRO):
                    sel = [train_rows[j] for j in order[i + m0 : i + m0 + MICRO]]
                    ids, attn, last, y = pack_rows(sel, pad_id)
                    logits = model(input_ids=ids, attention_mask=attn).logits
                    row = logits[torch.arange(len(ids)), last][:, [no_id, yes_id]]
                    loss = nn.functional.cross_entropy(row.float(), y) / ACCUM
                    loss.backward()
                opt.step()
                step += 1
                done += 1
                if done % 100 == 0:
                    print(f"    step {step}/{TOTAL_STEPS}  loss {float(loss) * ACCUM:.4f}",
                          flush=True)
                if done % 500 == 0:
                    save()
        save()
        print(f"[scale] burst done at step {step}/{TOTAL_STEPS}", flush=True)

    print(f"\n[scale] eval at step {step} ({MODEL}, seed {SEED}):", flush=True)
    for name, rows in evals.items():
        print(f"    {name:<10} {acc(model, rows, yes_id, no_id, pad_id):.3f}", flush=True)
    print("\n  refs (809K in-house, 3 seeds): cause 0.731/0.581, conf 0.190/0.160 — the trap is"
          "\n  the verdict to beat; GNN = 1.000/0.952 cause, 1.000/0.893 conf.", flush=True)


if __name__ == "__main__":
    main()
