"""Robust, scalable trainer for the small causal reasoner — CPU smoke test or full GPU run.

This is the hardened evolution of ``causal_reasoner_prototype.py`` (which stays as the minimal
didactic version). It teaches a from-scratch decoder transformer a structural causal rule from
traces generated and verified by causalrl, and is engineered to be *launched and trusted*:

* **device-agnostic** — auto-detects CUDA / MPS / CPU; AMP (mixed precision) on CUDA.
* **architecture fix for the generalisation wall** — positional scheme is selectable; the default
  is **RoPE** (rotary), which extrapolates to longer sequences far better than the learned
  positional embeddings the prototype used. That directly targets the size-extrapolation gap.
  ``--pos nope`` and ``--pos learned`` are available for ablation.
* **reproducible** — every RNG seeded; optional deterministic kernels.
* **robust training loop** — warmup + cosine LR, gradient clipping, best/last checkpointing with
  resume, periodic eval, JSON metrics, unbuffered logging to stdout and a file.
* **two tasks** — ``dsep`` (d-separation, the learnable bedrock) and ``ident`` (identifiability,
  the harder CHT gate); same pipeline, selectable with ``--task``.

Quick CPU sanity check (seconds)::

    uv run --extra torch python examples/causal_reasoner_train.py --smoke

Full run (scales itself up on GPU; override anything via flags)::

    uv run --extra torch python examples/causal_reasoner_train.py --task dsep --epochs 60 \
        --d-model 256 --layers 6 --heads 8 --n-train 60000 --out runs/dsep

Resume::

    uv run --extra torch python examples/causal_reasoner_train.py --out runs/dsep --resume

causalrl is the trace generator and the ground-truth verifier throughout. Didactic research
scaffold, not a performance guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from causalrl import CausalGraph, is_identifiable
from causalrl.identification._separation import d_separated

# ============================================================================================
# Config
# ============================================================================================


@dataclass
class Config:
    task: str = "dsep"  # "dsep" | "ident"
    # data
    train_sizes: tuple[int, ...] = (3, 4, 5)
    eval_sizes: tuple[int, ...] = (3, 4, 5)  # held-out instances, same sizes -> tests the rule
    extrap_sizes: tuple[int, ...] = (6, 7)  # larger graphs -> tests length generalisation
    n_train: int = 8000
    n_eval: int = 1000
    edge_prob: float = 0.4
    bidir_prob: float = 0.25  # only used by the "ident" task (latent confounders)
    max_cond: int = 2  # max conditioning-set size for "dsep"
    # model
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int | None = None  # defaults to 4 * d_model
    dropout: float = 0.0
    pos: str = "rope"  # "rope" | "nope" | "learned"
    # optimisation
    batch_size: int = 128
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    # runtime
    seed: int = 0
    device: str = "auto"
    amp: bool = True  # enabled only on CUDA
    deterministic: bool = False
    num_workers: int = 0
    eval_every: int = 5  # epochs
    log_every: int = 50  # steps
    out: str = "runs/causal_reasoner"
    resume: bool = False
    max_nodes: int = 12  # vocabulary supports graphs up to this many nodes

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ============================================================================================
# Reproducibility
# ============================================================================================


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


# ============================================================================================
# Vocabulary + serialisation
# ============================================================================================


class Vocab:
    def __init__(self, max_nodes: int) -> None:
        self.nodes = list(string.ascii_uppercase[:max_nodes])
        specials = ["[PAD]", "[G]", "->", "<>", "[Q]", "[C]", "[A]", "yes", "no", "[E]"]
        self.tokens = specials + self.nodes
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.pad = self.stoi["[PAD]"]
        self.yes = self.stoi["yes"]
        self.no = self.stoi["no"]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi[t] for t in toks]


# ============================================================================================
# Trace generation (causalrl as generator + oracle)
# ============================================================================================


def _random_graph(cfg: Config, n: int, rng: random.Random, vocab: Vocab
                  ) -> tuple[CausalGraph, list[str]]:
    nodes = rng.sample(vocab.nodes, n)
    order = nodes[:]
    rng.shuffle(order)
    directed = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < cfg.edge_prob
    ]
    bidirected = (
        [
            (order[i], order[j])
            for i in range(n)
            for j in range(i + 1, n)
            if rng.random() < cfg.bidir_prob
        ]
        if cfg.task == "ident"
        else []
    )
    return CausalGraph(directed_edges=directed, bidirected_edges=bidirected, nodes=nodes), order


def _make_example(cfg: Config, n: int, rng: random.Random, vocab: Vocab) -> tuple[list[str], bool]:
    graph, order = _random_graph(cfg, n, rng, vocab)
    toks = ["[G]"]
    for src, dst in graph.directed_edges:
        toks += [src, "->", dst]
    for a, b in graph.bidirected_edges:
        toks += [a, "<>", b]

    if cfg.task == "dsep":
        x, y = rng.sample(order, 2)
        rest = [v for v in order if v not in (x, y)]
        z = rng.sample(rest, rng.randint(0, min(cfg.max_cond, len(rest))))
        label = d_separated(graph, {x}, {y}, set(z))
        toks += ["[Q]", x, y, "[C]", *z, "[A]", "yes" if label else "no", "[E]"]
        return toks, label

    # ident: needs a directed path X ->...-> Y for a non-trivial effect
    pairs = [(x, y) for x in order for y in graph.descendants(x) if x != y]
    if not pairs:
        return _make_example(cfg, n, rng, vocab)  # resample
    x, y = rng.choice(pairs)
    label = is_identifiable(graph, x, y)
    toks += ["[Q]", x, y, "[A]", "yes" if label else "no", "[E]"]
    return toks, label


def build_split(cfg: Config, sizes: tuple[int, ...], n: int, seed: int, vocab: Vocab
                ) -> list[list[str]]:
    """Class-balanced traces over the given sizes (rejection-sampled)."""
    rng = random.Random(seed)
    pos: list[list[str]] = []
    neg: list[list[str]] = []
    target = n // 2
    guard = 0
    while len(pos) < target or len(neg) < target:
        toks, label = _make_example(cfg, rng.choice(sizes), rng, vocab)
        bucket = pos if label else neg
        if len(bucket) < target:
            bucket.append(toks)
        guard += 1
        if guard > 200 * n:  # pathological class imbalance safety valve
            break
    data = pos + neg
    rng.shuffle(data)
    return data


# ============================================================================================
# Dataset
# ============================================================================================


class TraceDataset(Dataset[list[int]]):
    def __init__(self, traces: list[list[str]], vocab: Vocab) -> None:
        self.rows = [vocab.encode(t) for t in traces]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> list[int]:
        return self.rows[i]


def make_collate(pad: int):
    def collate(batch: list[list[int]]) -> tuple[Tensor, Tensor]:
        width = max(len(r) for r in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        for i, r in enumerate(batch):
            ids[i, : len(r)] = torch.tensor(r)
        return ids, ids == pad  # (tokens, pad_mask)

    return collate


# ============================================================================================
# Model: decoder-only transformer with selectable positional scheme (RoPE default)
# ============================================================================================


def _rope_cache(t: int, dim: int, device: torch.device, base: float = 10000.0
                ) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(t, device=device).float()
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, T, Dh); cos/sin: (T, Dh)
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Llama/Qwen/Gemma standard): no mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward (Llama/PaLM/Mistral standard), bias-free. Hidden ≈ 8/3·d_model."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = ((8 * d_model // 3 + 31) // 32) * 32  # round 8/3·d to a multiple of 32
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.pos = cfg.pos
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=2)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)
        if self.pos == "rope":
            cos, sin = _rope_cache(t, self.dh, x.device)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, self.h * self.dh)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x), attn_mask)
        return x + self.mlp(self.n2(x))


class CausalReasoner(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab_size, cfg.d_model)
        self.learned_pos = (
            nn.Embedding(256, cfg.d_model) if cfg.pos == "learned" else None
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: Tensor, pad_mask: Tensor) -> Tensor:
        t = idx.shape[1]
        h = self.tok(idx)
        if self.learned_pos is not None:
            h = h + self.learned_pos(torch.arange(t, device=idx.device))[None]
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=idx.device))
        mask = causal[None, None] & (~pad_mask)[:, None, None, :]  # (B,1,T,T) True=attend
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.norm(h))


# ============================================================================================
# Schedule, evaluation, checkpointing
# ============================================================================================


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model: CausalReasoner, traces: list[list[str]], vocab: Vocab, device: str) -> float:
    model.eval()
    correct = 0
    for toks in traces:
        a = toks.index("[A]")
        ids = torch.tensor(vocab.encode(toks[: a + 1]), device=device)[None]
        pad = torch.zeros_like(ids, dtype=torch.bool)
        logits = model(ids, pad)[0, -1]
        pred = vocab.yes if logits[vocab.yes] > logits[vocab.no] else vocab.no
        correct += int(pred == vocab.stoi[toks[a + 1]])
    return correct / len(traces)


def save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, sched: object,
              scaler: object, epoch: int, best: float, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),  # type: ignore[attr-defined]
            "scaler": scaler.state_dict(),  # type: ignore[attr-defined]
            "epoch": epoch,
            "best": best,
            "cfg": asdict(cfg),
        },
        path,
    )


# ============================================================================================
# Train
# ============================================================================================


def train(cfg: Config) -> None:
    device = cfg.resolved_device()
    use_amp = cfg.amp and device == "cuda"
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    seed_everything(cfg.seed, cfg.deterministic)
    vocab = Vocab(cfg.max_nodes)
    log(f"device={device} amp={use_amp} task={cfg.task} pos={cfg.pos} vocab={len(vocab)}")

    log("generating traces from causalrl (generator + oracle) ...")
    train_traces = build_split(cfg, cfg.train_sizes, cfg.n_train, cfg.seed + 1, vocab)
    eval_traces = build_split(cfg, cfg.eval_sizes, cfg.n_eval, cfg.seed + 2, vocab)
    extrap_traces = build_split(cfg, cfg.extrap_sizes, cfg.n_eval, cfg.seed + 3, vocab)
    log(f"train={len(train_traces)} eval={len(eval_traces)} extrap={len(extrap_traces)}  "
        f"example: {' '.join(train_traces[0])}")

    loader = DataLoader(
        TraceDataset(train_traces, vocab),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate(vocab.pad),
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = CausalReasoner(cfg, len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {n_params / 1e6:.2f}M params, {cfg.n_layers} layers, d_model={cfg.d_model}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(loader))
    warmup = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, warmup))
    scaler = torch.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad, label_smoothing=cfg.label_smoothing)

    start_epoch, best = 0, 0.0
    ckpt_last, ckpt_best = out / "last.pt", out / "best.pt"
    if cfg.resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        log(f"resumed from {ckpt_last} at epoch {start_epoch} (best={best:.3f})")

    metrics: list[dict[str, float]] = []
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        for step, (ids, pad) in enumerate(loader):
            ids, pad = ids.to(device), pad.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):  # type: ignore[attr-defined]
                logits = model(ids, pad)
                loss = loss_fn(logits[:, :-1].reshape(-1, len(vocab)), ids[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            nb += 1
            if step % cfg.log_every == 0:
                log(f"  epoch {epoch + 1}/{cfg.epochs} step {step}/{len(loader)} "
                    f"loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e}")

        do_eval = (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1
        row: dict[str, float] = {"epoch": epoch + 1, "loss": running / max(1, nb)}
        if do_eval:
            row["held_out"] = evaluate(model, eval_traces, vocab, device)
            row["extrap"] = evaluate(model, extrap_traces, vocab, device)
            log(f"epoch {epoch + 1}: loss {row['loss']:.3f}  held_out {row['held_out']:.3f}  "
                f"extrap {row['extrap']:.3f}")
            if row["held_out"] > best:
                best = row["held_out"]
                save_ckpt(ckpt_best, model, opt, sched, scaler, epoch, best, cfg)
                log(f"  new best held_out {best:.3f} -> saved {ckpt_best}")
        metrics.append(row)
        save_ckpt(ckpt_last, model, opt, sched, scaler, epoch, best, cfg)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log(f"done. best held_out {best:.3f}. checkpoints in {out}/")


# ============================================================================================
# CLI
# ============================================================================================


def build_config() -> Config:
    d = Config()
    p = argparse.ArgumentParser(
        description="Robust trainer for the small causal reasoner (see module docstring).",
    )
    p.add_argument("--task", choices=["dsep", "ident"], default=d.task)
    p.add_argument("--pos", choices=["rope", "nope", "learned"], default=d.pos)
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-eval", type=int, default=d.n_eval)
    p.add_argument("--train-sizes", type=int, nargs="+", default=list(d.train_sizes))
    p.add_argument("--eval-sizes", type=int, nargs="+", default=list(d.eval_sizes))
    p.add_argument("--extrap-sizes", type=int, nargs="+", default=list(d.extrap_sizes))
    p.add_argument("--d-model", type=int, default=d.d_model)
    p.add_argument("--layers", type=int, default=d.n_layers)
    p.add_argument("--heads", type=int, default=d.n_heads)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--eval-every", type=int, default=d.eval_every, help="epochs between evals")
    p.add_argument("--log-every", type=int, default=d.log_every, help="steps between log lines")
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--max-nodes", type=int, default=d.max_nodes)
    p.add_argument("--out", default=d.out)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny fast CPU sanity run")
    a = p.parse_args()

    cfg = Config(
        task=a.task, pos=a.pos, n_train=a.n_train, n_eval=a.n_eval,
        train_sizes=tuple(a.train_sizes), eval_sizes=tuple(a.eval_sizes),
        extrap_sizes=tuple(a.extrap_sizes), d_model=a.d_model, n_layers=a.layers,
        n_heads=a.heads, dropout=a.dropout, batch_size=a.batch_size, epochs=a.epochs,
        lr=a.lr, seed=a.seed, device=a.device, amp=not a.no_amp,
        deterministic=a.deterministic, num_workers=a.num_workers, max_nodes=a.max_nodes,
        eval_every=a.eval_every, log_every=a.log_every, weight_decay=a.weight_decay,
        grad_clip=a.grad_clip, out=a.out, resume=a.resume,
    )
    if a.smoke:
        cfg = Config(
            task=a.task, pos=a.pos, n_train=600, n_eval=200, d_model=64, n_layers=2,
            n_heads=4, epochs=3, eval_every=1, log_every=5, device=a.device, amp=False,
            out=a.out,
        )
    return cfg


# auto-scale defaults when a GPU is present and the user did not override the scale flags
def maybe_autoscale(cfg: Config) -> Config:
    default = Config()
    untouched = cfg.d_model == default.d_model and cfg.n_train == default.n_train
    if cfg.resolved_device() == "cuda" and untouched:
        cfg.d_model, cfg.n_layers, cfg.n_heads = 256, 6, 8
        cfg.n_train, cfg.batch_size, cfg.epochs = 40000, 256, 60
        cfg.train_sizes, cfg.extrap_sizes = (3, 4, 5, 6, 7), (8, 9, 10)
    return cfg


if __name__ == "__main__":
    train(maybe_autoscale(build_config()))
