"""Causal as core design: a graph transformer whose ATTENTION IS THE CAUSAL GRAPH.

Every other example treats causality as something layered on top of a causally-naive transformer —
control tokens in the data, or a separate NCM module. Here causality is the core of the
computation. The transformer does not attend over a token stream with a temporal triangular mask;
it attends over the **variables of a causal graph**, and the graph structure is wired into the
attention itself:

* **Tokens are variables (nodes), not subwords.** A node carries only its *role* (treatment X,
  outcome Y, in the conditioning set Z, or plain) — never a fixed identity. The model is therefore
  permutation-invariant over variables and indifferent to graph size, which is the right inductive
  bias for a structural rule and is what the sequence model lacked (its size-extrapolation failure
  came from leaning on edge-listing order and absolute sequence position).
* **Attention is biased by the causal graph.** For every ordered pair of variables the attention
  score gets a learned, per-head bias keyed on their *causal relation* — parent, child, confounded
  (bidirected), or none (Graphormer-style relational encoding). Information flows along the graph.
* **do(X) is edge ablation, native to the architecture.** Intervening on X severs its incoming
  "parent" relations in the relation matrix — Pearl's graph surgery, implemented as a mask edit, not
  a learned token. ``CausalGraphTransformer.intervene`` does exactly this.

The internals are the modern small-LM standard (the "Llama recipe"): **RMSNorm**, **SwiGLU**,
bias-free linears, pre-norm residual blocks.

causalrl is the trace generator and the d-separation oracle. We train on small graphs (3-5 vars)
and test on larger ones (6-7) to measure whether making structure the core — rather than an add-in —
improves the size generalisation that defeated the sequence model.

Run (CPU smoke)::

    uv run --extra torch python examples/causal_graph_transformer.py --smoke

Full / GPU::

    uv run --extra torch python examples/causal_graph_transformer.py --epochs 40 --d-model 192 \
        --layers 6 --n-train 40000 --out runs/cgt

The next causal-core layers — exogenous-noise inputs with twin networks for L3 counterfactuals,
and an identifiability-aware objective that abstains on non-identifiable queries — are described in
examples/CAUSAL_LLM_RESEARCH.md. This file establishes the foundation they build on. Didactic
research scaffold, not a performance guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from causalrl import CausalGraph
from causalrl.identification._separation import d_separated

# Roles a variable can play in a query, and causal relations between two variables.
PLAIN, ROLE_X, ROLE_Y, ROLE_Z, ROLE_PAD = 0, 1, 2, 3, 4
SELF, PARENT, CHILD, CONFOUND, NONE = 0, 1, 2, 3, 4
N_ROLES, N_RELATIONS = 5, 5


# ============================================================================================
# Config
# ============================================================================================


@dataclass
class Config:
    task: str = "dsep"
    train_sizes: tuple[int, ...] = (3, 4, 5)
    eval_sizes: tuple[int, ...] = (3, 4, 5)
    extrap_sizes: tuple[int, ...] = (6, 7)
    n_train: int = 8000
    n_eval: int = 1000
    edge_prob: float = 0.4
    max_cond: int = 2
    max_nodes: int = 16
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0
    batch_size: int = 128
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    eval_every: int = 5
    curriculum: str = "balanced"  # "balanced" (class-balanced) | "stratified" (difficulty-balanced)
    ablate_structure: bool = False  # if True, attention is structure-blind (causal-bias ablation)
    out: str = "runs/causal_graph_transformer"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================================================
# Example: a graph rendered as (roles, relation matrix, query), labelled by causalrl
# ============================================================================================


@dataclass
class Example:
    roles: list[int]
    relations: list[list[int]]  # relations[i][j] = relation of j w.r.t. i (for attention i <- j)
    x: int
    y: int
    label: bool


def _make_example(cfg: Config, n: int, rng: random.Random, pool: list[str]) -> tuple[Example, str]:
    names = rng.sample(pool, n)
    order = names[:]
    rng.shuffle(order)
    directed = [
        (order[i], order[j])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < cfg.edge_prob
    ]
    graph = CausalGraph(directed_edges=directed, nodes=names)

    x, y = rng.sample(names, 2)
    rest = [v for v in names if v not in (x, y)]
    z = rng.sample(rest, rng.randint(0, min(cfg.max_cond, len(rest))))
    label = d_separated(graph, {x}, {y}, set(z))  # the oracle
    # difficulty stratum (compare separation with and without Z) — used by the stratified curriculum
    sep_empty = d_separated(graph, {x}, {y}, set())
    if (x, y) in directed or (y, x) in directed:
        stratum = "adjacent"
    elif sep_empty and not label:
        stratum = "collider_open"
    elif not sep_empty and label:
        stratum = "blocked"
    elif label:
        stratum = "sep_robust"
    else:
        stratum = "conn_robust"

    idx = {name: i for i, name in enumerate(names)}
    roles = [PLAIN] * n
    roles[idx[x]] = ROLE_X
    roles[idx[y]] = ROLE_Y
    for zz in z:
        roles[idx[zz]] = ROLE_Z

    rel = [[NONE] * n for _ in range(n)]
    for i in range(n):
        rel[i][i] = SELF
    for a, b in graph.directed_edges:  # a -> b
        rel[idx[b]][idx[a]] = PARENT  # a is a parent of b
        rel[idx[a]][idx[b]] = CHILD  # b is a child of a
    return Example(roles, rel, idx[x], idx[y], label), stratum


# the five difficulty strata; the stratified curriculum fills the hard ones (which are rare in
# random graphs) to equal share, so the model cannot coast on adjacency / no-path shortcuts.
STRATA = ("adjacent", "sep_robust", "conn_robust", "blocked", "collider_open")


def build_split(cfg: Config, sizes: tuple[int, ...], n: int, seed: int, pool: list[str]
                ) -> list[Example]:
    rng = random.Random(seed)
    pos: list[Example] = []
    neg: list[Example] = []
    target = n // 2
    while len(pos) < target or len(neg) < target:
        ex, _ = _make_example(cfg, rng.choice(sizes), rng, pool)
        bucket = pos if ex.label else neg
        if len(bucket) < target:
            bucket.append(ex)
    data = pos + neg
    rng.shuffle(data)
    return data


def build_stratified(cfg: Config, sizes: tuple[int, ...], n: int, seed: int, pool: list[str]
                     ) -> list[Example]:
    """Difficulty-balanced set: each of the five strata filled to ~n/5 (rejection-sampled).

    Forces the genuinely structural cases (multi-hop ``conn_robust``, ``collider_open``) — rare in
    random graphs — to ~20% each, so the model must learn path-tracing and the collider rule rather
    than the adjacency / no-path shortcuts a class-balanced set lets it coast on.
    """
    rng = random.Random(seed)
    per = n // len(STRATA)
    buckets: dict[str, list[Example]] = {s: [] for s in STRATA}
    guard = 0
    while any(len(buckets[s]) < per for s in STRATA):
        ex, st = _make_example(cfg, rng.choice(sizes), rng, pool)
        if len(buckets[st]) < per:
            buckets[st].append(ex)
        guard += 1
        if guard > 400 * n:  # safety valve if a stratum is unreachable at these sizes
            break
    data = [ex for s in STRATA for ex in buckets[s]]
    rng.shuffle(data)
    return data



class GraphDataset(Dataset[Example]):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def collate(batch: list[Example]) -> dict[str, Tensor]:
    b = len(batch)
    n = max(len(e.roles) for e in batch)
    roles = torch.full((b, n), ROLE_PAD, dtype=torch.long)
    rel = torch.full((b, n, n), NONE, dtype=torch.long)
    pad = torch.ones(b, n, dtype=torch.bool)  # True = padded
    x = torch.zeros(b, dtype=torch.long)
    y = torch.zeros(b, dtype=torch.long)
    label = torch.zeros(b, dtype=torch.long)
    for k, e in enumerate(batch):
        m = len(e.roles)
        roles[k, :m] = torch.tensor(e.roles)
        rel[k, :m, :m] = torch.tensor(e.relations)
        pad[k, :m] = False
        x[k], y[k], label[k] = e.x, e.y, int(e.label)
    return {"roles": roles, "rel": rel, "pad": pad, "x": x, "y": y, "label": label}


# ============================================================================================
# Llama-recipe internals: RMSNorm + SwiGLU + bias-free
# ============================================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = ((8 * d_model // 3 + 31) // 32) * 32
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


# ============================================================================================
# Graph attention: the causal graph IS the attention bias
# ============================================================================================


class GraphAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout
        # one learned scalar per (head, causal relation) — the structural attention bias
        self.rel_bias = nn.Parameter(torch.zeros(cfg.n_heads, N_RELATIONS))
        # ablation: make attention structure-blind to measure how much the causal bias contributes
        self.ablate = cfg.ablate_structure

    def forward(self, h: Tensor, rel: Tensor, pad: Tensor) -> Tensor:
        b, n, _ = h.shape
        q, k, v = self.qkv(h).split(h.shape[-1], dim=2)
        q = q.view(b, n, self.h, self.dh).transpose(1, 2)
        k = k.view(b, n, self.h, self.dh).transpose(1, 2)
        v = v.view(b, n, self.h, self.dh).transpose(1, 2)
        if self.ablate:
            # structure-blind: no relation bias, only role/value embeddings + padding mask
            bias = torch.zeros(b, 1, n, n, device=h.device)
        else:
            # structural bias: gather rel_bias[head, rel[b,i,j]] -> (B, H, N, N)
            bias = self.rel_bias[:, rel].permute(1, 0, 2, 3)
        # mask padded keys
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, self.h * self.dh)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = GraphAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.dropout)

    def forward(self, h: Tensor, rel: Tensor, pad: Tensor) -> Tensor:
        h = h + self.attn(self.n1(h), rel, pad)
        return h + self.mlp(self.n2(h))


class CausalGraphTransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(N_ROLES, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.readout = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2, bias=False),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def intervene(rel: Tensor, nodes: Tensor) -> Tensor:
        """do(nodes): graph surgery as a mask edit — sever each intervened node's parent relations.

        ``rel`` is (B, N, N); ``nodes`` is (B,) the intervened variable per example. Every incoming
        PARENT relation of the intervened node becomes NONE, exactly Pearl's edge deletion — the
        do-operator implemented natively in the attention structure, not learned from data.
        """
        rel = rel.clone()
        b = rel.shape[0]
        for k in range(b):
            row = rel[k, nodes[k]]
            row[row == PARENT] = NONE
        return rel

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.role_emb(batch["roles"])
        rel, pad = batch["rel"], batch["pad"]
        for blk in self.blocks:
            h = blk(h, rel, pad)
        h = self.norm(h)
        ar = torch.arange(h.shape[0], device=h.device)
        hx, hy = h[ar, batch["x"]], h[ar, batch["y"]]
        mean = (h * (~pad)[..., None]).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return self.readout(torch.cat([hx, hy, mean], dim=-1))


# ============================================================================================
# Training
# ============================================================================================


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model: CausalGraphTransformer, data: list[Example], device: str) -> float:
    model.eval()
    loader = DataLoader(GraphDataset(data), batch_size=256, collate_fn=collate)
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch).argmax(-1)
        correct += int((pred == batch["label"]).sum())
        total += pred.shape[0]
    return correct / total


def train(cfg: Config) -> None:
    device = cfg.resolved_device()
    use_amp = cfg.amp and device == "cuda"
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with (out / "train.log").open("a") as f:
            f.write(line + "\n")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    pool = list(string.ascii_uppercase[: cfg.max_nodes])
    log(f"device={device} amp={use_amp} task={cfg.task} (causal graph transformer)")

    builder = build_stratified if cfg.curriculum == "stratified" else build_split
    log(f"generating traces from causalrl (oracle); curriculum={cfg.curriculum} ...")
    train_data = builder(cfg, cfg.train_sizes, cfg.n_train, cfg.seed + 1, pool)
    eval_data = builder(cfg, cfg.eval_sizes, cfg.n_eval, cfg.seed + 2, pool)
    extrap_data = builder(cfg, cfg.extrap_sizes, cfg.n_eval, cfg.seed + 3, pool)

    loader = DataLoader(
        GraphDataset(train_data), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, pin_memory=(device == "cuda"),
    )
    model = CausalGraphTransformer(cfg).to(device)
    log(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{cfg.n_layers} layers, d_model={cfg.d_model}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * max(1, len(loader))
    warmup = int(cfg.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warmup))
    scaler = torch.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]
    loss_fn = nn.CrossEntropyLoss()

    best: float = 0.0
    metrics: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        running, nb = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device, enabled=use_amp):  # type: ignore[attr-defined]
                loss = loss_fn(model(batch), batch["label"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            nb += 1
        row: dict[str, float] = {"epoch": epoch + 1, "loss": running / max(1, nb)}
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            row["held_out"] = evaluate(model, eval_data, device)
            row["extrap"] = evaluate(model, extrap_data, device)
            log(f"epoch {epoch + 1}: loss {row['loss']:.3f}  held_out {row['held_out']:.3f}  "
                f"extrap {row['extrap']:.3f}")
            if row["held_out"] > best:
                best = row["held_out"]
                torch.save(model.state_dict(), out / "best.pt")
        metrics.append(row)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"done. best held_out {best:.3f}. checkpoints in {out}/")


def build_config() -> Config:
    d = Config()
    p = argparse.ArgumentParser(description="Causal graph transformer (attention = causal graph).")
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-eval", type=int, default=d.n_eval)
    p.add_argument("--train-sizes", type=int, nargs="+", default=list(d.train_sizes))
    p.add_argument("--extrap-sizes", type=int, nargs="+", default=list(d.extrap_sizes))
    p.add_argument("--d-model", type=int, default=d.d_model)
    p.add_argument("--layers", type=int, default=d.n_layers)
    p.add_argument("--heads", type=int, default=d.n_heads)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--curriculum", choices=["balanced", "stratified"], default=d.curriculum)
    p.add_argument("--ablate-structure", action="store_true",
                   help="make attention structure-blind (ablate the causal relational bias)")
    p.add_argument("--out", default=d.out)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        return Config(n_train=600, n_eval=200, d_model=64, n_layers=2, n_heads=4, epochs=4,
                      eval_every=1, device=a.device, amp=False, curriculum=a.curriculum,
                      ablate_structure=a.ablate_structure, out=a.out)
    return Config(
        n_train=a.n_train, n_eval=a.n_eval, train_sizes=tuple(a.train_sizes),
        extrap_sizes=tuple(a.extrap_sizes), d_model=a.d_model, n_layers=a.layers,
        n_heads=a.heads, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed,
        device=a.device, amp=not a.no_amp, eval_every=a.eval_every, curriculum=a.curriculum,
        ablate_structure=a.ablate_structure, out=a.out,
    )


if __name__ == "__main__":
    train(build_config())
