"""E8 — the open cell: does FARSIGHTED policy gradient collude? (matched horizon, decisive pair)

The adversarial review's verdict on E3b: myopic-by-construction learners cannot represent
punishment threats, and the E6 comparison was horizon-confounded (2M vs 20k). This experiment
closes both holes on the same Cournot game at E6's full horizon (T = 2M steps):

    two PG cells, both memory-1 (policy sees the last joint action), R replicates each:
      myopic     gamma = 0     — state but no motive   (control; E3b's cell at matched horizon)
      farsighted gamma = 0.95  — state AND motive      (THE cell)
    farsightedness via truncated n-step DISCOUNTED RETURNS with a running-mean baseline and NO
    bootstrapped critic — isolating the intertemporal objective from value bootstrapping;
    plus the memory-1 Q anchor (bootstrapped + farsighted, same horizon, from E6).

Stateless cells are omitted deliberately (E3b answered myopic-stateless; farsighted-stateless has
no channel to condition punishment on). Predictions under the "folk-theorem ingredient =
intertemporal objective" hypothesis: myopic -> Nash, farsighted -> collusion possible. If instead
only the bootstrapped Q anchor colludes, the driver narrows to bootstrapping specifically —
either way the mechanism claim sharpens.

Run:  uv run python experiments/eqcf/e8_farsighted_2x2.py   (CUDA; ~2-4h)
"""

from __future__ import annotations

import time

import numpy as np
import torch

from causalrl.magames import cce_polytope

import common
from e6_collusion import memory1_q_run, payoff_matrices

R = 64
T_TRAIN = 2_000_000
T_TAIL = 250_000
N_STEP = 32  # truncation for discounted returns (0.95^32 ~ 0.19)
HIDDEN = 32
LR = 1e-3
ENT_COEF = 0.005
N_ACT = 5
N_STATE = N_ACT * N_ACT


def run_pg_cell(gamma: float, seed: int, device: torch.device) -> tuple[np.ndarray, ...]:
    """One memory-1 PG cell with both agents stacked into a single (2R, ...) batch."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    u1_np, u2_np = payoff_matrices()
    u1 = torch.tensor(u1_np, dtype=torch.float32, device=device)
    u2 = torch.tensor(u2_np, dtype=torch.float32, device=device)

    def init(*shape: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(0.1 * torch.randn(*shape, generator=gen, device=device))

    b = 2 * R  # agent-replicate batch: [0:R] = firm 1, [R:2R] = firm 2
    w1, b1 = init(b, N_STATE, HIDDEN), init(b, HIDDEN)
    w2, b2 = init(b, HIDDEN, N_ACT), init(b, N_ACT)
    baseline = torch.zeros(b, device=device)
    opt = torch.optim.Adam([w1, b1, w2, b2], lr=LR)

    state = torch.zeros(R, dtype=torch.long, device=device)
    counts = torch.zeros(R, N_ACT, N_ACT, device=device)
    profit_sum = torch.zeros(R, device=device)
    welfare_sum = torch.zeros(R, device=device)
    ridx = torch.arange(R, device=device)

    rew_buf = torch.zeros(N_STEP, b, device=device)
    buf_lp: list[torch.Tensor] = []
    buf_ent: list[torch.Tensor] = []
    fill = 0

    for t in range(T_TRAIN):
        obs = torch.nn.functional.one_hot(state, N_STATE).float()
        obs2 = obs.repeat(2, 1)  # both agents observe the same last joint action
        hidden = torch.relu(torch.einsum("bih,bi->bh", w1, obs2) + b1)
        logits = torch.einsum("bha,bh->ba", w2, hidden) + b2
        logp = torch.log_softmax(logits, dim=-1)
        gumbel = -torch.log(-torch.log(torch.rand(b, N_ACT, device=device)))
        actions = torch.argmax(logits.detach() + gumbel, dim=-1)
        a1, a2 = actions[:R], actions[R:]
        r = torch.cat([u1[a1, a2], u2[a1, a2]])

        buf_lp.append(logp.gather(1, actions[:, None]).squeeze(1))
        buf_ent.append(-(logp.exp() * logp).sum(dim=-1))
        rew_buf[fill] = r
        fill += 1

        state = a1 * N_ACT + a2
        if t >= T_TRAIN - T_TAIL:
            counts[ridx, a1, a2] += 1.0
            profit_sum += r[:R]
            welfare_sum += r[:R] + r[R:]

        if fill == N_STEP:
            acc = torch.zeros(b, device=device)
            returns = torch.zeros(N_STEP, b, device=device)
            for s in range(N_STEP - 1, -1, -1):
                acc = rew_buf[s] + gamma * acc
                returns[s] = acc
            advantage = returns - baseline
            lp_all = torch.stack(buf_lp)
            ent_all = torch.stack(buf_ent)
            loss = -(lp_all * advantage.detach()).mean(dim=0).sum()
            loss = loss - ENT_COEF * ent_all.mean(dim=0).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            baseline = 0.95 * baseline + 0.05 * returns.mean(dim=0)
            buf_lp, buf_ent, fill = [], [], 0

    mu = (counts / counts.sum(dim=(1, 2), keepdim=True)).reshape(R, -1).cpu().numpy()
    game = common.bimatrix_game(u1_np, u2_np, names=("F1", "F2"))
    gains = cce_polytope(game).deviation_gains
    eps = np.maximum(mu @ gains.T, 0.0).max(axis=1)
    denom = counts.sum(dim=(1, 2)).cpu().numpy()
    return eps, profit_sum.cpu().numpy() / denom, welfare_sum.cpu().numpy() / denom


def report(label: str, eps, profits, welfare) -> None:
    q = np.percentile
    collusive = float(np.mean(np.asarray(welfare) > 33.0))
    print(f"\n--- {label} (n={len(eps)}) ---", flush=True)
    print(f"  eps_T q25/50/75: {q(eps, 25):.3f} / {q(eps, 50):.3f} / {q(eps, 75):.3f}")
    print(f"  profit q25/50/75: {q(profits, 25):.2f} / {q(profits, 50):.2f} / {q(profits, 75):.2f}")
    print(f"  welfare q25/50/75: {q(welfare, 25):.2f} / {q(welfare, 50):.2f} / "
          f"{q(welfare, 75):.2f}   (CCE-degenerate at 32, collusive 36)")
    print(f"  fraction clearly collusive (welfare > 33): {collusive:.3f}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"E8 matched-horizon decisive pair — device: {device}, T={T_TRAIN:,}, R={R}")
    print("=" * 78, flush=True)

    for gamma, tag in ((0.0, "myopic g=0 (control)"), (0.95, "farsighted g=0.95 (THE cell)")):
        start = time.perf_counter()
        eps, prof, wel = run_pg_cell(gamma, seed=42, device=device)
        report(f"PG memory-1 x {tag} [{time.perf_counter() - start:.0f}s]", eps, prof, wel)

    # Anchor: bootstrapped + farsighted memory-1 Q at the SAME horizon (E6 machinery).
    u1, u2 = payoff_matrices()
    game = common.bimatrix_game(u1, u2, names=("F1", "F2"))
    poly = cce_polytope(game)
    eps_q, prof_q, wel_q = [], [], []
    for seed in (0, 1, 2):
        mu = memory1_q_run(seed)
        vec = np.array([mu.get(p, 0.0) for p in poly.profiles])
        eps_q.append(float(np.max(np.maximum(poly.deviation_gains @ vec, 0.0))))
        prof_q.append(float(sum(w * u1[p] for p, w in mu.items())))
        wel_q.append(float(sum(w * (u1[p] + u2[p]) for p, w in mu.items())))
    report("Q memory-1, bootstrapped+farsighted (anchor, 3 seeds)",
           np.array(eps_q), np.array(prof_q), np.array(wel_q))


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:  # shape/loss-path check only
        T_TRAIN, T_TAIL, R = 2_000, 500, 8
    main()
