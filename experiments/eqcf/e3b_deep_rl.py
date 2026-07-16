"""E3b — Deep-RL populations vs the CCE bounds, batched on GPU (the proposal's real E3).

Neural policy-gradient learners (per-replicate 2-layer MLP policies, REINFORCE-with-baseline —
the core update of the PPO family) play the same Cournot stage game as E6. Two conditions,
R independent replicate populations each, trained simultaneously as one batched GPU computation:

  stateless : policy sees a constant input  -> the approximately-no-regret regime;
  memory-1  : policy sees the last joint action -> history-dependent strategies are learnable.

IMPORTANT design fact for interpreting the result: these learners maximize IMMEDIATE stage
reward (no discounting), so they are myopic by construction and cannot represent intertemporal
punishment threats even with memory — the memory-1 condition isolates "state" from
"farsightedness". The open cell (farsighted gamma > 0 policy gradient, matched to E6's horizon)
is the follow-up; this run says nothing about it.

Readout per replicate: measured stage-game regret eps_T and average profit over the tail window;
across replicates: quartiles, fraction outside the exact stage-CCE, fraction supra-competitive.
One decisive batched run instead of a battery — the replicate axis is the power.

Run:  uv run python experiments/eqcf/e3b_deep_rl.py   (needs torch; uses CUDA if available)
"""

from __future__ import annotations

import numpy as np
import torch

from causalrl.magames import cce_bounds, cce_polytope, certify_cce_do

import common
from e6_collusion import payoff_matrices

R = 256  # independent replicate populations per condition
T_TRAIN = 20_000
T_TAIL = 5_000  # measurement window
HIDDEN = 32
LR = 3e-3
ENT_COEF = 0.01
N_ACT = 5


class BatchedMLPAgents:
    """R independent 2-layer MLP policies as one batched tensor computation."""

    def __init__(self, n_in: int, device: torch.device, gen: torch.Generator) -> None:
        def init(*shape: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                0.1 * torch.randn(*shape, generator=gen, device=device)
            )

        self.w1, self.b1 = init(R, n_in, HIDDEN), init(R, HIDDEN)
        self.w2, self.b2 = init(R, HIDDEN, N_ACT), init(R, N_ACT)
        self.baseline = torch.zeros(R, device=device)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.w1, self.b1, self.w2, self.b2]

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(torch.einsum("rih,ri->rh", self.w1, obs) + self.b1)
        return torch.einsum("rha,rh->ra", self.w2, hidden) + self.b2


def run_condition(memory: bool, seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Train R replicate pairs; return (eps_T, mean profit of firm 1) per replicate."""
    torch.manual_seed(seed)  # Categorical.sample draws from the global per-device RNG
    gen = torch.Generator(device=device).manual_seed(seed)
    u1_np, u2_np = payoff_matrices()
    u1 = torch.tensor(u1_np, dtype=torch.float32, device=device)
    u2 = torch.tensor(u2_np, dtype=torch.float32, device=device)
    n_in = N_ACT * N_ACT if memory else 1
    agents = (BatchedMLPAgents(n_in, device, gen), BatchedMLPAgents(n_in, device, gen))
    opt = torch.optim.Adam([p for a in agents for p in a.parameters()], lr=LR)

    state = torch.zeros(R, dtype=torch.long, device=device)
    counts = torch.zeros(R, N_ACT, N_ACT, device=device)
    profit_sum = torch.zeros(R, device=device)
    replicate_index = torch.arange(R, device=device)

    for t in range(T_TRAIN):
        if memory:
            obs = torch.nn.functional.one_hot(state, N_ACT * N_ACT).float()
        else:
            obs = torch.ones(R, 1, device=device)
        logits = [a.logits(obs) for a in agents]
        dists = [torch.distributions.Categorical(logits=lg) for lg in logits]
        actions = [d.sample() for d in dists]
        rewards = (
            u1[actions[0], actions[1]],
            u2[actions[0], actions[1]],
        )
        loss = torch.zeros((), device=device)
        for agent, dist, act, reward in zip(agents, dists, actions, rewards, strict=True):
            advantage = reward - agent.baseline
            loss = loss - (dist.log_prob(act) * advantage.detach()).sum()
            loss = loss - ENT_COEF * dist.entropy().sum()
            agent.baseline = 0.99 * agent.baseline.detach() + 0.01 * reward
        opt.zero_grad()
        loss.backward()
        opt.step()
        state = actions[0] * N_ACT + actions[1]
        if t >= T_TRAIN - T_TAIL:
            counts[replicate_index, actions[0], actions[1]] += 1.0
            profit_sum += rewards[0]

    mu = (counts / T_TAIL).reshape(R, -1).cpu().numpy()
    game = common.bimatrix_game(u1_np, u2_np, names=("F1", "F2"))
    gains = cce_polytope(game).deviation_gains  # rows align with reshape(R, 25) profile order
    eps = np.maximum(mu @ gains.T, 0.0).max(axis=1)
    return eps, (profit_sum / T_TAIL).cpu().numpy()


def report(label: str, eps: np.ndarray, profits: np.ndarray, exact) -> None:
    outside = float(np.mean(profits > exact.upper + 1e-6) + np.mean(profits < exact.lower - 1e-6))
    supra = float(np.mean(profits > 16.0 + 0.25))
    q = np.percentile
    print(f"\n--- {label} (R={R} replicates) ---")
    print(f"  eps_T quartiles: {q(eps, 25):.3f} / {q(eps, 50):.3f} / {q(eps, 75):.3f}")
    print(f"  profit quartiles: {q(profits, 25):.2f} / {q(profits, 50):.2f} / {q(profits, 75):.2f}"
          f"   (Nash 16, collusive 18)")
    print(f"  fraction with profit outside exact CCE {list(np.round(exact, 3))}: {outside:.3f}")
    print(f"  fraction supra-competitive (> 16.25): {supra:.3f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"E3b deep-RL populations vs CCE bounds — device: {device}")
    print("=" * 78)
    u1_np, u2_np = payoff_matrices()
    game = common.bimatrix_game(u1_np, u2_np, names=("F1", "F2"))

    def profit1(profile) -> float:
        return float(u1_np[profile["F1"], profile["F2"]])

    exact = cce_bounds(game, profit1)
    print(f"exact stage-CCE interval for firm-1 profit: [{exact.lower:.3f}, {exact.upper:.3f}]")

    eps_s, prof_s = run_condition(memory=False, seed=0, device=device)
    report("stateless neural policy gradient", eps_s, prof_s, exact)

    eps_m, prof_m = run_condition(memory=True, seed=1, device=device)
    report("memory-1 neural policy gradient", eps_m, prof_m, exact)

    med_eps = float(np.median(np.concatenate([eps_s, eps_m])))
    cert = certify_cce_do(game, profit1, no_regret=False, epsilon=med_eps)
    print(f"\ncertificate at pooled median measured eps ({med_eps:.3f}): {cert}")
    sens = cert.witness.detail["epsilon_sensitivity"]
    print(f"epsilon sensitivity (marginal interval growth per unit regret): {sens}")


if __name__ == "__main__":
    main()
