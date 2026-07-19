"""E5 (garnish) — certificate throughput on large learner populations via JAX.

Vectorises the E2 Hedge population over N independent replicates with jax.vmap/lax.scan, then
measures per-replicate realized regrets against the (single, shared) CCE polytope and the
containment rate in the measured-epsilon intervals. Supports the library paper, not the theory.

Run:  uv run python experiments/eqcf/e5_jax_scale.py   (skips gracefully without the [jax] extra)
"""

from __future__ import annotations

import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None

from causalrl.magames import cce_bounds, cce_polytope

import common
from e2_saf_chaos import EPS_X, EPS_Y, TAU, saf_matrices

N_REPLICATES = 10_000
HORIZON = 2_000


def main() -> None:
    print("=" * 78)
    print("E5 JAX scale garnish")
    print("=" * 78)
    if jax is None:
        print("SKIP: jax is not installed (install the [jax] extra); nothing to run.")
        return

    ux, uy = saf_matrices(EPS_X, EPS_Y, tau=TAU)
    game = common.bimatrix_game(ux, uy, names=("X", "Y"))
    poly = cce_polytope(game)
    gains = jnp.asarray(poly.deviation_gains)
    profile_index = {p: k for k, p in enumerate(poly.profiles)}
    ux_j, uy_j = jnp.asarray(ux), jnp.asarray(uy)
    eta = float(np.sqrt(8.0 * np.log(3) / HORIZON))

    def step(carry, key):
        wx, wy, counts = carry
        kx, ky = jax.random.split(key)
        ax = jax.random.categorical(kx, wx)
        ay = jax.random.categorical(ky, wy)
        counts = counts.at[ax * 3 + ay].add(1.0)
        wx = wx + eta * ux_j[:, ay]
        wy = wy + eta * uy_j[ax, :]
        return (wx, wy, counts), None

    def replicate(key):
        keys = jax.random.split(key, HORIZON)
        (wx, wy, counts), _ = jax.lax.scan(
            step, (jnp.zeros(3), jnp.zeros(3), jnp.zeros(9)), keys
        )
        mu = counts / HORIZON
        eps = jnp.max(gains @ mu)
        return mu, eps

    keys = jax.random.split(jax.random.PRNGKey(0), N_REPLICATES)
    start = time.perf_counter()
    mus, epsilons = jax.jit(jax.vmap(replicate))(keys)
    epsilons.block_until_ready()
    elapsed = time.perf_counter() - start

    def payoff_x(profile) -> float:
        return float(ux[profile["X"], profile["Y"]])

    values = np.array(
        [payoff_x(dict(zip(poly.agents, p, strict=True))) for p in poly.profiles]
    )
    realized = np.asarray(mus) @ values
    eps_max = float(np.max(np.asarray(epsilons)))
    interval = cce_bounds(game, payoff_x, epsilon=max(eps_max, 0.0))
    inside = np.mean((realized >= interval.lower - 1e-9) & (realized <= interval.upper + 1e-9))

    print(f"{N_REPLICATES} replicates x {HORIZON} rounds in {elapsed:.2f}s "
          f"({N_REPLICATES * HORIZON / elapsed:,.0f} learner-steps/s)")
    print(f"max measured regret across replicates: {eps_max:.5f}")
    print(f"shared measured-eps CCE interval: [{interval.lower:.4f}, {interval.upper:.4f}]")
    print(f"containment rate of realized time-averages: {inside:.4f} (must be 1.0)")


if __name__ == "__main__":
    main()
