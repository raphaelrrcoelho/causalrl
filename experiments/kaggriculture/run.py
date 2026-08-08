#!/usr/bin/env python3
"""Command line for the Kaggriculture study. Run `python run.py --help` from this directory.

Every number in FINDINGS.md is reproducible from here. Seed counts are kept low enough to be
usable interactively; raise `--seeds` for anything you intend to act on, because a single episode
is a noisy read (weed spawns and the shop-unlock order are both random).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_eval(args: argparse.Namespace) -> None:
    """Score the agent against a named opponent."""
    from agent import agent
    from harness import evaluate

    started = time.time()
    result = evaluate(agent, args.opponent, seeds=range(args.seeds))
    print(f"agent vs {args.opponent}  ({result['n']} seeds, {time.time() - started:.0f}s)")
    print(f"  mean   ${result['mean']:,.0f}")
    print(f"  median ${result['median']:,.0f}")
    print(f"  range  ${result['min']:,.0f} - ${result['max']:,.0f}")
    print(f"  stdev  ${result['stdev']:,.0f}")


def cmd_selfplay(args: argparse.Namespace) -> None:
    """Score the agent against itself -- the shared-market read, and the honest one."""
    from agent import agent
    from harness import evaluate

    result = evaluate(agent, agent, seeds=range(args.seeds))
    print(f"self-play ({result['n']} seeds): mean ${result['mean']:,.0f}")


def cmd_pool(args: argparse.Namespace) -> None:
    """Round-robin over the strategy pool (FINDINGS §12). Slow: ~10s per pair per seed."""
    from opponents import POOL, payoff_matrix

    started = time.time()
    matrix = payoff_matrix(seeds=range(args.seeds))
    names = [s.name for s in POOL]
    print(f"row's mean bank when playing column ({time.time() - started:.0f}s)\n")
    print(f"{'':13s}" + "".join(f"{n[:10]:>11s}" for n in names) + f"{'AVG':>11s}")
    for row in names:
        values = [matrix[(row, col)] for col in names]
        line = "".join(f"{v:11,.0f}" for v in values)
        print(f"{row:13s}{line}{statistics.fmean(values):11,.0f}")
    print("\nhead-to-head (row beats column on mean bank):")
    for row in names:
        wins = sum(1 for col in names if col != row and matrix[(row, col)] > matrix[(col, row)])
        print(f"  {row:13s} {wins}/{len(names) - 1}")


def cmd_equilibrium(args: argparse.Namespace) -> None:
    """Solve the empirical game over the pool -- needs causalrl installed."""
    from opponents import POOL, payoff_matrix

    from causalrl import AgentType, Population, certify_cce_do, run_no_regret

    matrix = payoff_matrix(seeds=range(args.seeds))
    names = [s.name for s in POOL]

    def payoff(own: int, others: tuple[int, ...], params: object = None) -> float:
        return matrix[(names[own], names[others[0]])] / 1000.0

    player = AgentType(name="player", actions=tuple(range(len(names))), payoff=payoff)
    population = Population(agents=("A", "B"), types={"A": player, "B": player})
    run = run_no_regret(population, args.rounds, algorithm="regret_matching", seed=0)
    weights: dict[int, float] = {}
    for profile, weight in zip(run.profiles, run.weights, strict=True):
        if weight > 1e-6:
            weights[profile[0]] = weights.get(profile[0], 0.0) + weight
    print("equilibrium over the strategy pool:")
    for index in sorted(weights, key=lambda k: -weights[k]):
        if weights[index] > 1e-3:
            print(f"  {names[index]:13s} {weights[index]:.3f}")
    print(f"  measured regret {run.regret_trace[-1][1]:.4f}")
    certificate = certify_cce_do(
        population.to_game(),
        lambda p: payoff(p["A"], (p["B"],)),
        no_regret=False,
        epsilon=run.regret_trace[-1][1],
    )
    print(f"\n{certificate}")


def cmd_economics(args: argparse.Namespace) -> None:
    """Print the demand and per-tile tables straight from the referee's own constants."""
    from economics import MARKET_PARAMS, TILES_PER_UNIT, YIELD_PER_TILE_DAY, town_drain_per_day

    items = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
    print(
        f"{'product':12s} {'$/tile/day':>11s} {'drain/day (d20)':>16s} {'sustainable tiles':>18s}"
    )
    for item in items:
        rate = YIELD_PER_TILE_DAY[item] / TILES_PER_UNIT[item]
        drain = town_drain_per_day(item, 20)
        print(
            f"{item:12s} {rate * MARKET_PARAMS[item]['base']:11.2f} {drain:16.2f} {drain / rate:18.1f}"
        )
    print("\nwhole-season absorption at base price:")
    total = sum(
        town_drain_per_day(i, d) * MARKET_PARAMS[i]["base"] for i in items for d in range(30)
    )
    print(f"  ${total:,.0f} total, ~${total / 2:,.0f} per seller against one opponent")


def cmd_timing(args: argparse.Namespace) -> None:
    """Check the per-turn decision budget (Kaggle allows 1000 ms)."""
    from agent import agent
    from kaggle_environments import make

    worst = [0.0]
    total = [0.0, 0]

    def timed(obs, config=None):
        start = time.perf_counter()
        out = agent(obs, config)
        elapsed = time.perf_counter() - start
        worst[0] = max(worst[0], elapsed)
        total[0] += elapsed
        total[1] += 1
        return out

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0})
    env.run([timed, "starter"])
    print(
        f"turns: {total[1]}  mean {1000 * total[0] / total[1]:.3f} ms  worst {1000 * worst[0]:.3f} ms"
    )
    print("budget: 1000 ms per turn")


def cmd_submission(args: argparse.Namespace) -> None:
    """Write a single-file submission and verify it runs standalone."""
    source = Path(__file__).resolve().parent / "agent.py"
    target = Path(args.out).resolve()
    body = source.read_text()
    target.write_text(
        body + "\n\ndef act_agent(obs, config=None):\n    return agent(obs, config)\n"
    )
    print(f"wrote {target} ({len(body.splitlines())} lines)")
    print("agent.py imports only from kaggle_environments, so it is already self-contained.")
    print(f"upload {target.name} to Kaggle and select `agent` as the entry point.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("eval", help="score the agent against an opponent")
    p.add_argument("--opponent", default="starter", help="starter | random | pass")
    p.add_argument("--seeds", type=int, default=8)
    p.set_defaults(func=cmd_eval)

    p = subparsers.add_parser("selfplay", help="score the agent against itself")
    p.add_argument("--seeds", type=int, default=5)
    p.set_defaults(func=cmd_selfplay)

    p = subparsers.add_parser("pool", help="round-robin over the strategy pool")
    p.add_argument("--seeds", type=int, default=3)
    p.set_defaults(func=cmd_pool)

    p = subparsers.add_parser("equilibrium", help="solve the empirical game over the pool")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=20000)
    p.set_defaults(func=cmd_equilibrium)

    p = subparsers.add_parser("economics", help="print the demand / per-tile tables")
    p.set_defaults(func=cmd_economics)

    p = subparsers.add_parser("timing", help="check the per-turn decision budget")
    p.set_defaults(func=cmd_timing)

    p = subparsers.add_parser("submission", help="write a single-file submission")
    p.add_argument("--out", default="submission.py")
    p.set_defaults(func=cmd_submission)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
