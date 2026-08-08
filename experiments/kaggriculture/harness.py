"""Local evaluation: run episodes, report the spread, and say what the farm actually did.

A single episode is a noisy read -- weed spawns and the shop-unlock order are both random -- so a
score from one seed says very little about whether a change helped. Everything here reports across
seeds, and :func:`compare` reports the *paired* difference, which removes the shared episode
randomness that a difference of two independent means would leave in.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any


def run_episode(first: Any, second: Any, *, seed: int = 0, steps: int = 720) -> tuple[float, float]:
    """One full season; returns both players' final bank."""
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.run([first, second])
    rewards = [s.reward for s in env.steps[-1]]
    return float(rewards[0] or 0.0), float(rewards[1] or 0.0)


def evaluate(
    subject: Any, opponent: Any, *, seeds: Sequence[int] = range(8), steps: int = 720
) -> dict[str, float]:
    """Score ``subject`` against ``opponent`` over ``seeds``, from the subject's seat."""
    scores = [run_episode(subject, opponent, seed=s, steps=steps)[0] for s in seeds]
    return {
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "n": len(scores),
    }


def compare(
    challenger: Any, incumbent: Any, opponent: Any, *, seeds: Sequence[int] = range(8)
) -> dict[str, float]:
    """Paired comparison of two candidates against a common opponent on identical seeds.

    Pairing matters more than sample size here. Both candidates meet the same weed spawns and the
    same shop-unlock order on a given seed, so the per-seed difference cancels that shared
    randomness; an unpaired difference of means would leave it in and need far more episodes to see
    the same effect.
    """
    deltas = []
    for seed in seeds:
        a = run_episode(challenger, opponent, seed=seed)[0]
        b = run_episode(incumbent, opponent, seed=seed)[0]
        deltas.append(a - b)
    mean = statistics.fmean(deltas)
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    stderr = stdev / (len(deltas) ** 0.5) if deltas else 0.0
    return {
        "mean_delta": mean,
        "stderr": stderr,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "n": len(deltas),
        # Two standard errors is a rough screen, not a valid sequential test: it does not survive
        # peeking, so do not stop adding seeds the moment it goes green.
        "clear": abs(mean) > 2 * stderr if stderr else False,
    }


def diagnose(subject: Callable[..., Any], opponent: Any, *, seed: int = 0) -> dict[str, Any]:
    """Run one episode and report what the farm ended up holding -- the tuning read-out."""
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([subject, opponent])
    final = env.steps[-1][0].observation
    farm = final["farms"][0]
    counts: dict[str, int] = {}
    for row in farm["tiles"]:
        for tile in row:
            if tile == "LOCKED":
                counts["locked"] = counts.get("locked", 0) + 1
            elif tile is None:
                counts["empty"] = counts.get("empty", 0) + 1
            elif tile.get("kind") == "PLANT":
                counts[tile["crop"]] = counts.get(tile["crop"], 0) + 1
            else:
                counts[tile.get("kind", "?")] = counts.get(tile.get("kind", "?"), 0) + 1
    return {
        "money": farm["money"],
        "quadrants": farm.get("unlocked_quadrants"),
        "tiles": counts,
        "shed": dict(final.get("private", {}).get("shed", {})),
    }
