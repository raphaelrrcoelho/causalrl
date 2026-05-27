"""Emit reproducible structural-bandit benchmark reports as JSON."""

from __future__ import annotations

import argparse
import json

from causalrl.eval.benchmark import (
    report_to_dict,
    run_confounded_chain_benchmark,
    run_frontdoor_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("confounded-chain", "frontdoor"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tail-window", type=int, default=None)
    parser.add_argument("--n-mc", type=int, default=None)
    args = parser.parse_args()

    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    kwargs: dict[str, object] = {"seeds": seeds}
    if args.steps is not None:
        kwargs["n_steps"] = args.steps
    if args.tail_window is not None:
        kwargs["tail_window"] = args.tail_window
    if args.n_mc is not None:
        kwargs["n_mc"] = args.n_mc
    report = (
        run_confounded_chain_benchmark(**kwargs)
        if args.benchmark == "confounded-chain"
        else run_frontdoor_benchmark(**kwargs)
    )
    print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
