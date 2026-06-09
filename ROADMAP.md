# Roadmap

`causalrl` follows [semantic versioning](https://semver.org) from 1.0. This roadmap states
intentions, not promises; the library's honesty-about-scope rule applies here too.

## Now (1.0.x)
- Stabilize the 1.0 public API; fix bugs; sharpen docstrings and the [Tour](https://raphaelrrcoelho.github.io/causalrl/tour/) examples.
- A runnable example in [`examples/`](examples) for every taxonomy task that still lacks one.
- `certify_decision` ergonomics driven by real use (arm labels, pretty-printing).

## Next
- Wider sensitivity coverage in the decision layer (additional MSM kernels; clearer
  abstain/ship reporting).
- More Gymnasium-native surface: documented interventional-rollout recipes and a CGFA-PPO worked example.
- Tighter interop docs with DoWhy / EconML (shared estimands; when to use which).

## Later
- A scale path for confounded offline RL with [`d3rlpy`](https://github.com/takuseno/d3rlpy) as the
  backbone — `causalrl` supplies the causal layer, not a new trainer.
- Community-requested taxonomy slices and benchmarks.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the `good first issue` label.
