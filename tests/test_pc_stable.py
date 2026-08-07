"""PC-stable: the skeleton must not depend on the order the variables arrive in."""

from __future__ import annotations

import numpy as np

from causalrl.discovery import _pc_skeleton, conditional_mutual_information, discover


def _collider_data(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    """A -> C <- B with D downstream of C, so several levels of the search do real work.

    C is a noisy OR rather than a modular sum: ``(A + B) % k`` with uniform parents is independent
    of each parent marginally, which is a faithfulness violation and would make the collider
    genuinely undetectable -- a property of that generator, not of the algorithm under test.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    c = ((a | b) ^ (rng.random(n) < 0.05)).astype(int)
    d = (c ^ (rng.random(n) < 0.1)).astype(int)
    return {"A": a, "B": b, "C": c, "D": d}


def test_the_skeleton_is_independent_of_variable_order() -> None:
    """The property PC-stable exists for. In-place adjacency updates break exactly this."""
    data = _collider_data()
    orders = [
        ["A", "B", "C", "D"],
        ["D", "C", "B", "A"],
        ["C", "A", "D", "B"],
        ["B", "D", "A", "C"],
    ]
    skeletons = set()
    for order in orders:
        adj, _ = _pc_skeleton(data, order, threshold=0.01, max_conditioning_size=3)
        skeletons.add(frozenset(frozenset((a, b)) for a in adj for b in adj[a]))
    assert len(skeletons) == 1, f"skeleton depends on variable order: {len(skeletons)} distinct"


def test_discover_is_independent_of_variable_order() -> None:
    """The property has to survive all the way through orientation, not just the skeleton."""
    data = _collider_data(seed=1)
    edges = set()
    for order in (["A", "B", "C", "D"], ["D", "C", "B", "A"], ["C", "B", "D", "A"]):
        cpdag = discover(data, order, threshold=0.01, max_conditioning_size=3)
        edges.add((cpdag.directed_edges, cpdag.undirected_edges))
    assert len(edges) == 1


def test_the_collider_is_still_found() -> None:
    """Stability must not cost correctness: A -> C <- B is the structure in the data."""
    cpdag = discover(_collider_data(seed=2), ["A", "B", "C", "D"], threshold=0.01)
    assert ("A", "C") in cpdag.directed_edges
    assert ("B", "C") in cpdag.directed_edges


def test_conditional_mutual_information_matches_its_definition() -> None:
    """The vectorised counting must equal the textbook sum, not merely rank the same."""
    rng = np.random.default_rng(3)
    n = 500
    z = rng.integers(0, 3, size=n)
    data = {
        "X": (z + rng.integers(0, 2, size=n)) % 3,
        "Y": (z + rng.integers(0, 2, size=n)) % 3,
        "Z": z,
    }

    def by_definition(x: str, y: str, cond: list[str]) -> float:
        total = 0.0
        cols = [data[c] for c in cond]
        keys = [tuple(int(c[i]) for c in cols) for i in range(n)]
        for zk in set(keys):
            rows = [i for i in range(n) if keys[i] == zk]
            p_z = len(rows) / n
            for xv in set(int(data[x][i]) for i in rows):
                for yv in set(int(data[y][i]) for i in rows):
                    n_xy = sum(1 for i in rows if data[x][i] == xv and data[y][i] == yv)
                    if n_xy == 0:
                        continue
                    n_x = sum(1 for i in rows if data[x][i] == xv)
                    n_y = sum(1 for i in rows if data[y][i] == yv)
                    p_xy = n_xy / len(rows)
                    total += p_z * p_xy * np.log(p_xy / ((n_x / len(rows)) * (n_y / len(rows))))
        return max(total, 0.0)

    for cond in ([], ["Z"]):
        got = conditional_mutual_information(data, "X", "Y", cond)
        assert abs(got - by_definition("X", "Y", cond)) < 1e-10

    # X and Y are dependent marginally but independent given Z, which is the whole point.
    assert conditional_mutual_information(data, "X", "Y", []) > 0.05
    assert conditional_mutual_information(data, "X", "Y", ["Z"]) < 0.02


def test_it_handles_a_single_level_column_and_an_empty_log() -> None:
    """Degenerate inputs must return 0, not divide by a zero count."""
    constant = {"X": np.zeros(50, dtype=int), "Y": np.arange(50) % 3}
    assert conditional_mutual_information(constant, "X", "Y", []) == 0.0
    empty = {"X": np.array([], dtype=int), "Y": np.array([], dtype=int)}
    assert conditional_mutual_information(empty, "X", "Y", []) == 0.0
