"""sigma-separation, the Markov property for cyclic SCMs (experimental; plan §11).

sigma-separation (Forré & Mooij 2017; Bongers, Forré, Peters & Mooij 2021, *Ann. Statist.*) is the
graphical independence criterion for directed graphs *with cycles*; it replaces d-separation, to
which it reduces on acyclic graphs. We compute it by **acyclification** (Forré & Mooij 2018): a
cyclic graph's sigma-separations are exactly the m-separations of an acyclic ADMG in which each
strongly connected component is a bidirected clique and external parents are lifted onto every
component member. Delegating to the shipped, well-tested
:func:`eqcert.identification._separation.d_separated` makes the acyclic-coincidence guarantee
hold *by construction* (on a DAG the acyclification is the identity).
"""

from __future__ import annotations

from collections.abc import Iterable

from eqcert.experimental.cyclic.graph import CyclicCausalGraph
from eqcert.identification._separation import d_separated


def sigma_separated(
    graph: CyclicCausalGraph,
    x: Iterable[str],
    y: Iterable[str],
    z: Iterable[str] = (),
) -> bool:
    """Is ``x`` sigma-separated from ``y`` given ``z`` in the (possibly cyclic) ``graph``?

    On an acyclic ``graph`` this returns exactly what
    :func:`~eqcert.identification._separation.d_separated` returns for the same edges -- the
    cyclic theory degrades gracefully to the shipped d-separation.
    """
    return d_separated(graph.acyclification(), set(x), set(y), set(z))
