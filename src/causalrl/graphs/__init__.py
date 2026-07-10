"""Graph utilities home (plan §4 ``graphs/``).

A leaf package. Phase 0 provides certificate-provenance hashing plus a re-export of the shipped
:class:`~causalrl.scm.graph.CausalGraph`. Migrating the graph *types* into this package is a later
minor; for now this is the hashing home, added as a re-export shim rather than a code move (I9).
"""

from __future__ import annotations

from causalrl.graphs.hashing import graph_hash
from causalrl.scm.graph import CausalGraph

__all__ = ["CausalGraph", "graph_hash"]
