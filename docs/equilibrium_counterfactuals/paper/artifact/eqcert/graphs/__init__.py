"""Graph utilities home (plan §4 ``graphs/``).

A leaf package. Phase 0 provides certificate-provenance hashing plus a re-export of the shipped
:class:`~eqcert.scm.graph.CausalGraph`. Migrating the graph *types* into this package is a later
minor; for now this is the hashing home, added as a re-export shim rather than a code move (I9).
"""

from __future__ import annotations

from eqcert.graphs.hashing import graph_hash
from eqcert.scm.graph import CausalGraph

__all__ = ["CausalGraph", "graph_hash"]
