from __future__ import annotations

import hashlib
import json

from causalrl.scm.graph import CausalGraph


def graph_hash(graph: CausalGraph) -> str:
    """A stable, canonical SHA-256 hex digest of an ADMG's structure.

    Order-independent (node and edge lists are sorted before hashing), direction-sensitive for
    directed edges, and symmetric for bidirected (latent-confounding) edges. Isolated nodes are
    part of the structure and change the digest. Used to fingerprint the graph a certificate was
    computed against (provenance; invariant I8).
    """
    nodes = sorted(str(n) for n in graph.nodes)
    directed = sorted((str(u), str(v)) for u, v in graph.directed_edges)
    bidirected = sorted(tuple(sorted((str(u), str(v)))) for u, v in graph.bidirected_edges)
    canonical = json.dumps(
        {"nodes": nodes, "directed": directed, "bidirected": bidirected},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
