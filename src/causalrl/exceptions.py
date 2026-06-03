"""Typed exceptions. Causal misuse fails loudly, never silently."""


class CausalRLError(Exception):
    """Base class for all causalrl errors."""


class CausalGraphError(CausalRLError):
    """Invalid graph operation (unknown node, cycle, malformed edge)."""


class NotIdentifiableError(CausalRLError):
    """A causal query is not identifiable from the available data."""

    def __init__(self, message: str, witness: object | None = None) -> None:
        super().__init__(message)
        self.witness = witness


class RealizabilityError(CausalRLError):
    """A counterfactual query cannot be realized from the given evidence."""


class UnverifiedAssumptionError(CausalRLError):
    """A method's claimed guarantee requires an assumption the caller has not declared."""


class CausalInterfaceUnavailableError(CausalRLError):
    """The causal interface is not available on this wrapper.

    Raised when a method that requires a live SCM and a named reward node is called
    on a :class:`~causalrl.envs.wrapper.CausalEnvWrapper` that was constructed without
    them (e.g. wrapping a :class:`~causalrl.envs.base.ConfoundedMDP` that carries
    ``scm=None``, or without passing a ``reward_node``).
    """
