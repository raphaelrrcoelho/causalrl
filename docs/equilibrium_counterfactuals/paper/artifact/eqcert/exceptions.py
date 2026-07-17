"""Typed exceptions. Causal misuse fails loudly, never silently."""


class EqcertError(Exception):
    """Base class for all eqcert errors."""


class CausalGraphError(EqcertError):
    """Invalid graph operation (unknown node, cycle, malformed edge)."""


class NotIdentifiableError(EqcertError):
    """A causal query is not identifiable from the available data."""

    def __init__(self, message: str, witness: object | None = None) -> None:
        super().__init__(message)
        self.witness = witness


class RealizabilityError(EqcertError):
    """A counterfactual query cannot be realized from the given evidence."""


class UnverifiedAssumptionError(EqcertError):
    """A method's claimed guarantee requires an assumption the caller has not declared."""


class CausalInterfaceUnavailableError(EqcertError):
    """The causal interface is not available on this wrapper.

    Raised when a method that requires a live SCM and a named reward node is called
    on a :class:`~eqcert.envs.wrapper.CausalEnvWrapper` that was constructed without
    them (e.g. wrapping a :class:`~eqcert.envs.base.ConfoundedMDP` that carries
    ``scm=None``, or without passing a ``reward_node``).
    """
