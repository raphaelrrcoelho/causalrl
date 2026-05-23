import torch

from causalrl.scm.mechanisms import FunctionalMechanism, LinearGaussianMechanism


def test_functional_mechanism_uses_parents_and_noise():
    m = FunctionalMechanism(parents=["A", "B"], fn=lambda pa, u: pa["A"] + pa["B"] + u)
    out = m(
        parent_values={"A": torch.tensor([1.0]), "B": torch.tensor([2.0])},
        noise=torch.tensor([0.5]),
    )
    assert torch.allclose(out, torch.tensor([3.5]))


def test_functional_mechanism_root_ignores_parents():
    m = FunctionalMechanism(parents=[], fn=lambda pa, u: u)
    out = m(parent_values={}, noise=torch.tensor([7.0]))
    assert torch.allclose(out, torch.tensor([7.0]))


def test_linear_gaussian_mechanism():
    m = LinearGaussianMechanism(parents=["X"], weights={"X": 2.0}, bias=1.0)
    out = m(parent_values={"X": torch.tensor([3.0])}, noise=torch.tensor([0.0]))
    assert torch.allclose(out, torch.tensor([7.0]))  # 2*3 + 1 + 0
