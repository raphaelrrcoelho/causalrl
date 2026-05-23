import torch

from causalrl.scm.mechanisms import (
    FunctionalMechanism,
    LinearGaussianMechanism,
    NeuralMechanism,
)


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


def test_neural_mechanism_forwards_parents_and_noise():
    # A linear net over [parent, noise] with fixed weights/bias gives a known output.
    net = torch.nn.Linear(2, 1)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[2.0, 3.0]]))  # 2*parent + 3*noise
        net.bias.copy_(torch.tensor([1.0]))
    m = NeuralMechanism(parents=["X"], net=net)
    out = m(parent_values={"X": torch.tensor([4.0])}, noise=torch.tensor([5.0]))
    assert out.shape == (1,)
    assert torch.allclose(out, torch.tensor([24.0]))  # 2*4 + 3*5 + 1 = 24
