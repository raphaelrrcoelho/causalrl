"""Known-noise abduction: exact continuous counterfactuals via abduct/predict."""

from torch.distributions import Normal

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import LinearGaussianMechanism
from causalrl.scm.scm import ExogenousPosterior, StructuralCausalModel


def _lin_scm() -> StructuralCausalModel:
    # X = U_x ;  Y = 2*X + U_y
    g = CausalGraph(directed_edges=[("X", "Y")])
    mechs = {
        "X": LinearGaussianMechanism([], {}, bias=0.0),
        "Y": LinearGaussianMechanism(["X"], {"X": 2.0}, bias=0.0),
    }
    exo = {"X": Normal(0.0, 1.0), "Y": Normal(0.0, 1.0)}
    return StructuralCausalModel(g, mechs, exo)


def test_known_noise_abduction_is_exact():
    scm = _lin_scm()
    post = scm.abduct(known={"X": 0.5, "Y": 0.1}, n=1)
    assert isinstance(post, ExogenousPosterior)
    factual = post.predict()
    assert abs(float(factual["Y"]) - 1.1) < 1e-6  # 2*0.5 + 0.1
    cf = post.predict(do={"X": 3.0})
    assert abs(float(cf["Y"]) - 6.1) < 1e-6  # 2*3 + 0.1, same U_y


def test_abduct_predict_reused_across_interventions():
    scm = _lin_scm()
    post = scm.abduct(known={"X": 0.0, "Y": 0.0}, n=1)
    ys = [float(post.predict(do={"X": x})["Y"]) for x in (1.0, 2.0, 3.0)]
    assert ys == [2.0, 4.0, 6.0]
