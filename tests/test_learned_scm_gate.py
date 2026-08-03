from causalrl.eval.learned_scm_gate import run_learned_scm_oracle_gate


def test_learned_scm_beats_an_L1_equivalent_wrong_structure_on_do_queries():
    result = run_learned_scm_oracle_gate(seeds=(0, 1, 2), n=8000)
    assert result.causal_error < 0.03
    assert result.correlational_error > 0.15
    assert result.gap > 0.1
    assert result.passed is True
