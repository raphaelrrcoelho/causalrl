"""Independent optimality certificates for the E6 LP claims via weak duality.

For min c.mu s.t. G mu <= 0, 1.mu = 1, mu >= 0 with optimum v and duals y = -dual_ub >= 0:
weak duality certificate: (G^T y)_j >= c_j - v for all j  (then no feasible mu beats v).
This checks the simplex's answers WITHOUT trusting its internals.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/rapha/Documents/Code/causalrl/experiments/eqcf")

import numpy as np

import common
from e6_collusion import payoff_matrices
from causalrl.magames import cce_polytope
from causalrl.magames._lp import solve_lp

u1, u2 = payoff_matrices()
game = common.bimatrix_game(u1, u2, names=("F1", "F2"))
poly = cce_polytope(game)
G = poly.deviation_gains
n = len(poly.profiles)
welfare = np.array([u1[p] + u2[p] for p in poly.profiles])
profit1 = np.array([u1[p] for p in poly.profiles])

a_eq, b_eq = np.ones((1, n)), np.array([1.0])


def certify(c, label):
    res = solve_lp(c, a_ub=G, b_ub=np.zeros(G.shape[0]), a_eq=a_eq, b_eq=b_eq)
    assert res.status == "optimal", res.status
    v = res.value
    # primal feasibility
    mu = res.x
    assert np.all(mu >= -1e-9) and abs(mu.sum() - 1) < 1e-9, "primal infeasible"
    assert np.all(G @ mu <= 1e-8), f"primal violates CCE constraints: {np.max(G @ mu)}"
    assert abs(c @ mu - v) < 1e-8
    # dual certificate: y >= 0 and (G^T y)_j >= v - c_j for all j
    # (Lagrangian: inf_mu>=0 of sum_j mu_j (c_j + (G^T y)_j - lambda) finite iff each term >= 0)
    y = -res.dual_ub
    ok_sign = np.all(y >= -1e-9)
    slack = G.T @ np.clip(y, 0, None) - (v - np.asarray(c))
    ok_dual = np.all(slack >= -1e-7)
    print(f"{label}: value={v:+.6f}  y>=0: {ok_sign}  dual-feasible: {ok_dual} "
          f"(min slack {slack.min():.2e})")
    return v, ok_sign and ok_dual


results = []
v, ok = certify(welfare, "min welfare  "); results.append(ok); lo_w = v
v, ok = certify(-welfare, "max welfare  "); results.append(ok); hi_w = -v
v, ok = certify(profit1, "min profit1  "); results.append(ok); lo_p = v
v, ok = certify(-profit1, "max profit1  "); results.append(ok); hi_p = -v

print(f"\nwelfare CCE interval: [{lo_w:.6f}, {hi_w:.6f}]  (claimed degenerate {{32}})")
print(f"profit1 CCE interval: [{lo_p:.6f}, {hi_p:.6f}]  (claimed [12, 20])")
print(f"\nall four LP answers duality-certified: {all(results)}")

# Structural explanation check: the grid creates weak-tie Nash at (5,3)/(3,5) besides (4,4);
# all lie on the total-quantity-8 anti-diagonal where welfare = (12-8)*8 = 32.
from causalrl.games import pure_nash_equilibria
nash = pure_nash_equilibria(game)
print(f"pure Nash profiles (grid indices): {[tuple(p.values()) for p in nash]}")
print("quantities:", [(2 + p['F1'], 2 + p['F2']) for p in nash])
