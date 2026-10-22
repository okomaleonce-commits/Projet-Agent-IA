"""Tests du cœur Dixon-Coles (matrice de scores et correction τ)."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from betting_engine.dixon_coles.model import score_matrix, tau_matrix


def test_score_matrix_sums_to_one() -> None:
    matrix = score_matrix(1.6, 1.1, rho=-0.13)
    assert np.isclose(matrix.sum(), 1.0)


def test_score_matrix_rho_zero_is_independent_poisson() -> None:
    lam_h, lam_a, max_goals = 1.8, 1.2, 10
    matrix = score_matrix(lam_h, lam_a, rho=0.0, max_goals=max_goals)
    goals = np.arange(max_goals + 1)
    expected = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    expected /= expected.sum()
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_tau_matrix_values() -> None:
    tau = tau_matrix(1.5, 1.0, -0.1)
    assert np.isclose(tau[0, 0], 1 - 1.5 * 1.0 * -0.1)
    assert np.isclose(tau[0, 1], 1 + 1.5 * -0.1)
    assert np.isclose(tau[1, 0], 1 + 1.0 * -0.1)
    assert np.isclose(tau[1, 1], 1 - -0.1)


def test_score_matrix_clamps_nonpositive_rates() -> None:
    matrix = score_matrix(-5.0, 0.0, rho=-0.13)
    assert np.isclose(matrix.sum(), 1.0)
    assert (matrix >= 0).all()


def test_higher_home_lambda_raises_home_win_mass() -> None:
    strong_home = score_matrix(2.5, 0.8)
    weak_home = score_matrix(0.8, 2.5)
    assert np.tril(strong_home, -1).sum() > np.tril(weak_home, -1).sum()
