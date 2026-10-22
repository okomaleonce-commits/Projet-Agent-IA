"""Tests de la simulation Monte-Carlo (cohérence avec l'analytique)."""

from __future__ import annotations

import numpy as np
import pytest

from betting_engine.dixon_coles.model import score_matrix
from betting_engine.markets import one_x_two
from betting_engine.simulation import simulate_from_matrix, simulate_match


def test_mc_frequencies_converge_to_analytic() -> None:
    matrix = score_matrix(1.6, 1.2, rho=-0.13)
    rng = np.random.default_rng(42)
    result = simulate_from_matrix(matrix, n=200_000, rng=rng)
    # Les fréquences MC doivent approcher la matrice analytique.
    assert np.max(np.abs(result.frequency_matrix - matrix)) < 0.01


def test_mc_markets_match_analytic() -> None:
    matrix = score_matrix(1.9, 0.9, rho=-0.13)
    rng = np.random.default_rng(7)
    result = simulate_from_matrix(matrix, n=200_000, rng=rng)
    analytic = one_x_two(matrix)
    empirical = one_x_two(result.frequency_matrix)
    assert abs(analytic["home"] - empirical["home"]) < 0.01


def test_simulate_match_shapes_and_seed() -> None:
    a = simulate_match(1.5, 1.0, n=5000, rng=np.random.default_rng(1))
    b = simulate_match(1.5, 1.0, n=5000, rng=np.random.default_rng(1))
    assert a.home_goals.shape == (5000,)
    np.testing.assert_array_equal(a.home_goals, b.home_goals)  # reproductible


def test_simulate_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        simulate_match(1.5, 1.0, n=0)
