"""Simulation Monte-Carlo de scorelines à partir du modèle Dixon-Coles.

On échantillonne N scores depuis la distribution catégorielle donnée par la matrice
Dixon-Coles (donc la correction τ est respectée). Les fréquences empiriques
convergent vers la matrice analytique — les deux voies sont ainsi cohérentes par
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from betting_engine.config import (
    DEFAULT_MAX_GOALS,
    DEFAULT_RHO,
    DEFAULT_SIMULATIONS,
)
from betting_engine.dixon_coles.model import score_matrix


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Résultat d'une simulation : scores tirés + matrice de fréquences empiriques."""

    home_goals: np.ndarray
    away_goals: np.ndarray
    frequency_matrix: np.ndarray
    n_simulations: int


def simulate_from_matrix(
    matrix: np.ndarray,
    n: int = DEFAULT_SIMULATIONS,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Tire `n` scorelines depuis une matrice de scores normalisée.

    Args:
        matrix: matrice `[h, a]` sommant à 1.
        n: nombre de simulations (> 0).
        rng: générateur numpy (pour la reproductibilité). Créé si absent.
    """
    if n <= 0:
        raise ValueError("Le nombre de simulations doit être strictement positif.")
    generator = rng if rng is not None else np.random.default_rng()

    size = matrix.shape[0]
    flat = np.asarray(matrix, dtype=float).ravel()
    flat = flat / flat.sum()  # garde-fou de normalisation

    draws = generator.choice(flat.size, size=n, p=flat)
    home_goals, away_goals = np.divmod(draws, size)

    freq = np.zeros_like(matrix, dtype=float)
    np.add.at(freq, (home_goals, away_goals), 1.0)
    freq /= n

    return SimulationResult(
        home_goals=home_goals,
        away_goals=away_goals,
        frequency_matrix=freq,
        n_simulations=n,
    )


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    n: int = DEFAULT_SIMULATIONS,
    max_goals: int = DEFAULT_MAX_GOALS,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Construit la matrice Dixon-Coles puis simule `n` scorelines."""
    matrix = score_matrix(lambda_home, lambda_away, rho=rho, max_goals=max_goals)
    return simulate_from_matrix(matrix, n=n, rng=rng)
