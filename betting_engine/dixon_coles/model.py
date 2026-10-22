"""Cœur Dixon-Coles : correction τ et construction de la matrice de scores.

Modèle de Poisson bivarié avec correction des petits scores (Dixon & Coles, 1997).
Porté et vectorisé (numpy) depuis APEX OMEGA `models/dixon_coles.py`.

Convention : λ_home, λ_away sont les taux de buts attendus (xG) domicile/extérieur.
τ(h, a) :
    (0,0) → 1 − λ_h·λ_a·ρ      (0,1) → 1 + λ_h·ρ
    (1,0) → 1 + λ_a·ρ          (1,1) → 1 − ρ
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from betting_engine.config import DEFAULT_MAX_GOALS, DEFAULT_RHO

_MIN_RATE = 1e-6


def tau_matrix(lambda_home: float, lambda_away: float, rho: float) -> np.ndarray:
    """Matrice 2×2 des facteurs de correction τ pour les cellules (0/1, 0/1)."""
    return np.array(
        [
            [1.0 - lambda_home * lambda_away * rho, 1.0 + lambda_home * rho],
            [1.0 + lambda_away * rho, 1.0 - rho],
        ]
    )


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Construit la matrice P(buts_dom=h, buts_ext=a), indexée `[h, a]`.

    Produit externe de deux Poisson indépendants, puis correction τ sur les
    quatre cellules basses, puis normalisation à 1.

    Args:
        lambda_home: xG domicile (> 0).
        lambda_away: xG extérieur (> 0).
        rho: paramètre de correction Dixon-Coles.
        max_goals: nombre de buts max modélisé par camp (matrice (max+1)²).

    Returns:
        Matrice numpy (max_goals+1, max_goals+1) sommant à 1.
    """
    lam_h = max(float(lambda_home), _MIN_RATE)
    lam_a = max(float(lambda_away), _MIN_RATE)
    goals = np.arange(max_goals + 1)

    pmf_home = poisson.pmf(goals, lam_h)
    pmf_away = poisson.pmf(goals, lam_a)
    matrix = np.outer(pmf_home, pmf_away)

    matrix[:2, :2] *= tau_matrix(lam_h, lam_a, rho)
    matrix = np.clip(matrix, 0.0, None)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix
