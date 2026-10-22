"""Estimation par maximum de vraisemblance des paramètres Dixon-Coles.

On ajuste, par match, la log-vraisemblance d'un Poisson bivarié avec correction τ,
pondérée par une décroissance temporelle `exp(-ξ·Δt_jours)`. Les forces d'attaque
sont contraintes (moyenne nulle) via une pénalité, pour lever la non-identifiabilité.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from betting_engine.config import DEFAULT_RHO
from betting_engine.data import DataError, Match
from betting_engine.dixon_coles.strengths import TeamStrengths

log = logging.getLogger("betting_engine.fit")

_ATTACK_MEAN_PENALTY = 1e4
_TINY = 1e-10


@dataclass(frozen=True, slots=True)
class FitResult:
    """Résultat du fit : forces estimées + diagnostics."""

    strengths: TeamStrengths
    log_likelihood: float
    n_matches: int
    n_teams: int
    converged: bool


def fit_dixon_coles(
    matches: list[Match],
    xi: float = 0.0,
    rho_bounds: tuple[float, float] = (-0.25, 0.25),
    max_iter: int = 500,
) -> FitResult:
    """Ajuste le modèle Dixon-Coles sur des résultats historiques.

    Args:
        matches: résultats passés (au moins deux équipes et un match).
        xi: taux de décroissance temporelle (0 = poids uniforme).
        rho_bounds: bornes de ρ (garde τ positif).
        max_iter: itérations max de l'optimiseur.

    Returns:
        `FitResult` avec les `TeamStrengths` estimées.

    Raises:
        DataError: moins de deux équipes ou aucun match.
    """
    if not matches:
        raise DataError("Impossible de fitter sur une liste de matchs vide.")

    teams = sorted({t for m in matches for t in (m.home_team, m.away_team)})
    if len(teams) < 2:
        raise DataError("Le fit requiert au moins deux équipes distinctes.")

    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = np.array([index[m.home_team] for m in matches])
    away_idx = np.array([index[m.away_team] for m in matches])
    home_goals = np.array([m.home_goals for m in matches], dtype=float)
    away_goals = np.array([m.away_goals for m in matches], dtype=float)
    weights = _time_weights(matches, xi)

    gln_home = gammaln(home_goals + 1.0)
    gln_away = gammaln(away_goals + 1.0)
    mean_goals = max((home_goals.sum() + away_goals.sum()) / (2 * len(matches)), _TINY)

    def negative_log_likelihood(params: np.ndarray) -> float:
        attack = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        home_adv = params[2 * n_teams]
        rho = params[2 * n_teams + 1]

        log_lh = home_adv + attack[home_idx] + defense[away_idx]
        log_la = attack[away_idx] + defense[home_idx]
        lam_h = np.exp(log_lh)
        lam_a = np.exp(log_la)

        ll = (home_goals * log_lh - lam_h - gln_home) + (
            away_goals * log_la - lam_a - gln_away
        )
        ll += np.log(np.clip(_tau(home_goals, away_goals, lam_h, lam_a, rho), _TINY, None))

        penalty = _ATTACK_MEAN_PENALTY * attack.mean() ** 2
        return -float((weights * ll).sum()) + penalty

    x0 = np.concatenate(
        [np.zeros(n_teams), np.full(n_teams, np.log(mean_goals)), [0.1], [DEFAULT_RHO]]
    )
    bounds = [(None, None)] * (2 * n_teams) + [(None, None), rho_bounds]

    result = minimize(
        negative_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter},
    )
    if not result.success:
        log.warning("Convergence MLE incomplète : %s", result.message)

    attack = result.x[:n_teams] - result.x[:n_teams].mean()  # recentre sum(attack)=0
    defense = result.x[n_teams : 2 * n_teams]
    strengths = TeamStrengths(
        attack={t: float(attack[i]) for t, i in index.items()},
        defense={t: float(defense[i]) for t, i in index.items()},
        home_adv=float(result.x[2 * n_teams]),
        rho=float(result.x[2 * n_teams + 1]),
        teams=tuple(teams),
    )
    return FitResult(
        strengths=strengths,
        log_likelihood=-float(result.fun),
        n_matches=len(matches),
        n_teams=n_teams,
        converged=bool(result.success),
    )


def _time_weights(matches: list[Match], xi: float) -> np.ndarray:
    """Poids `exp(-ξ·Δt)` relatifs au match le plus récent (uniformes si ξ=0)."""
    if xi <= 0:
        return np.ones(len(matches))
    reference = max(m.date for m in matches)
    deltas = np.array([(reference - m.date).days for m in matches], dtype=float)
    return np.exp(-xi * deltas)


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Correction τ vectorisée sur les quatre cellules basses."""
    tau = np.ones_like(home_goals)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m11] = 1.0 - rho
    return tau
