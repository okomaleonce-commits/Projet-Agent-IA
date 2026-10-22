"""Fonctions de haut niveau : pricing d'un match depuis des xG ou des forces estimées."""

from __future__ import annotations

from betting_engine.config import DEFAULT_MAX_GOALS, DEFAULT_RHO
from betting_engine.dixon_coles.model import score_matrix
from betting_engine.dixon_coles.strengths import TeamStrengths
from betting_engine.markets import price


def price_match(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict:
    """Price l'ensemble des marchés à partir d'xG domicile/extérieur."""
    matrix = score_matrix(lambda_home, lambda_away, rho=rho, max_goals=max_goals)
    result = price(matrix)
    result["inputs"] = {"lambda_home": lambda_home, "lambda_away": lambda_away, "rho": rho}
    return result


def price_fixture(
    strengths: TeamStrengths,
    home_team: str,
    away_team: str,
    rho: float | None = None,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict:
    """Price une affiche à partir des forces estimées (fit MLE)."""
    lambda_home, lambda_away = strengths.expected_goals(home_team, away_team)
    effective_rho = strengths.rho if rho is None else rho
    return price_match(lambda_home, lambda_away, rho=effective_rho, max_goals=max_goals)
