"""Sous-package Dixon-Coles : matrice de scores, forces d'équipe, fit MLE."""

from __future__ import annotations

from betting_engine.dixon_coles.fit import FitResult, fit_dixon_coles
from betting_engine.dixon_coles.model import score_matrix, tau_matrix
from betting_engine.dixon_coles.strengths import TeamStrengths, UnknownTeamError

__all__ = [
    "FitResult",
    "TeamStrengths",
    "UnknownTeamError",
    "fit_dixon_coles",
    "score_matrix",
    "tau_matrix",
]
