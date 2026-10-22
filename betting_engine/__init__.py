"""betting_engine — moteur de pricing football Dixon-Coles.

Deux voies cohérentes de pricing : analytique (matrice de scores) et Monte-Carlo.
Alimenté soit par des xG directs, soit par un fit MLE des forces d'équipe sur
l'historique (CSV ou API-Football).
"""

from __future__ import annotations

from betting_engine.data import DataError, Match, load_matches_csv
from betting_engine.dixon_coles.fit import FitResult, fit_dixon_coles
from betting_engine.dixon_coles.model import score_matrix
from betting_engine.dixon_coles.strengths import TeamStrengths, UnknownTeamError
from betting_engine.engine import price_fixture, price_match
from betting_engine.markets import (
    asian_handicap,
    btts,
    double_chance,
    edge,
    kelly_fraction,
    one_x_two,
    over_under,
    price,
)
from betting_engine.simulation import (
    SimulationResult,
    simulate_from_matrix,
    simulate_match,
)

__version__ = "0.1.0"

__all__ = [
    "DataError",
    "FitResult",
    "Match",
    "SimulationResult",
    "TeamStrengths",
    "UnknownTeamError",
    "asian_handicap",
    "btts",
    "double_chance",
    "edge",
    "fit_dixon_coles",
    "kelly_fraction",
    "load_matches_csv",
    "one_x_two",
    "over_under",
    "price",
    "price_fixture",
    "price_match",
    "score_matrix",
    "simulate_from_matrix",
    "simulate_match",
]
