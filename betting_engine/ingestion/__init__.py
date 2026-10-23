"""Sous-package d'ingestion de données (résultats et cotes API-Football)."""

from __future__ import annotations

from betting_engine.ingestion.api_football import ApiFootballError, fetch_results
from betting_engine.ingestion.odds import (
    FixtureOdds,
    Odds1x2,
    OddsError,
    fetch_odds_1x2,
    fetch_upcoming,
    value_bets,
)

__all__ = [
    "ApiFootballError",
    "FixtureOdds",
    "Odds1x2",
    "OddsError",
    "fetch_odds_1x2",
    "fetch_results",
    "fetch_upcoming",
    "value_bets",
]
