"""Ingestion des cotes 1X2 depuis API-Football et calcul de value (edge + Kelly).

Les cotes ne sont disponibles que pour les matchs proches (api-sports les purge
~7 jours après le coup d'envoi). Ce module sert donc au pricing des matchs À VENIR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from betting_engine.config import API_FOOTBALL_BASE, api_football_key
from betting_engine.markets import edge, kelly_fraction

log = logging.getLogger("betting_engine.odds")

_BET_MATCH_WINNER = 1  # id du marché "Match Winner" (1X2) côté API-Football
_VALUE_TO_OUTCOME = {"Home": "home", "Draw": "draw", "Away": "away"}


class OddsError(RuntimeError):
    """Échec de récupération ou d'analyse des cotes."""


@dataclass(frozen=True, slots=True)
class Odds1x2:
    """Cotes décimales 1X2."""

    home: float
    draw: float
    away: float

    def implied_probabilities(self, remove_overround: bool = True) -> dict[str, float]:
        """Probabilités implicites ; `remove_overround` normalise pour retirer la marge."""
        raw = {"home": 1.0 / self.home, "draw": 1.0 / self.draw, "away": 1.0 / self.away}
        if not remove_overround:
            return raw
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}


@dataclass(frozen=True, slots=True)
class FixtureOdds:
    """Cotes agrégées d'un match : meilleure cote et moyenne par issue."""

    fixture_id: int
    best: Odds1x2
    average: Odds1x2
    n_books: int


def _headers() -> dict[str, str]:
    return {"x-apisports-key": api_football_key()}


def fetch_odds_1x2(
    fixture_id: int, timeout: int = 15, session: requests.Session | None = None
) -> FixtureOdds | None:
    """Récupère et agrège les cotes 1X2 d'un match. None si aucune cote publiée."""
    client = session or requests.Session()
    response = client.get(
        f"{API_FOOTBALL_BASE}/odds",
        headers=_headers(),
        params={"fixture": fixture_id, "bet": _BET_MATCH_WINNER},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OddsError(f"HTTP {response.status_code} sur /odds (fixture {fixture_id})")
    payload = response.json()
    rows = payload.get("response", [])
    if not rows:
        return None
    return _aggregate(fixture_id, rows[0].get("bookmakers", []))


def _aggregate(fixture_id: int, bookmakers: list[dict]) -> FixtureOdds | None:
    collected: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
    for book in bookmakers:
        bets = book.get("bets", [])
        if not bets:
            continue
        for value in bets[0].get("values", []):
            outcome = _VALUE_TO_OUTCOME.get(value.get("value"))
            if outcome is not None:
                collected[outcome].append(float(value["odd"]))
    if not all(collected.values()):
        return None
    best = Odds1x2(*(max(collected[k]) for k in ("home", "draw", "away")))
    average = Odds1x2(*(sum(collected[k]) / len(collected[k]) for k in ("home", "draw", "away")))
    return FixtureOdds(fixture_id, best=best, average=average, n_books=len(bookmakers))


def value_bets(
    model_1x2: dict[str, float], odds: Odds1x2, min_edge: float = 0.0
) -> list[dict]:
    """Liste des issues à valeur positive avec edge et fraction de Kelly.

    Args:
        model_1x2: probabilités du modèle {"home", "draw", "away"}.
        odds: cotes décimales (souvent la meilleure cote marché).
        min_edge: edge minimal pour retenir une issue.

    Returns:
        Liste triée par edge décroissant de dicts {outcome, model_prob, odd, edge, kelly}.
    """
    odds_map = {"home": odds.home, "draw": odds.draw, "away": odds.away}
    found = []
    for outcome, prob in model_1x2.items():
        if outcome not in odds_map:
            continue
        value_edge = edge(prob, odds_map[outcome])
        if value_edge is not None and value_edge > min_edge:
            found.append(
                {
                    "outcome": outcome,
                    "model_prob": round(prob, 4),
                    "odd": odds_map[outcome],
                    "edge": value_edge,
                    "kelly": kelly_fraction(prob, odds_map[outcome]),
                }
            )
    found.sort(key=lambda row: row["edge"], reverse=True)
    return found


def fetch_upcoming(
    league_id: int, season: int, n: int = 10, session: requests.Session | None = None,
    timeout: int = 15,
) -> list[dict]:
    """Récupère les `n` prochains matchs d'une ligue (id + noms d'équipes)."""
    client = session or requests.Session()
    response = client.get(
        f"{API_FOOTBALL_BASE}/fixtures",
        headers=_headers(),
        params={"league": league_id, "season": season, "next": n},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OddsError(f"HTTP {response.status_code} sur /fixtures (ligue {league_id})")
    fixtures = []
    for raw in response.json().get("response", []):
        teams = raw.get("teams", {})
        fixtures.append(
            {
                "fixture_id": raw["fixture"]["id"],
                "date": raw["fixture"]["date"][:10],
                "home_team": teams.get("home", {}).get("name", ""),
                "away_team": teams.get("away", {}).get("name", ""),
            }
        )
    return fixtures
