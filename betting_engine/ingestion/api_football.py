"""Ingestion des résultats historiques depuis API-Football (api-sports.io).

Récupère les matchs terminés (`status=FT`) d'une ligue/saison et les convertit en
`list[Match]`, prêts pour le fit. La clé est lue dans l'environnement
(`API_FOOTBALL_KEY`) ; en session cloud, le proxy l'injecte automatiquement.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import requests

from betting_engine.config import API_FOOTBALL_BASE, api_football_key
from betting_engine.data import Match

log = logging.getLogger("betting_engine.ingestion")

_FINISHED = {"FT", "AET", "PEN"}


class ApiFootballError(RuntimeError):
    """Échec d'appel à l'API-Football (HTTP non-200, réponse illisible)."""


def _headers() -> dict[str, str]:
    return {"x-apisports-key": api_football_key()}


def fetch_results(
    league_id: int,
    season: int,
    timeout: int = 15,
    session: requests.Session | None = None,
) -> list[Match]:
    """Récupère les résultats terminés d'une ligue/saison.

    Args:
        league_id: identifiant API-Football de la ligue (cf. `config.LEAGUES`).
        season: année de saison (ex. 2025).
        timeout: délai par requête en secondes.
        session: session requests réutilisable (optionnelle, utile pour les tests).

    Returns:
        Liste de `Match` triée par date croissante.

    Raises:
        ApiFootballError: statut HTTP non-200 ou charge utile invalide.
    """
    client = session or requests.Session()
    matches: list[Match] = []
    page = 1

    while True:
        payload = _get_page(client, league_id, season, page, timeout)
        for raw in payload.get("response", []):
            match = _to_match(raw, league_id, season)
            if match is not None:
                matches.append(match)

        paging = payload.get("paging", {}) or {}
        if page >= int(paging.get("total", 1)):
            break
        page += 1

    matches.sort(key=lambda m: m.date)
    log.info("API-Football : %d matchs pour ligue %s saison %s", len(matches), league_id, season)
    return matches


def _get_page(
    client: requests.Session, league_id: int, season: int, page: int, timeout: int
) -> dict:
    # API-Football renvoie une saison entière en une page et rejette un `page`
    # superflu ; on ne l'ajoute donc qu'à partir de la page 2.
    params: dict[str, object] = {"league": league_id, "season": season, "status": "FT-AET-PEN"}
    if page > 1:
        params["page"] = page

    response = client.get(
        f"{API_FOOTBALL_BASE}/fixtures", headers=_headers(), params=params, timeout=timeout
    )
    if response.status_code != 200:
        raise ApiFootballError(
            f"HTTP {response.status_code} sur /fixtures (ligue {league_id}, saison {season})"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiFootballError(f"Réponse JSON illisible : {exc}") from exc

    errors = payload.get("errors")
    if errors:
        raise ApiFootballError(f"API-Football a renvoyé des erreurs : {errors}")
    return payload


def _to_match(raw: dict, league_id: int, season: int) -> Match | None:
    """Convertit une fixture brute en `Match`, ou None si non exploitable."""
    status = (raw.get("fixture", {}).get("status") or {}).get("short")
    goals = raw.get("goals", {}) or {}
    home_goals, away_goals = goals.get("home"), goals.get("away")
    if status not in _FINISHED or home_goals is None or away_goals is None:
        return None

    teams = raw.get("teams", {})
    return Match(
        date=_parse_api_date(raw.get("fixture", {}).get("date", "")),
        home_team=teams.get("home", {}).get("name", ""),
        away_team=teams.get("away", {}).get("name", ""),
        home_goals=int(home_goals),
        away_goals=int(away_goals),
        league_id=league_id,
        season=season,
    )


def _parse_api_date(value: str) -> date:
    """Parse une date ISO 8601 d'API-Football (ex. `2026-09-21T15:00:00+00:00`)."""
    return datetime.fromisoformat(value).date()
