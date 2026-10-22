"""Configuration centrale : endpoint API-Football, défauts du modèle, registre de ligues.

Le registre `LEAGUES` est repris du bot APEX OMEGA (réutilisation d'une base éprouvée) :
chaque ligue porte son `rho` (correction Dixon-Coles) et son `home_adv` calibrés.
"""

from __future__ import annotations

import os
from typing import Final

# --- API-Football (api-sports.io) ------------------------------------------
# La clé est lue dans l'environnement ; dans la session cloud le proxy l'injecte.
API_FOOTBALL_BASE: Final[str] = "https://v3.football.api-sports.io"


def api_football_key() -> str:
    """Retourne la clé API-Football depuis l'environnement (vide si absente)."""
    return os.getenv("API_FOOTBALL_KEY", "")


# --- Défauts du modèle Dixon-Coles -----------------------------------------
DEFAULT_RHO: Final[float] = -0.13
DEFAULT_HOME_ADV: Final[float] = 1.10
DEFAULT_MAX_GOALS: Final[int] = 10
DEFAULT_SIMULATIONS: Final[int] = 50_000

# --- Registre de ligues (repris d'APEX OMEGA) ------------------------------
# id API-Football -> métadonnées. `rho`/`home_adv` sont des valeurs calibrées.
LEAGUES: Final[dict[int, dict]] = {
    2: {"name": "UEFA Champions League", "rho": -0.13, "home_adv": 1.08},
    3: {"name": "UEFA Europa League", "rho": -0.13, "home_adv": 1.08},
    848: {"name": "UEFA Conference League", "rho": -0.13, "home_adv": 1.08},
    39: {"name": "English Premier League", "rho": -0.13, "home_adv": 1.10},
    140: {"name": "La Liga", "rho": -0.13, "home_adv": 1.12},
    78: {"name": "Bundesliga", "rho": -0.13, "home_adv": 1.11},
    135: {"name": "Serie A", "rho": -0.13, "home_adv": 1.13},
    61: {"name": "Ligue 1", "rho": -0.13, "home_adv": 1.12},
    94: {"name": "Primeira Liga (POR)", "rho": -0.14, "home_adv": 1.14},
    88: {"name": "Eredivisie (NED)", "rho": -0.13, "home_adv": 1.11},
    144: {"name": "Jupiler Pro League", "rho": -0.13, "home_adv": 1.10},
    169: {"name": "Eliteserien (NOR)", "rho": -0.14, "home_adv": 1.10},
    207: {"name": "Super Lig (TUR)", "rho": -0.15, "home_adv": 1.15},
    119: {"name": "Superliga (DEN)", "rho": -0.13, "home_adv": 1.11},
}


def league_defaults(league_id: int | None) -> dict:
    """Retourne les défauts (`rho`, `home_adv`, `name`) d'une ligue, ou les défauts génériques."""
    if league_id is not None and league_id in LEAGUES:
        return LEAGUES[league_id]
    return {"name": "générique", "rho": DEFAULT_RHO, "home_adv": DEFAULT_HOME_ADV}
