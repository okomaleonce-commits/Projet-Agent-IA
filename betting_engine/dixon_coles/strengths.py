"""Forces d'équipe estimées et calcul des xG d'un match.

Modèle log-linéaire :
    λ_home = exp(home_adv + attack[dom] + defense[ext])
    λ_away = exp(           attack[ext] + defense[dom])
`home_adv` et les forces sont exprimés en log (c'est ce que produit le fit MLE).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class UnknownTeamError(KeyError):
    """Équipe absente des forces estimées (jamais vue à l'entraînement)."""


@dataclass(frozen=True, slots=True)
class TeamStrengths:
    """Forces d'attaque/défense (en log) par équipe, avantage domicile et ρ."""

    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    teams: tuple[str, ...] = field(default=())

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Retourne (λ_home, λ_away) pour une affiche donnée."""
        for team in (home_team, away_team):
            if team not in self.attack:
                raise UnknownTeamError(f"Équipe inconnue du modèle : {team!r}")
        lam_home = float(
            np.exp(self.home_adv + self.attack[home_team] + self.defense[away_team])
        )
        lam_away = float(np.exp(self.attack[away_team] + self.defense[home_team]))
        return lam_home, lam_away
