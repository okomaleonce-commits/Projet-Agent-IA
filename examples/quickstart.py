"""Exemple de bout en bout : pricing analytique + Monte-Carlo (données synthétiques).

Exécuter :  python examples/quickstart.py
Ne nécessite aucun fichier externe ni accès réseau.
"""

from __future__ import annotations

import numpy as np

from betting_engine import price_match, simulate_match
from betting_engine.markets import one_x_two


def main() -> None:
    home_xg, away_xg, rho = 1.75, 1.05, -0.13

    print("=== Pricing analytique (Dixon-Coles) ===")
    bundle = price_match(home_xg, away_xg, rho=rho)
    print(f"xG : {home_xg} - {away_xg}  (rho={rho})")
    print("1X2 :", bundle["prob_1x2"])
    print("Over/Under 2.5 :", bundle["prob_ou25"])
    print("BTTS :", bundle["prob_btts"])
    print("Scores exacts probables :", bundle["top_scores"][:3])

    print("\n=== Simulation Monte-Carlo (50k, cohérence) ===")
    sim = simulate_match(home_xg, away_xg, rho=rho, n=50_000, rng=np.random.default_rng(0))
    print("1X2 (MC) :", one_x_two(sim.frequency_matrix))
    mean_total = float((sim.home_goals + sim.away_goals).mean())
    print(f"Total de buts moyen simulé : {mean_total:.2f}  (attendu ≈ {home_xg + away_xg:.2f})")


if __name__ == "__main__":
    main()
