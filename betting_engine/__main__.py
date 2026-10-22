"""Interface en ligne de commande : `python -m betting_engine ...`.

Sous-commandes :
    price  — price un match à partir d'xG fournis.
    fit    — récupère l'historique d'une ligue/saison via API-Football, fitte le
             modèle, puis price une affiche.
"""

from __future__ import annotations

import argparse
import logging

from betting_engine import fit_dixon_coles, price_fixture, price_match
from betting_engine.config import league_defaults
from betting_engine.ingestion import fetch_results


def _print_bundle(bundle: dict) -> None:
    print("1X2            :", bundle["prob_1x2"])
    print("Double chance  :", bundle["prob_dc"])
    print("Over/Under 2.5 :", bundle["prob_ou25"])
    print("BTTS           :", bundle["prob_btts"])
    print("Scores probables:", bundle["top_scores"][:4])


def _cmd_price(args: argparse.Namespace) -> None:
    bundle = price_match(args.home_xg, args.away_xg, rho=args.rho)
    print(f"Pricing xG {args.home_xg} - {args.away_xg} (rho={args.rho})")
    _print_bundle(bundle)


def _cmd_fit(args: argparse.Namespace) -> None:
    meta = league_defaults(args.league)
    print(f"Récupération {meta['name']} (ligue {args.league}, saison {args.season})...")
    matches = fetch_results(args.league, args.season)
    print(f"{len(matches)} matchs récupérés — fit en cours...")
    result = fit_dixon_coles(matches, xi=args.xi)
    print(f"Fit : {result.n_teams} équipes, logL={result.log_likelihood:.1f}, "
          f"convergé={result.converged}")
    bundle = price_fixture(result.strengths, args.home, args.away)
    print(f"\nPricing {args.home} (dom) vs {args.away} (ext) :")
    _print_bundle(bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betting_engine", description="Pricing Dixon-Coles.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_price = sub.add_parser("price", help="Price un match à partir d'xG.")
    p_price.add_argument("--home-xg", type=float, required=True, dest="home_xg")
    p_price.add_argument("--away-xg", type=float, required=True, dest="away_xg")
    p_price.add_argument("--rho", type=float, default=-0.13)
    p_price.set_defaults(func=_cmd_price)

    p_fit = sub.add_parser("fit", help="Fit sur l'historique API-Football puis price une affiche.")
    p_fit.add_argument("--league", type=int, required=True)
    p_fit.add_argument("--season", type=int, required=True)
    p_fit.add_argument("--home", required=True)
    p_fit.add_argument("--away", required=True)
    p_fit.add_argument("--xi", type=float, default=0.0)
    p_fit.set_defaults(func=_cmd_fit)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
