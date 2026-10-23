"""Interface en ligne de commande : `python -m betting_engine ...`.

Sous-commandes :
    price     — price un match à partir d'xG fournis.
    fit       — récupère l'historique (API-Football), fitte, puis price une affiche.
    backtest  — backtest walk-forward du modèle sur une ligue/saison (log-loss, Brier...).
    value     — fitte la saison en cours, récupère les matchs à venir + cotes et
                liste les paris à valeur positive (edge + Kelly).
"""

from __future__ import annotations

import argparse
import logging

from betting_engine import (
    fetch_odds_1x2,
    fetch_results,
    fetch_upcoming,
    fit_dixon_coles,
    price_fixture,
    price_match,
    value_bets,
    walk_forward_backtest,
)
from betting_engine.config import league_defaults
from betting_engine.dixon_coles.strengths import UnknownTeamError

log = logging.getLogger("betting_engine.cli")


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


def _cmd_backtest(args: argparse.Namespace) -> None:
    meta = league_defaults(args.league)
    print(f"Backtest {meta['name']} (ligue {args.league}, saison {args.season})...")
    matches = fetch_results(args.league, args.season)
    print(f"{len(matches)} matchs — backtest walk-forward (min_train={args.min_train})...")
    result = walk_forward_backtest(
        matches, min_train=args.min_train, retrain_every=args.retrain_every, xi=args.xi
    )
    print(f"Matchs testés  : {result.n_test} (ignorés : {result.n_skipped})")
    print(f"Log-loss       : {result.log_loss}  (uniforme ≈ 1.0986)")
    print(f"Score de Brier : {result.brier_score}")
    print(f"Exactitude     : {result.accuracy:.1%}")


def _cmd_value(args: argparse.Namespace) -> None:
    meta = league_defaults(args.league)
    print(f"{meta['name']} — fit saison {args.season} puis value sur les prochains matchs...")
    matches = fetch_results(args.league, args.season)
    strengths = fit_dixon_coles(matches, xi=args.xi).strengths
    upcoming = fetch_upcoming(args.league, args.season, n=args.next)
    print(f"{len(matches)} matchs d'entraînement, {len(upcoming)} matchs à venir.\n")

    for fixture in upcoming:
        _report_fixture_value(fixture, strengths, args.min_edge)


def _report_fixture_value(fixture: dict, strengths, min_edge: float) -> None:
    label = f"{fixture['date']}  {fixture['home_team']} vs {fixture['away_team']}"
    try:
        bundle = price_fixture(strengths, fixture["home_team"], fixture["away_team"])
    except UnknownTeamError:
        print(f"{label} — équipe non vue à l'entraînement, ignoré.")
        return
    odds = fetch_odds_1x2(fixture["fixture_id"])
    if odds is None:
        print(f"{label} — pas de cotes publiées.")
        return
    bets = value_bets(bundle["prob_1x2"], odds.best, min_edge=min_edge)
    if not bets:
        print(f"{label} — aucune value (edge < {min_edge}).")
        return
    print(f"{label}")
    for bet in bets:
        print(f"    {bet['outcome']:<5} cote {bet['odd']:<5} "
              f"p={bet['model_prob']:.3f} edge={bet['edge']:+.3f} kelly={bet['kelly']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betting_engine", description="Pricing Dixon-Coles.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_price = sub.add_parser("price", help="Price un match à partir d'xG.")
    p_price.add_argument("--home-xg", type=float, required=True, dest="home_xg")
    p_price.add_argument("--away-xg", type=float, required=True, dest="away_xg")
    p_price.add_argument("--rho", type=float, default=-0.13)
    p_price.set_defaults(func=_cmd_price)

    p_fit = sub.add_parser("fit", help="Fit API-Football puis price une affiche.")
    p_fit.add_argument("--league", type=int, required=True)
    p_fit.add_argument("--season", type=int, required=True)
    p_fit.add_argument("--home", required=True)
    p_fit.add_argument("--away", required=True)
    p_fit.add_argument("--xi", type=float, default=0.0)
    p_fit.set_defaults(func=_cmd_fit)

    p_bt = sub.add_parser("backtest", help="Backtest walk-forward sur une ligue/saison.")
    p_bt.add_argument("--league", type=int, required=True)
    p_bt.add_argument("--season", type=int, required=True)
    p_bt.add_argument("--min-train", type=int, default=150, dest="min_train")
    p_bt.add_argument("--retrain-every", type=int, default=10, dest="retrain_every")
    p_bt.add_argument("--xi", type=float, default=0.0)
    p_bt.set_defaults(func=_cmd_backtest)

    p_val = sub.add_parser("value", help="Value + Kelly sur les prochains matchs.")
    p_val.add_argument("--league", type=int, required=True)
    p_val.add_argument("--season", type=int, required=True)
    p_val.add_argument("--next", type=int, default=10)
    p_val.add_argument("--min-edge", type=float, default=0.0, dest="min_edge")
    p_val.add_argument("--xi", type=float, default=0.0)
    p_val.set_defaults(func=_cmd_value)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
