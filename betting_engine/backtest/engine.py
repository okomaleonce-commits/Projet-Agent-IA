"""Backtest walk-forward du modèle Dixon-Coles.

Pour chaque match de test, on (ré)ajuste le modèle sur les matchs strictement
antérieurs, on prédit le 1X2, puis on compare à l'issue réelle. Évaluation du modèle
par log-loss / Brier / exactitude (sans cotes). Une simulation de mises optionnelle
calcule le ROI dès lors qu'un fournisseur de cotes historiques est injecté.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from betting_engine.data import DataError, Match
from betting_engine.dixon_coles.fit import fit_dixon_coles
from betting_engine.dixon_coles.model import score_matrix
from betting_engine.dixon_coles.strengths import TeamStrengths, UnknownTeamError
from betting_engine.ingestion.odds import Odds1x2, value_bets
from betting_engine.markets import one_x_two
from betting_engine.metrics import accuracy, brier_score, log_loss, outcome_of

log = logging.getLogger("betting_engine.backtest")

OddsLookup = Callable[[Match], Odds1x2 | None]


@dataclass(frozen=True, slots=True)
class BettingReport:
    """Résultat de la simulation de mises (si cotes fournies)."""

    n_bets: int
    staked: float
    profit: float
    roi: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Résultat d'un backtest walk-forward."""

    n_test: int
    n_skipped: int
    log_loss: float
    brier_score: float
    accuracy: float
    betting: BettingReport | None = None


def walk_forward_backtest(
    matches: list[Match],
    min_train: int = 150,
    retrain_every: int = 10,
    xi: float = 0.0,
    odds_lookup: OddsLookup | None = None,
    stake: str = "flat",
    min_edge: float = 0.0,
    kelly_cap: float = 0.1,
    max_goals: int = 10,
) -> BacktestResult:
    """Rejoue une séquence de matchs en walk-forward.

    Args:
        matches: historique complet (sera trié par date).
        min_train: nombre de matchs d'entraînement avant le premier test.
        retrain_every: ré-ajuste le modèle tous les N matchs de test.
        xi: décroissance temporelle passée au fit.
        odds_lookup: fonction match -> cotes 1X2 (ou None) pour la simulation de mises.
        stake: "flat" (1 unité) ou "kelly" (fraction de Kelly, plafonnée).
        min_edge: edge minimal pour parier.
        kelly_cap: plafond de la fraction de Kelly par pari.
        max_goals: taille de la matrice de scores.

    Raises:
        DataError: pas assez de matchs pour `min_train`.
    """
    if len(matches) <= min_train:
        raise DataError(f"Besoin de plus de {min_train} matchs ; reçu {len(matches)}.")

    ordered = sorted(matches, key=lambda m: m.date)
    predictions: list[dict] = []
    outcomes: list[str] = []
    skipped = 0
    strengths: TeamStrengths | None = None
    bet_log: list[tuple[float, float]] = []  # (mise, gain_net)

    for position, match in enumerate(ordered[min_train:], start=min_train):
        if strengths is None or (position - min_train) % retrain_every == 0:
            strengths = _safe_fit(ordered[:position], xi)

        probs = _predict(strengths, match, max_goals)
        if probs is None:
            skipped += 1
            continue

        predictions.append(probs)
        outcomes.append(outcome_of(match.home_goals, match.away_goals))

        if odds_lookup is not None:
            _simulate_bet(match, probs, odds_lookup, stake, min_edge, kelly_cap, bet_log)

    if not predictions:
        raise DataError("Aucun match de test exploitable (équipes jamais vues à l'entraînement).")

    return BacktestResult(
        n_test=len(predictions),
        n_skipped=skipped,
        log_loss=round(log_loss(predictions, outcomes), 4),
        brier_score=round(brier_score(predictions, outcomes), 4),
        accuracy=round(accuracy(predictions, outcomes), 4),
        betting=_summarise_bets(bet_log) if odds_lookup is not None else None,
    )


def _safe_fit(history: list[Match], xi: float) -> TeamStrengths | None:
    try:
        return fit_dixon_coles(history, xi=xi).strengths
    except DataError:
        return None


def _predict(strengths: TeamStrengths | None, match: Match, max_goals: int) -> dict | None:
    if strengths is None:
        return None
    try:
        lam_h, lam_a = strengths.expected_goals(match.home_team, match.away_team)
    except UnknownTeamError:
        return None
    matrix = score_matrix(lam_h, lam_a, rho=strengths.rho, max_goals=max_goals)
    return one_x_two(matrix)


def _simulate_bet(
    match: Match,
    probs: dict,
    odds_lookup: OddsLookup,
    stake: str,
    min_edge: float,
    kelly_cap: float,
    bet_log: list[tuple[float, float]],
) -> None:
    odds = odds_lookup(match)
    if odds is None:
        return
    candidates = value_bets(probs, odds, min_edge=min_edge)
    if not candidates:
        return
    pick = candidates[0]  # meilleure value
    amount = 1.0 if stake == "flat" else min(pick["kelly"], kelly_cap)
    if amount <= 0:
        return
    won = pick["outcome"] == {"H": "home", "D": "draw", "A": "away"}[
        outcome_of(match.home_goals, match.away_goals)
    ]
    net = amount * (pick["odd"] - 1.0) if won else -amount
    bet_log.append((amount, net))


def _summarise_bets(bet_log: list[tuple[float, float]]) -> BettingReport:
    staked = sum(amount for amount, _ in bet_log)
    profit = sum(net for _, net in bet_log)
    return BettingReport(
        n_bets=len(bet_log),
        staked=round(staked, 3),
        profit=round(profit, 3),
        roi=round(profit / staked, 4) if staked > 0 else 0.0,
    )
