"""Tests du backtest walk-forward (données synthétiques, cotes injectées)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from betting_engine.backtest import walk_forward_backtest
from betting_engine.data import DataError, Match
from betting_engine.dixon_coles.model import score_matrix
from betting_engine.ingestion.odds import Odds1x2

TEAMS = ["T0", "T1", "T2", "T3", "T4"]
ATTACK = {"T0": 0.35, "T1": 0.12, "T2": 0.0, "T3": -0.15, "T4": -0.32}
BASE = float(np.log(1.35))
DEFENSE = {"T0": 0.0, "T1": -0.05, "T2": 0.05, "T3": 0.10, "T4": -0.10}
HOME_ADV, RHO = 0.25, -0.10


def _synthetic(rounds: int, seed: int = 7) -> list[Match]:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    matches: list[Match] = []
    day = 0
    for _ in range(rounds):
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                lam_h = np.exp(HOME_ADV + ATTACK[home] + BASE + DEFENSE[away])
                lam_a = np.exp(ATTACK[away] + BASE + DEFENSE[home])
                matrix = score_matrix(lam_h, lam_a, rho=RHO, max_goals=10)
                flat = matrix.ravel() / matrix.sum()
                hg, ag = divmod(int(rng.choice(flat.size, p=flat)), matrix.shape[0])
                matches.append(Match(start + timedelta(days=day), home, away, hg, ag))
                day += 1
    return matches


def test_backtest_beats_uniform() -> None:
    result = walk_forward_backtest(
        _synthetic(rounds=35), min_train=200, retrain_every=50
    )
    assert result.n_test > 0
    assert result.log_loss < np.log(3)  # mieux que l'uniforme
    assert result.betting is None  # pas de cotes fournies


def test_backtest_with_odds_produces_report() -> None:
    def fixed_odds(_match: Match) -> Odds1x2:
        return Odds1x2(2.5, 3.3, 2.8)  # cotes généreuses -> des value bets

    result = walk_forward_backtest(
        _synthetic(rounds=35), min_train=200, retrain_every=50, odds_lookup=fixed_odds
    )
    assert result.betting is not None
    assert result.betting.n_bets >= 1
    assert result.betting.staked > 0


def test_backtest_requires_enough_matches() -> None:
    with pytest.raises(DataError):
        walk_forward_backtest(_synthetic(rounds=2), min_train=150)
