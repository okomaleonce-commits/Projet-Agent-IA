"""Tests du fit MLE : récupération de paramètres sur données synthétiques."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from betting_engine.data import DataError, Match
from betting_engine.dixon_coles.fit import fit_dixon_coles
from betting_engine.dixon_coles.model import score_matrix

TEAMS = ["T0", "T1", "T2", "T3", "T4"]
TRUE_ATTACK = {"T0": 0.35, "T1": 0.12, "T2": 0.0, "T3": -0.15, "T4": -0.32}
_BASELINE = float(np.log(1.3))
TRUE_DEFENSE = {"T0": 0.0, "T1": -0.05, "T2": 0.05, "T3": 0.10, "T4": -0.10}
TRUE_HOME_ADV = 0.25
TRUE_RHO = -0.10


def _true_lambdas(home: str, away: str) -> tuple[float, float]:
    lam_h = np.exp(TRUE_HOME_ADV + TRUE_ATTACK[home] + _BASELINE + TRUE_DEFENSE[away])
    lam_a = np.exp(TRUE_ATTACK[away] + _BASELINE + TRUE_DEFENSE[home])
    return float(lam_h), float(lam_a)


def _synthetic_matches(rounds: int, seed: int = 123) -> list[Match]:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    matches: list[Match] = []
    day = 0
    for _ in range(rounds):
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                lam_h, lam_a = _true_lambdas(home, away)
                matrix = score_matrix(lam_h, lam_a, rho=TRUE_RHO, max_goals=10)
                flat = matrix.ravel() / matrix.sum()
                draw = rng.choice(flat.size, p=flat)
                hg, ag = divmod(int(draw), matrix.shape[0])
                matches.append(Match(start + timedelta(days=day), home, away, hg, ag))
                day += 1
    return matches


def test_fit_recovers_expected_goals() -> None:
    result = fit_dixon_coles(_synthetic_matches(rounds=90), xi=0.0)
    assert result.n_teams == 5
    assert np.isfinite(result.log_likelihood)

    rel_errors = []
    for home in TEAMS:
        for away in TEAMS:
            if home == away:
                continue
            true_h, true_a = _true_lambdas(home, away)
            pred_h, pred_a = result.strengths.expected_goals(home, away)
            rel_errors.append(abs(pred_h - true_h) / true_h)
            rel_errors.append(abs(pred_a - true_a) / true_a)
    assert np.median(rel_errors) < 0.10


def test_fit_recovers_home_advantage_sign() -> None:
    result = fit_dixon_coles(_synthetic_matches(rounds=60), xi=0.0)
    assert result.strengths.home_adv > 0


def test_fit_time_weights_do_not_crash() -> None:
    result = fit_dixon_coles(_synthetic_matches(rounds=30), xi=0.0065)
    assert result.n_matches == 30 * 20


def test_fit_rejects_empty() -> None:
    with pytest.raises(DataError):
        fit_dixon_coles([])


def test_fit_rejects_single_team() -> None:
    matches = [Match(date(2024, 1, 1), "A", "A", 1, 1)]
    with pytest.raises(DataError):
        fit_dixon_coles(matches)
