"""Tests des métriques d'évaluation 1X2."""

from __future__ import annotations

import numpy as np
import pytest

from betting_engine.metrics import accuracy, brier_score, log_loss, outcome_of

UNIFORM = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}


def test_outcome_of() -> None:
    assert outcome_of(2, 0) == "H"
    assert outcome_of(1, 1) == "D"
    assert outcome_of(0, 3) == "A"


def test_log_loss_uniform_is_ln3() -> None:
    assert log_loss([UNIFORM, UNIFORM], ["H", "A"]) == pytest.approx(np.log(3), abs=1e-6)


def test_log_loss_perfect_is_near_zero() -> None:
    perfect = {"home": 1.0, "draw": 0.0, "away": 0.0}
    assert log_loss([perfect], ["H"]) < 1e-6


def test_brier_perfect_is_zero() -> None:
    perfect = {"home": 1.0, "draw": 0.0, "away": 0.0}
    assert brier_score([perfect], ["H"]) == pytest.approx(0.0)


def test_accuracy_counts_argmax() -> None:
    preds = [{"home": 0.6, "draw": 0.3, "away": 0.1}, {"home": 0.1, "draw": 0.2, "away": 0.7}]
    assert accuracy(preds, ["H", "H"]) == 0.5


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        log_loss([UNIFORM], ["H", "A"])  # longueurs différentes
    with pytest.raises(ValueError):
        log_loss([], [])  # vide
    with pytest.raises(ValueError):
        log_loss([UNIFORM], ["X"])  # issue invalide
