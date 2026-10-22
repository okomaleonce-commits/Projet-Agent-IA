"""Tests des dérivations de marché, edge et Kelly."""

from __future__ import annotations

import numpy as np
import pytest

from betting_engine.dixon_coles.model import score_matrix
from betting_engine.markets import (
    asian_handicap,
    btts,
    double_chance,
    edge,
    kelly_fraction,
    one_x_two,
    over_under,
    price,
    top_scores,
)

MATRIX = score_matrix(1.7, 1.1, rho=-0.13)


def test_one_x_two_sums_to_one() -> None:
    probs = one_x_two(MATRIX)
    assert abs(probs["home"] + probs["draw"] + probs["away"] - 1.0) < 1e-3


def test_over_under_complementary() -> None:
    probs = over_under(MATRIX, 2.5)
    assert abs(probs["over"] + probs["under"] - 1.0) < 1e-3


def test_btts_complementary() -> None:
    probs = btts(MATRIX)
    assert abs(probs["yes"] + probs["no"] - 1.0) < 1e-3


def test_double_chance() -> None:
    probs = one_x_two(MATRIX)
    dc = double_chance(probs)
    assert np.isclose(dc["1X"], probs["home"] + probs["draw"])


def test_top_scores_sorted_desc() -> None:
    scores = top_scores(MATRIX, top_n=5)
    probs = [p for _, _, p in scores]
    assert probs == sorted(probs, reverse=True)
    assert len(scores) == 5


def test_asian_handicap_half_line_sums_to_one() -> None:
    ah = asian_handicap(MATRIX, -0.5)
    assert abs(ah["home"] + ah["away"] - 1.0) < 1e-3


def test_asian_handicap_whole_line_refunds_push() -> None:
    ah = asian_handicap(MATRIX, -1.0)
    # push retiré du dénominateur -> home + away == 1
    assert abs(ah["home"] + ah["away"] - 1.0) < 1e-3


def test_asian_handicap_quarter_line_between_neighbours() -> None:
    low = asian_handicap(MATRIX, -0.5)["home"]
    high = asian_handicap(MATRIX, -1.0)["home"]
    quarter = asian_handicap(MATRIX, -0.75)["home"]
    assert min(low, high) - 1e-6 <= quarter <= max(low, high) + 1e-6


def test_asian_handicap_rejects_non_quarter() -> None:
    with pytest.raises(ValueError):
        asian_handicap(MATRIX, -0.3)


def test_price_bundle_keys() -> None:
    bundle = price(MATRIX)
    assert {"prob_1x2", "prob_dc", "prob_btts", "prob_ou25", "top_scores"} <= bundle.keys()


def test_edge_and_kelly() -> None:
    assert edge(0.6, 2.0) == pytest.approx(0.1)
    assert edge(0.6, None) is None
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 2.0) == 0.0  # pas de valeur -> 0


def test_markets_reject_non_square() -> None:
    with pytest.raises(ValueError):
        one_x_two(np.zeros((3, 4)))
