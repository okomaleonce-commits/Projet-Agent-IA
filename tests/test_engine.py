"""Tests des fonctions de haut niveau (pricing xG et pricing par forces)."""

from __future__ import annotations

import pytest

from betting_engine.dixon_coles.strengths import TeamStrengths, UnknownTeamError
from betting_engine.engine import price_fixture, price_match


def test_price_match_structure() -> None:
    bundle = price_match(1.7, 1.1, rho=-0.13)
    assert bundle["inputs"]["lambda_home"] == 1.7
    assert "prob_1x2" in bundle
    total = sum(bundle["prob_1x2"].values())
    assert abs(total - 1.0) < 1e-3


def test_price_fixture_uses_strengths() -> None:
    strengths = TeamStrengths(
        attack={"A": 0.3, "B": -0.3},
        defense={"A": 0.0, "B": 0.1},
        home_adv=0.25,
        rho=-0.12,
        teams=("A", "B"),
    )
    bundle = price_fixture(strengths, "A", "B")
    assert bundle["inputs"]["rho"] == -0.12
    assert bundle["prob_1x2"]["home"] > bundle["prob_1x2"]["away"]  # A plus fort à domicile


def test_price_fixture_unknown_team() -> None:
    strengths = TeamStrengths({"A": 0.0}, {"A": 0.0}, 0.2, -0.1, ("A",))
    with pytest.raises(UnknownTeamError):
        price_fixture(strengths, "A", "Z")
