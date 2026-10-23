"""Tests de l'ingestion des cotes et du calcul de value (session mockée)."""

from __future__ import annotations

import pytest

from betting_engine.ingestion.odds import (
    Odds1x2,
    OddsError,
    fetch_odds_1x2,
    fetch_upcoming,
    value_bets,
)


class _Resp:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _odds_payload() -> dict:
    def book(name, h, d, a):
        return {
            "name": name,
            "bets": [{"values": [
                {"value": "Home", "odd": str(h)},
                {"value": "Draw", "odd": str(d)},
                {"value": "Away", "odd": str(a)},
            ]}],
        }

    return {"response": [{"bookmakers": [book("A", 2.0, 3.4, 3.8), book("B", 2.1, 3.3, 3.6)]}]}


class _OddsSession:
    def get(self, url, headers=None, params=None, timeout=None) -> _Resp:
        return _Resp(200, _odds_payload())


def test_implied_probabilities_remove_overround() -> None:
    probs = Odds1x2(2.0, 3.4, 3.8).implied_probabilities(remove_overround=True)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_fetch_odds_aggregates_best_and_average() -> None:
    odds = fetch_odds_1x2(123, session=_OddsSession())
    assert odds is not None
    assert odds.best.home == 2.1  # meilleure cote domicile
    assert odds.average.home == pytest.approx(2.05)
    assert odds.n_books == 2


def test_fetch_odds_empty_returns_none() -> None:
    class _Empty:
        def get(self, url, headers=None, params=None, timeout=None) -> _Resp:
            return _Resp(200, {"response": []})

    assert fetch_odds_1x2(123, session=_Empty()) is None


def test_fetch_odds_http_error() -> None:
    class _Err:
        def get(self, url, headers=None, params=None, timeout=None) -> _Resp:
            return _Resp(500, {})

    with pytest.raises(OddsError):
        fetch_odds_1x2(123, session=_Err())


def test_value_bets_flags_positive_edge() -> None:
    model = {"home": 0.55, "draw": 0.25, "away": 0.20}
    odds = Odds1x2(2.0, 3.4, 3.8)  # implicite home=0.5 -> edge +0.05
    bets = value_bets(model, odds, min_edge=0.0)
    assert bets[0]["outcome"] == "home"
    assert bets[0]["edge"] == pytest.approx(0.05, abs=1e-4)
    assert bets[0]["kelly"] > 0


def test_value_bets_min_edge_filter() -> None:
    model = {"home": 0.50, "draw": 0.25, "away": 0.25}
    odds = Odds1x2(2.0, 3.4, 3.8)
    assert value_bets(model, odds, min_edge=0.05) == []


def test_fetch_upcoming_maps_fixtures() -> None:
    class _UpSession:
        def get(self, url, headers=None, params=None, timeout=None) -> _Resp:
            return _Resp(200, {"response": [{
                "fixture": {"id": 999, "date": "2026-10-24T14:00:00+00:00"},
                "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Everton"}},
            }]})

    up = fetch_upcoming(39, 2026, n=1, session=_UpSession())
    assert up[0]["fixture_id"] == 999
    assert up[0]["home_team"] == "Arsenal"
    assert up[0]["date"] == "2026-10-24"
