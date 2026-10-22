"""Tests de l'ingestion API-Football (session mockée, aucun appel réseau réel)."""

from __future__ import annotations

import pytest

from betting_engine.ingestion.api_football import ApiFootballError, fetch_results


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _fixture(home: str, away: str, hg, ag, status: str = "FT") -> dict:
    return {
        "fixture": {"date": "2026-09-21T15:00:00+00:00", "status": {"short": status}},
        "teams": {"home": {"name": home}, "away": {"name": away}},
        "goals": {"home": hg, "away": ag},
    }


class _PagingSession:
    """Session factice renvoyant deux pages de résultats."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, url, headers=None, params=None, timeout=None) -> _DummyResponse:
        self.calls += 1
        page = params.get("page", 1)
        if page == 1:
            return _DummyResponse(
                200,
                {
                    "response": [
                        _fixture("Fiji", "Vanuatu", 0, 1),
                        _fixture("A", "B", None, None),  # ignoré (buts manquants)
                        _fixture("C", "D", 2, 2, status="NS"),  # ignoré (non terminé)
                    ],
                    "paging": {"current": 1, "total": 2},
                },
            )
        return _DummyResponse(
            200,
            {
                "response": [_fixture("New Caledonia", "Solomon Islands", 3, 0)],
                "paging": {"current": 2, "total": 2},
            },
        )


def test_fetch_results_filters_and_paginates() -> None:
    session = _PagingSession()
    matches = fetch_results(39, 2026, session=session)
    assert session.calls == 2  # pagination suivie
    assert len(matches) == 2  # None-goals et non-FT exclus
    names = {m.home_team for m in matches}
    assert names == {"Fiji", "New Caledonia"}
    assert matches[0].league_id == 39


class _ErrorSession:
    def get(self, url, headers=None, params=None, timeout=None) -> _DummyResponse:
        return _DummyResponse(403, {})


def test_fetch_results_raises_on_http_error() -> None:
    with pytest.raises(ApiFootballError):
        fetch_results(39, 2026, session=_ErrorSession())
