"""Tests du contrat de données et du chargement CSV."""

from __future__ import annotations

from datetime import date

import pytest

from betting_engine.data import DataError, Match, load_matches_csv, parse_date


def test_match_valid() -> None:
    match = Match(date(2026, 9, 21), "Fiji", "Vanuatu", 0, 1)
    assert match.home_team == "Fiji"
    assert match.away_goals == 1


def test_match_rejects_negative_goals() -> None:
    with pytest.raises(DataError):
        Match(date(2026, 9, 21), "A", "B", -1, 0)


def test_match_rejects_empty_team() -> None:
    with pytest.raises(DataError):
        Match(date(2026, 9, 21), "", "B", 1, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2026-09-21", date(2026, 9, 21)), ("21/09/2026", date(2026, 9, 21))],
)
def test_parse_date_formats(text: str, expected: date) -> None:
    assert parse_date(text) == expected


def test_parse_date_invalid() -> None:
    with pytest.raises(DataError):
        parse_date("not-a-date")


def test_load_matches_csv_default_schema(tmp_path) -> None:
    csv_file = tmp_path / "hist.csv"
    csv_file.write_text(
        "date,home_team,away_team,home_score,away_score\n"
        "2026-09-21,New Caledonia,Solomon Islands,0,2\n"
        "2026-09-21,Fiji,Vanuatu,0,1\n",
        encoding="utf-8",
    )
    matches = load_matches_csv(csv_file)
    assert len(matches) == 2
    assert matches[0].home_team == "New Caledonia"
    assert matches[1].away_goals == 1


def test_load_matches_csv_missing_file(tmp_path) -> None:
    with pytest.raises(DataError):
        load_matches_csv(tmp_path / "absent.csv")


def test_load_matches_csv_missing_column(tmp_path) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("date,home_team,away_team,home_score\n2026-09-21,A,B,1\n", encoding="utf-8")
    with pytest.raises(DataError):
        load_matches_csv(csv_file)


def test_load_matches_csv_custom_mapping(tmp_path) -> None:
    csv_file = tmp_path / "custom.csv"
    csv_file.write_text("d;h;a;hg;ag\n2026-09-21;A;B;2;1\n", encoding="utf-8")
    mapping = {
        "date": "d",
        "home_team": "h",
        "away_team": "a",
        "home_goals": "hg",
        "away_goals": "ag",
    }
    matches = load_matches_csv(csv_file, columns=mapping, delimiter=";")
    assert matches[0].home_goals == 2
