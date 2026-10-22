"""Contrat de données et chargement CSV.

`Match` est la brique d'entrée du fit : un résultat passé (date, équipes, buts).
Le loader CSV est calé par défaut sur le schéma `date,home_team,away_team,home_score,
away_score` (cf. Apex-TSS `data/history/intl_supplement.csv`) et accepte un mapping
de colonnes pour tout autre format.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y")

# Mapping par défaut : nom logique -> nom de colonne dans le CSV.
DEFAULT_COLUMNS: dict[str, str] = {
    "date": "date",
    "home_team": "home_team",
    "away_team": "away_team",
    "home_goals": "home_score",
    "away_goals": "away_score",
}


class DataError(ValueError):
    """Entrée de données invalide (colonne manquante, buts négatifs, date illisible)."""


@dataclass(frozen=True, slots=True)
class Match:
    """Un résultat de match joué (données immuables)."""

    date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    league_id: int | None = None
    season: int | None = None

    def __post_init__(self) -> None:
        if not self.home_team or not self.away_team:
            raise DataError("Les noms d'équipes ne peuvent pas être vides.")
        if self.home_goals < 0 or self.away_goals < 0:
            raise DataError(
                f"Buts négatifs interdits : {self.home_team} {self.home_goals}-"
                f"{self.away_goals} {self.away_team}"
            )


def parse_date(value: str) -> date:
    """Parse une date selon plusieurs formats courants."""
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DataError(f"Date illisible : {value!r} (formats acceptés : {_DATE_FORMATS})")


def load_matches_csv(
    path: str | Path,
    columns: dict[str, str] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> list[Match]:
    """Charge des résultats depuis un CSV en `list[Match]`.

    Args:
        path: chemin du fichier CSV.
        columns: mapping nom logique -> nom de colonne (défaut : `DEFAULT_COLUMNS`).
        delimiter: séparateur de champs.
        encoding: encodage du fichier.

    Raises:
        DataError: fichier introuvable, en-tête manquant ou ligne invalide.
    """
    col = {**DEFAULT_COLUMNS, **(columns or {})}
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"CSV introuvable : {file_path}")

    with file_path.open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = reader.fieldnames or []
        missing = [
            col[key]
            for key in ("date", "home_team", "away_team", "home_goals", "away_goals")
            if col[key] not in header
        ]
        if missing:
            raise DataError(
                f"Colonnes manquantes dans {file_path.name} : {missing} "
                f"(en-tête trouvé : {header})"
            )
        return [_row_to_match(row, col, file_path.name, i) for i, row in enumerate(reader, 2)]


def _row_to_match(row: dict[str, str], col: dict[str, str], name: str, line: int) -> Match:
    try:
        return Match(
            date=parse_date(row[col["date"]]),
            home_team=row[col["home_team"]].strip(),
            away_team=row[col["away_team"]].strip(),
            home_goals=int(row[col["home_goals"]]),
            away_goals=int(row[col["away_goals"]]),
        )
    except (KeyError, ValueError) as exc:
        raise DataError(f"{name} ligne {line} : ligne invalide ({exc})") from exc
