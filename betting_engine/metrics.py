"""Métriques d'évaluation probabiliste pour les prédictions 1X2.

Fonctions pures (aucune dépendance réseau) : log-loss, score de Brier, exactitude.
Une prédiction est un dict {"home", "draw", "away"} ; le résultat observé est l'un
des libellés "H", "D", "A".
"""

from __future__ import annotations

import numpy as np

_OUTCOME_KEYS = {"H": "home", "D": "draw", "A": "away"}
_EPS = 1e-15


def _validate(predictions: list[dict], outcomes: list[str]) -> None:
    if len(predictions) != len(outcomes):
        raise ValueError("predictions et outcomes doivent avoir la même longueur.")
    if not predictions:
        raise ValueError("Aucune prédiction à évaluer.")
    bad = {o for o in outcomes if o not in _OUTCOME_KEYS}
    if bad:
        raise ValueError(f"Résultats invalides : {bad} (attendus : H, D, A).")


def log_loss(predictions: list[dict], outcomes: list[str]) -> float:
    """Log-loss moyen : −moyenne(log p_réalisé). Plus bas = meilleur."""
    _validate(predictions, outcomes)
    losses = [
        -np.log(max(pred[_OUTCOME_KEYS[out]], _EPS))
        for pred, out in zip(predictions, outcomes, strict=True)
    ]
    return float(np.mean(losses))


def brier_score(predictions: list[dict], outcomes: list[str]) -> float:
    """Score de Brier multiclasse moyen : moyenne de Σ_k (p_k − y_k)². Plus bas = meilleur."""
    _validate(predictions, outcomes)
    total = 0.0
    for pred, out in zip(predictions, outcomes, strict=True):
        for key in ("home", "draw", "away"):
            target = 1.0 if _OUTCOME_KEYS[out] == key else 0.0
            total += (pred[key] - target) ** 2
    return float(total / len(predictions))


def accuracy(predictions: list[dict], outcomes: list[str]) -> float:
    """Proportion de matchs où l'issue la plus probable du modèle est la bonne."""
    _validate(predictions, outcomes)
    hits = 0
    for pred, out in zip(predictions, outcomes, strict=True):
        best = max(("home", "draw", "away"), key=lambda k: pred[k])
        if best == _OUTCOME_KEYS[out]:
            hits += 1
    return hits / len(predictions)


def outcome_of(home_goals: int, away_goals: int) -> str:
    """Encode un score en issue 1X2 : 'H', 'D' ou 'A'."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"
