"""Dérivation des probabilités de marché depuis une matrice de scores.

Toutes les fonctions acceptent une matrice numpy `[h, a]` (cf. `dixon_coles.model`)
ou un échantillon Monte-Carlo converti en matrice. Inclut edge et Kelly, portés
d'APEX OMEGA.
"""

from __future__ import annotations

import numpy as np

Probabilities = dict[str, float]


def _as_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("La matrice de scores doit être carrée (shape [n, n]).")
    return arr


def one_x_two(matrix: np.ndarray) -> Probabilities:
    """P(domicile), P(nul), P(extérieur)."""
    m = _as_matrix(matrix)
    home = float(np.tril(m, -1).sum())  # h > a
    draw = float(np.trace(m))  # h == a
    away = float(np.triu(m, 1).sum())  # h < a
    return {"home": round(home, 4), "draw": round(draw, 4), "away": round(away, 4)}


def over_under(matrix: np.ndarray, line: float = 2.5) -> Probabilities:
    """P(total de buts > line) et P(total < line)."""
    m = _as_matrix(matrix)
    n = m.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = float(m[totals > line].sum())
    return {"line": line, "over": round(over, 4), "under": round(1.0 - over, 4)}


def btts(matrix: np.ndarray) -> Probabilities:
    """Both Teams To Score : P(dom ≥ 1 ET ext ≥ 1)."""
    m = _as_matrix(matrix)
    yes = float(m[1:, 1:].sum())
    return {"yes": round(yes, 4), "no": round(1.0 - yes, 4)}


def double_chance(probs_1x2: Probabilities) -> Probabilities:
    """Double chance (1X, X2, 12) à partir du 1X2."""
    return {
        "1X": round(probs_1x2["home"] + probs_1x2["draw"], 4),
        "X2": round(probs_1x2["draw"] + probs_1x2["away"], 4),
        "12": round(probs_1x2["home"] + probs_1x2["away"], 4),
    }


def top_scores(matrix: np.ndarray, top_n: int = 6) -> list[tuple[int, int, float]]:
    """Les `top_n` scores exacts les plus probables : liste de (dom, ext, proba)."""
    m = _as_matrix(matrix)
    flat_idx = np.argsort(m, axis=None)[::-1][:top_n]
    rows, cols = np.unravel_index(flat_idx, m.shape)
    return [(int(h), int(a), round(float(m[h, a]), 4)) for h, a in zip(rows, cols, strict=True)]


def asian_handicap(matrix: np.ndarray, line: float) -> Probabilities:
    """Handicap asiatique appliqué au score domicile.

    `line` est un multiple de 0.25. Retourne les probabilités de gain côté
    domicile et extérieur (remboursements de push partagés pour les lignes entières
    et quart de but), de sorte que `home + away == 1`.
    """
    if abs(line * 4 - round(line * 4)) > 1e-9:
        raise ValueError("La ligne de handicap doit être un multiple de 0.25.")
    quarter = round(line * 4) / 4

    # Les lignes quart de but se décomposent en deux demi-mises adjacentes.
    if (round(quarter * 4) % 2) != 0:
        low = _asian_half_or_whole(matrix, quarter - 0.25)
        high = _asian_half_or_whole(matrix, quarter + 0.25)
        return {
            "line": line,
            "home": round((low["home"] + high["home"]) / 2, 4),
            "away": round((low["away"] + high["away"]) / 2, 4),
        }
    result = _asian_half_or_whole(matrix, quarter)
    result["line"] = line
    return result


def _asian_half_or_whole(matrix: np.ndarray, line: float) -> Probabilities:
    """Résout un handicap demi (.5) ou entier (.0, push remboursé)."""
    m = _as_matrix(matrix)
    n = m.shape[0]
    margin = np.add.outer(np.arange(n), -np.arange(n)) + line  # (h - a) + line
    home = float(m[margin > 0].sum())
    push = float(m[np.isclose(margin, 0.0)].sum())
    away = float(m[margin < 0].sum())
    if push > 0:  # ligne entière : on retire la part remboursée du dénominateur
        staked = home + away
        home = home / staked if staked > 0 else 0.0
        away = away / staked if staked > 0 else 0.0
    return {"home": round(home, 4), "away": round(away, 4)}


def price(matrix: np.ndarray) -> dict:
    """Dérive l'ensemble des marchés usuels en une passe."""
    probs_1x2 = one_x_two(matrix)
    return {
        "prob_1x2": probs_1x2,
        "prob_dc": double_chance(probs_1x2),
        "prob_btts": btts(matrix),
        "prob_ou15": over_under(matrix, 1.5),
        "prob_ou25": over_under(matrix, 2.5),
        "prob_ou35": over_under(matrix, 3.5),
        "top_scores": top_scores(matrix),
    }


def edge(model_prob: float, bookmaker_odd: float | None) -> float | None:
    """Edge = proba modèle − proba implicite de la cote. None si cote invalide."""
    if not bookmaker_odd or bookmaker_odd <= 1.0:
        return None
    return round(model_prob - 1.0 / bookmaker_odd, 4)


def kelly_fraction(model_prob: float, bookmaker_odd: float | None) -> float:
    """Fraction de Kelly pleine : f* = (b·p − q) / b, plancher à 0."""
    if not bookmaker_odd or bookmaker_odd <= 1.0:
        return 0.0
    b = bookmaker_odd - 1.0
    f = (b * model_prob - (1.0 - model_prob)) / b
    return max(round(f, 4), 0.0)
