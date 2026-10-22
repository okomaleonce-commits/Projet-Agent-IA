"""Sous-package d'ingestion de données (API-Football)."""

from __future__ import annotations

from betting_engine.ingestion.api_football import ApiFootballError, fetch_results

__all__ = ["ApiFootballError", "fetch_results"]
