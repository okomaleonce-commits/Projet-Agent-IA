"""Tests du CLI (sous-commande `price` ; `fit` nécessite le réseau, non testé ici)."""

from __future__ import annotations

import pytest

from betting_engine.__main__ import build_parser, main


def test_parser_price_arguments() -> None:
    args = build_parser().parse_args(["price", "--home-xg", "1.7", "--away-xg", "1.1"])
    assert args.command == "price"
    assert args.home_xg == 1.7
    assert args.rho == -0.13


def test_main_price_runs(capsys) -> None:
    main(["price", "--home-xg", "1.7", "--away-xg", "1.1"])
    out = capsys.readouterr().out
    assert "1X2" in out


def test_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
