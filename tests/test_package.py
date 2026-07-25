"""Smoke tests for the package scaffold."""

from gridfm_app_distribution import __version__
from gridfm_app_distribution.cli import main


def test_version_is_defined() -> None:
    """The package exposes its initial version."""
    assert __version__ == "0.1.0"


def test_cli_runs() -> None:
    """The empty command line exits successfully."""
    assert main([]) == 0

