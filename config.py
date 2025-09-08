"""Compatibility shim so tests importing `config` find the project's config.

Tests import `config` at top-level; actual config lives in `scripts/config.py`.
This file re-exports everything for backwards compatibility.
"""

from scripts.config import *  # noqa: F401,F403
