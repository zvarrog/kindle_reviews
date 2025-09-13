from pathlib import Path
import importlib

from scripts.settings import FORCE_TRAIN


def test_force_flag_present():
    assert FORCE_TRAIN is not None
