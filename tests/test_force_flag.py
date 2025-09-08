from pathlib import Path
import importlib

from scripts import config


def test_force_flag_present():
    assert hasattr(config, "FORCE_TRAIN")
