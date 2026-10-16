import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phantom.config import load_config  # noqa: E402


@pytest.fixture
def sample_cfg():
    return load_config(ROOT / "configs" / "sample_mission.yaml")


@pytest.fixture
def decoy_cfg():
    return load_config(ROOT / "configs" / "decoy_mission.yaml")
