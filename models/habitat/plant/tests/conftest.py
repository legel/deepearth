"""Put the model directory on the path so tests import the package directly, and publish the collected count."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_collection_modifyitems(session, config, items):
    """Publish the collected count so `test_repo` can hold the README's number to it."""
    pytest.collected_count = len(items)
