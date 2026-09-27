"""Put the package root on the path so tests import the modules directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
