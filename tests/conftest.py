import sys
from pathlib import Path

# Make the packages in src/ importable in the tests (processing, evaluation).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))