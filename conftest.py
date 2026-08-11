"""Put the repo root on sys.path so a bare `pytest` works, not only `python -m pytest`.

`python -m pytest` prepends the current directory; plain `pytest` does not. Most
test modules here import theia inside their test functions and so survived
either way, but test_confabulation.py imports at module level and collection
failed with ModuleNotFoundError under bare `pytest`.

That is exactly how the CI workflow invokes it, so CI would have failed on its
first real run -- caught by scripts/reproduce.sh, which is the only thing that
ran pytest the way a newcomer would.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
