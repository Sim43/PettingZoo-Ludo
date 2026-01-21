import os
import sys
import numpy as np

# Ensure local PettingZoo repo is importable when running tests without installation
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PETTINGZOO_ROOT = os.path.join(ROOT, "PettingZoo")
if PETTINGZOO_ROOT not in sys.path:
    sys.path.insert(0, PETTINGZOO_ROOT)

from pettingzoo.test import (  # type: ignore  # imported from installed PettingZoo
    api_test,
    performance_benchmark,
    render_test,
    seed_test,
)

from ludo.ludo import env


def test_api():
    """Basic PettingZoo AEC API compliance test (single mode)."""
    api_test(env(), num_cycles=1000, verbose_progress=False)


def test_api_teams():
    """PettingZoo AEC API compliance test for 2v2 team mode."""
    api_test(env(mode="teams"), num_cycles=1000, verbose_progress=False)


def test_seed():
    """Determinism / seeding compliance test."""
    seed_test(env, num_cycles=10)


def test_render():
    """Check that rendering works and returns the correct types."""
    render_test(env)


def test_performance_benchmark():
    """Performance benchmark (manual inspection; skipped by default)."""
    performance_benchmark(env())
