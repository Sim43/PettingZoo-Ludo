import inspect
import os
import sys

import numpy as np
import pytest

# Ensure local PettingZoo repo is importable when running tests without installation
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PETTINGZOO_ROOT = os.path.join(ROOT, "PettingZoo")
if PETTINGZOO_ROOT not in sys.path:
    sys.path.insert(0, PETTINGZOO_ROOT)

from pettingzoo.test import (  # type: ignore  # imported from installed PettingZoo
    api_test,
    max_cycles_test,
    performance_benchmark,
    render_test,
    seed_test,
)

from ludo.ludo import env, raw_env


def test_api():
    """Basic PettingZoo AEC API compliance test."""
    api_test(env(), num_cycles=1000, verbose_progress=False)


def test_seed():
    """Determinism / seeding compliance test."""
    seed_test(env, num_cycles=10)


def test_max_cycles():
    """Run max_cycles_test only if the environment supports a max_cycles argument."""
    sig = inspect.signature(raw_env.__init__)
    if "max_cycles" not in sig.parameters:
        pytest.skip("max_cycles argument not supported by raw_env.")
    max_cycles_test(__import__("ludo.ludo", fromlist=["*"]))


def test_render():
    """Check that rendering works and returns the correct types."""
    render_test(env)


@pytest.mark.skip(reason="Performance benchmark is intended for manual inspection.")
def test_performance_benchmark():
    """Performance benchmark (manual inspection; skipped by default)."""
    performance_benchmark(env())


def test_save_observation():
    """Minimal save-observation-style test to ensure obs can be materialized."""
    e = env()
    e.reset()
    for _ in range(e.num_agents * 5):
        agent = e.agent_selection
        obs, rew, term, trunc, info = e.last()
        if isinstance(obs, dict) and "observation" in obs:
            arr = np.asarray(obs["observation"])
            assert arr.size > 0
        if term or trunc:
            e.step(None)
        else:
            if isinstance(obs, dict) and "action_mask" in obs:
                mask = obs["action_mask"]
                legal = [i for i, v in enumerate(mask) if v == 1]
                action = legal[0] if legal else None
            else:
                action = e.action_space(agent).sample()
            e.step(action)

