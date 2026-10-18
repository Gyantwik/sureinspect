"""Tests for core/interfaces.py (the frozen P1 <-> P2 contract)."""
import inspect

import numpy as np
import pytest

def test_interfaces_expose_the_frozen_signatures():
    from core import interfaces as I

    expected = {
        "standardize": ["image_bgr", "max_side"],
        "to_blob": ["image_std", "size"],
        "fit_gate_thresholds": ["images_std"],
        "check_quality": ["image_std", "thresholds"],
        "inspect_standardized": ["image_std", "category", "flags", "alpha_fail", "snapshot"],
        "decide": ["score", "lighting_shift", "evidence", "t_pass", "t_fail", "budget_state"],
    }
    for name, params in expected.items():
        fn = getattr(I, name)
        got = list(inspect.signature(fn).parameters)
        assert got[: len(params)] == params or got == params[: len(got)], (name, got)
    assert set(expected) <= set(I.__all__)


def test_engine_stubs_fail_loudly_until_agent_exists():
    from core import interfaces as I

    try:
        from core import agent  # noqa: F401
        pytest.skip("core.agent exists; stub behaviour no longer applies")
    except ImportError:
        pass
    with pytest.raises(NotImplementedError):
        I.inspect_standardized(np.zeros((8, 8, 3), np.uint8), "pcb1", {}, 0.05, None)
    with pytest.raises(NotImplementedError):
        I.decide(1.0, False, None, 1.0, 2.0, None)
