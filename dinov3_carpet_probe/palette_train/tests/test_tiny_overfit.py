from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Tiny-overfit gate is exercised by the dedicated run_tiny_overfit.py script.")
def test_tiny_overfit_gate_placeholder():
    assert True
