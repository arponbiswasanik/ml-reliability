"""Unit tests: PSI computation and drift classification."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring import classify, psi


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, size=5000)
    shifted = rng.normal(2, 1, size=5000)          # 2-sigma shift
    assert psi(ref, shifted) > 0.25                # must alarm


def test_psi_out_of_range_lands_in_edge_bins():
    ref = np.linspace(0, 1, 1000)
    far = np.full(500, 50.0)                        # all beyond reference max
    assert psi(ref, far) > 0.25


def test_psi_constant_reference_is_zero():
    assert psi(np.full(100, 5.0), np.full(100, 7.0)) == 0.0


def test_classify_thresholds():
    assert classify(0.05) == "ok"
    assert classify(0.15) == "warn"
    assert classify(0.25) == "alarm"