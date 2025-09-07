import os
import sys
import numpy as np

# Ensure project root on path so we can import the script module
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.evaluate_network_anomaly_detector import (
    best_f1_threshold,
    precision_at_k,
    detection_delays,
)

def test_best_f1_threshold_basic():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    th, stats = best_f1_threshold(y, s)
    assert 0.2 <= th <= 0.9
    assert 0 <= stats["precision"] <= 1
    assert 0 <= stats["recall"] <= 1


def test_precision_at_k():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.8, 0.2])
    p_at_2 = precision_at_k(y, s, 2)
    assert p_at_2 == 1.0


def test_detection_delays_no_detection():
    y = np.array([0, 0, 1, 1, 0, 0])
    s = np.array([0, 0, 0, 0, 0, 0])
    delays, mean_d, p95_d = detection_delays(y, s, 0.5)
    assert delays == []
    assert np.isnan(mean_d)
    assert np.isnan(p95_d)
