"""
Evaluate NetworkAnomalyDetector on synthetic data with injected anomalies.

This script generates realistic network metric streams, injects different
anomaly types, feeds them to NetworkAnomalyDetector, and computes metrics:
- AUROC, Average Precision (PR-AUC)
- Best-threshold F1/Precision/Recall (grid search)
- Precision@K, False Positives per hour
- Detection delay (mean, p95)
- Inference latency (mean, p95)

It optionally logs results to MLflow (if tracking server reachable), else falls
back to local file-based tracking under ./mlruns.

Usage:
  python scripts/evaluate_network_anomaly_detector.py --steps 600 --seed 42 \
         --mlflow true --exp "anomaly-detector-eval"
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# Ensure project root is on sys.path for local execution
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Local imports
from agent.network_monitor.network_anomaly_detector import NetworkAnomalyDetector


# ---------------------------- Simulation ----------------------------

@dataclass
class InjectionWindow:
    start: int
    end: int
    kind: str  # "bandwidth" | "latency" | "connections" | "packet_loss"


def _make_baseline_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _simulate_step(
    t: int,
    rng: np.random.Generator,
    recv_base: float,
    send_base: float,
    conn_base: int,
    latency_base: float,
    injections: List[InjectionWindow],
) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
    """Generate one timestep of metrics and labels.

    Returns:
      metrics_dict: as expected by NetworkAnomalyDetector.analyze_network_metrics
      y: binary label if any anomaly injected at t
      y_by_type: dict of per-type binary labels
    """
    noise = lambda s: rng.normal(0, s)

    # Baseline stats (bytes/s)
    bytes_recv_rate = max(0.0, recv_base + noise(recv_base * 0.05))
    bytes_sent_rate = max(0.0, send_base + noise(send_base * 0.06))
    packets_recv_rate = max(10.0, recv_base / 800.0 + noise(5))
    packets_sent_rate = max(10.0, send_base / 1000.0 + noise(5))

    # Connections
    total_connections = max(5, int(conn_base + noise(max(3, conn_base * 0.05))))
    tcp_established = int(max(1, total_connections * 0.7 + noise(2)))
    tcp_listen = int(max(1, total_connections * 0.1 + noise(1)))
    tcp_time_wait = int(max(0, total_connections * 0.05 + noise(1)))
    tcp_close_wait = int(max(0, total_connections * 0.02 + noise(1)))
    udp_connections = int(max(0, total_connections * 0.13 + noise(1)))

    # Latency (ms)
    g_dns = max(5.0, latency_base + noise(2))
    cf_dns = max(5.0, latency_base * 0.9 + noise(2))
    gw = max(2.0, latency_base * 0.5 + noise(1.5))

    # Packet errors/drops in %
    error_rate = max(0.0, 0.1 + abs(noise(0.02)))
    drop_rate = max(0.0, 0.1 + abs(noise(0.02)))

    # Determine active injections
    active_types: List[str] = [inj.kind for inj in injections if inj.start <= t < inj.end]
    y_by_type = {k: 1 if k in active_types else 0 for k in [
        "bandwidth", "latency", "connections", "packet_loss"
    ]}
    y = 1 if any(y_by_type.values()) else 0

    # Apply anomalies
    if y_by_type["bandwidth"]:
        # Big recv spike 4-8x
        factor = rng.uniform(4.0, 8.0)
        bytes_recv_rate *= factor
        packets_recv_rate *= min(10.0, factor * 1.5)
    if y_by_type["connections"]:
        # Sudden connection spike 3-10x
        factor = rng.uniform(3.0, 10.0)
        total_connections = int(total_connections * factor)
        tcp_established = int(tcp_established * factor)
    if y_by_type["latency"]:
        # Latency surge 4-10x (DNS/gateway)
        factor = rng.uniform(4.0, 10.0)
        g_dns *= factor
        cf_dns *= min(factor, 6.0)
        gw *= min(factor, 4.0)
    if y_by_type["packet_loss"]:
        error_rate = rng.uniform(3.0, 10.0)
        drop_rate = rng.uniform(3.0, 10.0)

    metrics: Dict[str, Any] = {
        "bandwidth_stats": {
            "eth0": {
                "bytes_sent": 0,  # absolute counters unused in detector
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0,
                "errin": 0,
                "errout": 0,
                "dropin": 0,
                "dropout": 0,
                "bytes_sent_rate": float(bytes_sent_rate),
                "bytes_recv_rate": float(bytes_recv_rate),
                "packets_sent_rate": float(packets_sent_rate),
                "packets_recv_rate": float(packets_recv_rate),
            }
        },
        "connections": {
            "total_connections": int(total_connections),
            "tcp_established": int(tcp_established),
            "tcp_listen": int(tcp_listen),
            "tcp_time_wait": int(tcp_time_wait),
            "tcp_close_wait": int(tcp_close_wait),
            "udp_connections": int(udp_connections),
        },
        "latency_metrics": {
            "google_dns_latency_ms": float(g_dns),
            "cloudflare_dns_latency_ms": float(cf_dns),
            "local_gateway_latency_ms": float(gw),
        },
        "packet_stats": {
            "overall": {
                "packets_sent": 0,
                "packets_recv": 0,
                "errors_in": 0,
                "errors_out": 0,
                "drops_in": 0,
                "drops_out": 0,
                "error_rate": float(error_rate),
                "drop_rate": float(drop_rate),
            }
        },
    }

    return metrics, y, y_by_type


def _generate_injections(steps: int, rng: np.random.Generator) -> List[InjectionWindow]:
    """Create several non-overlapping anomaly windows across timeline."""
    windows: List[InjectionWindow] = []
    kinds = ["bandwidth", "latency", "connections", "packet_loss"]
    for k in kinds:
        # 2 windows per kind
        for _ in range(2):
            dur = int(max(10, steps * rng.uniform(0.03, 0.08)))
            start = int(rng.integers(low=steps // 10, high=max(steps - dur - 1, steps // 2)))
            windows.append(InjectionWindow(start=start, end=start + dur, kind=k))
    # Sort by start time
    windows.sort(key=lambda w: w.start)
    return windows


# ---------------------------- Metrics ----------------------------

def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision, recall arrays include an extra point at threshold=inf; align lengths
    f1s = []
    threshs = []
    for p, r, th in zip(precision[:-1], recall[:-1], thresholds):
        f1 = (2 * p * r) / (p + r + 1e-12)
        f1s.append(f1)
        threshs.append(th)
    if not f1s:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    idx = int(np.argmax(f1s))
    th_star = float(threshs[idx])
    p_star = float(precision[idx])
    r_star = float(recall[idx])
    f_star = float(f1s[idx])
    return th_star, {"f1": f_star, "precision": p_star, "recall": r_star}


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(scores)))
    idx = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[idx]))


def detection_delays(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[List[int], float, float]:
    """Compute delays in steps from anomaly start to first detection."""
    detections = scores >= threshold
    delays: List[int] = []
    i = 0
    n = len(y_true)
    while i < n:
        if y_true[i] == 1 and (i == 0 or y_true[i - 1] == 0):
            # anomaly starts at i; search first detection >= i
            j = i
            while j < n and y_true[j] == 1 and not detections[j]:
                j += 1
            if j < n and detections[j]:
                delays.append(j - i)
            else:
                # no detection during this window; optionally search few steps after
                k = j
                while k < min(n, j + 20) and not detections[k]:
                    k += 1
                if k < min(n, j + 20) and detections[k]:
                    delays.append(k - i)
        i += 1
    if delays:
        mean_d = float(np.mean(delays))
        p95_d = float(np.percentile(delays, 95))
    else:
        mean_d = p95_d = float("nan")
    return delays, mean_d, p95_d


def false_positives_per_hour(y_true: np.ndarray, preds: np.ndarray, steps_per_sec: float) -> float:
    # Count positive predictions on non-anomalous timesteps
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    seconds = len(y_true) / steps_per_sec
    hours = max(1e-9, seconds / 3600.0)
    return float(fp / hours)


# ---------------------------- MLflow ----------------------------

def _setup_mlflow(experiment: str, enable: bool) -> Tuple[bool, str]:
    if not enable:
        return False, ""
    import mlflow
    # Try config from env or file
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        # Try config file
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "mlflow.yaml")
        cfg_path = os.path.abspath(cfg_path)
        try:
            with open(cfg_path, "r") as f:
                data = yaml.safe_load(f)
                tracking_uri = str(data.get("tracking_uri", ""))
        except Exception:
            tracking_uri = ""
    # Fallback to local file store
    if not tracking_uri:
        tracking_uri = "file:./mlruns"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        return True, tracking_uri
    except Exception:
        # Final fallback to file store
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(experiment)
            return True, "file:./mlruns"
        except Exception:
            return False, ""


# ---------------------------- Main Eval ----------------------------

def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    rng = _make_baseline_rng(args.seed)
    steps = int(args.steps)
    step_dt = float(args.step_dt)

    # Baselines
    recv_base = float(args.recv_base)
    send_base = float(args.send_base)
    conn_base = int(args.conn_base)
    latency_base = float(args.latency_base)

    # Detector config
    det = NetworkAnomalyDetector(
        {
            "history_size": max(100, int(steps * 0.25)),
            "baseline_samples": max(30, int(steps * 0.1)),
            "sensitivity": float(args.sensitivity),
        }
    )

    injections = _generate_injections(steps, rng)

    y: List[int] = []
    scores: List[float] = []
    latency_ms: List[float] = []

    # Simulate stream
    for t in range(steps):
        m, label, _ = _simulate_step(
            t,
            rng,
            recv_base,
            send_base,
            conn_base,
            latency_base,
            injections,
        )
        t0 = time.perf_counter()
        anomalies = det.analyze_network_metrics(m)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latency_ms.append(dt_ms)

        # Score: max confidence in this step
        if anomalies:
            s = max(float(getattr(a, "confidence", 0.0)) for a in anomalies)
        else:
            s = 0.0
        y.append(label)
        scores.append(s)

        # Optional: emulate real-time spacing (disabled by default)
        if args.realtime:
            time.sleep(step_dt)

    y_arr = np.array(y, dtype=int)
    s_arr = np.array(scores, dtype=float)

    # Metrics
    results: Dict[str, Any] = {}
    try:
        results["auroc"] = float(roc_auc_score(y_arr, s_arr))
    except Exception:
        results["auroc"] = float("nan")
    try:
        results["avg_precision"] = float(average_precision_score(y_arr, s_arr))
    except Exception:
        results["avg_precision"] = float("nan")

    th_star, pr_at_star = best_f1_threshold(y_arr, s_arr)
    results.update({
        "best_threshold": float(th_star),
        "f1_best": pr_at_star["f1"],
        "precision_best": pr_at_star["precision"],
        "recall_best": pr_at_star["recall"],
    })

    # Precision@K
    for k in (10, 20, 50):
        results[f"precision_at_{k}"] = precision_at_k(y_arr, s_arr, k)

    # Detection delay and FP/hr at threshold
    preds = (s_arr >= th_star).astype(int)
    delays, delay_mean, delay_p95 = detection_delays(y_arr, s_arr, th_star)
    results["delay_mean_steps"] = delay_mean
    results["delay_p95_steps"] = delay_p95
    results["false_positives_per_hour"] = false_positives_per_hour(y_arr, preds, 1.0 / step_dt)

    # Inference latency
    lat = np.array(latency_ms)
    results["inference_latency_ms_mean"] = float(np.mean(lat))
    results["inference_latency_ms_p95"] = float(np.percentile(lat, 95))

    # Attach run context
    results["steps"] = steps
    results["step_dt"] = step_dt
    results["seed"] = int(args.seed)
    results["baseline_recv_Bps"] = recv_base
    results["baseline_send_Bps"] = send_base
    results["baseline_connections"] = conn_base
    results["baseline_latency_ms"] = latency_base
    results["sensitivity"] = float(args.sensitivity)

    return {"y": y_arr, "scores": s_arr, "latency_ms": latency_ms, "results": results}


def _log_mlflow(run_data: Dict[str, Any], experiment: str, enable: bool) -> None:
    ok, uri = _setup_mlflow(experiment, enable)
    if not ok:
        return
    import mlflow
    y = run_data["y"]
    s = run_data["scores"]
    metrics = run_data["results"]
    with mlflow.start_run():
        # Log params
        param_keys = [
            "steps",
            "step_dt",
            "seed",
            "baseline_recv_Bps",
            "baseline_send_Bps",
            "baseline_connections",
            "baseline_latency_ms",
            "sensitivity",
        ]
        for k in param_keys:
            mlflow.log_param(k, metrics.get(k))

        # Log metrics
        metric_keys = [
            "auroc",
            "avg_precision",
            "best_threshold",
            "f1_best",
            "precision_best",
            "recall_best",
            "precision_at_10",
            "precision_at_20",
            "precision_at_50",
            "delay_mean_steps",
            "delay_p95_steps",
            "false_positives_per_hour",
            "inference_latency_ms_mean",
            "inference_latency_ms_p95",
        ]
        for k in metric_keys:
            v = metrics.get(k)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                continue
            mlflow.log_metric(k, float(v))

        # Curves as artifacts
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

            # PR curve
            fig, ax = plt.subplots(figsize=(5, 4))
            PrecisionRecallDisplay.from_predictions(y, s, ax=ax)
            ax.set_title("Precision-Recall")
            fig.tight_layout()
            mlflow.log_figure(fig, "pr_curve.png")
            plt.close(fig)

            # ROC curve
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y, s, ax=ax)
            ax.set_title("ROC Curve")
            fig.tight_layout()
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

            # Score timeline
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(s, label="score")
            ax.plot(y, label="label", alpha=0.6)
            ax.set_title("Score vs. Label over Time")
            ax.set_xlabel("Step")
            ax.legend()
            fig.tight_layout()
            mlflow.log_figure(fig, "timeline.png")
            plt.close(fig)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NetworkAnomalyDetector on synthetic data")
    p.add_argument("--steps", type=int, default=600, help="Number of timesteps")
    p.add_argument("--step-dt", dest="step_dt", type=float, default=1.0, help="Seconds per step (simulation)")
    p.add_argument("--recv-base", dest="recv_base", type=float, default=500_000.0, help="Baseline recv bytes/s")
    p.add_argument("--send-base", dest="send_base", type=float, default=200_000.0, help="Baseline send bytes/s")
    p.add_argument("--conn-base", dest="conn_base", type=int, default=80, help="Baseline total connections")
    p.add_argument("--latency-base", dest="latency_base", type=float, default=15.0, help="Baseline latency ms")
    p.add_argument("--sensitivity", type=float, default=0.8, help="Detector sensitivity (0-1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    p.add_argument("--exp", type=str, default="anomaly-detector-eval", help="MLflow experiment name")
    p.add_argument("--realtime", action="store_true", help="Sleep between steps by step_dt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_eval(args)
    # Print summary
    results = out["results"]
    summary_keys = [
        "auroc",
        "avg_precision",
        "f1_best",
        "precision_best",
        "recall_best",
        "best_threshold",
        "precision_at_10",
        "delay_mean_steps",
        "false_positives_per_hour",
        "inference_latency_ms_mean",
    ]
    print("Evaluation Summary:")
    for k in summary_keys:
        print(f"  {k}: {results.get(k)}")

    # MLflow logging
    _log_mlflow(out, args.exp, args.mlflow)


if __name__ == "__main__":
    main()
