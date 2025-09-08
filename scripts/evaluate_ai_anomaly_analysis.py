"""
AI-focused evaluation of the LLM anomaly analysis pipeline.

This script feeds synthetic time-series metrics into AnomalyDetector (server.ai_engine)
and measures the quality of the final AI analysis results:

- Structural validity (schema adherence already enforced by Pydantic)
- Findings coverage (recall of injected anomalies by metric)
- Precision (spurious findings not tied to any injected anomaly)
- Severity agreement (within 1 level of expected)
- Runtime latency (end-to-end)
- Counts and distribution of recommendations

If a Vertex AI endpoint is configured (VERTEX_ENDPOINT_ID set), it exercises the
real LLM. Otherwise, the detector falls back to a heuristic analysis; we still
compute structural and timing metrics, marking mode=fallback.

Logged to MLflow (uses MLFLOW_TRACKING_URI or falls back to file:./mlruns).

Usage:
  python scripts/evaluate_ai_anomaly_analysis.py --steps 180 --seed 13 --mlflow true \
         --exp ai-anomaly-eval
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo root on path
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.ai_engine.anomaly_detector import AnomalyDetector  # type: ignore


# ---------------------------- Synthetic data ----------------------------

@dataclass
class Inject:
    start: int
    end: int
    metric: str  # one of AnomalyDetector.key_metrics
    expected_type: str  # spike/drop/trend_change
    expected_severity: str  # low/medium/high/critical


SEV2NUM = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def gen_injections(steps: int, rng: np.random.Generator) -> List[Inject]:
    inj: List[Inject] = []
    # Choose a few metrics to perturb
    candidates = [
        ("total_connections", "spike"),
        ("google_dns_latency_ms", "spike"),
        ("cloudflare_dns_latency_ms", "spike"),
        ("local_gateway_latency_ms", "spike"),
        ("cpu_percent", "spike"),
    ]
    for metric, etype in candidates:
        dur = int(max(8, steps * rng.uniform(0.05, 0.12)))
        start = int(rng.integers(low=steps // 10, high=max(steps - dur - 1, steps // 2)))
        sev = np.random.choice(["medium", "high", "critical"], p=[0.4, 0.4, 0.2])
        inj.append(Inject(start=start, end=start + dur, metric=metric, expected_type=etype, expected_severity=sev))
    return inj


def synth_raw_metrics(steps: int, injections: List[Inject], seed: int) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    raw: List[Dict[str, Any]] = []
    gt_by_step: Dict[int, Dict[str, Any]] = {}

    # Baselines
    cpu_b = 22.0
    mem_b = 55.0
    disk_b = 40.0
    conn_b = 90
    lat_g = 18.0
    lat_cf = 16.0
    lat_gw = 8.0

    def active_injections(t: int) -> List[Inject]:
        return [inj for inj in injections if inj.start <= t < inj.end]

    for t in range(steps):
        act = active_injections(t)
        gt_by_step[t] = {"metrics": [i.metric for i in act]}
        # Noise helpers
        n = lambda s: rng.normal(0, s)

        cpu = max(0.0, cpu_b + n(2.0))
        mem = max(0.0, mem_b + n(1.5))
        disk = max(0.0, disk_b + n(1.0))
        total_conn = max(5, int(conn_b + n(5)))
        g = max(5.0, lat_g + n(1.2))
        cf = max(5.0, lat_cf + n(1.0))
        gw = max(2.0, lat_gw + n(0.6))

        # Apply spikes
        for inj in act:
            if inj.metric == "cpu_percent":
                cpu *= {"medium": 1.6, "high": 2.2, "critical": 3.0}[inj.expected_severity]
            elif inj.metric == "total_connections":
                total_conn = int(total_conn * {"medium": 2.0, "high": 3.0, "critical": 5.0}[inj.expected_severity])
            elif inj.metric == "google_dns_latency_ms":
                g *= {"medium": 3.0, "high": 5.0, "critical": 8.0}[inj.expected_severity]
            elif inj.metric == "cloudflare_dns_latency_ms":
                cf *= {"medium": 2.5, "high": 4.0, "critical": 6.0}[inj.expected_severity]
            elif inj.metric == "local_gateway_latency_ms":
                gw *= {"medium": 2.0, "high": 3.0, "critical": 4.0}[inj.expected_severity]

        raw.append(
            {
                "system_metrics": {
                    "cpu_percent": float(cpu),
                    "memory_percent": float(mem),
                    "disk_percent": float(disk),
                },
                "connections": {
                    "total_connections": int(total_conn),
                },
                "latency_metrics": {
                    "google_dns_latency_ms": float(g),
                    "cloudflare_dns_latency_ms": float(cf),
                    "local_gateway_latency_ms": float(gw),
                },
            }
        )

    return raw, gt_by_step


# ---------------------------- Metrics ----------------------------

def coverage_and_precision(
    findings: List[Dict[str, Any]],
    injections: List[Inject],
) -> Tuple[float, float]:
    """Compute recall (coverage) and precision at the finding level by metric name.
    A finding counts as correct if it references a metric with an injected spike.
    """
    inj_metrics = {inj.metric for inj in injections}
    if not findings:
        return 0.0, 0.0
    tp = 0
    for f in findings:
        m = f.get("metric_name") or ""
        if m in inj_metrics:
            tp += 1
    fn = len(inj_metrics) - len({f.get("metric_name") for f in findings if (f.get("metric_name") or "") in inj_metrics})
    fp = max(0, len(findings) - tp)
    recall = float(tp / max(1, tp + fn))
    precision = float(tp / max(1, tp + fp))
    return recall, precision


def severity_agreement(findings: List[Dict[str, Any]], injections: List[Inject]) -> float:
    """Measure severity agreement: fraction of matched findings with |pred - exp| <= 1 level."""
    exp_by_metric: Dict[str, int] = {}
    for inj in injections:
        exp_by_metric[inj.metric] = max(exp_by_metric.get(inj.metric, -1), SEV2NUM.get(inj.expected_severity, 0))
    ok = 0
    total = 0
    for f in findings:
        m = f.get("metric_name") or ""
        if m in exp_by_metric:
            total += 1
            pred = SEV2NUM.get(str(f.get("severity", "low")).lower(), 0)
            if abs(pred - exp_by_metric[m]) <= 1:
                ok += 1
    return float(ok / max(1, total))


# ---------------------------- MLflow helpers ----------------------------

def setup_mlflow(exp: str, enable: bool) -> bool:
    if not enable:
        return False
    try:
        import mlflow
        # Use a clean local store by default to avoid existing malformed experiments
        uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns_eval")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(exp)
        return True
    except Exception:
        return False


async def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    injections = gen_injections(args.steps, rng)
    raw, _ = synth_raw_metrics(args.steps, injections, args.seed + 1)

    config = {
        "ai": {
            "anomaly_detection": {"window_size": args.steps, "sensitivity": 0.8, "min_samples": 25}
        }
    }
    det = AnomalyDetector(config)

    t0 = time.perf_counter()
    result = await det.analyze_agent_metrics(agent_id=1, hostname="ai-eval-host", raw_metrics=raw)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Determine mode (LLM vs fallback) by checking summary field default and finding confidences
    ai_mode = "llm" if os.getenv("VERTEX_ENDPOINT_ID") else "fallback"

    # Coerce to dicts for metrics
    findings = [
        {
            "metric_name": f.metric_name,
            "anomaly_type": f.anomaly_type,
            "severity": f.severity,
            "confidence": float(f.confidence),
        }
        for f in (result.findings or [])
    ]

    recs = [
        {
            "category": r.category,
            "priority": r.priority,
        }
        for r in (result.recommendations or [])
    ]

    recall, precision = coverage_and_precision(findings, injections)
    sev_agree = severity_agreement(findings, injections)

    out = {
        "ai_mode": ai_mode,
        "latency_ms": dt_ms,
        "findings_count": len(findings),
        "recommendations_count": len(recs),
        "recall_by_metric": recall,
        "precision_by_metric": precision,
        "severity_agreement": sev_agree,
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AI anomaly analysis (LLM)")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--mlflow", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    p.add_argument("--exp", type=str, default="ai-anomaly-eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = asyncio.run(run_once(args))

    print("AI Anomaly Analysis Evaluation:")
    for k in [
        "ai_mode",
        "latency_ms",
        "findings_count",
        "recommendations_count",
        "recall_by_metric",
        "precision_by_metric",
        "severity_agreement",
    ]:
        print(f"  {k}: {res.get(k)}")

    if setup_mlflow(args.exp, args.mlflow):
        try:
            import mlflow
            with mlflow.start_run():
                for k, v in res.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                    else:
                        mlflow.log_param(k, v)
        except Exception:
            pass


if __name__ == "__main__":
    main()
