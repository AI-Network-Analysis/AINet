# Evaluation & Performance Metrics

This document describes how we evaluate AINet with an emphasis on the AI anomaly analysis for academic reporting.

## Scope
Primary focus: AI anomaly analysis quality, not raw packet capture.

Components under evaluation:
- AI pipeline: `server/ai_engine/anomaly_detector.py` (LLM-assisted analysis or heuristic fallback)
- Statistical baseline: `agent/network_monitor/network_anomaly_detector.py` (for discrimination metrics)

Data: synthetic metric streams with injected anomalies (bandwidth, latency, connections, packet loss). No packet sniffing evaluation is performed.

Outputs: structured findings (metric, type, severity, confidence), recommendations, risk summaries.

## Metrics
- Discrimination
  - AUROC: overall ranking quality.
  - Average Precision (PR-AUC): robust to class imbalance.
- Operating point quality (picked by best F1 via grid):
  - F1, Precision, Recall
  - Precision@K (K ∈ {10, 20, 50})
- Stability and timeliness
  - Detection Delay (mean and p95 in steps)
  - False Positives per hour (at best-F1 threshold)
- Performance
  - Inference latency (mean, p95 ms per timestep)

## Methodology
1. Generate a baseline traffic process: bytes/sec, connections, latencies, and packet error rates.
2. Inject multiple windows per anomaly type with controlled magnitudes and durations.
3. Stream metrics to the detector; record max confidence per step as the anomaly score.
4. Compute metrics listed above and plot PR/ROC/timeline curves.
5. Log to MLflow for experiment tracking.

## Reproducibility
- Random seeds fixed (`--seed`), baseline parameters (`--recv-base`, `--send-base`, etc.) documented and logged.
- MLflow logs: params, metrics, and artifacts (plots) under experiment `anomaly-detector-eval` by default.
- Tracking URI: takes `MLFLOW_TRACKING_URI`; if not set, falls back to `file:./mlruns`.

## How to run
### AI anomaly analysis (LLM or fallback)
```
python scripts/evaluate_ai_anomaly_analysis.py --steps 180 --seed 13 --mlflow true
```
- If `VERTEX_ENDPOINT_ID` is set, the LLM is used; otherwise the detector uses its fallback heuristic.
- Metrics: findings coverage/precision (by metric), severity agreement, end-to-end latency, recommendation counts.

### Statistical baseline (optional)
```
python scripts/evaluate_network_anomaly_detector.py --steps 600 --seed 42 --mlflow true
```
Optional flags:
- `--step-dt 1.0` seconds per simulated step (no real-time sleep by default)
- `--sensitivity 0.8` detector sensitivity
- `--exp my-experiment` MLflow experiment name

## Interpretation
- AUROC/PR-AUC close to 1.0 indicate strong separation.
- High F1/Precision/Recall at the best threshold show useful operating points.
- Low FP/hour is preferable for operations.
- Short detection delays (mean/p95) indicate responsiveness.
- Latency stats gauge efficiency and capacity planning.
 - For the AI layer, high coverage and precision mean it correctly captures injected anomalies without over-reporting; severity agreement reflects calibration.

## Limitations
- Synthetic data approximates realistic behavior but cannot capture full production diversity.
- In fallback mode (no Vertex LLM), analysis quality reflects heuristics only.
- Confidence values may need calibration; learned calibrators could improve probabilities.

## Next steps
- Benchmark on real labeled incidents and production telemetry.
- Calibrate confidence to probability via Platt/Isotonic regression.
- Add per-type AI metrics and error analysis (missed vs spurious findings).
- Human-in-the-loop evaluation for recommendation usefulness and clarity.
