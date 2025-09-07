"""
Fine-tuning dashboard and API
Separate UI to upload JSONL and trigger Vertex AI fine-tuning with MLflow tracking.
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
import mlflow

from ..mlops.pipelines.vertex_ai_mlops_pipeline import (
    VertexAIMLOpsManager,
    MLOpsPipeline,
    ModelConfig,
)
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

load_dotenv()

finetune_router = APIRouter(prefix="/finetune", tags=["Fine-tune"])


def _build_manager() -> VertexAIMLOpsManager:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEXAI_LOCATION", "us-central1")
    tracking = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    bucket = os.getenv("GCS_STAGING_BUCKET")
    if not project:
        raise HTTPException(status_code=500, detail="GOOGLE_CLOUD_PROJECT is not set")
    if not bucket:
        raise HTTPException(status_code=500, detail="GCS_STAGING_BUCKET is not set")
    return VertexAIMLOpsManager(
        project_id=project,
        location=location,
        mlflow_tracking_uri=tracking,
        credentials_path=creds,
        staging_bucket=bucket,
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "vertex-ai-llm-finetuning"),
    )


def _ensure_mlflow_tracking():
    """Set MLflow tracking URI explicitly to keep client/server in sync."""
    tracking = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(tracking)
    except Exception:
        pass


@finetune_router.get("/", response_class=HTMLResponse)
async def finetune_page() -> HTMLResponse:
    """Serve the separate fine-tuning dashboard HTML."""
    templates_dir = Path(__file__).parents[2] / "dashboard" / "templates"
    html_path = templates_dir / "finetune.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())

    # Fallback minimal UI
    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html>
        <head><title>Fine-tuning Dashboard</title></head>
        <body>
          <h1>Fine-tuning Dashboard</h1>
          <p>Upload a JSONL dataset and trigger fine-tuning.</p>
          <form id="form" enctype="multipart/form-data">
            <label>Dataset Name <input name="dataset_name" required /></label><br/>
            <label>Base Model <input name="base_model" value="gemini-1.0-pro-002" /></label><br/>
            <label>Training Steps <input name="training_steps" value="100" type="number" min="1" /></label><br/>
            <input type="file" name="file" accept=".jsonl" required />
            <button type="submit">Start Fine-tuning</button>
          </form>
          <pre id="out"></pre>
          <script>
            const form = document.getElementById('form');
            form.addEventListener('submit', async (e) => {
              e.preventDefault();
              const fd = new FormData(form);
              const res = await fetch('/finetune/upload', { method:'POST', body: fd });
              const data = await res.json();
              document.getElementById('out').textContent = JSON.stringify(data, null, 2);
            });
          </script>
        </body>
        </html>
        """,
    )


@finetune_router.post("/upload")
async def upload_and_start(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    base_model: str = Form("gemini-1.0-pro-002"),
    training_steps: int = Form(100),
) -> JSONResponse:
    """Upload local JSONL, push to GCS, and trigger a tuning job."""
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

    try:
        _ensure_mlflow_tracking()
        manager = _build_manager()
        pipeline = MLOpsPipeline(manager)

        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Upload + start job
        ds_cfg = manager.upload_and_version_dataset(tmp_path, dataset_name)
        job_id, run_id = manager.create_fine_tuning_job(
            ds_cfg, ModelConfig(base_model=base_model, training_steps=int(training_steps))
        )

        # Attach the dataset file as an artifact to the tuning run as well
        try:
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
            with mlflow.start_run(run_id=run_id):
                # Log only if small enough (<100MB)
                if os.path.getsize(tmp_path) <= 100 * 1024 * 1024:
                    mlflow.log_artifact(tmp_path, artifact_path=f"datasets/{dataset_name}/{ds_cfg.version}")
        except Exception:
            pass

        # Construct MLflow run URL if local UI
        tracking = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").rstrip("/")
        mlflow_url: Optional[str] = None
        try:
            from mlflow.tracking import MlflowClient
            _ensure_mlflow_tracking()
            client = MlflowClient()
            exp = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT", "vertex-ai-llm-finetuning"))
            if exp:
                mlflow_url = f"{tracking}/#/experiments/{exp.experiment_id}/runs/{run_id}"
        except Exception:
            pass

        return JSONResponse(
            {
                "status": "started",
                "dataset": ds_cfg.__dict__,
                "job_id": job_id,
                "mlflow_run_id": run_id,
                "mlflow_url": mlflow_url,
            }
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to start fine-tuning")
        raise HTTPException(status_code=500, detail=str(e))


@finetune_router.get("/status")
async def job_status(job_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch latest job state and echo MLflow link if available."""
    try:
        _ensure_mlflow_tracking()
        manager = _build_manager()
        res = manager.track_training_metrics(job_id, run_id) if run_id else {"job_state": "UNKNOWN"}

        # If succeeded, ensure a model version is registered in MLflow Models
        if run_id and res.get("job_state") == "JOB_STATE_SUCCEEDED":
            tuned_model_resource = res.get("model_resource_name")
            _ensure_mlflow_tracking()
            client = MlflowClient()
            run = client.get_run(run_id)
            params = run.data.params or {}
            dataset_name = params.get("dataset_name", "vertex_dataset")
            dataset_version = params.get("version") or params.get("dataset_version")
            base_model = params.get("base_model")
            model_registry_name = f"{dataset_name}_vertex_sft"

            # Avoid duplicate registration by checking for an existing version tagged with this tuning job id
            already = False
            try:
                versions = client.search_model_versions(f"name='{model_registry_name}'")
                for v in versions:
                    tags = getattr(v, "tags", {}) or {}
                    if tags.get("vertex_tuning_job_id") == job_id:
                        already = True
                        break
            except Exception:
                pass

            if not already:
                try:
                    manager.create_mlflow_model_version(
                        model_name=model_registry_name,
                        endpoint_id=None,
                        vertex_model_resource_name=tuned_model_resource,
                        job_id=job_id,
                        tags={
                            "dataset_version": dataset_version or "unknown",
                            "base_model": base_model or "unknown",
                            "registered_via": "finetune_status",
                        },
                    )
                    res["model_registered"] = True
                    res["model_name"] = model_registry_name
                except Exception:
                    res["model_registered"] = False

        return res
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to fetch status")
        raise HTTPException(status_code=500, detail=str(e))
