"""
MLOps Pipeline for Vertex AI Fine-tuned Models with MLflow Integration

This pipeline provides:
1. Model tracking and versioning with MLflow
2. Dataset versioning and management
3. Continuous fine-tuning capabilities
4. Model deployment automation
5. Performance monitoring
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import logging

import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from google.cloud import aiplatform, storage
from google.cloud.aiplatform import Model, Endpoint, CustomJob
from google.oauth2 import service_account
import vertexai
from vertexai.preview.tuning import sft
from vertexai.preview import tuning as vpt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training and deployment"""
    base_model: str = "gemini-1.0-pro-002"  # current tunable model
    training_steps: int = 100
    learning_rate: float = 0.001
    batch_size: int = 8
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    dataset_name: str
    gcs_path: str
    version: str
    split_ratio: float = 0.8
    validation_split: float = 0.1


class VertexAIMLOpsManager:
    """Manages MLOps pipeline for Vertex AI fine-tuned models"""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        staging_bucket: str | None = None,
        mlflow_tracking_uri: str | None = None,
        experiment_name: str = "vertex-ai-finetuning",
        credentials_path: str | None = None,
    ):
        self.project_id = project_id
        self.location = location
        # Resolve staging bucket strictly from param or environment; no hardcoded fallback
        _bucket = staging_bucket or os.getenv("GCS_STAGING_BUCKET")
        if not _bucket:
            raise ValueError(
                "GCS staging bucket not configured. Set env GCS_STAGING_BUCKET or pass staging_bucket."
            )
        # Accept with or without gs:// prefix; store as bucket name only
        self.staging_bucket = _bucket.replace("gs://", "").strip("/")
        self._credentials = None

        # Resolve Google credentials
        cred_file = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_file:
            if os.path.exists(cred_file):
                try:
                    self._credentials = service_account.Credentials.from_service_account_file(cred_file)
                    logger.info("Loaded Google credentials from file.")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to load credentials from {cred_file}: {e}")
            else:
                logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS path does not exist: {cred_file}")

        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location,
            staging_bucket=f"gs://{self.staging_bucket}",
            credentials=self._credentials,
        )
        # Initialize vertexai preview SDK for tuning
        try:
            vertexai.init(project=project_id, location=location, credentials=self._credentials)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"vertexai.init failed: {e}")

        # Initialize MLflow
        tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Clients
        self.mlflow_client = MlflowClient()
        self.storage_client = storage.Client(project=self.project_id, credentials=self._credentials)

    def upload_and_version_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        metadata: Dict[str, Any] | None = None,
    ) -> DatasetConfig:
        """
        Upload dataset to GCS and track version in MLflow

        Args:
            dataset_path: Local path to dataset file
            dataset_name: Name for the dataset
            metadata: Additional metadata to track

        Returns:
            DatasetConfig with versioning information
        """
        # Generate dataset version hash
        with open(dataset_path, "rb") as f:
            content = f.read()
            version_hash = hashlib.sha256(content).hexdigest()[:8]

        # Create versioned GCS path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_path = f"gs://{self.staging_bucket}/datasets/{dataset_name}/{timestamp}_{version_hash}/data.jsonl"

        # Upload to GCS
        bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(dataset_path)

        logger.info(f"Dataset uploaded to {gcs_path}")

        # Track in MLflow
        with mlflow.start_run(run_name=f"dataset_upload_{dataset_name}_{version_hash}") as _:
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("dataset_version", version_hash)
            mlflow.log_param("gcs_path", gcs_path)
            mlflow.log_param("upload_timestamp", timestamp)

            if metadata:
                for key, value in metadata.items():
                    mlflow.log_param(f"dataset_{key}", value)

            # Log dataset statistics
            df = pd.read_json(dataset_path, lines=True)
            mlflow.log_metric("dataset_size", len(df))
            mlflow.log_metric("dataset_columns", len(df.columns))

            # Log dataset artifacts: original file and lightweight previews
            try:
                # Guard against extremely large files (log up to 100MB)
                max_bytes = 100 * 1024 * 1024
                if os.path.getsize(dataset_path) <= max_bytes:
                    mlflow.log_artifact(dataset_path, artifact_path=f"datasets/{dataset_name}/{version_hash}")
                # Preview artifacts
                preview_dir = os.path.join("/tmp", f"mlflow_ds_preview_{version_hash}")
                os.makedirs(preview_dir, exist_ok=True)
                # Save head and schema
                head_path = os.path.join(preview_dir, "head.jsonl")
                df.head(50).to_json(head_path, orient="records", lines=True)
                schema_path = os.path.join(preview_dir, "schema.json")
                # Infer simple schema from first row
                first_row = df.head(1).to_dict(orient="records")
                import json as _json
                with open(schema_path, "w") as f:
                    _json.dump({"columns": list(df.columns), "sample": first_row}, f, indent=2)
                mlflow.log_artifacts(preview_dir, artifact_path=f"datasets/{dataset_name}/{version_hash}/preview")
            except Exception:
                # Non-fatal if artifact logging fails
                pass

        return DatasetConfig(dataset_name=dataset_name, gcs_path=gcs_path, version=version_hash)

    def create_fine_tuning_job(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        job_display_name: str | None = None,
    ) -> tuple[str, str]:
        """
        Create and submit a fine-tuning job to Vertex AI

        Args:
            dataset_config: Dataset configuration
            model_config: Model training configuration
            job_display_name: Display name for the job

        Returns:
            Tuple of (job_id, mlflow_run_id)
        """
        job_display_name = job_display_name or f"finetune_{dataset_config.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=job_display_name) as run:
            # Log configuration
            mlflow.log_params(asdict(model_config))
            mlflow.log_params(asdict(dataset_config))

            # Create tuning job using Supervised Fine-Tuning API
            if "bison" in model_config.base_model:
                logger.warning("Consider using a Gemini model (e.g., gemini-1.0-pro-002) for tuning.")

            tuning_job = sft.train(
                source_model=model_config.base_model,
                train_dataset=dataset_config.gcs_path,
                tuned_model_display_name=job_display_name,
                epochs=max(1, int(model_config.training_steps)),
                # Optional: learning_rate_multiplier=model_config.learning_rate,
            )

            # Track job in MLflow
            mlflow.log_param("vertex_job_id", tuning_job.name)
            # Use tag so subsequent updates don't violate MLflow param immutability
            mlflow.set_tag("vertex_job_state", "RUNNING")
            mlflow.set_tag("mlflow.runName", job_display_name)
            mlflow.set_tag("job_type", "fine_tuning")

            logger.info(f"Fine-tuning job started: {tuning_job.name}")

            return tuning_job.name, run.info.run_id

    def track_training_metrics(self, job_id: str, mlflow_run_id: str) -> Dict[str, Any]:
        """
        Track training metrics from Vertex AI job in MLflow

        Args:
            job_id: Vertex AI job ID
            mlflow_run_id: MLflow run ID to log metrics to

        Returns:
            Dictionary of final metrics
        """
        # Get tuning job via vertexai preview API
        job = vpt.TuningJob(job_id)
        job.refresh()

        with mlflow.start_run(run_id=mlflow_run_id):
            # Update job state as tag to avoid param overwrite errors
            state_name = getattr(job.state, "name", str(job.state))
            mlflow.set_tag("vertex_job_state", state_name)
            # Optional numeric metric for trend
            state_map = {
                "JOB_STATE_UNSPECIFIED": 0,
                "JOB_STATE_QUEUED": 1,
                "JOB_STATE_PENDING": 2,
                "JOB_STATE_RUNNING": 3,
                "JOB_STATE_SUCCEEDED": 4,
                "JOB_STATE_FAILED": -1,
                "JOB_STATE_CANCELLING": -2,
                "JOB_STATE_CANCELLED": -3,
                "JOB_STATE_PAUSED": -4,
                "JOB_STATE_EXPIRED": -5,
                "JOB_STATE_UPDATING": 5,
            }
            mlflow.log_metric("vertex_job_state_code", state_map.get(state_name, 0))

            if state_name == "JOB_STATE_SUCCEEDED":
                # Log completion time
                try:
                    mlflow.log_metric(
                        "training_duration_seconds",
                        (job.update_time - job.create_time).total_seconds(),
                    )
                except Exception:
                    pass

                # Log model resource name if available
                tuned_model_name = getattr(job, "tuned_model_name", None)
                if tuned_model_name:
                    mlflow.log_param("model_resource_name", tuned_model_name)

                mlflow.set_tag("training_status", "completed")

            elif state_name == "JOB_STATE_FAILED":
                mlflow.set_tag("training_status", "failed")
                if getattr(job, "error", None):
                    mlflow.log_param("error_message", str(job.error))

        result: Dict[str, Any] = {"job_state": state_name}
        if getattr(job, "tuned_model_name", None):
            result["model_resource_name"] = job.tuned_model_name
        return result

    def deploy_model(
        self,
        model_resource_name: str,
        endpoint_display_name: str | None = None,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 3,
    ) -> Tuple[str, str]:
        """
        Deploy a fine-tuned model to an endpoint

        Returns:
            Tuple of (endpoint_id, deployed_model_id)
        """
        endpoint_display_name = endpoint_display_name or f"endpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=f"deploy_{endpoint_display_name}"):
            # Get or create endpoint
            endpoints = Endpoint.list(
                filter=f'display_name="{endpoint_display_name}"',
                order_by="create_time desc",
                project=self.project_id,
                location=self.location,
            )

            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"Using existing endpoint: {endpoint.resource_name}")
            else:
                endpoint = Endpoint.create(
                    display_name=endpoint_display_name, project=self.project_id, location=self.location
                )
                logger.info(f"Created new endpoint: {endpoint.resource_name}")

            # Deploy model to endpoint
            model = Model(model_resource_name)
            deployed_model = endpoint.deploy(
                model=model,
                deployed_model_display_name=f"deployed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )

            # Log deployment info to MLflow
            mlflow.log_param("endpoint_id", endpoint.name.split("/")[-1])
            mlflow.log_param("endpoint_resource_name", endpoint.resource_name)
            mlflow.log_param("deployed_model_id", deployed_model.id)
            mlflow.log_param("model_resource_name", model_resource_name)
            mlflow.log_param("machine_type", machine_type)
            mlflow.log_metric("min_replicas", min_replica_count)
            mlflow.log_metric("max_replicas", max_replica_count)
            mlflow.set_tag("deployment_status", "active")

            # Store endpoint ID for use with your existing code
            endpoint_id = endpoint.name.split("/")[-1]
            logger.info(f"Model deployed. Endpoint ID: {endpoint_id}")

            return endpoint_id, deployed_model.id

    def create_mlflow_model_version(
        self,
        model_name: str,
        endpoint_id: str | None = None,
        vertex_model_resource_name: str | None = None,
        job_id: str | None = None,
        metrics: Dict[str, float] | None = None,
        tags: Dict[str, str] | None = None,
    ) -> str:
        """
        Register a model version in MLflow Models that tracks a Vertex AI tuned model.

        You can provide either an Endpoint ID (for live inference) or a Vertex Model
        resource name (model registry in Vertex AI). Both can be tagged for lineage.
        """
        with mlflow.start_run(run_name=f"register_{model_name}") as run:
            # Log lineage params
            if endpoint_id:
                mlflow.log_param("vertex_endpoint_id", endpoint_id)
            if vertex_model_resource_name:
                mlflow.log_param("vertex_model_resource_name", vertex_model_resource_name)
            if job_id:
                mlflow.log_param("vertex_tuning_job_id", job_id)

            # Log metrics if provided
            if metrics:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)

            # Create a custom model wrapper; prefers endpoint for inference, otherwise stores metadata only
            class VertexAIModelWrapper(mlflow.pyfunc.PythonModel):
                def __init__(
                    self,
                    project_id: str,
                    location: str,
                    endpoint_id: str | None = None,
                    vertex_model_resource_name: str | None = None,
                ):
                    self.project_id = project_id
                    self.location = location
                    self.endpoint_id = endpoint_id
                    self.vertex_model_resource_name = vertex_model_resource_name

                def predict(self, context, model_input):  # type: ignore[override]
                    # For safety, avoid making live calls by default.
                    # This wrapper captures lineage/config in predictions for traceability.
                    return {
                        "project_id": self.project_id,
                        "location": self.location,
                        "endpoint_id": self.endpoint_id,
                        "vertex_model_resource_name": self.vertex_model_resource_name,
                        "input": model_input.to_dict() if hasattr(model_input, "to_dict") else model_input,
                    }

            model_instance = VertexAIModelWrapper(
                project_id=self.project_id,
                location=self.location,
                endpoint_id=endpoint_id,
                vertex_model_resource_name=vertex_model_resource_name,
            )

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_instance,
                registered_model_name=model_name,
                conda_env={
                    "channels": ["defaults"],
                    "dependencies": [
                        "python=3.11",
                        "pip",
                        {"pip": ["mlflow", "google-cloud-aiplatform"]},
                    ],
                },
            )

            # Log lineage as an artifact for quick inspection
            try:
                mlflow.log_dict(
                    {
                        "project_id": self.project_id,
                        "location": self.location,
                        "vertex_endpoint_id": endpoint_id,
                        "vertex_model_resource_name": vertex_model_resource_name,
                        "vertex_tuning_job_id": job_id,
                    },
                    artifact_file="vertex_model_info.json",
                )
            except Exception:
                pass

            # Add tags to the registered model version
            client = MlflowClient()
            mv = client.get_latest_versions(model_name)[0]
            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(model_name, mv.version, key, value)
            # Always add helpful defaults
            if endpoint_id:
                client.set_model_version_tag(model_name, mv.version, "vertex_endpoint_id", endpoint_id)
            if vertex_model_resource_name:
                client.set_model_version_tag(
                    model_name, mv.version, "vertex_model_resource_name", vertex_model_resource_name
                )
            if job_id:
                client.set_model_version_tag(model_name, mv.version, "vertex_tuning_job_id", job_id)

            return run.info.run_id

    def monitor_endpoint_performance(
        self, endpoint_id: str, model_name: str, evaluation_dataset: str | None = None
    ) -> Dict[str, float]:
        """
        Monitor and track endpoint performance metrics
        """
        with mlflow.start_run(run_name=f"monitor_{model_name}_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_param("endpoint_id", endpoint_id)
            mlflow.log_param("model_name", model_name)

            # Placeholder metrics; replace with real monitoring
            metrics = {
                "avg_latency_ms": 150.5,
                "p99_latency_ms": 450.2,
                "requests_per_second": 10.5,
                "error_rate": 0.001,
                "model_quality_score": 0.92,
            }

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_param("monitoring_timestamp", datetime.now().isoformat())

            return metrics


class MLOpsPipeline:
    """Automated MLOps pipeline for continuous improvement"""

    def __init__(self, manager: VertexAIMLOpsManager):
        self.manager = manager

    def run_training_pipeline(
        self,
        dataset_path: str,
        dataset_name: str,
        model_config: ModelConfig | None = None,
        auto_deploy: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline from dataset to deployment
        """
        results: Dict[str, Any] = {}
        model_config = model_config or ModelConfig()

        try:
            # Step 1: Upload and version dataset
            logger.info("Step 1: Uploading dataset...")
            dataset_config = self.manager.upload_and_version_dataset(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                metadata={"pipeline_run": datetime.now().isoformat()},
            )
            results["dataset_version"] = dataset_config.version

            # Step 2: Create fine-tuning job
            logger.info("Step 2: Starting fine-tuning job...")
            job_id, mlflow_run_id = self.manager.create_fine_tuning_job(
                dataset_config=dataset_config, model_config=model_config
            )
            results["job_id"] = job_id
            results["mlflow_run_id"] = mlflow_run_id

            # Step 3: Wait for training and track metrics
            logger.info("Step 3: Monitoring training progress...")
            training_results = self.manager.track_training_metrics(job_id, mlflow_run_id)
            results["training_status"] = training_results["job_state"]

            # Step 4: If succeeded, register in MLflow Models (always), and deploy if requested
            if training_results["job_state"] == "JOB_STATE_SUCCEEDED":
                tuned_model_resource = training_results.get("model_resource_name")

                # Register model version in MLflow Models even before deployment
                model_registry_name = f"{dataset_name}_vertex_sft"
                _ = self.manager.create_mlflow_model_version(
                    model_name=model_registry_name,
                    endpoint_id=None,
                    vertex_model_resource_name=tuned_model_resource,
                    job_id=results.get("job_id"),
                    tags={"dataset_version": dataset_config.version, "base_model": model_config.base_model},
                )

            # Optional deployment
            if auto_deploy and training_results.get("job_state") == "JOB_STATE_SUCCEEDED":
                logger.info("Step 4: Deploying model...")
                # Use the tuned model resource produced by the job
                model_resource_name = training_results.get("model_resource_name")
                if not model_resource_name:
                    raise RuntimeError("Tuned model resource name not found after successful job.")

                endpoint_id, deployed_model_id = self.manager.deploy_model(
                    model_resource_name=model_resource_name,
                    endpoint_display_name=f"auto_deploy_{dataset_name}",
                )
                results["endpoint_id"] = endpoint_id
                results["deployed_model_id"] = deployed_model_id

                # Step 5: Register or update in MLflow Model Registry with endpoint linkage
                logger.info("Step 5: Registering model version with endpoint link...")
                model_version_run_id = self.manager.create_mlflow_model_version(
                    model_name=f"{dataset_name}_vertex_sft",
                    endpoint_id=endpoint_id,
                    vertex_model_resource_name=model_resource_name,
                    job_id=results.get("job_id"),
                    tags={"auto_deployed": "true", "dataset_version": dataset_config.version},
                )
                results["model_version_run_id"] = model_version_run_id

            logger.info("Pipeline completed successfully!")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Pipeline failed: {str(e)}")
            results["error"] = str(e)

        return results


def integrate_with_existing_code(endpoint_id: str) -> Dict[str, str]:
    """
    Generate environment variables for your existing code
    """
    return {
        "VERTEX_ENDPOINT_ID": endpoint_id,
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "VERTEXAI_LOCATION": os.getenv("VERTEXAI_LOCATION", "us-central1"),
    }


if __name__ == "__main__":
    # Example usage
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
    LOCATION = os.getenv("VERTEXAI_LOCATION", "us-central1")

    # Initialize MLOps manager
    manager = VertexAIMLOpsManager(
        project_id=PROJECT_ID,
        location=LOCATION,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "vertex-ai-llm-finetuning"),
    )

    # Run automated pipeline
    pipeline = MLOpsPipeline(manager)

    # Example: Run training pipeline with new dataset (adjust path)
    results = pipeline.run_training_pipeline(
        dataset_path="path/to/your/training_data.jsonl",
        dataset_name="customer_support_v2",
        auto_deploy=True,
    )

    # Use the deployed endpoint with your existing code
    if "endpoint_id" in results:
        env_vars = integrate_with_existing_code(results["endpoint_id"])
        print("Set these environment variables for your existing code:")
        for key, value in env_vars.items():
            print(f"export {key}={value}")
