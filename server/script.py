import os
from dotenv import load_dotenv
from mlops.pipelines.vertex_ai_mlops_pipeline import (
    VertexAIMLOpsManager,
    MLOpsPipeline,
    ModelConfig,
)

# Load environment variables from .env if present
load_dotenv()

credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

manager = VertexAIMLOpsManager(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("VERTEXAI_LOCATION", "us-central1"),
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    credentials_path=credentials_path,
    # staging_bucket="gs://your-staging-bucket"  # uncomment to override default
)

pipeline = MLOpsPipeline(manager)
results = pipeline.run_training_pipeline(
    dataset_path="/home/dushmin/datasetai.jsonl",
    dataset_name="my_dataset_v1",
    model_config=ModelConfig(base_model="gemini-2.5-flash-lite"),
    auto_deploy=False
)
print(results)