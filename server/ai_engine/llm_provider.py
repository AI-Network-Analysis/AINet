"""
LLM Provider utility (env-only)
Uses a fine-tuned Vertex AI model deployed to an endpoint.

Required env vars:
- VERTEX_ENDPOINT_ID: your deployed model endpoint ID (the numeric ID from your deployment)

Optional env vars:
- GOOGLE_CLOUD_PROJECT: GCP project id (ADC can also provide)
- VERTEXAI_LOCATION: Vertex AI region (default: us-central1)
- GOOGLE_APPLICATION_CREDENTIALS: path to service account json for local dev
"""

from typing import Any, Dict
import os

from langchain_google_vertexai import ChatVertexAI

def build_chat_model(model_config: Dict[str, Any] = None) -> ChatVertexAI:
    """
    Build a chat model (ChatVertexAI) using environment variables only.
    Uses a fine-tuned model deployed to a Vertex AI endpoint.
    Args:
        model_config: Ignored - all config comes from env vars
    Returns:
        ChatVertexAI instance for the deployed endpoint
    """
    endpoint_id = os.getenv('VERTEX_ENDPOINT_ID')
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('VERTEXAI_LOCATION', 'us-central1')
    
    if not endpoint_id:
        raise ValueError("VERTEX_ENDPOINT_ID environment variable is required")
    
    # Use the full endpoint path as the model name
    full_endpoint = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    
    # Create ChatVertexAI with minimal parameters to avoid validation errors
    return ChatVertexAI(
        model=full_endpoint,
        project=project,
        location=location,
        temperature=0.7,
        max_output_tokens=1024
    )
