from pydantic import BaseModel, Field
from typing import List


class MetadataResponse(BaseModel):
    """API metadata returned by the root endpoint."""
    api_name: str = Field(..., description="Name of the API", examples=["Big Five Personality Inference API"])
    description: str = Field(..., description="Brief description of the API", examples=["API for predicting Big Five personality traits"])
    version: str = Field(..., description="Current API version", examples=["1.2.0"])
    status: str = Field(..., description="Current API status", examples=["online"])
    documentation: str = Field(..., description="Path to interactive API docs", examples=["/docs"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status of the API", examples=["healthy"])
    device: str = Field(..., description="Compute device in use", examples=["cuda"])
    models_loaded: List[str] = Field(..., description="List of model IDs currently loaded in memory", examples=[["swinv2", "vit", "pvtv2"]])
    port: str = Field(..., description="Port configuration", examples=["auto"])


class ModelDetail(BaseModel):
    """Details of an available inference model."""
    id: str = Field(..., description="The ID of the model to be used in predictions", examples=["swinv2"])
    name: str = Field(..., description="Human-readable name of the model", examples=["Swin Transformer V2"])
    description: str = Field(..., description="Description of the model", examples=["SwinV2 Base model optimized for Big Five personality traits prediction"])
    repo_id: str = Field(..., description="Hugging Face model repository ID", examples=["lyfesan/swinv2_base_..."])

class ModelsListResponse(BaseModel):
    """List of available inference models."""
    available_models: List[ModelDetail] = Field(..., description="List of models available for inference")
