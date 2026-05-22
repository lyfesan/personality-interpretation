import json
from fastapi import APIRouter
from schemas.system import MetadataResponse, HealthResponse
from services.model_manager import model_manager, DEVICE

router = APIRouter(tags=["System"])

@router.get("/", response_model=MetadataResponse)
async def root():
    """Standard root endpoint providing API metadata."""
    with open("config/metadata.json", "r") as f:
        metadata = json.load(f)
    metadata["documentation"] = "/docs"
    return metadata

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API Health check"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": list(model_manager.models.keys()),
        "port": "auto"
    }

