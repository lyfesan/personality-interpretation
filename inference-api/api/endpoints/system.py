from fastapi import APIRouter
from services.model_manager import model_manager, DEVICE

router = APIRouter(tags=["System"])

@router.get("/")
async def root():
    """Standard root endpoint providing API metadata."""
    return {
        "api_name": "Big Five Personality Inference API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }

@router.get("/health")
async def health_check():
    """API Health check"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": list(model_manager.models.keys()),
        "port": "auto"
    }
