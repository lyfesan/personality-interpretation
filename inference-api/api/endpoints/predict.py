from fastapi import APIRouter
from schemas.predict import InferenceRequest, PredictionResponse
from services.model_manager import model_manager

router = APIRouter(tags=["Inference"])

@router.get("/models")
async def list_models():
    """List all available models loaded in memory for inference."""
    return {
        "available_models": list(model_manager.models.keys())
    }

@router.post("/predict_base64", response_model=PredictionResponse)
async def predict_base64(request: InferenceRequest):
    return model_manager.predict(request.model_type, request.image_base64)
