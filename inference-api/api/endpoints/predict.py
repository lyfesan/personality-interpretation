from fastapi import APIRouter
from schemas.predict import InferenceRequest, PredictionResponse
from schemas.system import ModelsListResponse
from services.model_manager import model_manager

router = APIRouter(tags=["Inference"])

@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all available models loaded in memory for inference."""
    return {
        "available_models": list(model_manager.model_configs.values())
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict_personality(request: InferenceRequest):
    """Predict Big Five personality traits from a base64 encoded face image."""
    return model_manager.predict(request.model_type, request.image_base64)
