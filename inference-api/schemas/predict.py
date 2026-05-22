from pydantic import BaseModel, Field
from typing import Optional


class InferenceRequest(BaseModel):
    """Request body for Big Five personality trait prediction."""
    model_type: str = Field(
        ...,
        description="The ID of the vision model to use for inference",
        examples=["swinv2", "vit", "pvtv2"],
    )
    image_base64: str = Field(
        ...,
        description="Base64-encoded image string (JPEG/PNG). Data URI prefix is optional.",
        examples=["iVBORw0KGgoAAAANSUhEUg..."],
    )


class OCEANTraits(BaseModel):
    """Big Five (OCEAN) personality trait scores, each ranging from 0.0 to 1.0."""
    Openness: float = Field(..., ge=0.0, le=1.0, description="Openness to experience", examples=[0.62])
    Conscientiousness: float = Field(..., ge=0.0, le=1.0, description="Conscientiousness", examples=[0.63])
    Extraversion: float = Field(..., ge=0.0, le=1.0, description="Extraversion", examples=[0.54])
    Agreeableness: float = Field(..., ge=0.0, le=1.0, description="Agreeableness", examples=[0.63])
    Neuroticism: float = Field(..., ge=0.0, le=1.0, description="Neuroticism", examples=[0.60])


class PredictionResponse(BaseModel):
    """Response containing the model used, predicted OCEAN traits, and the cropped face image."""
    model_used: str = Field(..., description="The ID of the model that produced the prediction", examples=["swinv2"])
    predictions: OCEANTraits = Field(..., description="Predicted Big Five personality trait scores")
    cropped_face_base64: Optional[str] = Field(None, description="Base64 encoded cropped face image, if face extraction was used.", examples=["/9j/4AAQSkZJRgABAQEASABIAAD/4..."])
