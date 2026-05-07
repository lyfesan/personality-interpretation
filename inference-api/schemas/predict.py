from pydantic import BaseModel

class InferenceRequest(BaseModel):
    model_type: str
    image_base64: str

# Define the OCEAN traits structure
class OCEANTraits(BaseModel):
    Openness: float
    Conscientiousness: float
    Extraversion: float
    Agreeableness: float
    Neuroticism: float

# Define the final JSON response structure
class PredictionResponse(BaseModel):
    model_used: str
    predictions: OCEANTraits
