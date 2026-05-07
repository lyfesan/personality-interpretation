import io
import base64
import torch
from PIL import Image
from torchvision import transforms
from fastapi import HTTPException
from services.inference import BigFiveRegressor
from schemas.predict import OCEANTraits, PredictionResponse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    def __init__(self):
        self.models = {}
        self.transforms_dict = {}

    def load_hf_model_pipeline(self, model_key: str, repo_id: str, timm_name: str, use_complex_head: bool):
        """Loads model from Hugging Face and creates its specific preprocessing transform."""
        try:
            model = BigFiveRegressor.from_pretrained(repo_id, timm_name=timm_name, use_complex_head=use_complex_head)
            model.to(DEVICE)
            model.eval()

            # SwinV2 uses 256x256, ViT/PVTv2 use 224x224
            IMG_SIZE = 256 if 'swinv2' in model_key else 224
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.models[model_key] = model
            self.transforms_dict[model_key] = transform
            print(f"✅ Loaded {model_key.upper()} from {repo_id}")
        except Exception as e:
            print(f"⚠️ Failed to load {model_key} from {repo_id}. Error: {e}")

    def predict(self, model_type: str, image_base64: str) -> PredictionResponse:
        model_type_lower = model_type.lower()
        if model_type_lower not in self.models:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Choose from: {list(self.models.keys())}")

        # Decode Base64 to Image
        try:
            # Strip header if frontend accidentally includes "data:image/jpeg;base64,"
            base64_data = image_base64.split(",")[-1] 
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 image payload.")

        # Transform and Infer
        transform = self.transforms_dict[model_type_lower]
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        model = self.models[model_type_lower]
        with torch.no_grad():
            with torch.amp.autocast('cuda' if DEVICE == 'cuda' else 'cpu'):
                output = model(input_tensor)
                probabilities = output.squeeze().cpu().numpy()

        # 1. Map the raw array to the order the model was trained on
        raw_traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        raw_results = {trait: float(score) for trait, score in zip(raw_traits, probabilities)}

        # 2. Standardize to the OCEAN format using Pydantic
        standardized_ocean = OCEANTraits(
            Openness=raw_results['Openness'],
            Conscientiousness=raw_results['Conscientiousness'],
            Extraversion=raw_results['Extraversion'],
            Agreeableness=raw_results['Agreeableness'],
            Neuroticism=raw_results['Neuroticism']
        )

        # 3. Return the strictly formatted Pydantic object
        return PredictionResponse(
            model_used=model_type_lower, 
            predictions=standardized_ocean
        )

# Global instance to be used across the application
model_manager = ModelManager()
