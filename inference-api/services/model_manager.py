import io
import base64
import torch
from PIL import Image
from torchvision import transforms
from fastapi import HTTPException
from services.inference import BigFiveRegressor
from schemas.predict import OCEANTraits, PredictionResponse
from services.face_extractor import FaceExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    def __init__(self):
        self.models = {}
        self.transforms_dict = {}
        self.model_configs = {}
        try:
            self.face_extractor = FaceExtractor()
        except Exception as e:
            print(f"Warning: Failed to initialize FaceExtractor: {e}")
            self.face_extractor = None

    def load_hf_model_pipeline(self, model_key: str, repo_id: str, model_info: dict = None):
        """Loads model from Hugging Face and creates its specific preprocessing transform."""
        try:
            model = BigFiveRegressor.from_pretrained(repo_id)
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
            if model_info:
                self.model_configs[model_key] = model_info
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

        # Face Extraction
        cropped_base64 = None
        if self.face_extractor:
            image = self.face_extractor.extract_main_face(image)
            if image is None:
                raise HTTPException(status_code=400, detail="No face detected in the image.")
            
            # Convert back to base64 for response
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            cropped_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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
            predictions=standardized_ocean,
            cropped_face_base64=cropped_base64
        )

# Global instance to be used across the application
model_manager = ModelManager()
