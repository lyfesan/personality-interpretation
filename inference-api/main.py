from contextlib import asynccontextmanager
import json
from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import HOST, PORT, DEBUG_MODE
from core.exceptions import custom_404_handler
from api.router import api_router
from services.model_manager import model_manager, DEVICE

# Load metadata from config file
with open("config/metadata.json", "r") as f:
    METADATA = json.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Downloading/Loading models into VRAM (this takes a moment on first run)...")
    
    try:
        with open("config/models.json", "r") as f:
            models_config = json.load(f)
            
        for model_info in models_config:
            model_manager.load_hf_model_pipeline(
                model_info["id"], 
                model_info["repo_id"], 
                model_info=model_info
            )
    except FileNotFoundError:
        print("⚠️ models.json not found in config/. No models loaded automatically.")

    yield
    print("Shutting down API and releasing resources...")
    model_manager.models.clear()
    model_manager.transforms_dict.clear()
    if hasattr(model_manager, "model_configs"):
        model_manager.model_configs.clear()

app = FastAPI(
    title=METADATA["api_name"],
    description=METADATA["description"],
    version=METADATA["version"],
    debug=DEBUG_MODE,
    lifespan=lifespan,
)
print(f"API Engine initialized on: {DEVICE.upper()}")

# Register exception handlers
app.add_exception_handler(StarletteHTTPException, custom_404_handler)

# Include routers
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)