from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import HOST, PORT, DEBUG_MODE
from core.exceptions import custom_404_handler
from api.router import api_router
from services.model_manager import model_manager, DEVICE

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Downloading/Loading models into VRAM (this takes a moment on first run)...")
    
    model_manager.load_hf_model_pipeline(
        "swinv2", 
        "lyfesan/swinv2_base_window12to16_192to256_ms_in22k_ft_in1k_Run_D_The_Ultimate_bigfive", 
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", 
        True
    )
    
    model_manager.load_hf_model_pipeline(
        "vit", 
        "lyfesan/vit_base_patch16_224_augreg_in21k_ft_in1k_Run_D_The_Ultimate_bigfive", 
        "vit_base_patch16_224.augreg_in21k", 
        True
    )

    model_manager.load_hf_model_pipeline(
        "pvtv2", 
        "lyfesan/pvt_v2_b5_in1k_Run_D_The_Ultimate_bigfive", 
        "pvt_v2_b5.in1k", 
        True
    )
    yield
    print("Shutting down API and releasing resources...")
    model_manager.models.clear()
    model_manager.transforms_dict.clear()

app = FastAPI(title="Big Five Personality Inference API", debug=DEBUG_MODE, lifespan=lifespan)
print(f"API Engine initialized on: {DEVICE.upper()}")

# Register exception handlers
app.add_exception_handler(StarletteHTTPException, custom_404_handler)

# Include routers
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)