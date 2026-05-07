# Big Five Personality Inference API

This directory contains the FastAPI backend responsible for running PyTorch vision models to infer Big Five (OCEAN) personality traits from face images.

## Architecture

The API has been designed using **Clean Architecture** principles. This ensures separation of concerns, high maintainability, and clear boundaries between configuration, data structures, business logic, and routing.

### Directory Structure

```text
inference-api/
├── main.py                     # Entrypoint & FastAPI setup
├── core/
│   ├── config.py               # Environment configuration and settings
│   ├── exceptions.py           # Custom exception handlers (e.g., 404 handler)
├── schemas/
│   ├── predict.py              # Pydantic schemas (Request & Response models)
├── services/
│   ├── inference.py            # PyTorch model architecture (BigFiveRegressor)
│   ├── model_manager.py        # ML State, Model caching, and VRAM management
├── api/
│   ├── router.py               # Main API router aggregating other routes
│   ├── endpoints/
│       ├── system.py           # Root (/) and /health endpoints
│       ├── predict.py          # /predict_base64 and /models endpoints
```

### Component Details

#### 1. `main.py`
The main entry point. It initializes the FastAPI application, registers global exception handlers, and mounts the API routers. It leverages the modern `asynccontextmanager` (`lifespan`) to load the Hugging Face models securely into GPU/CPU memory exactly once upon server startup.

#### 2. `core/`
- **`config.py`**: Loads environment variables from `.env` using `python-dotenv`. It manages host, port configurations, and handles the Hugging Face authentication token (`HF_TOKEN`) needed for downloading private models.
- **`exceptions.py`**: Centralizes custom API exception handlers, returning cleaner, standardized JSON error messages for standard HTTP errors like 404.

#### 3. `schemas/`
- **`predict.py`**: Uses **Pydantic** to strictly define and validate the data structures entering and leaving the API. 
  - `InferenceRequest`: Expects the requested `model_type` and an `image_base64` string.
  - `PredictionResponse`: Standardizes the output into an `OCEANTraits` dictionary (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

#### 4. `services/`
This is where the actual "business logic" and machine learning heavy-lifting occur.
- **`inference.py`**: Contains the `BigFiveRegressor` class. This PyTorch `nn.Module` uses the `timm` library for the vision backbone and a custom classification head. It integrates with `PyTorchModelHubMixin` for easy Hugging Face weight loading.
- **`model_manager.py`**: Contains the `ModelManager` class which acts as a singleton. It handles:
  - Downloading and caching the PyTorch models (`swinv2`, `vit`, `pvtv2`).
  - Generating and storing the specific image transformations (resizing and normalization) needed by each architecture.
  - Handling the execution of the `forward` pass in half-precision (if CUDA is available) and returning structured predictions.

#### 5. `api/`
- **`router.py`**: A central `APIRouter` that includes all underlying endpoint files.
- **`endpoints/system.py`**: Contains endpoints meant for orchestration health checks (`/health`) and basic API metadata (`/`).
- **`endpoints/predict.py`**: Contains the core functionality endpoints:
  - `GET /models`: Dynamically returns a list of models currently loaded into memory by the `model_manager`.
  - `POST /predict_base64`: Accepts a base64 encoded image, triggers the `model_manager` for inference, and returns the strictly typed JSON response.

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API Metadata and Status |
| `GET` | `/health` | Diagnostic check (device, port, loaded models) |
| `GET` | `/models` | Returns available inference model keys (e.g., swinv2) |
| `POST` | `/predict_base64` | Submits an image for OCEAN trait inference |

*To see the full interactive API schema, run the server and navigate to `http://127.0.0.1:8000/docs`.*
