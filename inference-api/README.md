# Big Five Personality Inference API

This directory contains the FastAPI backend responsible for running PyTorch vision models to infer Big Five (OCEAN) personality traits from face images.

## Architecture

The API has been designed using **Clean Architecture** principles. This ensures separation of concerns, high maintainability, and clear boundaries between configuration, data structures, business logic, and routing.

### Directory Structure

```text
inference-api/
├── main.py                     # Entrypoint & FastAPI setup
├── assets/                     # Model assets (e.g., blaze_face_short_range.tflite)
├── core/
│   ├── config.py               # Environment configuration and settings
│   ├── exceptions.py           # Custom exception handlers (e.g., 404 handler)
├── config/
│   ├── metadata.json           # API Metadata
│   ├── models.json             # Dynamic model configuration file
├── schemas/
│   ├── predict.py              # Pydantic schemas (Request & Response models)
│   ├── system.py               # Pydantic schemas (System & Model responses)
├── services/
│   ├── inference.py            # PyTorch model architecture (BigFiveRegressor)
│   ├── model_manager.py        # ML State, Model caching, and VRAM management
│   ├── face_extractor.py       # Preprocessing logic using Mediapipe
├── api/
│   ├── router.py               # Main API router aggregating other routes
│   ├── endpoints/
│       ├── system.py           # Root (/) and /health endpoints
│       ├── predict.py          # /predict and /models endpoints
```

### Component Details

#### 1. `main.py`
The main entry point. It initializes the FastAPI application, registers global exception handlers, and mounts the API routers. It leverages the modern `asynccontextmanager` (`lifespan`) to load the Hugging Face models securely into GPU/CPU memory exactly once upon server startup.

#### 2. `core/` & `config/`
- **`core/config.py`**: Loads environment variables from `.env` using `python-dotenv`.
- **`core/exceptions.py`**: Centralizes custom API exception handlers, returning cleaner, standardized JSON error messages for standard HTTP errors like 404.
- **`config/models.json`**: Contains the configurations (IDs, descriptions, repo IDs) of available models, making it easy to add new models without modifying code.

#### 3. `schemas/`
- **`predict.py`**: Uses **Pydantic** to strictly define and validate the data structures entering and leaving the API. 
  - `InferenceRequest`: Expects the requested `model_type` and an `image_base64` string.
  - `PredictionResponse`: Standardizes the output into an `OCEANTraits` dictionary and also returns `cropped_face_base64` to visualize the preprocessed face on the client.

#### 4. `services/`
- **`face_extractor.py`**: Uses MediaPipe to process the uploaded image, scores potential faces, and extracts the primary subject with a specific cropping offset to prepare it for neural network inference.
- **`inference.py`**: Contains the `BigFiveRegressor` class. This PyTorch `nn.Module` uses the `timm` library for the vision backbone and a custom classification head. It integrates with `PyTorchModelHubMixin` for easy Hugging Face weight loading.
- **`model_manager.py`**: Contains the `ModelManager` class which acts as a singleton. It handles:
  - Downloading and caching the PyTorch models based on `models.json`.
  - Managing the `FaceExtractor` preprocessing step.
  - Generating and storing the specific image transformations needed by each architecture.
  - Handling the execution of the `forward` pass in half-precision (if CUDA is available) and returning structured predictions.

#### 5. `api/`
- **`router.py`**: A central `APIRouter` that includes all underlying endpoint files.
- **`endpoints/system.py`**: Contains endpoints meant for orchestration health checks (`/health`) and basic API metadata (`/`).
- **`endpoints/predict.py`**: Contains the core functionality endpoints:
  - `GET /models`: Dynamically returns detailed metadata (names, descriptions, IDs) of models currently loaded into memory.
  - `POST /predict`: Accepts a base64 encoded image, extracts the main face, triggers the `model_manager` for inference, and returns the strictly typed JSON response containing both the OCEAN traits and the cropped face image.

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API Metadata and Status |
| `GET` | `/health` | Diagnostic check (device, port, loaded models) |
| `GET` | `/models` | Returns available inference models with detailed descriptions |
| `POST` | `/predict` | Submits an image, extracts the face, and returns OCEAN traits alongside the cropped face image |

*To see the full interactive API schema, run the server and navigate to `http://127.0.0.1:8000/docs`.*
