# Personality Interpretation: Big Five Traits

This project provides an end-to-end microservice solution for interpreting the "Big Five" (OCEAN) personality traits from face images. By leveraging advanced Vision Models and Large Language Models (LLMs), it not only extracts raw personality scores from images but also provides personalized, human-readable psychological interpretations.

## System Architecture

The project is split into three main components, each housed in its own directory:

### 1. `inference-api/` (Python / FastAPI)
A high-performance PyTorch backend that loads Hugging Face vision models (like SwinV2, ViT) into memory.
- **Features**: MediaPipe face extraction, dynamic model caching, and Big Five (OCEAN) regression predictions.
- **Run Instructions**: See [inference-api/README.md](./inference-api/README.md)

### 2. `interpretation-api/` (Go / Gin)
An API Gateway that orchestrates the workflow between the `inference-api` and external LLMs.
- **Features**: Communicates with OpenRouter to send image data and Big Five scores to Vision-Language Models (e.g. Qwen, Gemma) to generate natural language interpretations. Natively supports dynamic response styles (e.g., Short Summary, Comprehensive Analysis) configured via JSON.
- **Run Instructions**: See [interpretation-api/README.md](./interpretation-api/README.md)

### 3. `frontend-gradio-inference/` (Python / Gradio)
A user-friendly web UI for interacting with both the raw inference backend and the full interpretation pipeline.
- **Features**: Dual-tab interface (Raw Inference & Full Interpretation), model selection dropdowns, and a JSON result exporter for LLM-as-a-Judge evaluations.

---

## Quick Start Guide

To run the entire ecosystem locally, you will need three terminal windows running simultaneously.

### Step 1: Start the Inference API
1. Navigate to `inference-api/`
2. Activate your Python environment and install `requirements.txt`.
3. Set your `HF_TOKEN` in `.env`.
4. Run `python main.py` (Starts on port 8000).

### Step 2: Start the Interpretation API
1. Navigate to `interpretation-api/`
2. Configure `.env` with `INFERENCE_API_URL=http://localhost:8000` and your `OPENROUTER_API_KEY`.
3. Run `go run main.go` (Starts on port 8080).

### Step 3: Start the Gradio Frontend
1. Navigate to `frontend-gradio-inference/`
2. Activate your Python environment and install `requirements.txt`.
3. Run `python app.py` (Starts on port 7860).

Once all three services are running, open `http://127.0.0.1:7860` in your web browser.

---

## Evaluation (LLM-as-a-Judge)
This architecture is specifically designed to support evaluation using frameworks like DeepEval. The `interpretation-api` enforces language outputs based on unique response styles (English translated to Indonesian vs. Native Indonesian) to measure translation quality and context preservation in modern LLMs. Using the **Export Result as JSON** feature in the frontend allows you to easily save outputs for batch evaluation.