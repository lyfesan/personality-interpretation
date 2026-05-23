# Interpretation API (Go)

This directory contains the Go-based API Gateway built with the Gin framework. It acts as an orchestrator that bridges the gap between raw vision-based personality predictions and natural language generation via Large Language Models (LLMs).

## Features

- **Microservice Orchestration**: Forwards uploaded images to the `inference-api` to extract raw Big Five personality scores.
- **LLM Integration**: Communicates with OpenRouter to pass the extracted traits and the original image into Vision-Language Models (e.g., Gemma, Qwen) for deep personality analysis.
- **Dynamic Prompting System**: Supports multiple prompt styles (e.g., Comprehensive, Short Summary) natively integrated into the API. The prompts enforce strict Indonesian language output for downstream evaluation.
- **Configuration Driven**: LLM models and response styles are easily configurable via JSON files without changing the Go code.

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API Metadata and Status |
| `GET` | `/docs` | Interactive Swagger/OpenAPI UI |
| `GET` | `/inference-models` | Proxies available vision models from the `inference-api` |
| `GET` | `/llm-models` | Returns available LLM models defined in `config/llm_models.json` |
| `GET` | `/response-styles` | Returns available response styles and their prompts from `config/response_styles.json` |
| `POST` | `/interpret` | Accepts a `multipart/form-data` request with an image, inference model ID, LLM ID, and style ID. Returns the predicted traits and the LLM interpretation text. |

## Setup and Execution

### Prerequisites
- Go 1.20+ installed.
- The **Inference API must be running** locally.
- An **OpenRouter API Key** for LLM access.

### Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Update the variables in `.env`:
   ```env
   INFERENCE_API_URL=http://localhost:8000
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   APP_URL=http://localhost:8080
   ```

### Running the API

1. Navigate to the `interpretation-api` directory:
   ```bash
   cd interpretation-api
   ```

2. Run the application:
   ```bash
   go run main.go
   ```

The server will start on `http://127.0.0.1:8080`. You can view the API documentation at `http://127.0.0.1:8080/docs`.
