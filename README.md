# Personality Interpretation: Big Five Traits

This project provides an end-to-end solution for interpreting the "Big Five" (OCEAN) personality traits from face images. It consists of two main components:
1. **Inference API**: A clean-architecture FastAPI backend that handles loading Hugging Face PyTorch models and serving predictions.
2. **Gradio Frontend**: A web-based UI for easily testing the API and visualizing predictions.

---

## 1. Running the Inference API (Backend)

The backend runs on FastAPI and PyTorch, accelerating inference via CUDA if available. 

### Setup and Execution

1. Open a terminal and navigate to the API directory:
   ```bash
   cd inference-api
   ```

2. Create and activate a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure your environment variables are configured. You can use the existing `.env` file or copy `.env.example` to `.env` to set your `HF_TOKEN`.

5. Start the API server:
   ```bash
   python main.py
   ```

The API will start at `http://127.0.0.1:8000`. During the first startup, it will load the model weights into your VRAM.
You can view the interactive Swagger API documentation at `http://127.0.0.1:8000/docs`.

---

## 2. Running the Gradio Testing UI (Frontend)

The Gradio app provides a clean graphical interface to upload images, select models dynamically, and view the interpreted OCEAN results.

### Prerequisites
- The **Inference API must be running** (as instructed above) for the UI to fetch the available models and process predictions.

### Setup and Execution

1. Open a **new terminal window** and navigate to the frontend directory:
   ```bash
   cd frontend-gradio-inference
   ```

2. Activate your Python environment. You can use the same virtual environment created for the API:
   ```bash
   # Windows PowerShell
   ..\inference-api\venv\Scripts\Activate.ps1
   # Linux/macOS
   source ../inference-api/venv/bin/activate
   ```
   *Alternatively, you can create a separate virtual environment inside this folder.*

3. Install the required dependencies (Gradio, Requests, Pillow):
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Gradio application:
   ```bash
   python app.py
   ```

The terminal will provide a local URL (typically `http://127.0.0.1:7860`). Open this address in your web browser. Upload a face image, select your model, and click **Predict Personality** to see the traits!