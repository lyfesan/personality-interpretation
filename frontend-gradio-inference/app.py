import gradio as gr
import requests
import base64
from io import BytesIO

INFERENCE_API_URL = "http://127.0.0.1:8000"
INTERPRETATION_API_URL = "http://127.0.0.1:8080"

def get_available_models():
    """Fetch available models from the FastAPI server."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models", timeout=2)
        if response.status_code == 200:
            return response.json().get("available_models", [])
    except Exception as e:
        print(f"Warning: Could not fetch models from API ({e}). Using defaults.")
    # Fallback default models if API is unreachable during startup
    return ["swinv2", "vit", "pvtv2"]

def predict(image, model_type):
    if image is None:
        return {"error": "Please upload an image."}
    if not model_type:
        return {"error": "Please select a model."}
    
    # Convert PIL Image to Base64 string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    payload = {
        "model_type": model_type,
        "image_base64": img_str
    }
    
    try:
        response = requests.post(f"{INFERENCE_API_URL}/predict_base64", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("predictions", {})
        else:
            return {"error": f"HTTP {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": "Connection failed. Is the API running?", "details": str(e)}

# --- Interpretation API helpers ---

def get_inference_models():
    """Fetch inference models from the interpretation API."""
    try:
        response = requests.get(f"{INTERPRETATION_API_URL}/inference-models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                return data.get("available_models", [])
            return data
    except Exception as e:
        print(f"Warning: Could not fetch inference models ({e}).")
    return ["swinv2", "vit", "pvtv2"]

def get_llm_models():
    """Fetch allowed LLM models from the interpretation API."""
    try:
        response = requests.get(f"{INTERPRETATION_API_URL}/llm-models", timeout=2)
        if response.status_code == 200:
            models = response.json()
            return [(m["name"], m["id"]) for m in models]
    except Exception as e:
        print(f"Warning: Could not fetch LLM models ({e}).")
    return [("Gemma 4 31B (free)", "google/gemma-4-31b-it:free")]

def interpret(image, inference_model, llm_model):
    """Send image to the interpretation API via multipart/form-data."""
    if image is None:
        return {}, "Please upload an image."
    if not inference_model:
        return {}, "Please select an inference model."
    if not llm_model:
        return {}, "Please select an LLM model."

    # Convert PIL image to bytes for multipart upload
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    try:
        files = {"image": ("image.jpg", buffered, "image/jpeg")}
        data = {
            "inference_model": inference_model,
            "llm_model": llm_model,
        }
        response = requests.post(
            f"{INTERPRETATION_API_URL}/interpret",
            files=files,
            data=data,
            timeout=120,
        )
        if response.status_code == 200:
            result = response.json()
            traits = result.get("predictions", {})
            interpretation = result.get("interpretation", "No interpretation returned.")
            return traits, interpretation
        else:
            err = response.json().get("error", response.text)
            return {}, f"Error {response.status_code}: {err}"
    except Exception as e:
        return {}, f"Connection failed. Is the interpretation API running?\n{e}"


# --- Build combined app ---

def build_app():
    models = get_available_models()
    inf_models = get_inference_models()
    llm_models = get_llm_models()

    with gr.Blocks(title="Personality Interpretation") as demo:
        gr.Markdown("# 🧠 Personality Analysis")

        with gr.Tabs():
            # ===== Tab 1: Raw Inference (existing) =====
            with gr.TabItem("🔬 Inference"):
                gr.Markdown("Test the raw inference API. Upload an image, choose a vision model, and get OCEAN trait scores.")
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Face Image")
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=models,
                                value=models[0] if models else None,
                                label="Inference Model",
                            )
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        submit_btn = gr.Button("Predict Personality", variant="primary")
                    with gr.Column():
                        output_json = gr.JSON(label="Personality Traits (OCEAN)")

                submit_btn.click(
                    fn=predict,
                    inputs=[image_input, model_dropdown],
                    outputs=output_json,
                )

                def refresh_models_list():
                    new_models = get_available_models()
                    return gr.update(choices=new_models, value=new_models[0] if new_models else None)

                refresh_btn.click(
                    fn=refresh_models_list,
                    inputs=[],
                    outputs=[model_dropdown],
                )

            # ===== Tab 2: Full Interpretation =====
            with gr.TabItem("✨ Interpretation"):
                gr.Markdown("Upload an image and get a full personality analysis powered by vision models + LLM interpretation.")
                with gr.Row():
                    with gr.Column():
                        interp_image = gr.Image(type="pil", label="Face Image")
                        with gr.Row():
                            interp_inf_dropdown = gr.Dropdown(
                                choices=inf_models,
                                value=inf_models[0] if inf_models else None,
                                label="Inference Model",
                            )
                            interp_llm_dropdown = gr.Dropdown(
                                choices=[name for name, _ in llm_models],
                                value=llm_models[0][0] if llm_models else None,
                                label="LLM Model",
                            )
                        interp_btn = gr.Button("Interpret Personality", variant="primary")
                    with gr.Column():
                        interp_traits = gr.JSON(label="Predicted Traits (OCEAN)")
                        interp_text = gr.Markdown(label="LLM Interpretation", value="*Interpretation will appear here...*")

                # Map display name -> id for the LLM dropdown
                llm_name_to_id = {name: mid for name, mid in llm_models}

                def on_interpret(image, inf_model, llm_name):
                    llm_id = llm_name_to_id.get(llm_name, llm_name)
                    return interpret(image, inf_model, llm_id)

                interp_btn.click(
                    fn=on_interpret,
                    inputs=[interp_image, interp_inf_dropdown, interp_llm_dropdown],
                    outputs=[interp_traits, interp_text],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
