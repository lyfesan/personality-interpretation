import gradio as gr
import requests
import base64
import json
import tempfile
import os
from io import BytesIO
from PIL import Image

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://127.0.0.1:8000")
INTERPRETATION_API_URL = os.getenv("INTERPRETATION_API_URL", "http://127.0.0.1:8080")


def get_available_models():
    """Fetch available models from the FastAPI server."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/models", timeout=2)
        if response.status_code == 200:
            models_data = response.json().get("available_models", [])
            # Return list of tuples: (Display Name, model_id) for the dropdown
            return [(f"{m.get('name', m.get('id'))}", m.get("id")) for m in models_data]
    except Exception as e:
        print(f"Warning: Could not fetch models from API ({e}). Using defaults.")
    # Fallback default models if API is unreachable during startup
    return [("SwinV2 (swinv2)", "swinv2"), ("ViT (vit)", "vit"), ("PVTv2 (pvtv2)", "pvtv2")]

def predict(image, model_type):
    if image is None:
        return {"error": "Please upload an image."}, None
    if not model_type:
        return {"error": "Please select a model."}, None
    
    # Convert PIL Image to Base64 string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    payload = {
        "model_type": model_type,
        "image_base64": img_str
    }
    
    try:
        response = requests.post(f"{INFERENCE_API_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", {})
            cropped_b64 = data.get("cropped_face_base64")
            
            cropped_img = None
            if cropped_b64:
                try:
                    img_data = base64.b64decode(cropped_b64)
                    cropped_img = Image.open(BytesIO(img_data)).convert("RGB")
                except Exception:
                    pass
                    
            return predictions, cropped_img
        else:
            return {"error": f"HTTP {response.status_code}", "details": response.text}, None
    except Exception as e:
        return {"error": "Connection failed. Is the API running?", "details": str(e)}, None

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

def get_response_styles():
    """Fetch allowed response styles from the interpretation API."""
    try:
        response = requests.get(f"{INTERPRETATION_API_URL}/response-styles", timeout=2)
        if response.status_code == 200:
            styles = response.json()
            return [(s["name"], s["id"]) for s in styles]
    except Exception as e:
        print(f"Warning: Could not fetch response styles ({e}).")
    return [("Comprehensive (ID)", "comprehensive_id")]

def interpret(image, inference_model, llm_model, style_id):
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
            "style_id": style_id,
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

def export_result(image, inf_model, llm_id, style_id, traits, interpretation):
    """Exports the results to a JSON file and returns the temp file path."""
    if not traits and not interpretation:
        return None # Nothing to export
        
    img_b64 = None
    if image is not None:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    data = {
        "parameters": {
            "inference_model": inf_model,
            "llm_model": llm_id,
            "response_style": style_id
        },
        "results": {
            "predictions": traits,
            "interpretation": interpretation
        },
        "image_base64": img_b64
    }
    
    fd, path = tempfile.mkstemp(suffix=".json", prefix="personality_export_")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    return path


# --- Build combined app ---

def build_app():
    models = get_available_models()
    inf_models_raw = get_inference_models()
    
    # Map inference model IDs to display names (Name, ID)
    id_to_name = {m_id: m_name for m_name, m_id in models}
    
    inf_models = []
    for m in inf_models_raw:
        if isinstance(m, dict):
            inf_models.append((m.get("name", m.get("id")), m.get("id")))
        else:
            inf_models.append((id_to_name.get(m, m), m))

    llm_models = get_llm_models()
    response_styles = get_response_styles()

    with gr.Blocks(title="Personality Interpretation") as demo:
        gr.Markdown("# Personality Analysis")

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
                                value=models[0][1] if models else None, 
                                label="Inference Model"
                            )
                            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")

                        submit_btn = gr.Button("Predict Personality", variant="primary")
                        
                    with gr.Column():
                        output_json = gr.JSON(label="Personality Traits (OCEAN)")
                        cropped_output = gr.Image(type="pil", label="Extracted Face (Model Input)")
        
                # Action mappings
                submit_btn.click(
                    fn=predict,
                    inputs=[image_input, model_dropdown],
                    outputs=[output_json, cropped_output]
                )
                
                def refresh_models_list():
                    new_models = get_available_models()
                    return gr.update(choices=new_models, value=new_models[0][1] if new_models else None)
                    
                refresh_btn.click(
                    fn=refresh_models_list,
                    inputs=[],
                    outputs=[model_dropdown]
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
                                value=inf_models[0][1] if inf_models else None,
                                label="Inference Model",
                            )
                            interp_llm_dropdown = gr.Dropdown(
                                choices=llm_models,
                                value=llm_models[0][1] if llm_models else None,
                                label="LLM Model",
                            )
                        style_dropdown = gr.Dropdown(
                            choices=response_styles,
                            value=response_styles[0][1] if response_styles else None,
                            label="Response Style"
                        )
                        interp_btn = gr.Button("Interpret Personality", variant="primary")
                    with gr.Column():
                        interp_traits = gr.JSON(label="Predicted Traits (OCEAN)")
                        interp_text = gr.Markdown(label="LLM Interpretation", value="*Interpretation will appear here...*")
                        
                        export_btn = gr.DownloadButton("Export Result as JSON", variant="secondary")

                def on_interpret(image, inf_model, llm_id, style_id):
                    return interpret(image, inf_model, llm_id, style_id)

                interp_btn.click(
                    fn=on_interpret,
                    inputs=[interp_image, interp_inf_dropdown, interp_llm_dropdown, style_dropdown],
                    outputs=[interp_traits, interp_text],
                )
                
                export_btn.click(
                    fn=export_result,
                    inputs=[interp_image, interp_inf_dropdown, interp_llm_dropdown, style_dropdown, interp_traits, interp_text],
                    outputs=[export_btn]
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    app.launch(server_name=server_name, server_port=server_port, share=False, theme=gr.themes.Soft())

