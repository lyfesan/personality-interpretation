import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000"

def get_available_models():
    """Fetch available models from the FastAPI server."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=2)
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
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
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

def build_app():
    models = get_available_models()
    
    with gr.Blocks(title="Big Five Inference UI") as demo:
        gr.Markdown("# Big Five Personality Inference")
        gr.Markdown("Temporary UI for testing the inference API. Upload an image, choose a vision model, and predict OCEAN traits.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Face Image")
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=models, 
                        value=models[0] if models else None, 
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
            return gr.update(choices=new_models, value=new_models[0] if new_models else None)
            
        refresh_btn.click(
            fn=refresh_models_list,
            inputs=[],
            outputs=[model_dropdown]
        )
        
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
