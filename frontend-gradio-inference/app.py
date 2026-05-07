import gradio as gr
import requests
import base64
from io import BytesIO

API_URL = "http://127.0.0.1:8000"

def get_available_models():
    """Fetch available models from the FastAPI server."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=2)
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
        response = requests.post(f"{API_URL}/predict_base64", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("predictions", {})
        else:
            return {"error": f"HTTP {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": "Connection failed. Is the API running?", "details": str(e)}

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
        
        # Action mappings
        submit_btn.click(
            fn=predict,
            inputs=[image_input, model_dropdown],
            outputs=output_json
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
