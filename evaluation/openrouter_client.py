import os
import requests
from dotenv import load_dotenv

load_dotenv()

def evaluate_with_openrouter(prompt_text: str, image_base64: str = None, model: str = "openai/gpt-4o") -> str:
    """
    Directly calls OpenRouter's OpenAI-compatible API to perform evaluations.
    This allows us to seamlessly pass Multimodal inputs (Image + Text).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    content = [{"type": "text", "text": prompt_text}]
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": 0.0 # Greedy decoding for consistent evaluations
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]
