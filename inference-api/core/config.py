import os
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "t")
HF_TOKEN = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face
if HF_TOKEN and HF_TOKEN != "your_huggingface_access_token_here":
    print("Logging into Hugging Face Hub...")
    login(token=HF_TOKEN)
