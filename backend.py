import os
import base64
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import cv2
from dotenv import load_dotenv

import torch
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------
# Load API key
# ---------------------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------
# Load HuggingFace CLIP model
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------------------------------------
# Image utilities
# ---------------------------------------------------
def preprocess_image(uploaded_file):
    """
    Convert uploaded file (BytesIO) to base64 string
    """
    image = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_b64

def extract_basic_features(uploaded_file):
    """
    Return simple image info: dimensions and channels
    """
    file_bytes = uploaded_file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    return f"Image dimensions: {w}x{h}, channels: {c}"

# ---------------------------------------------------
# CLIP Vision Encoder
# ---------------------------------------------------
def encode_image_with_clip(uploaded_file):
    """
    Encode the image into a vector embedding using HuggingFace CLIP
    """
    image = Image.open(uploaded_file).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().tolist()[0]

# ---------------------------------------------------
# LLaMA 3.2 call
# ---------------------------------------------------
def ask_llama(prompt, image_desc="", image_embedding=None):
    """
    Sends user prompt + optional image description/embedding to LLaMA 3.2 via OpenRouter
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    content = f"{prompt}\n\nImage info: {image_desc}"
    
    if image_embedding:
        content += f"\n\nImage embedding (CLIP vector first 10 dims): {image_embedding[:10]}..."

    messages = [
        {"role": "system", "content": "You are an AI photo assistant."},
        {"role": "user", "content": content}
    ]

    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": messages,
        "temperature": 0.4
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❗ Error querying LLaMA: {e}"