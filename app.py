import streamlit as st
from backend import preprocess_image, ask_llama, extract_basic_features, encode_image_with_clip

st.set_page_config(page_title="AI Agent for Photos", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #00aaff;
        text-shadow: 2px 2px 5px black;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #cccccc;
        margin-bottom: 2em;
    }
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed #00aaff;
        border-radius: 12px;
        padding: 20px;
    }
    div.stButton > button {
        background-color: #0066ff;
        color: white;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        font-size: 1em;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #0044cc;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- APP TITLE ---
st.markdown('<div class="title">🤖 AI Agent for Photos</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Python, LLaMA 3.2 & CLIP</div>', unsafe_allow_html=True)

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload a photo", type=["jpg", "jpeg", "png"])

# --- User Query ---
user_query = st.text_input("💬 Ask something about this photo:")

# --- Submit Button ---
if st.button("🔎 Analyze with AI Agent"):
    if uploaded_file is not None and user_query.strip():
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            image_info = extract_basic_features(uploaded_file)

            uploaded_file.seek(0)
            image_embedding = encode_image_with_clip(uploaded_file)

            uploaded_file.seek(0)
            img_b64 = preprocess_image(uploaded_file)

            response = ask_llama(user_query, image_desc=image_info, image_embedding=image_embedding)

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Response from AI Agent:")
        st.write(response)
    else:
        st.warning("⚠ Please upload a photo and enter your question.")