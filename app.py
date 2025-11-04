import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import pipeline
from stage1.models import resnet18_finetune

# ============================================================
# 🎯 App Configuration
# ============================================================
st.set_page_config(page_title="AI Emotion Detector", layout="centered")
st.title("🎭 AI Emotion Detector")
st.markdown("""
Choose **Image** or **Text** mode below.  
This app can detect emotions from either a **face image** or a **written sentence**.
""")

# ============================================================
# ⚙️ Load Models
# ============================================================
@st.cache_resource
def load_face_model():
    model = resnet18_finetune(num_classes=7)
    model.load_state_dict(torch.load("stage1/artifacts/best_model.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

face_model = load_face_model()
text_model = load_text_model()
classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ============================================================
# 🧩 Mode Selection
# ============================================================
mode = st.radio("Choose input type:", ["🖼️ Image", "💬 Text"])

# ============================================================
# 🎭 IMAGE MODE
# ============================================================
if mode == "🖼️ Image":
    st.header("📷 Upload a Face Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = face_model(img_tensor).argmax(1).item()
        predicted_emotion = classes[pred]

        st.success(f"🎭 Detected Emotion: **{predicted_emotion.upper()}**")

# ============================================================
# 💬 TEXT MODE
# ============================================================
else:
    st.header("✍️ Enter a Sentence")
    user_input = st.text_area("Write something expressing your feeling:")

    if st.button("🔍 Detect Emotion"):
        if user_input.strip():
            result = text_model(user_input)[0]
            predicted_emotion = result["label"].lower()
            confidence = round(result["score"] * 100, 2)
            st.success(f"🎭 Detected Emotion: **{predicted_emotion.upper()}** ({confidence}%)")
        else:
            st.warning("Please type something first.")