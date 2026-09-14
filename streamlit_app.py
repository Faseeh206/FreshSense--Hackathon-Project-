import os
import time
from datetime import datetime

# Configure Keras backend before importing
os.environ["KERAS_BACKEND"] = "numpy"

import streamlit as st
import numpy as np
from PIL import Image
import keras

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FreshSense — AI Food Freshness Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design matching the FreshSense Figma design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .badge-card {
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .status-fresh {
        background-color: #ECFDF5;
        border: 1.5px solid #A7F3D0;
        color: #065F46;
    }
    .status-slightly-aged {
        background-color: #FFFBEB;
        border: 1.5px solid #FDE68A;
        color: #92400E;
    }
    .status-stale {
        background-color: #FFF7ED;
        border: 1.5px solid #FFEDD5;
        color: #9A3412;
    }
    .status-spoiled {
        background-color: #FFF1F2;
        border: 1.5px solid #FECDD3;
        color: #991B1B;
    }
    .status-rotten {
        background-color: #FEF2F2;
        border: 1.5px solid #FCA5A5;
        color: #881337;
    }
    
    .badge-title {
        font-family: 'Outfit', sans-serif;
        font-size: 22px;
        font-weight: 800;
        margin: 0;
    }
    
    .badge-conf {
        font-family: 'JetBrains Mono', monospace;
        font-size: 16px;
        font-weight: 700;
    }
    
    .desc-box {
        font-size: 14px;
        line-height: 1.6;
        color: #475569;
        background: #F8FAFC;
        border-left: 4px solid #10B981;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
    }
    
    .team-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & Model Loading
# ---------------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["Fresh", "Slightly Aged", "Stale", "Spoiled", "Rotten"]

CLASS_DESCRIPTIONS = {
    "Fresh": "Harvest-grade quality with firm cellular structure and prime pigmentation. Completely safe for raw consumption or prolonged cold storage.",
    "Slightly Aged": "Minor dehydration or mild oxidation detected. Quality remains safe for consumption; recommended for use within 24–48 hours.",
    "Stale": "Noticeable moisture loss, wrinkled skin, and decline in aroma. Safe to cook, stew, or puree, but not recommended for fresh raw salads.",
    "Spoiled": "Significant spoilage and onset of microbial breakdown. Discoloration or soft pitting present. Unfit for consumption; discard immediately.",
    "Rotten": "Severe decomposition and advanced fungal activity. Pungent biological breakdown. Hazardous bio-waste — discard into compost or sealed trash.",
}

CLASS_CSS_MAP = {
    "Fresh": "status-fresh",
    "Slightly Aged": "status-slightly-aged",
    "Stale": "status-stale",
    "Spoiled": "status-spoiled",
    "Rotten": "status-rotten",
}

@st.cache_resource(show_spinner="Loading MobileNetV2 Neural Network…")
def load_model():
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "models", "MobileNetV2_best_model.keras"),
        os.path.join(os.path.dirname(__file__), "FreshSense", "models", "MobileNetV2_best_model.keras"),
        os.path.join(
            os.path.dirname(__file__),
            "Food-Freshness-Detection-Using-Deep-Learning-main",
            "Output Files",
            "agglomerative_saved_models",
            "MobileNetV2_best_model.keras",
        ),
    ]
    model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            "Could not locate MobileNetV2_best_model.keras in any candidate path: "
            + ", ".join(candidate_paths)
        )
    return keras.saving.load_model(model_path, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and add batch dimension."""
    image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------------------------------------------------------------------
# Sidebar Navigation & Project Info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🍃 FreshSense")
    st.caption("AI Food Freshness Diagnostic Platform")
    st.markdown("**Pak-Angels Mid Program Hackathon**")
    
    st.divider()
    st.subheader("📋 5 Freshness Stages")
    st.markdown("""
    - 🟢 **Fresh:** Harvest grade, raw eating
    - 🟡 **Slightly Aged:** Eat within 48 hours
    - 🟠 **Stale:** Cook, stew, or puree
    - 🔴 **Spoiled:** Discard, do not eat
    - 🟣 **Rotten:** Hazardous bio-waste
    """)

    st.divider()
    st.subheader("🛠 Tech Stack")
    st.markdown("""
    - **Google Antigravity:** Agentic workflow
    - **MobileNetV2:** 11.5 MB weights, 82% F1
    - **Keras 3 + NumPy:** CPU inference (&lt;850ms)
    - **Dataset:** 13 produce categories (6.4 GB)
    """)

    st.divider()
    st.subheader("👥 Team Members")
    st.markdown("""
    <div class="team-box">
        <b>Muhammad Faseeh (GL)</b><br><small>Group Leader & AI Lead</small><br><br>
        <b>Umamah Ibreeq Zafar</b><br><small>CV Research & Dataset</small><br><br>
        <b>Anumta Nadeem</b><br><small>UI/UX Design Systems</small><br><br>
        <b>Zafar Aman Khattak</b><br><small>System Architecture & QA</small>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🍃 FreshSense — Food Freshness Detection")
st.markdown(
    "Upload a photo of any fruit or vegetable to instantly evaluate its freshness "
    "across **five biological degradation stages** using MobileNetV2 computer vision."
)

# Test sample selector or file uploader
tabs = st.tabs(["📤 Upload Your Image", "🧪 Test Sample Images"])

image_to_process = None
image_name = ""

with tabs[0]:
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of produce for freshness evaluation"
    )
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file)
            image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

with tabs[1]:
    test_samples_dir = os.path.join(os.path.dirname(__file__), "Test Samples")
    if os.path.exists(test_samples_dir):
        samples = [f for f in os.listdir(test_samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if samples:
            selected_sample = st.selectbox("Select a sample produce image:", ["None"] + samples)
            if selected_sample != "None":
                sample_path = os.path.join(test_samples_dir, selected_sample)
                image_to_process = Image.open(sample_path)
                image_name = selected_sample

# ---------------------------------------------------------------------------
# Prediction & Results Display
# ---------------------------------------------------------------------------
if image_to_process is not None:
    col_img, col_results = st.columns([1, 1.2], gap="large")
    
    with col_img:
        st.subheader("📷 Input Image")
        st.image(image_to_process, caption=image_name, use_container_width=True)
    
    with col_results:
        st.subheader("🧠 Diagnostic Analysis")
        
        with st.spinner("Analyzing image features…"):
            start_time = time.time()
            img_tensor = preprocess_image(image_to_process)
            
            # Silent inference to prevent terminal issues
            preds = model.predict(img_tensor, verbose=0)
            latency = (time.time() - start_time) * 1000
            
            probs = [float(p) for p in preds[0]]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100.0

        # Status badge
        css_class = CLASS_CSS_MAP.get(pred_class, "status-fresh")
        st.markdown(f"""
        <div class="badge-card {css_class}">
            <div>
                <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Classification Result</span>
                <div class="badge-title">{pred_class}</div>
            </div>
            <div class="badge-conf">{confidence:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Plain-language culinary advice
        st.markdown(f"""
        <div class="desc-box">
            <b>Culinary & Safety Guidance:</b><br>
            {CLASS_DESCRIPTIONS[pred_class]}
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution bars
        st.markdown("##### 📊 Confidence Distribution Across 5 Classes")
        for name, prob in zip(CLASS_NAMES, probs):
            percentage = prob * 100.0
            st.write(f"**{name}** — `{percentage:.1f}%`")
            st.progress(min(max(float(prob), 0.0), 1.0))
        
        st.caption(f"⚡ Local CPU inference latency: `{latency:.1f} ms` &bull; Resolution: `128×128`")
        
        # Record session history
        if "history" not in st.session_state:
            st.session_state["history"] = []
            
        record = {
            "name": image_name,
            "class": pred_class,
            "confidence": f"{confidence:.1f}%",
            "time": datetime.now().strftime("%I:%M:%S %p"),
        }
        # Avoid duplicate consecutive additions
        if not st.session_state["history"] or st.session_state["history"][0].get("time") != record["time"]:
            st.session_state["history"].insert(0, record)

# ---------------------------------------------------------------------------
# Session History Table
# ---------------------------------------------------------------------------
if "history" in st.session_state and st.session_state["history"]:
    st.divider()
    st.subheader("🕒 Recent Scans in This Session")
    st.table(st.session_state["history"][:10])
