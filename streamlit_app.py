import os
import time
import base64
from datetime import datetime

# ---------------------------------------------------------------------------
# Backend & Model Setup (Robust fallback for Linux / Windows / Python 3.10-3.14)
# ---------------------------------------------------------------------------
os.environ.setdefault("KERAS_BACKEND", "numpy")

import streamlit as st
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FreshSense — AI Food Freshness Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helper: Read Logo as Base64 for bulletproof image rendering
# ---------------------------------------------------------------------------
def get_logo_base64():
    candidates = [
        os.path.join(os.path.dirname(__file__), "static", "logo.jpg"),
        os.path.join(os.path.dirname(__file__), "FreshSense", "static", "logo.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return ""

LOGO_B64 = get_logo_base64()
LOGO_SRC = f"data:image/jpeg;base64,{LOGO_B64}" if LOGO_B64 else ""

# ---------------------------------------------------------------------------
# High-End Design System Injection (Matching FreshSense Figma & Flask App)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Work+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* CSS Reset for Streamlit Container */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {visibility: hidden; height: 0px;}
    
    .stApp {
        background-color: #F8FAFC !important;
        font-family: 'Work Sans', sans-serif !important;
        color: #1E293B !important;
    }
    
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* Custom Header Bar */
    .custom-navbar {
        background: rgba(11, 15, 25, 0.96);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 14px 28px;
        margin: -1rem -2rem 24px -2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .nav-brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .nav-logo {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        object-fit: cover;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }
    
    .nav-title {
        font-family: 'Outfit', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1;
    }
    
    .nav-sub {
        font-size: 11px;
        color: #94A3B8;
        letter-spacing: 0.04em;
        margin-top: 3px;
    }
    
    .nav-badge {
        background: linear-gradient(135deg, #10B981, #059669);
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Hero Section */
    .hero-banner {
        position: relative;
        background: linear-gradient(145deg, #0B0F19 0%, #111827 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 42px 40px 36px 40px;
        margin-bottom: 30px;
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.15);
    }
    
    .hero-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(70px);
        pointer-events: none;
        opacity: 0.25;
    }
    .orb-1 {
        width: 320px;
        height: 320px;
        top: -80px;
        right: -60px;
        background: #10B981;
    }
    .orb-2 {
        width: 260px;
        height: 260px;
        bottom: -60px;
        left: 25%;
        background: #3B82F6;
    }
    
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        color: #34D399;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    .hero-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34D399;
        box-shadow: 0 0 10px #34D399;
    }
    
    .hero-h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: 12px;
    }
    
    .hero-gradient {
        background: linear-gradient(135deg, #34D399 0%, #6EE7B7 50%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-p {
        color: #94A3B8;
        font-size: 15px;
        line-height: 1.6;
        max-width: 680px;
        margin-bottom: 26px;
    }
    
    .stat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 20px;
    }
    
    .stat-card {
        flex: 1;
        min-width: 140px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 16px;
    }
    
    .stat-val {
        font-family: 'Outfit', sans-serif;
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    .stat-lbl {
        font-size: 11px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Workspace Containers */
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 24px;
    }
    
    .panel-header {
        font-family: 'Outfit', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Diagnosis Badges */
    .diag-badge {
        padding: 18px 22px;
        border-radius: 14px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    .badge-fresh {
        background: #ECFDF5;
        border: 1.5px solid #A7F3D0;
        color: #065F46;
    }
    .badge-slightly-aged {
        background: #FFFBEB;
        border: 1.5px solid #FDE68A;
        color: #92400E;
    }
    .badge-stale {
        background: #FFF7ED;
        border: 1.5px solid #FFEDD5;
        color: #9A3412;
    }
    .badge-spoiled {
        background: #FFF1F2;
        border: 1.5px solid #FECDD3;
        color: #991B1B;
    }
    .badge-rotten {
        background: #FEF2F2;
        border: 1.5px solid #FCA5A5;
        color: #881337;
    }
    
    .badge-label-small {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.8;
    }
    
    .badge-class-name {
        font-family: 'Outfit', sans-serif;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-top: 2px;
    }
    
    .badge-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.8);
        padding: 6px 14px;
        border-radius: 9999px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .guidance-box {
        background: #F8FAFC;
        border-left: 4px solid #10B981;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.6;
        color: #334155;
        margin-bottom: 20px;
    }
    
    .empty-state-box {
        text-align: center;
        padding: 44px 20px;
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        color: #64748B;
    }

    /* Bento Taxonomy Grid */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 32px;
    }
    
    .bento-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    
    .bento-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    
    .bento-title {
        font-family: 'Outfit', sans-serif;
        font-size: 16px;
        font-weight: 700;
    }
    
    .bento-desc {
        font-size: 13px;
        color: #64748B;
        line-height: 1.5;
    }

    /* Custom Streamlit Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navbar Component
# ---------------------------------------------------------------------------
logo_html = f'<img src="{LOGO_SRC}" class="nav-logo" alt="Logo">' if LOGO_SRC else '<span style="font-size:26px;">🍃</span>'

st.markdown(f"""
<div class="custom-navbar">
    <div class="nav-brand-group">
        {logo_html}
        <div>
            <div class="nav-title">FreshSense</div>
            <div class="nav-sub">AI FOOD FRESHNESS DIAGNOSTIC PLATFORM</div>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span class="nav-badge">AI 2.0</span>
        <span style="color:#94A3B8; font-size:12px; font-weight:600;">Pak-Angels Mid Program Hackathon</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Banner Component
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-orb orb-1"></div>
    <div class="hero-orb orb-2"></div>
    <div class="hero-pill">
        <span class="hero-pill-dot"></span>
        Deep Learning Vision System &bull; 82% F1 Accuracy &bull; Pak-Angels Hackathon
    </div>
    <div class="hero-h1">
        Know your food freshness.<br>
        <span class="hero-gradient">Before you taste or serve.</span>
    </div>
    <div class="hero-p">
        Instant computer vision diagnostics powered by MobileNetV2. Upload any fruit or vegetable photo to evaluate quality across five distinct biological degradation stages with automated culinary guidance.
    </div>
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">82%</div>
            <div class="stat-lbl">Validation F1 Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">13</div>
            <div class="stat-lbl">Produce Categories</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">5 Stages</div>
            <div class="stat-lbl">Freshness Spectrum</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">&lt; 350ms</div>
            <div class="stat-lbl">Local CPU Latency</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model Loader (Resilient Fallback: TF / Keras 3 with compile=False)
# ---------------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["Fresh", "Slightly Aged", "Stale", "Spoiled", "Rotten"]

CLASS_DESCRIPTIONS = {
    "Fresh": "Harvest-grade quality with firm cellular structure and prime pigmentation. Completely safe for raw consumption, salads, or extended cold storage.",
    "Slightly Aged": "Minor dehydration or light oxidation detected. Organoleptic quality remains safe; recommended for consumption within 24–48 hours.",
    "Stale": "Noticeable moisture loss, skin wrinkling, and aroma decline. Safe to cook, stew, bake, or puree, but not recommended for fresh raw eating.",
    "Spoiled": "Onset of active microbial breakdown and deep tissue softening. Unfit for human consumption — discard immediately to prevent contamination.",
    "Rotten": "Severe decomposition and heavy fungal / spore coverage. Pungent biological decay. Hazardous bio-waste — dispose in compost or sealed trash.",
}

CLASS_BADGE_MAP = {
    "Fresh": "badge-fresh",
    "Slightly Aged": "badge-slightly-aged",
    "Stale": "badge-stale",
    "Spoiled": "badge-spoiled",
    "Rotten": "badge-rotten",
}

CLASS_BAR_COLORS = {
    "Fresh": "#10B981",
    "Slightly Aged": "#F59E0B",
    "Stale": "#F97316",
    "Spoiled": "#EF4444",
    "Rotten": "#9333EA",
}

@st.cache_resource(show_spinner=False)
def load_food_model():
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
    if not model_path:
        raise FileNotFoundError(f"Model file not found in paths: {candidate_paths}")

    # Try TensorFlow first (if on Streamlit Cloud Python 3.10/3.11 Linux)
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        # Fallback to Keras 3 with numpy backend (Windows Python 3.14 or light environments)
        os.environ.setdefault("KERAS_BACKEND", "numpy")
        import keras
        return keras.saving.load_model(model_path, compile=False)

try:
    model = load_food_model()
except Exception as e:
    st.error(f"⚠️ Model initialization warning: {e}")
    model = None

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------------------------------------------------------------------
# Main Interactive Diagnostic Workspace
# ---------------------------------------------------------------------------
st.markdown("### 🔬 AI Diagnostic Workspace")
st.caption("Upload a fruit/vegetable photo or click a sample to run real-time MobileNetV2 freshness evaluation.")

col_input, col_output = st.columns([1, 1.15], gap="large")

# Persistent selection in session_state
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None
if "current_image_name" not in st.session_state:
    st.session_state["current_image_name"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []

with col_input:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-header">
            <span>📷</span> Produce Image Input
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop produce photo here (JPG, JPEG, PNG, WebP):",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader",
        help="Upload a clear produce image for biological decay evaluation"
    )

    if uploaded_file is not None:
        try:
            st.session_state["current_image"] = Image.open(uploaded_file)
            st.session_state["current_image_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Could not read uploaded image: {e}")

    # Instant Produce Sample Chips
    st.markdown("<p style='font-size:13px; font-weight:600; color:#64748B; margin:14px 0 6px 0;'>Or select a test sample:</p>", unsafe_allow_html=True)
    
    test_samples_dir = os.path.join(os.path.dirname(__file__), "Test Samples")
    sample_options = [
        ("Fresh Okra", "fresh ocra.jpg"),
        ("Fresh Mango", "FreshMango_40.png"),
        ("Rotten Apple", "820_RottenApple_71.jpg"),
        ("Rotten Banana", "251_RottenBanana_99.png"),
        ("Fresh Strawberry", "723_FreshStrawberry_74.jpg"),
        ("Rotten Capsicum", "403_RottenCapsicum_86.jpg"),
    ]
    
    chip_cols = st.columns(3)
    for idx, (label, fname) in enumerate(sample_options):
        c_idx = idx % 3
        with chip_cols[c_idx]:
            if st.button(label, key=f"chip_{fname}"):
                sample_file = os.path.join(test_samples_dir, fname)
                if os.path.exists(sample_file):
                    st.session_state["current_image"] = Image.open(sample_file)
                    st.session_state["current_image_name"] = label

    # Image Preview
    if st.session_state["current_image"] is not None:
        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        st.image(
            st.session_state["current_image"],
            caption=f"Selected: {st.session_state['current_image_name']}",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Reset Button
        if st.button("🔄 Clear / Select Another Image", key="clear_btn"):
            st.session_state["current_image"] = None
            st.session_state["current_image_name"] = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-header">
            <span>🧠</span> Diagnostic Results & Quality Assessment
        </div>
    """, unsafe_allow_html=True)

    if st.session_state["current_image"] is None:
        st.markdown("""
        <div class="empty-state-box">
            <div style="font-size: 38px; margin-bottom: 8px;">🥑</div>
            <div style="font-family:'Outfit', sans-serif; font-size:18px; font-weight:700; color:#1E293B;">Awaiting Produce Photo</div>
            <p style="font-size:13px; max-width:320px; margin:8px auto 0 auto; line-height:1.5;">
                Upload a photo on the left or select a sample produce button to run the MobileNetV2 neural network diagnostic.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Run inference
        if model is None:
            st.error("Model is not loaded. Please verify model weights.")
        else:
            with st.spinner("Evaluating cellular decay & pigmentation..."):
                t_start = time.time()
                input_tensor = preprocess_image(st.session_state["current_image"])
                preds = model.predict(input_tensor, verbose=0)
                latency_ms = (time.time() - t_start) * 1000

                probs = [float(p) for p in preds[0]]
                pred_idx = int(np.argmax(probs))
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100.0

            # Status Badge
            badge_class = CLASS_BADGE_MAP.get(pred_class, "badge-fresh")
            st.markdown(f"""
            <div class="diag-badge {badge_class}">
                <div>
                    <div class="badge-label-small">AI Freshness Classification</div>
                    <div class="badge-class-name">{pred_class}</div>
                </div>
                <div class="badge-conf-val">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Culinary Guidance
            st.markdown(f"""
            <div class="guidance-box">
                <b style="color:#0F172A;">Culinary & Handling Advice:</b><br>
                {CLASS_DESCRIPTIONS[pred_class]}
            </div>
            """, unsafe_allow_html=True)

            # Probability Bars
            st.markdown("<p style='font-family:Outfit, sans-serif; font-size:15px; font-weight:700; margin-bottom:10px;'>📊 Stage Confidence Breakdown</p>", unsafe_allow_html=True)
            for name, prob in zip(CLASS_NAMES, probs):
                pct = prob * 100.0
                bar_color = CLASS_BAR_COLORS.get(name, "#10B981")
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:600; margin-bottom:3px;">
                    <span>{name}</span>
                    <span style="font-family:'JetBrains Mono', monospace;">{pct:.1f}%</span>
                </div>
                <div style="background:#F1F5F9; border-radius:9999px; height:8px; width:100%; margin-bottom:10px; overflow:hidden;">
                    <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:9999px;"></div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(f"⚡ Latency: `{latency_ms:.1f}ms` &bull; Input: `128×128 RGB` &bull; Weights: `MobileNetV2 (11.5 MB)`")

            # Add to session history
            now_str = datetime.now().strftime("%I:%M:%S %p")
            item_record = {
                "Item": st.session_state["current_image_name"],
                "Result": pred_class,
                "Confidence": f"{confidence:.1f}%",
                "Time": now_str,
            }
            if not st.session_state["history"] or st.session_state["history"][0]["Time"] != now_str:
                st.session_state["history"].insert(0, item_record)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5-Stage Biological Quality Spectrum (Bento Grid)
# ---------------------------------------------------------------------------
st.markdown("<br><hr style='border:none; border-top:1px solid #E2E8F0; margin:24px 0;'><br>", unsafe_allow_html=True)
st.markdown("### 📋 5-Stage Quality Spectrum")
st.caption("Standardized biological taxonomy used by FreshSense to grade produce deterioration.")

st.markdown("""
<div class="bento-grid">
    <div class="bento-card" style="border-top: 3px solid #10B981;">
        <div class="bento-header">
            <span style="font-size:20px;">🟢</span>
            <span class="bento-title" style="color:#065F46;">Fresh</span>
        </div>
        <div class="bento-desc">Peak firmness, bright natural pigmentation, no microbial or oxidation activity. Ideal for raw eating.</div>
    </div>
    <div class="bento-card" style="border-top: 3px solid #F59E0B;">
        <div class="bento-header">
            <span style="font-size:20px;">🟡</span>
            <span class="bento-title" style="color:#92400E;">Slightly Aged</span>
        </div>
        <div class="bento-desc">Minor surface dulling or slight moisture loss. Fully safe; recommend preparing within 48 hours.</div>
    </div>
    <div class="bento-card" style="border-top: 3px solid #F97316;">
        <div class="bento-header">
            <span style="font-size:20px;">🟠</span>
            <span class="bento-title" style="color:#9A3412;">Stale</span>
        </div>
        <div class="bento-desc">Wrinkling, loss of crunch, and mild aromatic decline. Great for purees, soups, or cooking.</div>
    </div>
    <div class="bento-card" style="border-top: 3px solid #EF4444;">
        <div class="bento-header">
            <span style="font-size:20px;">🔴</span>
            <span class="bento-title" style="color:#991B1B;">Spoiled</span>
        </div>
        <div class="bento-desc">Early fungal invasion, deep discoloration, and sour aroma. Unfit for consumption; discard.</div>
    </div>
    <div class="bento-card" style="border-top: 3px solid #9333EA;">
        <div class="bento-header">
            <span style="font-size:20px;">🟣</span>
            <span class="bento-title" style="color:#6B21A8;">Rotten</span>
        </div>
        <div class="bento-desc">Advanced biological decay, spores, and liquid liquefaction. Hazardous bio-waste — dispose safely.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session History Table
# ---------------------------------------------------------------------------
if st.session_state["history"]:
    st.markdown("### 🕒 Recent Scans in This Session")
    st.table(st.session_state["history"][:8])

# ---------------------------------------------------------------------------
# Team Members & Hackathon Footer
# ---------------------------------------------------------------------------
st.markdown("<br><hr style='border:none; border-top:1px solid #E2E8F0; margin:24px 0;'><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#0F172A; border-radius:18px; padding:32px 36px; color:#F8FAFC;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:20px;">
        <div>
            <div style="font-family:'Outfit', sans-serif; font-size:22px; font-weight:800; color:#34D399; margin-bottom:6px;">
                Pak-Angels Mid Program Hackathon
            </div>
            <div style="font-size:14px; color:#94A3B8; max-width:440px;">
                FreshSense AI Food Freshness Detection Platform. Built with Google Antigravity, MobileNetV2, and Python.
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:12px 24px; font-size:13px;">
            <div>
                <b style="color:#FFFFFF;">Muhammad Faseeh (GL)</b><br>
                <span style="color:#94A3B8;">Group Leader & AI Lead</span>
            </div>
            <div>
                <b style="color:#FFFFFF;">Umamah Ibreeq Zafar</b><br>
                <span style="color:#94A3B8;">CV Research & Dataset</span>
            </div>
            <div>
                <b style="color:#FFFFFF;">Anumta Nadeem</b><br>
                <span style="color:#94A3B8;">UI/UX Design Systems</span>
            </div>
            <div>
                <b style="color:#FFFFFF;">Zafar Aman Khattak</b><br>
                <span style="color:#94A3B8;">System Architecture & QA</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
