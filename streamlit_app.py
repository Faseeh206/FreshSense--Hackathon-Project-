import os
import time
import base64
from datetime import datetime

# Configure Keras backend
os.environ.setdefault("KERAS_BACKEND", "numpy")

import streamlit as st
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FreshSense — AI Food Freshness Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Load and Embed Assets (Logo & CSS)
# ---------------------------------------------------------------------------
def get_base64_image(image_path: str) -> str:
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

# Candidate paths for logo
logo_candidates = [
    os.path.join(os.path.dirname(__file__), "static", "logo.jpg"),
    os.path.join(os.path.dirname(__file__), "FreshSense", "static", "logo.jpg"),
]
logo_path = next((p for p in logo_candidates if os.path.exists(p)), "")
LOGO_B64 = get_base64_image(logo_path)
LOGO_SRC = f"data:image/jpeg;base64,{LOGO_B64}" if LOGO_B64 else ""

# Load EXACT style.css from FreshSense/static/css/style.css
css_candidates = [
    os.path.join(os.path.dirname(__file__), "FreshSense", "static", "css", "style.css"),
    os.path.join(os.path.dirname(__file__), "static", "css", "style.css"),
]
css_path = next((p for p in css_candidates if os.path.exists(p)), "")
ORIGINAL_CSS = ""
if css_path and os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        ORIGINAL_CSS = f.read()

# Replace any /static/logo.jpg in the CSS or HTML with base64 data
if LOGO_SRC:
    ORIGINAL_CSS = ORIGINAL_CSS.replace("/static/logo.jpg", LOGO_SRC)

# ---------------------------------------------------------------------------
# Streamlit Specific CSS Integration
# ---------------------------------------------------------------------------
STREAMLIT_INTEGRATION_CSS = """
/* Streamlit UI Reset to make custom HTML look 100% native */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header[data-testid="stHeader"] {visibility: hidden !important; height: 0px !important;}

.stApp {
    background-color: #F8FAFC !important;
    font-family: 'Work Sans', sans-serif !important;
    color: #475569 !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Ensure container max-width matches the original navbar and hero */
.container {
    max-width: 1160px !important;
    margin: 0 auto !important;
    padding: 0 28px !important;
}

/* File Uploader styling to blend with the original dropzone */
[data-testid="stFileUploader"] {
    margin-top: 10px !important;
    margin-bottom: 12px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FAFAFC !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 24px 16px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #059669 !important;
    background: #F0FDF4 !important;
}

/* Sample chip buttons */
.stButton > button {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #334155 !important;
    font-family: 'Work Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    border-color: #10B981 !important;
    color: #059669 !important;
    background: #ECFDF5 !important;
    transform: translateY(-1px) !important;
}
"""

st.markdown(f"<style>{ORIGINAL_CSS}\n{STREAMLIT_INTEGRATION_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Neural Network Model Loader (MobileNetV2)
# ---------------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["Fresh", "Slightly Aged", "Stale", "Spoiled", "Rotten"]

CLASS_DESCRIPTIONS = {
    "Fresh": "This item looks fresh and safe to consume. No visible signs of aging or spoilage.",
    "Slightly Aged": "Minor signs of aging detected. Still safe for consumption but best used soon.",
    "Stale": "Noticeable aging present. Quality has declined — consume with caution.",
    "Spoiled": "Significant spoilage detected. This item is not recommended for consumption.",
    "Rotten": "Severe decay detected. This item should be discarded immediately.",
}

CLASS_KEYS = {
    "Fresh": "fresh",
    "Slightly Aged": "slightly-aged",
    "Stale": "stale",
    "Spoiled": "spoiled",
    "Rotten": "rotten"
}

@st.cache_resource(show_spinner=False)
def load_mobilenet():
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
    path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("MobileNetV2_best_model.keras not found.")

    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        os.environ.setdefault("KERAS_BACKEND", "numpy")
        import keras
        return keras.saving.load_model(path, compile=False)

try:
    model = load_mobilenet()
except Exception as e:
    model = None

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------------------------------------------------------------------
# Top Navigation (EXACT from index.html)
# ---------------------------------------------------------------------------
nav_logo_html = f'<img src="{LOGO_SRC}" alt="FreshSense Logo" class="nav-logo-img">' if LOGO_SRC else '<span style="font-size:24px;">🍃</span>'

st.markdown(f"""
<header class="navbar">
    <div class="container nav-inner">
        <a href="/" class="nav-brand">
            {nav_logo_html}
            <span class="brand-title">FreshSense</span>
            <span class="brand-badge">AI 2.0</span>
        </a>
        <nav class="nav-links">
            <a href="#detect">Detect Freshness</a>
            <a href="#scale">Quality Scale</a>
            <a href="#about">Architecture</a>
            <a href="#history">Scan History</a>
        </nav>
    </div>
</header>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Section (EXACT from index.html)
# ---------------------------------------------------------------------------
st.markdown("""
<section class="hero">
    <div class="hero-bg-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
    </div>
    <div class="container hero-inner">
        <div class="hero-tag">
            <span class="pulse-dot"></span>
            Deep Learning Vision System &bull; 82% F1 Accuracy
        </div>
        <h1 class="hero-title">
            Know your food freshness.<br>
            <span class="gradient-text">Before you taste or serve.</span>
        </h1>
        <p class="hero-desc">
            Instant computer vision classification powered by MobileNetV2. Upload any fruit or vegetable photo to evaluate quality across five distinct biological degradation stages.
        </p>
        <div class="hero-actions">
            <a href="#detect" class="btn btn-primary">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Analyze Food Image
            </a>
            <a href="#about" class="btn btn-secondary">
                View Architecture
            </a>
        </div>
        <div class="hero-stats">
            <div class="stat-box">
                <span class="stat-value">82%</span>
                <span class="stat-name">Validation Accuracy</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat-box">
                <span class="stat-value">13</span>
                <span class="stat-name">Food Classes Covered</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat-box">
                <span class="stat-value">5</span>
                <span class="stat-name">Freshness Categories</span>
            </div>
            <div class="stat-sep"></div>
            <div class="stat-box">
                <span class="stat-value">&lt; 1s</span>
                <span class="stat-name">Local CPU Inference</span>
            </div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Detection Workspace Section (EXACT structure from index.html)
# ---------------------------------------------------------------------------
st.markdown("""
<section id="detect" class="section section-detect">
    <div class="container">
        <div class="section-header text-center">
            <div class="section-label">AI Diagnostic Workspace</div>
            <h2 class="section-title">Upload Food for Freshness Analysis</h2>
            <p class="section-subtitle">Supports photos of apples, bananas, tomatoes, carrots, mangoes, cucumbers, and more.</p>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# Session State initialization
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "selected_name" not in st.session_state:
    st.session_state["selected_name"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []

# Workspace layout container
ws_container = st.container()
with ws_container:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Check if an image is selected
        if st.session_state["selected_image"] is None:
            st.markdown("""
            <div class="dropzone-card" id="uploadZone" style="cursor:default; min-height:360px;">
                <div class="dropzone-body" id="uploadContent">
                    <div class="dropzone-icon-circle">
                        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                    </div>
                    <h3 class="dropzone-title">Drag & drop your food image here</h3>
                    <p class="dropzone-instruction">or select an image using the uploader below</p>
                    <div class="dropzone-badges">
                        <span class="file-chip">JPG</span>
                        <span class="file-chip">JPEG</span>
                        <span class="file-chip">PNG</span>
                        <span class="file-chip">WEBP</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload food image",
                type=["jpg", "jpeg", "png", "webp"],
                key="file_up",
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                st.session_state["selected_image"] = Image.open(uploaded_file)
                st.session_state["selected_name"] = uploaded_file.name
                st.rerun()

            # Fast Test Produce Chips
            st.markdown("<p style='font-size:12px; font-weight:600; color:#64748B; margin:16px 0 6px 0; text-transform:uppercase; letter-spacing:0.05em;'>Or Select Sample Produce:</p>", unsafe_allow_html=True)
            test_samples_dir = os.path.join(os.path.dirname(__file__), "Test Samples")
            sample_chips = [
                ("🥒 Fresh Okra", "fresh ocra.jpg"),
                ("🥭 Fresh Mango", "FreshMango_40.png"),
                ("🍎 Rotten Apple", "820_RottenApple_71.jpg"),
                ("🍌 Rotten Banana", "251_RottenBanana_99.png"),
                ("🍓 Fresh Strawberry", "723_FreshStrawberry_74.jpg"),
                ("🫑 Rotten Capsicum", "403_RottenCapsicum_86.jpg"),
            ]
            chip_cols = st.columns(3)
            for idx, (lbl, fn) in enumerate(sample_chips):
                with chip_cols[idx % 3]:
                    if st.button(lbl, key=f"btn_{fn}"):
                        p = os.path.join(test_samples_dir, fn)
                        if os.path.exists(p):
                            st.session_state["selected_image"] = Image.open(p)
                            st.session_state["selected_name"] = lbl
                            st.rerun()
        else:
            # Render image preview using original .preview-wrapper
            img = st.session_state["selected_image"]
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_data_url = f"data:image/jpeg;base64,{img_b64}"

            st.markdown(f"""
            <div class="dropzone-card" style="padding:0; min-height:400px; cursor:default;">
                <div class="preview-wrapper" style="display:flex; width:100%; min-height:400px; align-items:center; justify-content:center; background:#0F172A; position:relative;">
                    <img src="{img_data_url}" alt="Food scan preview" class="preview-img" style="max-height:340px; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,0.4);">
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑️ Remove Image / Scan Another", key="btn_remove"):
                st.session_state["selected_image"] = None
                st.session_state["selected_name"] = ""
                st.rerun()

    with col_right:
        if st.session_state["selected_image"] is None:
            # Idle State (EXACT from index.html)
            st.markdown("""
            <div class="result-card-container" style="min-height:400px;">
                <div class="state-placeholder" id="resultEmpty">
                    <div class="placeholder-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 14 14"></polyline>
                        </svg>
                    </div>
                    <h4>Awaiting Input Image</h4>
                    <p>Upload a food photo on the left. The neural network will extract features, classify freshness, and calculate confidence distribution in real-time.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Run MobileNetV2 prediction
            input_arr = preprocess(st.session_state["selected_image"])
            preds = model.predict(input_arr, verbose=0)
            probs = [float(p) for p in preds[0]]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100.0
            badge_key = CLASS_KEYS.get(pred_class, "fresh")
            desc = CLASS_DESCRIPTIONS.get(pred_class, "")

            # Generate Probability Bars HTML (EXACT from app.js)
            bars_html = ""
            for name, prob in zip(CLASS_NAMES, probs):
                pct = prob * 100.0
                bar_k = CLASS_KEYS.get(name, "fresh")
                bars_html += f"""
                <div class="prob-row">
                    <span class="prob-label">{name}</span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill bar-{bar_k}" style="width:{pct:.1f}%;"></div>
                    </div>
                    <span class="prob-value">{pct:.1f}%</span>
                </div>
                """

            # Save to history
            now_str = datetime.now().strftime("%b %d, %Y at %I:%M %p")
            record = {
                "filename": st.session_state["selected_name"],
                "predicted_class": pred_class,
                "confidence": round(confidence, 1),
                "timestamp": now_str,
                "badge": badge_key
            }
            if not st.session_state["history"] or st.session_state["history"][0].get("timestamp") != now_str:
                st.session_state["history"].insert(0, record)

            # Complete State (EXACT from index.html)
            st.markdown(f"""
            <div class="result-card-container" style="min-height:400px;">
                <div class="state-result" id="resultCard">
                    <div class="result-headline">
                        <div class="badge-wrapper">
                            <span class="result-tag badge-{badge_key}" id="resultBadge">{pred_class}</span>
                        </div>
                        <span class="confidence-indicator" id="resultConfidence">{confidence:.1f}% Confidence</span>
                    </div>

                    <p class="result-summary" id="resultDescription">
                        {desc}
                    </p>

                    <div class="probability-section">
                        <div class="prob-title">Confidence Distribution Across 5 Classes</div>
                        <div class="chart-bars" id="probChart">
                            {bars_html}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5-Stage Freshness Spectrum Section (EXACT from index.html)
# ---------------------------------------------------------------------------
st.markdown("""
<section id="scale" class="section section-scale">
    <div class="container">
        <div class="section-header">
            <div class="section-label">Classification Taxonomy</div>
            <h2 class="section-title">The 5-Stage Freshness Spectrum</h2>
            <p class="section-subtitle">Our Agglomerative clustering pipeline groups images into five visually distinctive stages of deterioration.</p>
        </div>

        <div class="scale-grid">
            <div class="scale-card scale-c1">
                <div class="scale-card-header">
                    <span class="scale-number">01</span>
                    <span class="scale-status">Fresh</span>
                </div>
                <div class="scale-indicator"></div>
                <p class="scale-desc">Harvest-grade quality, vibrant skin pigmentation, crisp texture, and zero noticeable biological degradation.</p>
            </div>
            <div class="scale-card scale-c2">
                <div class="scale-card-header">
                    <span class="scale-number">02</span>
                    <span class="scale-status">Slightly Aged</span>
                </div>
                <div class="scale-indicator"></div>
                <p class="scale-desc">Early oxidation or minor moisture depletion. Completely safe to consume; recommended for near-term preparation.</p>
            </div>
            <div class="scale-card scale-c3">
                <div class="scale-card-header">
                    <span class="scale-number">03</span>
                    <span class="scale-status">Stale</span>
                </div>
                <div class="scale-indicator"></div>
                <p class="scale-desc">Noticeable dehydration, softened cellular structure, or loss of aroma. Significant decline in culinary quality.</p>
            </div>
            <div class="scale-card scale-c4">
                <div class="scale-card-header">
                    <span class="scale-number">04</span>
                    <span class="scale-status">Spoiled</span>
                </div>
                <div class="scale-indicator"></div>
                <p class="scale-desc">Discoloration, surface softening, and onset of mold or bacterial presence. Not recommended for human intake.</p>
            </div>
            <div class="scale-card scale-c5">
                <div class="scale-card-header">
                    <span class="scale-number">05</span>
                    <span class="scale-status">Rotten</span>
                </div>
                <div class="scale-indicator"></div>
                <p class="scale-desc">Advanced microbial degradation, severe fungal growth, structural breakdown. Discard immediately to protect surroundings.</p>
            </div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Architecture & Engineering Section (EXACT from index.html)
# ---------------------------------------------------------------------------
st.markdown("""
<section id="about" class="section section-about">
    <div class="container">
        <div class="section-header">
            <div class="section-label">Engineering & Tech Stack</div>
            <h2 class="section-title">Built on Computer Vision Research</h2>
            <p class="section-subtitle">Trained on thousands of curated fruit and vegetable images across 13 diverse species.</p>
        </div>

        <div class="cards-bento">
            <div class="bento-card bento-emerald">
                <div class="bento-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                </div>
                <h3>MobileNetV2 Neural Network</h3>
                <p>Features inverted residual bottlenecks and depthwise separable convolutions to achieve high accuracy with minimal parameter overhead.</p>
                <div class="bento-footer">Agglomerative weights &bull; 82% accuracy</div>
            </div>

            <div class="bento-card bento-amber">
                <div class="bento-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                </div>
                <h3>Keras 3 NumPy Engine</h3>
                <p>Silent CPU inference powered by Keras 3 with NumPy math primitives. Zero GPU dependencies or heavy CUDA setup needed.</p>
                <div class="bento-footer">Zero-latency backend pipeline</div>
            </div>

            <div class="bento-card bento-rose">
                <div class="bento-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"></path><line x1="4" y1="22" x2="4" y2="15"></line></svg>
                </div>
                <h3>13 Produce Categories</h3>
                <p>Trained on Apple, Banana, Bell Pepper, Bitter Gourd, Capsicum, Carrot, Cucumber, Mango, Okra, Orange, Potato, Strawberry, and Tomato.</p>
                <div class="bento-footer">6.4 GB curated dataset</div>
            </div>

            <div class="bento-card bento-indigo">
                <div class="bento-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>
                </div>
                <h3>Microservice Architecture</h3>
                <p>RESTful JSON endpoints handling multipart image streaming, image normalization (128&times;128), and transaction history persistence.</p>
                <div class="bento-footer">Production-grade Python API</div>
            </div>

            <div class="bento-card bento-cyan">
                <div class="bento-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                </div>
                <h3>Computer Vision Preprocessing</h3>
                <p>Gaussian blur noise reduction, Canny edge detection, and RGB normalization at 128&times;128 resolution for consistent inference.</p>
                <div class="bento-footer">Robust invariant transforms</div>
            </div>

            <div class="bento-card bento-purple">
                <div class="bento-icon">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>
                </div>
                <h3>Agglomerative Cluster Hierarchy</h3>
                <p>Euclidean metric with Ward linkage grouping to discover non-linear quality degradation boundaries without manual annotation bias.</p>
                <div class="bento-footer">Unsupervised quality discovery</div>
            </div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scan History Section (EXACT from index.html)
# ---------------------------------------------------------------------------
history_html = ""
if st.session_state["history"]:
    cards_html = ""
    for item in st.session_state["history"][:6]:
        b_key = item.get("badge", "fresh")
        cards_html += f"""
        <div class="history-item-card" style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:18px 20px; box-shadow:0 2px 8px rgba(15,23,42,0.04);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span class="result-tag badge-{b_key}" style="font-size:12px; padding:4px 12px;">{item['predicted_class']}</span>
                <span style="font-family:'JetBrains Mono', monospace; font-size:13px; font-weight:600; color:#475569;">{item['confidence']}%</span>
            </div>
            <div style="font-size:14px; font-weight:700; color:#0F172A; margin-bottom:4px;">{item['filename']}</div>
            <div style="font-size:12px; color:#94A3B8;">{item['timestamp']}</div>
        </div>
        """
    history_html = f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:16px; margin-top:20px;">
        {cards_html}
    </div>
    """
else:
    history_html = """
    <div class="history-empty-box" id="historyEmpty">
        <div class="empty-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        </div>
        <h4>No previous scan records</h4>
        <p>Upload a food picture in the workspace above to generate your first freshness record.</p>
    </div>
    """

st.markdown(f"""
<section id="history" class="section section-history">
    <div class="container">
        <div class="history-topbar">
            <div>
                <div class="section-label">Audit Log</div>
                <h2 class="section-title">Diagnostic History</h2>
                <p class="section-subtitle">Review recent food classification records and confidence metrics.</p>
            </div>
        </div>
        {history_html}
    </div>
</section>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer (EXACT from index.html)
# ---------------------------------------------------------------------------
footer_logo_html = f'<img src="{LOGO_SRC}" alt="FreshSense" class="footer-logo">' if LOGO_SRC else '<span style="font-size:24px;">🍃</span>'

st.markdown(f"""
<footer class="footer">
    <div class="container footer-content">
        <div class="footer-left">
            {footer_logo_html}
            <div>
                <div class="footer-brand-name">FreshSense</div>
                <div class="footer-sub">Deep Learning Food Spoilage Diagnostic Platform &bull; Pak-Angels Mid Program Hackathon</div>
            </div>
        </div>
        <div class="footer-right">
            <span>Muhammad Faseeh (GL)</span> &bull; 
            <span>Umamah Ibreeq Zafar</span> &bull; 
            <span>Anumta Nadeem</span> &bull; 
            <span>Zafar Aman Khattak</span>
        </div>
    </div>
</footer>
""", unsafe_allow_html=True)
