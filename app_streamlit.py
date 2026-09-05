import streamlit as st
import requests
import base64

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Emotion Recognition | Multimodal BEiT-3",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= MAIN APP CSS (professional / single-accent theme) =================
st.markdown("""
<style>
    :root {
        --accent: #3B82F6;
        --accent-dark: #2563EB;
        --bg-deep: #0F172A;
        --bg-panel: #111827;
        --border-soft: rgba(148, 163, 184, 0.18);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
    }

    .stApp {
        background: linear-gradient(180deg, #0B1120 0%, #0F172A 100%);
        color: var(--text-primary);
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        min-height: 100vh;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1.5rem;}

    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        letter-spacing: -3px;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .hero-title .accent { color: var(--text-primary); }

    .hero-subtitle {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 3rem;
        text-align: center;
    }

    .section-header {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary);
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-left: 3px solid var(--accent);
        padding-left: 10px;
    }

    .stButton>button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 900 !important;
        padding: 1rem 2.5rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 50px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    .stButton>button:hover {
        background: var(--accent-dark) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    .stTextArea textarea {
        background: #F8FAFC !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-size: 1.05rem !important;
    }
    .stTextArea textarea::placeholder { color: #64748B !important; opacity: 1 !important; }
    .stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important; }

    /* Widget labels ("Text content:", "Image content:") — default Streamlit
       gray label color is too dark to read against this dark background */
    .stTextArea label p, .stFileUploader label p,
    [data-testid="stWidgetLabel"] p {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    div[data-testid="stAlert"] {
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stAlert"] * { color: var(--text-secondary) !important; }

    .image-preview-frame {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        border: 1px solid var(--border-soft);
    }

    [data-testid="stSidebar"] { background: var(--bg-panel) !important; border-right: 1px solid var(--border-soft) !important; }

    .custom-upload-wrapper [data-testid='stFileUploader'] { width: 100%; }
    .custom-upload-wrapper [data-testid='stFileUploader'] section {
        background-color: #F8FAFC;
        border: 1.5px dashed #CBD5E1;
        border-radius: 10px;
        padding: 26px;
        transition: all 0.2s ease;
        text-align: center;
        color: #0F172A;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section:hover {
        border-color: var(--accent);
        background-color: #F1F5F9;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button {
        display: inline-block !important;
        width: auto !important;
        height: auto !important;
        opacity: 1 !important;
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.3rem !important;
        margin-top: 10px;
        transition: all 0.2s ease !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button p,
    .custom-upload-wrapper [data-testid='stFileUploader'] section button span,
    .custom-upload-wrapper [data-testid='stFileUploader'] section button svg {
        color: #000000 !important;
        fill: #000000 !important;
        stroke: #000000 !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button:hover {
        background: var(--accent) !important;
        color: #ffffff !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button:hover p,
    .custom-upload-wrapper [data-testid='stFileUploader'] section button:hover span,
    .custom-upload-wrapper [data-testid='stFileUploader'] section button:hover svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section > div > div span {
        color: #0F172A !important;
        font-family: inherit !important;
        font-weight: 600 !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section small {
        color: #334155 !important;
        opacity: 1 !important;
    }

    /* ---- Metric cards (Valence / Arousal) ---- */
    .metric-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-top: 3px solid var(--accent);
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover { border-color: var(--accent); transform: translateY(-2px); }
    .metric-label {
        font-size: 1.2rem !important;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.6rem;
    }
    .metric-value {
        font-size: 3rem !important;
        font-weight: 900;
        color: var(--text-primary);
        margin-bottom: 0.6rem;
    }
    .metric-interval {
        font-size: 1rem !important;
        font-weight: 600;
        color: var(--accent);
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
    }

    /* ---- Final emotion banner ---- */
    .emotion-banner {
        background: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--accent);
        color: var(--text-primary);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 1.8rem;
    }
    .emotion-banner-title {
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: var(--text-secondary);
    }
    .emotion-banner-value {
        font-size: 3.2rem !important;
        font-weight: 900;
        margin-top: 0.5rem;
        letter-spacing: 1px;
        color: var(--accent);
    }

    hr { border-color: var(--border-soft) !important; }

    /* --- MOBILE RESPONSIVENESS --- */
    @media only screen and (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .hero-title { font-size: 3rem !important; }
        .hero-subtitle { font-size: 1.1rem !important; letter-spacing: 2px !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: auto !important; }
        .metric-value { font-size: 2.2rem !important; }
        .emotion-banner-value { font-size: 2.2rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="hero-title">🎭 Emotion <span class="accent">Recognition</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">BEiT-3 and Uncertainty Quantification via Adaptive OTCP+</div>',
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">Input Data</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Text content:",
        placeholder="Type a message or sentence here...",
        height=140
    )
    st.markdown('<div class="custom-upload-wrapper">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Image content:", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">Image Preview</div>', unsafe_allow_html=True)
    if uploaded_file:
        img_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        st.markdown(f"""
            <div class="image-preview-frame">
                <img src="data:{uploaded_file.type};base64,{img_b64}"
                     width="100%" style="display: block;">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No image uploaded yet. Upload an image to view the preview.")

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Analyse Contents and Predict Emotion", type="primary", use_container_width=True)

# ================= API ENDPOINT =================
API_URL = "https://beit3-emotion-api-1011510519247.europe-west1.run.app/predict"

# ================= ACTION LOGIC =================
if analyze_btn:
    if not user_text and not uploaded_file:
        st.error("⚠️ Please provide at least a text snippet or an image to run the prediction.")
    else:
        files = {}
        data = {}

        if uploaded_file:
            files['image'] = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        if user_text:
            data['text'] = user_text

        with st.spinner("Processing multimodal representation..."):
            try:
                response = requests.post(API_URL, data=data, files=files if files else None)

                if response.status_code == 200:
                    result = response.json()

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

                    res_col1, res_col2 = st.columns(2, gap="medium")

                    with res_col1:
                        val_score = round(result.get("Valence", 0.0), 4)
                        val_int = result.get("Valence_Interval", "N/A")
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Valence</div>
                                <div class="metric-value">{val_score}</div>
                                <div class="metric-interval">OT-CP+: <b>{val_int}</b></div>
                            </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        aro_score = round(result.get("Arousal", 0.0), 4)
                        aro_int = result.get("Arousal_Interval", "N/A")
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Arousal</div>
                                <div class="metric-value">{aro_score}</div>
                                <div class="metric-interval">OT-CP+: <b>{aro_int}</b></div>
                            </div>
                        """, unsafe_allow_html=True)

                    predicted_emotion = result.get("Emotion", "N/A")
                    st.markdown(f"""
                        <div class="emotion-banner">
                            <div class="emotion-banner-title">Predicted Emotion Category</div>
                            <div class="emotion-banner-value">{predicted_emotion}</div>
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error(f"❌ API Error ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"🔌 Connection Error: could not reach the FastAPI server at `{API_URL}`. Details: {e}")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p><strong>Emotion Recognition — Multimodal System</strong></p>
    <p>BEiT-3 Made Multilingual via the VecMap Framework + Uncertainty Quantification with Optimal Transport-Based Conformal Prediction</p>
    <p><a href="https://github.com/manueldg1" target="_blank" style="color:#3B82F6; text-decoration:none; font-weight:600;">github.com/manueldg1</a></p>
    <p style="font-size: 0.85rem; color: #64748B; max-width: 700px; margin: 12px auto 0 auto; line-height: 1.5;">
        This system was developed exclusively for academic research purposes.
        It is not intended for use in personnel selection, psychological assessment, surveillance,
        or any other decision-making process involving real individuals.
    </p>
</div>
""", unsafe_allow_html=True)
