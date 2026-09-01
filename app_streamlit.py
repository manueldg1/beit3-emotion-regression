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

# ================= MAIN APP CSS (same visual identity as app (1).py) =================
st.markdown("""
<style>
    :root {
        --primary: #00f2fe;
        --primary-glow: #00f2feaa;
        --primary-dark: #4facfe;
        --neon-pink: #ff00ff;
        --neon-purple: #8a2be2;
        --matrix-green: #00ff41;
        --bg-deep: #0a0a12;
        --bg-card: rgba(20, 20, 40, 0.8);
        --text-glow: 0 0 10px rgba(0, 242, 254, 0.7);
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 242, 254, 0.05) 0%, transparent 50%),
            linear-gradient(135deg, #0a0a12 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', system-ui, sans-serif;
        min-height: 100vh;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem;}

    @keyframes float { 0%, 100% { transform: translateY(0px) rotate(0deg); } 50% { transform: translateY(-10px) rotate(1deg); } }
    @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.05); opacity: 0.8; } 100% { transform: scale(1); opacity: 1; } }
    @keyframes slideIn { 0% { opacity: 0; transform: translateX(-30px); } 100% { opacity: 1; transform: translateX(0); } }
    @keyframes scanline { 0% { top: 0%; } 100% { top: 100%; } }
    .floating { animation: float 6s ease-in-out infinite; }
    .pulse-glow { animation: pulse 2s ease-in-out infinite; }
    .slide-in { animation: slideIn 0.6s ease-out; }

    .hero-title { font-size: 5rem; font-weight: 900; letter-spacing: -3px; background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(0, 242, 254, 0.5), 0 0 60px rgba(0, 242, 254, 0.3); margin-bottom: 0; text-align: center; }
    .hero-subtitle {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 3rem;
        text-align: center;
        background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.35);
    }

    .section-header { font-size: 1.5rem !important; font-weight: 700 !important; color: #e2e8f0; margin-top: 1rem; margin-bottom: 0.8rem; letter-spacing: 1px; text-transform: uppercase; }

    .glass-card { background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 24px; padding: 30px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.2); transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1); }
    .glass-card:hover { transform: translateY(-8px) scale(1.02); border-color: var(--primary); box-shadow: 0 15px 40px 0 rgba(0, 242, 254, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.3); }

    .stButton>button { background: linear-gradient(135deg, var(--primary) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%) !important; color: #0f172a !important; border: none !important; font-weight: 800 !important; padding: 1rem 2.5rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; border-radius: 50px !important; transition: all 0.3s ease !important; box-shadow: 0 5px 15px rgba(0, 242, 254, 0.3) !important; }
    .stButton>button:hover { transform: translateY(-3px) scale(1.05) !important; box-shadow: 0 10px 25px rgba(0, 242, 254, 0.5), 0 0 30px rgba(0, 242, 254, 0.3) !important; letter-spacing: 3px !important; }

    .stTextArea textarea { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0, 242, 254, 0.3) !important; border-radius: 16px !important; color: #000000 !important; font-size: 1.05rem !important; }
    .stTextArea textarea::placeholder { color: #334155 !important; opacity: 1 !important; }
    .stTextArea textarea:focus { border-color: var(--primary) !important; box-shadow: 0 0 15px rgba(0, 242, 254, 0.3) !important; }

    .image-preview-frame { position: relative; overflow: hidden; border-radius: 20px; border: 1px solid rgba(0, 242, 254, 0.4); box-shadow: 0 0 25px rgba(0, 242, 254, 0.15), inset 0 0 20px rgba(0, 242, 254, 0.05); transition: all 0.3s ease; }
    .image-preview-frame:hover { border-color: var(--primary); box-shadow: 0 0 35px rgba(0, 242, 254, 0.25); }

    [data-testid="stSidebar"] { background: rgba(10, 10, 18, 0.9) !important; backdrop-filter: blur(20px) !important; border-right: 1px solid rgba(0, 242, 254, 0.2) !important; }

    .custom-upload-wrapper [data-testid='stFileUploader'] { width: 100%; }
    .custom-upload-wrapper [data-testid='stFileUploader'] section {
        background-color: rgba(0, 242, 254, 0.03);
        border: 2px dashed var(--primary);
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
        text-align: center;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section:hover {
        background-color: rgba(0, 242, 254, 0.1);
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
        border-color: var(--neon-pink);
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button {
        display: inline-block !important;
        width: auto !important;
        height: auto !important;
        opacity: 1 !important;
        background: transparent !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 50px !important;
        padding: 0.5rem 1.5rem !important;
        margin-top: 10px;
        transition: all 0.3s ease !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section button:hover {
        background: var(--primary) !important;
        color: var(--bg-deep) !important;
        box-shadow: 0 0 15px var(--primary) !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section > div > div span {
        color: var(--primary) !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
    }
    .custom-upload-wrapper [data-testid='stFileUploader'] section small {
        color: #000000 !important;
        opacity: 1 !important;
    }

    /* ---- Metric cards (Valence / Arousal) ---- */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.36);
        transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: var(--primary); box-shadow: 0 0 25px rgba(0, 242, 254, 0.25); transform: translateY(-4px); }
    .metric-label { font-size: 1.2rem !important; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.6rem; }
    .metric-value {
        font-size: 3rem !important;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        margin-bottom: 0.6rem;
    }
    .metric-interval {
        font-size: 1rem !important;
        font-weight: 600;
        color: #e2e8f0;
        background-color: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
    }

    /* ---- Final emotion banner ---- */
    .emotion-banner {
        background: linear-gradient(135deg, var(--neon-purple) 0%, var(--neon-pink) 50%, var(--primary-dark) 100%);
        color: #0f172a;
        border-radius: 24px;
        padding: 28px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 40px -5px rgba(255, 0, 255, 0.4), 0 0 60px rgba(0, 242, 254, 0.15);
    }
    .emotion-banner-title { font-size: 1.1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 3px; opacity: 0.85; }
    .emotion-banner-value { font-size: 3.2rem !important; font-weight: 900; margin-top: 0.5rem; letter-spacing: 1px; }

    /* --- MOBILE RESPONSIVENESS --- */
    @media only screen and (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .hero-title { font-size: 3rem !important; }
        .hero-subtitle { font-size: 1.1rem !important; letter-spacing: 2px !important; }
        .glass-card { padding: 15px !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: auto !important; }
        .metric-value { font-size: 2.2rem !important; }
        .emotion-banner-value { font-size: 2.2rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="hero-title floating">🎭 EMOTION RECOGNITION</div>', unsafe_allow_html=True)
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
                     width="100%" style="border-radius: 18px; display: block;">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No image uploaded yet. Upload an image to view the preview.")

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Analyse Contents and Predict Emotion", type="primary", use_container_width=True)

# ================= API ENDPOINT =================
API_URL = "https://writing-makes-counting-missouri.trycloudflare.com"

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
                        <div class="emotion-banner pulse-glow">
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
    <p><a href="https://github.com/manueldg1" target="_blank" style="color:#00f2fe; text-decoration:none; font-weight:600;">github.com/manueldg1</a></p>
    <p style="font-size: 0.85rem; color: #94a3b8; max-width: 700px; margin: 12px auto 0 auto; line-height: 1.5;">
        This system was developed exclusively for academic research purposes (Master's thesis).
        It is not intended for use in personnel selection, psychological assessment, surveillance,
        or any other decision-making process involving real individuals.
    </p>
</div>
""", unsafe_allow_html=True)
