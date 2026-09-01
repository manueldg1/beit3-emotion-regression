import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling and larger typography
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle and header styling */
    .sub-title {
        font-size: 1.3rem !important;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Input section header */
    .section-header {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Result metric cards */
    .metric-card {
        background-color: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 1.3rem !important;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .metric-interval {
        font-size: 1.1rem !important;
        font-weight: 500;
        color: #475569;
        background-color: #E0F2FE;
        padding: 6px 12px;
        border-radius: 20px;
        display: inline-block;
    }

    /* Final emotion outcome banner */
    .emotion-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
    }
    .emotion-banner-title {
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    .emotion-banner-value {
        font-size: 3rem !important;
        font-weight: 900;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">🎭 Multimodal Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by <b>BEiT-3</b> architecture and <b>OT-CP+</b> interval estimation</div>', unsafe_allow_html=True)

st.divider()

# Layout layout: Left column for Inputs, Right column for Info / Preview
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">📥 Provide Input Data</div>', unsafe_allow_html=True)
    user_text = st.text_area("Text Content:", placeholder="Type a message or sentence here...", height=120)
    uploaded_file = st.file_uploader("Image Content:", type=["jpg", "jpeg", "png"])

with col_right:
    st.markdown('<div class="section-header">🖼️ Image Preview</div>', unsafe_allow_html=True)
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
    else:
        st.info("No image uploaded yet. Upload an image to view preview.")

st.markdown("<br>", unsafe_allow_html=True)

# Submit Button
btn_container = st.container()
with btn_container:
    analyze_btn = st.button("🚀 Analyze Emotion", type="primary", use_container_width=True)

# API Endpoint URL
API_URL = "https://writing-makes-counting-missouri.trycloudflare.com/predict"

# Action logic
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
                    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

                    # Display metrics in custom styled cards
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

                    # Final emotion output banner
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
                st.error(f"🔌 Connection Error: Could not reach FastAPI server at `{API_URL}`. Details: {e}")
