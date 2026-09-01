import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="BEiT-3 Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Cyberpunk / Synthwave CSS Theme (Oculus Prime Style)
st.markdown("""
    <style>
    /* Main Background with Radial Purple/Dark Blue Gradient */
    .stApp {
        background: radial-gradient(circle at 80% 20%, #1e0936 0%, #0a0c16 60%, #05060b 100%);
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Neon Title Styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00f0ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
        margin-bottom: 5px;
    }
    
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }
    
    /* Container / Card Styling */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(112, 0, 255, 0.3) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border-radius: 12px;
        padding: 24px;
    }
    
    /* Metrics Box Customization */
    div[data-testid="stMetric"] {
        background: rgba(20, 15, 38, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.08);
    }
    div[data-testid="stMetricLabel"] {
        color: #A78BFA !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(0, 240, 255, 0.5);
    }
    
    /* Neon Action Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff 0%, #7000ff 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 14px 28px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border-radius: 25px !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(112, 0, 255, 0.8) !important;
        transform: translateY(-2px);
    }
    
    /* Form Elements Styling */
    .stTextArea textarea, .stFileUploader section {
        background-color: rgba(10, 14, 26, 0.8) !important;
        border: 1px solid rgba(112, 0, 255, 0.3) !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cloudflare FastAPI Endpoint
FASTAPI_URL = "https://writing-makes-counting-missouri.trycloudflare.com/predict"

# 4. Header Section
st.markdown("<h1 class='main-title'>BEiT-3 EMOTION AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Multimodal Emotion Recognition & OT-CP+ Interval Estimation</p>", unsafe_allow_html=True)
st.markdown("---")

# 5. Main Layout
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Input Data")
    
    text_input = st.text_area(
        "Text Content:", 
        placeholder="Type a sentence or prompt here...",
        height=120
    )
    
    uploaded_file = st.file_uploader(
        "Image Content:", 
        type=["jpg", "jpeg", "png"]
    )
    
    image = None
    if uploaded_file:
        st.markdown("**Image Preview**")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    analyze_btn = st.button("⚡ ANALYZE EMOTION")

with col_output:
    st.subheader("📊 Model Predictions")
    
    if analyze_btn:
        if not text_input and not uploaded_file:
            st.warning("Please provide at least a text prompt or an image to analyze.")
        else:
            with st.spinner("Processing through BEiT-3 neural network..."):
                try:
                    files = {}
                    data = {}
                    
                    if uploaded_file:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                        files = {'file': (uploaded_file.name, img_byte_arr.getvalue(), uploaded_file.type)}
                    
                    if text_input:
                        data = {'text': text_input}

                    # Request to FastAPI Backend
                    response = requests.post(FASTAPI_URL, files=files, data=data)

                    if response.status_code == 200:
                        res_data = response.json()
                        
                        emotion = res_data.get("predicted_emotion", "N/A")
                        val = res_data.get("valence", 0.0)
                        aro = res_data.get("arousal", 0.0)
                        
                        # 1. Predicted Emotion Banner
                        st.markdown(f"""
                            <div style="background: linear-gradient(90deg, rgba(0, 240, 255, 0.2), rgba(112, 0, 255, 0.3)); 
                                        border: 1px solid #00F0FF; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                                <h3 style="margin: 0; color: #FFFFFF; letter-spacing: 1px;">Predicted Emotion: <b style="color: #00F0FF;">{emotion}</b></h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # 2. VAD Metrics Cards
                        m1, m2 = st.columns(2)
                        m1.metric("Valence (V)", f"{val:+.3f}")
                        m2.metric("Arousal (A)", f"{aro:+.3f}")

                        # 3. OT-CP+ Interval Details
                        otcp = res_data.get("otcp_intervals", {})
                        if otcp:
                            with st.expander("🔍 View OT-CP+ Intervals", expanded=False):
                                st.write(f"**Valence Interval:** `[{otcp.get('valence_low', 0):.3f}, {otcp.get('valence_high', 0):.3f}]`")
                                st.write(f"**Arousal Interval:** `[{otcp.get('arousal_low', 0):.3f}, {otcp.get('arousal_high', 0):.3f}]`")

                        # 4. Cyberpunk Circumplex 2D Plot
                        fig = go.Figure()

                        # Quadrant Axis Lines
                        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="rgba(112, 0, 255, 0.3)", dash="dash"))
                        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="rgba(112, 0, 255, 0.3)", dash="dash"))

                        # Prediction Point
                        fig.add_trace(go.Scatter(
                            x=[val], y=[aro],
                            mode="markers+text",
                            text=[f"  {emotion}"],
                            textposition="top right",
                            marker=dict(size=14, color="#00F0FF", line=dict(width=2, color="#FFFFFF")),
                            textfont=dict(color="#00F0FF", size=14, family="Inter")
                        ))

                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(10, 14, 26, 0.7)',
                            font=dict(color='#A78BFA'),
                            xaxis=dict(title="Valence", range=[-1, 1], gridcolor='rgba(112, 0, 255, 0.15)', zeroline=False),
                            yaxis=dict(title="Arousal", range=[-1, 1], gridcolor='rgba(112, 0, 255, 0.15)', zeroline=False),
                            height=360,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")
