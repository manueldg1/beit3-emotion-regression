import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="BEiT-3 Emotion AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Cyberpunk / Oculus Prime Theme Style (Larger Fonts, No Emojis)
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at 80% 20%, #1e0936 0%, #0a0c16 60%, #05060b 100%);
        color: #C4B5FD;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Main Title */
    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00f0ff 0%, #b026ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 240, 255, 0.5);
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    
    .sub-title {
        text-align: center;
        color: #A78BFA;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 30px;
        text-shadow: 0 0 12px rgba(167, 139, 250, 0.4);
    }

    /* Subheaders (Section Titles) */
    h2, h3, .stSubheader {
        color: #00F0FF !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        letter-spacing: 2.5px !important;
        text-transform: uppercase !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }
    
    /* Labels (Input field labels) */
    label, .stTextArea label, .stFileUploader label {
        color: #A78BFA !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    
    /* Text Input Area & Uploader */
    .stTextArea textarea {
        background-color: rgba(10, 14, 26, 0.85) !important;
        border: 1px solid rgba(112, 0, 255, 0.4) !important;
        color: #00F0FF !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
    }
    
    .stTextArea textarea::placeholder {
        color: #6B7280 !important;
    }
    
    .stFileUploader section {
        background-color: rgba(10, 14, 26, 0.85) !important;
        border: 1px dashed rgba(0, 240, 255, 0.4) !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader span, .stFileUploader small {
        color: #C4B5FD !important;
        font-size: 0.9rem !important;
    }

    /* Container / Card Boxes */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(112, 0, 255, 0.35) !important;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 24px;
    }

    /* Metrics Box Customization */
    div[data-testid="stMetric"] {
        background: rgba(15, 10, 30, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.12);
    }
    div[data-testid="stMetricLabel"] {
        color: #A78BFA !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px rgba(0, 240, 255, 0.7);
    }

    /* Glow Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff 0%, #b026ff 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 16px 32px !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 25px !important;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        box-shadow: 0 0 32px rgba(176, 38, 255, 0.85) !important;
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #A78BFA !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        background-color: rgba(10, 14, 26, 0.6) !important;
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cloudflare FastAPI Endpoint
FASTAPI_URL = "https://writing-makes-counting-missouri.trycloudflare.com/predict"

# 4. Header Section
st.markdown("<h1 class='main-title'>BEiT-3 EMOTION AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>MULTIMODAL EMOTION RECOGNITION & OT-CP+ INTERFACE</p>", unsafe_allow_html=True)
st.markdown("---")

# 5. Main Layout
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("INPUT DATA")
    
    text_input = st.text_area(
        "TEXT CONTENT", 
        placeholder="Type a sentence or prompt here...",
        height=120
    )
    
    uploaded_file = st.file_uploader(
        "IMAGE CONTENT", 
        type=["jpg", "jpeg", "png"]
    )
    
    image = None
    if uploaded_file:
        st.markdown("<p style='color: #A78BFA; font-weight: 700; font-size: 0.95rem; letter-spacing: 2px; text-transform: uppercase;'>IMAGE PREVIEW</p>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    analyze_btn = st.button("ANALYZE CONTENT & PREDICT EMOTION")

with col_output:
    st.subheader("MODEL PREDICTIONS")
    
    if analyze_btn:
        if not text_input and not uploaded_file:
            st.warning("Please provide at least a text prompt or an image to analyze.")
        else:
            with st.spinner("PROCESSING NEURAL EMBEDDINGS..."):
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
                            <div style="background: linear-gradient(90deg, rgba(0, 240, 255, 0.15), rgba(176, 38, 255, 0.25)); 
                                        border: 1px solid #00F0FF; padding: 16px; border-radius: 8px; text-align: center; margin-bottom: 20px;
                                        box-shadow: 0 0 18px rgba(0, 240, 255, 0.25);">
                                <h3 style="margin: 0; color: #C4B5FD; font-size: 1.05rem; letter-spacing: 2.5px; text-transform: uppercase;">PREDICTED EMOTION</h3>
                                <div style="font-size: 2.2rem; font-weight: 900; color: #00F0FF; letter-spacing: 3px; text-shadow: 0 0 12px rgba(0, 240, 255, 0.8);">{emotion}</div>
                            </div>
                        """, unsafe_allow_html=True)

                        # 2. VAD Metrics Cards
                        m1, m2 = st.columns(2)
                        m1.metric("VALENCE (V)", f"{val:+.3f}")
                        m2.metric("AROUSAL (A)", f"{aro:+.3f}")

                        # 3. OT-CP+ Interval Details
                        otcp = res_data.get("otcp_intervals", {})
                        if otcp:
                            with st.expander("VIEW OT-CP+ INTERVALS", expanded=False):
                                st.markdown(f"<span style='color:#C4B5FD; font-size: 1rem;'><b>Valence Interval:</b> <code>[{otcp.get('valence_low', 0):.3f}, {otcp.get('valence_high', 0):.3f}]</code></span>", unsafe_allow_html=True)
                                st.markdown(f"<span style='color:#C4B5FD; font-size: 1rem;'><b>Arousal Interval:</b> <code>[{otcp.get('arousal_low', 0):.3f}, {otcp.get('arousal_high', 0):.3f}]</code></span>", unsafe_allow_html=True)

                        # 4. Cyberpunk Circumplex 2D Plot
                        fig = go.Figure()

                        # Quadrant Axis Lines
                        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="rgba(176, 38, 255, 0.35)", dash="dash"))
                        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="rgba(176, 38, 255, 0.35)", dash="dash"))

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
                            plot_bgcolor='rgba(10, 14, 26, 0.85)',
                            font=dict(color='#A78BFA', size=12),
                            xaxis=dict(title=dict(text="VALENCE", font=dict(color="#A78BFA", size=13)), range=[-1, 1], gridcolor='rgba(176, 38, 255, 0.15)', zeroline=False),
                            yaxis=dict(title=dict(text="AROUSAL", font=dict(color="#A78BFA", size=13)), range=[-1, 1], gridcolor='rgba(176, 38, 255, 0.15)', zeroline=False),
                            height=350,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")
