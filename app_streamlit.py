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

# 2. Precise Oculus Prime CSS Theme Styling
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at 75% 15%, #2a0845 0%, #0d0e1b 55%, #070811 100%);
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Main Title (OCULUS PRIME Matching Style) */
    .main-title {
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 3px;
        background: linear-gradient(90deg, #00f0ff 0%, #9d4edf 50%, #b026ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 25px rgba(0, 240, 255, 0.55));
        margin-bottom: 2px;
        text-transform: uppercase;
    }
    
    /* Subtitle (NEURAL RETINAL INTERFACE Matching Style) */
    .sub-title {
        text-align: center;
        color: #8E8FA2;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 5px;
        text-transform: uppercase;
        margin-bottom: 40px;
    }

    /* Container Top Bar Titles (Like PATIENT DATA) */
    .section-header-box {
        background: linear-gradient(90deg, rgba(30, 41, 75, 0.8) 0%, rgba(76, 29, 149, 0.7) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px 12px 0 0;
        padding: 16px 24px;
        color: #FFFFFF;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Container Body Box */
    .section-body-box {
        background: rgba(13, 17, 30, 0.75);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }

    /* Field Labels (NAME, AGE, TEXT CONTENT) */
    label, .stTextArea label, .stFileUploader label {
        color: #8E8FA2 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.8px !important;
        text-transform: uppercase !important;
        margin-bottom: 6px !important;
    }
    
    /* Inputs Styling */
    .stTextArea textarea {
        background-color: rgba(9, 11, 20, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #E2E8F0 !important;
        font-size: 0.95rem !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00F0FF !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.3) !important;
    }
    
    .stFileUploader section {
        background-color: rgba(9, 11, 20, 0.9) !important;
        border: 1px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
    }

    /* Metrics Box Customization */
    div[data-testid="stMetric"] {
        background: rgba(13, 10, 28, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.08);
    }
    div[data-testid="stMetricLabel"] {
        color: #8E8FA2 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
    }

    /* Main Action Button (INITIATE NEURAL ANALYSIS Gradient Bar) */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff 0%, #7000ff 50%, #b026ff 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 16px 32px !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 30px !important;
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.45) !important;
        transition: all 0.3s ease !important;
        margin-top: 10px;
    }

    .stButton>button:hover {
        box-shadow: 0 0 35px rgba(176, 38, 255, 0.75) !important;
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #8E8FA2 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        background-color: rgba(9, 11, 20, 0.6) !important;
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cloudflare FastAPI Endpoint
FASTAPI_URL = "https://writing-makes-counting-missouri.trycloudflare.com/predict"

# 4. Header Section
st.markdown("<h1 class='main-title'>BEiT-3 EMOTION AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>MULTIMODAL EMOTION RECOGNITION & OT-CP+ INTERFACE</p>", unsafe_allow_html=True)

# 5. Main Layout
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("<div class='section-header-box'>INPUT DATA</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-body-box'>", unsafe_allow_html=True)
    
    text_input = st.text_area(
        "TEXT CONTENT", 
        placeholder="Type a sentence or prompt here...",
        height=110
    )
    
    uploaded_file = st.file_uploader(
        "IMAGE CONTENT", 
        type=["jpg", "jpeg", "png"]
    )
    
    image = None
    if uploaded_file:
        st.markdown("<p style='color: #8E8FA2; font-weight: 700; font-size: 0.75rem; letter-spacing: 1.8px; text-transform: uppercase; margin-top: 15px;'>IMAGE PREVIEW</p>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    analyze_btn = st.button("INITIATE NEURAL ANALYSIS")
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown("<div class='section-header-box'>MODEL PREDICTIONS</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-body-box'>", unsafe_allow_html=True)
    
    if analyze_btn:
        if not text_input and not uploaded_file:
            st.warning("Please provide at least a text prompt or an image to analyze.")
        else:
            with st.spinner("PROCESSING NEURAL EMBEDDINGS..."):
                try:
                    files = None
                    data = {}
                    
                    if uploaded_file:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                        files = {'file': (uploaded_file.name, img_byte_arr.getvalue(), uploaded_file.type)}
                    
                    if text_input:
                        data['text'] = text_input

                    # Request to FastAPI Backend
                    response = requests.post(FASTAPI_URL, files=files, data=data)

                    if response.status_code == 200:
                        res_data = response.json()
                        
                        emotion = res_data.get("predicted_emotion", "N/A")
                        val = res_data.get("valence", 0.0)
                        aro = res_data.get("arousal", 0.0)
                        
                        # 1. Predicted Emotion Banner
                        st.markdown(f"""
                            <div style="background: linear-gradient(90deg, rgba(0, 240, 255, 0.12), rgba(176, 38, 255, 0.22)); 
                                        border: 1px solid #00F0FF; padding: 16px; border-radius: 8px; text-align: center; margin-bottom: 20px;
                                        box-shadow: 0 0 20px rgba(0, 240, 255, 0.2);">
                                <h3 style="margin: 0; color: #8E8FA2; font-size: 0.8rem; letter-spacing: 2.5px; text-transform: uppercase;">PREDICTED EMOTION</h3>
                                <div style="font-size: 2.2rem; font-weight: 900; color: #00F0FF; letter-spacing: 3px; text-shadow: 0 0 15px rgba(0, 240, 255, 0.8);">{emotion}</div>
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
                                st.markdown(f"<span style='color:#8E8FA2; font-size: 0.9rem;'><b>Valence Interval:</b> <code>[{otcp.get('valence_low', 0):.3f}, {otcp.get('valence_high', 0):.3f}]</code></span>", unsafe_allow_html=True)
                                st.markdown(f"<span style='color:#8E8FA2; font-size: 0.9rem;'><b>Arousal Interval:</b> <code>[{otcp.get('arousal_low', 0):.3f}, {otcp.get('arousal_high', 0):.3f}]</code></span>", unsafe_allow_html=True)

                        # 4. Cyberpunk Circumplex 2D Plot
                        fig = go.Figure()

                        # Quadrant Axis Lines
                        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="rgba(139, 92, 246, 0.35)", dash="dash"))
                        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="rgba(139, 92, 246, 0.35)", dash="dash"))

                        # Prediction Point
                        fig.add_trace(go.Scatter(
                            x=[val], y=[aro],
                            mode="markers+text",
                            text=[f"  {emotion}"],
                            textposition="top right",
                            marker=dict(size=14, color="#00F0FF", line=dict(width=2, color="#FFFFFF")),
                            textfont=dict(color="#00F0FF", size=13, family="Inter")
                        ))

                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(9, 11, 20, 0.85)',
                            font=dict(color='#8E8FA2', size=11),
                            xaxis=dict(title=dict(text="VALENCE", font=dict(color="#8E8FA2", size=12)), range=[-1, 1], gridcolor='rgba(139, 92, 246, 0.15)', zeroline=False),
                            yaxis=dict(title=dict(text="AROUSAL", font=dict(color="#8E8FA2", size=12)), range=[-1, 1], gridcolor='rgba(139, 92, 246, 0.15)', zeroline=False),
                            height=340,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")
    else:
        st.markdown("<p style='color: #8E8FA2; font-size: 0.9rem; text-align: center; margin-top: 40px;'>Provide input on the left and click <b>INITIATE NEURAL ANALYSIS</b> to view predictions.</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
