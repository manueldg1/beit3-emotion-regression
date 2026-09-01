import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go

# 1. Configuration della Pagina
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Styling Custom (Ispirato allo stile A-Team Strive To Win)
st.markdown("""
    <style>
    /* Sfondo principale scuro */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Stile per i container e le card */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background-color: #161B22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363D;
    }
    /* Customization per st.metric */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        color: #8B949E;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #58A6FF;
        font-weight: bold;
    }
    /* Styling del pulsante primario */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        border-radius: 6px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2EA043;
    }
    </style>
""", unsafe_allow_html=True)

# URL del backend FastAPI tramite Cloudflare Tunnel
FASTAPI_URL = "https://writing-makes-counting-missouri.trycloudflare.com/predict"

# 3. Header
st.title("🎭 Multimodal Emotion Recognition & OT-CP+")
st.markdown("<p style='color: #8B949E;'>BEiT-3 Fine-tuned Model for Continuous Valence/Arousal & Discrete Emotion Estimation</p>", unsafe_allow_html=True)
st.markdown("---")

# 4. Layout a 2 Colonne
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
    
    if uploaded_file:
        st.markdown("**Image Preview**")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    analyze_btn = st.button("🚀 Analyze Emotion")

with col_output:
    st.subheader("📊 Model Predictions")
    
    if analyze_btn:
        if not text_input and not uploaded_file:
            st.warning("Please provide at least a text prompt or an image to analyze.")
        else:
            with st.spinner("Processing through BEiT-3 model..."):
                try:
                    # Preparazione Payload per FastAPI
                    files = {}
                    data = {}
                    
                    if uploaded_file:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                        files = {'file': (uploaded_file.name, img_byte_arr.getvalue(), uploaded_file.type)}
                    
                    if text_input:
                        data = {'text': text_input}

                    # Chiamata al Backend
                    response = requests.post(FASTAPI_URL, files=files, data=data)

                    if response.status_code == 200:
                        res_data = response.json()
                        
                        # Emozione e metriche
                        emotion = res_data.get("predicted_emotion", "N/A")
                        val = res_data.get("valence", 0.0)
                        aro = res_data.get("arousal", 0.0)
                        
                        # 1. Category Banner
                        st.markdown(f"""
                            <div style="background-color: #1F6FEB; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
                                <h3 style="margin: 0; color: white;">Predicted Emotion: <b>{emotion}</b></h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # 2. Schede Metriche VAD
                        m1, m2 = st.columns(2)
                        m1.metric("Valence (V)", f"{val:+.3f}")
                        m2.metric("Arousal (A)", f"{aro:+.3f}")

                        # 3. Intervalli OT-CP+
                        otcp = res_data.get("otcp_intervals", {})
                        if otcp:
                            with st.expander("🔍 OT-CP+ Interval Details", expanded=False):
                                st.write(f"**Valence Interval:** `[{otcp.get('valence_low', 0):.3f}, {otcp.get('valence_high', 0):.3f}]`")
                                st.write(f"**Arousal Interval:** `[{otcp.get('arousal_low', 0):.3f}, {otcp.get('arousal_high', 0):.3f}]`")

                        # 4. Grafico Plotly Scuro per Circumplex Space
                        fig = go.Figure()

                        # Assi centrali
                        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="#30363D", dash="dash"))
                        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="#30363D", dash="dash"))

                        # Prediction Point
                        fig.add_trace(go.Scatter(
                            x=[val], y=[aro],
                            mode="markers+text",
                            text=[f"Prediction ({emotion})"],
                            textposition="top center",
                            marker=dict(size=14, color="#58A6FF"),
                            textfont=dict(color="#FFFFFF")
                        ))

                        fig.update_layout(
                            paper_bgcolor='#161B22',
                            plot_bgcolor='#161B22',
                            font=dict(color='#8B949E'),
                            xaxis=dict(title="Valence", range=[-1, 1], gridcolor='#30363D', zeroline=False),
                            yaxis=dict(title="Arousal", range=[-1, 1], gridcolor='#30363D', zeroline=False),
                            height=360,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")
