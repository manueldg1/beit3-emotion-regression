import os
import json
import re
from io import BytesIO

import streamlit as st
from PIL import Image

# Optional: use Gemini for multimodal image + text emotion recognition.
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="EMOTIA | Multimodal Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = ""


# ============================================================
# CUSTOM STYLE
# ============================================================
st.markdown(
    """
<style>
:root {
    --bg: #080b14;
    --card: rgba(17, 24, 39, 0.78);
    --border: rgba(255,255,255,0.10);
    --cyan: #00f2fe;
    --blue: #4facfe;
    --pink: #ff00ff;
    --muted: #94a3b8;
    --text: #e5e7eb;
}

.stApp {
    background:
        radial-gradient(circle at 15% 85%, rgba(79,172,254,.12), transparent 35%),
        radial-gradient(circle at 85% 15%, rgba(255,0,255,.10), transparent 35%),
        linear-gradient(135deg, #080b14 0%, #111827 52%, #0b1220 100%);
    color: var(--text);
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 15px 0 35px 0;
}

.hero h1 {
    font-size: clamp(2.5rem, 6vw, 5rem);
    margin: 0;
    font-weight: 900;
    letter-spacing: -3px;
    background: linear-gradient(135deg, var(--cyan), var(--pink), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 35px rgba(0,242,254,.20);
}

.hero p {
    color: var(--muted);
    letter-spacing: 5px;
    text-transform: uppercase;
    font-size: .95rem;
    margin-top: 10px;
}

.glass {
    background: linear-gradient(
        135deg,
        rgba(255,255,255,.08),
        rgba(255,255,255,.025)
    );
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 12px 40px rgba(0,0,0,.25);
    backdrop-filter: blur(18px);
}

.section-title {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 14px;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid rgba(0,242,254,.35);
    color: var(--cyan);
    background: rgba(0,242,254,.06);
    font-size: .75rem;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.result-card {
    margin-top: 20px;
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(0,242,254,.08),
        rgba(255,0,255,.06)
    );
    border: 1px solid rgba(0,242,254,.22);
    box-shadow: 0 0 35px rgba(0,242,254,.08);
}

.emotion {
    font-size: 2.7rem;
    font-weight: 900;
    margin: 5px 0 2px 0;
    background: linear-gradient(90deg, var(--cyan), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-box {
    background: rgba(0,0,0,.20);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 15px;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
}

.metric-label {
    color: var(--muted);
    font-size: .75rem;
    margin-top: 3px;
}

div[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(0,242,254,.55) !important;
    background: rgba(0,242,254,.035) !important;
    border-radius: 18px !important;
}

div[data-testid="stFileUploader"] section:hover {
    background: rgba(0,242,254,.08) !important;
    box-shadow: 0 0 25px rgba(0,242,254,.12);
}

.stTextArea textarea {
    background: rgba(0,0,0,.18) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 16px !important;
    color: white !important;
}

.stButton > button {
    border-radius: 14px;
    border: 1px solid rgba(0,242,254,.45);
    background: linear-gradient(90deg, rgba(0,242,254,.16), rgba(255,0,255,.12));
    color: white;
    font-weight: 800;
    letter-spacing: .5px;
    min-height: 3rem;
    transition: .25s ease;
}

.stButton > button:hover {
    border-color: var(--cyan);
    box-shadow: 0 0 25px rgba(0,242,254,.18);
    transform: translateY(-1px);
}

.small {
    color: var(--muted);
    font-size: .85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

def clean_json(text: str) -> dict:
    """Extract a JSON object even if the model wraps it in markdown."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain JSON.")
    return json.loads(match.group(0))


def analyze_with_gemini(image: Image.Image, text: str) -> dict:
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. Add it to requirements.txt."
        )
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets or environment variables."
        )

    genai.configure(api_key=API_KEY)

    # Keep the model name in one place so it is easy to change later.
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
You are a multimodal emotion recognition system.

Analyze BOTH modalities:
1. The uploaded image.
2. The provided text.

Determine the dominant emotional state expressed by the combined content.
Do not infer protected/sensitive personal attributes. Focus only on emotional
expression conveyed by the image and text.

Choose exactly one dominant emotion from:
{", ".join(EMOTIONS)}

Also provide:
- confidence: a number from 0 to 100
- image_emotion: emotion suggested by the image alone
- text_emotion: emotion suggested by the text alone
- explanation: concise explanation of how image and text support the result
- modality_agreement: "high", "medium", or "low"

Return ONLY valid JSON using exactly this schema:
{{
  "emotion": "joy",
  "confidence": 0,
  "image_emotion": "neutral",
  "text_emotion": "neutral",
  "modality_agreement": "medium",
  "explanation": "..."
}}

Text content:
{text}
"""

    response = model.generate_content([prompt, image])
    return clean_json(response.text)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="hero">
    <div class="badge">MULTIMODAL AI SYSTEM</div>
    <h1>EMOTIA</h1>
    <p>Multimodal Emotion Recognition</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# INPUTS
# ============================================================
col_img, col_text = st.columns(2, gap="large")

with col_img:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🖼️ IMAGE CONTENT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="small">Upload an image containing the visual emotional cues to analyze.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.success("Image loaded")

    st.markdown("</div>", unsafe_allow_html=True)

with col_text:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📝 TEXT CONTENT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="small">Enter the text that should be interpreted together with the image.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    text_content = st.text_area(
        "Text content",
        height=260,
        placeholder="Example: I can't believe this finally happened...",
        label_visibility="collapsed",
    )

    if text_content.strip():
        st.success("Text loaded")
    else:
        st.info("Waiting for text content...")

    st.markdown("</div>", unsafe_allow_html=True)


st.write("")
st.markdown(
    '<div style="text-align:center; color:#94a3b8; margin-bottom:10px;">'
    "The model combines visual and linguistic evidence before predicting the dominant emotion."
    "</div>",
    unsafe_allow_html=True,
)

analyze = st.button(
    "🧠 ANALYZE CONTENTS & PREDICT EMOTION",
    use_container_width=True,
    type="primary",
)


# ============================================================
# ANALYSIS
# ============================================================
if analyze:
    if image is None:
        st.error("Please upload an image first.")
    elif not text_content.strip():
        st.error("Please enter some text content first.")
    else:
        with st.spinner("Analyzing image + text and predicting emotion..."):
            try:
                result = analyze_with_gemini(image, text_content)

                # Normalize model output for the UI.
                result["emotion"] = str(result.get("emotion", "neutral")).lower()
                result["confidence"] = float(result.get("confidence", 0))
                result["confidence"] = max(0, min(100, result["confidence"]))

                st.session_state["emotion_result"] = result

            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ============================================================
# OUTPUT
# ============================================================
if "emotion_result" in st.session_state:
    result = st.session_state["emotion_result"]

    st.markdown("---")
    st.markdown(
        '<div class="section-title">📊 ANALYSIS OUTPUT</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    emotion = result.get("emotion", "neutral").upper()
    confidence = result.get("confidence", 0)
    image_emotion = str(result.get("image_emotion", "neutral")).upper()
    text_emotion = str(result.get("text_emotion", "neutral")).upper()
    agreement = str(result.get("modality_agreement", "medium")).upper()
    explanation = result.get("explanation", "No explanation returned.")

    st.markdown("### Dominant emotion")
    st.markdown(f'<div class="emotion">{emotion}</div>', unsafe_allow_html=True)

    st.write("")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{confidence:.0f}%</div>
                <div class="metric-label">CONFIDENCE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{image_emotion}</div>
                <div class="metric-label">IMAGE SIGNAL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{text_emotion}</div>
                <div class="metric-label">TEXT SIGNAL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{agreement}</div>
                <div class="metric-label">MODALITY AGREEMENT</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Interpretation")
    st.write(explanation)

    with st.expander("View raw model output"):
        st.json(result)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div style="text-align:center; color:#64748b; padding:30px 0 10px 0;">
    <div style="font-weight:700;">EMOTIA · Multimodal Emotion Recognition</div>
    <div style="font-size:.8rem; margin-top:6px;">
        Research / educational prototype · Emotion prediction is probabilistic.
    </div>
</div>
""",
    unsafe_allow_html=True,
)
