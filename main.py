# Import dependencies
import sys
import os
import io
import math
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Setup BEiT-3 directory
BEIT3_DIR = os.getenv("BEIT3_DIR", "/app/unilm/beit3")
if BEIT3_DIR not in sys.path:
    sys.path.append(BEIT3_DIR)

# Import BEiT-3 modules
import modeling_finetune
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import XLMRobertaTokenizer

# ------------------------------------------------------------------------------
# 1. CARICAMENTO MODELLO BEiT-3 LARGE
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = os.getenv("HF_CHECKPOINT_REPO", "manueldg1/beit3-valence-arousal")
HF_FILENAME = os.getenv("HF_CHECKPOINT_FILENAME", "model_only_fp16.pth")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"Downloading checkpoint from Hugging Face Hub: {HF_REPO_ID}/{HF_FILENAME} ...")
CHECKPOINT_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_FILENAME,
    token=HF_TOKEN,
)
print(f"Checkpoint cached locally at: {CHECKPOINT_PATH}")

# Load Tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Image preprocessing (480x480 pixels, bicubic interpolation)
transform_image = transforms.Compose([
    transforms.Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

print(f"Loading BEiT-3 Large on device: {device}...")
model = modeling_finetune.beit3_large_patch16_480_valence_arousal(
    vocab_size=250002,
    nb_classes=2,
    drop_path_rate=0.0
)

try:
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)

    if device.type == "cpu":
        model = model.float()
    else:
        model = model.half()

    model.to(device)
    model.eval()
    print(f"BEiT-3 Checkpoint ({CHECKPOINT_PATH}) loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load checkpoint: {e}")
    model = None

# ------------------------------------------------------------------------------
# 2. VALORI DI CALIBRAZIONE OTCP+ E MODELLI PYDANTIC
# ------------------------------------------------------------------------------
CALIB_Q_V = 0.1842
CALIB_Q_A = 0.1521


class Interval_VA_Values(BaseModel):
    Valence: float
    Arousal: float
    Emotion: str
    Valence_Interval: Tuple[float, float]
    Arousal_Interval: Tuple[float, float]

# ------------------------------------------------------------------------------
# 3. FUNZIONI AUSILIARIE (MAPPA EMOZIONI & CALCOLO INTERVALLI OTCP+)
# ------------------------------------------------------------------------------
VAD_MAPPING = {
    'Amusement':   {'Valence': 0.858,  'Arousal': 0.674},
    'Anger':       {'Valence': -0.666, 'Arousal': 0.730},
    'Awe':         {'Valence': -0.062, 'Arousal': 0.480},
    'Contentment': {'Valence': 0.750,  'Arousal': 0.220},
    'Disgust':     {'Valence': -0.896, 'Arousal': 0.550},
    'Excitement':  {'Valence': 0.792,  'Arousal': 0.368},
    'Fear':        {'Valence': -0.854, 'Arousal': 0.680},
    'Sadness':     {'Valence': -0.896, 'Arousal': -0.424},
}

def map_va_to_nrc_v2_emotion(valence: float, arousal: float, threshold: float = 0.45) -> str:
    if abs(valence) <= 0.15 and abs(arousal) <= 0.15:
        return "Neutral"

    closest_emotion = "something else"
    min_distance = float('inf')
    for emotion, coords in VAD_MAPPING.items():
        dist = math.sqrt((valence - coords['Valence']) ** 2 + (arousal - coords['Arousal']) ** 2)
        if dist < min_distance:
            min_distance = dist
            closest_emotion = emotion

    if min_distance > threshold:
        return "something else"

    return closest_emotion


def compute_otcp_intervals(v_pred: float, a_pred: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    v_min = max(-1.0, v_pred - CALIB_Q_V)
    v_max = min(1.0, v_pred + CALIB_Q_V)
    a_min = max(-1.0, a_pred - CALIB_Q_A)
    a_max = min(1.0, a_pred + CALIB_Q_A)
    return (round(v_min, 4), round(v_max, 4)), (round(a_min, 4), round(a_max, 4))

# ------------------------------------------------------------------------------
# 4. FASTAPI APP & ENDPOINT POST
# ------------------------------------------------------------------------------
app = FastAPI(
    title="BEiT-3 Multimodal Emotion Recognition & OT-CP+ API",
    description="API for continuous Valence/Arousal estimation and OT-CP+ intervals."
)


@app.post("/predict", response_model=Interval_VA_Values)
async def predict(
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
) -> Interval_VA_Values:

    has_image = image is not None and image.filename != ""
    has_text = text is not None and text.strip() != ""

    if not has_image and not has_text:
        raise HTTPException(
            status_code=400,
            detail="Send at least text or an image for inference."
        )

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="BEiT-3 model was not loaded properly at startup."
        )

    # 1. Processing Immagine
    image_tensor = None
    if has_image:
        try:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_tensor = transform_image(pil_image).unsqueeze(0).to(device)
            if device.type == "cpu":
                image_tensor = image_tensor.float()
            else:
                image_tensor = image_tensor.half()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Unable to process the provided image file."
            )

    # 2. Processing Testo
    text_tokens = None
    padding_mask = None
    if has_text:
        tokens = tokenizer(
            text.strip(),
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        text_tokens = tokens["input_ids"].to(device)
        padding_mask = (tokens["attention_mask"] == 0).to(device)

    # 3. Inferenza condizionale senza passare mai 'None' per le modalità assenti
    try:
        with torch.no_grad():
            if has_text and not has_image:
                # SOLO TESTO: passa esclusivamente text_segment e padding_mask
                outputs = model(
                    text_segment=text_tokens,
                    padding_mask=padding_mask
                )
            elif has_image and not has_text:
                # SOLO IMMAGINE: passa esclusivamente image
                outputs = model(
                    image=image_tensor
                )
            else:
                # MULTIMODALE: passa sia image che testo
                outputs = model(
                    image=image_tensor,
                    text_segment=text_tokens,
                    padding_mask=padding_mask
                )

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            v_pred = float(outputs[0, 0].item())
            a_pred = float(outputs[0, 1].item())

    except Exception as e:
        import traceback
        print(f"[ERROR INFERENCE]: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error in BEiT-3 model: {e}"
        )

    # 4. Calcolo intervalli OTCP+ e Mappatura Emozione
    v_interval, a_interval = compute_otcp_intervals(v_pred, a_pred)
    emotion_label = map_va_to_nrc_v2_emotion(v_pred, a_pred)

    # 5. Output JSON
    return Interval_VA_Values(
        Valence=round(v_pred, 4),
        Arousal=round(a_pred, 4),
        Emotion=emotion_label,
        Valence_Interval=v_interval,
        Arousal_Interval=a_interval
    )
