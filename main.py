# Import dependencies
import sys
import os
import io
import math
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from contextlib import asynccontextmanager

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
# CONFIGURAZIONE AMBIENTE E VARIABILI GLOBALI
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = os.getenv("HF_CHECKPOINT_REPO", "manueldg1/beit3-valence-arousal")
HF_FILENAME = os.getenv("HF_CHECKPOINT_FILENAME", "model_only_fp16.pth")
HF_TOKEN = os.getenv("HF_TOKEN")

# Variabili che verranno popolate all'avvio nel lifespan
model = None
tokenizer = None
transform_image = None

# ------------------------------------------------------------------------------
# LIFESPAN: CARICA IL MODELLO DOPO CHE UVICORN È PARTITO
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, transform_image
    
    print("Inizializzazione Tokenizer e Trasformazioni...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    transform_image = transforms.Compose([
        transforms.Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    print(f"Download checkpoint da Hugging Face Hub: {HF_REPO_ID}/{HF_FILENAME} ...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN,
        )
        print(f"Checkpoint salvato in cache: {checkpoint_path}")

        print(f"Caricamento BEiT-3 Large sul dispositivo: {device}...")
        model_instance = modeling_finetune.beit3_large_patch16_480_valence_arousal(
            vocab_size=250002,
            nb_classes=2,
            drop_path_rate=0.0
        )

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model_instance.load_state_dict(state_dict)

        if device.type == "cpu":
            model_instance = model_instance.float()
        else:
            model_instance = model_instance.half()

        model_instance.to(device)
        model_instance.eval()
        
        model = model_instance
        print(">>> MODELLO BEiT-3 CARICATO CON SUCCESSO! <<<")
    except Exception as e:
        print(f"[ERRORE CRITICO] Impossibile caricare il modello: {e}")
        model = None

    yield

# ------------------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------------------
app = FastAPI(
    title="BEiT-3 Multimodal Emotion Recognition & OT-CP+ API",
    description="API per stima continua Valence/Arousal ed intervalli OT-CP+.",
    lifespan=lifespan
)

# ------------------------------------------------------------------------------
# VALORI DI CALIBRAZIONE OTCP+ E MODELLI PYDANTIC
# ------------------------------------------------------------------------------
CALIB_Q_V = 0.1842
CALIB_Q_A = 0.1521

class Interval_VA_Values(BaseModel):
    Valence: float
    Arousal: float
    Emotion: str
    Valence_Interval: Tuple[float, float]
    Arousal_Interval: Tuple[float, float]

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
# ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/")
def health_check():
    """Endpoint per l'health check di Google Cloud Run."""
    return {
        "status": "online",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=Interval_VA_Values)
async def predict(
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
) -> Interval_VA_Values:

    if not image and not text:
        raise HTTPException(
            status_code=400,
            detail="Invia almeno un testo o un'immagine per l'inferenza."
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Il modello BEiT-3 si sta ancora caricando in memoria o ha riscontrato un errore."
        )

    # 1. Processamento Immagine
    image_tensor = None
    if image is not None:
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
                detail="Impossibile elaborare l'immagine caricata."
            )

    # 2. Processamento Testo
    text_tokens = None
    padding_mask = None
    if text is not None and text.strip():
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        text_tokens = tokens["input_ids"].to(device)
        padding_mask = (tokens["attention_mask"] == 0).to(device)

    # 3. Inferenza Modello
    try:
        with torch.no_grad():
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
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante l'inferenza del modello BEiT-3: {e}"
        )

    # 4. Calcolo intervalli OTCP+ ed Emozione
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
