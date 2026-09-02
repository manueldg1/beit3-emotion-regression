FROM python:3.10-slim

# System dependencies: git is needed to clone the unilm repo for BEiT-3's
# support modules (utils.py, modeling_utils.py, torchscale glue code).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- BEiT-3 support code from the official Microsoft repo ---
# We only need the "beit3" subfolder, so a shallow sparse checkout keeps
# the image small instead of cloning the entire unilm monorepo.
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/unilm.git /app/unilm \
    && cd /app/unilm && git sparse-checkout set beit3

# --- Your custom code overrides the stock files ---
# modeling_finetune.py is YOUR version (with the valence/arousal head),
# it replaces the generic one that ships in the official repo.
COPY modeling_finetune.py /app/unilm/beit3/modeling_finetune.py
COPY main.py /app/main.py

ENV BEIT3_DIR=/app/unilm/beit3
ENV PYTHONUNBUFFERED=1

# Cloud Run injects the PORT environment variable (defaults to 8080).
# The checkpoint itself is NOT baked into the image — main.py downloads
# it from your Hugging Face Hub repo the first time the container starts.
EXPOSE 8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
