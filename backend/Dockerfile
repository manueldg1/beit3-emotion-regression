FROM python:3.10-slim

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Your BEiT-3 code, as-is from your repository ---
# We copy every .py file already present in your repo (main.py,
# modeling_finetune.py, modeling_utils.py, utils.py, etc.) instead of
# re-cloning the official microsoft/unilm repo. This guarantees the
# container runs the EXACT same code you use on the university cluster.
COPY *.py /app/

# --- Compatibility patch ---
# The official BEiT-3 utility code was written for an older PyTorch that
# had an internal "torch._six" module (removed in modern PyTorch). This
# rewrites that specific import wherever it appears, without touching
# anything else in the files.
RUN grep -rl "from torch._six import inf" /app/*.py | \
    xargs --no-run-if-empty sed -i 's/from torch\._six import inf/from torch import inf/'

# --- Bake the model checkpoint into the image at build time ---
# Downloading here (on Google's Cloud Build network) is much faster and
# more reliable than downloading it at container runtime, and means
# every request-time cold start avoids network I/O for the 1.7GB file
# entirely — it's just a local disk read from now on.
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='manueldg1/beit3-valence-arousal', filename='model_only_fp16.pth', local_dir='/app', local_dir_use_symlinks=False)"

ENV BEIT3_DIR=/app
ENV PYTHONUNBUFFERED=1

# Cloud Run injects the PORT environment variable (defaults to 8080).
# The checkpoint is now baked into the image (see above) — no network
# download needed at runtime anymore.
EXPOSE 8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
