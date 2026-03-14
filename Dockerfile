FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (blazing-fast Python package manager) ─────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Non-root user (required by HuggingFace Spaces) ──────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# ── Install CPU-only PyTorch first (saves ~1.8GB vs CUDA) ───────
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ────────────────────────
COPY --chown=user requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# ── Pre-download EasyOCR English model (~100MB, avoids runtime delay) 
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# ── Copy application code ────────────────────────────────────────
COPY --chown=user . .

# ── Create .streamlit config (gitignored locally, needed at runtime)
RUN mkdir -p .streamlit && \
    printf '[server]\nmaxUploadSize = 5\nheadless = true\nport = 7860\naddress = "0.0.0.0"\n' \
    > .streamlit/config.toml

EXPOSE 7860

CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
