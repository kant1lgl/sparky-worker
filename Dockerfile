FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 ffmpeg \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-jpn \
        cmake pkg-config build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Force la version CPU de torch/torchvision AVANT les autres dépendances
# (argostranslate/stanza et transformers en ont besoin, et sans ça pip
# installe la variante CUDA par défaut — inutile sur ce serveur sans GPU).
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY install_translation_models.py .
RUN python install_translation_models.py

COPY download_bubble_model.py .
RUN python download_bubble_model.py

COPY worker.py .

CMD ["python", "worker.py"]