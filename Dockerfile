FROM python:3.13-slim

# libmagic1 : détection du vrai type MIME d'un fichier à partir de son contenu.
# ffmpeg : extraction d'une frame depuis une vidéo pour générer sa miniature.
# tesseract-ocr(+langs) : OCR pour le pipeline de traduction (FR/EN/JP).
# cmake/pkg-config : nécessaires pour compiler sentencepiece (dépendance d'argostranslate).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 ffmpeg \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-jpn \
        cmake pkg-config build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Force la version CPU de torch AVANT argostranslate : argostranslate dépend de
# stanza qui dépend de torch, et sans ça pip installe la variante CUDA par
# défaut (~3 Go de libs GPU totalement inutiles sur ce serveur sans GPU).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge et installe les modèles de traduction Argos au moment du build
# (pas au runtime) : démarrage instantané, pas de dépendance réseau au cold start.
COPY install_translation_models.py .
RUN python install_translation_models.py

COPY worker.py .

CMD ["python", "worker.py"]