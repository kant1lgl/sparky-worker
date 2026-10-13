"""
Spark — Worker de traitement asynchrone (Railway)

Mode "réveillé à la demande" : ce service n'est PAS une boucle qui tourne en
continu. C'est un petit serveur HTTP qui ne fait rien tant que personne ne
l'appelle. Un Database Webhook Supabase appelle POST /process à chaque
insertion ou mise à jour dans la table `jobs` ; ça réveille le conteneur
Railway (mode Serverless/Sleep), le worker vide toute la file de jobs
"pending" d'un coup, puis redevient silencieux — Railway le remet en veille
automatiquement après ~5-10 min sans trafic sortant.

Chaque type de job (malware_scan, duplicate_check, thumbnail, translate...)
a son propre handler dans HANDLERS.
"""

import json
import os
import re
import subprocess
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from urllib.parse import urlparse

import imagehash
import magic
import requests
from flask import Flask, jsonify, request
from PIL import Image
from supabase import create_client, Client
import pytesseract
import argostranslate.translate
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
WORKER_SECRET = os.environ.get("WORKER_SECRET")  # partagé avec le Database Webhook Supabase

MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))

# Seuils de détection de doublon/plagiat.
PHASH_HAMMING_THRESHOLD = 8      # distance de Hamming <= 8 sur un phash 64 bits -> quasi-identique
TEXT_JACCARD_THRESHOLD = 0.85    # similarité de Jaccard sur shingles >= 0.85 -> texte quasi-identique
TEXT_SHINGLE_SIZE = 8            # taille des n-grammes de caractères pour le texte

# Types MIME réels autorisés sur Spark (lus dans le contenu binaire du fichier,
# pas dans son extension déclarée ni son Content-Type HTTP).
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "application/pdf",
}
MAX_FILE_SIZE_BYTES = 600 * 1024 * 1024  # garde-fou large ; les limites précises par type sont déjà imposées à l'upload

# Miniatures.
THUMBNAIL_MAX_DIMENSION = 600
THUMBNAIL_WEBP_QUALITY = 82

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
BUBBLE_MODEL_DIR = os.environ.get("BUBBLE_MODEL_DIR", "/app/models/bubble-detector")
BUBBLE_CONFIDENCE_THRESHOLD = 0.4

bubble_processor = AutoImageProcessor.from_pretrained(BUBBLE_MODEL_DIR)
bubble_model = AutoModelForObjectDetection.from_pretrained(BUBBLE_MODEL_DIR)
bubble_model.eval()


def detect_bubbles(local_path: str) -> list[list[int]]:
    """Détecte les bulles de texte sur une page (RT-DETR-v2, licence Apache-2.0).
    Renvoie une liste de bbox [x, y, w, h] en pixels, triées haut-en-bas puis
    gauche-à-droite (approximation simple de l'ordre de lecture — pas une vraie
    inférence droite-à-gauche pour le manga, à affiner plus tard si besoin)."""
    with Image.open(local_path) as img:
        img = img.convert("RGB")
        inputs = bubble_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = bubble_model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = bubble_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=BUBBLE_CONFIDENCE_THRESHOLD
        )[0]

    boxes = []
    for box in results["boxes"].tolist():
        x1, y1, x2, y2 = box
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def ocr_bubble(local_path: str, bbox: list[int], source_lang: str) -> str:
    """OCR sur une bulle isolée (recadrée depuis la page) : le texte d'une
    bulle appartient à la même réplique, donc pas besoin de regroupement par
    ligne comme pour l'OCR pleine page."""
    tess_lang = TESSERACT_LANG_MAP.get(source_lang)
    if not tess_lang:
        raise ValueError(f"langue OCR non supportée: {source_lang}")

    x, y, w, h = bbox
    with Image.open(local_path) as img:
        crop = img.crop((x, y, x + w, y + h))
        text = pytesseract.image_to_string(crop, lang=tess_lang)
    return " ".join(text.split())

# Pattern des URLs publiques Supabase Storage :
# https://<project>.supabase.co/storage/v1/object/public/<bucket>/<path...>
STORAGE_URL_RE = re.compile(r"/storage/v1/object/public/([^/]+)/(.+)$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_storage_url(url: str):
    """Extrait (bucket, path) d'une URL publique Supabase Storage, ou None si non reconnue."""
    match = STORAGE_URL_RE.search(urlparse(url).path)
    if not match:
        return None
    bucket, path = match.group(1), match.group(2)
    return bucket, path


def download_to_temp(url: str) -> str:
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    suffix = os.path.splitext(urlparse(url).path)[1] or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"fichier trop volumineux (> {MAX_FILE_SIZE_BYTES} octets): {url}")
            f.write(chunk)
    return path


def validate_file_is_safe(local_path: str) -> tuple[bool, str]:
    """
    Vérifie que le contenu RÉEL du fichier (lu par libmagic à partir de ses
    premiers octets, pas de son extension) correspond à un type autorisé sur
    Spark. Rejette tout fichier dont la signature binaire ne correspond pas à
    un format attendu (ex: un exécutable ou un script renommé en .jpg).

    Ne remplace pas une analyse antivirus signature par signature (abandonnée
    pour l'instant, trop gourmande en RAM pour l'infra actuelle) : ça bloque
    le cas le plus fréquent (fichier déguisé), pas un payload malveillant
    caché dans un fichier par ailleurs valide dans son format.
    """
    real_mime = magic.from_file(local_path, mime=True)
    if real_mime not in ALLOWED_MIME_TYPES:
        return False, real_mime

    if real_mime.startswith("image/"):
        try:
            with Image.open(local_path) as img:
                img.verify()
        except Exception:
            return False, real_mime

    return True, real_mime


def resolve_owner_id(table: str, row_id: str):
    """Retourne l'owner_id d'une ligne publications/chapters (via la publication
    parente pour un chapitre). Renvoie None si la résolution échoue."""
    try:
        row = (
            supabase.table(table)
            .select("owner_id" if table == "publications" else "publication_id")
            .eq("id", row_id)
            .single()
            .execute()
        )
        if table == "publications":
            return row.data.get("owner_id")
        pub_id = row.data.get("publication_id")
        if not pub_id:
            return None
        pub = supabase.table("publications").select("owner_id").eq("id", pub_id).single().execute()
        return pub.data.get("owner_id")
    except Exception as exc:
        print(f"[worker] résolution owner_id échouée pour {table}/{row_id}: {exc}")
        return None


def notify(owner_id, notif_type: str, title: str, message: str, kind_for_log: str) -> None:
    if not owner_id:
        return
    try:
        supabase.table("notifications").insert({
            "user_id": owner_id,
            "type": notif_type,
            "title": title,
            "message": message,
            "link": "/publications",
            "read": False,
        }).execute()
    except Exception as exc:
        print(f"[{kind_for_log}] notification échouée: {exc}")


# ── Handlers par type de job ────────────────────────────────────────────

def handle_ping(payload: dict) -> dict:
    """Job de test pour valider que la queue fonctionne de bout en bout."""
    return {"pong": True, "echo": payload}


def handle_malware_scan(payload: dict) -> dict:
    """
    payload attendu :
      { "table": "publications" | "chapters", "row_id": "<uuid>", "files": ["<url>", ...] }

    Vérifie que chaque fichier est bien ce qu'il prétend être (type MIME réel
    lu dans son contenu binaire, décodage image valide) parmi les formats
    autorisés sur Spark. Si tout est conforme -> scan_status = 'clean'.
    Si un fichier est rejeté -> scan_status = 'infected', les fichiers rejetés
    sont supprimés du Storage, et l'auteur est notifié.
    """
    table = payload["table"]
    row_id = payload["row_id"]
    files = [f for f in (payload.get("files") or []) if f]

    if not files:
        supabase.table(table).update({"scan_status": "clean"}).eq("id", row_id).execute()
        return {"scanned": 0, "infected": 0}

    infected_urls = []
    for url in files:
        local_path = None
        try:
            local_path = download_to_temp(url)
            ok, real_mime = validate_file_is_safe(local_path)
            if not ok:
                print(f"[malware_scan] fichier rejeté ({real_mime}) : {url}")
                infected_urls.append(url)
        except Exception as exc:
            print(f"[malware_scan] vérification échouée pour {url}: {exc}")
            infected_urls.append(url)
        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)

    if infected_urls:
        for url in infected_urls:
            parsed = parse_storage_url(url)
            if parsed:
                bucket, path = parsed
                try:
                    supabase.storage.from_(bucket).remove([path])
                except Exception as exc:
                    print(f"[malware_scan] suppression storage échouée pour {url}: {exc}")

        supabase.table(table).update(
            {"scan_status": "infected", "status": "rejected"} if table == "publications"
            else {"scan_status": "infected"}
        ).eq("id", row_id).execute()

        owner_id = resolve_owner_id(table, row_id)
        notify(
            owner_id,
            "content_rejected",
            "Contenu refusé",
            "Un fichier que tu as uploadé ne correspond pas à un format autorisé et a été supprimé.",
            "malware_scan",
        )

        return {"scanned": len(files), "infected": len(infected_urls)}

    supabase.table(table).update({"scan_status": "clean"}).eq("id", row_id).execute()
    return {"scanned": len(files), "infected": 0}


def compute_phash(local_path: str) -> str:
    """Empreinte perceptuelle d'une image (64 bits, en hex). Deux images visuellement
    proches (recadrage, léger filtre, recompression) ont des phash proches."""
    with Image.open(local_path) as img:
        return str(imagehash.phash(img))


def hamming_distance_hex(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def text_shingles(text: str, k: int = TEXT_SHINGLE_SIZE) -> set:
    """Découpe un texte normalisé en n-grammes de caractères, pour comparaison
    de similarité indépendante de la casse/espacement."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return set()
    if len(normalized) < k:
        return {normalized}
    return {normalized[i:i + k] for i in range(len(normalized) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def handle_duplicate_check(payload: dict) -> dict:
    """
    payload attendu :
      {
        "table": "publications" | "chapters",
        "row_id": "<uuid>",
        "files": ["<url image>", ...],   # optionnel : pages/couverture à comparer
        "text": "<texte à comparer>"      # optionnel : texte de l'œuvre à comparer
      }

    Compare les images (perceptual hash, distance de Hamming) et le texte
    (similarité de Jaccard sur shingles de caractères) du contenu contre ce qui
    a déjà été publié par d'AUTRES créateurs uniquement — un créateur qui
    republie ou traduit sa propre œuvre (même dessin, bulles de texte
    traduites) ne doit jamais être flagué contre lui-même. Ne supprime rien
    automatiquement — une ressemblance forte avec le contenu d'un tiers flague
    juste le contenu pour revue admin (duplicate_status = 'flagged'), les
    empreintes du nouveau contenu sont stockées dans tous les cas pour servir
    de référence aux prochaines comparaisons.
    """
    table = payload["table"]
    row_id = payload["row_id"]
    files = [f for f in (payload.get("files") or []) if f]
    text = (payload.get("text") or "").strip()

    owner_id = resolve_owner_id(table, row_id)

    flags = []

    # --- Images ---
    for url in files:
        local_path = None
        try:
            local_path = download_to_temp(url)
            phash = compute_phash(local_path)
        except Exception as exc:
            print(f"[duplicate_check] phash échoué pour {url}: {exc}")
            continue
        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)

        query = (
            supabase.table("content_hashes")
            .select("row_id, source_table, file_url, hash_value, owner_id")
            .eq("hash_type", "phash")
            .neq("row_id", row_id)
        )
        if owner_id:
            query = query.neq("owner_id", owner_id)
        existing = query.execute()
        for row in existing.data or []:
            dist = hamming_distance_hex(phash, row["hash_value"])
            if dist <= PHASH_HAMMING_THRESHOLD:
                flags.append({
                    "type": "image",
                    "matched_table": row["source_table"],
                    "matched_row_id": row["row_id"],
                    "matched_url": row["file_url"],
                    "distance": dist,
                })

        supabase.table("content_hashes").insert({
            "source_table": table,
            "row_id": row_id,
            "owner_id": owner_id,
            "hash_type": "phash",
            "file_url": url,
            "hash_value": phash,
        }).execute()

    # --- Texte ---
    if text:
        shingles = text_shingles(text)

        query = (
            supabase.table("content_hashes")
            .select("row_id, source_table, hash_value, owner_id")
            .eq("hash_type", "text_shingles")
            .neq("row_id", row_id)
        )
        if owner_id:
            query = query.neq("owner_id", owner_id)
        existing = query.execute()
        for row in existing.data or []:
            try:
                other = set(json.loads(row["hash_value"]))
            except Exception:
                continue
            sim = jaccard(shingles, other)
            if sim >= TEXT_JACCARD_THRESHOLD:
                flags.append({
                    "type": "text",
                    "matched_table": row["source_table"],
                    "matched_row_id": row["row_id"],
                    "similarity": round(sim, 3),
                })

        supabase.table("content_hashes").insert({
            "source_table": table,
            "row_id": row_id,
            "owner_id": owner_id,
            "hash_type": "text_shingles",
            "hash_value": json.dumps(list(shingles)),
        }).execute()

    if flags:
        supabase.table(table).update({"duplicate_status": "flagged"}).eq("id", row_id).execute()
        notify(
            owner_id,
            "content_flagged_duplicate",
            "Contenu en cours de vérification",
            "Une ressemblance a été détectée avec le contenu d'un autre créateur. Ton contenu est en revue par la modération.",
            "duplicate_check",
        )
    else:
        supabase.table(table).update({"duplicate_status": "clean"}).eq("id", row_id).execute()

    return {"images_checked": len(files), "text_checked": bool(text), "flags": len(flags)}


def generate_image_thumbnail(local_path: str, max_dim: int) -> str:
    """Redimensionne une image (en conservant les proportions) et l'encode en
    WebP. Renvoie le chemin du fichier temporaire généré, à nettoyer par l'appelant."""
    out_fd, out_path = tempfile.mkstemp(suffix=".webp")
    os.close(out_fd)
    with Image.open(local_path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        img.save(out_path, "WEBP", quality=THUMBNAIL_WEBP_QUALITY)
    return out_path


def extract_video_frame(local_path: str) -> str:
    """Extrait une frame représentative d'une vidéo via ffmpeg (à ~10% de la
    durée pour éviter un éventuel écran noir de début, repli sur 1s si la
    durée n'a pas pu être déterminée)."""
    frame_fd, frame_path = tempfile.mkstemp(suffix=".jpg")
    os.close(frame_fd)

    seek = 1.0
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", local_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip())
        if duration > 0:
            seek = max(0.5, duration * 0.1)
    except Exception:
        pass

    proc = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(seek), "-i", local_path, "-frames:v", "1", "-q:v", "3", frame_path],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0 or not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
        raise RuntimeError(f"ffmpeg erreur (code {proc.returncode}): {proc.stderr}")
    return frame_path


def handle_thumbnail(payload: dict) -> dict:
    """
    payload attendu :
      {
        "table": "publications" | "chapters" | "profiles",
        "row_id": "<uuid>",
        "source_url": "<url image ou vidéo>",
        "field": "<colonne à mettre à jour avec l'URL de la miniature>",
        "max_dimension": 600   # optionnel
      }

    Génère une miniature (image redimensionnée en WebP, ou frame extraite via
    ffmpeg pour une vidéo) et l'upload dans le MÊME bucket Storage que la
    source, sous 'thumbnails/<table>/<row_id>.webp'. Met à jour la colonne
    `field` de la ligne avec l'URL publique de la miniature.
    """
    table = payload["table"]
    row_id = payload["row_id"]
    source_url = payload["source_url"]
    field = payload["field"]
    max_dim = int(payload.get("max_dimension") or THUMBNAIL_MAX_DIMENSION)

    parsed_source = parse_storage_url(source_url)
    if not parsed_source:
        raise ValueError(f"URL Storage non reconnue, impossible de déterminer le bucket cible: {source_url}")
    bucket, _source_path = parsed_source

    local_path = None
    frame_path = None
    thumb_path = None
    try:
        local_path = download_to_temp(source_url)
        real_mime = magic.from_file(local_path, mime=True)

        if real_mime.startswith("video/"):
            frame_path = extract_video_frame(local_path)
            thumb_path = generate_image_thumbnail(frame_path, max_dim)
        elif real_mime.startswith("image/"):
            thumb_path = generate_image_thumbnail(local_path, max_dim)
        else:
            raise ValueError(f"type non pris en charge pour une miniature: {real_mime}")

        dest_path = f"thumbnails/{table}/{row_id}.webp"
        with open(thumb_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                dest_path, f, {"content-type": "image/webp", "upsert": "true"}
            )
        public_url = supabase.storage.from_(bucket).get_public_url(dest_path)
    finally:
        for p in (local_path, frame_path, thumb_path):
            if p and os.path.exists(p):
                os.remove(p)

    supabase.table(table).update({field: public_url}).eq("id", row_id).execute()
    return {"thumbnail_url": public_url}

TESSERACT_LANG_MAP = {"fr": "fra", "en": "eng", "ja": "jpn"}
SUPPORTED_TRANSLATE_LANGS = {"fr", "en", "ja"}
OCR_MIN_CONFIDENCE = 40


def ocr_page(local_path: str, source_lang: str) -> list[dict]:
    """OCR direct sur l'image entière (pas de détection de bulles au MVP) :
    regroupe les mots par ligne (Tesseract fournit block/par/line_num) pour
    traduire des phrases complètes plutôt que des mots isolés sans contexte
    de phrase, et renvoie les lignes de texte avec leur position (bbox
    englobante de la ligne)."""
    tess_lang = TESSERACT_LANG_MAP.get(source_lang)
    if not tess_lang:
        raise ValueError(f"langue OCR non supportée: {source_lang}")

    with Image.open(local_path) as img:
        data = pytesseract.image_to_data(img, lang=tess_lang, output_type=pytesseract.Output.DICT)

    lines: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if conf < OCR_MIN_CONFIDENCE:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left, top = data["left"][i], data["top"][i]
        right, bottom = left + data["width"][i], top + data["height"][i]

        if key not in lines:
            lines[key] = {"words": [], "left": left, "top": top, "right": right, "bottom": bottom}
        line = lines[key]
        line["words"].append(word)
        line["left"] = min(line["left"], left)
        line["top"] = min(line["top"], top)
        line["right"] = max(line["right"], right)
        line["bottom"] = max(line["bottom"], bottom)

    blocks = []
    for key in sorted(lines.keys()):
        line = lines[key]
        text = " ".join(line["words"])
        blocks.append({
            "text": text,
            "bbox": [line["left"], line["top"], line["right"] - line["left"], line["bottom"] - line["top"]],
        })
    return blocks

def handle_translate(payload: dict) -> dict:
    """
    payload attendu :
      {
        "table": "chapters",
        "row_id": "<uuid du chapitre>",
        "pages": ["<url image page 1>", "<url image page 2>", ...],
        "source_lang": "fr" | "en" | "ja"
      }

    MVP : pas de détection de bulles, OCR direct (Tesseract) sur l'image
    entière. Les blocs texte+position sont stockés tels quels pour la langue
    source, et traduits (Argos) vers les 2 autres langues supportées. Aucune
    image n'est régénérée : le frontend superpose le texte traduit à la
    position d'origine (bbox).
    """
    row_id = payload["row_id"]
    pages = payload.get("pages") or []
    source_lang = payload["source_lang"]

    if source_lang not in SUPPORTED_TRANSLATE_LANGS:
        raise ValueError(f"source_lang non supportée: {source_lang}")

    target_langs = [l for l in SUPPORTED_TRANSLATE_LANGS if l != source_lang]

    pages_processed = 0
    blocks_translated = 0

    for page_index, url in enumerate(pages):
        local_path = None
        try:
            local_path = download_to_temp(url)
            bubble_boxes = detect_bubbles(local_path)
            blocks = [
                {"text": text, "bbox": bbox}
                for bbox in bubble_boxes
                if (text := ocr_bubble(local_path, bbox, source_lang))
            ]
            if not blocks:
                # Repli : aucune bulle détectée, OCR pleine page avec
                # regroupement par ligne (comportement précédent).
                blocks = ocr_page(local_path, source_lang)
        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)

        supabase.table("chapter_translations").upsert({
            "chapter_id": row_id,
            "page_index": page_index,
            "lang": source_lang,
            "blocks": blocks,
        }, on_conflict="chapter_id,page_index,lang").execute()

        for target_lang in target_langs:
            translated_blocks = [
                {**b, "text": argostranslate.translate.translate(b["text"], source_lang, target_lang)}
                for b in blocks
            ]
            supabase.table("chapter_translations").upsert({
                "chapter_id": row_id,
                "page_index": page_index,
                "lang": target_lang,
                "blocks": translated_blocks,
            }, on_conflict="chapter_id,page_index,lang").execute()
            blocks_translated += len(translated_blocks)

        pages_processed += 1

    return {"pages_processed": pages_processed, "blocks_translated": blocks_translated}

HANDLERS = {
    "ping": handle_ping,
    "malware_scan": handle_malware_scan,
    "duplicate_check": handle_duplicate_check,
    "thumbnail": handle_thumbnail,
    "translate": handle_translate,
}


def fetch_next_job():
    res = (
        supabase.table("jobs")
        .select("*")
        .eq("status", "pending")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def process_job(job: dict) -> None:
    job_id = job["id"]
    job_type = job["type"]
    payload = job.get("payload") or {}

    supabase.table("jobs").update(
        {"status": "processing", "started_at": now_iso()}
    ).eq("id", job_id).execute()

    handler = HANDLERS.get(job_type)
    if handler is None:
        supabase.table("jobs").update(
            {
                "status": "failed",
                "error": f"Type de job inconnu: {job_type}",
                "completed_at": now_iso(),
            }
        ).eq("id", job_id).execute()
        print(f"[worker] job {job_id} : type inconnu '{job_type}'")
        return

    try:
        result = handler(payload)
        supabase.table("jobs").update(
            {"status": "done", "result": result, "completed_at": now_iso()}
        ).eq("id", job_id).execute()
        print(f"[worker] job {job_id} ({job_type}) : terminé")
    except Exception as exc:
        attempts = (job.get("attempts") or 0) + 1
        error_text = f"{exc}\n{traceback.format_exc()}"
        if attempts < MAX_ATTEMPTS:
            supabase.table("jobs").update(
                {"status": "pending", "attempts": attempts, "error": error_text}
            ).eq("id", job_id).execute()
            print(f"[worker] job {job_id} ({job_type}) : échec, retry {attempts}/{MAX_ATTEMPTS}")
        else:
            supabase.table("jobs").update(
                {
                    "status": "failed",
                    "attempts": attempts,
                    "error": error_text,
                    "completed_at": now_iso(),
                }
            ).eq("id", job_id).execute()
            print(f"[worker] job {job_id} ({job_type}) : échec définitif après {attempts} tentatives")


def drain_queue() -> int:
    """Traite tous les jobs 'pending' en attente, un par un, jusqu'à ce que la
    file soit vide. Renvoie le nombre de jobs traités."""
    count = 0
    while True:
        job = fetch_next_job()
        if job is None:
            break
        process_job(job)
        count += 1
    return count


# ── Serveur HTTP (réveillé par le Database Webhook Supabase) ───────────

app = Flask(__name__)
_processing_lock = threading.Lock()


@app.route("/", methods=["GET"])
def health():
    return "ok", 200


@app.route("/process", methods=["POST"])
def process_endpoint():
    if WORKER_SECRET and request.headers.get("X-Worker-Secret") != WORKER_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    if not _processing_lock.acquire(blocking=False):
        return jsonify({"status": "already_processing"}), 202

    try:
        processed = drain_queue()
    finally:
        _processing_lock.release()

    print(f"[worker] réveil terminé, {processed} job(s) traité(s), retour en veille.")
    return jsonify({"status": "done", "processed": processed}), 200


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    print(f"[worker] Spark worker (HTTP, mode veille) démarré sur le port {port}.")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()