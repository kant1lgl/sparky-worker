"""Télécharge au moment du build Docker les poids du détecteur de bulles
(RT-DETR-v2, licence Apache-2.0) pour fonctionner hors-ligne au runtime."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ogkalu/comic-text-and-bubble-detector",
    local_dir="/app/models/bubble-detector",
    allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
)