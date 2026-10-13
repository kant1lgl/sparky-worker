"""Télécharge et installe au moment du build Docker les modèles de traduction
Argos nécessaires au pipeline FR/EN/JP (évite tout téléchargement au runtime)."""

import argostranslate.package

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

# Argos n'a pas forcément de paquet direct fr<->ja : les paires manquantes
# passent automatiquement par l'anglais comme langue pivot à la traduction.
PAIRS = [("en", "fr"), ("fr", "en"), ("en", "ja"), ("ja", "en")]

for from_code, to_code in PAIRS:
    pkg = next(
        (p for p in available_packages if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if pkg is None:
        print(f"ATTENTION: aucun paquet Argos trouvé pour {from_code} -> {to_code}")
        continue
    print(f"Installation {from_code} -> {to_code}")
    argostranslate.package.install_from_path(pkg.download())