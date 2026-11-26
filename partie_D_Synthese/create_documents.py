import json
from pathlib import Path

import pymupdf

from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent

DOC_DIR = BASE_DIR / "documents"
IMG_DIR = BASE_DIR / "images"
DATA_DIR = BASE_DIR / "data"

for directory in [DOC_DIR, IMG_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =====================================
# 1. CREATION DE LA FIGURE
# =====================================

image_path = IMG_DIR / "img_px200_01.png"

image = Image.new(
    "RGB",
    (900, 450),
    "white"
)

draw = ImageDraw.Draw(image)

draw.rounded_rectangle(
    (100, 100, 800, 350),
    radius=30,
    fill="#263238"
)

# Voyant alimentation
draw.ellipse(
    (200, 180, 280, 260),
    fill="#22c55e"
)

# Voyant statut
draw.ellipse(
    (410, 180, 490, 260),
    fill="#f59e0b"
)

# Voyant erreur
draw.ellipse(
    (620, 180, 700, 260),
    fill="#ef4444"
)

draw.text(
    (190, 280),
    "POWER",
    fill="white"
)

draw.text(
    (405, 280),
    "STATUS",
    fill="white"
)

draw.text(
    (625, 280),
    "ERROR",
    fill="white"
)

draw.text(
    (350, 40),
    "PX-200 - Front panel",
    fill="black"
)

image.save(image_path)


# =====================================
# 2. CREATION DU MANUEL PDF
# =====================================

pdf = pymupdf.open()

pages = [

    (
        "PX-200 - Presentation",
        """
Le PX-200 est un produit electronique de demonstration.

Le panneau avant contient trois voyants :
POWER, STATUS et ERROR.

POWER indique l'alimentation.
STATUS indique l'etat de fonctionnement.
ERROR indique une anomalie.
"""
    ),

    (
        "PX-200 - Diagnostic des voyants",
        """
Un voyant STATUS orange clignotant indique
une phase d'initialisation.

Attendre 60 secondes.

Si le voyant continue de clignoter,
eteindre l'appareil, attendre 30 secondes,
puis redemarrer.

Si le probleme persiste, contacter le SAV.

Voir figure img_px200_01 :
panneau avant et voyants du PX-200.
"""
    ),

    (
        "PX-200 - Entretien",
        """
Debrancher l'appareil avant le nettoyage.

Utiliser un chiffon doux et sec.

Ne pas utiliser de liquide directement
sur les composants electroniques.

En cas de panne persistante,
contacter le service apres-vente.
"""
    )
]

for index, (title, content) in enumerate(pages):

    page = pdf.new_page()

    page.insert_text(
        (50, 60),
        title,
        fontsize=20
    )

    page.insert_textbox(
        pymupdf.Rect(50, 100, 550, 400),
        content,
        fontsize=12
    )

    if index == 1:

        page.insert_image(
            pymupdf.Rect(50, 420, 550, 670),
            filename=str(image_path)
        )

        page.insert_text(
            (50, 690),
            "Figure : img_px200_01",
            fontsize=11
        )

pdf.save(str(DOC_DIR / "manuel_PX200.pdf"))

pdf.close()


# =====================================
# 3. METADONNEES DE LA FIGURE
# =====================================

figures = [

    {
        "id": "img_px200_01",
        "source": "manuel_PX200.pdf",
        "page": 2,
        "file": "img_px200_01.png",
        "description": (
            "Panneau avant du PX-200 avec trois "
            "voyants POWER vert, STATUS orange "
            "et ERROR rouge. Diagnostic du voyant "
            "STATUS orange clignotant."
        )
    }
]

with open(
    DATA_DIR / "figures.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        figures,
        f,
        ensure_ascii=False,
        indent=4
    )


print("[OK] Manuel PDF cree.")
print("[OK] Figure img_px200_01 creee.")
print("[OK] Metadonnees creees.")
