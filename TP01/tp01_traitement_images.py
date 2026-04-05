"""
TP01 : Traitement des images avec PIL (Pillow)
===============================================
Cours : Biométrie et Tatouage Numérique
Classe : ING-4-SSIR

Ce script implémente 9 transformations d'images fondamentales
utilisées en prétraitement biométrique :
  1. Lecture et affichage
  2. Redimensionnement
  3. Ajustement de luminosité
  4. Conversion en niveaux de gris
  5. Binarisation
  6. Détection de contours
  7. Filtrage / débruitage (flou gaussien)
  8. Histogramme
  9. Égalisation d'histogramme
"""

import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour sauvegarde
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────
INPUT_IMAGE = "sample_image.jpg"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Charger l'image originale une seule fois
img_original = Image.open(INPUT_IMAGE)
print(f"Image chargée : {INPUT_IMAGE}")
print(f"  Taille : {img_original.size}")
print(f"  Mode   : {img_original.mode}")
print(f"  Format : {img_original.format}")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 1 — Lecture et affichage de l'image originale
# ═══════════════════════════════════════════════════════════════════
# Objectif : Charger et visualiser l'image source.
# Effet observé : L'image est affichée telle quelle, sans transformation.

fig = plt.figure(figsize=(6, 5))
plt.subplot(1, 1, 1)
plt.imshow(img_original)
plt.axis("off")
plt.title("Image Originale")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_originale.png"), dpi=150)
plt.close(fig)
print("[Partie 1] image_originale.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 2 — Redimensionnement
# ═══════════════════════════════════════════════════════════════════
# Objectif : Réduire ou adapter la taille de l'image pour normaliser
#            les entrées d'un système biométrique.
# Effet observé : L'image est réduite à 200×200, perdant son ratio
#                 original mais gagnant en uniformité de taille.

img_resized = img_original.resize((200, 200))

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.axis("off")
plt.title(f"Originale ({img_original.size[0]}×{img_original.size[1]})")

plt.subplot(1, 2, 2)
plt.imshow(img_resized)
plt.axis("off")
plt.title(f"Redimensionnée (200×200)")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_redimensionnee.png"), dpi=150)
plt.close(fig)
print("[Partie 2] image_redimensionnee.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 3 — Ajustement de la luminosité
# ═══════════════════════════════════════════════════════════════════
# Objectif : Modifier la luminosité pour compenser un éclairage
#            insuffisant lors de la capture biométrique.
# Effet observé : Avec un facteur de 1.5, l'image devient plus claire.
#                 Les détails dans les zones sombres deviennent visibles.

enhancer = ImageEnhance.Brightness(img_original)
img_bright = enhancer.enhance(1.5)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.axis("off")
plt.title("Originale")

plt.subplot(1, 2, 2)
plt.imshow(img_bright)
plt.axis("off")
plt.title("Luminosité ×1.5")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_luminosite_augmente.png"), dpi=150)
plt.close(fig)
print("[Partie 3] image_luminosite_augmente.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 4 — Conversion en niveaux de gris
# ═══════════════════════════════════════════════════════════════════
# Objectif : Passer en niveaux de gris pour simplifier le traitement.
#            En biométrie (empreinte, iris), on travaille quasi toujours
#            sur un seul canal d'intensité.
# Effet observé : L'image perd ses couleurs ; chaque pixel est
#                 représenté par une seule valeur d'intensité [0-255].

img_gray = img_original.convert("L")

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_original)
plt.axis("off")
plt.title("Originale (RGB)")

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Niveaux de gris (L)")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_gris.png"), dpi=150)
plt.close(fig)
print("[Partie 4] image_gris.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 5 — Binarisation
# ═══════════════════════════════════════════════════════════════════
# Objectif : Convertir l'image en noir et blanc pur (0 ou 255).
#            C'est une étape clé dans le traitement d'empreintes
#            digitales : les crêtes deviennent noires, le fond blanc.
# Effet observé : Avec un seuil de 128, les pixels d'intensité < 128
#                 passent à 0 (noir), les autres à 255 (blanc).

SEUIL = 128
img_binary = img_gray.point(lambda p: 255 if p >= SEUIL else 0)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Niveaux de gris")

plt.subplot(1, 2, 2)
plt.imshow(img_binary, cmap="gray")
plt.axis("off")
plt.title(f"Binarisée (seuil = {SEUIL})")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_binarisee.png"), dpi=150)
plt.close(fig)
print("[Partie 5] image_binarisee.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 6 — Détection des contours
# ═══════════════════════════════════════════════════════════════════
# Objectif : Extraire les contours de l'image. En biométrie, les
#            contours correspondent aux transitions crête/vallée
#            d'une empreinte ou aux limites de l'iris.
# Effet observé : Seules les zones de forte variation d'intensité
#                 (les bords) sont conservées, le reste devient noir.

img_edges = img_gray.filter(ImageFilter.FIND_EDGES)

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Niveaux de gris")

plt.subplot(1, 2, 2)
plt.imshow(img_edges, cmap="gray")
plt.axis("off")
plt.title("Contours (FIND_EDGES)")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_contours.png"), dpi=150)
plt.close(fig)
print("[Partie 6] image_contours.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 7 — Filtrage et débruitage (flou gaussien)
# ═══════════════════════════════════════════════════════════════════
# Objectif : Réduire le bruit dans l'image avant extraction de
#            caractéristiques. Un flou gaussien lisse les variations
#            aléatoires tout en préservant les structures principales.
# Effet observé : Plus le rayon augmente, plus l'image est floue.
#                 Rayon 1 = léger lissage, rayon 3 = flou prononcé.

radii = [1, 2, 3]
blurred_images = []
for r in radii:
    blurred_images.append(img_original.filter(ImageFilter.GaussianBlur(radius=r)))

fig = plt.figure(figsize=(14, 4))
plt.subplot(1, 4, 1)
plt.imshow(img_original)
plt.axis("off")
plt.title("Originale")

for i, (r, img_blur) in enumerate(zip(radii, blurred_images)):
    plt.subplot(1, 4, i + 2)
    plt.imshow(img_blur)
    plt.axis("off")
    plt.title(f"Gaussien r={r}")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_flou_gaussien.png"), dpi=150)
plt.close(fig)
print("[Partie 7] image_flou_gaussien.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 8 — Histogramme de l'image
# ═══════════════════════════════════════════════════════════════════
# Objectif : Visualiser la distribution des intensités de pixels.
#            L'histogramme révèle le contraste et l'exposition de
#            l'image — information critique pour le prétraitement.
# Effet observé : La courbe montre la fréquence de chaque niveau
#                 de gris (0 à 255). Un pic concentré = faible contraste.

histogram = img_gray.histogram()

fig = plt.figure(figsize=(8, 5))
plt.plot(range(256), histogram, color="black", linewidth=1)
plt.fill_between(range(256), histogram, alpha=0.3, color="gray")
plt.xlabel("Niveau de gris (0-255)")
plt.ylabel("Nombre de pixels")
plt.title("Histogramme de l'image en niveaux de gris")
plt.xlim(0, 255)
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "histogramme.png"), dpi=150)
plt.close(fig)
print("[Partie 8] histogramme.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
# PARTIE 9 — Égalisation de l'histogramme
# ═══════════════════════════════════════════════════════════════════
# Objectif : Redistribuer les intensités pour améliorer le contraste
#            global. Technique standard en prétraitement biométrique
#            pour normaliser l'éclairage entre captures différentes.
# Effet observé : L'histogramme s'étale sur toute la plage [0-255],
#                 les détails deviennent plus visibles dans les zones
#                 sombres et claires.

img_equalized = ImageOps.equalize(img_gray)
hist_equalized = img_equalized.histogram()

fig = plt.figure(figsize=(12, 8))

# Images : avant / après
plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Avant égalisation")

plt.subplot(2, 2, 2)
plt.imshow(img_equalized, cmap="gray")
plt.axis("off")
plt.title("Après égalisation")

# Histogrammes : avant / après
plt.subplot(2, 2, 3)
plt.plot(range(256), histogram, color="black", linewidth=1)
plt.fill_between(range(256), histogram, alpha=0.3, color="gray")
plt.title("Histogramme original")
plt.xlim(0, 255)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(range(256), hist_equalized, color="darkblue", linewidth=1)
plt.fill_between(range(256), hist_equalized, alpha=0.3, color="steelblue")
plt.title("Histogramme égalisé")
plt.xlim(0, 255)
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "image_egalisee.png"), dpi=150)
plt.close(fig)
print("[Partie 9] image_egalisee.png sauvegardée")


# ═══════════════════════════════════════════════════════════════════
print("\n✅ TP01 terminé — toutes les images sont dans le dossier results/")
