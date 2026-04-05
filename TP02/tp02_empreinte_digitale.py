"""
TP02 : Reconnaissance d'Empreinte Digitale
============================================
Cours : Biométrie et Tatouage Numérique
Classe : ING-4-SSIR

Implémentation et comparaison de quatre approches de matching :
  A. Méthode locale : ORB (Oriented FAST and Rotated BRIEF)
  B. Méthode globale fréquentielle : FFT
  C. Méthode texture orientée : Filtres de Gabor
  D. Méthode similarité structurelle : SSIM + extraction de contours
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as compare_ssim
from scipy.ndimage import convolve

# ─── Configuration ───────────────────────────────────────────────
SAMPLES_DIR = "samples"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Images de test
REF_IMAGE = os.path.join(SAMPLES_DIR, "fingerprint_ref.png")
TEST_SAME = os.path.join(SAMPLES_DIR, "fingerprint_test_same.png")
TEST_DIFF = os.path.join(SAMPLES_DIR, "fingerprint_test_diff.png")

SEUIL_GLOBAL = 0.75  # Seuil de décision par défaut


# ═══════════════════════════════════════════════════════════════════
# Fonctions utilitaires communes
# ═══════════════════════════════════════════════════════════════════

def load_gray(path, size=(300, 300)):
    """Charge une image en niveaux de gris et la redimensionne."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    return cv2.resize(img, size)


def decision(score, seuil):
    """Retourne la décision d'acceptation/rejet."""
    return "ACCEPTÉE" if score >= seuil else "REJETÉE"


def print_result(method, pair_name, score, seuil):
    """Affiche le résultat d'une comparaison."""
    d = decision(score, seuil)
    status = "✅" if d == "ACCEPTÉE" else "❌"
    print(f"  {pair_name:30s} → Score = {score:.4f}  | Seuil = {seuil}  | {status} {d}")


# ═══════════════════════════════════════════════════════════════════
# PARTIE A — Méthode locale : ORB
# ═══════════════════════════════════════════════════════════════════
# Principe : ORB détecte des points d'intérêt (keypoints) et calcule
# des descripteurs binaires locaux. Le matching compare ces descripteurs
# entre deux images via BFMatcher (Brute Force) + distance de Hamming.
# Le score est le ratio de "bons" matchs / total de matchs possibles.

def method_orb(img1, img2, seuil=SEUIL_GLOBAL):
    """
    Matching par ORB (Oriented FAST and Rotated BRIEF).
    
    Étapes :
    1. Détection des keypoints et calcul des descripteurs ORB
    2. Matching des descripteurs par BFMatcher (Hamming)
    3. Filtrage des bons matchs (distance < 75% du max)
    4. Score = nombre de bons matchs / min(keypoints)
    """
    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0, None, None, None

    # Matching brute force avec distance de Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    # Filtrage : garder les matchs dont la distance < 75% du max
    if len(matches) > 0:
        max_dist = max(m.distance for m in matches)
        good_matches = [m for m in matches if m.distance < 0.75 * max_dist]
    else:
        good_matches = []

    # Score = ratio de bons matchs
    min_kp = min(len(kp1), len(kp2))
    score = len(good_matches) / min_kp if min_kp > 0 else 0.0

    return score, kp1, kp2, good_matches


def visualize_orb(img1, img2, kp1, kp2, matches, title, filename):
    """Visualise les correspondances ORB entre deux images."""
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:50],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(img_matches, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PARTIE B — Méthode globale fréquentielle : FFT
# ═══════════════════════════════════════════════════════════════════
# Principe : La FFT (Transformée de Fourier rapide) convertit l'image
# dans le domaine fréquentiel. On compare les spectres de magnitude
# normalisés des deux images via corrélation croisée.
# Les empreintes similaires ont des spectres fréquentiels proches
# (mêmes fréquences de crêtes).

def method_fft(img1, img2, seuil=SEUIL_GLOBAL):
    """
    Matching par FFT (Transformée de Fourier).
    
    Étapes :
    1. Calcul de la FFT 2D de chaque image
    2. Extraction du spectre de magnitude (log)
    3. Normalisation des spectres
    4. Score = corrélation croisée normalisée entre les spectres
    """
    # FFT 2D + centrage du spectre
    f1 = np.fft.fft2(img1.astype(np.float64))
    f2 = np.fft.fft2(img2.astype(np.float64))

    # Spectre de magnitude (log pour meilleure visualisation)
    mag1 = np.log1p(np.abs(np.fft.fftshift(f1)))
    mag2 = np.log1p(np.abs(np.fft.fftshift(f2)))

    # Normalisation [0, 1]
    mag1_norm = (mag1 - mag1.min()) / (mag1.max() - mag1.min() + 1e-10)
    mag2_norm = (mag2 - mag2.min()) / (mag2.max() - mag2.min() + 1e-10)

    # Corrélation croisée normalisée
    score = np.corrcoef(mag1_norm.flatten(), mag2_norm.flatten())[0, 1]

    return score, mag1_norm, mag2_norm


def visualize_fft(img1, img2, mag1, mag2, title, filename):
    """Visualise les spectres FFT des deux images."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(img1, cmap="gray")
    axes[0, 0].set_title("Image 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2, cmap="gray")
    axes[0, 1].set_title("Image 2")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mag1, cmap="hot")
    axes[1, 0].set_title("Spectre FFT - Image 1")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mag2, cmap="hot")
    axes[1, 1].set_title("Spectre FFT - Image 2")
    axes[1, 1].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PARTIE C — Méthode texture orientée : Filtres de Gabor
# ═══════════════════════════════════════════════════════════════════
# Principe : Les filtres de Gabor capturent l'orientation et la
# fréquence locale des crêtes de l'empreinte. On applique une banque
# de filtres à différentes orientations, puis on compare les réponses
# moyennes (vecteurs de caractéristiques) par similarité cosinus.

def build_gabor_filters(num_orientations=8, frequency=0.1, ksize=31):
    """
    Construit une banque de filtres de Gabor à différentes orientations.
    
    Paramètres :
    - num_orientations : nombre de directions (défaut 8 → tous les 22.5°)
    - frequency : fréquence spatiale du filtre
    - ksize : taille du noyau
    """
    filters = []
    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma=4.0, theta=theta,
            lambd=1.0 / frequency, gamma=0.5, psi=0
        )
        kernel /= kernel.sum() + 1e-10
        filters.append(kernel)
    return filters


def gabor_features(img, filters):
    """
    Extrait un vecteur de caractéristiques Gabor.
    Pour chaque filtre, on calcule la moyenne et l'écart-type
    de la réponse → vecteur de taille 2 × num_filters.
    """
    features = []
    for kernel in filters:
        response = cv2.filter2D(img, cv2.CV_64F, kernel)
        features.append(response.mean())
        features.append(response.std())
    return np.array(features)


def method_gabor(img1, img2, seuil=SEUIL_GLOBAL):
    """
    Matching par filtres de Gabor.
    
    Étapes :
    1. Construire une banque de 8 filtres de Gabor
    2. Extraire les vecteurs de caractéristiques pour chaque image
    3. Score = similarité cosinus entre les deux vecteurs
    """
    filters = build_gabor_filters(num_orientations=8)

    feat1 = gabor_features(img1, filters)
    feat2 = gabor_features(img2, filters)

    # Similarité cosinus
    dot = np.dot(feat1, feat2)
    norm = (np.linalg.norm(feat1) * np.linalg.norm(feat2)) + 1e-10
    score = dot / norm

    return score, feat1, feat2, filters


def visualize_gabor(img, filters, title, filename):
    """Visualise les réponses des filtres de Gabor."""
    n = len(filters)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 6))
    axes = axes.flatten()
    for i, kernel in enumerate(filters):
        response = cv2.filter2D(img, cv2.CV_64F, kernel)
        axes[i].imshow(response, cmap="gray")
        axes[i].set_title(f"θ = {i * 180 // n}°")
        axes[i].axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PARTIE D — Méthode SSIM (Similarité Structurelle)
# ═══════════════════════════════════════════════════════════════════
# Principe : SSIM mesure la similarité structurelle entre deux images
# en tenant compte de la luminance, du contraste et de la structure
# locale. Avant comparaison, les images subissent un prétraitement
# complet : niveaux de gris, égalisation, binarisation, contours.

def preprocess(image_path):
    """
    Prétraitement complet d'une image d'empreinte.
    
    Étapes obligatoires (selon l'énoncé) :
    1. Conversion en niveaux de gris
    2. Redimensionnement (300×300)
    3. Égalisation d'histogramme
    4. Binarisation (seuil = 128)
    5. Extraction des contours (FIND_EDGES)
    
    Retourne : np.ndarray (300×300)
    """
    img = Image.open(image_path)

    # 1. Conversion en niveaux de gris
    img = img.convert("L")

    # 2. Redimensionnement
    img = img.resize((300, 300))

    # 3. Égalisation d'histogramme
    img = ImageOps.equalize(img)

    # 4. Binarisation (seuil = 128)
    img = img.point(lambda p: 255 if p >= 128 else 0)

    # 5. Extraction des contours
    img = img.filter(ImageFilter.FIND_EDGES)

    return np.array(img)


def method_ssim(path1, path2, seuil=SEUIL_GLOBAL):
    """
    Matching par SSIM après prétraitement.
    
    Étapes :
    1. Prétraiter les deux images (preprocess)
    2. Calculer la similarité structurelle (SSIM)
    3. Comparer au seuil pour la décision
    """
    img1 = preprocess(path1)
    img2 = preprocess(path2)

    score = compare_ssim(img1, img2, data_range=255)

    return score, img1, img2


def visualize_ssim(img1, img2, score, title, filename):
    """Visualise le prétraitement et le score SSIM."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Référence (contours)")
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title("Test (contours)")
    axes[1].axis("off")

    plt.suptitle(f"{title}\nSSIM = {score:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL — Exécution et comparaison des 4 méthodes
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  TP02 : Reconnaissance d'Empreinte Digitale")
    print("  Comparaison de 4 méthodes de matching")
    print("=" * 65)

    # Charger les images en niveaux de gris pour les méthodes A, B, C
    ref = load_gray(REF_IMAGE)
    test_same = load_gray(TEST_SAME)
    test_diff = load_gray(TEST_DIFF)

    pairs = [
        ("Même doigt (same)", test_same, TEST_SAME),
        ("Doigt différent (diff)", test_diff, TEST_DIFF),
    ]

    # Tableau récapitulatif
    results_table = []

    # ─── PARTIE A : ORB ─────────────────────────────────────────
    print("\n🔹 PARTIE A — Méthode ORB (keypoints locaux)")
    print("-" * 50)
    seuil_orb = 0.15  # Seuil adapté pour ORB (scores plus bas)
    for name, test_img, test_path in pairs:
        score, kp1, kp2, matches = method_orb(ref, test_img, seuil_orb)
        print_result("ORB", name, score, seuil_orb)
        suffix = "same" if "same" in name.lower() or "même" in name.lower() else "diff"
        if matches is not None:
            visualize_orb(ref, test_img, kp1, kp2, matches,
                         f"ORB — {name} (score={score:.4f})",
                         f"orb_{suffix}.png")
        results_table.append(("ORB", name, score, seuil_orb, decision(score, seuil_orb)))

    # ─── PARTIE B : FFT ─────────────────────────────────────────
    print("\n🔹 PARTIE B — Méthode FFT (domaine fréquentiel)")
    print("-" * 50)
    seuil_fft = 0.85
    for name, test_img, test_path in pairs:
        score, mag1, mag2 = method_fft(ref, test_img, seuil_fft)
        print_result("FFT", name, score, seuil_fft)
        suffix = "same" if "same" in name.lower() or "même" in name.lower() else "diff"
        visualize_fft(ref, test_img, mag1, mag2,
                     f"FFT — {name} (corrélation={score:.4f})",
                     f"fft_{suffix}.png")
        results_table.append(("FFT", name, score, seuil_fft, decision(score, seuil_fft)))

    # ─── PARTIE C : Gabor ───────────────────────────────────────
    print("\n🔹 PARTIE C — Méthode Gabor (texture orientée)")
    print("-" * 50)
    seuil_gabor = 0.90
    for name, test_img, test_path in pairs:
        score, feat1, feat2, filters = method_gabor(ref, test_img, seuil_gabor)
        print_result("Gabor", name, score, seuil_gabor)
        suffix = "same" if "same" in name.lower() or "même" in name.lower() else "diff"
        results_table.append(("Gabor", name, score, seuil_gabor, decision(score, seuil_gabor)))

    # Visualiser les filtres de Gabor sur l'image de référence
    gabor_filters = build_gabor_filters()
    visualize_gabor(ref, gabor_filters, "Réponses Gabor — Image de référence", "gabor_filters.png")

    # ─── PARTIE D : SSIM ────────────────────────────────────────
    print("\n🔹 PARTIE D — Méthode SSIM (similarité structurelle)")
    print("-" * 50)
    seuil_ssim = 0.75
    for name, test_img, test_path in pairs:
        score, proc1, proc2 = method_ssim(REF_IMAGE, test_path, seuil_ssim)
        print_result("SSIM", name, score, seuil_ssim)
        suffix = "same" if "same" in name.lower() or "même" in name.lower() else "diff"
        visualize_ssim(proc1, proc2, score,
                      f"SSIM — {name}",
                      f"ssim_{suffix}.png")
        results_table.append(("SSIM", name, score, seuil_ssim, decision(score, seuil_ssim)))

    # ─── Tableau comparatif ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TABLEAU COMPARATIF DES 4 MÉTHODES")
    print("=" * 65)
    print(f"  {'Méthode':<10} {'Paire':<30} {'Score':>8} {'Seuil':>8} {'Décision':>12}")
    print("  " + "-" * 68)
    for method, pair, score, seuil, dec in results_table:
        status = "✅" if dec == "ACCEPTÉE" else "❌"
        print(f"  {method:<10} {pair:<30} {score:>8.4f} {seuil:>8.2f} {status:>4} {dec}")

    # ─── Figure récapitulative ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("Référence")
    axes[0].axis("off")

    axes[1].imshow(test_same, cmap="gray")
    axes[1].set_title("Test — Même doigt")
    axes[1].axis("off")

    axes[2].imshow(test_diff, cmap="gray")
    axes[2].set_title("Test — Doigt différent")
    axes[2].axis("off")

    plt.suptitle("Images utilisées pour la comparaison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "images_comparaison.png"), dpi=150)
    plt.close(fig)

    # Graphique comparatif des scores
    methods = ["ORB", "FFT", "Gabor", "SSIM"]
    scores_same = [r[2] for r in results_table if "même" in r[1].lower() or "same" in r[1].lower()]
    scores_diff = [r[2] for r in results_table if "différ" in r[1].lower() or "diff" in r[1].lower()]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, scores_same, width, label="Même doigt", color="steelblue")
    bars2 = ax.bar(x + width/2, scores_diff, width, label="Doigt différent", color="indianred")

    ax.set_ylabel("Score de similarité")
    ax.set_title("Comparaison des 4 méthodes de matching")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Ajouter les valeurs sur les barres
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "comparaison_methodes.png"), dpi=150)
    plt.close(fig)

    print(f"\n✅ TP02 terminé — toutes les images sont dans {RESULTS_DIR}/")
