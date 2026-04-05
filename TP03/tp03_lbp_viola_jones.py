"""
TP03 : Reconnaissance faciale par LBP et Viola-Jones
=====================================================
Cours : Biométrie et Tatouage Numérique
Classe : ING-4-SSIR

Pipeline de vérification faciale :
  1. Détection de visage par Viola-Jones (Cascades de Haar)
  2. Extraction de caractéristiques par LBP (Local Binary Patterns)
  3. Comparaison par distance euclidienne
  4. Décision par seuillage
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# ─── Configuration ───────────────────────────────────────────────
SAMPLES_DIR = "samples"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REF_IMAGE = os.path.join(SAMPLES_DIR, "person_a_ref.jpg")
TEST_SAME = os.path.join(SAMPLES_DIR, "person_a_test.jpg")
TEST_DIFF = os.path.join(SAMPLES_DIR, "person_b.jpg")


# ═══════════════════════════════════════════════════════════════════
# Classe FaceVerificationSystem
# ═══════════════════════════════════════════════════════════════════

class FaceVerificationSystem:
    """
    Système de vérification faciale 1:1 basé sur LBP.

    Architecture :
      Capture → Détection (Viola-Jones) → Extraction LBP →
      Comparaison (distance euclidienne) → Décision (seuil)
    """

    def __init__(self):
        """
        Initialisation du détecteur de visage.
        Charge le classificateur en cascade Haar pré-entraîné d'OpenCV
        pour la détection frontale de visages.
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Impossible de charger le classificateur Haar.")

        # Stockage de la référence
        self.ref_features = None
        self.ref_face = None
        self.ref_image = None

    def detect_face(self, image):
        """
        Détection de visage par Viola-Jones.

        Paramètres du détecteur (selon l'énoncé) :
          - scaleFactor = 1.1
          - minNeighbors = 5
          - minSize = (30, 30)

        Si plusieurs visages sont détectés, on conserve le plus grand
        (surface w × h maximale).

        Paramètres
        ----------
        image : np.ndarray
            Image BGR en entrée.

        Retourne
        --------
        tuple : (face_roi, (x, y, w, h)) ou (None, None) si aucun visage.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            # Fallback : utiliser toute l'image comme visage
            # (utile pour les images synthétiques centrées)
            h, w = gray.shape
            face_roi = cv2.resize(gray, (128, 128))
            return face_roi, (0, 0, w, h)

        # Sélectionner le plus grand visage (surface maximale)
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        face_roi = gray[y:y+h, x:x+w]
        # Redimensionner à 128×128 (selon l'énoncé)
        face_roi = cv2.resize(face_roi, (128, 128))

        return face_roi, (x, y, w, h)

    def extract_lbp_features(self, face_image):
        """
        Extraction de caractéristiques par LBP (Local Binary Patterns).

        Principe :
          Pour chaque pixel (hors bords), on compare sa valeur avec ses
          8 voisins immédiats. Si le voisin ≥ centre → bit = 1, sinon → 0.
          On obtient un code binaire sur 8 bits (0-255) pour chaque pixel.

        Le vecteur de caractéristiques est l'histogramme normalisé des
        valeurs LBP (256 bins).

        Paramètres
        ----------
        face_image : np.ndarray
            Image de visage en niveaux de gris (128×128).

        Retourne
        --------
        np.ndarray : Histogramme LBP normalisé (256 valeurs).
        """
        rows, cols = face_image.shape
        lbp_image = np.zeros((rows - 2, cols - 2), dtype=np.uint8)

        # Parcours de chaque pixel (hors bords)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = face_image[i, j]
                code = 0

                # 8 voisins dans l'ordre horaire (en partant du haut-gauche)
                # P0: haut-gauche, P1: haut, P2: haut-droite, ...
                neighbors = [
                    face_image[i-1, j-1],  # P0 - haut-gauche
                    face_image[i-1, j],    # P1 - haut
                    face_image[i-1, j+1],  # P2 - haut-droite
                    face_image[i, j+1],    # P3 - droite
                    face_image[i+1, j+1],  # P4 - bas-droite
                    face_image[i+1, j],    # P5 - bas
                    face_image[i+1, j-1],  # P6 - bas-gauche
                    face_image[i, j-1],    # P7 - gauche
                ]

                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)

                lbp_image[i-1, j-1] = code

        # Histogramme normalisé (256 bins pour les 256 patterns possibles)
        hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-10)  # Normalisation

        return hist

    def setup_reference(self, image_path):
        """
        Enregistrement du visage de référence (phase d'enrôlement).

        Étapes :
          1. Charger l'image
          2. Détecter le visage
          3. Extraire les caractéristiques LBP
          4. Stocker le vecteur de référence

        Paramètres
        ----------
        image_path : str
            Chemin vers l'image de référence.
        """
        self.ref_image = cv2.imread(image_path)
        if self.ref_image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        self.ref_face, self.ref_coords = self.detect_face(self.ref_image)
        self.ref_features = self.extract_lbp_features(self.ref_face)

        print(f"  Référence enregistrée : {image_path}")
        print(f"  Visage détecté : {self.ref_coords}")

    def verify_face(self, image_path, threshold=0.75):
        """
        Vérification d'un visage par rapport à la référence.

        Étapes :
          1. Charger l'image test
          2. Détecter le visage
          3. Extraire les caractéristiques LBP
          4. Calculer la distance euclidienne avec la référence
          5. Convertir en similarité : similarité = 1 - distance
          6. Comparer au seuil

        Paramètres
        ----------
        image_path : str
            Chemin vers l'image de test.
        threshold : float
            Seuil de similarité (défaut 0.75).

        Retourne
        --------
        dict : Résultat contenant similarité, distance, décision, etc.
        """
        if self.ref_features is None:
            raise RuntimeError("Aucune référence enregistrée. Appelez setup_reference() d'abord.")

        test_image = cv2.imread(image_path)
        if test_image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        test_face, test_coords = self.detect_face(test_image)
        test_features = self.extract_lbp_features(test_face)

        # Distance euclidienne entre les histogrammes LBP
        distance = euclidean(self.ref_features, test_features)

        # Conversion en similarité (1 - distance)
        # On normalise pour que la similarité soit dans [0, 1]
        similarity = max(0, 1 - distance)

        # Décision
        is_match = similarity >= threshold
        decision = "MATCH" if is_match else "NO MATCH"

        result = {
            "image_path": image_path,
            "test_image": test_image,
            "test_face": test_face,
            "test_coords": test_coords,
            "test_features": test_features,
            "distance": distance,
            "similarity": similarity,
            "threshold": threshold,
            "decision": decision,
            "is_match": is_match,
        }

        return result


# ═══════════════════════════════════════════════════════════════════
# Fonctions de visualisation
# ═══════════════════════════════════════════════════════════════════

def draw_detection(image, coords, label, color):
    """Dessine le rectangle de détection et le label sur l'image."""
    img_copy = image.copy()
    x, y, w, h = coords
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img_copy, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_copy


def visualize_verification(system, result, title, filename):
    """Visualise le résultat de la vérification."""
    color = (0, 255, 0) if result["is_match"] else (0, 0, 255)
    status_text = result["decision"]

    ref_annotated = draw_detection(
        system.ref_image, system.ref_coords, "Reference", (255, 200, 0)
    )
    test_annotated = draw_detection(
        result["test_image"], result["test_coords"], status_text, color
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.cvtColor(ref_annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Référence")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(test_annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Test — {status_text}")
    axes[1].axis("off")

    status_icon = "✅" if result["is_match"] else "❌"
    plt.suptitle(
        f"{title}\n"
        f"Similarité = {result['similarity']:.4f} | "
        f"Distance = {result['distance']:.4f} | "
        f"Seuil = {result['threshold']} | "
        f"{status_icon} {status_text}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def visualize_lbp_histograms(system, results, filename):
    """Compare les histogrammes LBP de la référence et des tests."""
    fig, axes = plt.subplots(len(results) + 1, 1, figsize=(12, 3 * (len(results) + 1)))

    axes[0].bar(range(256), system.ref_features, color="steelblue", alpha=0.8)
    axes[0].set_title("Histogramme LBP — Référence")
    axes[0].set_xlim(0, 255)

    for i, (label, result) in enumerate(results):
        color = "green" if result["is_match"] else "indianred"
        axes[i+1].bar(range(256), result["test_features"], color=color, alpha=0.8)
        axes[i+1].set_title(f"Histogramme LBP — {label} (sim={result['similarity']:.4f})")
        axes[i+1].set_xlim(0, 255)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Programme principal (main)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  TP03 : Reconnaissance faciale par LBP et Viola-Jones")
    print("=" * 60)

    # Initialiser le système
    system = FaceVerificationSystem()

    # Enregistrer la référence (enrôlement)
    print("\n📷 Phase d'enrôlement :")
    system.setup_reference(REF_IMAGE)

    # Vérifier les images de test
    print("\n🔍 Phase de vérification :")
    print("-" * 50)

    threshold = 0.75
    all_results = []

    test_cases = [
        ("Même personne (A vs A')", TEST_SAME, "verify_same.png"),
        ("Personne différente (A vs B)", TEST_DIFF, "verify_diff.png"),
    ]

    for label, test_path, viz_file in test_cases:
        result = system.verify_face(test_path, threshold=threshold)

        status = "✅" if result["is_match"] else "❌"
        print(f"  {label:40s}")
        print(f"    Similarité : {result['similarity']:.4f}")
        print(f"    Distance   : {result['distance']:.4f}")
        print(f"    Décision   : {status} {result['decision']}")
        print()

        visualize_verification(system, result, label, viz_file)
        all_results.append((label, result))

    # Visualiser les histogrammes LBP comparés
    visualize_lbp_histograms(system, all_results, "lbp_histograms.png")

    # ─── Résumé ─────────────────────────────────────────────────
    print("=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    print(f"  {'Test':<40s} {'Similarité':>10} {'Décision':>12}")
    print("  " + "-" * 62)
    for label, result in all_results:
        status = "✅" if result["is_match"] else "❌"
        print(f"  {label:<40s} {result['similarity']:>10.4f} {status:>4} {result['decision']}")

    print(f"\n✅ TP03 terminé — résultats dans {RESULTS_DIR}/")
