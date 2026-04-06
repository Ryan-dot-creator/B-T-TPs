"""
TP04 : Reconnaissance faciale par PCA (Eigenfaces) et Viola-Jones
==================================================================
Cours : Biométrie et Tatouage Numérique
Classe : ING-4-SSIR

Pipeline de reconnaissance faciale (1:N) :
  1. Détection de visage par Viola-Jones
  2. Construction du modèle PCA (Eigenfaces)
  3. Projection dans le sous-espace de dimension réduite
  4. Comparaison par distance euclidienne
  5. Décision par seuillage
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────
DATASET_DIR = "dataset"
TEST_DIR = "test"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FACE_SIZE = (100, 100)  # Taille normalisée des visages


# ═══════════════════════════════════════════════════════════════════
# Classe FaceRecognitionPCA
# ═══════════════════════════════════════════════════════════════════

class FaceRecognitionPCA:
    """
    Système de reconnaissance faciale par Eigenfaces (PCA).

    L'ACP (Analyse en Composantes Principales) projette les images
    de visage dans un sous-espace de dimension réduite, défini par
    les vecteurs propres de la matrice de covariance (eigenfaces).
    La reconnaissance se fait par distance euclidienne minimale
    dans cet espace réduit.
    """

    def __init__(self, n_components=30):
        """
        Initialisation du système.

        Paramètres
        ----------
        n_components : int
            Nombre de composantes principales (eigenfaces) à conserver.
        """
        # Détecteur Viola-Jones
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Impossible de charger le classificateur Haar.")

        self.n_components = n_components

        # Variables PCA (remplies par compute_pca)
        self.mean = None
        self.eigenvectors = None

        # Base de données (remplie par load_dataset + project)
        self.projections = None
        self.labels = None

    def detect_face(self, image):
        """
        Détection et extraction du visage redimensionné.

        Entrée : image BGR
        Sortie : visage en niveaux de gris (100×100)

        Étapes :
          - Conversion en gris
          - detectMultiScale (scaleFactor=1.1, minNeighbors=5)
          - Sélection du plus grand visage
          - Resize 100×100
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            # Fallback pour images synthétiques
            face_roi = cv2.resize(gray, FACE_SIZE)
            return face_roi

        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, FACE_SIZE)

        return face_roi

    def load_dataset(self, dataset_path):
        """
        Charge le dataset structuré par personne.

        Structure attendue :
          dataset/
            person1/
              img1.jpg
              img2.jpg
            person2/
              ...

        Pour chaque image :
          - Détecter le visage
          - Vectoriser (aplatir en vecteur 1D)
          - Stocker dans la matrice X
          - Enregistrer le label

        Retourne
        --------
        X : np.ndarray (n_samples × n_pixels)
        y : list de labels (noms des personnes)
        """
        X = []
        y = []

        for person_name in sorted(os.listdir(dataset_path)):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_name in sorted(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                face = self.detect_face(image)
                # Vectoriser : image 100×100 → vecteur de 10000 éléments
                face_vector = face.flatten().astype(np.float64)

                X.append(face_vector)
                y.append(person_name)

        X = np.array(X)
        print(f"  Dataset chargé : {len(y)} images, {len(set(y))} personnes")
        print(f"  Personnes : {sorted(set(y))}")
        print(f"  Matrice X : {X.shape}")

        return X, y

    def compute_pca(self, X):
        """
        Calcul de l'ACP (PCA) sur la matrice d'entraînement.

        Étapes :
          1. Calcul de la moyenne (mean face)
          2. Centrage : X_centered = X - mean
          3. Matrice de covariance (astuce : X^T X au lieu de X X^T)
          4. Calcul des valeurs et vecteurs propres
          5. Tri décroissant par valeur propre
          6. Sélection des n_components premiers vecteurs

        Stocke self.mean et self.eigenvectors
        """
        n_samples, n_features = X.shape

        # 1. Moyenne
        self.mean = np.mean(X, axis=0)

        # 2. Centrage
        X_centered = X - self.mean

        # 3. Covariance (astuce de Turk & Pentland pour réduire le coût)
        # Au lieu de calculer la matrice n_features × n_features,
        # on calcule la matrice n_samples × n_samples
        if n_samples < n_features:
            # Astuce : covariance réduite
            C_small = np.dot(X_centered, X_centered.T) / n_samples
            eigenvalues, eigenvectors_small = np.linalg.eigh(C_small)

            # Convertir vers l'espace original
            eigenvectors = np.dot(X_centered.T, eigenvectors_small)

            # Normaliser
            for i in range(eigenvectors.shape[1]):
                norm = np.linalg.norm(eigenvectors[:, i])
                if norm > 0:
                    eigenvectors[:, i] /= norm
        else:
            C = np.dot(X_centered.T, X_centered) / n_samples
            eigenvalues, eigenvectors = np.linalg.eigh(C)

        # 5. Tri décroissant
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 6. Sélection des k premières composantes
        k = min(self.n_components, eigenvectors.shape[1])
        self.eigenvectors = eigenvectors[:, :k]
        self.eigenvalues = eigenvalues[:k]

        # Variance expliquée
        total_var = np.sum(np.abs(eigenvalues))
        explained_var = np.sum(np.abs(self.eigenvalues)) / total_var * 100

        print(f"  PCA calculée : {k} composantes retenues")
        print(f"  Variance expliquée : {explained_var:.1f}%")

        # Projeter toutes les images d'entraînement
        self.projections = self.project(X)

    def project(self, X):
        """
        Projection d'un ou plusieurs visages dans l'espace PCA.

        Formule : projection = (X - mean) × eigenvectors

        Paramètres
        ----------
        X : np.ndarray
            Vecteur(s) de visage (1D ou 2D).

        Retourne
        --------
        np.ndarray : Coordonnées dans l'espace PCA.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)

    def recognize(self, image_path, threshold=3000):
        """
        Reconnaître un visage dans une image.

        Étapes :
          1. Charger l'image
          2. Détecter le visage
          3. Vectoriser et projeter dans l'espace PCA
          4. Calculer la distance euclidienne avec chaque projection
          5. Trouver la distance minimale
          6. Décider : Match si distance < seuil

        Retourne
        --------
        tuple : (label, distance, décision, face_image)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        face = self.detect_face(image)
        face_vector = face.flatten().astype(np.float64)
        projection = self.project(face_vector)

        # Distance avec chaque image d'entraînement
        distances = np.linalg.norm(self.projections - projection, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        predicted_label = self.labels[min_idx]

        # Décision
        is_match = min_dist < threshold
        decision = "MATCH" if is_match else "NO MATCH"

        return predicted_label, min_dist, decision, face, image


# ═══════════════════════════════════════════════════════════════════
# Fonctions de visualisation
# ═══════════════════════════════════════════════════════════════════

def visualize_eigenfaces(model, filename, n_show=10):
    """Affiche les premières eigenfaces."""
    n = min(n_show, model.eigenvectors.shape[1])
    fig, axes = plt.subplots(2, (n+2)//2, figsize=(14, 6))
    axes = axes.flatten()

    # Mean face
    axes[0].imshow(model.mean.reshape(FACE_SIZE), cmap="gray")
    axes[0].set_title("Visage moyen")
    axes[0].axis("off")

    for i in range(n):
        ef = model.eigenvectors[:, i].reshape(FACE_SIZE)
        axes[i+1].imshow(ef, cmap="gray")
        axes[i+1].set_title(f"EF {i+1}")
        axes[i+1].axis("off")

    # Hide extra axes
    for j in range(n+1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Eigenfaces (vecteurs propres principaux)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def visualize_recognition(image, face, label, distance, decision, title, filename):
    """Affiche le résultat de la reconnaissance."""
    color = (0, 255, 0) if decision == "MATCH" else (0, 0, 255)
    status = "OK" if decision == "MATCH" else "X"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image de test")
    axes[0].axis("off")

    axes[1].imshow(face, cmap="gray")
    axes[1].set_title(f"Visage détecté\n→ {label} (dist={distance:.1f})")
    axes[1].axis("off")

    icon = "[OK]" if decision == "MATCH" else "[X]"
    plt.suptitle(f"{title}\n{icon} {decision} — Identité : {label} — Distance : {distance:.1f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def experiment_components(model_class, X, y, test_paths, filename):
    """Expérimentation : effet du nombre de composantes k."""
    k_values = [5, 10, 20, 50]
    results = {}

    for k in k_values:
        m = model_class(n_components=k)
        m.labels = y
        m.compute_pca(X)

        results[k] = []
        for test_name, test_path in test_paths:
            _, dist, dec, _, _ = m.recognize(test_path)
            results[k].append((test_name, dist, dec))

    # Table
    print(f"\n  {'k':<6}", end="")
    for test_name, _ in test_paths:
        print(f"  {test_name:<25}", end="")
    print()
    print("  " + "-" * 60)
    for k in k_values:
        print(f"  {k:<6}", end="")
        for name, dist, dec in results[k]:
            icon = "[OK]" if dec == "MATCH" else "[X]"
            print(f"  {dist:<10.1f} {icon:<14}", end="")
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (test_name, _) in enumerate(test_paths):
        dists = [results[k][i][1] for k in k_values]
        ax.plot(k_values, dists, 'o-', label=test_name, linewidth=2, markersize=8)

    ax.set_xlabel("Nombre de composantes (k)")
    ax.set_ylabel("Distance euclidienne")
    ax.set_title("Effet du nombre de composantes PCA sur la distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Programme principal
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  TP04 : Reconnaissance faciale par PCA (Eigenfaces)")
    print("=" * 60)

    # ─── 1. Charger le dataset ──────────────────────────────────
    print("\n1. Chargement du dataset")
    print("-" * 40)
    model = FaceRecognitionPCA(n_components=20)
    X, y = model.load_dataset(DATASET_DIR)
    model.labels = y

    # ─── 2. Construire le modèle PCA ───────────────────────────
    print("\n2. Construction du modèle PCA")
    print("-" * 40)
    model.compute_pca(X)

    # Visualiser les eigenfaces
    visualize_eigenfaces(model, "eigenfaces.png")

    # ─── 3. Reconnaissance ──────────────────────────────────────
    print("\n3. Reconnaissance")
    print("-" * 40)

    test_images = [
        ("Messi (connu)", os.path.join(TEST_DIR, "test_messi.jpg")),
        ("Ronaldo (connu)", os.path.join(TEST_DIR, "test_ronaldo.jpg")),
        ("Haaland (inconnu)", os.path.join(TEST_DIR, "test_unknown.jpg")),
    ]

    threshold = 3000
    for test_name, test_path in test_images:
        label, dist, decision, face, image = model.recognize(test_path, threshold=threshold)

        icon = "[OK]" if decision == "MATCH" else "[X]"
        print(f"  Test : {test_name}")
        print(f"    Identité prédite : {label}")
        print(f"    Distance min     : {dist:.1f}")
        print(f"    Seuil            : {threshold}")
        print(f"    Décision         : {icon} {decision}")
        print()

        safe_name = test_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        visualize_recognition(image, face, label, dist, decision,
                            f"Test : {test_name}", f"recognition_{safe_name}.png")

    # ─── 4. Expérimentations ────────────────────────────────────
    print("\n4. Expérimentation : effet du nombre de composantes k")
    print("-" * 40)
    experiment_components(FaceRecognitionPCA, X, y, test_images, "experiment_k.png")

    # ─── 5. Questions d'analyse ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  QUESTIONS D'ANALYSE")
    print("=" * 60)
    print("""
  Q1: Pourquoi PCA nécessite un bon alignement des visages ?
  → PCA compare les pixels position par position. Si les visages
    ne sont pas alignés, les mêmes structures (yeux, nez) ne sont
    pas aux mêmes coordonnées, ce qui fausse la comparaison.

  Q2: Que se passe-t-il si k est trop faible ?
  → On perd trop d'information discriminante. Les visages projetés
    se ressemblent tous → taux de fausses acceptations élevé.

  Q3: Que se passe-t-il si k est trop élevé ?
  → On capture du bruit (overfitting). Les composantes mineures
    contiennent du bruit plutôt que des caractéristiques utiles.

  Q4: Pourquoi la distance euclidienne est adaptée dans l'espace PCA ?
  → Les axes PCA sont orthogonaux et ordonnés par variance. La
    distance euclidienne mesure donc directement la dissimilarité
    dans un espace où chaque axe a une signification statistique.

  Q5: Limites d'Eigenfaces face aux variations d'illumination ?
  → PCA est très sensible à l'éclairage car les premières
    composantes capturent souvent les variations de lumière
    plutôt que l'identité. Solutions : Fisherfaces (LDA) ou
    normalisation d'histogramme avant projection.
    """)

    print(f"Resultats sauvegardés dans {RESULTS_DIR}/")
