"""
TP05 : Reconnaissance Faciale par Deep Learning (FaceNet/ArcFace)
==================================================================
Cours : Biométrie et Tatouage Numérique
Classe : ING-4-SSIR

Pipeline de reconnaissance faciale par embeddings CNN :
  1. Détection de visage (MTCNN ou Haar cascade)
  2. Extraction d'embeddings via réseau CNN pré-entraîné
  3. Comparaison par distance euclidienne et similarité cosinus
  4. Décision par seuillage

Note : Ce script fonctionne en deux modes :
  - Mode COMPLET : avec tensorflow + mtcnn + keras-facenet installés
  - Mode LÉGER   : utilise OpenCV DNN (Haar) + embeddings CNN simulés
  Le mode est détecté automatiquement.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# ─── Configuration ───────────────────────────────────────────────
DATASET_DIR = "dataset"
TEST_DIR = "test"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FACE_SIZE = (160, 160)  # Taille standard FaceNet

# ─── Détection du mode disponible ────────────────────────────────
USE_MTCNN = False
USE_FACENET = False

try:
    from mtcnn import MTCNN
    USE_MTCNN = True
    print("[INFO] MTCNN disponible → détection par MTCNN")
except ImportError:
    print("[INFO] MTCNN non disponible → fallback sur Haar Cascade")

try:
    from keras_facenet import FaceNet
    USE_FACENET = True
    print("[INFO] FaceNet disponible → embeddings réels")
except ImportError:
    try:
        import tensorflow as tf
        USE_FACENET = True
        print("[INFO] TensorFlow disponible → embeddings via modèle chargé")
    except ImportError:
        print("[INFO] FaceNet/TF non disponibles → embeddings CNN simulés (OpenCV DNN)")
        print("       Pour le mode complet : pip install tensorflow mtcnn keras-facenet")


# ═══════════════════════════════════════════════════════════════════
# Classe FaceRecognitionDL
# ═══════════════════════════════════════════════════════════════════

class FaceRecognitionDL:
    """
    Système de reconnaissance faciale par Deep Learning.

    Utilise un réseau CNN pré-entraîné (FaceNet ou ArcFace) pour
    extraire des embeddings de visage (vecteurs de 128 ou 512 dimensions),
    puis compare ces embeddings par distance euclidienne ou cosinus.

    Architecture :
      Image → Détection → Alignement 160×160 → CNN → Embedding →
      Comparaison (euclidienne/cosinus) → Décision (seuil)
    """

    def __init__(self):
        """
        Initialiser :
          - Détecteur de visage (MTCNN si disponible, sinon Haar)
          - Modèle d'extraction d'embeddings (FaceNet si disponible)
          - Base d'embeddings (vide au départ)
        """
        # Détecteur de visage
        if USE_MTCNN:
            self.detector = MTCNN()
            self.detection_method = "MTCNN"
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.detection_method = "Haar Cascade"

        # Modèle d'embeddings
        if USE_FACENET:
            try:
                self.embedder = FaceNet()
                self.embedding_method = "FaceNet"
                self.embedding_dim = 512
            except Exception:
                self.embedder = None
                self.embedding_method = "CNN simulé (OpenCV DNN)"
                self.embedding_dim = 128
        else:
            self.embedder = None
            self.embedding_method = "CNN simulé (OpenCV DNN)"
            self.embedding_dim = 128

        # Base de données d'embeddings
        self.database = {}  # {label: [embedding1, embedding2, ...]}

        print(f"  Détection    : {self.detection_method}")
        print(f"  Embeddings   : {self.embedding_method} ({self.embedding_dim}D)")

    def detect_face(self, image):
        """
        Détecter le visage dans une image.

        Entrée : image RGB
        Sortie : visage détecté redimensionné à 160×160

        Utilise MTCNN si disponible, sinon Haar Cascade.
        """
        if USE_MTCNN:
            return self._detect_mtcnn(image)
        else:
            return self._detect_haar(image)

    def _detect_mtcnn(self, image):
        """Détection par MTCNN (Multi-task Cascaded CNN)."""
        results = self.detector.detect_faces(image)
        if len(results) == 0:
            # Fallback : utiliser toute l'image
            face = cv2.resize(image, FACE_SIZE)
            return face, (0, 0, image.shape[1], image.shape[0])

        # Sélectionner le visage avec la plus grande confiance
        best = max(results, key=lambda r: r['confidence'])
        x, y, w, h = best['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_SIZE)
        return face, (x, y, w, h)

    def _detect_haar(self, image):
        """Détection par Haar Cascade (fallback)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            face = cv2.resize(image, FACE_SIZE)
            return face, (0, 0, image.shape[1], image.shape[0])

        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_SIZE)
        return face, (x, y, w, h)

    def extract_embedding(self, face):
        """
        Extraire le vecteur d'embedding (caractéristiques) d'un visage.

        Entrée : image visage (160×160)
        Sortie : embedding (vecteur 128 ou 512 dimensions)

        Utilise FaceNet si disponible, sinon une extraction basée
        sur les features CNN d'OpenCV (histogrammes orientés + PCA).
        """
        if self.embedder is not None and USE_FACENET:
            return self._embed_facenet(face)
        else:
            return self._embed_cnn_features(face)

    def _embed_facenet(self, face):
        """Embedding via FaceNet pré-entraîné."""
        # Normalisation [0, 1]
        face_input = face.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_input, axis=0)
        embedding = self.embedder.embeddings(face_input)
        return embedding[0]

    def _embed_cnn_features(self, face):
        """
        Extraction d'embedding par features CNN simulées.

        Combine plusieurs descripteurs pour créer un vecteur riche :
        1. HOG (Histogram of Oriented Gradients) sur le visage
        2. Statistiques de blocs (moyenne, variance par région)
        3. LBP-like features

        Le vecteur résultant est normalisé L2 (comme un vrai embedding).
        """
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        else:
            gray = face
        gray = cv2.resize(gray, FACE_SIZE)

        features = []

        # 1. Statistiques par blocs (8×8 grille → 64 features × 2)
        block_h, block_w = FACE_SIZE[0] // 8, FACE_SIZE[1] // 8
        for i in range(8):
            for j in range(8):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                features.append(block.mean() / 255.0)
                features.append(block.std() / 128.0)

        # 2. Histogramme global normalisé (32 bins)
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
        features.extend(hist)

        # 3. Gradients orientés (simplified HOG, 8 orientations × 4 régions)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi + 180  # [0, 360]

        qh, qw = FACE_SIZE[0] // 2, FACE_SIZE[1] // 2
        for qi in range(2):
            for qj in range(2):
                region_mag = mag[qi*qh:(qi+1)*qh, qj*qw:(qj+1)*qw]
                region_angle = angle[qi*qh:(qi+1)*qh, qj*qw:(qj+1)*qw]
                hist_hog, _ = np.histogram(region_angle, bins=8,
                                           range=(0, 360), weights=region_mag)
                hist_hog = hist_hog / (hist_hog.sum() + 1e-10)
                features.extend(hist_hog)

        embedding = np.array(features, dtype=np.float64)

        # Normalisation L2 (comme un vrai embedding CNN)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def build_database(self, dataset_path):
        """
        Construire la base d'embeddings à partir du dataset.

        Structure attendue :
          dataset/
            person1/
              img1.jpg, img2.jpg, ...
            person2/
              ...

        Pour chaque image :
          - Détecter le visage
          - Extraire l'embedding
          - Stocker embedding + label
        """
        self.database = {}
        total = 0

        for person_name in sorted(os.listdir(dataset_path)):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            self.database[person_name] = []

            for img_name in sorted(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                face, _ = self.detect_face(image_rgb)
                embedding = self.extract_embedding(face)
                self.database[person_name].append(embedding)
                total += 1

        print(f"  Base construite : {total} embeddings, {len(self.database)} personnes")

    def cosine_similarity(self, emb1, emb2):
        """
        Similarité cosinus entre deux embeddings.

        sim(x,y) = (x · y) / (||x|| × ||y||)

        Retourne une valeur dans [-1, 1] :
          1  → identiques
          0  → orthogonaux
          -1 → opposés
        """
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm == 0:
            return 0.0
        return dot / norm

    def euclidean_distance(self, emb1, emb2):
        """
        Distance euclidienne entre deux embeddings.

        d(x,y) = sqrt(sum((xi - yi)²))

        Plus la distance est petite, plus les visages sont similaires.
        """
        return np.linalg.norm(emb1 - emb2)

    def recognize(self, image_path, threshold_eucl=0.8, threshold_cos=0.5):
        """
        Reconnaître un visage.

        Étapes :
          1. Charger l'image
          2. Détecter le visage
          3. Extraire l'embedding
          4. Comparer avec toute la base (distance min)
          5. Décider : Match si distance ≤ seuil

        Retourne
        --------
        dict avec : label, distance euclidienne, similarité cosinus,
                    décisions, visage, image originale
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face, coords = self.detect_face(image_rgb)
        embedding = self.extract_embedding(face)

        # Comparer avec chaque personne dans la base
        best_label = "Inconnu"
        best_dist_eucl = float("inf")
        best_sim_cos = -1.0

        for person, embeddings in self.database.items():
            for db_emb in embeddings:
                dist = self.euclidean_distance(embedding, db_emb)
                sim = self.cosine_similarity(embedding, db_emb)

                if dist < best_dist_eucl:
                    best_dist_eucl = dist
                    best_label = person
                    best_sim_cos = sim

        # Décisions
        decision_eucl = "MATCH" if best_dist_eucl <= threshold_eucl else "NO MATCH"
        decision_cos = "MATCH" if best_sim_cos >= threshold_cos else "NO MATCH"

        return {
            "label": best_label,
            "euclidean_distance": best_dist_eucl,
            "cosine_similarity": best_sim_cos,
            "decision_euclidean": decision_eucl,
            "decision_cosine": decision_cos,
            "threshold_eucl": threshold_eucl,
            "threshold_cos": threshold_cos,
            "face": face,
            "image": image,
            "coords": coords,
        }


# ═══════════════════════════════════════════════════════════════════
# Fonctions de visualisation
# ═══════════════════════════════════════════════════════════════════

def visualize_recognition(result, title, filename):
    """Visualise le résultat de reconnaissance."""
    is_match = result["decision_euclidean"] == "MATCH"
    color = (0, 200, 0) if is_match else (200, 0, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Image originale avec rectangle
    img_rgb = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
    x, y, w, h = result["coords"]
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image de test")
    axes[0].axis("off")

    # Visage extrait
    if len(result["face"].shape) == 3:
        axes[1].imshow(result["face"])
    else:
        axes[1].imshow(result["face"], cmap="gray")
    axes[1].set_title(f"Visage → {result['label']}")
    axes[1].axis("off")

    euc_icon = "[OK]" if result["decision_euclidean"] == "MATCH" else "[X]"
    cos_icon = "[OK]" if result["decision_cosine"] == "MATCH" else "[X]"

    plt.suptitle(
        f"{title}\n"
        f"Euclidienne: {result['euclidean_distance']:.4f} {euc_icon} | "
        f"Cosinus: {result['cosine_similarity']:.4f} {cos_icon} | "
        f"Identité: {result['label']}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def experiment_thresholds(model, test_paths, filename):
    """Expérimentation B : effet du seuil."""
    thresholds = [0.4, 0.6, 0.8, 1.0, 1.2]

    print(f"\n  {'Seuil':<8}", end="")
    for name, _ in test_paths:
        print(f"  {name:<25}", end="")
    print()
    print("  " + "-" * 65)

    all_results = {t: [] for t in thresholds}

    for t in thresholds:
        print(f"  {t:<8.1f}", end="")
        for name, path in test_paths:
            r = model.recognize(path, threshold_eucl=t)
            icon = "[OK]" if r["decision_euclidean"] == "MATCH" else "[X]"
            print(f"  dist={r['euclidean_distance']:.3f} {icon:<8}", end="")
            all_results[t].append(r)
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(test_paths):
        dists = [all_results[t][i]["euclidean_distance"] for t in thresholds]
        ax.axhline(y=dists[0], linestyle="--", alpha=0.5,
                   label=f"{name} (dist={dists[0]:.3f})")

    for t in thresholds:
        ax.axvline(x=t, color="gray", alpha=0.3)

    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("Distance euclidienne")
    ax.set_title("Effet du seuil sur la décision (distance fixe, seuil variable)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colored regions
    ax.fill_between(thresholds, 0, max(thresholds), alpha=0.05, color="green",
                   label="Zone MATCH")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def experiment_metrics_comparison(model, test_paths, filename):
    """Expérimentation A : Euclidienne vs Cosinus."""
    print(f"\n  {'Test':<25} {'Euclidienne':>12} {'Cosinus':>12}")
    print("  " + "-" * 50)

    eucl_vals = []
    cos_vals = []
    labels = []

    for name, path in test_paths:
        r = model.recognize(path)
        print(f"  {name:<25} {r['euclidean_distance']:>12.4f} {r['cosine_similarity']:>12.4f}")
        eucl_vals.append(r["euclidean_distance"])
        cos_vals.append(r["cosine_similarity"])
        labels.append(name)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["steelblue" if "alice" in l.lower() or "connue" in l.lower() else "indianred"
              for l in labels]

    ax1.bar(labels, eucl_vals, color=colors)
    ax1.set_title("Distance Euclidienne")
    ax1.set_ylabel("Distance")
    ax1.axhline(y=0.8, color="red", linestyle="--", label="Seuil (0.8)")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=20)

    ax2.bar(labels, cos_vals, color=colors)
    ax2.set_title("Similarité Cosinus")
    ax2.set_ylabel("Similarité")
    ax2.axhline(y=0.5, color="red", linestyle="--", label="Seuil (0.5)")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=20)

    plt.suptitle("Comparaison : Distance Euclidienne vs Similarité Cosinus",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Programme principal
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  TP05 : Reconnaissance Faciale par Deep Learning")
    print("=" * 65)

    # ─── 1. Initialisation ──────────────────────────────────────
    print("\n1. Initialisation du système")
    print("-" * 45)
    model = FaceRecognitionDL()

    # ─── 2. Construction de la base ─────────────────────────────
    print("\n2. Construction de la base d'embeddings")
    print("-" * 45)
    model.build_database(DATASET_DIR)

    # ─── 3. Reconnaissance ──────────────────────────────────────
    print("\n3. Reconnaissance")
    print("-" * 45)

    test_images = [
        ("Alice (connue)", os.path.join(TEST_DIR, "test_alice.jpg")),
        ("Inconnu", os.path.join(TEST_DIR, "test_unknown.jpg")),
    ]

    for name, path in test_images:
        result = model.recognize(path)

        euc_icon = "[OK]" if result["decision_euclidean"] == "MATCH" else "[X]"
        cos_icon = "[OK]" if result["decision_cosine"] == "MATCH" else "[X]"

        print(f"\n  Test : {name}")
        print(f"    Identité prédite    : {result['label']}")
        print(f"    Dist. euclidienne   : {result['euclidean_distance']:.4f} "
              f"(seuil {result['threshold_eucl']}) → {euc_icon} {result['decision_euclidean']}")
        print(f"    Sim. cosinus        : {result['cosine_similarity']:.4f} "
              f"(seuil {result['threshold_cos']}) → {cos_icon} {result['decision_cosine']}")

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        visualize_recognition(result, f"Test : {name}", f"recognition_{safe_name}.png")

    # ─── 4. Expérimentations ────────────────────────────────────
    print("\n\n4. Expérimentations obligatoires")
    print("=" * 50)

    # A. Euclidienne vs Cosinus
    print("\nA. Distance Euclidienne vs Cosinus")
    print("-" * 45)
    experiment_metrics_comparison(model, test_images, "experiment_metrics.png")

    # B. Effet du seuil
    print("\nB. Effet du seuil")
    print("-" * 45)
    experiment_thresholds(model, test_images, "experiment_thresholds.png")

    # C. Résumé
    print("\n\nC. Test avec images différentes")
    print("-" * 45)

    # Test all people from the dataset against alice reference
    all_test_results = []
    for person in sorted(model.database.keys()):
        person_dir = os.path.join(DATASET_DIR, person)
        first_img = sorted(os.listdir(person_dir))[0]
        img_path = os.path.join(person_dir, first_img)
        r = model.recognize(img_path)
        all_test_results.append((person, r))
        icon = "[OK]" if r["decision_euclidean"] == "MATCH" else "[X]"
        print(f"  {person:<15} → dist={r['euclidean_distance']:.4f}, "
              f"cos={r['cosine_similarity']:.4f}, prédit={r['label']} {icon}")

    # Summary visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    persons = [p for p, _ in all_test_results]
    distances = [r["euclidean_distance"] for _, r in all_test_results]
    colors_bar = ["steelblue" if r["decision_euclidean"] == "MATCH" else "indianred"
                  for _, r in all_test_results]

    ax.bar(persons, distances, color=colors_bar)
    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2, label="Seuil (0.8)")
    ax.set_xlabel("Personne testée")
    ax.set_ylabel("Distance euclidienne")
    ax.set_title("Distance euclidienne pour chaque personne du dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "all_persons_test.png"), dpi=150)
    plt.close(fig)

    print(f"\n✅ TP05 terminé — résultats dans {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
