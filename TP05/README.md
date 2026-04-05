# TP05 : Reconnaissance Faciale par Deep Learning (FaceNet/ArcFace)

## Objectif
Implémenter un système de reconnaissance faciale basé sur des embeddings CNN, avec comparaison par distance euclidienne et similarité cosinus.

## Prérequis

Mode léger (par défaut) :
```bash
pip install numpy opencv-python matplotlib scikit-learn
```

Mode complet (avec vrais embeddings FaceNet) :
```bash
pip install tensorflow mtcnn keras-facenet
```

Le script détecte automatiquement les bibliothèques disponibles et s'adapte.

## Exécution
```bash
python tp05_deep_learning.py
```

## Architecture
```
Image → Détection (MTCNN/Haar) → Resize 160×160 → CNN → Embedding (128/512D)
                                                          ↓
Base d'embeddings ← dataset/                    Comparaison (Eucl. + Cosinus)
                                                          ↓
                                                    Décision (seuil)
```

## Classe `FaceRecognitionDL`

| Méthode | Rôle |
|---------|------|
| `__init__` | Initialise détecteur + modèle CNN |
| `detect_face(image)` | Détection via MTCNN ou Haar |
| `extract_embedding(face)` | Vecteur 128D ou 512D |
| `build_database(path)` | Construit la base d'embeddings |
| `cosine_similarity(emb1, emb2)` | sim = (x·y)/(‖x‖·‖y‖) |
| `euclidean_distance(emb1, emb2)` | d = √Σ(xi-yi)² |
| `recognize(image_path)` | Reconnaissance + double décision |

## Expérimentations
- **A.** Euclidienne vs Cosinus — tableau comparatif
- **B.** Effet du seuil (0.4, 0.6, 0.8, 1.0, 1.2)
- **C.** Test sur toutes les personnes du dataset

## Structure
```
TP05/
├── tp05_deep_learning.py
├── README.md
├── dataset/          # 5 personnes × 4 images
├── test/
│   ├── test_alice.jpg
│   └── test_unknown.jpg
└── results/
    ├── recognition_alice_connue.png
    ├── recognition_inconnu.png
    ├── experiment_metrics.png
    ├── experiment_thresholds.png
    └── all_persons_test.png
```

## Seuils typiques

| Méthode | Seuil | Interprétation |
|---------|-------|---------------|
| Euclidienne | ~0.8 | distance ≤ seuil → Match |
| Cosinus | ~0.5 | similarité ≥ seuil → Match |
