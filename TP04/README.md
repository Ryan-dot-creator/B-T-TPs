# TP04 : Reconnaissance faciale par PCA (Eigenfaces) et Viola-Jones

## Objectif
Construire un système de reconnaissance faciale (1:N) basé sur l'Analyse en Composantes Principales (ACP/PCA), avec détection par Viola-Jones et comparaison par distance euclidienne dans l'espace réduit.

## Prérequis
```bash
pip install numpy opencv-python matplotlib
```

## Exécution
```bash
python tp04_pca_eigenfaces.py
```

## Architecture
```
Images → Viola-Jones → Vectorisation → PCA (projection) → Distance Euclidienne → Décision
                        (100×100 → 10000D)  (10000D → kD)     (min distance)        (seuil)
```

## Classe `FaceRecognitionPCA`

| Méthode | Rôle |
|---------|------|
| `__init__(n_components)` | Initialise Viola-Jones et paramètre k |
| `detect_face(image)` | Détecte et extrait le visage (100×100) |
| `load_dataset(path)` | Charge le dataset structuré par personne |
| `compute_pca(X)` | Calcule moyenne, centrage, vecteurs propres |
| `project(face_vector)` | Projette un visage dans l'espace PCA |
| `recognize(image_path, threshold)` | Reconnaissance 1:N avec décision |

## Expérimentations
- Effet du nombre de composantes k (5, 10, 20, 50)
- Analyse Distance vs Décision
- Questions d'analyse (alignement, k trop faible/élevé, illumination)

## Structure
```
TP04/
├── tp04_pca_eigenfaces.py
├── README.md
├── dataset/          # 5 personnes × 4 images
│   ├── alice/
│   ├── bob/
│   ├── charlie/
│   ├── diana/
│   └── eve/
├── test/
│   ├── test_alice.jpg
│   └── test_unknown.jpg
└── results/
    ├── eigenfaces.png
    ├── experiment_k.png
    ├── recognition_alice_connue.png
    └── recognition_inconnu.png
```
