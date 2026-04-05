# TP03 : Reconnaissance faciale par LBP et Viola-Jones

## Objectif
Implémenter un système de vérification faciale (1:1) combinant la détection de visage par Viola-Jones et l'extraction de caractéristiques par LBP (Local Binary Patterns).

## Prérequis
```bash
pip install numpy opencv-python scipy matplotlib
```

## Exécution
```bash
python tp03_lbp_viola_jones.py
```

## Architecture du système

```
Image → Détection Viola-Jones → Extraction LBP → Distance Euclidienne → Décision
         (Haar Cascades)        (histogramme 256 bins)  (similarité = 1 - dist)   (seuil 0.75)
```

## Classe `FaceVerificationSystem`

| Méthode | Rôle |
|---------|------|
| `__init__` | Charge le classificateur Haar |
| `detect_face(image)` | Détecte et extrait le plus grand visage (128×128) |
| `extract_lbp_features(face)` | Calcule l'histogramme LBP normalisé (256 bins) |
| `setup_reference(path)` | Enrôlement du visage de référence |
| `verify_face(path, threshold)` | Vérification 1:1 avec décision Match/No Match |

## Structure
```
TP03/
├── tp03_lbp_viola_jones.py
├── README.md
├── samples/
│   ├── person_a_ref.jpg
│   ├── person_a_test.jpg
│   └── person_b.jpg
└── results/
    ├── verify_same.png
    ├── verify_diff.png
    └── lbp_histograms.png
```

## Note
Les images de test sont synthétiques. Pour de meilleurs résultats de discrimination, remplacez les fichiers dans `samples/` par de vraies photos de visages.
