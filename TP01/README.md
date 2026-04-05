# TP01 : Traitement des images avec PIL (Pillow)

## Objectif
Maîtriser les opérations fondamentales de traitement d'image utilisées en prétraitement biométrique : lecture, redimensionnement, conversion, binarisation, détection de contours, filtrage et analyse d'histogramme.

## Prérequis
```bash
pip install Pillow matplotlib
```

## Exécution
```bash
python tp01_traitement_images.py
```

## Structure
```
TP01/
├── tp01_traitement_images.py   # Script principal (9 parties)
├── sample_image.jpg            # Image d'entrée
├── README.md
└── results/                    # Images générées
    ├── image_originale.png
    ├── image_redimensionnee.png
    ├── image_luminosite_augmente.png
    ├── image_gris.png
    ├── image_binarisee.png
    ├── image_contours.png
    ├── image_flou_gaussien.png
    ├── histogramme.png
    └── image_egalisee.png
```

## Parties implémentées

| # | Transformation | Fonction PIL | Lien avec la biométrie |
|---|---------------|-------------|----------------------|
| 1 | Lecture / affichage | `Image.open()` | Acquisition du capteur |
| 2 | Redimensionnement | `resize()` | Normalisation des entrées |
| 3 | Luminosité | `ImageEnhance.Brightness()` | Compensation d'éclairage |
| 4 | Niveaux de gris | `convert("L")` | Réduction à 1 canal |
| 5 | Binarisation | `point()` | Séparation crêtes/vallées |
| 6 | Détection contours | `ImageFilter.FIND_EDGES` | Extraction des structures |
| 7 | Flou gaussien | `ImageFilter.GaussianBlur()` | Débruitage |
| 8 | Histogramme | `histogram()` | Analyse de contraste |
| 9 | Égalisation | `ImageOps.equalize()` | Normalisation inter-captures |
