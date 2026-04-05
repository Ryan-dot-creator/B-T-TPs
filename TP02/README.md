# TP02 : Reconnaissance d'Empreinte Digitale

## Objectif
Implémenter et comparer quatre approches de matching d'empreintes digitales, en lien avec le pipeline vu en cours (acquisition → prétraitement → extraction → comparaison → décision).

## Prérequis
```bash
pip install Pillow matplotlib numpy opencv-python scikit-image scipy
```

## Exécution
```bash
python tp02_empreinte_digitale.py
```

## Méthodes implémentées

| Partie | Méthode | Principe | Métrique |
|--------|---------|----------|----------|
| A | ORB | Keypoints locaux + descripteurs binaires | Ratio de bons matchs (Hamming) |
| B | FFT | Spectre fréquentiel des crêtes | Corrélation croisée normalisée |
| C | Gabor | Banque de filtres orientés (8 directions) | Similarité cosinus |
| D | SSIM | Similarité structurelle après binarisation + contours | SSIM (luminance, contraste, structure) |

## Structure
```
TP02/
├── tp02_empreinte_digitale.py
├── README.md
├── samples/
│   ├── fingerprint_ref.png
│   ├── fingerprint_test_same.png
│   └── fingerprint_test_diff.png
└── results/
    ├── comparaison_methodes.png
    ├── images_comparaison.png
    ├── orb_same.png / orb_diff.png
    ├── fft_same.png / fft_diff.png
    ├── gabor_filters.png
    └── ssim_same.png / ssim_diff.png
```

## Lien avec le cours
- **Prétraitement** : niveaux de gris, égalisation, binarisation, contours (Partie A du chapitre empreintes)
- **Matching** : comparaison de templates via scores de similarité
- **Décision** : seuillage → acceptation/rejet, lié directement au compromis FAR/FRR
